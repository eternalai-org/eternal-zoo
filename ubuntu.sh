#!/bin/bash
set -o pipefail

# Function: log_message
# Logs informational messages with a specific format.
log_message() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    if [[ -n "${message// }" ]]; then
        echo "[$timestamp] [INFO] [MODEL_INSTALL_LLAMA] $message"
    fi
}

# Function: log_error
# Logs error messages with a specific format.
log_error() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    if [[ -n "${message// }" ]]; then
        echo "[$timestamp] [ERROR] [MODEL_INSTALL_LLAMA] $message" >&2
    fi
}

# Function: log_success
# Logs success messages with a specific format.
log_success() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    if [[ -n "${message// }" ]]; then
        echo "[$timestamp] [SUCCESS] [MODEL_INSTALL_LLAMA] $message"
    fi
}

# Function: log_warning
# Logs warning messages with a specific format.
log_warning() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    if [[ -n "${message// }" ]]; then
        echo "[$timestamp] [WARNING] [MODEL_INSTALL_LLAMA] $message" >&2
    fi
}

# Function: log_section
# Logs section headers with a specific format.
log_section() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    if [[ -n "${message// }" ]]; then
        echo
        echo "[$timestamp] [SECTION] [MODEL_INSTALL_LLAMA] =========================================="
        echo "[$timestamp] [SECTION] [MODEL_INSTALL_LLAMA] $message"
        echo "[$timestamp] [SECTION] [MODEL_INSTALL_LLAMA] =========================================="
        echo
    fi
}

# Function: handle_error
# Handles errors, logs the error, deactivates the virtual environment if active, and exits.
handle_error() {
    local exit_code=$1
    local error_msg=$2
    log_error "$error_msg (Exit code: $exit_code)"
    if [[ -n "$VIRTUAL_ENV" ]]; then
        log_warning "Deactivating virtual environment due to error..."
        deactivate 2>/dev/null || true
    fi
    exit $exit_code
}

# Function: command_exists
# Checks if a command exists in the system.
command_exists() {
    command -v "$1" &> /dev/null
}

# Function: ensure_podman
# Ensures Podman is installed; attempts installation via apt-get if missing.
ensure_podman() {
    log_section "Checking Podman Installation"
    if command_exists podman; then
        log_success "Podman is installed and available."
        return
    fi

    log_warning "Podman not found. Attempting to install podman via apt-get..."
    sudo apt-get update
    if sudo apt-get install -y podman; then
        if command_exists podman; then
            log_success "Podman installed successfully."
            return
        fi
    fi
    handle_error 1 "Failed to install podman automatically. Please install podman and rerun this script."
}

# Function: ensure_containers_conf
# Guarantees that the specified containers.conf has the NVIDIA hook/CDI entries.
ensure_containers_conf() {
    local target_path="$1"
    local target_label="$2"
    local needs_update=0

    if [[ ! -f "$target_path" ]]; then
        needs_update=1
    else
        if ! grep -q "hooks_dir" "$target_path" >/dev/null 2>&1; then
            needs_update=1
        elif ! grep -q "cdi_config_dir" "$target_path" >/dev/null 2>&1; then
            needs_update=1
        fi
    fi

    if [[ "$needs_update" -eq 0 ]]; then
        log_message "$target_label already contains Podman NVIDIA configuration."
        return
    fi

    local tmp_conf
    tmp_conf=$(mktemp)
    cat <<'CONF' > "$tmp_conf"
[engine]
hooks_dir = ["/usr/share/containers/oci/hooks.d", "/etc/containers/oci/hooks.d"]
cdi_config_dir = ["/etc/cdi", "/usr/local/etc/cdi"]
CONF

    local timestamp
    timestamp=$(date +%Y%m%d%H%M%S)

    if [[ "$target_path" == /etc/* ]]; then
        sudo install -d -m 0755 "$(dirname "$target_path")"
        if [[ -f "$target_path" ]]; then
            sudo cp "$target_path" "$target_path.backup.$timestamp"
        fi
        sudo cp "$tmp_conf" "$target_path"
    else
        mkdir -p "$(dirname "$target_path")"
        if [[ -f "$target_path" ]]; then
            cp "$target_path" "$target_path.backup.$timestamp"
        fi
        cp "$tmp_conf" "$target_path"
    fi

    rm -f "$tmp_conf"
    log_success "Applied Podman NVIDIA configuration to $target_label."
}

# Function: configure_podman_gpu
# Installs NVIDIA Container Toolkit pieces, generates CDI specs, and updates Podman configs.
configure_podman_gpu() {
    log_section "Configuring Podman for NVIDIA GPUs"

    local gpu_packages=(nvidia-container-toolkit nvidia-container-toolkit-base)
    for pkg in "${gpu_packages[@]}"; do
        if ! dpkg -s "$pkg" >/dev/null 2>&1; then
            log_message "Installing $pkg..."
            sudo apt-get install -y "$pkg" || handle_error $? "Failed to install $pkg."
        else
            log_message "$pkg already installed."
        fi
    done

    if command_exists nvidia-ctk; then
        sudo install -d -m 0755 /etc/cdi
        if sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml >/dev/null 2>&1; then
            log_success "Generated NVIDIA CDI spec at /etc/cdi/nvidia.yaml."
        else
            log_warning "nvidia-ctk cdi generation failed; GPU passthrough may require manual intervention."
        fi
    else
        log_warning "nvidia-ctk not found; skipping automatic CDI generation."
    fi

    ensure_containers_conf "/etc/containers/containers.conf" "system containers.conf"
    ensure_containers_conf "$HOME/.config/containers/containers.conf" "user containers.conf"

    if command_exists systemctl; then
        sudo systemctl restart podman >/dev/null 2>&1 || log_warning "Could not restart system podman service (continuing)."
        systemctl --user daemon-reload >/dev/null 2>&1 || true
        systemctl --user restart podman.socket podman.service >/dev/null 2>&1 || log_message "User-level podman units not restarted (may be unused)."
    fi
}

# Function: check_apt_get
# Checks if apt-get is available.
check_apt_get() {
    log_section "Checking Package Manager"
    if ! command_exists apt-get; then
        log_error "apt-get is not available. This script requires Ubuntu or a compatible system."
        exit 1
    fi
    log_success "apt-get package manager is available."
}

# Function: check_sudo
# Checks if the user has sudo privileges.
check_sudo() {
    log_section "Checking Sudo Privileges"
    if ! sudo -n true 2>/dev/null; then
        log_error "This script requires sudo privileges. Please run as a user with sudo access."
        exit 1
    fi
    log_success "Sudo privileges confirmed."
}

# Function: check_internet_connectivity
# Verifies outbound connectivity by calling GitHub (curl) or ICMP ping to 8.8.8.8.
check_internet_connectivity() {
    log_section "Checking Internet Connectivity"
    if command_exists curl; then
        if ! curl -sSf https://github.com >/dev/null 2>&1; then
            handle_error 1 "No internet connectivity or GitHub unreachable."
        fi
    elif command_exists ping; then
        if ! ping -c1 -W2 8.8.8.8 >/dev/null 2>&1; then
            handle_error 1 "No internet connectivity (ping to 8.8.8.8 failed)."
        fi
    else
        handle_error 1 "Neither curl nor ping is available to verify connectivity."
    fi
    log_success "Internet connectivity verified."
}

# Function: check_python_installable
# Ensures Python >= 3.12 is present (installs python3.12 if necessary).
check_python_installable() {
    log_section "Checking Python Installation"
    REQUIRED_PY_VERSION="3.12"
    log_message "Looking for python${REQUIRED_PY_VERSION} or newer..."

    if command_exists python3.12; then
        PYTHON_CMD="python3.12"
        PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
        log_success "Found python3.12: $PYTHON_CMD ($PYTHON_VERSION)"
        return
    fi

    log_message "Searching for Python installations >= ${REQUIRED_PY_VERSION}..."
    PYTHON_VERSIONS=()
    for py_exec in /usr/bin/python3.*; do
        if [[ -x "$py_exec" && "$py_exec" =~ python3\.[0-9]+$ ]]; then
            version_output=$("$py_exec" --version 2>&1)
            if [[ $version_output =~ Python\ ([0-9]+\.[0-9]+) ]]; then
                version="${BASH_REMATCH[1]}"
                PYTHON_VERSIONS+=("$version:$py_exec")
            fi
        fi
    done

    HIGHEST_VERSION=""
    HIGHEST_PATH=""
    for version_path in "${PYTHON_VERSIONS[@]}"; do
        version="${version_path%%:*}"
        path="${version_path##*:}"
        if [[ $(printf '%s\n' "$REQUIRED_PY_VERSION" "$version" | sort -V | head -n1) == "$REQUIRED_PY_VERSION" ]]; then
            if [[ -z "$HIGHEST_VERSION" ]] || [[ $(printf '%s\n' "$HIGHEST_VERSION" "$version" | sort -V | tail -n1) == "$version" ]]; then
                HIGHEST_VERSION="$version"
                HIGHEST_PATH="$path"
            fi
        fi
    done

    if [[ -n "$HIGHEST_PATH" ]]; then
        PYTHON_CMD="$HIGHEST_PATH"
        PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
        log_success "Found Python >= ${REQUIRED_PY_VERSION}: $PYTHON_CMD ($PYTHON_VERSION)"
        return
    fi

    log_warning "No Python >= ${REQUIRED_PY_VERSION} found. Attempting to install python3.12 via deadsnakes PPA..."
    sudo apt-get update
    sudo apt-get install -y software-properties-common || handle_error $? "Failed to install software-properties-common."
    sudo add-apt-repository -y ppa:deadsnakes/ppa || handle_error $? "Failed to add deadsnakes PPA."
    sudo apt-get update
    sudo apt-get install -y python3.12 python3.12-venv python3.12-pip || handle_error $? "Failed to install python3.12."

    if command_exists python3.12; then
        PYTHON_CMD="python3.12"
        PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
        log_success "Successfully installed python3.12: $PYTHON_CMD ($PYTHON_VERSION)"
    else
        handle_error 1 "python3.12 installation failed. Please install Python >= 3.11 manually."
    fi
}

# Function: preflight_checks
# Runs all pre-installation checks and prints a summary.
preflight_checks() {
    log_section "Running Preflight Checks"
    check_apt_get
    check_sudo
    check_internet_connectivity
    ensure_podman
    check_python_installable
    log_success "All preflight checks passed successfully."
    echo
    echo "========================================="
    echo "Preflight checks summary:"
    echo "- apt-get: OK"
    echo "- Sudo: OK"
    echo "- Internet connectivity: OK"
    echo "- Podman: OK"
    echo "- Python: Using $PYTHON_CMD ($PYTHON_VERSION)"
    echo "========================================="
    echo
}

# Run all preflight checks before proceeding.
preflight_checks

# Python selection is now handled in check_python_installable function

log_message "Using Python at: $(which $PYTHON_CMD)"
log_message "Python setup complete."

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
MODELS_DIR="$HOME/.eternal-zoo/models"
TEMPLATES_DIR="$PROJECT_DIR/eternal_zoo/examples/templates"
log_message "Resolved project directory: $PROJECT_DIR"
log_message "Models directory will be created at: $MODELS_DIR"
log_message "Template directory will be mounted from: $TEMPLATES_DIR"
mkdir -p "$MODELS_DIR"
if [[ ! -d "$TEMPLATES_DIR" ]]; then
    log_warning "Template directory $TEMPLATES_DIR was not found. Continuing, but llama-server templates may be unavailable."
fi

# -----------------------------------------------------------------------------
# Step 2: Install required system packages
# -----------------------------------------------------------------------------
log_message "Installing required packages..."
if command -v apt-get &> /dev/null; then
    REQUIRED_PACKAGES=(pigz cmake libcurl4-openssl-dev python3-venv python3-pip build-essential git ninja-build nvidia-cuda-toolkit nvidia-container-toolkit nvidia-container-toolkit-base)
    MISSING_PACKAGES=()
    for pkg in "${REQUIRED_PACKAGES[@]}"; do
        if ! dpkg -s "$pkg" &> /dev/null; then
            MISSING_PACKAGES+=("$pkg")
        fi
    done
    if [ "${#MISSING_PACKAGES[@]}" -ne 0 ]; then
        log_message "Installing missing packages: ${MISSING_PACKAGES[*]}"
        sudo apt-get update
        sudo apt-get install -y "${MISSING_PACKAGES[@]}" || handle_error $? "Failed to install required packages."
    else
        log_message "All required packages are already installed."
    fi
else
    log_error "No supported package manager found (apt-get)"
    exit 1
fi
log_message "All required packages installed successfully."
configure_podman_gpu


# -----------------------------------------------------------------------------
# Step 3: Pull llama-server CUDA container image via Podman
# -----------------------------------------------------------------------------
LLAMA_IMAGE="${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}"
log_message "Pulling llama-server CUDA image with podman: $LLAMA_IMAGE"
if ! podman pull "$LLAMA_IMAGE"; then
    handle_error 1 "Failed to pull llama.cpp server Docker image."
fi
log_message "Container image $LLAMA_IMAGE is available locally."

# -----------------------------------------------------------------------------
# Step 4: Create llama-server wrapper script for Docker usage
# -----------------------------------------------------------------------------
log_message "Creating llama-server wrapper script..."
LLAMA_WRAPPER_DIR="$HOME/.local/bin"
mkdir -p "$LLAMA_WRAPPER_DIR"

cat > "$LLAMA_WRAPPER_DIR/llama-server" <<EOF
#!/bin/bash
set -euo pipefail

CONTAINER_RUNTIME="\${CONTAINER_RUNTIME:-podman}"
PROJECT_DIR="$PROJECT_DIR"
LLAMA_IMAGE="\${LLAMA_IMAGE:-$LLAMA_IMAGE}"
MODELS_DIR="\${MODELS_DIR:-\$HOME/.eternal-zoo/models}"
TEMPLATES_DIR="\${TEMPLATES_DIR:-$TEMPLATES_DIR}"

mkdir -p "\$MODELS_DIR"

RUNTIME_ARGS=()
DEFAULT_PODMAN_GPU_FLAGS="--hooks-dir=/usr/share/containers/oci/hooks.d --device nvidia.com/gpu=all"
if [[ -n "\${PODMAN_GPU_FLAGS:-}" ]]; then
    echo "[INFO] Passing PODMAN_GPU_FLAGS=\${PODMAN_GPU_FLAGS}"
    # shellcheck disable=SC2206
    RUNTIME_ARGS+=(\${PODMAN_GPU_FLAGS})
else
    echo "[INFO] Using default GPU flags: \${DEFAULT_PODMAN_GPU_FLAGS}"
    # shellcheck disable=SC2206
    RUNTIME_ARGS+=(\${DEFAULT_PODMAN_GPU_FLAGS})
fi

# Remove unsupported --cdi-device flags for Podman builds that do not recognize them.
if [[ "\${RUNTIME_ARGS[*]:-}" == *"--cdi-device"* ]]; then
    PODMAN_VERSION_OUTPUT=\$("\$CONTAINER_RUNTIME" --version 2>/dev/null || echo "unknown")
    echo "[WARNING] Detected '--cdi-device' in GPU flags, but \${CONTAINER_RUNTIME} (\${PODMAN_VERSION_OUTPUT}) may not support it. Removing the flag."
    FILTERED_ARGS=()
    skip_next=0
    for ((idx=0; idx<\${#RUNTIME_ARGS[@]}; idx++)); do
        arg="\${RUNTIME_ARGS[idx]}"
        if [[ "\$skip_next" -eq 1 ]]; then
            skip_next=0
            continue
        fi
        if [[ "\$arg" == "--cdi-device" ]]; then
            echo "[WARNING] Dropping '--cdi-device' and its value '\${RUNTIME_ARGS[idx+1]:-}' if present."
            skip_next=1
            continue
        fi
        if [[ "\$arg" == --cdi-device=* ]]; then
            echo "[WARNING] Dropping flag '\$arg'."
            continue
        fi
        FILTERED_ARGS+=("\$arg")
    done
    RUNTIME_ARGS=("\${FILTERED_ARGS[@]}")
fi

CONTAINER_CMD=("\$CONTAINER_RUNTIME" run -it --rm --network=host)
if [[ \${#RUNTIME_ARGS[@]} -gt 0 ]]; then
    CONTAINER_CMD+=("\${RUNTIME_ARGS[@]}")
fi
CONTAINER_CMD+=(
    -v "\$MODELS_DIR:\$MODELS_DIR"
    -v "\$TEMPLATES_DIR:\$TEMPLATES_DIR"
    -v "\$PROJECT_DIR:\$PROJECT_DIR"
    "\$LLAMA_IMAGE"
)

if [[ \$# -gt 0 ]]; then
    CONTAINER_CMD+=("\$@")
else
    echo "[DEBUG] No CLI arguments provided; relying on container entrypoint."
fi

GPU_ACCESS_METHOD="\${GPU_ACCESS:-}"
if [[ -n "\$GPU_ACCESS_METHOD" ]]; then
    echo "[INFO] llama-server wrapper detected GPU_ACCESS=\$GPU_ACCESS_METHOD"
fi

run_with_gpu_helper() {
    local helper="\$1"
    shift
    if command -v "\$helper" >/dev/null 2>&1; then
        "\$helper" "\$@"
    else
        echo "[WARNING] GPU helper '\$helper' not found. Running container command directly."
        "\$@"
    fi
}

case "\$GPU_ACCESS_METHOD" in
    prime-run)
        run_with_gpu_helper prime-run "\${CONTAINER_CMD[@]}"
        ;;
    optirun)
        run_with_gpu_helper optirun "\${CONTAINER_CMD[@]}"
        ;;
    primusrun)
        run_with_gpu_helper primusrun "\${CONTAINER_CMD[@]}"
        ;;
    *)
        "\${CONTAINER_CMD[@]}"
        ;;
esac
EOF

chmod +x "$LLAMA_WRAPPER_DIR/llama-server"
log_message "Created llama-server wrapper at $LLAMA_WRAPPER_DIR/llama-server."

# -----------------------------------------------------------------------------
# Step 5: Add llama-server wrapper directory to PATH in shell rc file
# -----------------------------------------------------------------------------
# Function: update_shell_rc_path
# Updates the specified shell rc file to include the wrapper directory in PATH.
update_shell_rc_path() {
    local shell_rc="$1"
    local path_line="export PATH=\"$LLAMA_WRAPPER_DIR:\$PATH\""
    if [ -f "$shell_rc" ]; then
        log_message "Backing up $shell_rc..."
        cp "$shell_rc" "$shell_rc.backup.$(date +%Y%m%d%H%M%S)" || log_error "Failed to backup $shell_rc."
        if grep -Fxq "$path_line" "$shell_rc"; then
            log_message "$LLAMA_WRAPPER_DIR already in PATH in $shell_rc. No update needed."
        else
            # Remove any previous lines that add $LLAMA_WRAPPER_DIR to PATH.
            sed -i "\|export PATH=\"$LLAMA_WRAPPER_DIR:\$PATH\"|d" "$shell_rc"
            echo "$path_line" >> "$shell_rc"
            log_message "Updated PATH in $shell_rc."
        fi
    else
        log_message "$shell_rc does not exist. Creating and adding PATH update."
        echo "$path_line" > "$shell_rc"
    fi
}

if [[ ":$PATH:" != *":$LLAMA_WRAPPER_DIR:"* ]]; then
    log_message "Adding $LLAMA_WRAPPER_DIR to PATH..."
    export PATH="$LLAMA_WRAPPER_DIR:$PATH"
    # Detect which shell rc file to update based on the user's shell.
    SHELL_NAME=$(basename "$SHELL")
    if [ "$SHELL_NAME" = "zsh" ]; then
        update_shell_rc_path "$HOME/.zshrc"
    else
        update_shell_rc_path "$HOME/.bashrc"
    fi
    # Set a flag to print an informative message at the end
    PATH_UPDATE_NEEDED=1
    log_message "PATH updated for current session and future sessions."
fi

# -----------------------------------------------------------------------------
# Step 6: Create and activate Python virtual environment
# -----------------------------------------------------------------------------
VENV_PATH=".eternal-zoo"
log_message "Creating virtual environment '$VENV_PATH'..."
"$PYTHON_CMD" -m venv "$VENV_PATH" || handle_error $? "Failed to create virtual environment."

log_message "Activating virtual environment..."
source "$VENV_PATH/bin/activate" || handle_error $? "Failed to activate virtual environment."
log_message "Virtual environment activated."

# -----------------------------------------------------------------------------
# Step 7: Install eternal-zoo toolkit in the virtual environment
# -----------------------------------------------------------------------------
log_message "Upgrading pip/setuptools inside the virtual environment..."
pip install --upgrade pip setuptools wheel || log_warning "pip bootstrap upgrade failed; continuing..."

log_message "Installing eternal-zoo from the current workspace..."
pip install . || handle_error $? "Failed to install eternal-zoo toolkit."
log_message "eternal-zoo toolkit installed."

log_message "Setup completed successfully."

# At the end of the script, print an informative message if PATH was updated
if [ "${PATH_UPDATE_NEEDED:-0}" = "1" ]; then
    echo
    echo "[INFO] The llama-server command directory was added to your PATH in your shell rc file."
    echo "      To use it in this session, run: export PATH=\"$LLAMA_WRAPPER_DIR:\$PATH\""
    echo "      Or restart your terminal or run: source ~/.bashrc (or ~/.zshrc)"
    echo
fi  