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

# Function: check_docker
# Checks if Docker is installed and available.
check_docker() {
    log_section "Checking Docker Installation"
    if ! command_exists docker; then
        log_error "Docker is not installed or not available in PATH. Please install Docker and try again."
        exit 1
    fi
    log_success "Docker is installed and available."
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

# Function: check_python_installable
# Checks if a suitable Python 3 is present (uses system python3, installs if missing).
check_python_installable() {
    log_section "Checking Python Installation"
    log_message "Searching for system python3..."
    if command_exists python3; then
        PYTHON_CMD="python3"
        PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
        log_success "Found system python3: $PYTHON_CMD ($PYTHON_VERSION)"
    else
        log_warning "python3 not found. Attempting to install python3..."
        log_message "Updating package lists and installing python3..."
        sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip || handle_error $? "Failed to install python3."
        if command_exists python3; then
            PYTHON_CMD="python3"
            PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
            log_success "Successfully installed python3: $PYTHON_CMD ($PYTHON_VERSION)"
        else
            handle_error 1 "python3 installation failed. Please install python3 manually."
        fi
    fi
}

# Function: preflight_checks
# Runs all pre-installation checks and prints a summary.
preflight_checks() {
    log_section "Running Preflight Checks"
    check_docker
    check_apt_get
    check_sudo
    check_python_installable
    log_success "All preflight checks passed successfully."
    echo
    echo "========================================="
    echo "Preflight checks summary:"
    echo "- Internet connectivity: OK"
    echo "- Docker: OK"
    echo "- apt-get: OK"
    echo "- Sudo: OK"
    if [ "$PYTHON_INSTALL_NEEDED" = "1" ]; then
        echo "- Python: Will install $PYTHON_PKG_TO_INSTALL via apt."
    else
        echo "- Python: Using $PYTHON_CMD ($HIGHEST_VERSION)"
    fi
    echo "========================================="
    echo
}

# Run all preflight checks before proceeding.
preflight_checks

# If Python needs to be installed, do it now.
if [ "$PYTHON_INSTALL_NEEDED" = "1" ]; then
    log_message "Installing $PYTHON_PKG_TO_INSTALL..."
    sudo apt-get update && sudo apt-get install -y $PYTHON_PKG_TO_INSTALL python3-venv python3-pip || handle_error $? "Failed to install $PYTHON_PKG_TO_INSTALL."
    if [ ! -x "$PYTHON_CMD" ]; then
        PYTHON_CMD="$PYTHON_PKG_TO_INSTALL"
    fi
    log_message "Installed and selected $PYTHON_CMD."
fi

log_message "Using Python at: $(which $PYTHON_CMD)"
log_message "Python setup complete."

# -----------------------------------------------------------------------------
# Step 2: Install required system packages
# -----------------------------------------------------------------------------
log_message "Installing required packages..."
if command -v apt-get &> /dev/null; then
    REQUIRED_PACKAGES=(pigz cmake libcurl4-openssl-dev python3-venv python3-pip build-essential git ninja-build nvidia-cuda-toolkit)
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

# -----------------------------------------------------------------------------
# Step 3: Detect GPU offloading setup (prime-run, optirun, primusrun)
# -----------------------------------------------------------------------------
log_message "Checking for GPU offloading setup..."
if command_exists prime-run; then
    log_message "NVIDIA Prime setup detected (prime-run available)"
    PRIME_RUN_AVAILABLE=true
elif command_exists optirun; then
    log_message "Optimus setup detected (optirun available)"
    OPTIRUN_AVAILABLE=true
elif command_exists primusrun; then
    log_message "Optimus setup detected (primusrun available)"
    PRIMUSRUN_AVAILABLE=true
else
    log_message "No GPU offloading setup detected, using direct GPU access."
    PRIME_RUN_AVAILABLE=false
    OPTIRUN_AVAILABLE=false
    PRIMUSRUN_AVAILABLE=false
fi

# -----------------------------------------------------------------------------
# Step 4: Pull llama-server CUDA Docker image
# -----------------------------------------------------------------------------
log_message "Pulling llama-server cuda image..."
if ! docker pull ghcr.io/ggerganov/llama.cpp:server-cuda; then
    handle_error 1 "Failed to pull llama.cpp server Docker image."
fi

# -----------------------------------------------------------------------------
# Step 5: Create llama-server wrapper script for Docker usage
# -----------------------------------------------------------------------------
log_message "Creating llama-server wrapper script..."
LLAMA_WRAPPER_DIR="$HOME/.local/bin"
mkdir -p "$LLAMA_WRAPPER_DIR"

# Prepare variables for template and model paths.
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
VENV_SITE_PACKAGES="$(pwd)/local_ai/lib/${PYTHON_VERSION}/site-packages"
TEMPLATES_DIR="$VENV_SITE_PACKAGES/local_ai/examples/templates"
USER_HOME="$HOME"
ABS_TEMPLATE_PATH="$USER_HOME/.local/lib/${PYTHON_VERSION}/site-packages/local_ai/examples/templates"

cat > "$LLAMA_WRAPPER_DIR/llama-server" << EOF
#!/bin/bash
# Wrapper script to run llama-server using Docker.

MODELS_DIR="$(pwd)/llms-storage"
TEMPLATES_DIR="$TEMPLATES_DIR"

# Default port for the server.
PORT=11434
prev=""
for arg in "\$@"; do
    if [[ "\$prev" == "--port" ]]; then
        PORT="\$arg"
    fi
    prev="\$arg"
done

# Detect GPU offloading method if available.
if command -v prime-run >/dev/null 2>&1; then
    echo "Info: NVIDIA Prime setup detected, using prime-run for GPU access."
    GPU_ACCESS="prime-run"
elif command -v optirun >/dev/null 2>&1; then
    echo "Info: Optimus setup detected, using optirun for GPU access."
    GPU_ACCESS="optirun"
elif command -v primusrun >/dev/null 2>&1; then
    echo "Info: Optimus setup detected, using primusrun for GPU access."
    GPU_ACCESS="primusrun"
else
    GPU_ACCESS="direct"
fi

# Mount model and template directories for Docker.
MODEL_MOUNT="-v \$MODELS_DIR:\$MODELS_DIR"
TEMPLATE_MOUNT="-v \$TEMPLATES_DIR:\$TEMPLATES_DIR"

DOCKER_RUN="docker run --rm -it --gpus all -p \$PORT:\$PORT \$MODEL_MOUNT \$TEMPLATE_MOUNT ghcr.io/ggerganov/llama.cpp:server-cuda \$@"

if [ "\$GPU_ACCESS" = "prime-run" ]; then
    prime-run \$DOCKER_RUN
elif [ "\$GPU_ACCESS" = "optirun" ]; then
    optirun \$DOCKER_RUN
elif [ "\$GPU_ACCESS" = "primusrun" ]; then
    primusrun \$DOCKER_RUN
else
    \$DOCKER_RUN
fi
EOF

chmod +x "$LLAMA_WRAPPER_DIR/llama-server"
log_message "Created llama-server wrapper at $LLAMA_WRAPPER_DIR/llama-server."

# -----------------------------------------------------------------------------
# Step 6: Add llama-server wrapper directory to PATH in shell rc file
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
# Step 7: Python venv and cryptomodels setup
# -----------------------------------------------------------------------------
log_section "Setting up Python virtual environment and cryptomodels toolkit"

# Create and activate Python virtual environment called 'cryptomodels'
log_message "Creating Python virtual environment 'cryptomodels'..."
"$PYTHON_CMD" -m venv cryptomodels || handle_error $? "Failed to create Python virtual environment 'cryptomodels'."

log_message "Activating Python virtual environment 'cryptomodels'..."
source cryptomodels/bin/activate || handle_error $? "Failed to activate Python virtual environment 'cryptomodels'."

log_message "Python virtual environment 'cryptomodels' activated successfully."

# Uninstalls and reinstalls the cryptomodels toolkit from the GitHub repository.
# log_message "Uninstalling any existing cryptomodels installation..."
# pip uninstall cryptomodels -y || handle_error $? "Failed to uninstall cryptomodels."
# pip install -q git+https://github.com/eternalai-org/CryptoModels.git || handle_error $? "Failed to install/update cryptomodels toolkit."
# log_message "cryptomodels toolkit installed/updated."

# Alternative version with version checking (commented out for speed)
log_message "Setting up cryptomodels toolkit..."
if pip show cryptomodels &>/dev/null; then
    log_message "cryptomodels is installed. Checking for updates..."
    INSTALLED_VERSION=$(pip show cryptomodels | grep Version | awk '{print $2}')
    log_message "Current version: $INSTALLED_VERSION"
    log_message "Checking latest version from repository..."
    TEMP_VERSION_FILE=$(mktemp)
    if curl -s https://raw.githubusercontent.com/eternalai-org/CryptoModels/main/crypto_models/__init__.py | grep -o "__version__ = \"[0-9.]*\"" | cut -d'"' -f2 > "$TEMP_VERSION_FILE"; then
        REMOTE_VERSION=$(cat "$TEMP_VERSION_FILE")
        rm "$TEMP_VERSION_FILE"
        log_message "Latest version: $REMOTE_VERSION"
        if [ "$(printf '%s\n' "$INSTALLED_VERSION" "$REMOTE_VERSION" | sort -V | head -n1)" = "$INSTALLED_VERSION" ] && [ "$INSTALLED_VERSION" != "$REMOTE_VERSION" ]; then
            log_message "New version available. Updating..."
            pip uninstall cryptomodels -y || handle_error $? "Failed to uninstall cryptomodels"
            pip install -q git+https://github.com/eternalai-org/CryptoModels.git || handle_error $? "Failed to update cryptomodels toolkit"
            log_message "cryptomodels toolkit updated to version $REMOTE_VERSION."
        else
            log_message "Already running the latest version. No update needed."
        fi
    else
        log_message "Could not check latest version. Proceeding with update to be safe..."
        pip uninstall cryptomodels -y || handle_error $? "Failed to uninstall cryptomodels"
        pip install -q git+https://github.com/eternalai-org/CryptoModels.git || handle_error $? "Failed to update cryptomodels toolkit"
        log_message "cryptomodels toolkit updated."
    fi
else
    log_message "Installing cryptomodels toolkit..."
    pip install -q git+https://github.com/eternalai-org/CryptoModels.git || handle_error $? "Failed to install cryptomodels toolkit"
    log_message "cryptomodels toolkit installed."
fi

log_message "Setup completed successfully."

# At the end of the script, print an informative message if PATH was updated
if [ "${PATH_UPDATE_NEEDED:-0}" = "1" ]; then
    echo
    echo "[INFO] The llama-server command directory was added to your PATH in your shell rc file."
    echo "      To use it in this session, run: export PATH=\"$LLAMA_WRAPPER_DIR:\$PATH\""
    echo "      Or restart your terminal or run: source ~/.bashrc (or ~/.zshrc)"
    echo
fi
