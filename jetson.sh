#!/bin/bash
set -o pipefail

# Function: log_message
# Logs informational messages with a specific format.
log_message() {
    local message="$1"
    if [[ -n "${message// }" ]]; then
        echo "[JETSON_LAUNCHER_LOG] --message \"$message\""
    fi
}

# Function: log_error
# Logs error messages with a specific format.
log_error() {
    local message="$1"
    if [[ -n "${message// }" ]]; then
        echo "[JETSON_LAUNCHER_LOG] --error \"$message\"" >&2
    fi
}

# Function: handle_error
# Handles errors, logs the error, deactivates the virtual environment if active, and exits.
handle_error() {
    local exit_code=$1
    local error_msg=$2
    log_error "$error_msg (Exit code: $exit_code)"
    if [[ -n "$VIRTUAL_ENV" ]]; then
        log_message "Deactivating virtual environment..."
        deactivate 2>/dev/null || true
    fi
    exit $exit_code
}

# Function: command_exists
# Checks if a command exists in the system.
command_exists() {
    command -v "$1" &> /dev/null
}

# Function: check_apt_get
# Checks if apt-get is available.
check_apt_get() {
    if ! command_exists apt-get; then
        log_error "apt-get is not available. This script requires Ubuntu or a compatible system."
        exit 1
    fi
    log_message "apt-get is available."
}

# Function: check_sudo
# Checks if the user has sudo privileges.
check_sudo() {
    if ! sudo -n true 2>/dev/null; then
        log_error "This script requires sudo privileges. Please run as a user with sudo access."
        exit 1
    fi
    log_message "Sudo privileges confirmed."
}

# Function: check_jetson_device
# Checks if the script is running on a Jetson device.
check_jetson_device() {
    IS_JETSON=0
    if [[ -f /proc/device-tree/model ]] && grep -qi "NVIDIA Jetson" /proc/device-tree/model; then
        IS_JETSON=1
    fi
    if [[ "$(uname -m)" == "aarch64" ]] && command_exists jetson_release; then
        IS_JETSON=1
    fi
    if [[ "$IS_JETSON" != "1" ]]; then
        handle_error 1 "This script is intended for NVIDIA Jetson devices."
    fi
    log_message "NVIDIA Jetson device detected."
}

# Function: check_python_installable
# Checks if a suitable Python 3 is present (uses system python3, installs if missing).
check_python_installable() {
    log_message "Checking for system python3..."
    if command_exists python3; then
        PYTHON_CMD="python3"
        PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
        log_message "Using system python3: $PYTHON_CMD ($PYTHON_VERSION)"
    else
        log_message "python3 not found. Attempting to install python3..."
        sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip || handle_error $? "Failed to install python3."
        if command_exists python3; then
            PYTHON_CMD="python3"
            PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
            log_message "Successfully installed python3: $PYTHON_CMD ($PYTHON_VERSION)"
        else
            handle_error 1 "python3 installation failed. Please install python3 manually."
        fi
    fi
}

# Function: preflight_checks
# Runs all pre-installation checks and prints a summary.
preflight_checks() {
    log_message "Running preflight checks..."
    check_jetson_device
    check_apt_get
    check_sudo
    check_python_installable
    log_message "All preflight checks passed."
    echo
    echo "========================================="
    echo "Preflight checks summary:"
    echo "- Jetson device: OK"
    echo "- Internet connectivity: OK"
    echo "- apt-get: OK"
    echo "- Sudo: OK"
    echo "- Python: Using $PYTHON_CMD ($PYTHON_VERSION)"
    echo "========================================="
    echo
}

# Run all preflight checks before proceeding.
preflight_checks

# Step 1: Install required system packages
log_message "Installing required packages..."
REQUIRED_PACKAGES=(pigz cmake libcurl4-openssl-dev python3-venv python3-pip)
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
log_message "All required packages installed successfully."

# Step 2: Check NVIDIA Container Toolkit (usually preinstalled on Jetson images)
if ! command_exists nvidia-container-toolkit; then
    log_message "NVIDIA Container Toolkit not found. Installing..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/ubuntu$(lsb_release -rs)/arm64/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
else
    log_message "NVIDIA Container Toolkit already installed."
fi

# Step 3: Check jetson-containers
if command_exists jetson-containers; then
    JETSON_CONTAINERS_PATH="$(jetson-containers root 2>/dev/null)"
    if [ -d "$JETSON_CONTAINERS_PATH/.git" ]; then
        pushd "$JETSON_CONTAINERS_PATH"
        git reset --hard HEAD
        git clean -fd
        git pull
        popd
    else
        log_message "Could not find .git directory in $JETSON_CONTAINERS_PATH, skipping manual update."
    fi
    jetson-containers update || log_error "Failed to update jetson-containers."
else
    if [ -d jetson-containers ]; then
        rm -rf jetson-containers || handle_error $? "Failed to remove existing jetson-containers directory."
    fi
    log_message "Installing jetson-containers CLI..."
    git clone https://github.com/dusty-nv/jetson-containers
    bash jetson-containers/install.sh || handle_error $? "Failed to install jetson-containers"
    export PATH="$HOME/.local/bin:$PATH"
fi

# Ensure autotag is executable
if [ -f "/usr/local/bin/autotag" ]; then
    if [ ! -x "/usr/local/bin/autotag" ]; then
        log_message "autotag is not executable. Attempting to make it executable..."
        sudo chmod +x /usr/local/bin/autotag || handle_error $? "Failed to make /usr/local/bin/autotag executable."
        log_message "autotag is now executable."
    else
        log_message "autotag is already executable."
    fi
else
    log_error "/usr/local/bin/autotag not found. jetson-containers may not be installed correctly."
    handle_error 1 "/usr/local/bin/autotag not found."
fi

# Step 4: Pull/build llama_cpp Docker image
log_message "Building llama_cpp Docker image with jetson-containers..."
jetson-containers run $(autotag llama_cpp) /bin/true || handle_error $? "Failed to pull llama_cpp image"

# Only launch the build container if the image does not already exist
if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^my-llama-build-mmsupport:latest$"; then
    log_message "Building my-llama-build-mmsupport image (container will be launched)..."
    nohup script -q -c "jetson-containers run --name my-llama-build-mmsupport $(autotag llama_cpp) bash -c 'apt-get update && apt-get install -y git cmake build-essential && rm -rf /opt/llama.cpp && git clone https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp && cd /opt/llama.cpp && rm -rf build && mkdir build && cd build && cmake .. -DGGML_CUDA=ON -DLLAVA_BUILD=ON -DLLAMA_BUILD_SERVER=ON && make -j\$(nproc) && make install && echo \"/usr/local/lib\" > /etc/ld.so.conf.d/local.conf && ldconfig && touch /tmp/build_complete && tail -f /dev/null'" /dev/null > /tmp/build.log 2>&1 &

    # Wait for the container to appear (timeout after 60 seconds)
    for i in {1..12}; do
        if docker ps -a --format '{{.Names}}' | grep -q "^my-llama-build-mmsupport$"; then
            log_message "Container my-llama-build-mmsupport is now running."
            break
        fi
        log_message "Waiting for container my-llama-build-mmsupport to start..."
        sleep 5
    done

    # If still not found, exit with error
    if ! docker ps -a --format '{{.Names}}' | grep -q "^my-llama-build-mmsupport$"; then
        log_error "Container my-llama-build-mmsupport did not start within expected time."
        exit 1
    fi

    # Now enter the build-complete waiting loop as before
    while true; do
        OUTPUT=$(docker exec my-llama-build-mmsupport test -f /tmp/build_complete 2>&1)
        if echo "$OUTPUT" | grep -q "No such container"; then
            log_error "No such container: my-llama-build-mmsupport"
            exit 1
        fi
        if docker exec my-llama-build-mmsupport test -f /tmp/build_complete 2>/dev/null; then
            log_message "Build complete! Committing container..."
            docker commit my-llama-build-mmsupport my-llama-build-mmsupport
            log_message "Committed to my-llama-build-mmsupport!"
            docker stop my-llama-build-mmsupport
            break
        fi
        log_message "Waiting for build to finish inside container..."
        sleep 10
    done
else
    log_message "Image my-llama-build-mmsupport:latest already exists. Skipping build container launch."
fi

# Step 5: Create wrapper script for llama-server
log_message "Creating llama-server wrapper script..."
LLAMA_WRAPPER_DIR="$HOME/.local/bin"
mkdir -p "$LLAMA_WRAPPER_DIR"

# Prepare variables for template and model paths.
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
MODELS_DIR="$(pwd)/llms-storage"
TEMPLATES_DIR="$(pwd)/local_ai/examples/templates"
USER_HOME="$HOME"
ABS_TEMPLATE_PATH="$(pwd)/local_ai/lib/$PYTHON_VERSION/site-packages/local_ai/examples/templates"

cat > "$LLAMA_WRAPPER_DIR/llama-server" << EOF
#!/bin/bash
# Wrapper script to run llama-server using jetson-containers docker image.

MODELS_DIR="$MODELS_DIR"
TEMPLATES_DIR="$TEMPLATES_DIR"
ABS_TEMPLATE_PATH="$ABS_TEMPLATE_PATH"
CONTAINER="my-llama-build-mmsupport"

# Mount model and template directories for Docker.
MODEL_MOUNT="-v \$MODELS_DIR:\$MODELS_DIR"
TEMPLATE_MOUNT="-v \$TEMPLATES_DIR:\$TEMPLATES_DIR"
ABS_TEMPLATE_PATH="-v \$ABS_TEMPLATE_PATH:\$ABS_TEMPLATE_PATH"

docker run --runtime nvidia -it --rm --network=host \$MODEL_MOUNT \$TEMPLATE_MOUNT \$ABS_TEMPLATE_PATH \$CONTAINER llama-server "\$@"
EOF
chmod +x "$LLAMA_WRAPPER_DIR/llama-server"
log_message "llama-server wrapper created at $LLAMA_WRAPPER_DIR/llama-server"

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

# Step 6: Python venv and cryptomodels setup
log_section "Setting up Python virtual environment and cryptomodels toolkit"

# Create and activate Python virtual environment called 'cryptomodels'
log_message "Creating Python virtual environment 'cryptomodels'..."
"$PYTHON_CMD" -m venv cryptomodels || handle_error $? "Failed to create Python virtual environment 'cryptomodels'."

log_message "Activating Python virtual environment 'cryptomodels'..."
source cryptomodels/bin/activate || handle_error $? "Failed to activate Python virtual environment 'cryptomodels'."

log_message "Python virtual environment 'cryptomodels' activated successfully."

# Uninstalls and reinstalls the cryptomodels toolkit from the GitHub repository.
log_message "Uninstalling any existing cryptomodels installation..."
pip uninstall cryptomodels -y || handle_error $? "Failed to uninstall cryptomodels."
pip install -q git+https://github.com/eternalai-org/CryptoModels.git || handle_error $? "Failed to install/update cryptomodels toolkit."
log_message "cryptomodels toolkit installed/updated."

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

log_message "Setup complete. You can now use 'llama-server' and your Python virtual environment."

# At the end of the script, print an informative message if PATH was updated
if [ "${PATH_UPDATE_NEEDED:-0}" = "1" ]; then
    echo
    echo "[INFO] The llama-server command directory was added to your PATH in your shell rc file."
    echo "      To use it in this session, run: export PATH=\"$LLAMA_WRAPPER_DIR:\$PATH\""
    echo "      Or restart your terminal or run: source ~/.bashrc (or ~/.zshrc)"
    echo
fi
