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

# Function: check_internet_connectivity
# Verifies outbound connectivity by trying GitHub (curl) or ICMP ping as fallback.
check_internet_connectivity() {
    log_message "Checking internet connectivity..."
    if command_exists curl; then
        if ! curl -sSf https://github.com >/dev/null 2>&1; then
            handle_error 1 "No internet connectivity or GitHub unreachable."
        fi
    else
        if command_exists ping; then
            if ! ping -c1 -W2 8.8.8.8 >/dev/null 2>&1; then
                handle_error 1 "No internet connectivity (ping to 8.8.8.8 failed)."
            fi
        else
            handle_error 1 "Neither curl nor ping are available to verify connectivity."
        fi
    fi
    log_message "Internet connectivity verified."
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
# Checks if Python 3.12 is present, and if not, installs it.
check_python_installable() {
    log_message "Checking for python3.12..."
    if command_exists python3.12; then
        PYTHON_CMD="python3.12"
        PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
        log_message "Using python3.12: $PYTHON_CMD ($PYTHON_VERSION)"
    else
        log_message "python3.12 not found. Attempting to install it..."
        sudo apt-get update
        sudo apt-get install -y software-properties-common || handle_error $? "Failed to install software-properties-common."
        sudo add-apt-repository -y ppa:deadsnakes/ppa || handle_error $? "Failed to add deadsnakes PPA."
        sudo apt-get update
        sudo apt-get install -y python3.12 python3.12-venv python3.12-pip || handle_error $? "Failed to install python3.12."
        if command_exists python3.12; then
            PYTHON_CMD="python3.12"
            PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
            log_message "Successfully installed python3.12: $PYTHON_CMD ($PYTHON_VERSION)"
        else
            handle_error 1 "python3.12 installation failed. Please install it manually."
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
    check_internet_connectivity
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
if ! command_exists nvidia-container-toolkit && ! command_exists nvidia-ctk; then
    log_message "NVIDIA Container Toolkit not found. Installing..."
    # Parse ID and VERSION_ID from /etc/os-release to construct the correct repo URL
    . /etc/os-release
    UBUNTU_VERSION_SHORT="${VERSION_ID}"
    # Fallback if VERSION_ID not present
    if [ -z "$UBUNTU_VERSION_SHORT" ]; then
        UBUNTU_VERSION_SHORT=$(lsb_release -rs 2>/dev/null || true)
    fi
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/ubuntu${UBUNTU_VERSION_SHORT}/arm64/nvidia-container-toolkit.list | \
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

# Check if the current llama_cpp image supports --mmproj
log_message "Checking if the current llama_cpp image supports --mmproj..."
MM_SUPPORT_CHECK_OUTPUT=$(mktemp)
# Run the command and capture both stdout and stderr
jetson-containers run $(autotag llama_cpp) llama-server --mmproj > "$MM_SUPPORT_CHECK_OUTPUT" 2>&1
MM_SUPPORT_RESULT=$?

if grep -q "error: invalid argument: --mmproj" "$MM_SUPPORT_CHECK_OUTPUT"; then
    log_message "Current llama_cpp image does NOT support --mmproj. Proceeding with custom build."
    NEED_CUSTOM_LLAMA_BUILD=1
elif grep -q "error while handling argument \"--mmproj\": expected value for argument" "$MM_SUPPORT_CHECK_OUTPUT"; then
    log_message "Current llama_cpp image supports --mmproj. Skipping custom build."
    NEED_CUSTOM_LLAMA_BUILD=0
else
    log_error "Could not determine --mmproj support from output. Output was:"
    cat "$MM_SUPPORT_CHECK_OUTPUT"
    log_message "Proceeding with custom build to be safe."
    NEED_CUSTOM_LLAMA_BUILD=1
fi
rm -f "$MM_SUPPORT_CHECK_OUTPUT"

# Only launch the build container if the image does not already exist and NEED_CUSTOM_LLAMA_BUILD=1
# decide if we need a custom build
if [ "$NEED_CUSTOM_LLAMA_BUILD" = "1" ]; then
    # find latest commit on llama.cpp master
    LLAMA_REPO="https://github.com/ggerganov/llama.cpp.git"
    LATEST_SHA=$(git ls-remote "$LLAMA_REPO" HEAD | awk '{print $1}')
    SHORT_SHA=${LATEST_SHA:0:7}
    IMAGE_TAG="my-llama-build-mmsupport:${SHORT_SHA}"

    if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${IMAGE_TAG}$"; then
        log_message "Building ${IMAGE_TAG} (latest llama.cpp @ ${SHORT_SHA})..."

        # run a build container that checks out the exact commit
        nohup script -q -c "jetson-containers run --name my-llama-build-mmsupport $(autotag llama_cpp) bash -c '
            set -euo pipefail
            apt-get update
            apt-get install -y git cmake build-essential
            rm -rf /opt/llama.cpp
            git clone $LLAMA_REPO /opt/llama.cpp
            cd /opt/llama.cpp
            git fetch origin ${LATEST_SHA}
            git checkout ${LATEST_SHA}
            rm -rf build && mkdir build && cd build
            cmake .. -DGGML_CUDA=ON -DLLAVA_BUILD=ON -DLLAMA_BUILD_SERVER=ON
            make -j\$(nproc)
            make install
            echo \"/usr/local/lib\" > /etc/ld.so.conf.d/local.conf
            ldconfig
            echo ${LATEST_SHA} > /opt/llama.cpp/.built_commit
            touch /tmp/build_complete
            tail -f /dev/null
        '" /dev/null > /tmp/build.log 2>&1 &

        # wait for container to appear (timeout after 60s)
        for i in {1..12}; do
            docker ps -a --format '{{.Names}}' | grep -q "^my-llama-build-mmsupport$" && {
                log_message "Container my-llama-build-mmsupport is now running."; break; }
            log_message "Waiting for container my-llama-build-mmsupport to start..."
            sleep 5
        done

        if ! docker ps -a --format '{{.Names}}' | grep -q "^my-llama-build-mmsupport$"; then
            log_error "Container my-llama-build-mmsupport did not start within expected time."
            exit 1
        fi

        # wait for build to finish
        while true; do
            if docker exec my-llama-build-mmsupport test -f /tmp/build_complete 2>/dev/null; then
                log_message "Build complete! Committing container..."
                docker commit \
                    --change "LABEL llama_repo=${LLAMA_REPO}" \
                    --change "LABEL llama_commit=${LATEST_SHA}" \
                    --change "LABEL llama_commit_short=${SHORT_SHA}" \
                    my-llama-build-mmsupport "${IMAGE_TAG}"

                # also update 'latest' to this commit for convenience
                docker tag "${IMAGE_TAG}" my-llama-build-mmsupport:latest

                log_message "Committed to ${IMAGE_TAG} and retagged :latest"
                docker stop my-llama-build-mmsupport >/dev/null
                docker rm my-llama-build-mmsupport >/dev/null
                break
            fi
            log_message "Waiting for build to finish inside container..."
            sleep 10
        done
    else
        log_message "Image ${IMAGE_TAG} already exists (up to date). Skipping build."
    fi
else
    log_message "Skipping custom build of llama_cpp; native image supports --mmproj."
fi

# Step 5: Create llama-server wrapper
log_message "Creating llama-server wrapper script..."
LLAMA_WRAPPER_DIR="$HOME/.local/bin"
mkdir -p "$LLAMA_WRAPPER_DIR"

if [ "$NEED_CUSTOM_LLAMA_BUILD" = "1" ]; then
    CONTAINER="$(autotag my-llama-build-mmsupport)"
else
    CONTAINER="$(autotag llama_cpp)"
fi

JETSON_SH_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")"
log_message "Using project directory based on jetson.sh location: $JETSON_SH_DIR"

echo '#!/bin/bash' > "$LLAMA_WRAPPER_DIR/llama-server"
echo '# Wrapper script to run llama-server using jetson-containers docker image.' >> "$LLAMA_WRAPPER_DIR/llama-server"
echo '' >> "$LLAMA_WRAPPER_DIR/llama-server"
echo "PROJECT_DIR=\"$JETSON_SH_DIR\"" >> "$LLAMA_WRAPPER_DIR/llama-server"
echo "MODELS_DIR=\"$HOME/.eternal-zoo/models\"" >> "$LLAMA_WRAPPER_DIR/llama-server"
echo "CONTAINER=\"$CONTAINER\"" >> "$LLAMA_WRAPPER_DIR/llama-server"
echo '' >> "$LLAMA_WRAPPER_DIR/llama-server"
echo '# Mount model and template directories for Docker.' >> "$LLAMA_WRAPPER_DIR/llama-server"
echo 'PROJECT_MOUNT="-v $PROJECT_DIR:$PROJECT_DIR"' >> "$LLAMA_WRAPPER_DIR/llama-server"
echo 'MODEL_MOUNT="-v $MODELS_DIR:$MODELS_DIR"' >> "$LLAMA_WRAPPER_DIR/llama-server"
echo '' >> "$LLAMA_WRAPPER_DIR/llama-server"
echo 'docker run --runtime nvidia -it --rm --network=host $PROJECT_MOUNT $MODEL_MOUNT $CONTAINER llama-server "$@"' >> "$LLAMA_WRAPPER_DIR/llama-server"

chmod +x "$LLAMA_WRAPPER_DIR/llama-server"
log_message "llama-server wrapper created at $LLAMA_WRAPPER_DIR/llama-server"

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

# Step 6: Create and activate virtual environment
VENV_PATH=".eternal-zoo"
log_message "Creating virtual environment '.eternal-zoo'..."
"$PYTHON_CMD" -m venv "$VENV_PATH" || handle_error $? "Failed to create virtual environment"

log_message "Activating virtual environment..."
source "$VENV_PATH/bin/activate" || handle_error $? "Failed to activate virtual environment"
log_message "Virtual environment activated."

# Step 6: Install eternal-zoo
log_message "Installing eternal-zoo..."
pip install . || handle_error $? "Failed to install eternal-zoo."
log_message "eternal-zoo installed successfully."

log_message "Setup completed successfully."
