#!/bin/bash
set -o pipefail

# Logging functions
log_message() {
    local message="$1"
    if [[ -n "${message// }" ]]; then
        echo "[LAUNCHER_LOGGER] [MODEL_INSTALL_LLAMA] --message \"$message\""
    fi
}

log_error() {
    local message="$1"
    if [[ -n "${message// }" ]]; then
        echo "[LAUNCHER_LOGGER] [MODEL_INSTALL_LLAMA] --error \"$message\"" >&2
    fi
}

# Error handling function
handle_error() {
    local exit_code=$1
    local error_msg=$2
    log_error "$error_msg (Exit code: $exit_code)"
    
    # Clean up if needed
    if [[ -n "$VIRTUAL_ENV" ]]; then
        log_message "Deactivating virtual environment..."
        deactivate 2>/dev/null || true
    fi
    
    exit $exit_code
}

command_exists() {
    command -v "$1" &> /dev/null
}

export PATH="/opt/homebrew/bin/:$PATH"
export PATH="$HOME/homebrew/bin:$PATH"

# Step 1: Ensure Homebrew is installed and set PATH
if ! command_exists brew; then
    log_error "Homebrew is not installed. Please install Homebrew first: https://brew.sh/"
    exit 1
fi

BREW_PREFIX=$(brew --prefix)
export PATH="$BREW_PREFIX/bin:$PATH"
log_message "Homebrew found at $BREW_PREFIX. PATH updated for this session."

# Step 2: Check system Python version and decide on installation
PYTHON_CMD=""
log_message "Searching for suitable Python (>= 3.11) in PATH..."

# Find all python3* executables in PATH and select the highest version >= 3.11
HIGHEST_VERSION=""
HIGHEST_CMD=""
PYTHON_CANDIDATES=$(compgen -c | grep -E '^python3(\.[0-9]+)?$' | sort -u)
for candidate in $PYTHON_CANDIDATES; do
    if command_exists "$candidate"; then
        VERSION=$($candidate --version 2>&1 | awk '{print $2}')
        MAJOR=$(echo "$VERSION" | cut -d. -f1)
        MINOR=$(echo "$VERSION" | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 11 ]; then
            # Compare versions to select the highest
            if [[ -z "$HIGHEST_VERSION" ]] || [[ $(printf '%s\n' "$VERSION" "$HIGHEST_VERSION" | sort -V | tail -n1) == "$VERSION" ]]; then
                HIGHEST_VERSION="$VERSION"
                HIGHEST_CMD="$candidate"
            fi
        fi
    fi
done

if [ -n "$HIGHEST_CMD" ]; then
    log_message "Found suitable Python: $HIGHEST_CMD ($HIGHEST_VERSION)"
    PYTHON_CMD="$HIGHEST_CMD"
fi

if [ -z "$PYTHON_CMD" ]; then
    handle_error 1 "No suitable Python (>= 3.11) found in PATH. Please install Python 3.11 or higher."
fi

log_message "Using Python at: $(which $PYTHON_CMD)"
log_message "Python setup complete."

# Step 3: Update PATH in .zshrc for future sessions
log_message "Checking if PATH update is needed in .zshrc..."
if ! grep -q "export PATH=\"/usr/local/bin:\$PATH\"" ~/.zshrc 2>/dev/null; then
    if [ -f ~/.zshrc ]; then
        log_message "Backing up current .zshrc..."
        cp ~/.zshrc ~/.zshrc.backup.$(date +%Y%m%d%H%M%S) || handle_error $? "Failed to backup .zshrc"
    else
        log_message "No existing .zshrc found. Skipping backup."
    fi
    
    log_message "Updating PATH in .zshrc for brew at /usr/local/bin..."
    echo "export PATH=\"/usr/local/bin:\$PATH\"" >> ~/.zshrc || handle_error $? "Failed to update .zshrc"
    log_message "Please restart your terminal or run 'source ~/.zshrc' for future sessions."
else
    log_message "PATH already updated in .zshrc."
fi

# Step 4: Install pigz
log_message "Checking for pigz installation..."
if command_exists pigz; then
    log_message "pigz is already installed."
else
    log_message "Installing pigz..."
    brew install pigz || handle_error $? "Failed to install pigz"
    log_message "pigz installed successfully."
fi

# Step 5: Install llama.cpp
log_message "Checking llama.cpp version..."
if command_exists llama-cli; then
    INSTALLED_VERSION=$(llama-cli --version 2>&1 | grep -oE 'version: [0-9]+' | cut -d' ' -f2 || echo "0")
    log_message "Current llama.cpp version: $INSTALLED_VERSION"
    
    # Get latest version from the formula file
    LATEST_VERSION=$(grep -oE 'tag: *"b[0-9]+"' llama.cpp.rb | sed 's/tag: *"b//;s/"//' || echo "0")
    log_message "Latest available version: $LATEST_VERSION"
    
    if [ "$INSTALLED_VERSION" -ne "$LATEST_VERSION" ]; then
        log_message "Version mismatch detected. Reinstalling llama.cpp to match formula version..."
        brew uninstall --force llama.cpp
        brew install llama.cpp.rb || handle_error $? "Failed to install llama.cpp"
    else
        log_message "Already running the correct version of llama.cpp."
    fi
else
    log_message "llama.cpp not found. Installing..."
    brew install llama.cpp.rb || handle_error $? "Failed to install llama.cpp"
fi

log_message "Verifying llama.cpp version..."
hash -r
llama-cli --version || handle_error $? "llama.cpp verification failed"
log_message "llama.cpp setup complete."


# Step 6: Create and activate virtual environment
log_message "Creating virtual environment 'cryptomodels'..."
"$PYTHON_CMD" -m venv cryptomodels || handle_error $? "Failed to create virtual environment"

log_message "Activating virtual environment..."
source cryptomodels/bin/activate || handle_error $? "Failed to activate virtual environment"
log_message "Virtual environment activated."


# Step 7: Install cryptomodels toolkit
log_message "Setting up cryptomodels toolkit..."
if pip show cryptomodels &>/dev/null; then
    log_message "cryptomodels is installed. Checking for updates..."
    
    # Get installed version
    INSTALLED_VERSION=$(pip show cryptomodels | grep Version | awk '{print $2}')
    log_message "Current version: $INSTALLED_VERSION"
    
    # Get remote version (from GitHub repository without installing)
    log_message "Checking latest version from repository..."
    TEMP_VERSION_FILE=$(mktemp)
    if curl -s https://raw.githubusercontent.com/eternalai-org/CryptoModels/main/crypto_models/__init__.py | grep -o "__version__ = \"[0-9.]*\"" | cut -d'"' -f2 > "$TEMP_VERSION_FILE"; then
        REMOTE_VERSION=$(cat "$TEMP_VERSION_FILE")
        rm "$TEMP_VERSION_FILE"
        
        log_message "Latest version: $REMOTE_VERSION"
        
        # Compare versions
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