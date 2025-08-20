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

# Version comparison function
compare_versions() {
    local installed="v$1"
    local remote="$2"
    
    # Validate version strings (basic semver check)
    if [[ ! "$installed" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?([.-][a-zA-Z0-9]+)*$ ]]; then
        log_error "Invalid installed version format: $installed"
        return 2
    fi
    
    if [[ ! "$remote" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?([.-][a-zA-Z0-9]+)*$ ]]; then
        log_error "Invalid remote version format: $remote"
        return 2
    fi
    
    # Return 0 if update needed (remote > installed), 1 if no update needed, 2 if error
    if [ "$installed" != "$remote" ] && [ "$(printf '%s\n' "$installed" "$remote" | sort -V | head -n1)" = "$installed" ]; then
        return 0  # Update needed
    else
        return 1  # No update needed
    fi
}

# Generic package update function
update_package() {
    local package_name="$1"
    local github_url="$2"
    local version_source_url="$3"
    local version_regex="$4"
    local install_cmd="$5"
    local specific_tag="$6"  # Optional: if provided, install this specific tag
    
    log_message "Checking $package_name installation..."
    
    # If specific tag is provided, use tag-based installation
    if [ -n "$specific_tag" ]; then
        install_package_by_tag "$package_name" "$github_url" "$specific_tag"
        return $?
    fi
    
    # Otherwise, use version-based update logic
    if pip show "$package_name" &>/dev/null; then
        log_message "$package_name is installed. Checking current version..."
        
        # Get installed version
        local installed_version=$(pip show "$package_name" | grep Version | awk '{print $2}')
        log_message "Current $package_name version: $installed_version"
        
        # Check if versions are actually different (only if specific tag is provided)
        if [ -n "$specific_tag" ]; then
            if [ "$installed_version" = "$specific_tag" ]; then
                log_message "$package_name is already at the specified version ($installed_version). No update needed."
            else
                log_message "Version mismatch detected. Updating $package_name from $installed_version to $specific_tag..."
                if pip uninstall "$package_name" -y && eval "$install_cmd"; then
                    log_message "$package_name updated to version $specific_tag."
                else
                    log_error "Failed to update $package_name. Continuing with installation..."
                fi
            fi
        else
            log_message "No specific tag provided. Skipping version check for $package_name."
        fi
    else
        log_message "Installing $package_name..."
        if eval "$install_cmd"; then
            log_message "$package_name installed successfully."
        else
            log_error "Failed to install $package_name. Continuing with installation..."
        fi
    fi
}

# Install package by specific tag
install_package_by_tag() {
    local package_name="$1"
    local github_url="$2"
    local tag="$3"
    
    log_message "Checking $package_name installation for tag: $tag"
    
    # Check if package is already installed
    if pip show "$package_name" &>/dev/null; then
        local installed_version=$(pip show "$package_name" | grep Version | awk '{print $2}')
        log_message "Current $package_name version: $installed_version"
        
        # Try to get more detailed information about the installed package
        local installed_location=$(pip show "$package_name" | grep Location | awk '{print $2}')
        local is_git_install=false
        local installed_git_ref=""
        
        # Check if this is a git installation
        if [ -d "$installed_location/$package_name/.git" ]; then
            is_git_install=true
            # Try to get the git reference (tag/branch/commit)
            if [ -f "$installed_location/$package_name/.git/HEAD" ]; then
                installed_git_ref=$(cat "$installed_location/$package_name/.git/HEAD" 2>/dev/null | sed 's|refs/heads/||' | sed 's|refs/tags/||')
            fi
        fi
        
        # Check for tag mismatch
        local needs_update=false
        local mismatch_reason=""
        
        if [ "$is_git_install" = true ] && [ -n "$installed_git_ref" ]; then
            # For git installations, compare the git reference
            if [ "$installed_git_ref" != "$tag" ]; then
                needs_update=true
                mismatch_reason="Git reference mismatch: installed=$installed_git_ref, desired=$tag"
            fi
        else
            # For non-git installations or when we can't determine git ref, compare version
            if [ "$installed_version" != "$tag" ]; then
                needs_update=true
                mismatch_reason="Version mismatch: installed=$installed_version, desired=$tag"
            fi

        fi
        
        if [ "$needs_update" = true ]; then
            log_message "Tag mismatch detected: $mismatch_reason"
            log_message "Updating $package_name to tag $tag..."
            pip uninstall "$package_name" -y || {
                log_error "Failed to uninstall $package_name"
                return 1
            }
        else
            log_message "$package_name is already installed with the correct tag ($tag). No update needed."
            return 0
        fi
    else
        log_message "$package_name not found. Installing from tag $tag..."
    fi
    
    # Install from specific tag
    local install_cmd="pip install git+${github_url}@v${tag}"
    log_message "Installing with command: $install_cmd"
    
    if eval "$install_cmd"; then
        log_message "$package_name installed successfully from tag $tag."
        return 0
    else
        log_error "Failed to install $package_name from tag $tag."
        return 1
    fi
}

# List available tags for a GitHub repository
list_available_tags() {
    local repo_url="$1"
    local package_name="$2"
    
    log_message "Fetching available tags for $package_name..."
    
    # Extract owner/repo from GitHub URL
    local repo_path=$(echo "$repo_url" | sed 's|https://github.com/||' | sed 's|\.git$||')
    
    # Use GitHub API to get tags
    local api_url="https://api.github.com/repos/$repo_path/tags"
    local temp_file=$(mktemp)
    
    if curl -s --connect-timeout 10 --max-time 30 "$api_url" > "$temp_file" 2>/dev/null; then
        local tags=$(cat "$temp_file" | grep '"name"' | cut -d'"' -f4 | head -10)
        rm -f "$temp_file"
        
        if [ -n "$tags" ]; then
            log_message "Available tags for $package_name (showing latest 10):"
            echo "$tags" | while read -r tag; do
                echo "  - $tag"
            done
        else
            log_message "No tags found for $package_name"
        fi
    else
        rm -f "$temp_file"
        log_error "Failed to fetch tags for $package_name"
    fi
}

# Show detailed installation status for a package
show_package_status() {
    local package_name="$1"
    local desired_tag="$2"
    
    log_message "Checking status of $package_name..."
    
    if pip show "$package_name" &>/dev/null; then
        local installed_version=$(pip show "$package_name" | grep Version | awk '{print $2}')
        local installed_location=$(pip show "$package_name" | grep Location | awk '{print $2}')
        
        log_message "  ✓ $package_name is installed"
        log_message "  Version: $installed_version"
        log_message "  Location: $installed_location"
        
        # Check if it's a git installation
        if [ -d "$installed_location/$package_name/.git" ]; then
            local git_ref=""
            if [ -f "$installed_location/$package_name/.git/HEAD" ]; then
                git_ref=$(cat "$installed_location/$package_name/.git/HEAD" 2>/dev/null | sed 's|refs/heads/||' | sed 's|refs/tags/||')
            fi
            log_message "  Git reference: $git_ref"
            
            if [ -n "$desired_tag" ]; then
                if [ "$git_ref" = "$desired_tag" ]; then
                    log_message "  ✓ Tag match: Installed tag matches desired tag ($desired_tag)"
                else
                    log_message "  ✗ Tag mismatch: Installed tag ($git_ref) != desired tag ($desired_tag)"
                fi
            fi
        else
            log_message "  Installation type: Standard pip package"
            if [ -n "$desired_tag" ]; then
                if [ "$installed_version" = "$desired_tag" ]; then
                    log_message "  ✓ Version match: Installed version matches desired tag ($desired_tag)"
                else
                    log_message "  ✗ Version mismatch: Installed version ($installed_version) != desired tag ($desired_tag)"
                fi
            fi
        fi
    else
        log_message "  ✗ $package_name is not installed"
        if [ -n "$desired_tag" ]; then
            log_message "  Will install from tag: $desired_tag"
        else
            log_message "  Will install latest version"
        fi
    fi
}

# Install mlx-openai-server using pip with specific version
install_mlx_openai_server_pip() {
    local version="$1"
    local package_name="mlx-openai-server"
    
    log_message "Checking $package_name installation..."
    
    if [ -n "$version" ]; then
        log_message "Will install $package_name version: $version"
        
        # Check if already installed with the correct version
        if pip show "$package_name" &>/dev/null; then
            local installed_version=$(pip show "$package_name" | grep Version | awk '{print $2}')
            log_message "Current $package_name version: $installed_version"
            
            if [ "$installed_version" = "$version" ]; then
                log_message "$package_name is already installed with the correct version ($version). No update needed."
                return 0
            else
                log_message "Version mismatch detected. Updating $package_name from $installed_version to $version..."
                pip uninstall "$package_name" -y || {
                    log_error "Failed to uninstall $package_name"
                    return 1
                }
            fi
        fi
        
        # Install specific version
        local install_cmd="pip install ${package_name}==${version}"
        log_message "Installing with command: $install_cmd"
        
        if eval "$install_cmd"; then
            log_message "$package_name installed successfully with version $version."
            return 0
        else
            log_error "Failed to install $package_name version $version."
            return 1
        fi
    else
        log_message "No version specified for $package_name. Installing latest version..."
        
        # Check if already installed
        if pip show "$package_name" &>/dev/null; then
            local installed_version=$(pip show "$package_name" | grep Version | awk '{print $2}')
            log_message "Current $package_name version: $installed_version"
            log_message "$package_name is already installed. No update needed."
            return 0
        fi
        
        # Install latest version
        local install_cmd="pip install ${package_name}"
        log_message "Installing with command: $install_cmd"
        
        if eval "$install_cmd"; then
            log_message "$package_name installed successfully with latest version."
            return 0
        else
            log_error "Failed to install $package_name."
            return 1
        fi
    fi
}

# Configuration section for package versions
# 
# USAGE:
# 1. Edit the variables below to specify exact tags/versions
# 2. Or set environment variables before running the script:
#    export MLX_FLUX_TAG="v1.0.0"
#    export ETERNAL_ZOO_TAG="v0.1.0"
#    ./mac.sh
# 3. Or modify the script to call list_available_tags() to see available versions
#
# Examples:
#   MLX_FLUX_TAG="v1.0.0"        # Install specific version
#   MLX_FLUX_TAG=""              # Install latest version (default)
#   MLX_FLUX_TAG="main"          # Install from main branch
#   MLX_FLUX_TAG="feature-xyz"   # Install from specific branch

# Use environment variable if set, otherwise use default (empty means latest)
: "${MLX_FLUX_TAG:=}"         # Leave empty for latest, or set specific tag
: "${MLX_OPENAI_SERVER_TAG:=}" # Leave empty for latest, or set specific tag
: "${ETERNAL_ZOO_TAG:=}"     # Leave empty for latest, or set specific tag

# Uncomment the lines below to see available tags before installation
# list_available_tags "https://github.com/0x9334/mlx-flux.git" "mlx-flux"
# list_available_tags "https://github.com/eternalai-org/eternal-zoo.git" "eternal-zoo"

# Display current configuration
if [ -n "$MLX_FLUX_TAG" ]; then
    log_message "Will install mlx-flux from tag: $MLX_FLUX_TAG"
else
    log_message "Will install mlx-flux from latest version"
fi

if [ -n "$ETERNAL_ZOO_TAG" ]; then
    log_message "Will install eternal-zoo from tag: $ETERNAL_ZOO_TAG"
else
    log_message "Will install eternal-zoo from latest version"
fi

if [ -n "$MLX_OPENAI_SERVER_TAG" ]; then
    log_message "Will install mlx-openai-server version: $MLX_OPENAI_SERVER_TAG"
else
    log_message "Will install mlx-openai-server from latest version"
fi

# Show current installation status
log_message "Current installation status:"
show_package_status "mlx-flux" "$MLX_FLUX_TAG"
show_package_status "mlx-openai-server" "$MLX_OPENAI_SERVER_TAG"
show_package_status "eternal-zoo" "$ETERNAL_ZOO_TAG"

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

# Step 2: Install Python 3.11
if brew list python@3.11 >/dev/null 2>&1; then
    PYTHON_CMD=$(brew --prefix python@3.11)/bin/python3.11
else
    brew install python@3.11
    PYTHON_CMD=$(brew --prefix python@3.11)/bin/python3.11
fi

log_message "Using Python at: $PYTHON_CMD"

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

#Step 5: Install llama.cpp
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
        brew install llama.cpp || handle_error $? "Failed to install llama.cpp"
    else
        log_message "Already running the correct version of llama.cpp."
    fi
else
    log_message "llama.cpp not found. Installing..."
    brew install llama.cpp || handle_error $? "Failed to install llama.cpp"
fi

# Step 6: Create and activate virtual environment

VENV_PATH=".eternal-zoo"
log_message "Checking if virtual environment exists..."
if [ -d "$VENV_PATH" ]; then
    # Determine Python version used by the existing venv
    if [ -x "$VENV_PATH/bin/python3" ]; then
        VENV_PYTHON="$VENV_PATH/bin/python3"
    elif [ -x "$VENV_PATH/bin/python" ]; then
        VENV_PYTHON="$VENV_PATH/bin/python"
    else
        VENV_PYTHON=""
    fi

    if [ -n "$VENV_PYTHON" ]; then
        VENV_PY_VERSION=$("$VENV_PYTHON" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null || echo "unknown")
        if [ "$VENV_PY_VERSION" != "3.11" ]; then
            log_message "Existing virtual environment uses Python $VENV_PY_VERSION. Removing..."
            rm -rf "$VENV_PATH"
        else
            log_message "Existing virtual environment uses Python 3.11. Keeping it."
        fi
    else
        log_message "Existing virtual environment has no valid Python executable. Removing..."
        rm -rf "$VENV_PATH"
    fi
fi

# Create venv only if it does not exist (or was removed above)
if [ ! -d "$VENV_PATH" ]; then
    log_message "Creating virtual environment 'eternal-zoo'..."
    "$PYTHON_CMD" -m venv "$VENV_PATH" || handle_error $? "Failed to create virtual environment"
fi

log_message "Activating virtual environment..."
source "$VENV_PATH/bin/activate" || handle_error $? "Failed to activate virtual environment"
log_message "Virtual environment activated."

# Step 7: Install mlx-openai-server dependencies
if install_mlx_openai_server_pip "$MLX_OPENAI_SERVER_TAG"; then
    log_message "mlx-openai-server installation completed successfully."
else
    log_error "mlx-openai-server installation failed. This may happen on Intel Macs or due to compatibility issues. Continuing with installation..."
fi

# Step 8: Install eternal-zoo toolkit
update_package "eternal-zoo" "https://github.com/eternalai-org/eternal-zoo.git" "https://raw.githubusercontent.com/eternalai-org/eternal-zoo/main/eternal_zoo/__init__.py" "__version__ = \"[0-9.]*\"" "pip install git+https://github.com/eternalai-org/eternal-zoo.git" "$ETERNAL_ZOO_TAG"

log_message "Setup completed successfully."
log_message "Setup completed successfully."
