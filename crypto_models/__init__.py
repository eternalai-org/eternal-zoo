import os
import shutil
import warnings
from loguru import logger
from crypto_models._version import __version__
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning)

# Import template functions for easier access


COMMAND_DIRS = [
    "/opt/homebrew/bin",
    os.path.join(os.path.expanduser("~"), "homebrew", "bin"),
]

# Get the current PATH
current_path = os.environ.get("PATH", "")

# Create a search path with COMMAND_DIRS followed by the system's PATH
search_path = os.pathsep.join(COMMAND_DIRS + [current_path])

def find_and_set_command(cmd_name, env_var_name, search_path):
    """
    Find a command in the search path, set its environment variable, and return its path.
    
    Args:
        cmd_name (str): Name of the command to find.
        env_var_name (str): Environment variable name to set with the command path.
        search_path (str): Path string to search for the command.
    
    Returns:
        str: Path to the command if found.
    
    Raises:
        RuntimeError: If the command is not found or an error occurs.
    """
    try:
        cmd_path = shutil.which(cmd_name, path=search_path)
        if not cmd_path:
            logger.error(f"{cmd_name} command not found in command directories or PATH")
            raise RuntimeError(f"{cmd_name} command not found in command directories or PATH")
        os.environ[env_var_name] = cmd_path
        return cmd_path
    except Exception as e:
        logger.error(f"Failed to find {cmd_name}: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to find {cmd_name}: {str(e)}")

# Define required commands and their corresponding environment variables
required_commands = [
    ("llama-server", "LLAMA_SERVER"),
    ("tar", "TAR_COMMAND"),
    ("pigz", "PIGZ_COMMAND"),
    ("cat", "CAT_COMMAND")
]

# Find all required commands and set their environment variables
for cmd_name, env_var_name in required_commands:
    find_and_set_command(cmd_name, env_var_name, search_path)