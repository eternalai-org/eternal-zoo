import sys
import asyncio
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import print as rprint
    
from crypto_models import __version__
from crypto_models.config import config
from crypto_models.core import CryptoModelsManager
from crypto_models.upload import upload_folder_to_lighthouse
from crypto_models.download import download_model_from_filecoin_async, check_downloaded_model
from crypto_models.preseved_models import PRESERVED_MODELS

manager = CryptoModelsManager()

def print_banner():
    """Display a beautiful banner for the CLI"""
    console = Console()
    banner_text = """
 ██████╗██████╗ ██╗   ██╗██████╗ ████████╗ ██████╗ 
██╔════╝██╔══██╗╚██╗ ██╔╝██╔══██╗╚══██╔══╝██╔═══██╗
██║     ██████╔╝ ╚████╔╝ ██████╔╝   ██║   ██║   ██║
██║     ██╔══██╗  ╚██╔╝  ██╔═══╝    ██║   ██║   ██║
╚██████╗██║  ██║   ██║   ██║        ██║   ╚██████╔╝
 ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝        ╚═╝    ╚═════╝ 
                                                    
███╗   ███╗ ██████╗ ██████╗ ███████╗██╗     ███████╗
████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║     ██╔════╝
██╔████╔██║██║   ██║██║  ██║█████╗  ██║     ███████╗
██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║     ╚════██║
██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗███████║
╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚══════╝
        """
        
    panel = Panel(
        Text(banner_text, style="bold cyan", justify="center"),
        title=f"[bold green]CryptoModels CLI v{__version__}[/bold green]",
        subtitle="[italic]Decentralized AI Model Management[/italic]",
        border_style="bright_blue",
        padding=(1, 2)
    )
    console.print(panel)

def print_success(message):
    """Print success message with styling"""
    rprint(f"[bold green]✅ {message}[/bold green]")

def print_error(message):
    """Print error message with styling"""
    rprint(f"[bold red]❌ {message}[/bold red]")

def print_info(message):
    """Print info message with styling"""
    rprint(f"[bold blue]ℹ️  {message}[/bold blue]")

def print_warning(message):
    """Print warning message with styling"""
    rprint(f"[bold yellow]⚠️  {message}[/bold yellow]")

def show_available_models():
    """Display available models in a beautiful table"""
    console = Console()
    table = Table(title="🤖 Available Preserved Models", border_style="cyan")
    table.add_column("Model Name", style="bold magenta", justify="left")
    table.add_column("Hash", style="dim", justify="left")
    
    for model_name, model_hash in PRESERVED_MODELS.items():
        table.add_row(model_name, model_hash[:16] + "...")
        
    console.print(table)

class CustomHelpFormatter(argparse.HelpFormatter):
    """Custom help formatter for better styling"""
    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super()._format_action_invocation(action)
        default = super()._format_action_invocation(action)
        return f"[bold cyan]{default}[/bold cyan]"

def parse_args():
    """Parse command line arguments with beautiful help formatting"""
    parser = argparse.ArgumentParser(
        description="🚀 CryptoModels - Decentralized AI Model Management Tool",
        formatter_class=CustomHelpFormatter,
        epilog="💡 For more information, visit: https://github.com/your-repo/crypto-models"
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"cryptomodels v{__version__} 🎉"
    )
    
    subparsers = parser.add_subparsers(
        dest='command', 
        help="🛠️  Available commands for managing AI models",
        metavar="COMMAND"
    )
    
    # Model command group
    model_command = subparsers.add_parser(
        "model", 
        help="🤖 Model management operations",
        description="Manage your decentralized AI models"
    )
    model_subparsers = model_command.add_subparsers(
        dest='model_command', 
        help="Model operations",
        metavar="OPERATION"
    )
    
    # Model run command
    run_command = model_subparsers.add_parser(
        "run", 
        help="🚀 Launch a local AI model server",
        description="Start serving a model locally with customizable settings"
    )
    
    run_command.add_argument(
        "model_name", 
        nargs='?', 
        help="🏷️  Model name (e.g., qwen3-1.7b) - automatically mapped to hash"
    )
    run_command.add_argument(
        "--hash", 
        type=str,
        help="🔗 Filecoin hash of the model (alternative to model name)",
        metavar="HASH"
    )
    run_command.add_argument(
        "--port", 
        type=int, 
        default=config.network.DEFAULT_PORT,
        help=f"🌐 Port number for the server (default: {config.network.DEFAULT_PORT})",
        metavar="PORT"
    )
    run_command.add_argument(
        "--host", 
        type=str, 
        default=config.network.DEFAULT_HOST,
        help=f"🏠 Host address for the server (default: {config.network.DEFAULT_HOST})",
        metavar="HOST"
    )
    run_command.add_argument(
        "--context-length", 
        type=int, 
        default=config.model.DEFAULT_CONTEXT_LENGTH,
        help=f"📏 Context length for the model (default: {config.model.DEFAULT_CONTEXT_LENGTH})",
        metavar="LENGTH"
    )
    
    # Model stop command
    stop_command = model_subparsers.add_parser(
        "stop", 
        help="🛑 Stop the running model server",
        description="Gracefully shutdown the currently running model server"
    )
    stop_command.add_argument(
        "--force", 
        action="store_true",
        help="💥 Force kill processes immediately without graceful termination (use when normal stop fails)"
    )
    
    # Model download command
    download_command = model_subparsers.add_parser(
        "download", 
        help="⬇️  Download model from IPFS",
        description="Download and extract model files from the decentralized network"
    )
    download_command.add_argument(
        "--hash", 
        required=True,
        help="🔗 IPFS hash of the model metadata",
        metavar="HASH"
    )
    download_command.add_argument(
        "--chunk-size", 
        type=int, 
        default=config.network.DEFAULT_CHUNK_SIZE,
        help=f"📦 Download chunk size in bytes (default: {config.network.DEFAULT_CHUNK_SIZE})",
        metavar="SIZE"
    )
    download_command.add_argument(
        "--output-dir", 
        type=Path, 
        default=None,
        help="📁 Output directory for model files",
        metavar="DIR"
    )
    
    # Model check command
    check_command = model_subparsers.add_parser(
        "check", 
        help="🔍 Check if model is downloaded",
        description="Check if a model with the specified hash has been downloaded"
    )
    check_command.add_argument(
        "--hash", 
        required=True,
        help="🔗 IPFS hash of the model to check",
        metavar="HASH"
    )
    
    # Model status command
    status_command = model_subparsers.add_parser(
        "status", 
        help="📊 Check running model status",
        description="Display information about the currently running model"
    )
    
    # Model preserve command
    preserve_command = model_subparsers.add_parser(
        "preserve", 
        help="💾 Preserve model to IPFS",
        description="Upload and preserve your model files to the decentralized network"
    )
    preserve_command.add_argument(
        "--task", 
        type=str, 
        default="chat", 
        choices=["chat", "embed", "image-generation", "image-edit"],
        help="🎯 Model task type (default: chat)",
        metavar="TYPE"
    )
    preserve_command.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="🔍 Model config name (default: None), need for image-generation and image-edit models",
        metavar="CONFIG"
    )
    preserve_command.add_argument(
        "--folder-path", 
        type=str, 
        required=True,
        help="📂 Path to folder containing model files",
        metavar="PATH"
    ) 
    preserve_command.add_argument(
        "--zip-chunk-size", 
        type=int, 
        default=512,
        help="🗜️  Chunk size for splitting compressed files in MB (default: 512)",
        metavar="SIZE"
    )
    preserve_command.add_argument(
        "--threads", 
        type=int, 
        default=16,
        help="🧵 Number of compression threads (default: 16)",
        metavar="COUNT"
    )
    preserve_command.add_argument(
        "--max-retries", 
        type=int, 
        default=5,
        help="🔄 Maximum upload retry attempts (default: 5)",
        metavar="NUM"
    )
    preserve_command.add_argument(
        "--hf-repo", 
        type=str, 
        default=None,
        help="🤗 Hugging Face model repository",
        metavar="REPO"
    )
    preserve_command.add_argument(
        "--hf-file", 
        type=str, 
        default=None,
        help="📄 Hugging Face model file",
        metavar="FILE"
    )
    preserve_command.add_argument(
        "--ram", 
        type=float, 
        default=None,
        help="🧠 Required RAM in GB for serving at 4096 context length",
        metavar="GB"
    )
    
    return parser.parse_known_args()

def handle_download(args):
    """Handle model download with beautiful output"""
    print_info(f"Starting download for hash: {args.hash}")
    try:
        asyncio.run(download_model_from_filecoin_async(args.hash))
        print_success("Model downloaded successfully!")
    except Exception as e:
        print_error(f"Download failed: {str(e)}")
        sys.exit(1)

def handle_run(args):
    """Handle model run with enhanced error handling and output"""
    # Determine the hash to use
    if args.hash and args.model_name:
        print_error("Please specify either a model name OR --hash, not both")
        print_info("Usage examples:")
        print("  • cryptomodels model run qwen3-1.7b")
        print("  • cryptomodels model run --hash QmYourHashHere")
        sys.exit(1)
    elif args.hash:
        model_hash = args.hash
        print_info(f"Using custom model hash: {model_hash[:16]}...")
    elif args.model_name:
        if args.model_name in PRESERVED_MODELS:
            model_hash = PRESERVED_MODELS[args.model_name]
            print_success(f"Found model '{args.model_name}' → {model_hash[:16]}...")
        else:
            print_error(f"Model '{args.model_name}' not found in preserved models")
            print_warning("Here are the available models:")
            show_available_models()
            print_info("For custom models, use: cryptomodels model run --hash <your_hash>")
            sys.exit(1)
    else:
        print_error("Either model name or --hash must be provided")
        print_info("Usage examples:")
        print("  • cryptomodels model run <model_name>")
        print("  • cryptomodels model run --hash <hash>")
        print_warning("Available models:")
        show_available_models()
        sys.exit(1)
    
    print_info(f"Starting model server...")
    print_info(f"Host: {args.host}, Port: {args.port}, Context: {args.context_length}")
    
    if not manager.start(model_hash, args.port, args.host, args.context_length):
        print_error("Failed to start model server")
        sys.exit(1)
    else:
        print_success(f"Model server started successfully on {args.host}:{args.port}")

def handle_stop(args):
    """Handle model stop with beautiful output"""
    if args.force:
        print_info("Force stopping model server...")
    else:
        print_info("Stopping model server...")
    if not manager.stop(force=args.force):
        print_error("Failed to stop model server or no server running")
        sys.exit(1)
    else:
        print_success("Model server stopped successfully")

def handle_status(args):
    """Handle status check with beautiful formatting"""
    running_model = manager.get_running_model()
    console = Console()
    if running_model:
        panel = Panel(
            f"[bold green]Status:[/bold green] Running\n"
            f"[bold blue]Details:[/bold blue] {running_model}",
            title="🤖 Model Server Status",
            border_style="green"
        )
        console.print(panel)
    else:
        panel = Panel(
            "[bold red]No model server is currently running[/bold red]",
            title="🤖 Model Server Status",
            border_style="red"
        )
        console.print(panel)

def handle_preserve(args):
    """Handle model preservation with beautiful output"""
    print_info(f"Starting preservation of: {args.folder_path}")
    print_info(f"Task: {args.task}, Threads: {args.threads}, Chunk size: {args.zip_chunk_size}MB")
    
    kwargs = {
        "task": args.task,
        "ram": args.ram,
        "config_name": args.config_name,
        "hf_repo": args.hf_repo,
        "hf_file": args.hf_file,
    }
    
    try:
        upload_folder_to_lighthouse(args.folder_path, args.zip_chunk_size, args.max_retries, args.threads, **kwargs)
        print_success("Model preserved successfully to IPFS!")
    except Exception as e:
        print_error(f"Preservation failed: {str(e)}")
        sys.exit(1)

def handle_check(args):
    """Handle model check with beautiful output"""
    print_info(f"Checking if model is downloaded for hash: {args.hash}")
    try:
        is_downloaded = check_downloaded_model(args.hash)
        if is_downloaded:
            print_success("True")
        else:
            print_info("False")
    except Exception as e:
        print_error(f"Check failed: {str(e)}")
        print_info("False")
        sys.exit(1)

def main():
    """Main CLI entry point with enhanced error handling"""
    # Show banner
    print_banner()
    
    known_args, unknown_args = parse_args()
    
    # Handle unknown arguments
    if unknown_args:
        for arg in unknown_args:
            print_error(f'Unknown command or argument: {arg}')
        print_info("Use --help for available commands and options")
        sys.exit(2)

    # Handle commands
    if known_args.command == "model":
        if known_args.model_command == "run":
            handle_run(known_args)
        elif known_args.model_command == "stop":
            handle_stop(known_args)
        elif known_args.model_command == "download":
            handle_download(known_args)
        elif known_args.model_command == "status":
            handle_status(known_args)
        elif known_args.model_command == "preserve":
            handle_preserve(known_args)
        elif known_args.model_command == "check":
            handle_check(known_args)
        else:
            print_error(f"Unknown model command: {known_args.model_command}")
            print_info("Available model commands: run, stop, download, status, preserve, check")
            sys.exit(2)
    else:
        print_error(f"Unknown command: {known_args.command}")
        print_info("Available commands: model")
        print_info("Use --help for more information")
        sys.exit(2)


if __name__ == "__main__":
    main()