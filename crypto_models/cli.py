import sys
import asyncio
import argparse
from pathlib import Path
from loguru import logger
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    
from crypto_models import __version__
from crypto_models.core import CryptoModelsManager
from crypto_models.upload import upload_folder_to_lighthouse
from crypto_models.download import download_model_from_filecoin_async
from crypto_models.preseved_models import PRESERVED_MODELS

manager = CryptoModelsManager()

def print_banner():
    """Display a beautiful banner for the CLI"""
    if RICH_AVAILABLE:
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
            Text(banner_text, style="bold cyan"),
            title=f"[bold green]CryptoModels CLI v{__version__}[/bold green]",
            subtitle="[italic]Decentralized AI Model Management[/italic]",
            border_style="bright_blue",
            padding=(1, 2)
        )
        console.print(panel)
    else:
        print(f"""
╔═══════════════════════════════════════════════════════════╗
║                    CRYPTO MODELS CLI                      ║
║                     Version {__version__:<10}                    ║
║               Decentralized AI Model Management           ║
╚═══════════════════════════════════════════════════════════╝
        """)

def print_success(message):
    """Print success message with styling"""
    if RICH_AVAILABLE:
        rprint(f"[bold green]✅ {message}[/bold green]")
    else:
        print(f"✅ {message}")

def print_error(message):
    """Print error message with styling"""
    if RICH_AVAILABLE:
        rprint(f"[bold red]❌ {message}[/bold red]")
    else:
        print(f"❌ {message}")

def print_info(message):
    """Print info message with styling"""
    if RICH_AVAILABLE:
        rprint(f"[bold blue]ℹ️  {message}[/bold blue]")
    else:
        print(f"ℹ️  {message}")

def print_warning(message):
    """Print warning message with styling"""
    if RICH_AVAILABLE:
        rprint(f"[bold yellow]⚠️  {message}[/bold yellow]")
    else:
        print(f"⚠️  {message}")

def show_available_models():
    """Display available models in a beautiful table"""
    if RICH_AVAILABLE:
        console = Console()
        table = Table(title="🤖 Available Preserved Models", border_style="cyan")
        table.add_column("Model Name", style="bold magenta", justify="left")
        table.add_column("Hash", style="dim", justify="left")
        
        for model_name, model_hash in PRESERVED_MODELS.items():
            table.add_row(model_name, model_hash[:16] + "...")
            
        console.print(table)
    else:
        print("\n🤖 Available Preserved Models:")
        print("=" * 50)
        for model_name, model_hash in PRESERVED_MODELS.items():
            print(f"  • {model_name:<20} {model_hash[:16]}...")
        print("=" * 50)

class CustomHelpFormatter(argparse.HelpFormatter):
    """Custom help formatter for better styling"""
    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super()._format_action_invocation(action)
        default = super()._format_action_invocation(action)
        if RICH_AVAILABLE:
            return f"[bold cyan]{default}[/bold cyan]"
        return default

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
        default=8080,
        help="🌐 Port number for the server (default: 8080)",
        metavar="PORT"
    )
    run_command.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0",
        help="🏠 Host address for the server (default: 0.0.0.0)",
        metavar="HOST"
    )
    run_command.add_argument(
        "--context-length", 
        type=int, 
        default=32768,
        help="📏 Context length for the model (default: 32768)",
        metavar="LENGTH"
    )
    
    # Model stop command
    stop_command = model_subparsers.add_parser(
        "stop", 
        help="🛑 Stop the running model server",
        description="Gracefully shutdown the currently running model server"
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
        default=8192,
        help="📦 Download chunk size in bytes (default: 8192)",
        metavar="SIZE"
    )
    download_command.add_argument(
        "--output-dir", 
        type=Path, 
        default=None,
        help="📁 Output directory for model files",
        metavar="DIR"
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
        choices=["chat", "embed"],
        help="🎯 Model task type (default: chat)",
        metavar="TYPE"
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
    print_info("Stopping model server...")
    if not manager.stop():
        print_error("Failed to stop model server or no server running")
        sys.exit(1)
    else:
        print_success("Model server stopped successfully")

def handle_status(args):
    """Handle status check with beautiful formatting"""
    running_model = manager.get_running_model()
    if running_model:
        if RICH_AVAILABLE:
            console = Console()
            panel = Panel(
                f"[bold green]Status:[/bold green] Running\n"
                f"[bold blue]Details:[/bold blue] {running_model}",
                title="🤖 Model Server Status",
                border_style="green"
            )
            console.print(panel)
        else:
            print("🤖 Model Server Status")
            print("=" * 30)
            print(f"Status: ✅ Running")
            print(f"Details: {running_model}")
            print("=" * 30)
    else:
        if RICH_AVAILABLE:
            console = Console()
            panel = Panel(
                "[bold red]No model server is currently running[/bold red]",
                title="🤖 Model Server Status",
                border_style="red"
            )
            console.print(panel)
        else:
            print("🤖 Model Server Status")
            print("=" * 30)
            print("Status: ❌ Not Running")
            print("=" * 30)

def handle_preserve(args):
    """Handle model preservation with beautiful output"""
    print_info(f"Starting preservation of: {args.folder_path}")
    print_info(f"Task: {args.task}, Threads: {args.threads}, Chunk size: {args.zip_chunk_size}MB")
    
    kwargs = {
        "task": args.task,
        "ram": args.ram,
        "hf_repo": args.hf_repo,
        "hf_file": args.hf_file,
    }
    
    try:
        upload_folder_to_lighthouse(args.folder_path, args.zip_chunk_size, args.max_retries, args.threads, **kwargs)
        print_success("Model preserved successfully to IPFS!")
    except Exception as e:
        print_error(f"Preservation failed: {str(e)}")
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
        else:
            print_error(f"Unknown model command: {known_args.model_command}")
            print_info("Available model commands: run, stop, download, status, preserve")
            sys.exit(2)
    else:
        print_error(f"Unknown command: {known_args.command}")
        print_info("Available commands: model")
        print_info("Use --help for more information")
        sys.exit(2)


if __name__ == "__main__":
    main()