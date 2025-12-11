import sys
import os
import asyncio
import argparse
import json
import yaml
import random
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

from eternal_zoo.version import __version__
from eternal_zoo.config import DEFAULT_CONFIG
from eternal_zoo.utils import find_gguf_files
from eternal_zoo.manager import EternalZooManager
from eternal_zoo.upload import upload_folder_to_lighthouse
from eternal_zoo.constants import DEFAULT_MODEL_DIR, POSTFIX_MODEL_PATH
from eternal_zoo.models import HASH_TO_MODEL, FEATURED_MODELS, MODEL_TO_HASH
from eternal_zoo.download import download_model_async, fetch_model_metadata_async

manager = EternalZooManager()

def get_all_downloaded_models() -> list:
    """
    Get all downloaded model hashes from the llms-storage directory.

    Returns:
        list: List of model hashes that have been downloaded
    """
    downloaded_models = []

    if not DEFAULT_MODEL_DIR.exists():
        return downloaded_models

    # Look for all .gguf files in the directory
    for model_file in DEFAULT_MODEL_DIR.glob(f"*.json"):
        model_id = model_file.stem
        if model_id:  # Make sure it's not empty
            downloaded_models.append(model_id)

    return downloaded_models

def print_banner():
    """Display a beautiful banner for the CLI"""
    console = Console()
    banner_text = """
███████╗████████╗███████╗██████╗ ███╗   ██╗ █████╗ ██╗          ███████╗ ██████╗  ██████╗
██╔════╝╚══██╔══╝██╔════╝██╔══██╗████╗  ██║██╔══██╗██║          ╚══███╔╝██╔═══██╗██╔═══██╗
█████╗     ██║   █████╗  ██████╔╝██╔██╗ ██║███████║██║            ███╔╝ ██║   ██║██║   ██║
██╔══╝     ██║   ██╔══╝  ██╔══██╗██║╚██╗██║██╔══██║██║           ███╔╝  ██║   ██║██║   ██║
███████╗   ██║   ███████╗██║  ██║██║ ╚████║██║  ██║███████╗     ███████╗╚██████╔╝╚██████╔╝
╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝     ╚══════╝ ╚═════╝  ╚═════╝
"""

    panel = Panel(
        Text(banner_text, style="bold cyan", justify="center"),
        title=f"[bold green]Eternal Zoo CLI v{__version__}[/bold green]",
        subtitle="[italic]Peer-to-Peer AI Model Management[/italic]",
        border_style="bright_blue",
        padding=(0, 0)
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
    """Display available models"""
    for model_name in FEATURED_MODELS:
        print(f"  {model_name}")

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
        description="🚀 Eternal Zoo - Peer-to-Peer AI Model Management Tool",
        formatter_class=CustomHelpFormatter,
        epilog="💡 For more information, visit: https://github.com/eternalai-org/eternal-zoo"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"eternal-zoo v{__version__} 🎉"
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
        help="🚀 Launch AI model server with multi-model support",
        description="Start serving models locally with multi-model and on-demand loading support"
    )
    run_command.add_argument(
        "--config",
        type=str,
        help="🔍 Config file path",
        metavar="CONFIG"
    )
    run_command.add_argument(
        "model_name",
        nargs='?',
        help="🏷️  Model name(s) - single: qwen3-1.7b or multi: qwen3-14b,qwen3-4b (first is main, others on-demand)"
    )
    run_command.add_argument(
        "--hash",
        type=str,
        help="🔗 Comma-separated Filecoin hashes (alternative to model names)",
        metavar="HASH1,HASH2,..."
    )
    run_command.add_argument(
        "--hf-repo",
        type=str,
        help="🤗 Hugging Face model repository",
        metavar="REPO"
    )
    run_command.add_argument(
        "--hf-file",
        type=str,
        help="🤗 Hugging Face model file",
        metavar="FILE"
    )
    run_command.add_argument(
        "--mmproj",
        type=str,
        help="🔍 Multimodal Projector File",
        metavar="MMProj"
    )
    run_command.add_argument(
        "--pattern",
        type=str,
        help="🔍 Pattern to download from Hugging Face",
        metavar="PATTERN"
    )
    run_command.add_argument(
        "--port",
        type=int,
        default=DEFAULT_CONFIG.network.DEFAULT_PORT,
        help=f"🌐 Port number for the server (default: {DEFAULT_CONFIG.network.DEFAULT_PORT})",
        metavar="PORT"
    )
    run_command.add_argument(
        "--host",
        type=str,
        default=DEFAULT_CONFIG.network.DEFAULT_HOST,
        help=f"🏠 Host address for the server (default: {DEFAULT_CONFIG.network.DEFAULT_HOST})",
        metavar="HOST"
    )
    run_command.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONFIG.model.DEFAULT_CONTEXT_LENGTH,
        help=f"📏 Context length for the model (default: {DEFAULT_CONFIG.model.DEFAULT_CONTEXT_LENGTH})",
        metavar="LENGTH"
    )
    run_command.add_argument(
        "--task",
        type=str,
        default="chat",
        choices=["chat", "embed", "image-generation", "image-edit"],
        help="🎯 Model task type (default: chat)",
        metavar="TYPE"
    )
    run_command.add_argument(
        "--backend",
        type=str,
        default="gguf",
        choices=["gguf", "mlx-flux", "mlx-lm"],
        help="🔍 Tensor backend for the model (default: gguf)",
        metavar="BACKEND"
    )
    run_command.add_argument(
        "--foreground",
        action="store_true",
        help="🖥️  Run in foreground mode (keep CLI running to monitor subprocesses)",
    )

    # Model serve command
    serve_command = model_subparsers.add_parser(
        "serve",
        help="🎯 Serve all downloaded models with optional main model selection",
        description="Run all models in llms-storage with a main model (randomly selected if not specified)"
    )
    serve_command.add_argument(
        "--main-hash",
        type=str,
        help="🔗 Hash of the main model to serve (if not specified, uses random model)",
        metavar="HASH"
    )
    serve_command.add_argument(
        "--main-model",
        type=str,
        help="🏷️  Model name(s) - single: qwen3-1.7b or multi: qwen3-14b,qwen3-4b (first is main, others on-demand)",
        metavar="MODEL"
    )
    serve_command.add_argument(
        "--hf-repo",
        type=str,
        help="🤗 Hugging Face model repository",
        metavar="REPO"
    )
    serve_command.add_argument(
        "--hf-file",
        type=str,
        help="🤗 Hugging Face model file",
        metavar="FILE"
    )
    serve_command.add_argument(
        "--mmproj",
        type=str,
        help="🔍 Multimodal Projector File",
        metavar="MMProj"
    )
    serve_command.add_argument(
        "--pattern",
        type=str,
        help="🔍 Pattern to download from Hugging Face",
        metavar="PATTERN"
    )
    serve_command.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONFIG.model.DEFAULT_CONTEXT_LENGTH,
        help=f"📏 Context length for the model (default: {DEFAULT_CONFIG.model.DEFAULT_CONTEXT_LENGTH})",
        metavar="LENGTH"
    )
    serve_command.add_argument(
        "--port",
        type=int,
        default=DEFAULT_CONFIG.network.DEFAULT_PORT,
        help=f"🌐 Port number for the server (default: {DEFAULT_CONFIG.network.DEFAULT_PORT})",
        metavar="PORT"
    )
    serve_command.add_argument(
        "--host",
        type=str,
        default=DEFAULT_CONFIG.network.DEFAULT_HOST,
        help=f"🏠 Host address for the server (default: {DEFAULT_CONFIG.network.DEFAULT_HOST})",
        metavar="HOST"
    )
    serve_command.add_argument(
        "--backend",
        type=str,
        default="gguf",
        choices=["gguf", "mlx-flux", "mlx-lm"],
        help="🔍 Tensor backend for the model (default: gguf)",
        metavar="BACKEND"
    )
    serve_command.add_argument(
        "--task",
        type=str,
        default="chat",
        choices=["chat", "embed", "image-generation", "image-edit"],
        help="🎯 Model task type (default: chat)",
        metavar="TYPE"
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
    stop_command.add_argument(
        "--port",
        type=int,
        default=DEFAULT_CONFIG.network.DEFAULT_PORT,
        help=f"🌐 Port number for the server (default: {DEFAULT_CONFIG.network.DEFAULT_PORT})",
        metavar="PORT"
    )
    # Model download command
    download_command = model_subparsers.add_parser(
        "download",
        help="⬇️  Download model from IPFS",
        description="Download and extract model files from the decentralized network"
    )
    download_command.add_argument(
        "model_name",
        nargs='?',
        help="🏷️  Model name(s) - single: qwen3-1.7b or multi: qwen3-14b,qwen3-4b (first is main, others on-demand)"
    )
    download_command.add_argument(
        "--hash",
        type=str,
        help="🔗 Comma-separated Filecoin hashes (alternative to model names)",
        metavar="HASH"
    )
    download_command.add_argument(
        "--hf-repo",
        type=str,
        help="🤗 Hugging Face model repository",
        metavar="REPO"
    )
    download_command.add_argument(
        "--hf-file",
        type=str,
        help="🤗 Hugging Face model file",
        metavar="FILE"
    )
    download_command.add_argument(
        "--mmproj",
        type=str,
        help="🔍 Multimodal Projector File",
        metavar="MMProj"
    )
    download_command.add_argument(
        "--pattern",
        type=str,
        help="🔍 Pattern to download from Hugging Face",
        metavar="PATTERN"
    )
    download_command.add_argument(
        "--task",
        type=str,
        default="chat",
        choices=["chat", "embed", "image-generation", "image-edit"],
        help="🎯 Model task type (default: chat)",
        metavar="TYPE"
    )
    download_command.add_argument(
        "--backend",
        type=str,
        default="gguf",
        choices=["gguf", "mlx-flux", "mlx-lm"],
        help="🔍 Tensor backend for the model (default: gguf)",
        metavar="BACKEND"
    )

    # Model check command
    check_command = model_subparsers.add_parser(
        "check",
        help="🔍 Check if model is downloaded",
        description="Check if a model with the specified hash has been downloaded"
    )
    check_command.add_argument(
        "model_name",
        nargs='?',
        help="🏷️  Model name(s) - single: qwen3-1.7b or multi: qwen3-14b,qwen3-4b (first is main, others on-demand)",
        metavar="MODEL"
    )
    check_command.add_argument(
        "--hash",
        help="🔗 IPFS hash of the model to check",
        metavar="HASH"
    )
    check_command.add_argument(
        "--hf-repo",
        help="🤗 Hugging Face model repository",
        metavar="REPO"
    )
    check_command.add_argument(
        "--hf-file",
    )
    check_command.add_argument(
        "--mmproj",
        help="🔍 Multimodal Projector File",
        metavar="MMProj"
    )
    check_command.add_argument(
        "--pattern",
        help="🔍 Pattern to download from Hugging Face",
        metavar="PATTERN"
    )
    check_command.add_argument(
        "--task",
        type=str,
        default="chat",
        choices=["chat", "embed", "image-generation", "image-edit"],
        help="🎯 Model task type (default: chat)",
        metavar="TYPE"
    )
    check_command.add_argument(
        "--backend",
        type=str,
        default="gguf",
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
        choices=["flux-dev", "flux-schnell"],
        help="🔍 Model config name (default: None), need for image-generation and image-edit models",
        metavar="CONFIG"
    )
    preserve_command.add_argument(
        "--gguf-folder",
        action="store_true",
        help="🔍 Indicate if this is a gguf folder include multiple files",
    )
    preserve_command.add_argument(
        "--lora",
        action="store_true",
        help="🔍 Indicate if this is a lora model (default: False)",
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

def handle_download(args) -> bool:
    """Handle model download with beautiful output"""
    if args.hash:
        # Download by hash
        if args.hash not in HASH_TO_MODEL:
            print_error(f"Hash {args.hash} not found in HASH_TO_MODEL")
            sys.exit(1)
        model_name = HASH_TO_MODEL[args.hash]
        hf_data = FEATURED_MODELS[model_name]
        success, local_path = asyncio.run(download_model_async(hf_data, args.hash))
        # Prepare and persist model metadata JSON (merge with fetched metadata if exists)
        if success:
            try:
                # Try to fetch gateway metadata to enrich saved file
                meta_success, fetched_meta = asyncio.run(fetch_model_metadata_async(args.hash))
            except Exception:
                meta_success, fetched_meta = False, None

            projector_path = f"{local_path}-projector"
            model_metadata_path = os.path.join(DEFAULT_MODEL_DIR, f"{args.hash}.json")
            existing_meta = {}
            if os.path.exists(model_metadata_path):
                try:
                    with open(model_metadata_path, "r") as f:
                        existing_meta = json.load(f)
                except Exception:
                    existing_meta = {}

            updates = {
                "task": (existing_meta.get("task")
                          or (fetched_meta or {}).get("task")
                          or hf_data.get("task")
                          or getattr(args, "task", None)
                          or "chat"),
                "model_id": args.hash,
                "model_name": (existing_meta.get("model_name")
                                or (fetched_meta or {}).get("folder_name")
                                or HASH_TO_MODEL.get(args.hash, args.hash)),
                "lora": (existing_meta.get("lora")
                          if existing_meta.get("lora") is not None else (fetched_meta or {}).get("lora", hf_data.get("lora", False))),
                "architecture": existing_meta.get("architecture") or hf_data.get("architecture", None),
                "multimodal": bool(os.path.exists(projector_path)),
                "hf_data": hf_data
            }
            merged = {**(fetched_meta or {}), **existing_meta, **updates}
            with open(model_metadata_path, "w") as f:
                json.dump(merged, f)
    elif args.model_name:
        if args.model_name in MODEL_TO_HASH:
            args.hash = MODEL_TO_HASH[args.model_name]
        if args.model_name not in FEATURED_MODELS:
            print_error(f"Model name {args.model_name} not found in FEATURED_MODELS")
            sys.exit(1)
        hf_data = FEATURED_MODELS[args.model_name]
        success, local_path = asyncio.run(download_model_async(hf_data, args.hash))
        # Save metadata for named featured models
        if success:
            projector_path = f"{local_path}-projector"
            model_id = args.hash if getattr(args, 'hash', None) else args.model_name
            model_metadata_path = os.path.join(DEFAULT_MODEL_DIR, f"{model_id}.json")
            existing_meta = {}
            if os.path.exists(model_metadata_path):
                try:
                    with open(model_metadata_path, "r") as f:
                        existing_meta = json.load(f)
                except Exception:
                    existing_meta = {}

            # Enrich with gateway metadata if we have a hash
            fetched_meta = None
            if getattr(args, 'hash', None):
                try:
                    meta_success, fetched_meta = asyncio.run(fetch_model_metadata_async(args.hash))
                    if not meta_success:
                        fetched_meta = None
                except Exception:
                    fetched_meta = None

            updates = {
                "task": (existing_meta.get("task")
                          or (fetched_meta or {}).get("task")
                          or hf_data.get("task")
                          or getattr(args, "task", None)
                          or "chat"),
                "model_id": model_id,
                "model_name": (existing_meta.get("model_name")
                                or (fetched_meta or {}).get("folder_name")
                                or args.model_name),
                "lora": (existing_meta.get("lora")
                          if existing_meta.get("lora") is not None else (fetched_meta or {}).get("lora", hf_data.get("lora", False))),
                "architecture": existing_meta.get("architecture") or hf_data.get("architecture", None),
                "multimodal": bool(os.path.exists(projector_path)),
                "hf_data": hf_data
            }
            merged = {**(fetched_meta or {}), **existing_meta, **updates}
            with open(model_metadata_path, "w") as f:
                json.dump(merged, f)
    else:
        # Download from Hugging Face
        hf_data = {
            "repo": args.hf_repo,
            "model": args.hf_file,
            "projector": args.mmproj,
            "pattern": args.pattern,
            "backend": args.backend
        }
        success, local_path = asyncio.run(download_model_async(hf_data))

        # Save metadata for HF downloads
        if success:
            projector_path = f"{local_path}-projector"
            final_local_path = local_path
            # If pattern is provided, try to resolve to a specific GGUF file like in handle_run
            if getattr(args, 'pattern', None):
                base_dir = os.path.dirname(local_path) if os.path.isfile(local_path) else local_path
                pattern_dir = os.path.join(base_dir, args.pattern)
                gguf_path = None
                if os.path.exists(pattern_dir) and os.path.isdir(pattern_dir):
                    gguf_files = find_gguf_files(pattern_dir)
                    if gguf_files:
                        gguf_path = os.path.join(pattern_dir, gguf_files[0])
                if not gguf_path:
                    gguf_files = find_gguf_files(base_dir)
                    if gguf_files:
                        gguf_path = os.path.join(base_dir, gguf_files[0])
                if gguf_path:
                    final_local_path = gguf_path

            model_id = os.path.basename(final_local_path)
            model_metadata = {
                "task": getattr(args, "task", "chat"),
                "model_id": model_id,
                "model_name": model_id,
                "multimodal": bool(projector_path and os.path.exists(projector_path)),
                "lora": False,
                "architecture": None,
                "hf_data": hf_data
            }
            model_metadata_path = os.path.join(DEFAULT_MODEL_DIR, f"{model_id}.json")
            with open(model_metadata_path, "w") as f:
                json.dump(model_metadata, f)
    
    # Handle download result
    if success:
        print_success(f"Model downloaded successfully to {local_path}")
        return True
    else:
        print_error("Download failed")
        sys.exit(1)

def _determine_model_id_from_args(args) -> str:
    """Determine the model_id used for metadata after a download.

    This mirrors the logic used when saving metadata in `handle_download`.
    Assumes the download for the given args has already completed.
    """
    # Hash or featured model name cases
    if getattr(args, 'hash', None):
        return args.hash
    if getattr(args, 'model_name', None):
        return MODEL_TO_HASH.get(args.model_name, args.model_name)

    # HF repo case
    if getattr(args, 'hf_repo', None):
        # Reconstruct local path similarly to the existing logic
        if getattr(args, 'hf_file', None):
            local_path = os.path.join(str(DEFAULT_MODEL_DIR), args.hf_file)
        else:
            base_dir = os.path.join(str(DEFAULT_MODEL_DIR), args.hf_repo.replace("/", "_"))
            if getattr(args, 'pattern', None):
                base_dir = f"{base_dir}_{args.pattern}"
            local_path = base_dir

        # If a directory, attempt to pick a GGUF file when pattern is specified or none found at top level
        if os.path.isdir(local_path):
            search_dir = local_path
            if getattr(args, 'pattern', None):
                pattern_dir = os.path.join(local_path, args.pattern)
                if os.path.exists(pattern_dir) and os.path.isdir(pattern_dir):
                    search_dir = pattern_dir
            gguf_files = find_gguf_files(search_dir)
            if gguf_files:
                local_path = os.path.join(search_dir, gguf_files[0])

        return os.path.basename(local_path)

    # Fallback
    raise ValueError("Insufficient arguments to determine model_id")

def handle_run(args):
    """Handle model loading and configuration based on provided arguments."""
    # Handle config file case with multi-model support
    if args.config:
        # read config file from yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        port = config.get("port", 8080)
        host = config.get("host", "0.0.0.0")
        models = config.get("models", {})
        
        # Build configurations for all models by reusing handle_download + load_model_metadata
        configs = []
        main_model_key = None

        # First pass: find the main model (on_demand=False)
        for model_key, model_config in models.items():
            on_demand = model_config.get("on_demand", False)
            if not on_demand:
                main_model_key = model_key
                break
        if not main_model_key:
            print_warning("No main model found in config file (on_demand=False), using first model")
            main_model_key = list(models.keys())[0]

        # Second pass: ensure downloads and load configs from metadata
        for model_key, model_config in models.items():
            is_main = (model_key == main_model_key)

            # Compose an args-like object for handle_download
            # Priority: featured name in `model`, else `hf_repo`
            if "model" in model_config and model_config["model"] in FEATURED_MODELS:
                featured_model_name = model_config["model"]
                temp_args = argparse.Namespace(
                    hash=MODEL_TO_HASH.get(featured_model_name),
                    model_name=featured_model_name,
                    hf_repo=None,
                    hf_file=None,
                    mmproj=None,
                    pattern=None,
                    backend=model_config.get("backend"),
                    task=model_config.get("task", FEATURED_MODELS[featured_model_name].get("task", "chat"))
                )
            elif "hf_repo" in model_config:
                temp_args = argparse.Namespace(
                    hash=None,
                    model_name=None,
                    hf_repo=model_config.get("hf_repo"),
                    hf_file=model_config.get("model"),
                    mmproj=model_config.get("mmproj"),
                    pattern=model_config.get("pattern"),
                    backend=model_config.get("backend"),
                    task=model_config.get("task", "chat")
                )
            else:
                print_warning(f"Model entry '{model_key}' missing required fields; skipping")
                continue

            # Ensure download + metadata
            handle_download(temp_args)

            # Determine model_id and load config from metadata
            try:
                model_id = _determine_model_id_from_args(temp_args)
            except Exception as e:
                print_warning(f"Failed to determine model_id for '{model_key}': {e}")
                continue

            success, cfg = load_model_metadata(model_id, is_main=is_main)
            if not success or not cfg:
                print_warning(f"Failed to load model '{model_key}' from metadata")
                continue

            # Apply YAML config overrides
            context_override = False
            if "context_length" in model_config:
                cfg["context_length"] = model_config["context_length"]
                context_override = True

            # Calculate memory requirements for all models after overrides are applied
            try:
                from eternal_zoo.manager import EternalZooManager
                manager = EternalZooManager()
                memory_requirements = manager._calculate_model_memory_requirements(cfg)
                cfg["memory_requirements"] = memory_requirements
                override_note = " (after context_length override)" if context_override else ""
                print_info(f"Calculated memory for {model_id}{override_note}: RAM={memory_requirements['ram_gb']:.2f}GB, GPU={memory_requirements['gpu_ram_gb']:.2f}GB")
            except Exception as e:
                print_warning(f"Failed to calculate memory requirements for {model_id}: {e}")
                # Continue without memory requirements - will fall back to on-demand calculation

            configs.append(cfg)

        if not configs:
            print_error("No valid models found in config file")
            sys.exit(1)

        # Start the multi-model server
        success = manager.start(configs, port, host)
        if not success:
            print_error("Failed to start multi-model server")
            sys.exit(1)

        print_success(f"Multi-model server started with {len(configs)} models")

        if getattr(args, 'foreground', False):
            print_info("Running in foreground mode - press Ctrl+C to stop")
            try:
                # Keep the process running to prevent subprocess termination in some environments
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                print_info("Stopping server...")
                manager.stop()
        return success
    
    # Handle Hugging Face repository case separately
    if args.hf_repo:
        # Ensure model and metadata are downloaded via the download flow
        handle_download(args)

        # Determine model_id and load config from metadata
        model_id = _determine_model_id_from_args(args)
        success, config = load_model_metadata(model_id, is_main=True)
        if not success:
            print_error(f"Failed to load model {model_id}")
            sys.exit(1)

        # Respect provided context length override
        if getattr(args, 'context_length', None):
            config["context_length"] = args.context_length

        success = manager.start([config], args.port, args.host)

        if not success:
            print_error(f"Failed to start model {model_id}")
            sys.exit(1)

        if getattr(args, 'foreground', False):
            print_info("Running in foreground mode - press Ctrl+C to stop")
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                print_info("Stopping server...")
                manager.stop()
        return success

    # Handle hash or model_name cases
    if args.hash:
        if args.hash not in HASH_TO_MODEL:
            print_error(f"Hash {args.hash} not found in HASH_TO_MODEL")
            sys.exit(1)
        model_name = HASH_TO_MODEL[args.hash]
    elif args.model_name:
        if args.model_name not in FEATURED_MODELS:
            print_error(f"Model name {args.model_name} not found in FEATURED_MODELS")
            sys.exit(1)
        model_name = args.model_name
        if model_name in MODEL_TO_HASH:
            args.hash = MODEL_TO_HASH[model_name]
    else:
        print_error("Either hash, model_name, or hf_repo must be provided")
        sys.exit(1)

    # Resolve model_id and ensure download+metadata exist, then start
    # Determine model name and model_id
    if args.hash:
        model_id = args.hash
    elif args.model_name:
        model_id = MODEL_TO_HASH.get(args.model_name, args.model_name)
    else:
        print_error("Either hash, model_name, or hf_repo must be provided")
        sys.exit(1)

    # If metadata not present, perform download (also writes metadata)
    metadata_path = os.path.join(DEFAULT_MODEL_DIR, f"{model_id}.json")
    if not os.path.exists(metadata_path):
        handle_download(args)

    success, config = load_model_metadata(model_id, is_main=True)
    if not success:
        print_error(f"Failed to load model {model_id}")
        sys.exit(1)

    # Respect provided context length
    if getattr(args, 'context_length', None):
        config["context_length"] = args.context_length

    success = manager.start([config], args.port, args.host)
    if not success:
        print_error(f"Failed to start model {model_id}")
        sys.exit(1)

    if getattr(args, 'foreground', False):
        print_info("Running in foreground mode - press Ctrl+C to stop")
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print_info("Stopping server...")
            manager.stop()
    return success


def load_model_metadata(model_id, is_main=False) -> tuple[bool, dict | None]:
    """Load model metadata from JSON file and prepare configuration.

    Args:
        model_id (str): The identifier of the model.
        is_main (bool): Whether this is the main model (default: False).

    Returns:
        tuple[bool, dict | None]: Success status and configuration dictionary.
    """
    # Use pathlib for consistent path operations
    model_dir = DEFAULT_MODEL_DIR
    json_path = model_dir / f"{model_id}.json"
    
    # Load metadata with optimized error handling
    try:
        with json_path.open("r") as f:
            metadata = json.load(f)
    except (FileNotFoundError, OSError):
        print_warning(f"Metadata file not found for model {model_id}")
        return False, None
    except json.JSONDecodeError as e:
        print_warning(f"Invalid JSON in metadata file for model {model_id}: {e}")
        return False, None

    # Optimize path checking - check both paths at once
    local_path = model_dir / model_id
    hf_data = metadata.get("hf_data", None)
    if hf_data:
        repo = hf_data["repo"]
        local_path = model_dir / f"{repo.replace('/', '_')}"
        model = hf_data.get("model", None)
        pattern = hf_data.get("pattern", None)
        if model:
            local_path = model_dir / model
        elif pattern:
            pattern_dir = model_dir / f"{repo.replace('/', '_')}_{pattern}" / pattern
            if pattern_dir.exists() and pattern_dir.is_dir():
                gguf_files = find_gguf_files(pattern_dir)
                if gguf_files:
                    local_path = pattern_dir / gguf_files[0]
            else:
                local_path = model_dir / f"{repo.replace('/', '_')}_{pattern}"
        
    if not local_path.exists():
        local_path = model_dir / f"{model_id}{POSTFIX_MODEL_PATH}"
        if not local_path.exists():
            print_warning(f"Model file not found for model {model_id}")
            return False, None

    # Extract metadata values once
    is_multimodal = metadata.get("multimodal", False)
    is_lora = metadata.get("lora", False)
    model_name = metadata.get("model_name") or metadata.get("folder_name", model_id)
    context_length = metadata.get("context_length", DEFAULT_CONFIG.model.DEFAULT_CONTEXT_LENGTH)
    
    # Handle projector path for multimodal models
    projector_path = str(model_dir / f"{model_id}-projector") if is_multimodal else None
    
    lora_config = None
    if is_lora:
        lora_config = _load_lora_config(local_path, model_dir)
        if lora_config is None:
            return False, None
        
        # Handle base model for LoRA
        base_model_result = _handle_lora_base_model(lora_config.pop("_metadata", {}))
        if base_model_result is None:
            return False, None
        local_path, model_name = base_model_result

    # Build configuration dictionary
    config = {
        "model_id": model_id,
        "model_name": model_name,
        "task": metadata.get("task", "chat"),
        "model": str(local_path),
        "multimodal": is_multimodal,
        "projector": projector_path,
        "on_demand": not is_main,
        "is_lora": is_lora,
        "architecture": metadata.get("architecture", "flux-dev"),
        "lora_config": lora_config,
        "context_length": context_length,
        "backend": hf_data.get("backend", "gguf") if hf_data else "gguf"
    }

    return True, config


def _load_lora_config(local_path, model_dir):
    """Helper function to load LoRA configuration."""
    lora_metadata_path = local_path / "metadata.json"
    if not lora_metadata_path.exists():
        print_warning("LoRA model found but metadata.json is missing")
        return None
    
    try:
        with lora_metadata_path.open("r") as f:
            lora_metadata = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print_warning(f"Failed to read LoRA metadata: {e}")
        return None
    
    # Build LoRA configuration efficiently
    lora_paths = lora_metadata.get("lora_paths", [])
    lora_scales = lora_metadata.get("lora_scales", [])
    
    if len(lora_paths) != len(lora_scales):
        print_warning("Mismatch between LoRA paths and scales")
        return None
    
    lora_config = {
        os.path.basename(lora_path): {
            "path": str(local_path / lora_path),
            "scale": lora_scales[i]
        }
        for i, lora_path in enumerate(lora_paths)
    }
    
    # Store metadata for base model handling
    lora_config["_metadata"] = lora_metadata
    return lora_config


def _handle_lora_base_model(lora_metadata):
    """Helper function to handle LoRA base model resolution."""
    base_model_hash = lora_metadata.get("base_model")
    if not base_model_hash:
        print_warning("LoRA metadata missing base_model hash")
        return None
        
    if base_model_hash not in HASH_TO_MODEL:
        print_warning(f"Base model hash {base_model_hash} not found")
        return None
    
    base_model_name = HASH_TO_MODEL[base_model_hash]
    if base_model_name not in FEATURED_MODELS:
        print_warning(f"Base model {base_model_name} not in featured models")
        return None
        
    base_model_hf_data = FEATURED_MODELS[base_model_name]
    
    try:
        success, base_model_local_path = asyncio.run(
            download_model_async(base_model_hf_data, base_model_hash)
        )
        if not success:
            print_warning(f"Failed to download base model {base_model_name}")
            return None
        return base_model_local_path, base_model_name
    except Exception as e:
        print_warning(f"Error downloading base model: {e}")
        return None

def handle_serve(args):
    """Handle model serve command - run all downloaded models with specified main model.

    Args:
        args: Command-line arguments containing host, port, main_hash, and main_model.
    """
    print_info("Discovering downloaded models in models directory...")
            
    # Get all downloaded models
    downloaded_models = get_all_downloaded_models()

    if not downloaded_models:
        print_error("No downloaded models found in models directory")
        sys.exit(1)

    print_success(f"Found {len(downloaded_models)} downloaded model(s)")

    # Determine the main model ID
    if args.main_hash:
        main_model_id = args.main_hash
    elif args.main_model:
        main_model_id = args.main_model
    elif args.hf_repo:
        main_model_id = args.hf_repo.replace("/", "_")
        if args.hf_file:
            main_model_id = args.hf_file
        elif args.pattern:
            pattern_dir = os.path.join(str(DEFAULT_MODEL_DIR), main_model_id + "_" + args.pattern,  args.pattern)
            if os.path.exists(pattern_dir) and os.path.isdir(pattern_dir):
                gguf_files = find_gguf_files(pattern_dir)
                if gguf_files:
                    main_model_id = os.path.basename(gguf_files[0])
            else:
                main_model_id = main_model_id + "_" + args.pattern
    else:
        main_model_id = downloaded_models[0]  # Default to first model instead of random

    # Validate that the main model exists in downloaded models
    if main_model_id not in downloaded_models:
        print_error(f"Specified main model {main_model_id} not found in downloaded models")
        sys.exit(1)

    # Prepare configurations
    success, main_config = load_model_metadata(main_model_id, is_main=True)
    if not success:
        print_error(f"Failed to load main model {main_model_id}")
        sys.exit(1)
    configs = [main_config]

    other_models = [model_id for model_id in downloaded_models if model_id != main_model_id]
    for model_id in other_models:
        success, config = load_model_metadata(model_id, is_main=False)
        if not success:
            print_warning(f"Failed to load model {model_id}")
            continue
        configs.append(config)
    # Start the model server
    success = manager.start(configs, args.port, args.host)

    if not success:
        print_error("Failed to start model server")
        sys.exit(1)
    else:
        print_success(f"Model server started successfully on {args.host}:{args.port}")
        if other_models:
            print_info(f"Serving {len(configs)} model(s) with main model: {main_model_id} and other models: {', '.join(other_models)}")
        else:
            print_info(f"Serving only the main model: {main_model_id}")
   

def handle_stop(args):
    """Handle model stop with beautiful output"""
    if not manager.stop():
        print_error("Failed to stop model server or no server running")
    else:
        print_success("Model server stopped successfully")

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
        "lora": args.lora,
        "gguf_folder": args.gguf_folder,
    }

    try:
        upload_folder_to_lighthouse(args.folder_path, args.zip_chunk_size, args.max_retries, args.threads, **kwargs)
        print_success("Model preserved successfully to IPFS!")
    except Exception as e:
        print_error(f"Preservation failed: {str(e)}")
        sys.exit(1)

def handle_check(args):
    """Handle model check with beautiful output"""
    if getattr(args, 'hash', None):
        model_hash = args.hash
        local_path = DEFAULT_MODEL_DIR / (model_hash + POSTFIX_MODEL_PATH)
        print_info(f"Local path: {local_path}")
        if local_path.exists():
            print_success("True")
        else:
            print_info("False")
        return

    hf_data = {
        "repo": args.hf_repo,
        "model": args.hf_file,
        "pattern": args.pattern,
        "mmproj": args.mmproj,
    }
    
    if getattr(args, 'model_name', None):
        model_name = args.model_name
        if model_name not in FEATURED_MODELS:
            print_error(f"Model name {model_name} not found in FEATURED_MODELS")
            sys.exit(1)
        if model_name in MODEL_TO_HASH:
            model_hash = MODEL_TO_HASH[model_name]
            local_path = DEFAULT_MODEL_DIR / (model_hash + POSTFIX_MODEL_PATH)
            print_info(f"Local path: {local_path}")
            if local_path.exists():
                print_success("True")
            else:
                print_info("False")
            return
        else:
            hf_data = FEATURED_MODELS[model_name]

    local_path = DEFAULT_MODEL_DIR / hf_data.get("repo", "").replace("/", "_")
        
    if hf_data.get("model", None):
        local_path = DEFAULT_MODEL_DIR / hf_data["model"]
    
    if hf_data.get("pattern", None):
        pattern_dir =  DEFAULT_MODEL_DIR / (hf_data["repo"].replace("/", "_") + "_" + hf_data["pattern"]) / hf_data["pattern"]
        if os.path.exists(pattern_dir) and os.path.isdir(pattern_dir):
            gguf_files = find_gguf_files(pattern_dir)
            if gguf_files:
                local_path = pattern_dir / gguf_files[0]
        else:
            local_path = DEFAULT_MODEL_DIR / (hf_data["repo"].replace("/", "_") + "_" + hf_data["pattern"])
        
    print_info(f"Local path: {local_path}")
    if local_path.exists():
        print_success("True")
    else:
        print_info("False")
    return

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
        elif known_args.model_command == "serve":
            handle_serve(known_args)
        elif known_args.model_command == "stop":
            handle_stop(known_args)
        elif known_args.model_command == "download":
            handle_download(known_args)
        elif known_args.model_command == "preserve":
            handle_preserve(known_args)
        elif known_args.model_command == "check":
            handle_check(known_args)
        else:
            print_error(f"Unknown model command: {known_args.model_command}")
            print_info("Available model commands: run, serve, stop, download, status, preserve, check")
            sys.exit(2)
    else:
        print_error(f"Unknown command: {known_args.command}")
        print_info("Available commands: model")
        print_info("Use --help for more information")
        sys.exit(2)


if __name__ == "__main__":
    main()