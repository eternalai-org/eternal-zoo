import sys
import asyncio
import argparse
from pathlib import Path
from loguru import logger
from crypto_models import __version__
from crypto_models.core import CryptoModelsManager
from crypto_models.upload import upload_folder_to_lighthouse
from crypto_models.download import download_model_from_filecoin_async
from crypto_models.preseved_models import PRESERVED_MODELS

manager = CryptoModelsManager()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tool for managing local large language models"
    )
    parser.add_argument(
        "--version", action="version", version=f"eai {__version__}"
    )
    subparsers = parser.add_subparsers(
        dest='command', help="Commands for managing local language models"  
    )
    
    # Model command group
    model_command = subparsers.add_parser(
        "model", help="Model management commands"
    )
    model_subparsers = model_command.add_subparsers(
        dest='model_command', help="Model operations"
    )
    
    # Model run command (previously start)
    run_command = model_subparsers.add_parser(
        "run", help="Run a local language model server"
    )
    
    # Add positional argument for model name
    run_command.add_argument(
        "model_name", nargs='?', 
        help="Model name (e.g., qwen3-1.7b) - will be mapped to hash automatically"
    )
    run_command.add_argument(
        "--hash", type=str,
        help="Filecoin hash of the model to run (alternative to model name)"
    )
    
    run_command.add_argument(
        "--port", type=int, default=8080,
        help="Port number for the local language model server"
    )
    run_command.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host address for the local language model server"
    )
    run_command.add_argument(
        "--context-length", type=int, default=32768,
        help="Context length for the local language model server"
    )
    
    # Model stop command
    stop_command = model_subparsers.add_parser(
        "stop", help="Stop a local language model server"
    )
    
    # Model download command
    download_command = model_subparsers.add_parser(
        "download", help="Download and extract model files from IPFS"
    )
    download_command.add_argument(
        "--hash", required=True,
        help="IPFS hash of the model metadata"
    )
    download_command.add_argument(
        "--chunk-size", type=int, default=8192,
        help="Chunk size for downloading files"
    )
    download_command.add_argument(
        "--output-dir", type=Path, default = None,
        help="Output directory for model files"
    )
    
    # Model status command
    status_command = model_subparsers.add_parser(
        "status", help="Check the running model"
    )
    
    # Model preserve command (previously upload)
    preserve_command = model_subparsers.add_parser(
        "preserve", help="Preserve model files to IPFS"
    )
    preserve_command.add_argument(
        "--task", type=str, default="chat", choices=["chat", "embed"],
        help = "Task type (chat or embed)"
    )
    preserve_command.add_argument(
        "--folder-path", type=str, required=True,
        help="Folder containing model files"
    ) 
    preserve_command.add_argument(
        "--zip-chunk-size", type=int, default=512,
        help="Chunk size for splitting compressed files"
    )
    preserve_command.add_argument(
        "--threads", type=int, default=16,
        help="Number of threads for compressing files"
    )
    preserve_command.add_argument(
        "--max-retries", type=int, default=5,
        help="Maximum number of retries for uploading files"
    )
    preserve_command.add_argument(
        "--hf-repo", type=str, default = None,
        help="Hugging Face model repository"
    )
    preserve_command.add_argument(
        "--hf-file", type=str, default = None,
        help="Hugging Face model file"
    )
    preserve_command.add_argument(
        "--ram", type=float, default=None,
        help="RAM in GB for the serving model at 4096 context length"
    )
    
    return parser.parse_known_args()

def handle_download(args):
    asyncio.run(download_model_from_filecoin_async(args.hash))

def handle_run(args):
    # Determine the hash to use
    if args.hash and args.model_name:
        logger.error("Please specify either a model name OR --hash, not both")
        sys.exit(1)
    elif args.hash:
        model_hash = args.hash
    elif args.model_name:
        if args.model_name in PRESERVED_MODELS:
            model_hash = PRESERVED_MODELS[args.model_name]
            logger.info(f"Mapping model name '{args.model_name}' to hash: {model_hash}")
        else:
            logger.error(f"Model '{args.model_name}' not found in preserved models.")
            logger.error("Available models:")
            for model_name in PRESERVED_MODELS.keys():
                logger.error(f"  - {model_name}")
            logger.error("Please use 'eai model run --hash <your_hash>' for custom models.")
            sys.exit(1)
    else:
        logger.error("Either model name or --hash must be provided")
        logger.error("Usage: eai model run <model_name> OR eai model run --hash <hash>")
        logger.error("Available models:")
        for model_name in PRESERVED_MODELS.keys():
            logger.error(f"  - {model_name}")
        sys.exit(1)
    
    if not manager.start(model_hash, args.port, args.host, args.context_length):
        sys.exit(1)

def handle_stop(args):
    if not manager.stop():
        sys.exit(1)

def handle_status(args):
    running_model = manager.get_running_model()
    if running_model:
        print(running_model)

def handle_preserve(args):
    kwargs = {
        "task": args.task,
        "ram": args.ram,
        "hf_repo": args.hf_repo,
        "hf_file": args.hf_file,
    }
    upload_folder_to_lighthouse(args.folder_path, args.zip_chunk_size, args.max_retries, args.threads, **kwargs)

def main():
    known_args, unknown_args = parse_args()
    for arg in unknown_args:
        logger.error(f'unknown command or argument: {arg}')
        sys.exit(2)

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
            logger.error(f"Unknown model command: {known_args.model_command}")
            sys.exit(2)
    else:
        logger.error(f"Unknown command: {known_args.command}")
        sys.exit(2)


if __name__ == "__main__":
    main()