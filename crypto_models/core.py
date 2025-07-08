import os
import json
import time
import signal
import msgpack
import psutil
import asyncio
import socket
import requests
import subprocess
import pkg_resources
import shutil
from pathlib import Path
from loguru import logger
from typing import Optional, Dict, Any
from crypto_models.config import config
from crypto_models.utils import wait_for_health
from crypto_models.download import download_model_from_filecoin_async

class CryptoAgentsServiceError(Exception):
    """Base exception for CryptoModels service errors."""
    pass

class ServiceStartError(CryptoAgentsServiceError):
    """Exception raised when service fails to start."""
    pass

class ModelNotFoundError(CryptoAgentsServiceError):
    """Exception raised when model file is not found."""
    pass

class CryptoModelsManager:
    """Manages a CryptoModels service with optimized performance."""
    
    def __init__(self):
        """Initialize the CryptoModelsManager with optimized defaults."""
        # Performance constants from config
        self.LOCK_TIMEOUT = config.core.LOCK_TIMEOUT
        self.PORT_CHECK_TIMEOUT = config.core.PORT_CHECK_TIMEOUT
        self.HEALTH_CHECK_TIMEOUT = config.core.HEALTH_CHECK_TIMEOUT
        self.PROCESS_TERM_TIMEOUT = config.core.PROCESS_TERM_TIMEOUT
        self.MAX_PORT_RETRIES = config.core.MAX_PORT_RETRIES
        
        # File paths from config
        self.msgpack_file = Path(config.file_paths.RUNNING_SERVICE_FILE)
        self.loaded_models: Dict[str, Any] = {}
        self.llama_server_path = config.file_paths.LLAMA_SERVER or os.getenv("LLAMA_SERVER")
        if not self.llama_server_path or not os.path.exists(self.llama_server_path):
            raise CryptoAgentsServiceError("llama-server executable not found in LLAMA_SERVER or PATH")
        self.start_lock_file = Path(config.file_paths.START_LOCK_FILE)
        self.logs_dir = Path(config.file_paths.LOGS_DIR)
        self.logs_dir.mkdir(exist_ok=True)
        
    def _get_free_port(self) -> int:
        """Get a free port number."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    
    def restart(self):
        """
        Restart the currently running CryptoModels service.

        Returns:
            bool: True if the service restarted successfully, False otherwise.
        """
        if not self.msgpack_file.exists():
            logger.warning("No running CryptoModels service to restart.")
            return False
        
        try:
            # Load service details from the msgpack file
            with open(self.msgpack_file, "rb") as f:
                service_info = msgpack.load(f)
            
            hash = service_info.get("hash")
            port = service_info.get("app_port")
            context_length = service_info.get("context_length")

            logger.info(f"Restarting CryptoModels service '{hash}' running on port {port}...")

            # Stop the current service with force for restart
            self.stop(force=True)

            # Start the service with the same parameters
            return self.start(hash, port, context_length=context_length)
        except Exception as e:
            logger.error(f"Error restarting CryptoModels service: {str(e)}", exc_info=True)
            return False

    def _get_family_template_and_practice(self, model_family: str):
        """Helper to get template and best practice paths based on folder name."""
        return (
            self._get_model_template_path(model_family),
            self._get_model_best_practice_path(model_family)
        )
    
    def _retry_request_json(self, url: str, retries: int = None, delay: int = None, timeout: int = None) -> Optional[dict]:
        """
        Utility to retry a GET request for JSON data with optimized parameters.
        Returns parsed JSON data or None on failure.
        """
        # Use config values as defaults
        retries = retries or config.core.REQUEST_RETRIES
        delay = delay or config.core.REQUEST_DELAY
        timeout = timeout or config.core.REQUEST_TIMEOUT
        
        backoff_delay = delay
        last_error = None
        
        for attempt in range(retries):
            try:
                # Use session for connection pooling if making multiple requests
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()  # Raise exception for HTTP errors
                
                # Parse JSON once and return
                return response.json()
                
            except requests.exceptions.Timeout as e:
                last_error = f"Timeout after {timeout}s"
                logger.warning(f"Attempt {attempt+1}/{retries} timed out for {url}")
            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {str(e)[:100]}"
                logger.warning(f"Attempt {attempt+1}/{retries} connection failed for {url}")
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP error {e.response.status_code}"
                logger.warning(f"Attempt {attempt+1}/{retries} HTTP error for {url}: {e}")
                # Don't retry on 4xx errors (client errors)
                if 400 <= e.response.status_code < 500:
                    break
            except (requests.exceptions.RequestException, ValueError) as e:
                last_error = f"Request error: {str(e)[:100]}"
                logger.warning(f"Attempt {attempt+1}/{retries} failed for {url}: {e}")
            
            # Sleep with exponential backoff (except on last attempt)
            if attempt < retries - 1:
                time.sleep(backoff_delay)
                backoff_delay = min(backoff_delay * 1.5, 8)  # Cap at 8 seconds
        
        logger.error(f"Failed to fetch {url} after {retries} attempts. Last error: {last_error}")
        return None

    def start(self, hash: str, port: int = None, host: str = None, context_length: int = None) -> bool:
        """
        Start the CryptoModels service in the background.

        Args:
            hash (str): Filecoin hash of the model to download and run.
            port (int): Port number for the CryptoModels service (default from config).
            host (str): Host address for the CryptoModels service (default from config).
            context_length (int): Context length for the model (default from config).

        Returns:
            bool: True if service started successfully, False otherwise.

        Raises:
            ValueError: If hash is not provided when no model is running.
            ModelNotFoundError: If model file is not found.
            ServiceStartError: If service fails to start.
        """
        if not hash:
            raise ValueError("Filecoin hash is required to start the service")
        
        # Use config defaults if not provided
        port = port or config.network.DEFAULT_PORT
        host = host or config.network.DEFAULT_HOST
        context_length = context_length or config.model.DEFAULT_CONTEXT_LENGTH

        # Acquire process lock to prevent concurrent starts
        if not self._acquire_start_lock(hash, host, port):
            return False
        
        # Track processes for cleanup on interruption
        ai_process = None
        apis_process = None
        
        def cleanup_processes():
            """Clean up any running processes."""
            if ai_process and ai_process.poll() is None:
                logger.info("Cleaning up AI process due to interruption...")
                try:
                    ai_process.terminate()
                    ai_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    ai_process.kill()
                except Exception as e:
                    logger.error(f"Error cleaning up AI process: {e}")
            
            if apis_process and apis_process.poll() is None:
                logger.info("Cleaning up API process due to interruption...")
                try:
                    apis_process.terminate()
                    apis_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    apis_process.kill()
                except Exception as e:
                    logger.error(f"Error cleaning up API process: {e}")
        
        def signal_handler(signum, frame):
            """Handle interruption signals."""
            logger.info(f"Received signal {signum}, cleaning up processes...")
            cleanup_processes()
            self._release_start_lock()
            exit(1)
        
        # Set up signal handlers for graceful cleanup
        original_sigint = signal.signal(signal.SIGINT, signal_handler)
        original_sigterm = signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Check if the requested port is available before doing expensive operations
            if not self._check_port_availability(host, port):
                raise ServiceStartError(f"Port {port} is already in use on {host}")
                # Connection refused means port is free, which is what we want

            try:
                logger.info(f"Starting CryptoModels service for model with hash: {hash}")
                
                local_model_path = asyncio.run(download_model_from_filecoin_async(hash))
                if not isinstance(local_model_path, str) or not local_model_path:
                    raise ModelNotFoundError(f"Model file not found for hash: {hash}")
                local_projector_path = local_model_path + "-projector"
                model_running = self.get_running_model()
                local_ai_port = self._get_free_port()
                if model_running:
                    if model_running == hash:
                        logger.warning(f"Model '{hash}' already running on port {port}")
                        return True
                    logger.info(f"Stopping existing model '{model_running}' on port {port}")
                    self.stop(force=True)

                if not os.path.exists(local_model_path):
                    raise ModelNotFoundError(f"Model file not found at: {local_model_path}")

                # Optimized metadata and multimodal checking
                model_dir = os.path.dirname(local_model_path)
                metadata = self._load_or_fetch_metadata(hash, model_dir)
                folder_name = metadata.get("folder_name", "")
                family = metadata.get("family", None)
                ram = metadata.get("ram", None)
                task = metadata.get("task", "chat")
                config_name = metadata.get("config_name", "flux-dev")
                if task == "embed":
                    running_ai_command = self._build_embed_command(local_model_path, local_ai_port, host)
                    service_metadata = self._create_service_metadata(
                        hash, local_model_path, local_ai_port, port, context_length, task, False, None
                    )
                elif task == "image-generation":
                    # require command `mlx-flux`
                    if not shutil.which("mlx-flux"):
                        raise CryptoAgentsServiceError("mlx-flux command not found in PATH")
                    running_ai_command = self._build_image_generation_command(local_model_path, local_ai_port, host, config_name)
                    service_metadata = self._create_service_metadata(
                        hash, local_model_path, local_ai_port, port, context_length, task, False, None
                    )
                else:
                    is_multimodal, projector_path = self._check_multimodal_support(local_model_path)
                    
                    service_metadata = self._create_service_metadata(
                        hash, local_model_path, local_ai_port, port, context_length, 
                        task, is_multimodal, projector_path
                    )

                    if "gemma-3n" in folder_name.lower():
                        template_path, best_practice_path = self._get_family_template_and_practice("gemma-3n")
                        running_ai_command = self._build_ai_command(
                            local_model_path, local_ai_port, host, context_length, template_path
                        )
                    elif "gemma-3" in folder_name.lower():
                        context_length = context_length // 2
                        template_path, best_practice_path = self._get_family_template_and_practice("gemma-3")
                        running_ai_command = self._build_ai_command(
                            local_model_path, local_ai_port, host, context_length, template_path
                        )
                    elif "qwen25" in folder_name.lower():
                        template_path, best_practice_path = self._get_family_template_and_practice("qwen25")
                        running_ai_command = self._build_ai_command(
                            local_model_path, local_ai_port, host, context_length, template_path, best_practice_path
                        )
                    elif "qwen3" in folder_name.lower():
                        template_path, best_practice_path = self._get_family_template_and_practice("qwen3")
                        running_ai_command = self._build_ai_command(
                            local_model_path, local_ai_port, host, context_length, template_path, best_practice_path
                        )
                    elif "llama" in folder_name.lower():
                        template_path, best_practice_path = self._get_family_template_and_practice("llama")
                        running_ai_command = self._build_ai_command(
                            local_model_path, local_ai_port, host, context_length, template_path, best_practice_path
                        )
                    else:
                        running_ai_command = self._build_ai_command(
                            local_model_path, local_ai_port, host, context_length
                        )

                    if service_metadata["multimodal"]:
                        running_ai_command.extend([
                            "--mmproj", str(local_projector_path)
                        ])

                service_metadata["family"] = family
                service_metadata["folder_name"] = folder_name
                service_metadata["ram"] = ram
                
                logger.info(f"Starting process: {' '.join(running_ai_command)}")
                service_metadata["running_ai_command"] = running_ai_command
                # Create log files for stdout and stderr for AI process
                ai_log_stderr = self.logs_dir / "ai.log"
                try:
                    with open(ai_log_stderr, 'w') as stderr_log:
                        ai_process = subprocess.Popen(
                            running_ai_command,
                            stderr=stderr_log,
                            preexec_fn=os.setsid
                        )
                    logger.info(f"AI logs written to {ai_log_stderr}")
                except Exception as e:
                    logger.error(f"Error starting CryptoModels service: {str(e)}", exc_info=True)
                    cleanup_processes()
                    return False
        
                if not wait_for_health(local_ai_port):
                    logger.error(f"Service failed to start within 600 seconds")
                    cleanup_processes()
                    return False
                
                logger.info(f"[CRYPTOMODELS] CryptoModels service started on port {local_ai_port}")

                # start the FastAPI app in the background           
                uvicorn_command = [
                    "uvicorn",
                    "crypto_models.apis:app",
                    "--host", host,
                    "--port", str(port),
                    "--log-level", "info"
                ]
                logger.info(f"Starting process: {' '.join(uvicorn_command)}")
                # Create log files for stdout and stderr
                api_log_stderr = self.logs_dir / "api.log"
                try:
                    with open(api_log_stderr, 'w') as stderr_log:
                        apis_process = subprocess.Popen(
                            uvicorn_command,
                            stderr=stderr_log,
                            preexec_fn=os.setsid
                        )
                    logger.info(f"API logs written to {api_log_stderr}")
                except Exception as e:
                    logger.error(f"Error starting FastAPI app: {str(e)}", exc_info=True)
                    cleanup_processes()
                    return False
                
                if not wait_for_health(port):
                    logger.error(f"API service failed to start within 600 seconds")
                    cleanup_processes()
                    return False

                logger.info(f"Service started on port {port} for model: {hash}")

                # Update service metadata with process IDs
                service_metadata.update({
                    "pid": ai_process.pid,
                    "app_pid": apis_process.pid
                })

                self._dump_running_service(service_metadata)    

                # update service metadata to the FastAPI app
                try:
                    update_url = f"http://localhost:{port}/update"
                    response = requests.post(update_url, json=service_metadata, timeout=10)
                    response.raise_for_status()  # Raise exception for HTTP error responses
                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to update service metadata: {str(e)}")
                    # Stop the partially started service
                    cleanup_processes()
                    return False
                
                return True

            except Exception as e:
                logger.error(f"Error starting CryptoModels service: {str(e)}", exc_info=True)
                cleanup_processes()
                return False
                
        finally:
            # Restore original signal handlers
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)
            # Always remove the lock when done (success or failure)
            self._release_start_lock()

    def _dump_running_service(self, metadata: dict):
        """Dump the running service details to a file."""
        try:
            with open(self.msgpack_file, "wb") as f:
                msgpack.dump(metadata, f)
        except Exception as e:
            logger.error(f"Error dumping running service: {str(e)}", exc_info=True)
            return False

    def get_running_model(self) -> Optional[str]:
        """
        Get currently running model hash if the service is healthy.

        Returns:
            Optional[str]: Running model hash or None if no healthy service exists.
        """
        if not self.msgpack_file.exists():
            return None

        try:
            # Load service info from msgpack file
            with open(self.msgpack_file, "rb") as f:
                service_info = msgpack.load(f)   
            model_hash = service_info.get("hash")      
            return model_hash

        except Exception as e:
            logger.error(f"Error getting running model: {str(e)}")
            return None
    
    def stop(self, force: bool = False) -> bool:
        """
        Stop the running CryptoModels service with optimized process termination.

        Args:
            force (bool): If True, force kill processes immediately without graceful termination.

        Returns:
            bool: True if the service stopped successfully, False otherwise.
        """
        if not os.path.exists(self.msgpack_file):
            logger.warning("No running CryptoModels service to stop.")
            return False

        try:
            # Load service details from the msgpack file
            with open(self.msgpack_file, "rb") as f:
                service_info = msgpack.load(f)
            
            hash_val = service_info.get("hash")
            pid = service_info.get("pid")
            app_pid = service_info.get("app_pid")
            app_port = service_info.get("app_port")
            local_ai_port = service_info.get("port")

            if force:
                logger.info(f"Force stopping CryptoModels service '{hash_val}' running on port {app_port} (AI PID: {pid}, API PID: {app_pid})...")
            else:
                logger.info(f"Stopping CryptoModels service '{hash_val}' running on port {app_port} (AI PID: {pid}, API PID: {app_pid})...")
            
            # Use the optimized termination methods with force parameter
            timeout = 0 if force else 15
            ai_stopped = self._terminate_process_safely(pid, "CryptoModels service", timeout=timeout, force=force)
            api_stopped = self._terminate_process_safely(app_pid, "API service", timeout=timeout, force=force)
            
            # Brief pause to allow system cleanup
            time.sleep(1)
            
            # Verify ports are freed
            ports_info = [(app_port, "API"), (local_ai_port, "AI")]
            ports_freed = self._check_ports_freed(ports_info, max_retries=3)
            
            # Clean up metadata file
            metadata_cleaned = False
            if ai_stopped and api_stopped:
                metadata_cleaned = self._cleanup_service_metadata(force=False)
            else:
                logger.warning("Keeping service metadata file since not all processes were successfully stopped")
            
            # Determine overall success
            success = ai_stopped and api_stopped and metadata_cleaned
            
            if success:
                logger.info("CryptoModels service stopped successfully.")
                if not ports_freed:
                    logger.warning("Service stopped but some ports may still be in use temporarily")
            else:
                logger.error("CryptoModels service stop completed with some failures")
                logger.error(f"CryptoModels stopped: {ai_stopped}, API stopped: {api_stopped}, metadata cleaned: {metadata_cleaned}")
            
            return success

        except Exception as e:
            logger.error(f"Error stopping CryptoModels service: {str(e)}", exc_info=True)
            return False

    def _terminate_process_safely(self, pid: int, process_name: str, timeout: int = 15, use_process_group: bool = True, force: bool = False) -> bool:
        """
        Safely terminate a process with graceful fallback to force kill.
        
        Args:
            pid: Process ID to terminate
            process_name: Human-readable process name for logging
            timeout: Timeout for graceful termination in seconds
            use_process_group: Whether to try process group termination first
            force: If True, force kill processes immediately without graceful termination
            
        Returns:
            bool: True if process was terminated successfully, False otherwise
        """
        if not pid:
            logger.warning(f"No PID provided for {process_name}")
            return True
        
        # Quick existence check
        if not psutil.pid_exists(pid):
            logger.info(f"Process {process_name} (PID: {pid}) not found, assuming already stopped")
            return True
        
        try:
            process = psutil.Process(pid)
            
            # Check if already in terminal state
            try:
                status = process.status()
                if status in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD, psutil.STATUS_STOPPED]:
                    logger.info(f"Process {process_name} (PID: {pid}) already terminated (status: {status})")
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return True
            
            logger.info(f"Terminating {process_name} (PID: {pid})...")
            
            # Collect children before termination
            children = []
            try:
                children = process.children(recursive=True)
                logger.debug(f"Found {len(children)} child processes for {process_name}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            
            # Phase 1: Graceful termination (skip if force=True)
            if not force:
                try:
                    if use_process_group:
                        # Try process group termination first (more efficient)
                        try:
                            pgid = os.getpgid(pid)
                            os.killpg(pgid, signal.SIGTERM)
                            logger.debug(f"Sent SIGTERM to process group {pgid}")
                        except (ProcessLookupError, OSError, PermissionError):
                            # Fall back to individual process termination
                            process.terminate()
                            logger.debug(f"Sent SIGTERM to process {pid}")
                            
                            # Terminate children individually
                            for child in children:
                                try:
                                    child.terminate()
                                    logger.debug(f"Sent SIGTERM to child process {child.pid}")
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                    else:
                        # Individual process termination
                        process.terminate()
                        logger.debug(f"Sent SIGTERM to process {pid}")
                        
                        for child in children:
                            try:
                                child.terminate()
                                logger.debug(f"Sent SIGTERM to child process {child.pid}")
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    logger.info(f"Process {process_name} disappeared during termination")
                    return True
                
                # Wait for graceful termination with exponential backoff
                wait_time = 0.1
                elapsed = 0
                while elapsed < timeout and psutil.pid_exists(pid):
                    time.sleep(wait_time)
                    elapsed += wait_time
                    wait_time = min(wait_time * 1.5, 2.0)  # Cap at 2 seconds
                    
                    # Check if process became zombie
                    try:
                        if process.status() == psutil.STATUS_ZOMBIE:
                            logger.info(f"{process_name} became zombie, considering stopped")
                            return True
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        return True
            else:
                logger.info(f"Force mode enabled - skipping graceful termination for {process_name} (PID: {pid})")
            
            # Phase 2: Force termination if still running
            if psutil.pid_exists(pid):
                logger.warning(f"Force killing {process_name} (PID: {pid})")
                try:
                    # Refresh children list
                    children = []
                    try:
                        process = psutil.Process(pid)
                        children = process.children(recursive=True)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    
                    if use_process_group:
                        # Try process group kill
                        try:
                            pgid = os.getpgid(pid)
                            os.killpg(pgid, signal.SIGKILL)
                            logger.debug(f"Sent SIGKILL to process group {pgid}")
                        except (ProcessLookupError, OSError, PermissionError):
                            # Fall back to individual kill
                            process.kill()
                            logger.debug(f"Sent SIGKILL to process {pid}")
                            
                            for child in children:
                                try:
                                    child.kill()
                                    logger.debug(f"Sent SIGKILL to child process {child.pid}")
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                    else:
                        # Individual process kill
                        process.kill()
                        logger.debug(f"Sent SIGKILL to process {pid}")
                        
                        for child in children:
                            try:
                                child.kill()
                                logger.debug(f"Sent SIGKILL to child process {child.pid}")
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    logger.info(f"Process {process_name} disappeared during force kill")
                    return True
                
                # Final wait for force termination (shorter timeout)
                force_timeout = 5
                for _ in range(force_timeout * 10):  # 0.1s intervals
                    if not psutil.pid_exists(pid):
                        break
                    time.sleep(0.1)
            
            # Final status check
            success = not psutil.pid_exists(pid)
            if success:
                logger.info(f"{process_name} terminated successfully")
            else:
                try:
                    process = psutil.Process(pid)
                    status = process.status()
                    if status == psutil.STATUS_ZOMBIE:
                        logger.warning(f"{process_name} is zombie but considered stopped")
                        success = True
                    else:
                        logger.error(f"Failed to terminate {process_name} (PID: {pid}), status: {status}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    success = True
                    
            return success
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            logger.info(f"Process {process_name} (PID: {pid}) no longer accessible")
            return True
        except Exception as e:
            logger.error(f"Error terminating {process_name} (PID: {pid}): {e}")
            return False

    async def _terminate_process_safely_async(self, pid: int, process_name: str, timeout: int = 15, use_process_group: bool = True) -> bool:
        """
        Async version of _terminate_process_safely for use in async contexts.
        
        Args:
            pid: Process ID to terminate
            process_name: Human-readable process name for logging
            timeout: Timeout for graceful termination in seconds
            use_process_group: Whether to try process group termination first
            
        Returns:
            bool: True if process was terminated successfully, False otherwise
        """
        if not pid:
            logger.warning(f"No PID provided for {process_name}")
            return True
        
        # Quick existence check
        if not psutil.pid_exists(pid):
            logger.info(f"Process {process_name} (PID: {pid}) not found, assuming already stopped")
            return True
        
        try:
            process = psutil.Process(pid)
            
            # Check if already in terminal state
            try:
                status = process.status()
                if status in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD, psutil.STATUS_STOPPED]:
                    logger.info(f"Process {process_name} (PID: {pid}) already terminated (status: {status})")
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return True
            
            logger.info(f"Terminating {process_name} (PID: {pid})...")
            
            # Collect children before termination
            children = []
            try:
                children = process.children(recursive=True)
                logger.debug(f"Found {len(children)} child processes for {process_name}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            
            # Phase 1: Graceful termination
            try:
                if use_process_group:
                    # Try process group termination first
                    try:
                        pgid = os.getpgid(pid)
                        os.killpg(pgid, signal.SIGTERM)
                        logger.debug(f"Sent SIGTERM to process group {pgid}")
                    except (ProcessLookupError, OSError, PermissionError):
                        # Fall back to individual process termination
                        process.terminate()
                        logger.debug(f"Sent SIGTERM to process {pid}")
                        
                        for child in children:
                            try:
                                child.terminate()
                                logger.debug(f"Sent SIGTERM to child process {child.pid}")
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                else:
                    # Individual process termination
                    process.terminate()
                    logger.debug(f"Sent SIGTERM to process {pid}")
                    
                    for child in children:
                        try:
                            child.terminate()
                            logger.debug(f"Sent SIGTERM to child process {child.pid}")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                            
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                logger.info(f"Process {process_name} disappeared during termination")
                return True
            
            # Wait for graceful termination with async sleep
            wait_time = 0.1
            elapsed = 0
            while elapsed < timeout and psutil.pid_exists(pid):
                await asyncio.sleep(wait_time)
                elapsed += wait_time
                wait_time = min(wait_time * 1.5, 2.0)  # Cap at 2 seconds
                
                # Check if process became zombie
                try:
                    if process.status() == psutil.STATUS_ZOMBIE:
                        logger.info(f"{process_name} became zombie, considering stopped")
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    return True
            
            # Phase 2: Force termination if still running
            if psutil.pid_exists(pid):
                logger.warning(f"Force killing {process_name} (PID: {pid})")
                try:
                    # Refresh children list
                    children = []
                    try:
                        process = psutil.Process(pid)
                        children = process.children(recursive=True)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    
                    if use_process_group:
                        # Try process group kill
                        try:
                            pgid = os.getpgid(pid)
                            os.killpg(pgid, signal.SIGKILL)
                            logger.debug(f"Sent SIGKILL to process group {pgid}")
                        except (ProcessLookupError, OSError, PermissionError):
                            # Fall back to individual kill
                            process.kill()
                            logger.debug(f"Sent SIGKILL to process {pid}")
                            
                            for child in children:
                                try:
                                    child.kill()
                                    logger.debug(f"Sent SIGKILL to child process {child.pid}")
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                    else:
                        # Individual process kill
                        process.kill()
                        logger.debug(f"Sent SIGKILL to process {pid}")
                        
                        for child in children:
                            try:
                                child.kill()
                                logger.debug(f"Sent SIGKILL to child process {child.pid}")
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    logger.info(f"Process {process_name} disappeared during force kill")
                    return True
                
                # Final wait for force termination with async sleep
                force_timeout = 5
                for _ in range(force_timeout * 10):  # 0.1s intervals
                    if not psutil.pid_exists(pid):
                        break
                    await asyncio.sleep(0.1)
            
            # Final status check
            success = not psutil.pid_exists(pid)
            if success:
                logger.info(f"{process_name} terminated successfully")
            else:
                try:
                    process = psutil.Process(pid)
                    status = process.status()
                    if status == psutil.STATUS_ZOMBIE:
                        logger.warning(f"{process_name} is zombie but considered stopped")
                        success = True
                    else:
                        logger.error(f"Failed to terminate {process_name} (PID: {pid}), status: {status}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    success = True
                    
            return success
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            logger.info(f"Process {process_name} (PID: {pid}) no longer accessible")
            return True
        except Exception as e:
            logger.error(f"Error terminating {process_name} (PID: {pid}): {e}")
            return False

    def _cleanup_service_metadata(self, force: bool = False) -> bool:
        """
        Clean up service metadata file with proper error handling.
        
        Args:
            force: If True, remove file even if processes might still be running
            
        Returns:
            bool: True if cleanup was successful, False otherwise
        """
        try:
            if not os.path.exists(self.msgpack_file):
                logger.debug("Service metadata file already removed")
                return True
                
            if not force:
                # Verify that processes are actually stopped before cleanup
                try:
                    with open(self.msgpack_file, "rb") as f:
                        service_info = msgpack.load(f)
                    
                    pid = service_info.get("pid")
                    app_pid = service_info.get("app_pid")
                    
                    # Check if any processes are still running
                    running_processes = []
                    if pid and psutil.pid_exists(pid):
                        running_processes.append(f"AI server (PID: {pid})")
                    if app_pid and psutil.pid_exists(app_pid):
                        running_processes.append(f"API server (PID: {app_pid})")
                    
                    if running_processes:
                        logger.warning(f"Not cleaning up metadata - processes still running: {', '.join(running_processes)}")
                        return False
                        
                except Exception as e:
                    logger.warning(f"Could not verify process status, proceeding with cleanup: {e}")
            
            os.remove(self.msgpack_file)
            logger.info("Service metadata file removed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error removing service metadata file: {str(e)}")
            return False

    def _get_model_template_path(self, model_family: str) -> str:
        """Get the template path for a specific model family."""
        chat_template_path = pkg_resources.resource_filename("crypto_models", f"examples/templates/{model_family}.jinja")
        # check if the template file exists
        if not os.path.exists(chat_template_path):
            return None
        return chat_template_path

    def _get_model_best_practice_path(self, model_family: str) -> str:
        """Get the best practices for a specific model family."""
        best_practice_path = pkg_resources.resource_filename("crypto_models", f"examples/best_practices/{model_family}.json")
        # check if the best practices file exists
        if not os.path.exists(best_practice_path):
            return None
        return best_practice_path
    
    def _build_embed_command(self, model_path: str, port: int, host: str) -> list:
        """Build the embed command with common parameters."""
        command = [
            self.llama_server_path,
            "--model", str(model_path),
            "--port", str(port),
            "--host", host,
            "--embedding",
            "--pooling", "cls",
            "-ub", "8192",
            "-ngl", "9999"
        ]
        return command

    def _build_ai_command(self, model_path: str, port: int, host: str, context_length: int, template_path: Optional[str] = None, best_practice_path: Optional[str] = None) -> list:
        """Build the AI command with common parameters."""
        command = [
            self.llama_server_path,
            "--model", str(model_path),
            "--port", str(port),
            "--host", host,
            "-c", str(context_length),
            "-fa",
            "--pooling", "cls",
            "--embeddings",
            "--no-webui",
            "-ngl", "9999",
            "--jinja",
            "--reasoning-format", "none"
        ]
        
        if template_path:
            command.extend(["--chat-template-file", template_path])
        
        if best_practice_path:
            with open(best_practice_path, "r") as f:
                best_practice = json.load(f)
                for key, value in best_practice.items():
                    command.extend([f"--{key}", str(value)])
        return command

    def _load_or_fetch_metadata(self, hash: str, model_dir: str) -> dict:
        """
        Load metadata from cache or fetch from remote with optimized I/O.
        Returns metadata_dict
        """
        metadata_file = os.path.join(model_dir, f"{hash}.json")
        
        # Try to load from cache first
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    logger.info(f"Loaded metadata from cache: {metadata_file}")
                    return metadata
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Invalid metadata file {metadata_file}, will fetch from remote: {e}")
                # Remove corrupted file
                try:
                    os.remove(metadata_file)
                except OSError:
                    pass
        
        # Fetch from remote
        filecoin_url = f"https://gateway.mesh3.network/ipfs/{hash}"
        response_json = self._retry_request_json(filecoin_url, retries=3, delay=5, timeout=10)
        
        if not response_json:
            logger.error(f"Failed to fetch metadata from {filecoin_url}")
            return {}
        
        # Save to cache (create directory if needed)
        try:
            os.makedirs(model_dir, exist_ok=True)
            with open(metadata_file, "w") as f:
                json.dump(response_json, f)
            logger.info(f"Cached metadata to {metadata_file}")
        except IOError as e:
            logger.warning(f"Failed to cache metadata: {e}")
        
        return response_json
    
    def _check_multimodal_support(self, local_model_path: str) -> tuple[bool, Optional[str]]:
        """
        Check if model supports multimodal with single file check.
        Returns (is_multimodal, projector_path)
        """
        projector_path = f"{local_model_path}-projector"
        is_multimodal = os.path.exists(projector_path)
        return is_multimodal, projector_path if is_multimodal else None

    def _create_service_metadata(self, hash: str, local_model_path: str, local_ai_port: int, 
                                port: int, context_length: int, task: str, 
                                is_multimodal: bool, projector_path: Optional[str]) -> dict:
        """Create service metadata dictionary with all required fields."""
        return {
            "task": task,
            "hash": hash,
            "port": local_ai_port,
            "local_text_path": local_model_path,
            "app_port": port,
            "context_length": context_length,
            "last_activity": time.time(),
            "multimodal": is_multimodal,
            "local_projector_path": projector_path
        }

    def _acquire_start_lock(self, hash: str, host: str, port: int) -> bool:
        """
        Acquire the start lock efficiently with minimal I/O operations.
        Returns True if lock acquired, False otherwise.
        """
        current_pid = os.getpid()
        current_time = time.time()
        
        # Check and handle existing lock
        if self.start_lock_file.exists():
            try:
                with open(self.start_lock_file, "r") as f:
                    lock_data = json.load(f)
                
                lock_pid = lock_data.get("pid")
                lock_timestamp = lock_data.get("timestamp", 0)
                
                # Check if lock is stale (over 30 minutes) or process is dead
                if current_time - lock_timestamp > self.LOCK_TIMEOUT or not psutil.pid_exists(lock_pid):
                    logger.warning(f"Removing stale lock file (PID: {lock_pid})")
                    self.start_lock_file.unlink()
                elif lock_pid != current_pid:
                    logger.error(f"Another process (PID: {lock_pid}) is already starting service")
                    return False
                    
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning(f"Corrupted lock file, removing: {e}")
                try:
                    self.start_lock_file.unlink()
                except OSError:
                    pass
        
        # Create new lock
        lock_data = {
            "pid": current_pid,
            "hash": hash,
            "timestamp": current_time,
            "host": host,
            "port": port
        }
        
        try:
            with open(self.start_lock_file, "w") as f:
                json.dump(lock_data, f)
            logger.info(f"Acquired start lock for PID {current_pid}")
            return True
        except OSError as e:
            logger.error(f"Failed to create lock file: {e}")
            return False
    
    def _release_start_lock(self) -> None:
        """Release the start lock if we own it."""
        if not self.start_lock_file.exists():
            return
            
        current_pid = os.getpid()
        try:
            with open(self.start_lock_file, "r") as f:
                lock_data = json.load(f)
            
            if lock_data.get("pid") == current_pid:
                self.start_lock_file.unlink()
                logger.info(f"Released start lock for PID {current_pid}")
            else:
                logger.warning(f"Lock file PID mismatch, not removing")
                
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"Error releasing lock: {e}")
            try:
                self.start_lock_file.unlink()
                logger.info("Removed potentially corrupted lock file")
            except OSError:
                pass

    def _check_port_availability(self, host: str, port: int, timeout: int = 2) -> bool:
        """
        Check if a port is available (not in use).
        Returns True if port is free, False if in use.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                result = s.connect_ex((host, port))
                return result != 0  # 0 means connection successful (port in use)
        except socket.error:
            return True  # Assume port is free if we can't check
        
    def _check_ports_freed(self, ports_info: list, max_retries: int = 5) -> bool:
        """
        Check if multiple ports are freed with optimized retry logic.
        ports_info: list of (port, service_name) tuples
        Returns True if all ports are freed.
        """
        if not ports_info:
            return True
            
        freed_ports = set()
        
        for retry in range(max_retries):
            for port, service_name in ports_info:
                if port and port not in freed_ports:
                    if self._check_port_availability('localhost', port, timeout=1):
                        freed_ports.add(port)
                        logger.debug(f"{service_name} port {port} is now free")
            
            # If all ports are freed, return early
            if len(freed_ports) == len([p for p, _ in ports_info if p]):
                return True
                
            if retry < max_retries - 1:
                time.sleep(1)
        
        # Log which ports are still in use
        for port, service_name in ports_info:
            if port and port not in freed_ports:
                logger.warning(f"{service_name} port {port} still in use after {max_retries} checks")
        
        return len(freed_ports) == len([p for p, _ in ports_info if p])
    
    async def kill_ai_server(self) -> bool:
        """Kill the AI server process if it's running (optimized async version)."""
        try:
            if not os.path.exists(self.msgpack_file):
                logger.warning("No service info found, cannot kill AI server")
                return False
                
            # Load service details from the msgpack file
            with open(self.msgpack_file, "rb") as f:
                service_info = msgpack.load(f)
                
            pid = service_info.get("pid")
            if not pid:
                logger.warning("No PID found in service info, cannot kill AI server")
                return False
                
            logger.info(f"Attempting to kill AI server with PID {pid}")
            
            # Use the optimized async termination method
            success = await self._terminate_process_safely_async(pid, "AI server", timeout=15)
            
            # Clean up service info if process was successfully killed
            if success:
                try:
                    # Remove PID from service info to indicate server is no longer running
                    service_info.pop("pid", None)
                    
                    with open(self.msgpack_file, "wb") as f:
                        msgpack.dump(service_info, f)
                    
                    logger.info("AI server stopped successfully and service info cleaned up")
                except Exception as e:
                    logger.warning(f"AI server stopped but failed to clean up service info: {str(e)}")
                
                return True
            else:
                logger.error("Failed to stop AI server")
                return False
            
        except Exception as e:
            logger.error(f"Error killing AI server: {str(e)}", exc_info=True)
            return False
    
    async def reload_ai_server(self, service_start_timeout: int = 120) -> bool:
        """Reload the AI server process (async version for API usage)."""
        try:
            if not os.path.exists(self.msgpack_file):
                logger.error("No service info found, cannot reload AI server")
                return False
                
            # Load service details from the msgpack file
            with open(self.msgpack_file, "rb") as f:
                service_info = msgpack.load(f)
                
            running_ai_command = service_info.get("running_ai_command")
            if not running_ai_command:
                logger.error("No running_ai_command found in service info, cannot reload AI server")
                return False
                
            logger.info(f"Reloading AI server with command: {running_ai_command}")

            ai_log_stderr = self.logs_dir / "ai.log"
            
            try:
                with open(ai_log_stderr, 'w') as stderr_log:
                    ai_process = subprocess.Popen(
                        running_ai_command,
                        stderr=stderr_log,
                        preexec_fn=os.setsid
                    )
                logger.info(f"AI logs written to {ai_log_stderr}")
            except Exception as e:
                logger.error(f"Error starting CryptoModels service: {str(e)}", exc_info=True)
                return False
            
            # Wait for the process to start by checking the health endpoint
            port = service_info["port"]
            if not wait_for_health(port, timeout=service_start_timeout):
                logger.error(f"AI server failed to start within {service_start_timeout} seconds")
                return False
            
            # Check if the process is running
            if ai_process.poll() is None:
                # Update the service info with new PID
                service_info["pid"] = ai_process.pid
                with open(self.msgpack_file, "wb") as f:
                    msgpack.dump(service_info, f)
                logger.info(f"Successfully reloaded AI server with PID {ai_process.pid}")
                return True
            else:
                logger.error(f"Failed to reload AI server: Process exited with code {ai_process.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"Error reloading AI server: {str(e)}", exc_info=True)
            return False
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service info from msgpack file with error handling."""
        if not os.path.exists(self.msgpack_file):
            raise CryptoAgentsServiceError("Service information not available")
        
        try:
            with open(self.msgpack_file, "rb") as f:
                return msgpack.load(f)
        except Exception as e:
            raise CryptoAgentsServiceError(f"Failed to load service info: {str(e)}")
    
    def update_service_info(self, updates: Dict[str, Any]) -> bool:
        """Update service information in the msgpack file."""
        try:
            if os.path.exists(self.msgpack_file):
                with open(self.msgpack_file, "rb") as f:
                    service_info = msgpack.load(f)
            else:
                service_info = {}
            
            service_info.update(updates)
            
            with open(self.msgpack_file, "wb") as f:
                msgpack.dump(service_info, f)
            
            return True
        except Exception as e:
            logger.error(f"Failed to update service info: {str(e)}")
            return False

    def _build_image_generation_command(self, model_path: str, port: int, host: str, config_name: str) -> list:
        """Build the image-generation command with MLX Flux parameters."""
        command = [
            "mlx-flux",
            "serve",
            "--model-path", str(model_path),
            "--config-name", config_name,
            "--port", str(port),
            "--host", host
        ]
        return command