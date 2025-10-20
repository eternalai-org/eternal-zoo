import os
import sys
import json
import time
import shutil
import signal
import msgpack
import psutil
import asyncio
import socket
import subprocess
import pkg_resources
from pathlib import Path
from loguru import logger
from eternal_zoo.config import DEFAULT_CONFIG
from eternal_zoo.utils import wait_for_health
from typing import Optional, Dict, Any, List

class EternalZooServiceError(Exception):
    """Base exception for EternalZoo service errors."""
    pass

class ServiceStartError(EternalZooServiceError):
    """Exception raised when service fails to start."""
    pass

class ModelNotFoundError(EternalZooServiceError):
    """Exception raised when model file is not found."""
    pass

class EternalZooManager:
    """Manages an EternalZoo service with optimized performance."""
    
    def __init__(self):
        """Initialize the EternalZooManager with optimized defaults.""" 
        # Performance constants from config
        self.LOCK_TIMEOUT = DEFAULT_CONFIG.core.LOCK_TIMEOUT
        self.PORT_CHECK_TIMEOUT = DEFAULT_CONFIG.core.PORT_CHECK_TIMEOUT
        self.HEALTH_CHECK_TIMEOUT = DEFAULT_CONFIG.core.HEALTH_CHECK_TIMEOUT
        self.PROCESS_TERM_TIMEOUT = DEFAULT_CONFIG.core.PROCESS_TERM_TIMEOUT
        self.MAX_PORT_RETRIES = DEFAULT_CONFIG.core.MAX_PORT_RETRIES
        
        # File paths from config
        self.ai_service_file = Path(DEFAULT_CONFIG.file_paths.AI_SERVICE_FILE)
        self.api_service_file = Path(DEFAULT_CONFIG.file_paths.API_SERVICE_FILE)
        self.service_info_file = Path(DEFAULT_CONFIG.file_paths.SERVICE_INFO_FILE)

        self.llama_server_path = DEFAULT_CONFIG.file_paths.LLAMA_SERVER
        self.logs_dir = Path(DEFAULT_CONFIG.file_paths.LOGS_DIR)
        self.logs_dir.mkdir(exist_ok=True)
        self.ai_log_file = self.logs_dir / "ai.log"
        self.api_log_file = self.logs_dir / "api.log"
        
        # Last model switch error for surfacing to API layer
        self.last_switch_error: Optional[str] = None
        # Optional context length override supplied by request layer
        self.switch_ctx_override: Optional[int] = None
        
    def _get_free_port(self) -> int:
        """Get a free port number."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def _get_family_template_and_practice(self, model_family: str):
        """Helper to get template and best practice paths based on folder name."""
        return (
            self._get_model_template_path(model_family),
            self._get_model_best_practice_path(model_family)
        )

    def _check_port_availability(self, host: str, port: int) -> bool:
        """Check if a port is available on the given host."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex((host, port))
                return result != 0  # Port is available if connection fails
        except Exception:
            return False
        
    def start(self, configs: List[dict], port: int = 8080, host: str = "0.0.0.0") -> bool:
        """
        Start the EternalZoo service with a given config.

        Args:
            config (dict): The config to start the service with.

        Returns:
            bool: True if service started successfully, False otherwise.
        """

        if not self._check_port_availability(host, port):
            raise ServiceStartError(f"Port {port} is already in use on {host}")
        
        # stop the service if it is already running
        self.stop()

        ai_services = []
        api_service = {
            "host": host,
            "port": port,
        }

        for config in configs:

            task = config.get("task", "chat")
            running_ai_command = None
            ai_service = config.copy()

            local_model_port = self._get_free_port()    

            if task == "embed":
                logger.info(f"Starting embed model: {config}")
                running_ai_command = self._build_embed_command(config)
            elif task == "chat":
                logger.info(f"Starting chat model: {config}")
                running_ai_command = self._build_chat_command(config)
            elif task == "image-generation":
                logger.info(f"Starting image generation model: {config}")
                if not shutil.which("mlx-openai-server"):
                    raise EternalZooServiceError("mlx-openai-server command not found in PATH")
                running_ai_command = self._build_image_generation_command(config)
            elif task == "image-edit":
                logger.warning(f"Image edit is not implemented yet: {config}")
                continue
            else:
                continue

            if running_ai_command is None:
                raise ValueError(f"Invalid running AI command: {running_ai_command}")
            
            ai_service["running_ai_command"] = running_ai_command
            logger.info(f"Running command: {' '.join(running_ai_command)}")

            if not config.get("on_demand", False):
                try:
                    # append port and host to the running_ai_command
                    running_ai_command.extend(["--port", str(local_model_port), "--host", host])
                    with open(self.ai_log_file, 'w') as stderr_log:
                        ai_process = subprocess.Popen(
                            running_ai_command,
                            stderr=stderr_log,
                            preexec_fn=os.setsid
                        )
                        logger.info(f"AI logs written to {self.ai_log_file}")
                    ai_service["created"] = int(time.time())
                    ai_service["owned_by"] = "user"
                    ai_service["active"] = True
                    ai_service["pid"] = ai_process.pid
                    ai_service["port"] = local_model_port
                    ai_service["host"] = host
                    ai_services.append(ai_service)
                    with open(self.ai_service_file, 'wb') as f:
                        msgpack.pack(ai_services, f)
                    logger.info(f"AI service metadata written to {self.ai_service_file}")
                    if not wait_for_health(local_model_port):
                        self.stop()
                        logger.error(f"Service failed to start within 120 seconds")
                        return False
                except Exception as e:
                    self.stop()
                    logger.error(f"Error starting EternalZoo service: {str(e)}", exc_info=True)
                    return False
            else:
                ai_service["created"] = int(time.time())
                ai_service["owned_by"] = "user"
                ai_service["active"] = False
                ai_services.append(ai_service)
                with open(self.ai_service_file, 'wb') as f:
                    msgpack.pack(ai_services, f)
                logger.info(f"AI service metadata written to {self.ai_service_file}")
                
            logger.info(f"[ETERNALZOO] Model service started on port {local_model_port}")
       
        # Start the FastAPI app
        uvicorn_command = [
            "uvicorn",
            "eternal_zoo.apis:app",
            "--host", host,
            "--port", str(port),
            "--log-level", "info"
        ]

        logger.info(f"Starting API process: {' '.join(uvicorn_command)}")

        try:
            with open(self.api_log_file, 'w') as stderr_log:
                api_process = subprocess.Popen(
                    uvicorn_command,
                    stderr=stderr_log,
                    preexec_fn=os.setsid
                )

                api_service["pid"] = api_process.pid

                logger.info(f"API logs written to {self.api_log_file}")

            with open(self.api_service_file, 'wb') as f:
                msgpack.pack(api_service, f)

            logger.info(f"API service metadata written to {self.api_service_file}")
            
        except Exception as e:
            self.stop()
            logger.error(f"Error writing proxy service metadata: {str(e)}", exc_info=True)
            return False
        
        self.update_service_info({            
            "api_service": api_service,
            "ai_services": ai_services,
        })

        return True
    
    def stop(self) -> bool:

        if not self.ai_service_file.exists() and not self.api_service_file.exists() and not self.service_info_file.exists():
            logger.warning("No running EternalZoo service to stop.")
            return False
        
        if self.service_info_file.exists():
            os.remove(self.service_info_file)
            logger.info(f"Service info file removed: {self.service_info_file}")
        
        # always force kill the service
        ai_services = []
        ai_service_stop = False
        api_service_stop = False
        
        if self.ai_service_file.exists():
            with open(self.ai_service_file, 'rb') as f:
                ai_services = msgpack.unpack(f)
            for ai_service in ai_services:
                pid = ai_service.get("pid", None)
                if pid and psutil.pid_exists(pid):
                    ai_service_stop = self._terminate_process_safely(pid, "EternalZoo AI Service", force=True)
            
            if ai_service_stop:
                os.remove(self.ai_service_file)
                logger.info(f"AI service metadata file removed: {self.ai_service_file}")
            else:
                logger.warning("Failed to stop EternalZoo AI Service")
        
        if self.api_service_file.exists():
            with open(self.api_service_file, 'rb') as f:
                api_service = msgpack.unpack(f)
            pid = api_service.get("pid", None)
            if pid and psutil.pid_exists(pid):
                api_service_stop = self._terminate_process_safely(pid, "EternalZoo API Service", force=True)
            
            if api_service_stop:
                os.remove(self.api_service_file)
                logger.info(f"API service metadata file removed: {self.api_service_file}")
            else:
                logger.warning("Failed to stop EternalZoo API Service")
        
        return True

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
                    # In force mode, also kill any remaining child processes
                    if force:
                        try:
                            children = process.children(recursive=True)
                            for child in children:
                                try:
                                    child.kill()
                                    logger.debug(f"Force killed child process {child.pid}")
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
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
                    
                    # Check if any processes are still running (excluding zombies)
                    running_processes = []
                    if pid and psutil.pid_exists(pid):
                        try:
                            process = psutil.Process(pid)
                            status = process.status()
                            if status not in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD, psutil.STATUS_STOPPED]:
                                running_processes.append(f"AI server (PID: {pid})")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    if app_pid and psutil.pid_exists(app_pid):
                        try:
                            process = psutil.Process(app_pid)
                            status = process.status()
                            if status not in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD, psutil.STATUS_STOPPED]:
                                running_processes.append(f"API server (PID: {app_pid})")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    
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

    def _get_model_template_path(self, model_family: str | None = None) -> str:
        if model_family is None:
            return None
        """Get the template path for a specific model family."""
        chat_template_path = pkg_resources.resource_filename("eternal_zoo", f"examples/templates/{model_family}.jinja")
        # check if the template file exists
        if not os.path.exists(chat_template_path):
            return None
        return chat_template_path

    def _get_model_best_practice_path(self, model_family: str | None = None) -> str:
        if model_family is None:
            return None
        """Get the best practices for a specific model family."""
        best_practice_path = pkg_resources.resource_filename("eternal_zoo", f"examples/best_practices/{model_family}.json")
        # check if the best practices file exists
        if not os.path.exists(best_practice_path):
            return None
        return best_practice_path
    
    def _estimate_model_ram_gb(self, ai_service: Dict[str, Any], context_length_override: Optional[int] = None) -> float:
        """Estimate RAM needed to load a model.
        Priority:
        1) Use explicit 'estimated_ram_gb' if provided in service config
        2) For GGUF models, base on model file size with overhead multiplier
        3) Fallback default
        """
        try:
            if "estimated_ram_gb" in ai_service and ai_service["estimated_ram_gb"] is not None:
                return float(ai_service["estimated_ram_gb"])  # trusted override
        except (TypeError, ValueError):
            pass

        backend = ai_service.get("backend", "gguf")
        model_path = ai_service.get("model")
        if backend == "gguf" and model_path and os.path.exists(model_path):
            # Prefer a gguf-based estimate including KV cache if library is available
            configured_context = ai_service.get("context_length", 32768)
            context_length = int(context_length_override) if context_length_override is not None else int(configured_context)
            try:
                file_size_gb_dbg = os.path.getsize(model_path) / (1024 ** 3)
            except Exception:
                file_size_gb_dbg = None
            file_size_str = f"{file_size_gb_dbg:.2f}GB" if file_size_gb_dbg is not None else "unknown"
            logger.info(
                f"RAM estimate inputs: backend={backend}, model='{model_path}', "
                f"configured_ctx={configured_context}, override_ctx={context_length_override}, used_ctx={context_length}, "
                f"file_size={file_size_str}"
            )
            estimate_gb = None
            try:
                import gguf  # type: ignore
                try:
                    # Newer gguf API may accept a path directly
                    reader = gguf.GGUFReader(model_path)
                except Exception:
                    # Fallback: open file handle
                    with open(model_path, "rb") as f:
                        reader = gguf.GGUFReader(f)  # type: ignore

                # Try multiple common metadata keys for layer and embedding size
                meta_get = getattr(reader, "get_value", None)
                n_layer = None
                n_embd = None
                if callable(meta_get):
                    for key in [
                        "llama.block_count",
                        "llama.layers",
                        "general.block_count",
                        "general.layers",
                    ]:
                        try:
                            val = meta_get(key)
                            if isinstance(val, (int, float)):
                                n_layer = int(val)
                                break
                        except Exception:
                            pass
                    for key in [
                        "llama.embedding_length",
                        "llama.hidden_size",
                        "general.embedding_length",
                        "general.hidden_size",
                    ]:
                        try:
                            val = meta_get(key)
                            if isinstance(val, (int, float)):
                                n_embd = int(val)
                                break
                        except Exception:
                            pass

                # If metadata unavailable, try to use configuration values before falling back
                if not n_layer or not n_embd:
                    n_layer_cfg = None
                    n_embd_cfg = None
                    try:
                        for key in [
                            "n_layer", "num_layers", "layers", "llama.block_count", "general.block_count"
                        ]:
                            if key in ai_service and ai_service.get(key) is not None:
                                n_layer_cfg = int(ai_service.get(key))
                                break
                    except Exception:
                        n_layer_cfg = None

                    try:
                        for key in [
                            "n_embd", "hidden_size", "embedding_length", "general.embedding_length"
                        ]:
                            if key in ai_service and ai_service.get(key) is not None:
                                n_embd_cfg = int(ai_service.get(key))
                                break
                    except Exception:
                        n_embd_cfg = None

                    if n_layer_cfg and n_embd_cfg:
                        n_layer = n_layer_cfg
                        n_embd = n_embd_cfg
                        logger.info(f"Using config-supplied metadata: n_layer={n_layer}, n_embd={n_embd}")
                        try:
                            print(f"[RAM EST META CONFIG] n_layer={n_layer} n_embd={n_embd}")
                        except Exception:
                            pass
                    else:
                        raise RuntimeError("gguf metadata missing for n_layer or n_embd")

                # Assume f16 KV cache unless overridden by runtime flags or config
                bytes_per_element = 2
                try:
                    kv_bpe_cfg = ai_service.get("kv_bytes_per_element", None)
                    if kv_bpe_cfg is not None:
                        bytes_per_element = int(kv_bpe_cfg)
                    else:
                        kv_dtype = ai_service.get("kv_dtype", None)
                        if isinstance(kv_dtype, str):
                            kv_dtype_l = kv_dtype.lower()
                            if kv_dtype_l in ("f16", "bf16"):
                                bytes_per_element = 2
                            elif kv_dtype_l in ("f32", "float32"):
                                bytes_per_element = 4
                            elif kv_dtype_l in ("q8", "q8_kv", "int8"):
                                bytes_per_element = 1
                except Exception:
                    # Ignore malformed overrides and keep default
                    pass
                logger.info(f"GGUF metadata: n_layer={n_layer}, n_embd={n_embd}, bytes_per_element={bytes_per_element}")
                try:
                    print(f"[RAM EST META] n_layer={n_layer} n_embd={n_embd} bytes_per_element={bytes_per_element}")
                except Exception:
                    pass
                kv_cache_bytes = 2 * n_layer * int(context_length) * n_embd * bytes_per_element
                kv_cache_gb = kv_cache_bytes / (1024 ** 3)

                # Weight residency: mostly mmapped, but budget a small residency buffer
                file_size_gb = os.path.getsize(model_path) / (1024 ** 3)
                weight_residency_buffer_gb = min(2.0, file_size_gb * 0.05)
                logger.info(
                    f"GGUF calc parts: file_size={file_size_gb:.2f}GB, kv_cache={kv_cache_gb:.2f}GB, weight_buffer={weight_residency_buffer_gb:.2f}GB"
                )
                try:
                    print(f"[RAM EST PARTS] file_size={file_size_gb:.2f}GB kv_cache={kv_cache_gb:.2f}GB weight_buffer={weight_residency_buffer_gb:.2f}GB")
                except Exception:
                    pass

                estimate_gb = kv_cache_gb + weight_residency_buffer_gb + 2.0  # extra headroom
                logger.info(
                    f"GGUF estimate using metadata: layers={n_layer}, emb={n_embd}, ctx={context_length}, "
                    f"kv~{kv_cache_gb:.2f}GB, buffer~{weight_residency_buffer_gb:.2f}GB => total~{estimate_gb:.2f}GB"
                )
            except Exception as e:
                # Fallback to file-size heuristic when gguf parsing not available
                try:
                    file_size_gb = os.path.getsize(model_path) / (1024 ** 3)
                    estimate_gb = file_size_gb * 1.25 + 4.0
                    logger.warning(
                        f"GGUF-based estimation failed ({type(e).__name__}: {str(e)}); "
                        f"using file-size heuristic size={file_size_gb:.2f}GB -> estimate={estimate_gb:.2f}GB"
                    )
                except Exception:
                    pass

            if estimate_gb is None:
                estimate_gb = 12.0
            return max(estimate_gb, 8.0)

        # Default conservative fallback
        return 12.0

    def _get_model_family(self, model_name: str | None = None) -> str:
        if model_name is None:
            return None
        model_name = model_name.lower()

        if "gpt-oss" in model_name:
            return "gpt-oss"
        if "jan-v1" in model_name:
            return "jan-v1"
        if "qwen3-coder" in model_name:
            return "qwen3-coder"
        if "qwen3" in model_name:
            if "2507" and "thinking" in model_name:
                return "qwen3-thinking-2507"
            elif "2507" and "instruct" in model_name:
                return "qwen3-instruct-2507"
            else:
                return "qwen3"
        if "qwen2.5" in model_name:
            return "qwen2.5"
        if "lfm2" in model_name:
            return "lfm2"
        if "openreasoning-nemotron" in model_name:
            return "openreasoning-nemotron"
        if "dolphin-3.0" in model_name:
            return "dolphin-3.0"
        if "dolphin-3.1" in model_name:
            return "dolphin-3.1"
        if "devstral-small" in model_name:
            return "devstral-small"
        if "gemma-3n" in model_name:
            return "gemma-3n"
        if "gemma-3" in model_name:
            return "gemma-3"
    
        return None
    
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

    def get_service_info(self) -> Dict[str, Any]:
        """Get service info from msgpack file with error handling."""
        if not os.path.exists(self.service_info_file):
            raise EternalZooServiceError("Service information not available")
        
        try:
            with open(self.service_info_file, "rb") as f:
                return msgpack.load(f)
        except Exception as e:
            raise EternalZooServiceError(f"Failed to load service info: {str(e)}")
    
    def update_service_info(self, updates: Dict[str, Any]) -> bool:
        """Update service information in the msgpack file."""
        try:
            if os.path.exists(self.service_info_file):
                with open(self.service_info_file, "rb") as f:
                    service_info = msgpack.load(f)
            else:
                service_info = {}
            
            service_info.update(updates)
            
            with open(self.service_info_file, "wb") as f:
                msgpack.dump(service_info, f)
            
            return True
        except Exception as e:
            logger.error(f"Failed to update service info: {str(e)}")
            return False
        
    def update_lora(self, request: Dict[str, Any]) -> bool:
        """Update the LoRA for a given model hash."""
        try:
            service_info = self.get_service_info()
            ai_services = service_info.get("ai_services", [])
            for ai_service in ai_services:
                if ai_service["model_id"] == request["model"]:
                    ai_service["lora_config"] = request["lora_config"]
                    break
            self.update_service_info({
                "ai_services": ai_services, 
            })
            return True
        except Exception as e:
            logger.error(f"Error updating LoRA: {str(e)}")
            return False
        
    def _build_chat_command(self, config: dict) -> list:
        """Build the chat command with common parameters."""
        model_path = config.get("model", None)
        if model_path is None:
            raise ValueError("Model path is required to start the service")
        
        hf_data = config.get("hf_data", None)
        model_name = config.get("model_name", None)
        model_family = self._get_model_family(model_name)
        template_path = self._get_model_template_path(model_family)
        best_practice_path = self._get_model_best_practice_path(model_family)
        projector = config.get("projector", None)
        context_length = config.get("context_length", 32768)
        backend = config.get("backend", "gguf")

        if backend == "gguf":
            command = [
                self.llama_server_path,
                "--model", str(model_path),
                "--pooling", "mean",
                "--no-webui",
                "--no-context-shift",
                "-fa", "on",
                "-ngl", "9999",
                "-c", str(context_length),
                "--embeddings",
                "--jinja",
            ]

            if projector is not None:
                if os.path.exists(projector):
                    command.extend(["--mmproj", str(projector)])
                else:
                    raise ValueError(f"Projector file not found: {projector}")
            
            if template_path is not None:
                if os.path.exists(template_path):
                    command.extend(["--chat-template-file", template_path])
                else:
                    raise ValueError(f"Template file not found: {template_path}")
            
            if best_practice_path is not None:
                if os.path.exists(best_practice_path):
                    with open(best_practice_path, "r") as f:
                        best_practice = json.load(f)
                        for key, value in best_practice.items():
                            command.extend([f"--{key}", str(value)])
                else:
                    raise ValueError(f"Best practices file not found: {best_practice_path}")
        elif backend == "mlx-lm":
            command = [
                "mlx-openai-server",
                "launch",
                "--model-path", str(model_path),
                "--model-type", "lm"
            ]
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return command

    def _build_embed_command(self, config: dict) -> list:
        """Build the embed command with common parameters."""
        model_path = config.get("model", None)
        if model_path is None:
            raise ValueError("Model path is required to start the service")
        
        command = [
            self.llama_server_path,
            "--model", str(model_path),
            "--embedding",
            "-fa", "on",
            "--pooling", "mean",
            "-ub", "4096",
            "-ngl", "9999"
        ]
        return command

    def _build_image_generation_command(self, config: dict) -> list:
        """Build the image-generation command with MLX Flux parameters and optional LoRA support."""

        model_path = config["model"]
        lora_config = config.get("lora_config", None)
        is_lora = config.get("is_lora", False)
        architecture = config.get("architecture", "flux-dev")
        lora_paths = []
        lora_scales = []

        if is_lora:
            for key, value in lora_config.items():
                lora_paths.append(value["path"])
                lora_scales.append(value["scale"])

        command = [
            "mlx-openai-server",
            "launch",
            "--model-path", str(model_path),
            "--config-name", architecture,
            "--model-type", "image-generation",
        ]
        
        # Validate LoRA parameters
        if lora_paths and lora_scales and len(lora_paths) != len(lora_scales):
            raise ValueError(f"LoRA paths count ({len(lora_paths)}) must match scales count ({len(lora_scales)})")
        
        # Add LoRA paths if provided
        if lora_paths:
            lora_path_str = ",".join(lora_paths)
            command.extend(["--lora-paths", lora_path_str])
        
        # Add LoRA scales if provided
        if lora_scales:
            lora_scale_str = ",".join(str(scale) for scale in lora_scales)
            command.extend(["--lora-scales", lora_scale_str])
        
        return command


    async def switch_model(self, target_model_id: str, context_length_override: Optional[int] = None) -> bool:
        """
        Switch to a different model that was registered during multi-model start.
        This will offload the currently active model and load the requested model.

        Args:
            target_hash (str): Hash of the model to switch to.
            service_start_timeout (int): Timeout for service startup in seconds.

        Returns:
            bool: True if model switch was successful, False otherwise.
        """
        service_info = self.get_service_info()
        ai_services = service_info.get("ai_services", [])
        active_service_index = 0
        target_service_index = 0
        active_ai_service = None
        target_ai_service = None
        
        for i, ai_service in enumerate(ai_services):
            if ai_service["active"]:
                active_service_index = i
                active_ai_service = ai_service

            if ai_service.get("model_id", None) == target_model_id:
                target_service_index = i
                target_ai_service = ai_service
        
        if target_ai_service is None:
            self.last_switch_error = f"Target model {target_model_id} not found"
            logger.error(self.last_switch_error)
            return False
        
        active_pid = active_ai_service.get("pid", None)
        if active_pid and psutil.pid_exists(active_pid):
            # Terminate current model process first
            await self._terminate_process_safely_async(active_pid, "EternalZoo AI Service", timeout=self.PROCESS_TERM_TIMEOUT)
            # Wait for memory to stabilize up to a short window
            logger.info("Waiting up to 3s for memory to stabilize after termination...")
            start_ts = time.time()
            last_avail = psutil.virtual_memory().available
            while time.time() - start_ts < 3.0:
                await asyncio.sleep(0.25)
                cur_avail = psutil.virtual_memory().available
                # Break early if available memory rises by >256MB indicating reclaim
                if cur_avail - last_avail > 256 * 1024 * 1024:
                    break
                last_avail = cur_avail
        else:
            logger.warning(f"Active model {active_ai_service.get('model_id', 'unknown')} not found")

        active_ai_service["active"] = False
        host = active_ai_service.get("host", "0.0.0.0")
        active_ai_service.pop("pid", None)
        active_ai_service.pop("host", None)
        active_ai_service.pop("port", None)
        ai_services[active_service_index] = active_ai_service
    
        running_ai_command = target_ai_service["running_ai_command"]
        if running_ai_command is None:
            self.last_switch_error = f"Target model {target_model_id} has no running AI command"
            logger.error(self.last_switch_error)
            return False

        # Pre-loading memory checks (RAM and optional VRAM)
        # Improved memory estimation
        effective_ctx_override = context_length_override if context_length_override is not None else self.switch_ctx_override
        model_memory_gb = self._estimate_model_ram_gb(target_ai_service, context_length_override=effective_ctx_override)

        available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        configured_context = target_ai_service.get("context_length", 32768)
        eff_ctx = effective_ctx_override if effective_ctx_override is not None else configured_context
        logger.info(
            f"Memory check for {target_model_id}: required={model_memory_gb:.2f}GB, available RAM={available_ram_gb:.2f}GB, "
            f"ctx_used={eff_ctx}, ctx_configured={configured_context}, ctx_override={effective_ctx_override}"
        )

        if available_ram_gb < model_memory_gb:
            self.last_switch_error = (
                f"Insufficient RAM for {target_model_id} - required {model_memory_gb:.2f}GB, "
                f"available {available_ram_gb:.2f}GB"
            )
            logger.error(self.last_switch_error)
            return False

        # Optional: Linux/NVIDIA VRAM check using pynvml if available
        if sys.platform.startswith("linux"):
            try:
                import pynvml  # type: ignore
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                available_vram_gb = info.free / (1024 ** 3)
                logger.info(f"VRAM check: available VRAM={available_vram_gb:.2f}GB")
                if available_vram_gb < model_memory_gb:
                    self.last_switch_error = (
                        f"Insufficient VRAM for {target_model_id} - required {model_memory_gb:.2f}GB, "
                        f"available VRAM {available_vram_gb:.2f}GB"
                    )
                    logger.error(self.last_switch_error)
                    return False
            except Exception as e:
                logger.warning(f"VRAM check skipped/unavailable: {e}")

        local_model_port = self._get_free_port()
        # avoid mutating stored command in service info
        launch_command = [*running_ai_command, "--port", str(local_model_port), "--host", host]
        logger.info(f"Switching to model: {target_model_id} with command: {' '.join(launch_command)}")
        with open(self.ai_log_file, 'w') as stderr_log:
            ai_process = await asyncio.create_subprocess_exec(
                *launch_command,
                stderr=stderr_log,
                preexec_fn=os.setsid
            )
            target_ai_service["pid"] = ai_process.pid
            target_ai_service["active"] = True
            target_ai_service["port"] = local_model_port
            target_ai_service["host"] = host
            ai_services[target_service_index] = target_ai_service

        with open(self.ai_service_file, 'wb') as f:
            msgpack.pack(ai_services, f)
            
        # wait for the service to be healthy (run sync check in a thread)
        # After spawn, observe memory stabilization until health or short window
        # Log RAM before health wait
        pre_health_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        logger.info(f"Available RAM before health wait: {pre_health_ram_gb:.2f}GB")
        is_healthy = await asyncio.to_thread(wait_for_health, local_model_port)
        if not is_healthy:
            await self._terminate_process_safely_async(ai_process.pid, "EternalZoo AI Service", timeout=self.PROCESS_TERM_TIMEOUT)
            self.last_switch_error = f"Health check failed for {target_model_id} on port {local_model_port}"
            logger.error(self.last_switch_error)
            return False
        post_health_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        logger.info(f"Available RAM after health success: {post_health_ram_gb:.2f}GB")
        
        with open(self.ai_service_file, 'wb') as f:
            msgpack.pack(ai_services, f)
        logger.info(f"AI service metadata written to {self.ai_service_file}")

        self.update_service_info({
            "ai_services": ai_services, 
        })

        # Clear last switch error on success
        self.last_switch_error = None
        # Clear transient context override
        self.switch_ctx_override = None

        return True
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get the list of available models.
        """
        service_info = self.get_service_info()
        ai_services = service_info.get("ai_services", [])
        return ai_services
        
    def get_models_by_task(self, tasks: List[str]) -> List[Dict[str, Any]]:
        """
        Get the list of models by task.
        """
        models = []
        service_info = self.get_service_info()
        ai_services = service_info.get("ai_services", [])
        for ai_service in ai_services:
            if ai_service["task"] in tasks:
                models.append(ai_service)
        return models