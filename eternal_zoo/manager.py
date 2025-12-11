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
from eternal_zoo.utils import wait_for_health, is_jetson, get_cuda_memory_jetson
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

        # Perform GPU memory cleanup to ensure clean state from previous runs
        self._perform_startup_gpu_cleanup()

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
                    # Memory validation before starting subprocess
                    # Use pre-calculated memory requirements from config (calculated in load_model_metadata)
                    memory_reqs = ai_service.get("memory_requirements", {})
                    model_memory_gb = memory_reqs.get("ram_gb", self._estimate_model_ram_gb(ai_service))
                    available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)

                    logger.info(f"Memory check for {config.get('model_name', 'unknown')}: required={model_memory_gb:.2f}GB, available RAM={available_ram_gb:.2f}GB")
                    if "memory_requirements" in ai_service:
                        mem_info = ai_service["memory_requirements"]
                        logger.info(f"Using pre-calculated memory: RAM={mem_info['ram_gb']:.2f}GB, GPU={mem_info['gpu_ram_gb']:.2f}GB, KV={mem_info['kv_cache_gb']:.2f}GB")

                    if available_ram_gb < model_memory_gb:
                        logger.error(f"Insufficient RAM to start model: need {model_memory_gb:.2f}GB, have {available_ram_gb:.2f}GB")
                        return False

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
        
        # attempt graceful shutdown first, fallback to force if needed
        ai_services = []
        ai_service_stop = False
        api_service_stop = False

        if self.ai_service_file.exists():
            with open(self.ai_service_file, 'rb') as f:
                ai_services = msgpack.unpack(f)
            for ai_service in ai_services:
                pid = ai_service.get("pid", None)
                if pid and psutil.pid_exists(pid):
                    ai_service_stop = self._terminate_process_safely(pid, "EternalZoo AI Service", force=False)

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
                api_service_stop = self._terminate_process_safely(pid, "EternalZoo API Service", force=False)
            
            if api_service_stop:
                os.remove(self.api_service_file)
                logger.info(f"API service metadata file removed: {self.api_service_file}")
            else:
                logger.warning("Failed to stop EternalZoo API Service")
        
        # Verify GPU memory cleanup if applicable
        if not self._verify_cuda_memory_released():
            logger.warning("GPU memory may not have been fully released after shutdown")
            # Attempt cleanup
            import asyncio
            asyncio.run(self._attempt_cuda_reset())

        return True

    def _get_platform_info(self) -> dict:
        """Detect platform and GPU information for cleanup strategy."""
        import platform
        system = platform.system().lower()

        gpu_info = self._detect_gpu_info()
        return {
            'os': system,
            'gpu_vendor': gpu_info['vendor'],
            'gpu_present': gpu_info['present'],
            'gpu_memory_mb': gpu_info['memory_mb']
        }

    def _detect_gpu_info(self) -> dict:
        """Detect GPU vendor and capabilities."""
        # Try NVIDIA first
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mb = info.total / (1024 * 1024)
            return {'vendor': 'nvidia', 'present': True, 'memory_mb': total_mb}
        except:
            pass

        # Try AMD/NVIDIA via nvidia-smi (works for some AMD cards too)
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                total_mb = int(result.stdout.strip().split('\n')[0])
                return {'vendor': 'nvidia', 'present': True, 'memory_mb': total_mb}
        except:
            pass

        # Mac OS detection
        import platform
        if platform.system() == 'Darwin':
            try:
                # Try to detect Apple Silicon vs Intel
                import subprocess
                result = subprocess.run(['system_profiler', 'SPHardwareDataType'],
                                      capture_output=True, text=True, timeout=5)
                if 'Apple M' in result.stdout or 'Apple Silicon' in result.stdout:
                    return {'vendor': 'apple_silicon', 'present': True, 'memory_mb': None}
                else:
                    return {'vendor': 'mac_intel', 'present': True, 'memory_mb': None}
            except:
                return {'vendor': 'mac_unknown', 'present': True, 'memory_mb': None}

        # Fallback: assume some GPU is present
        return {'vendor': 'unknown', 'present': True, 'memory_mb': None}

    def _get_gpu_memory_usage(self) -> str:
        """Get current GPU memory usage as a formatted string. Returns 'unavailable' if checks fail."""
        # Try modern nvidia-ml-py first (replacement for deprecated pynvml)
        try:
            import nvidia_ml_py as nvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            info = nvml.nvmlDeviceGetMemoryInfo(handle)
            used_mb = info.used / (1024 * 1024)
            total_mb = info.total / (1024 * 1024)
            return f"{used_mb:.1f}MB/{total_mb:.1f}MB"
        except Exception:
            pass

        # Fallback to deprecated pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_mb = info.used / (1024 * 1024)
            total_mb = info.total / (1024 * 1024)
            return f"{used_mb:.1f}MB/{total_mb:.1f}MB"
        except Exception:
            pass

        # Fallback to nvidia-smi (may not work on Jetson)
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and ',' in lines[0]:
                    parts = lines[0].split(', ')
                    if len(parts) == 2 and parts[0] != '[Not Supported]' and parts[1] != '[Not Supported]':
                        try:
                            used_mb, total_mb = map(float, parts)
                            return f"{used_mb:.1f}MB/{total_mb:.1f}MB"
                        except ValueError:
                            pass
        except Exception:
            pass

        # Final fallback: try tegrastats for Jetson devices
        try:
            import subprocess
            result = subprocess.run(['tegrastats', '--interval', '1', '--count', '1'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and 'GR3D' in result.stdout:
                # Parse tegrastats output for GPU memory (simplified)
                # This is a basic implementation - could be improved
                return "tegrastats-available"
        except Exception:
            pass

        return "unavailable"

    def _perform_startup_gpu_cleanup(self) -> None:
        """Perform GPU memory cleanup at startup to ensure clean state from previous runs."""
        try:
            platform_info = self._get_platform_info()
            logger.info(f"Performing startup GPU cleanup on {platform_info['os']} with {platform_info['gpu_vendor']} GPU")

            if platform_info['gpu_vendor'] == 'nvidia':
                self._cleanup_nvidia_gpu_memory()
            elif platform_info['os'] == 'darwin':
                self._cleanup_metal_gpu_memory()
            else:
                logger.info("No specific GPU cleanup needed for this platform")

        except Exception as e:
            logger.warning(f"Startup GPU cleanup failed: {e}")

    def _cleanup_nvidia_gpu_memory(self) -> None:
        """Attempt to clean up NVIDIA GPU memory state."""
        try:
            # Try to reset GPU memory state
            import subprocess
            # Use nvidia-smi to reset GPU memory if possible
            result = subprocess.run(['nvidia-smi', '--gpu-reset'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("NVIDIA GPU reset completed")
            else:
                logger.debug(f"NVIDIA GPU reset not available: {result.stderr}")

            # Check current GPU memory state
            gpu_mem = self._get_gpu_memory_usage()
            logger.info(f"GPU memory state after cleanup: {gpu_mem}")

        except Exception as e:
            logger.debug(f"NVIDIA GPU cleanup failed: {e}")

    def _cleanup_metal_gpu_memory(self) -> None:
        """Attempt to clean up Metal GPU memory state on macOS."""
        try:
            # On macOS, we can't directly reset GPU memory, but we can log the current state
            logger.info("Metal GPU cleanup - monitoring memory state")
            # The async cleanup monitoring will handle any issues during actual loading
        except Exception as e:
            logger.debug(f"Metal GPU cleanup failed: {e}")

    async def _monitor_memory_cleanup_async(self, platform_info: dict, initial_ram_gb: float, initial_gpu_mb: float) -> bool:
        """Perform GPU memory cleanup at startup to ensure clean state from previous runs."""
        try:
            platform_info = self._get_platform_info()
            logger.info(f"Performing startup GPU cleanup on {platform_info['os']} with {platform_info['gpu_vendor']} GPU")

            if platform_info['gpu_vendor'] == 'nvidia':
                self._cleanup_nvidia_gpu_memory()
            elif platform_info['os'] == 'darwin':
                self._cleanup_metal_gpu_memory()
            else:
                logger.info("No specific GPU cleanup needed for this platform")

        except Exception as e:
            logger.warning(f"Startup GPU cleanup failed: {e}")

    def _cleanup_nvidia_gpu_memory(self) -> None:
        """Attempt to clean up NVIDIA GPU memory state."""
        try:
            # Try to reset GPU memory state
            import subprocess
            # Use nvidia-smi to reset GPU memory if possible
            result = subprocess.run(['nvidia-smi', '--gpu-reset'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("NVIDIA GPU reset completed")
            else:
                logger.debug(f"NVIDIA GPU reset not available: {result.stderr}")

            # Check current GPU memory state
            gpu_mem = self._get_gpu_memory_usage()
            logger.info(f"GPU memory state after cleanup: {gpu_mem}")

        except Exception as e:
            logger.debug(f"NVIDIA GPU cleanup failed: {e}")

    def _cleanup_metal_gpu_memory(self) -> None:
        """Attempt to clean up Metal GPU memory state on macOS."""
        try:
            # On macOS, we can't directly reset GPU memory, but we can log the current state
            logger.info("Metal GPU cleanup - monitoring memory state")
            # The async cleanup monitoring will handle any issues during actual loading
        except Exception as e:
            logger.debug(f"Metal GPU cleanup failed: {e}")

    async def _monitor_memory_cleanup_async(self, platform_info: dict, initial_ram_gb: float, initial_gpu_mb: float) -> bool:
        """Monitor memory cleanup completion using platform-specific callbacks."""
        completion_event = asyncio.Event()
        monitoring_tasks = []

        try:
            # Create platform-specific monitoring tasks
            if platform_info['gpu_vendor'] == 'nvidia':
                monitoring_tasks.append(self._monitor_cuda_cleanup_async(completion_event, initial_gpu_mb))
            elif platform_info['os'] == 'darwin':
                monitoring_tasks.append(self._monitor_metal_cleanup_async(completion_event))
            else:
                # Generic RAM-only monitoring
                monitoring_tasks.append(self._monitor_ram_cleanup_async(completion_event, initial_ram_gb))

            # Always monitor RAM reclamation
            monitoring_tasks.append(self._monitor_ram_cleanup_async(completion_event, initial_ram_gb))

            # Start all monitoring tasks
            started_tasks = []
            for coro in monitoring_tasks:
                task = asyncio.create_task(coro)
                started_tasks.append(task)

            # Wait for completion with platform-specific timeout
            timeout = self._get_cleanup_timeout(platform_info)
            try:
                await asyncio.wait_for(completion_event.wait(), timeout=timeout)
                logger.info("✅ Memory cleanup completed successfully")
                return True
            except asyncio.TimeoutError:
                logger.warning(f"⚠️ Memory cleanup monitoring timed out after {timeout}s, proceeding anyway")
                return True  # Don't block model loading

        except Exception as e:
            logger.error(f"Memory cleanup monitoring failed: {e}, proceeding anyway")
            return True
        finally:
            # Cancel any remaining monitoring tasks
            for task in started_tasks:
                if not task.done():
                    task.cancel()

    async def _monitor_cuda_cleanup_async(self, completion_event: asyncio.Event, initial_gpu_mb: float) -> None:
        """Monitor CUDA context cleanup via nvidia-ml-py or pynvml."""
        if initial_gpu_mb <= 0:
            return

        try:
            # Try modern nvidia-ml-py first
            try:
                import nvidia_ml_py as nvml
            except ImportError:
                import pynvml as nvml

            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)

            while not completion_event.is_set():
                info = nvml.nvmlDeviceGetMemoryInfo(handle)
                current_used_mb = info.used / (1024 * 1024)
                freed_mb = initial_gpu_mb - current_used_mb

                # Consider cleanup complete if we've freed 80% of initial GPU memory
                if freed_mb > (initial_gpu_mb * 0.8):
                    logger.info(f"CUDA cleanup detected: {freed_mb:.1f}MB GPU memory freed")
                    completion_event.set()
                    return

                await asyncio.sleep(0.2)  # Check every 200ms

        except Exception as e:
            logger.debug(f"CUDA cleanup monitoring failed: {e}")

    async def _monitor_metal_cleanup_async(self, completion_event: asyncio.Event) -> None:
        """Monitor Metal framework cleanup on macOS."""
        try:
            # Metal cleanup is typically much faster than CUDA
            # Wait a bit then assume cleanup is complete
            await asyncio.sleep(0.5)
            logger.info("Metal cleanup assumed complete (fast GPU context)")
            completion_event.set()
        except Exception as e:
            logger.debug(f"Metal cleanup monitoring failed: {e}")

    async def _monitor_ram_cleanup_async(self, completion_event: asyncio.Event, initial_ram_gb: float) -> None:
        """Monitor RAM reclamation as a fallback indicator."""
        try:
            while not completion_event.is_set():
                current_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
                reclaimed_gb = initial_ram_gb - current_ram_gb

                # Consider cleanup complete if we've reclaimed significant RAM
                if reclaimed_gb > 0.5:  # 512MB threshold
                    logger.info(f"RAM cleanup detected: {reclaimed_gb:.2f}GB reclaimed")
                    completion_event.set()
                    return

                await asyncio.sleep(0.5)  # Check every 500ms

        except Exception as e:
            logger.debug(f"RAM cleanup monitoring failed: {e}")

    def _get_cleanup_timeout(self, platform_info: dict) -> float:
        """Get platform-specific cleanup timeout."""
        timeouts = {
            'nvidia': 15.0,    # CUDA cleanup can take time
            'apple_silicon': 2.0,  # Very fast
            'mac_intel': 3.0,  # Metal is reasonably fast
            'cpu_only': 1.0    # Minimal for CPU-only
        }
        return timeouts.get(platform_info.get('gpu_vendor', 'cpu_only'), 5.0)

    async def _attempt_cuda_reset(self) -> bool:
        """Attempt CUDA device reset to force memory cleanup. Returns True if successful."""
        logger.info("Attempting CUDA device reset for memory cleanup...")

        # Jetson-specific handling
        if is_jetson():
            logger.warning("Jetson detected - CUDA memory may persist after process termination. Attempting cleanup.")
            # For Jetson, try driver reload if we have sudo
            try:
                import subprocess
                # Check if we have sudo
                sudo_check = subprocess.run(['sudo', '-n', 'true'], capture_output=True, timeout=5)
                if sudo_check.returncode == 0:
                    logger.info("Attempting NVIDIA driver reload on Jetson...")
                    # Unload and reload NVIDIA driver
                    unload = await asyncio.create_subprocess_exec(
                        'sudo', 'rmmod', 'nvidia',
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL
                    )
                    await asyncio.wait_for(unload.wait(), timeout=10.0)
                    await asyncio.sleep(0.5)
                    load = await asyncio.create_subprocess_exec(
                        'sudo', 'nvidia-smi', '-a',  # This should reload the driver
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL
                    )
                    await asyncio.wait_for(load.wait(), timeout=10.0)
                    logger.info("NVIDIA driver reload completed on Jetson")
                    await asyncio.sleep(1.0)  # Allow time for reload
                    return True
                else:
                    logger.warning("No sudo access for driver reload on Jetson")
            except (asyncio.TimeoutError, subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.warning(f"Jetson driver reload failed: {e}")
            return False  # Don't try other methods on Jetson

        try:
            # Try modern nvidia-ml-py first for device reset
            try:
                import nvidia_ml_py as nvml
            except ImportError:
                import pynvml as nvml

            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)

            # Try to reset GPU by toggling persistence mode
            try:
                current_mode = nvml.nvmlDeviceGetPersistenceMode(handle)
                # Toggle persistence mode to force cleanup
                nvml.nvmlDeviceSetPersistenceMode(handle, 0)  # Disable
                await asyncio.sleep(0.1)
                nvml.nvmlDeviceSetPersistenceMode(handle, 1)  # Re-enable
                logger.info("CUDA device reset completed")
                return True
            except Exception as e:
                logger.warning(f"NVML device reset failed: {e}")

        except Exception:
            logger.debug("NVML not available for CUDA reset")

        # Fallback: try nvidia-smi gpu reset
        try:
            logger.info("Attempting CUDA reset via nvidia-smi...")
            result = await asyncio.create_subprocess_exec(
                'nvidia-smi', '--gpu-reset',
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await asyncio.wait_for(result.wait(), timeout=10.0)
            if result.returncode == 0:
                logger.info("CUDA device reset via nvidia-smi completed")
                await asyncio.sleep(0.5)  # Allow time for reset to take effect
                return True
            else:
                logger.warning(f"nvidia-smi gpu reset failed with return code: {result.returncode}")
        except (asyncio.TimeoutError, FileNotFoundError) as e:
            logger.warning(f"nvidia-smi reset unavailable: {e}")

        logger.warning("All CUDA reset methods failed")
        return False

    def _verify_cuda_memory_released(self) -> bool:
        """Verify CUDA memory cleanup after shutdown. Returns True if memory appears clean or check unavailable."""
        # On Jetson, memory checks are unreliable due to unified memory and persistent allocations
        # Always attempt cleanup on Jetson
        if is_jetson():
            logger.info("Jetson detected - CUDA memory verification unreliable, will attempt cleanup")
            return False  # Force cleanup attempt

        try:
            # Try pynvml first
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_mb = info.used / (1024 * 1024)
            logger.info(f"GPU memory after shutdown: {used_mb:.1f}MB used out of {info.total / (1024*1024):.1f}MB")
            return used_mb < 500  # Consider clean if less than 500MB used
        except ImportError:
            # Fallback to nvidia-smi
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines:
                        used_mb, total_mb = map(int, lines[0].split(', '))
                        logger.info(f"GPU memory after shutdown: {used_mb}MB used out of {total_mb}MB")
                        return used_mb < 500
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
                pass
        except Exception as e:
            logger.debug(f"CUDA memory verification failed: {e}")

        # If all checks fail, assume it's clean to avoid false positives
        logger.debug("CUDA memory verification unavailable - assuming clean")
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

    async def _terminate_process_safely_async(self, pid: int, process_name: str, timeout: int = 15, use_process_group: bool = True, force: bool = False) -> bool:
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
            if not force:
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
            else:
                logger.info(f"Force mode enabled - skipping graceful termination for {process_name} (PID: {pid})")

            # Phase 2: Force termination if still running
            if psutil.pid_exists(pid):
                logger.warning(f"Force killing {process_name} (PID: {pid})")

                # Attempt CUDA reset before force kill to ensure GPU memory cleanup
                await self._attempt_cuda_reset()

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
    
    def _calculate_model_memory_requirements(self, ai_service: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate and return detailed memory requirements for a model.
        Returns dict with: ram_gb, gpu_ram_gb, kv_cache_gb, weights_gb

        Uses GGUF metadata when available, falls back to file-size heuristics.
        """
        backend = ai_service.get("backend", "gguf")
        model_path = ai_service.get("model")

        # Get context length from config or default
        context_length = ai_service.get("context_length", DEFAULT_CONFIG.model.DEFAULT_CONTEXT_LENGTH)

        # Get GPU layers from config (default to full offload for GGUF)
        gpu_layers = ai_service.get("gpu_layers", 9999 if backend == "gguf" else 0)

        # Initialize with conservative defaults
        memory_reqs = {
            "ram_gb": 12.0,
            "gpu_ram_gb": 0.0,
            "kv_cache_gb": 0.0,
            "weights_gb": 0.0
        }

        if backend == "gguf" and model_path and os.path.exists(model_path):
            try:
                import gguf
                try:
                    reader = gguf.GGUFReader(model_path)
                except Exception:
                    with open(model_path, "rb") as f:
                        reader = gguf.GGUFReader(f)

                # Extract metadata using gguf-mem-info.py approach
                arch = self._get_scalar_field(reader, "general.architecture")
                if arch is None:
                    raise RuntimeError("general.architecture not found in GGUF metadata")

                arch = str(arch)
                prefix = arch

                def fp(suffix: str) -> Optional[int]:
                    v = self._get_scalar_field(reader, f"{prefix}.{suffix}")
                    return int(v) if v is not None else None

                n_layer = fp("block_count")
                n_embd = fp("embedding_length")
                n_head = fp("attention.head_count")
                n_head_kv = fp("attention.head_count_kv") or n_head

                if n_layer is None or n_embd is None:
                    raise RuntimeError("Missing n_layer or n_embd in GGUF metadata")

                # KV cache calculation (same as gguf-mem-info.py)
                kv_bytes_per_elem = 2  # Default f16
                kv_dtype = ai_service.get("kv_dtype", "f16")
                if isinstance(kv_dtype, str):
                    kv_dtype_l = kv_dtype.lower()
                    if kv_dtype_l in ("f16", "bf16"):
                        kv_bytes_per_elem = 2
                    elif kv_dtype_l in ("f32", "float32"):
                        kv_bytes_per_elem = 4
                    elif kv_dtype_l in ("q8", "q8_kv", "int8"):
                        kv_bytes_per_elem = 1

                kv_bytes = 2 * n_layer * context_length * n_embd * kv_bytes_per_elem
                kv_cache_gb = kv_bytes / (1024 ** 3)

                # File size for weights
                file_size_gb = os.path.getsize(model_path) / (1024 ** 3)

                # GPU/CPU split calculation (similar to gguf-mem-info.py)
                gpu_layers_clamped = max(0, min(gpu_layers, n_layer))
                if n_layer and gpu_layers_clamped > 0:
                    frac = float(gpu_layers_clamped) / float(n_layer)
                    gpu_weight_bytes = int(os.path.getsize(model_path) * frac * 0.9)  # ~90% weights are per-layer
                    cpu_weight_bytes = os.path.getsize(model_path) - gpu_weight_bytes
                else:
                    gpu_weight_bytes = 0
                    cpu_weight_bytes = os.path.getsize(model_path)

                gpu_weights_gb = gpu_weight_bytes / (1024 ** 3)
                cpu_weights_gb = cpu_weight_bytes / (1024 ** 3)

                # Total RAM: CPU weights + KV cache + small buffer
                weight_residency_buffer_gb = min(2.0, file_size_gb * 0.05)
                ram_gb = cpu_weights_gb + kv_cache_gb + weight_residency_buffer_gb + 2.0  # headroom

                # GPU RAM: GPU weights + KV cache (KV usually on GPU when offloading)
                gpu_ram_gb = gpu_weights_gb + kv_cache_gb

                memory_reqs = {
                    "ram_gb": max(ram_gb, 8.0),
                    "gpu_ram_gb": gpu_ram_gb,
                    "kv_cache_gb": kv_cache_gb,
                    "weights_gb": file_size_gb
                }

                logger.info(
                    f"Calculated memory for {model_path}: RAM={memory_reqs['ram_gb']:.2f}GB, "
                    f"GPU={memory_reqs['gpu_ram_gb']:.2f}GB, KV={kv_cache_gb:.2f}GB, "
                    f"ctx={context_length}, gpu_layers={gpu_layers_clamped}"
                )

            except Exception as e:
                logger.warning(f"GGUF memory calculation failed for {model_path}: {e}, using file-size fallback")
                # File-size fallback
                file_size_gb = os.path.getsize(model_path) / (1024 ** 3)
                fallback_ram_gb = file_size_gb * 1.25 + 4.0
                memory_reqs = {
                    "ram_gb": max(fallback_ram_gb, 8.0),
                    "gpu_ram_gb": 0.0,
                    "kv_cache_gb": 0.0,
                    "weights_gb": file_size_gb
                }

        elif backend in ["mlx-lm", "image-generation"]:
            # For non-GGUF backends, use file-size based estimation if model path exists
            if model_path and os.path.exists(model_path):
                file_size_gb = os.path.getsize(model_path) / (1024 ** 3)
                # More conservative estimate for non-GGUF models
                memory_reqs = {
                    "ram_gb": max(file_size_gb * 1.5 + 4.0, 8.0),
                    "gpu_ram_gb": 0.0,  # Assume CPU-only for now
                    "kv_cache_gb": 0.0,
                    "weights_gb": file_size_gb
                }
            else:
                logger.warning(f"No model path for {backend} backend, using default memory estimate")

        return memory_reqs

    def _get_scalar_field(self, reader, key: str) -> Optional[object]:
        """Extract scalar field from GGUF reader (similar to gguf-mem-info.py)."""
        field = reader.get_field(key)
        if field is None:
            return None
        val = field.contents()
        # strings and scalars are fine
        if isinstance(val, (str, bytes, int, float)):
            return val
        # Handle arrays/lists - for attention.head_count_kv, use max value
        try:
            if hasattr(val, "__len__") and len(val) > 0:
                if len(val) == 1:
                    return val[0]
                elif key.endswith("attention.head_count_kv"):
                    # For KV heads, use the maximum value (some layers may have 0)
                    return max(val)
                else:
                    # For other arrays, try to use the first element
                    return val[0]
        except TypeError:
            pass
        return val

    def _calculate_model_memory_requirements(self, ai_service: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate and return detailed memory requirements for a model.
        Returns dict with: ram_gb, gpu_ram_gb, kv_cache_gb, weights_gb

        Uses GGUF metadata when available, falls back to file-size heuristics.
        """
        backend = ai_service.get("backend", "gguf")
        model_path = ai_service.get("model")

        # Get context length from config or default
        context_length = ai_service.get("context_length", DEFAULT_CONFIG.model.DEFAULT_CONTEXT_LENGTH)

        # Get GPU layers from config (default to full offload for GGUF)
        gpu_layers = ai_service.get("gpu_layers", 9999 if backend == "gguf" else 0)

        # Initialize with conservative defaults
        memory_reqs = {
            "ram_gb": 12.0,
            "gpu_ram_gb": 0.0,
            "kv_cache_gb": 0.0,
            "weights_gb": 0.0
        }

        if backend == "gguf" and model_path and os.path.exists(model_path):
            try:
                import gguf
                try:
                    reader = gguf.GGUFReader(model_path)
                except Exception:
                    with open(model_path, "rb") as f:
                        reader = gguf.GGUFReader(f)

                # Extract metadata using gguf-mem-info.py approach
                arch = self._get_scalar_field(reader, "general.architecture")
                if arch is None:
                    raise RuntimeError("general.architecture not found in GGUF metadata")

                arch = str(arch)
                prefix = arch

                def fp(suffix: str) -> Optional[int]:
                    v = self._get_scalar_field(reader, f"{prefix}.{suffix}")
                    return int(v) if v is not None else None

                n_layer = fp("block_count")
                n_embd = fp("embedding_length")
                n_head = fp("attention.head_count")
                n_head_kv = fp("attention.head_count_kv") or n_head

                if n_layer is None or n_embd is None:
                    raise RuntimeError("Missing n_layer or n_embd in GGUF metadata")

                # KV cache calculation (same as gguf-mem-info.py)
                kv_bytes_per_elem = 2  # Default f16
                kv_dtype = ai_service.get("kv_dtype", "f16")
                if isinstance(kv_dtype, str):
                    kv_dtype_l = kv_dtype.lower()
                    if kv_dtype_l in ("f16", "bf16"):
                        kv_bytes_per_elem = 2
                    elif kv_dtype_l in ("f32", "float32"):
                        kv_bytes_per_elem = 4
                    elif kv_dtype_l in ("q8", "q8_kv", "int8"):
                        kv_bytes_per_elem = 1

                kv_bytes = 2 * n_layer * context_length * n_embd * kv_bytes_per_elem
                kv_cache_gb = kv_bytes / (1024 ** 3)

                # File size for weights
                file_size_gb = os.path.getsize(model_path) / (1024 ** 3)

                # GPU/CPU split calculation (similar to gguf-mem-info.py)
                gpu_layers_clamped = max(0, min(gpu_layers, n_layer))
                if n_layer and gpu_layers_clamped > 0:
                    frac = float(gpu_layers_clamped) / float(n_layer)
                    gpu_weight_bytes = int(os.path.getsize(model_path) * frac * 0.9)  # ~90% weights are per-layer
                    cpu_weight_bytes = os.path.getsize(model_path) - gpu_weight_bytes
                else:
                    gpu_weight_bytes = 0
                    cpu_weight_bytes = os.path.getsize(model_path)

                gpu_weights_gb = gpu_weight_bytes / (1024 ** 3)
                cpu_weights_gb = cpu_weight_bytes / (1024 ** 3)

                # Total RAM: CPU weights + KV cache + small buffer
                weight_residency_buffer_gb = min(2.0, file_size_gb * 0.05)
                ram_gb = cpu_weights_gb + kv_cache_gb + weight_residency_buffer_gb + 2.0  # headroom

                # GPU RAM: GPU weights + KV cache (KV usually on GPU when offloading)
                gpu_ram_gb = gpu_weights_gb + kv_cache_gb

                memory_reqs = {
                    "ram_gb": max(ram_gb, 8.0),
                    "gpu_ram_gb": gpu_ram_gb,
                    "kv_cache_gb": kv_cache_gb,
                    "weights_gb": file_size_gb
                }

                logger.info(
                    f"Calculated memory for {model_path}: RAM={memory_reqs['ram_gb']:.2f}GB, "
                    f"GPU={memory_reqs['gpu_ram_gb']:.2f}GB, KV={kv_cache_gb:.2f}GB, "
                    f"ctx={context_length}, gpu_layers={gpu_layers_clamped}"
                )

            except Exception as e:
                logger.warning(f"GGUF memory calculation failed for {model_path}: {e}, using file-size fallback")
                # File-size fallback
                file_size_gb = os.path.getsize(model_path) / (1024 ** 3)
                fallback_ram_gb = file_size_gb * 1.25 + 4.0
                memory_reqs = {
                    "ram_gb": max(fallback_ram_gb, 8.0),
                    "gpu_ram_gb": 0.0,
                    "kv_cache_gb": 0.0,
                    "weights_gb": file_size_gb
                }

        elif backend in ["mlx-lm", "image-generation"]:
            # For non-GGUF backends, use file-size based estimation if model path exists
            if model_path and os.path.exists(model_path):
                file_size_gb = os.path.getsize(model_path) / (1024 ** 3)
                # More conservative estimate for non-GGUF models
                memory_reqs = {
                    "ram_gb": max(file_size_gb * 1.5 + 4.0, 8.0),
                    "gpu_ram_gb": 0.0,  # Assume CPU-only for now
                    "kv_cache_gb": 0.0,
                    "weights_gb": file_size_gb
                }
            else:
                logger.warning(f"No model path for {backend} backend, using default memory estimate")

        return memory_reqs

    def _estimate_model_ram_gb(self, ai_service: Dict[str, Any], context_length_override: Optional[int] = None) -> float:
        """Estimate RAM needed to load a model.
        Priority:
        1) Use stored 'memory_requirements' if available
        2) Use explicit 'estimated_ram_gb' if provided in service config
        3) For GGUF models, base on model file size with overhead multiplier
        4) Fallback default
        """
        # First priority: use pre-calculated memory requirements
        if "memory_requirements" in ai_service and ai_service["memory_requirements"]:
            stored_ram = ai_service["memory_requirements"].get("ram_gb")
            if stored_ram is not None:
                # If context override provided, use stored values as-is (ignore context difference for performance)
                if context_length_override is not None:
                    configured_ctx = ai_service.get("context_length", DEFAULT_CONFIG.model.DEFAULT_CONTEXT_LENGTH)
                    if context_length_override != configured_ctx:
                        logger.info(f"Context override provided ({context_length_override} vs configured {configured_ctx}), "
                                  f"using stored memory estimate {stored_ram:.2f}GB (may be conservative)")
                return stored_ram

        # Fallback to original estimation logic
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

                # Use the same simple approach as gguf-mem-info.py
                arch = self._get_scalar_field(reader, "general.architecture")
                if arch is None:
                    raise RuntimeError("general.architecture not found in GGUF metadata")

                arch = str(arch)
                prefix = arch  # e.g. 'llama', 'gemma', 'qwen3moe', etc.

                logger.info(f"[GGUF DEBUG] Architecture detected: {arch}, using prefix: {prefix}")

                def fp(suffix: str) -> Optional[int]:
                    key = f"{prefix}.{suffix}"
                    v = self._get_scalar_field(reader, key)
                    logger.debug(f"[GGUF DEBUG] Tried key '{key}' -> {v}")
                    return int(v) if v is not None else None

                n_layer = fp("block_count")
                n_embd = fp("embedding_length")
                n_head = fp("attention.head_count")
                n_head_kv = fp("attention.head_count_kv") or n_head

                logger.info(f"[GGUF DEBUG] Extracted metadata: n_layer={n_layer}, n_embd={n_embd}, n_head={n_head}, n_head_kv={n_head_kv}")

                if n_layer is None or n_embd is None:
                    raise RuntimeError(f"GGUF metadata missing for n_layer ({n_layer}) or n_embd ({n_embd}) - architecture: {arch}")

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

                logger.info(f"[GGUF DEBUG] Using bytes_per_element={bytes_per_element} for KV cache")
                kv_cache_bytes = 2 * n_layer * int(context_length) * n_embd * bytes_per_element
                kv_cache_gb = kv_cache_bytes / (1024 ** 3)

                # Weight residency: mostly mmapped, but budget a small residency buffer
                file_size_gb = os.path.getsize(model_path) / (1024 ** 3)
                weight_residency_buffer_gb = min(2.0, file_size_gb * 0.05)

                logger.info(
                    f"[GGUF DEBUG] Calculation: layers={n_layer}, emb={n_embd}, ctx={context_length}, "
                    f"file_size={file_size_gb:.2f}GB, kv_cache={kv_cache_gb:.2f}GB, weight_buffer={weight_residency_buffer_gb:.2f}GB"
                )

                estimate_gb = kv_cache_gb + weight_residency_buffer_gb + 2.0  # extra headroom
                logger.info(
                    f"GGUF estimate using metadata: arch={arch}, layers={n_layer}, emb={n_embd}, ctx={context_length}, "
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
        
        # Log memory state before termination
        pre_term_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        gpu_memory_before = self._get_gpu_memory_usage()
        logger.info(f"Memory before model termination: RAM={pre_term_ram_gb:.2f}GB, GPU={gpu_memory_before}")

        active_pid = active_ai_service.get("pid", None)
        if active_pid and psutil.pid_exists(active_pid):
            # Terminate current model process first
            logger.info(f"Terminating active model process (PID: {active_pid})")
            await self._terminate_process_safely_async(active_pid, "EternalZoo AI Service", timeout=self.PROCESS_TERM_TIMEOUT)

            # Use async callback-based memory cleanup monitoring
            platform_info = self._get_platform_info()
            logger.info(f"Using {platform_info['gpu_vendor']} cleanup monitoring on {platform_info['os']}")

            # Monitor cleanup completion with platform-specific callbacks
            cleanup_success = await self._monitor_memory_cleanup_async(
                platform_info,
                pre_term_ram_gb,
                float(gpu_memory_before.split('/')[0]) if '/' in gpu_memory_before else 0
            )

            # Log final memory state
            post_term_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
            gpu_memory_after_term = self._get_gpu_memory_usage()
            ram_reclaimed = post_term_ram_gb - pre_term_ram_gb
            logger.info(f"Memory after model termination: RAM={post_term_ram_gb:.2f}GB (+{ram_reclaimed:.2f}GB), GPU={gpu_memory_after_term}")
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

        # Log detailed memory breakdown
        memory_reqs = target_ai_service.get("memory_requirements", {})
        ram_needed = memory_reqs.get("ram_gb", model_memory_gb)
        gpu_needed = memory_reqs.get("gpu_ram_gb", 0.0)
        kv_cache = memory_reqs.get("kv_cache_gb", 0.0)
        weights = memory_reqs.get("weights_gb", 0.0)

        logger.info(
            f"🔄 MODEL SWITCH MEMORY CHECK for {target_model_id}:"
        )
        logger.info(
            f"   Required: RAM={ram_needed:.2f}GB, GPU={gpu_needed:.2f}GB | "
            f"Available: RAM={available_ram_gb:.2f}GB"
        )
        logger.info(
            f"   Breakdown: Weights={weights:.2f}GB, KV Cache={kv_cache:.2f}GB, Context={eff_ctx}"
        )
        logger.info(
            f"   Context: configured={configured_context}, override={effective_ctx_override}, used={eff_ctx}"
        )

        if available_ram_gb < model_memory_gb:
            self.last_switch_error = (
                f"Insufficient RAM for {target_model_id} - required {model_memory_gb:.2f}GB, "
                f"available {available_ram_gb:.2f}GB"
            )
            logger.error(self.last_switch_error)
            return False

        # Optional: GPU VRAM check using modern libraries
        if self._get_platform_info()['gpu_present']:
            try:
                # Try modern nvidia-ml-py first
                try:
                    import nvidia_ml_py as nvml
                except ImportError:
                    import pynvml as nvml

                nvml.nvmlInit()
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                info = nvml.nvmlDeviceGetMemoryInfo(handle)
                available_vram_gb = info.free / (1024 ** 3)

                # Use stored GPU memory requirements if available, otherwise use RAM estimate
                gpu_memory_required = target_ai_service.get("memory_requirements", {}).get("gpu_ram_gb", model_memory_gb)
                logger.info(f"🔄 MODEL SWITCH GPU CHECK for {target_model_id}: required={gpu_memory_required:.2f}GB, available VRAM={available_vram_gb:.2f}GB")

                if available_vram_gb < gpu_memory_required:
                    self.last_switch_error = (
                        f"Insufficient VRAM for {target_model_id} - required {gpu_memory_required:.2f}GB, "
                        f"available VRAM {available_vram_gb:.2f}GB"
                    )
                    logger.error(f"❌ {self.last_switch_error}")
                    return False

            except Exception as e:
                logger.warning(f"VRAM check skipped/unavailable: {e}")

        # Log memory state before starting new model
        pre_launch_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        gpu_memory_pre_launch = self._get_gpu_memory_usage()
        logger.info(f"Memory before launching new model: RAM={pre_launch_ram_gb:.2f}GB, GPU={gpu_memory_pre_launch}")

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

        logger.info(f"✅ MODEL SWITCH SUCCESSFUL: {target_model_id} is now active")
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