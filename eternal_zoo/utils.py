import os
import shutil
import hashlib
import subprocess
import shutil
import tempfile
import subprocess
import asyncio
import requests
import time
from typing import List
from loguru import logger
from pathlib import Path


def find_gguf_files(directory):
    """Find all .gguf files in the given directory with proper error handling"""
    try:
        if not os.path.exists(directory):
            return []
        if not os.path.isdir(directory):
            return []
        
        gguf_files = [
            f for f in os.listdir(directory) 
            if f.lower().endswith(".gguf") and os.path.isfile(os.path.join(directory, f))
        ]
        return sorted(gguf_files)  # Sort for consistent selection
    except (OSError, PermissionError) as e:
        logger.warning(f"Error accessing directory {directory}: {e}")
        return []


# Add file logger for extract_zip and related operations
logger.add("utils.log", rotation="10 MB", retention="10 days", encoding="utf-8")

def compress_folder(model_folder: str, zip_chunk_size: int = 128, threads: int = 1) -> str:
    """
    Compress a folder into split parts using tar, pigz, and split.
    """
    temp_dir = tempfile.mkdtemp()
    output_prefix = os.path.join(temp_dir, os.path.basename(model_folder) + ".zip.part-")
    tar_command = (
        f"{os.environ['TAR_COMMAND']} -cf - '{model_folder}' | "
        f"{os.environ['PIGZ_COMMAND']} --best -p {threads} | "
        f"split -b {zip_chunk_size}M - '{output_prefix}'"
    )
    try:
        subprocess.run(tar_command, shell=True, check=True)
        logger.info(f"{tar_command} completed successfully")
        return temp_dir
    except subprocess.CalledProcessError as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Compression failed: {e}")


def run_with_retries(cmd: str, max_retries: int = 3, delay: int = 2):
    """
    Run a shell command with retries. Raise error if all attempts fail.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[extract_zip] Attempt {attempt}/{max_retries}: {cmd}")
            subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            logger.info(f"[extract_zip] Command succeeded: {cmd}")
            return
        except subprocess.CalledProcessError as e:
            logger.error(f"[extract_zip] Command failed (attempt {attempt}): {e}")
            if attempt < max_retries:
                logger.warning(f"[extract_zip] Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"[extract_zip] All {max_retries} attempts failed for: {cmd}")
                raise


def extract_zip(paths: List[Path]):
    # Use the absolute path only once.
    target_abs = Path.cwd().absolute()
    target_dir = f"'{target_abs}'"
    logger.info(f"📦 Extracting files to: {target_dir}")

    # Check disk space before extraction
    total_parts_size = sum(p.stat().st_size for p in paths)
    # Estimate required space: parts + tar + extracted files (safety factor 2.5)
    required_bytes = int(total_parts_size * 2.5)
    usage = shutil.disk_usage(target_abs)
    free = usage.free
    logger.info(f"Disk space check: required={required_bytes/1024/1024/1024:.2f} GB, available={free/1024/1024/1024:.2f} GB")
    if free < required_bytes:
        logger.error(f"Not enough disk space: required {required_bytes/1024/1024/1024:.2f} GB, available {free/1024/1024/1024:.2f} GB")
        raise RuntimeError(f"Not enough disk space: required {required_bytes/1024/1024/1024:.2f} GB, available {free/1024/1024/1024:.2f} GB")

    # Get absolute paths for required commands.
    cat_path = os.environ.get("CAT_COMMAND")
    pigz_cmd = os.environ.get("PIGZ_COMMAND")
    tar_cmd = os.environ.get("TAR_COMMAND")
    if not (cat_path and pigz_cmd and tar_cmd):
        logger.error("Required commands (cat, TAR_COMMAND, PIGZ_COMMAND) not found.")
        raise RuntimeError("Required commands (cat, TAR_COMMAND, PIGZ_COMMAND) not found.")

    # Sort paths by their string representation.
    sorted_paths = sorted(paths, key=lambda p: str(p))
    # Quote each path after converting to its absolute path.
    paths_str = " ".join(f"'{p.absolute()}'" for p in sorted_paths)
    logger.info(f"🗂️ Extracting files: {paths_str}")

    cpus = os.cpu_count() or 1

    # Create temporary files for each step
    temp_gz = Path(tempfile.mktemp(suffix=".gz"))
    temp_tar = Path(tempfile.mktemp(suffix=".tar"))

    try:
        # Step 1: Concatenate all parts into a single gzipped file
        cat_command = f"{cat_path} {paths_str} > '{temp_gz}'"
        run_with_retries(cat_command)
        logger.info(f"✅ [extract_zip] Step 1 completed: {temp_gz}")

        # Step 2: Decompress the gzipped file to a tar file using pigz
        pigz_command = f"{pigz_cmd} -p {cpus} -d -c '{temp_gz}' > '{temp_tar}'"
        run_with_retries(pigz_command)
        logger.info(f"✅ [extract_zip] Step 2 completed: {temp_tar}")

        # Step 3: Extract the tar file to the target directory
        tar_command = f"{tar_cmd} -xf '{temp_tar}' -C {target_dir}"
        run_with_retries(tar_command)
        logger.info(f"🎉 [extract_zip] Step 3 completed: extracted to {target_dir}")
    finally:
        # Remove temporary files if they exist
        if temp_gz.exists():
            temp_gz.unlink()
        if temp_tar.exists():
            temp_tar.unlink()

def compute_file_hash(file_path: Path, hash_algo: str = "sha256") -> str:
    """Compute the hash of a file."""
    hash_func = getattr(hashlib, hash_algo)()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


async def async_move(src: str, dst: str) -> None:
    """Asynchronously move a file or directory from src to dst, with retries and source existence check."""
    loop = asyncio.get_event_loop()
    max_retries = 3
    delay = 2
    for attempt in range(1, max_retries + 1):
        if not Path(src).exists():
            logger.warning(f"😕 [async_move] Source does not exist: {src} (attempt {attempt})")
            if attempt < max_retries:
                logger.info(f"⏳ [async_move] Waiting {delay} seconds for source to appear...")
                await asyncio.sleep(delay)
                continue
            else:
                logger.error(f"❌ [async_move] Source does not exist after {max_retries} attempts: {src}")
                raise FileNotFoundError(f"Source does not exist after {max_retries} attempts: {src}")
        try:
            await loop.run_in_executor(None, shutil.move, src, dst)
            logger.info(f"🚚 [async_move] Move succeeded: {src} -> {dst}")
            return
        except Exception as e:
            logger.warning(f"⚠️ [async_move] Move failed (attempt {attempt}): {e}")
            if attempt < max_retries:
                logger.info(f"🔁 [async_move] Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"💥 [async_move] All {max_retries} attempts failed for: {src} -> {dst}")
                raise

async def async_rmtree(path: str) -> None:
    """Asynchronously remove a directory tree."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, shutil.rmtree, path, True)

async def async_extract_zip(paths: list) -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, extract_zip, paths)  # Assuming extract_zip is defined

def wait_for_health(port: int, timeout: int = 120) -> bool:
    """
    Wait for the service to become healthy with optimized retry logic.
    """
    health_check_url = f"http://localhost:{port}/health"
    start_time = time.time()
    wait_time = 0.5  # Start with shorter wait time for faster startup detection
    last_error = None

    logger.info(f"Waiting for service health at {health_check_url} (timeout: {timeout}s)")

    while time.time() - start_time < timeout:
        try:
            # Use shorter timeout for faster failure detection
            response = requests.get(health_check_url, timeout=3)
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    if response_data.get("status") == "ok":
                        elapsed = time.time() - start_time
                        logger.info(f"Service healthy at {health_check_url} (took {elapsed:.1f}s)")
                        return True
                except ValueError:
                    # If JSON parsing fails, just check status code
                    pass

        except requests.exceptions.ConnectionError:
            last_error = "Connection refused"
        except requests.exceptions.Timeout:
            last_error = "Request timeout"
        except requests.exceptions.RequestException as e:
            last_error = str(e)[:100]

        # Log progress every 30 seconds to avoid spam
        elapsed = time.time() - start_time
        if elapsed > 0 and int(elapsed) % 30 == 0:
            logger.debug(f"Still waiting for health check... ({elapsed:.0f}s elapsed, last error: {last_error})")

        time.sleep(wait_time)
        # Exponential backoff with cap at 10 seconds
        wait_time = min(wait_time * 1.5, 10)

    logger.error(f"Health check failed after {timeout}s. Last error: {last_error}")
    return False


def is_jetson():
    """Check if running on NVIDIA Jetson device."""
    return os.path.exists('/etc/nv_tegra_release')


def get_jetson_llama_container():
    """Get the correct llama-server container for Jetson using autotag."""
    if not is_jetson():
        return None

    try:
        # Use subprocess to run autotag commands
        import subprocess

        # Try custom build first
        result = subprocess.run(['autotag', 'my-llama-build-mmsupport'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()

        # Fallback to standard llama_cpp
        result = subprocess.run(['autotag', 'llama_cpp'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Final fallback
    return 'my-llama-build-mmsupport:latest'


def get_cuda_memory_jetson(project_dir=None, models_dir=None, container=None):
    """Get CUDA memory info on Jetson using docker container with CUDA program."""
    if not is_jetson():
        return None, None

    # Use provided paths or defaults
    if project_dir is None:
        project_dir = os.environ.get('ETERNAL_ZOO_PROJECT_DIR', '/home/sniffski/Documents/EternalAI/eternal-zoo')
    if models_dir is None:
        models_dir = os.environ.get('ETERNAL_ZOO_MODELS_DIR', '/home/sniffski/.eternal-zoo/models')
    if container is None:
        container = get_jetson_llama_container()

    cmd = f"""docker run --runtime nvidia --rm --network=host -v {project_dir}:{project_dir} -v {models_dir}:{models_dir} {container} sh -c "
cat > memcheck.cu << 'EOF'
#include <iostream>
#include <cuda_runtime.h>
int main() {{
    size_t free_byte, total_byte;
    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
    if (cudaSuccess != cuda_status) {{
        std::cout << \"Error: \" << cudaGetErrorString(cuda_status) << std::endl;
        return 1;
    }}
    std::cout << free_byte << \" \" << total_byte << std::endl;
    return 0;
}}
EOF
nvcc memcheck.cu -o memcheck 2>/dev/null && ./memcheck" """

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            if len(parts) == 2:
                free = int(parts[0])
                total = int(parts[1])
                return free, total
    except:
        pass
    return None, None

