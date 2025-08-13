import os
import json
import random
import aiohttp
import asyncio
import subprocess
from loguru import logger
from huggingface_hub import HfApi
from eternal_zoo.utils import async_move, async_rmtree, compute_file_hash
from eternal_zoo.models import FEATURED_MODELS, HASH_TO_MODEL
from eternal_zoo.constants import DEFAULT_MODEL_DIR, POSTFIX_MODEL_PATH, GATEWAY_URLS, ETERNAL_AI_METADATA_GW, PREFIX_DOWNLOAD_LOG

SLEEP_TIME = 2
CONNECTION_POOL_SIZE = 32  # Increased connection pool

hf_api = HfApi(
    token = os.getenv("HF_TOKEN")
)

def get_all_files_from_hf_repo(repo_id: str) -> list[str]:
    """
    Get all files from a Hugging Face repository.
    """
    files = hf_api.list_repo_files(repo_id, revision="main")
    return files

def get_infos_from_paths(repo_id: str, paths: list[str]) -> dict:
    """
    Get the size of a list of paths from a Hugging Face repository.
    """
    infos = {
        "total_size": 0,
        "files": {}
    }
    paths_info = hf_api.get_paths_info(repo_id=repo_id, paths=paths)
    for path in paths_info:
        infos["total_size"] += path.size
        sha256 = path.lfs.sha256 if path.lfs else None
        infos["files"][path.rfilename] = {
            "sha256": sha256,
            "size": path.size,
        }
    return infos

def check_valid_folder(infos: dict, folder_path: str) -> bool:
    """
    Check if the folder is valid by comparing the sha256 of the files in the folder with the sha256 in the infos.
    """
    for file_name, file_info in infos["files"].items():
        if file_info["sha256"] is None:
            continue
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(file_path):
            return False
        computed_sha256 = compute_file_hash(file_path)
        if computed_sha256 != file_info["sha256"]:
            return False
        logger.info(f"File {file_name} is valid")
    return True


async def pick_fastest_gateway(filecoin_hash: str, gateways: list[str], timeout: int = 5) -> str:
    """
    Check the speed of each gateway and return the fastest one for the given filecoin_hash.
    If only one gateway is provided, return it immediately.
    Args:
        filecoin_hash (str): The IPFS hash to test download speed for.
        gateways (list[str]): List of gateway URLs.
        timeout (int): Timeout in seconds for each speed test.
    Returns:
        str: The fastest gateway URL, or the first in the list if all fail.
    """

    # If there is only one gateway, return it immediately (no need to check)
    if len(gateways) == 1:
        logger.info(f"[pick_fastest_gateway] ✅ Only one gateway provided, returning: {gateways[0]}")
        return gateways[0]

    logger.info(f"[pick_fastest_gateway] 🚦 Checking speed for {len(gateways)} gateways with hash: {filecoin_hash}")

    async def check_gateway(gateway: str) -> tuple[str, float]:
        url = f"{gateway}{filecoin_hash}"
        logger.info(f"[pick_fastest_gateway] 🔍 Testing gateway: {url}")
        start = asyncio.get_event_loop().time()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                # Use GET with Range header to fetch only the first 1KB, since some gateways may not support HEAD
                headers = {"Range": "bytes=0-1023"}
                async with session.get(url, headers=headers) as resp:
                    if resp.status in (200, 206):  # 206 = Partial Content
                        elapsed = asyncio.get_event_loop().time() - start
                        logger.info(f"[pick_fastest_gateway] ⏱️ Gateway {gateway} responded in {elapsed:.3f} seconds.")
                        return gateway, elapsed
                    else:
                        logger.warning(f"[pick_fastest_gateway] ⚠️ Gateway {gateway} returned status {resp.status}.")
        except Exception as e:
            logger.warning(f"[pick_fastest_gateway] ❌ Error with gateway {gateway}: {e}")
        # Return infinity if the gateway is not available or too slow
        return gateway, float('inf')

    # Run all gateway checks concurrently
    tasks = [check_gateway(gw) for gw in gateways]
    results = await asyncio.gather(*tasks)
    # Filter for gateways that responded successfully
    valid_results = [r for r in results if r[1] != float('inf')]
    if valid_results:
        fastest = min(valid_results, key=lambda x: x[1])
        logger.info(f"[pick_fastest_gateway] 🚀 Fastest gateway selected: {fastest[0]} (time: {fastest[1]:.3f} seconds)\n")
        return fastest[0]
    else:
        logger.warning(f"[pick_fastest_gateway] ⚠️ All gateways timed out or failed. Using the first gateway as fallback: {gateways[0]}")
        return gateways[0]
    
def calculate_backoff(attempt: int) -> int:
    return min(SLEEP_TIME * (2 ** (attempt - 1)), 300)
    

async def fetch_model_metadata_async(filecoin_hash: str) -> tuple[bool, dict | None]:
    """
    Asynchronously fetch model metadata from the fastest gateway with infinite retry logic.
    
    Args:
        filecoin_hash (str): The IPFS hash of the model
        
    Returns:
        dict: Model metadata JSON, or None if all attempts fail
    """
    metadata_path = DEFAULT_MODEL_DIR / f"{filecoin_hash}.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return True, json.load(f)
    
    # Select the fastest gateway before downloading
    logger.info("Checking gateway speeds...")
    # for meta-data add more with ETERNAL_AI_METADATA_GW
    gateways_with_eternal = GATEWAY_URLS + [ETERNAL_AI_METADATA_GW]
    best_gateway = await pick_fastest_gateway(filecoin_hash, gateways_with_eternal)
    logger.info(f"Using fastest gateway: {best_gateway}")
    input_link = f"{best_gateway}{filecoin_hash}"

    # Set up session parameters with optimized limits
    timeout = aiohttp.ClientTimeout(total=180, connect=60)
    connector = aiohttp.TCPConnector(limit=CONNECTION_POOL_SIZE)

    # Use infinite retry loop with exponential backoff
    attempt = 1
    while True:  # Infinite loop until success or user cancellation
        backoff = calculate_backoff(attempt)

        try:
            logger.info(f"Downloading model metadata (attempt {attempt})")

            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                async with session.get(input_link) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch metadata: HTTP {response.status}")
                        logger.warning(f"Retrying in {backoff} seconds")
                        await asyncio.sleep(backoff)
                        attempt += 1
                        continue

                    # Parse metadata
                    data = await response.json()
                    data["filecoin_hash"] = filecoin_hash
                    with open(metadata_path, "w") as f:
                        json.dump(data, f)
                    return True, data

        except KeyboardInterrupt:
            logger.info("Metadata download canceled by user")
            return False, None
        except aiohttp.ClientError as e:
            logger.warning(f"HTTP error on attempt {attempt}: {e}")
            logger.warning(f"Retrying in {backoff} seconds")
            await asyncio.sleep(backoff)
            attempt += 1
            continue
        except Exception as e:
            logger.warning(f"Download attempt {attempt} failed: {e}")
            logger.warning(f"Retrying in {backoff} seconds")
            await asyncio.sleep(backoff)
            attempt += 1
            continue

async def download_model_async_by_hash(hf_data: dict, filecoin_hash: str) -> tuple[bool, str | None]:
    """
    Asynchronously download a model from Filecoin using its IPFS hash.
    This function will select the fastest gateway from a list of gateways by testing their response times,
    and use the fastest one to download the model.
    
    Args:
        filecoin_hash (str): The IPFS hash of the model
        
    Returns:
        tuple[bool, str | None]: (success, local_path) - True and path if successful, False and None if failed
    """
    # Ensure output directory exists

    is_lora = False
    lora_metadata = None

    local_path = DEFAULT_MODEL_DIR / f"{filecoin_hash}{POSTFIX_MODEL_PATH}"
    local_path_str = str(local_path.absolute())
    
    # Fetch model metadata using the dedicated async function
    success, data = await fetch_model_metadata_async(filecoin_hash)

    if not success:
        logger.error("Failed to fetch model metadata")
        return False, None
        
    data["filecoin_hash"] = filecoin_hash
    data["local_path"] = local_path_str
    is_lora = data.get("lora", False)   

    if is_lora:
        # First, try to load metadata from local LoRA model if it exists
        if os.path.exists(local_path_str):
            metadata_path = os.path.join(local_path_str, "metadata.json")
            try:
                with open(metadata_path, "r") as f:
                    lora_metadata = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load local LoRA metadata: {e}")
                lora_metadata = {}
        
            base_model_hash = lora_metadata["base_model"]
            lora_base_model_path_str = str(DEFAULT_MODEL_DIR / f"{base_model_hash}{POSTFIX_MODEL_PATH}")
            if os.path.exists(lora_base_model_path_str):
                logger.info(f"Using existing LoRA model at {local_path_str}")
                logger.info(f"Using existing base model at {lora_base_model_path_str}")
                return True, local_path_str
            else:
                logger.warning(f"LoRA model exists but base model not found at {lora_base_model_path_str}")
                logger.info(f"Downloading missing base model: {base_model_hash}")
                base_model_hf_data = None
                if base_model_hash in HASH_TO_MODEL:
                    base_model_hf_data = FEATURED_MODELS[HASH_TO_MODEL[base_model_hash]]
                success, base_model_path = await download_model_async_by_hash(base_model_hf_data, base_model_hash)
                if not success:
                    logger.error(f"Failed to download base model: {base_model_hash}")
                    return False, None
                logger.info(f"Successfully downloaded base model and using existing LoRA model")
                return True, local_path_str
    else:
        if os.path.exists(local_path_str):
            logger.info(f"Using existing model at {local_path_str}")
            return True, local_path_str

    if is_lora:
        success, hf_res = await download_model_from_hf(hf_data)
        print(f"success: {success}")
        print(f"hf_res: {hf_res}")
        if not success:
            logger.error("Failed to download LoRA model")
            return False, None
        
        tmp_dir = hf_res["tmp_dir"]
        model_path = hf_res["model_path"]

        await async_move(model_path, local_path_str)
        await async_rmtree(tmp_dir)
            
        lora_metadata_path = os.path.join(local_path_str, "metadata.json")
        try:
            with open(lora_metadata_path, "r") as f:
                lora_metadata = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load LoRA metadata from downloaded model: {e}")
            return False, None
        
        base_model_hash = lora_metadata.get("base_model", None)
        if base_model_hash is None:
            logger.error("No base_model found in LoRA metadata")
            return False, None

        base_model_hf_data = None

        if base_model_hash in HASH_TO_MODEL:
            base_model_hf_data = FEATURED_MODELS[HASH_TO_MODEL[base_model_hash]]

        print(f"base_model_hf_data: {base_model_hf_data}")
        print(f"base_model_hash: {base_model_hash}")

        success, base_model_path = await download_model_async_by_hash(base_model_hf_data, base_model_hash)
        if not success:
            logger.error(f"Failed to download base model: {base_model_hash}")
            return False, None
        logger.info(f"Successfully downloaded LoRA model and base model: {local_path_str}")
        return True, local_path_str

    success, hf_res = await download_model_from_hf(hf_data)
    print(f"success: {success}")
    if not success:
        logger.error("Failed to download model")
        return False, None

    tmp_dir = hf_res["tmp_dir"]
    model_path = hf_res["model_path"]
    projector_path = hf_res.get("projector_path", None)

    if projector_path:
        await async_move(projector_path, local_path_str + "-projector")

    await async_move(model_path, local_path_str)
    await async_rmtree(tmp_dir)

    return True, local_path_str

class HuggingFaceProgressTracker:
    def __init__(self, total_size: int):
        """Initialize the tracker with the total size of files to download.

        Args:
            total_size (int): Total size of files in bytes.
        """
        self.total_size = total_size * random.uniform(1.2, 1.5)
        self.current_size = 0
        self.percentage = 0.0

    def update_current_size(self, current_size: int):
        """Update the current size and recalculate the percentage.

        Args:
            current_size (int): Current size of tmp_dir in bytes.
        """
        self.current_size = current_size
        if self.total_size > 0:
            self.percentage = min((self.current_size / self.total_size) * 100, 95)
        else:
            self.percentage = 0.0

    def get_progress(self) -> dict:
        """Return the current progress as a dictionary.

        Returns:
            dict: Contains current_size, total_size, and percentage.
        """
        return {
            "current_size": self.current_size,
            "total_size": self.total_size,
            "percentage": self.percentage
        }
    
async def calculate_current_size_of_folder(folder_path: str) -> int:
    process = await asyncio.create_subprocess_shell(
        f"du -s {folder_path}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        size_str = stdout.decode().split()[0]
        current_size = int(size_str) * 1024
        return current_size
    else:
        logger.warning(f"Failed to get directory size: {stderr.decode()}")
        return 0

async def track_progress(tracker: HuggingFaceProgressTracker, folder_path: str):
    """Periodically track the size of tmp_dir and update the tracker.

    Args:
        tracker (HuggingFaceProgressTracker): The progress tracker instance.
        tmp_dir (str): Path to the temporary directory.
    """
    last_logged = -1  # Track the last logged percentage
    while True:
        try:
           current_size = await calculate_current_size_of_folder(folder_path)
           tracker.update_current_size(current_size)
           if tracker.percentage - last_logged >= 1:
               logger.info(f"{PREFIX_DOWNLOAD_LOG} --progress {tracker.percentage:.2f}%")
               last_logged = tracker.percentage
        except Exception as e:
            logger.warning(f"Error in progress tracking: {e}")
        await asyncio.sleep(2)  # Check every 2 second 

async def download_model_from_hf(data: dict, final_dir: str | None = None) -> tuple[bool, dict | None]:
    """
    Download model from HuggingFace Hub with infinite retry logic and exponential backoff.
    Downloads forever until finished or canceled by user.
    
    Args:
        data: Model metadata dictionary containing hf_repo, hf_file, and local_path
        
    Returns:
        tuple[bool, str | None]: (success, local_path) - True and path if successful, False and None if failed
    """

    res = {}
    repo_id = data["repo"]
    model = data.get("model", None)
    projector = data.get("projector", None)
    pattern = data.get("pattern", None) 
    tmp_dir = str(DEFAULT_MODEL_DIR/f"tmp_{repo_id.replace('/', '_')}")   
    os.makedirs(tmp_dir, exist_ok=True)

    # Remove all lock files in the cache directory
    CACHE_DIR = os.path.join(tmp_dir, ".cache", "huggingface", "download")
    if os.path.exists(CACHE_DIR):
        for root, dirs, files in os.walk(CACHE_DIR):
            for file in files:
                if file.endswith(".lock"):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        logger.info(f"Removed lock file: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to remove lock file: {file_path}: {e}")

    final_path = None
    final_projector_path = None
      
    files = []
    if model:
        files.append(model)
    
    if projector:
        files.append(projector)

    if not files:
        files = get_all_files_from_hf_repo(repo_id)

    if pattern:
        # filter files by pattern
        filter = f"-{pattern}-"
        files = [file for file in files if filter in file]

    infos = get_infos_from_paths(repo_id, files)
    tracker = HuggingFaceProgressTracker(infos["total_size"])
    progress_task = asyncio.create_task(track_progress(tracker, tmp_dir))
    res["tmp_dir"] = tmp_dir

    if final_dir:
        
        final_path = os.path.join(final_dir, repo_id.replace("/", "_"))

        if model:
            final_path = os.path.join(final_dir, model)
        
        if pattern:
            final_path = os.path.join(final_dir, repo_id.replace("/", "_") + "_" + pattern)

        if projector:
            final_projector_path = final_path + "-projector"

    try:
        attempt = 1
        while True:  # Infinite loop until success or user cancellation
            try:
                logger.info(f"Download attempt {attempt} for {repo_id}")
                            
                # Run the synchronous download in a thread executor to avoid blocking
                loop = asyncio.get_event_loop()
                
                if model is None:
                    
                    if final_path:
                        if os.path.exists(final_path):
                            logger.info(f"Model {final_path} already exists")
                            return True, {"model_path": final_path, "tmp_dir": tmp_dir}

                    if pattern:                        
                        command = f"hf download {repo_id} --local-dir {tmp_dir} --include \"*{pattern}*\""

                        await loop.run_in_executor(
                            None,
                            lambda: subprocess.run(command, shell=True)
                        )
                        
                        res["model_path"] = tmp_dir
                        if not check_valid_folder(infos, tmp_dir):
                            raise Exception("Model is not the same as the one on Hugging Face")

                        if final_path:
                            await async_move(tmp_dir, final_path)
                            res["model_path"] = final_path
                                                   
                    else:
                        command = f"hf download {repo_id} --local-dir {tmp_dir}"
                        await loop.run_in_executor(
                            None,
                            lambda: subprocess.run(command, shell=True)
                        )

                        res["model_path"] = tmp_dir
                        if not check_valid_folder(infos, tmp_dir):
                            raise Exception("Model is not the same as the one on Hugging Face")
                        
                        if final_path:
                            await async_move(tmp_dir, final_path)
                            res["model_path"] = final_path
                        
                    return True, res
                
                skip_download_model = False
                skip_download_projector = False

                if final_path:
                    if os.path.exists(final_path):
                        skip_download_model = True
                        res["model_path"] = final_path
                        logger.info(f"Model {final_path} already exists")

                if final_projector_path:
                    if os.path.exists(final_projector_path):
                        skip_download_projector = True
                        res["projector_path"] = final_projector_path
                        logger.info(f"Projector {final_projector_path} already exists")

                if skip_download_model and skip_download_projector:
                    return True, res

                if not skip_download_model:
                    command = f"hf download {repo_id} {model} --local-dir {tmp_dir}"
                    await loop.run_in_executor(
                        None,
                        lambda: subprocess.run(command, shell=True)
                    )
                    res["model_path"] = os.path.join(tmp_dir, model)

                if projector:
                    if final_projector_path:
                        if os.path.exists(final_projector_path):
                            skip_download_projector = True
                            res["projector_path"] = final_projector_path
                            logger.info(f"Projector {final_projector_path} already exists")
                            
                    if not skip_download_projector:
                        command = f"hf download {repo_id} {projector} --local-dir {tmp_dir}"
                        await loop.run_in_executor(
                            None,
                            lambda: subprocess.run(command, shell=True)
                        )
                        res["projector_path"] = os.path.join(tmp_dir, projector)
                
                if not skip_download_model:
                    sha256_model = compute_file_hash(res["model_path"])
                    if sha256_model != infos["files"][model]["sha256"]:
                        raise Exception("Model is not the same as the one on Hugging Face")
                    else:
                        logger.info(f"Model {res['model_path']} is valid")
                
                if projector:
                    if not skip_download_projector:
                        sha256_projector = compute_file_hash(res["projector_path"])
                        if sha256_projector != infos["files"][projector]["sha256"]:
                            raise Exception("Projector is not the same as the one on Hugging Face")
                        else:
                            logger.info(f"Projector {res['projector_path']} is valid")

                if final_path:
                    if not os.path.exists(final_path):
                        await async_move(res["model_path"], final_path)
                        res["model_path"] = final_path
                
                if final_projector_path:
                    if not os.path.exists(final_projector_path):
                        await async_move(res["projector_path"], final_projector_path)
                        res["projector_path"] = final_projector_path

                return True, res
                                    
            except KeyboardInterrupt:
                logger.info("Download canceled by user")

            except Exception as e:
                logger.warning(f"Download attempt {attempt} failed: {e}")
                
                # Calculate exponential backoff with jitter
                backoff_time = calculate_backoff(attempt)
                logger.warning(f"Retrying in {backoff_time}s (attempt {attempt + 1})")
                await asyncio.sleep(backoff_time)
                
                attempt += 1
        
    except KeyboardInterrupt:
        logger.info("Download canceled by user")
        return False, None

    finally:
        # Cancel the progress task to stop it from running indefinitely
        progress_task.cancel()
        try:
            await progress_task  # Wait for cancellation to complete
        except asyncio.CancelledError:
            logger.info(f"{PREFIX_DOWNLOAD_LOG} --progress {tracker.percentage:.2f}%")

async def download_model_async(hf_data: dict, filecoin_hash: str | None = None) -> tuple[bool, str | None]:
    """
    Download model with infinite retry logic until success or cancellation.
    
    Args:
        hf_data: Model metadata dictionary
        filecoin_hash: Optional IPFS hash for Filecoin download
        
    Returns:
        tuple[bool, str | None]: (success, local_path) - True and path if successful, False and None if failed
    """
    logger.info(f"Downloading model: {hf_data}")
    logger.info(f"Filecoin hash: {filecoin_hash}")
    
    attempt = 1
    while True:  # Infinite loop until success or user cancellation
        try:
            logger.info(f"Download attempt {attempt}")
            path = None
            
            if filecoin_hash:
                print(f"DOWNLOADING MODEL: {hf_data}")
                print(f"FILECOIN HASH: {filecoin_hash}")
                success, path = await download_model_async_by_hash(hf_data, filecoin_hash)

                print(f"success: {success}")
                print(f"path: {path}")
                if success:
                    logger.info(f"Successfully downloaded model to {path}")
                    return True, path
                else:
                    logger.warning(f"Download attempt {attempt} failed")
            else:
                final_dir = str(DEFAULT_MODEL_DIR)
                success, hf_res = await download_model_from_hf(hf_data, final_dir)

                if success:
                    tmp_dir = hf_res.get("tmp_dir", None)
                    
                    if tmp_dir:
                        await async_rmtree(tmp_dir)
                    
                    path = hf_res["model_path"]
                    logger.info(f"Successfully downloaded model to {path}")
                    return True, path
                else:
                    logger.warning(f"Download attempt {attempt} failed")
            
            # Calculate exponential backoff with jitter
            backoff_time = calculate_backoff(attempt)
            logger.warning(f"Retrying in {backoff_time}s (attempt {attempt + 1})")
            await asyncio.sleep(backoff_time)
            attempt += 1
            
        except KeyboardInterrupt:
            logger.info("Download canceled by user")
            return False, None
        except Exception as e:
            logger.warning(f"Download attempt {attempt} failed with exception: {e}")
            backoff_time = calculate_backoff(attempt)
            logger.warning(f"Retrying in {backoff_time}s (attempt {attempt + 1})")
            await asyncio.sleep(backoff_time)
            attempt += 1