import os
import json
import time
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from lighthouseweb3 import Lighthouse
from concurrent.futures import ThreadPoolExecutor, as_completed
from crypto_models.utils import compute_file_hash, compress_folder, extract_zip

load_dotenv()

def upload_to_lighthouse(file_path: Path):
    """
    Upload a file to Lighthouse.storage and measure the time taken.
    Note: Assumes lighthouse_web3.upload is synchronous; adjust if async.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        file_hash = compute_file_hash(file_path)

        start_time = time.time()
        file_name = os.path.basename(file_path)
        lh = Lighthouse(token=os.getenv("LIGHTHOUSE_API_KEY"))
        response = lh.upload(str(file_path))  # Convert Path to string
        elapsed_time = time.time() - start_time
        upload_speed = file_size / elapsed_time if elapsed_time > 0 else 0

        print(f"Uploaded {file_path}: {elapsed_time:.2f}s, {upload_speed:.2f} MB/s")
        if "data" in response and "Hash" in response["data"]:
            cid = response["data"]["Hash"]
            return {"cid": cid, "file_hash": file_hash, "size_mb": file_size, "file_name": file_name}, None
        else:
            return None, "No CID in response"
    except Exception as e:
        print(f"Upload failed for {file_path}: {str(e)}")
        return None, str(e)

def upload_folder_to_lighthouse(
    folder_path: str, zip_chunk_size=512, max_retries=20, threads=16, max_workers=4, **kwargs
):
    """
    Upload a folder to Lighthouse.storage by compressing it into parts and uploading in parallel.
    """
    folder_path = Path(folder_path)
    folder_name = folder_path.name
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    metadata = {
        "folder_name": folder_name,
        "chunk_size_mb": zip_chunk_size,
        "files": [],
        **kwargs,
    }
    # Need to import tempfile at the top if not already
    metadata_fd, metadata_path_str = tempfile.mkstemp(suffix='.json', prefix=f"{folder_name}_")
    os.close(metadata_fd)  # Close the file descriptor as we'll open it later
    metadata_path = Path(metadata_path_str)
    temp_dir = None

    try:
        # Compress the folder
        temp_dir = compress_folder(folder_path, zip_chunk_size, threads)
        part_files = [
            os.path.join(temp_dir, f) for f in sorted(os.listdir(temp_dir))
            if f.startswith(f"{folder_name}.zip.part-")
        ]
        metadata["num_of_files"] = len(part_files)
        print(f"Uploading {len(part_files)} parts to Lighthouse.storage...")

        # Parallel upload with optimized retries
        errors = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            def upload_with_retry(part_path):
                backoff_delay = 1  # Start with 1 second, exponential backoff
                for attempt in range(max_retries):
                    file_info, error = upload_to_lighthouse(part_path)
                    if file_info:
                        return file_info, None
                    print(f"Retry {attempt + 1}/{max_retries} for {part_path}")
                    # Exponential backoff instead of fixed delay
                    time.sleep(min(backoff_delay, 8))  # Cap at 8 seconds
                    backoff_delay *= 1.5
                return None, f"Failed after {max_retries} attempts"

            # Submit all uploads at once for better parallelism
            futures = {executor.submit(upload_with_retry, part): part for part in part_files}
            
            # Process completed uploads as they finish
            for future in as_completed(futures):
                part_path = futures[future]
                file_info, error = future.result()
                if file_info:
                    metadata["files"].append(file_info)
                else:
                    errors.append((part_path, error))

        # Save metadata and handle results
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        if errors:
            error_msg = "\n".join(f"{path}: {err}" for path, err in errors)
            print(f"Completed with {len(errors)} errors:\n{error_msg}")
            return None, f"Partial upload failure: {len(errors)} parts failed"
        print("All parts uploaded successfully!")
        
        # upload metadata to Lighthouse
        metadata_info, error = upload_to_lighthouse(metadata_path)

        if metadata_info:
            metadata['cid'] = metadata_info['cid']
            final_metadata_path = Path.cwd()/f"{folder_name}_metadata.json"
            with open(final_metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            print(f"Metadata uploaded: {metadata_info['cid']}")
            return metadata, None
        else:
            return None, error
        
    except Exception as e:
        print(f"Upload process failed: {str(e)}")
        return None, str(e)
    finally:
        # merge the parts to the original folder
        all_parts = [f for f in os.listdir(temp_dir) if f.startswith(f"{folder_name}.zip.part-")]
        sorted_parts = sorted(all_parts)
        extract_zip([Path(temp_dir) / part for part in sorted_parts])