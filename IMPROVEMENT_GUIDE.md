# Guide: Improving On-Demand Model Swapping in Eternal Zoo

## 1. Introduction

This document outlines critical improvements for the on-demand model swapping mechanism in the Eternal Zoo. The current implementation successfully loads and unloads models, but it lacks safeguards for memory management, which can lead to out-of-memory (OOM) errors and system instability, especially when switching between large models.

The following instructions provide a clear path to making the model swapping process more robust, efficient, and reliable on all platforms, including **Linux** and **macOS**.

## 2. Core Issue: Risk of Out-of-Memory Errors

The most critical issue is that the `EternalZooManager.switch_model` function attempts to load a new model process without first verifying if the system has sufficient available memory (RAM and VRAM). If a large model is requested on a system with limited resources, this can crash the application or the entire system.

## 3. Proposed Improvements

Please apply the following changes. The primary files for modification are `eternal_zoo/manager.py` and `eternal_zoo/apis.py`.

### Improvement 1: Implement Pre-loading Memory Checks (Critical)

**File:** `eternal_zoo/manager.py`
**Function:** `switch_model`

Before loading the new model subprocess, you must check for sufficient available memory.

**Instructions:**

1.  **Estimate Model Memory:** Determine the estimated memory requirement for the model being loaded. This could be a value stored in the model's metadata JSON file.
2.  **Check Available System RAM:** Use the `psutil` library to check for available system memory.
3.  **Check Available GPU VRAM (Platform-Aware):**
    *   **For NVIDIA GPUs (Linux):** Use the `pynvml` library to get available VRAM.
    *   **For macOS (Apple Silicon/AMD):** There is no standard tool like `pynvml`. On Apple Silicon's unified memory architecture, system RAM is shared with the GPU. Therefore, checking the available system RAM via `psutil` is the most practical and effective approach.
4.  **Conditionally Load Model:** If the available memory is less than the estimated requirement, abort the switch and return an error.

**Conceptual Code Example (inside `switch_model`):**

```python
# In EternalZooManager.switch_model, before starting the new process...

import psutil
# You may need to add pynvml for NVIDIA GPUs
# from pynvml import *

# 1. Get model memory requirement from its metadata
model_memory_gb = target_ai_service.get("estimated_ram_gb", 8) # Default to 8GB if not specified

# 2. Check available system RAM
available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)

print(f"Required memory: {model_memory_gb}GB, Available RAM: {available_ram_gb:.2f}GB")

if available_ram_gb < model_memory_gb:
    logger.error(f"Not enough RAM to load model {target_model_id}. Required: {model_memory_gb}GB, Available: {available_ram_gb:.2f}GB")
    return False # Abort the switch

# (Optional) Platform-specific GPU check for Linux/NVIDIA
# if sys.platform == "linux":
#     try:
#         nvmlInit()
#         handle = nvmlDeviceGetHandleByIndex(0) # Assuming GPU 0
#         info = nvmlDeviceGetMemoryInfo(handle)
#         available_vram_gb = info.free / (1024 ** 3)
#         if available_vram_gb < model_memory_gb:
#             logger.error(f"Not enough VRAM to load model...")
#             return False
#     except Exception as e:
#         logger.warning(f"Could not check VRAM: {e}")


# ... if checks pass, proceed to load the model process ...
logger.info("Sufficient memory available. Proceeding with model switch.")
```

### Improvement 2: Introduce a Post-Termination Cool-down Period

**File:** `eternal_zoo/manager.py`
**Function:** `switch_model`

Give the operating system a moment to reclaim all resources from the terminated model process before starting a new one.

**Instructions:**

After terminating the old model's process, add a short, non-blocking sleep.

**Conceptual Code Example (inside `switch_model`):**

```python
# In EternalZooManager.switch_model...

# Terminate the old active process
active_pid = active_ai_service.get("pid", None)
if active_pid and psutil.pid_exists(active_pid):
    self._terminate_process_safely(active_pid, "EternalZoo AI Service", force=True)

# --- ADD THIS ---
# Add a cool-down period to allow the OS to reclaim resources
logger.info("Allowing a 2-second cool-down period...")
await asyncio.sleep(2)
# ----------------

# Update metadata for the old service
active_ai_service["active"] = False
# ... etc ...
```

### Improvement 3: Convert `switch_model` to Asynchronous

**Files:** `eternal_zoo/manager.py`, `eternal_zoo/apis.py`

To prevent the model switching process from blocking the entire request worker, convert `switch_model` and its call site to be fully asynchronous.

**Instructions:**

1.  **In `eternal_zoo/manager.py`:**
    *   Change the `switch_model` function signature from `def` to `async def`.
    *   Replace the call to `subprocess.Popen` with `await asyncio.create_subprocess_exec()`.
    *   Replace any `time.sleep()` calls with `await asyncio.sleep()`.

2.  **In `eternal_zoo/apis.py`:**
    *   In the `_ensure_model_active_in_queue` function, use `await` when calling `eternal_zoo_manager.switch_model()`.

**Conceptual Code Example:**

**`eternal_zoo/manager.py`:**
```python
# Change signature
async def switch_model(self, target_model_id: str) -> bool:
    # ... (logic to find services) ...

    # Use async sleep for cool-down
    await asyncio.sleep(2)

    # ... (logic to build command) ...

    # Use asyncio.create_subprocess_exec
    ai_process = await asyncio.create_subprocess_exec(
        *running_ai_command,
        stderr=stderr_log,
        preexec_fn=os.setsid
    )

    # ... (rest of the logic) ...
```

**`eternal_zoo/apis.py`:**
```python
# In RequestProcessor._ensure_model_active_in_queue...

# Await the async function call
success = await eternal_zoo_manager.switch_model(model_requested)
```

## 4. Code Review Focus

When implementing these changes, please pay close attention to the following areas of the codebase:

-   **`eternal_zoo/manager.py`**:
    -   `switch_model()`: This is the primary function to modify.
    -   `_terminate_process_safely()`: Review this to understand how processes are killed, but note that it is already a good cross-platform implementation. **Do not add Linux-specific `cgroups`** to maintain macOS compatibility.
-   **`eternal_zoo/apis.py`**:
    -   `RequestProcessor._ensure_model_active_in_queue()`: This is the function that triggers the model switch and needs to `await` the refactored `switch_model` call.

## 5. Conclusion

By implementing these improvements, the Eternal Zoo will be significantly more stable and reliable. The memory checks will prevent system-wide crashes, and the asynchronous refactoring will improve the responsiveness of the service during model switches.
