"""
This module provides a FastAPI application that acts as a proxy or processor for chat completion and embedding requests,
forwarding them to an underlying service running on a local port. It handles both text and vision-based chat completions,
as well as embedding generation, with support for streaming responses.
"""

import logging
import httpx
import asyncio
import time
import json
import uuid
import psutil
# Import configuration settings
from crypto_models.config import config
from json_repair import repair_json
from typing import Dict, Any, Optional
from crypto_models.core import CryptoModelsManager, CryptoAgentsServiceError
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

# Import schemas from schema.py
from crypto_models.schema import (
    Choice,
    Message,
    ModelCard,
    ModelList,
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ChatCompletionChunk,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
    ChatCompletionResponse,
    ImageGenerationRequest,
    ImageGenerationResponse
)

# Set up logging with both console and file output
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI()

# Initialize CryptoModels Manager for service management
crypto_models_manager = CryptoModelsManager()


# Performance constants from config
IDLE_TIMEOUT = config.performance.IDLE_TIMEOUT
UNLOAD_CHECK_INTERVAL = config.performance.UNLOAD_CHECK_INTERVAL
SERVICE_START_TIMEOUT = config.performance.SERVICE_START_TIMEOUT
POOL_CONNECTIONS = config.performance.POOL_CONNECTIONS
POOL_KEEPALIVE = config.performance.POOL_KEEPALIVE
HTTP_TIMEOUT = config.performance.HTTP_TIMEOUT
STREAM_TIMEOUT = config.performance.STREAM_TIMEOUT
MAX_RETRIES = config.performance.MAX_RETRIES
RETRY_DELAY = config.performance.RETRY_DELAY
MAX_QUEUE_SIZE = config.performance.MAX_QUEUE_SIZE
HEALTH_CHECK_INTERVAL = config.performance.HEALTH_CHECK_INTERVAL
STREAM_CHUNK_SIZE = config.performance.STREAM_CHUNK_SIZE

# Utility functions
def get_service_info() -> Dict[str, Any]:
    """Get service info from CryptoModelsManager with error handling."""
    try:
        return crypto_models_manager.get_service_info()
    except CryptoAgentsServiceError as e:
        raise HTTPException(status_code=503, detail=str(e))

def get_service_port() -> int:
    """Get service port with caching and error handling."""
    service_info = get_service_info()
    if "port" not in service_info:
        raise HTTPException(status_code=503, detail="Service port not configured")
    return service_info["port"]

def convert_request_to_dict(request) -> Dict[str, Any]:
    """Convert request object to dictionary, supporting both Pydantic v1 and v2."""
    return request.model_dump() if hasattr(request, "model_dump") else request.dict()

def generate_request_id() -> str:
    """Generate a short request ID for tracking."""
    return str(uuid.uuid4())[:8]

def generate_chat_completion_id() -> str:
    """Generate a chat completion ID."""
    return f"chatcmpl-{uuid.uuid4().hex}"

# Service Functions
class ServiceHandler:
    """Handler class for making requests to the underlying service."""
    
    @staticmethod
    def _create_vision_error_response(request: ChatCompletionRequest, content: str):
        """Create error response for vision requests when multimodal is not supported."""
        if request.stream:
            async def error_stream():
                chunk = {
                    "id": generate_chat_completion_id(),
                    "choices": [{
                        "delta": {"content": content},
                        "finish_reason": "stop",
                        "index": 0
                    }],
                    "created": int(time.time()),
                    "model": request.model,
                    "object": "chat.completion.chunk"
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream")
        else:
            return ChatCompletionResponse(
                id=generate_chat_completion_id(),
                object="chat.completion",
                created=int(time.time()),
                model=request.model,
                choices=[Choice(
                    finish_reason="stop",
                    index=0,
                    message=Message(role="assistant", content=content)
                )]
            )
    
    @staticmethod
    async def generate_text_response(request: ChatCompletionRequest):
        """Generate a response for chat completion requests, supporting both streaming and non-streaming."""
        port = get_service_port()
        
        if request.is_vision_request():
            service_info = get_service_info()
            if not service_info.get("multimodal", False):
                content = "Unfortunately, I'm not equipped to interpret images at this time. Please provide a text description if possible."
                return ServiceHandler._create_vision_error_response(request, content)
                
        request.clean_messages()
        request.enhance_tool_messages()
        request_dict = convert_request_to_dict(request)
        
        if request.stream:
            return StreamingResponse(
                ServiceHandler._stream_generator(port, request_dict),
                media_type="text/event-stream"
            )

        # Make a non-streaming API call
        response_data = await ServiceHandler._make_api_call(port, "/v1/chat/completions", request_dict)
        return ChatCompletionResponse(
            id=response_data.get("id", generate_chat_completion_id()),
            object=response_data.get("object", "chat.completion"),
            created=response_data.get("created", int(time.time())),
            model=request.model,
            choices=response_data.get("choices", [])
        )
    
    @staticmethod
    async def generate_embeddings_response(request: EmbeddingRequest):
        """Generate a response for embedding requests."""
        port = get_service_port()
        request_dict = convert_request_to_dict(request)
        response_data = await ServiceHandler._make_api_call(port, "/v1/embeddings", request_dict)
        return EmbeddingResponse(
            object=response_data.get("object", "list"),
            data=response_data.get("data", []),
            model=request.model
        )
    
    @staticmethod
    async def generate_image_response(request: ImageGenerationRequest):
        """Generate a response for image generation requests."""
        port = get_service_port()
        request_dict = convert_request_to_dict(request)
        response_data = await ServiceHandler._make_api_call(port, "/v1/images/generations", request_dict)
        return ImageGenerationResponse(
            created=response_data.get("created", int(time.time())),
            data=response_data.get("data", [])
        )
    
    @staticmethod
    async def _make_api_call(port: int, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a non-streaming API call to the specified endpoint and return the JSON response."""
        try:
            response = await app.state.client.post(
                f"http://localhost:{port}{endpoint}", 
                json=data,
                timeout=HTTP_TIMEOUT
            )
            logger.info(f"Received response with status code: {response.status_code}")
            
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"Error: {response.status_code} - {error_text}")
                if response.status_code < 500:
                    raise HTTPException(status_code=response.status_code, detail=error_text)
            
            # Cache JSON parsing to avoid multiple calls
            json_response = response.json()
            return json_response
            
        except httpx.TimeoutException as e:
            raise HTTPException(status_code=504, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @staticmethod
    async def _stream_generator(port: int, data: Dict[str, Any]):
        """Generator for streaming responses from the service."""
        buffer = ""
        tool_calls = {}
        
        def _extract_json_data(line: str) -> Optional[str]:
            """Extract JSON data from SSE line, return None if not valid data."""
            line = line.strip()
            if not line or line.startswith(': ping'):
                return None
            if line.startswith('data: '):
                return line[6:].strip()
            return None
        
        def _process_tool_call_delta(delta_tool_call, tool_calls: dict):
            """Process tool call delta and update tool_calls dict."""
            tool_call_index = str(delta_tool_call.index)
            if tool_call_index not in tool_calls:
                tool_calls[tool_call_index] = {"arguments": ""}
            
            if delta_tool_call.id is not None:
                tool_calls[tool_call_index]["id"] = delta_tool_call.id
                
            function = delta_tool_call.function
            if function.name is not None:
                tool_calls[tool_call_index]["name"] = function.name
            if function.arguments is not None:
                tool_calls[tool_call_index]["arguments"] += function.arguments
        
        def _create_tool_call_chunks(tool_calls: dict, chunk_obj):
            """Create tool call chunks for final output - yields each chunk separately."""
            chunk_obj_copy = chunk_obj.copy()
            
            for tool_call_index, tool_call in tool_calls.items():
                try:
                    tool_call_obj = json.loads(repair_json(json.dumps(tool_call)))
                    tool_call_id = tool_call_obj.get("id", None)
                    tool_call_name = tool_call_obj.get("name", "")
                    tool_call_arguments = tool_call_obj.get("arguments", "")
                    if tool_call_arguments == "":
                        tool_call_arguments = "{}"
                    function_call = ChoiceDeltaFunctionCall(
                        name=tool_call_name,
                        arguments=tool_call_arguments
                    )
                    delta_tool_call = ChoiceDeltaToolCall(
                        index=int(tool_call_index),
                        id=tool_call_id,
                        function=function_call,
                        type="function"
                    )
                    chunk_obj_copy.choices[0].delta.content = None
                    chunk_obj_copy.choices[0].delta.tool_calls = [delta_tool_call]  
                    chunk_obj_copy.choices[0].finish_reason = "tool_calls"
                    yield f"data: {chunk_obj_copy.json()}\n\n"
                except Exception as e:
                    logger.error(f"Failed to create tool call chunk: {e}")
                    chunk_obj_copy.choices[0].delta.content = None
                    chunk_obj_copy.choices[0].delta.tool_calls = []
                    yield f"data: {chunk_obj_copy.json()}\n\n"
        try:
            async with app.state.client.stream(
                "POST", 
                f"http://localhost:{port}/v1/chat/completions", 
                json=data,
                timeout=STREAM_TIMEOUT
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    error_msg = f"data: {{\"error\":{{\"message\":\"{error_text.decode()}\",\"code\":{response.status_code}}}}}\n\n"
                    logger.error(f"Streaming error: {response.status_code} - {error_text.decode()}")
                    yield error_msg
                    return
                
                async for chunk in response.aiter_bytes():
                    buffer += chunk.decode('utf-8', errors='replace')
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        json_str = _extract_json_data(line)
                        
                        if json_str is None:
                            continue
                            
                        if json_str == '[DONE]':
                            yield 'data: [DONE]\n\n'
                            continue
                        
                        try:
                            chunk_obj = ChatCompletionChunk.parse_raw(json_str)
                            choice = chunk_obj.choices[0]
                            
                            # Handle finish reason - output accumulated tool calls
                            if choice.finish_reason and tool_calls:
                                for tool_call_chunk in _create_tool_call_chunks(tool_calls, chunk_obj):
                                    yield tool_call_chunk

                                yield f"data: [DONE]\n\n"
                                return
                            
                            # Handle tool call deltas
                            if choice.delta.tool_calls:
                                _process_tool_call_delta(choice.delta.tool_calls[0], tool_calls)
                            else:
                                # Regular content chunk
                                yield f"data: {chunk_obj.json()}\n\n"
                                    
                        except Exception as e:
                            logger.error(f"Failed to parse streaming chunk: {e}")
                            # Pass through unparseable data (except ping messages)
                            if not line.strip().startswith(': ping'):
                                yield f"data: {line}\n\n"
                            
                # Process any remaining buffer content
                if buffer.strip():
                    json_str = _extract_json_data(buffer)
                    if json_str and json_str != '[DONE]':
                        try:
                            chunk_obj = ChatCompletionChunk.parse_raw(json_str)
                            yield f"data: {chunk_obj.json()}\n\n"
                        except Exception as e:
                            logger.error(f"Failed to parse trailing chunk: {e}")
                    elif json_str == '[DONE]':
                        yield 'data: [DONE]\n\n'
                            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {{\"error\":{{\"message\":\"{str(e)}\",\"code\":500}}}}\n\n"


# Request Processor
class RequestProcessor:
    """Process requests sequentially using a queue to accommodate single-threaded backends."""
    
    queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
    processing_lock = asyncio.Lock()
    
    # Define which endpoints need to be processed sequentially
    MODEL_ENDPOINTS = {
        "/v1/chat/completions": (ChatCompletionRequest, ServiceHandler.generate_text_response),
        "/v1/embeddings": (EmbeddingRequest, ServiceHandler.generate_embeddings_response),
        "/v1/images/generations": (ImageGenerationRequest, ServiceHandler.generate_image_response),
        "/chat/completions": (ChatCompletionRequest, ServiceHandler.generate_text_response),
        "/embeddings": (EmbeddingRequest, ServiceHandler.generate_embeddings_response),
    }
    
    @staticmethod
    async def _ensure_server_running(request_id: str) -> bool:
        """
        Optimized server state checking and reloading with better error handling.
        """
        try:
            service_info = get_service_info()
            pid = service_info.get("pid")
            
            # Quick validation checks
            if not pid:
                logger.info(f"[{request_id}] No PID found, reloading CryptoModels server...")
                return await crypto_models_manager.reload_ai_server()
        
            # Efficient process existence and status check
            if not psutil.pid_exists(pid):
                logger.info(f"[{request_id}] PID {pid} not found, reloading CryptoModels server...")
                return await crypto_models_manager.reload_ai_server()
            
            # Check process status in a single call
            try:
                process = psutil.Process(pid)
                status = process.status()
                
                # Check for non-running states
                if status in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD, psutil.STATUS_STOPPED]:
                    logger.info(f"[{request_id}] Process {pid} status is {status}, reloading CryptoModels server...")
                    return await crypto_models_manager.reload_ai_server()
                
                # Additional check for processes that might be hung
                try:
                    # Quick responsiveness check - verify process is actually doing something
                    cpu_percent = process.cpu_percent(interval=0.1)  # Non-blocking check
                    memory_info = process.memory_info()
                    
                    logger.debug(f"[{request_id}] CryptoModels server PID {pid} is healthy "
                               f"(status: {status}, memory: {memory_info.rss // 1024 // 1024}MB)")
                    
                except psutil.AccessDenied:
                    # Process exists but we can't get detailed info - assume it's running
                    logger.debug(f"[{request_id}] CryptoModels server PID {pid} running (limited access)")
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                logger.info(f"[{request_id}] Process {pid} not accessible, reloading CryptoModels server...")
                return await crypto_models_manager.reload_ai_server()
            
            return True
            
        except HTTPException:
            logger.info(f"[{request_id}] Service info not available, attempting reload...")
            return await crypto_models_manager.reload_ai_server()
        except Exception as e:
            logger.error(f"[{request_id}] Unexpected error in _ensure_server_running: {str(e)}")
            return False
    
    @staticmethod
    async def process_request(endpoint: str, request_data: Dict[str, Any]):
        """Process a request by adding it to the queue and waiting for the result."""
        request_id = generate_request_id()
        queue_size = RequestProcessor.queue.qsize()
        
        logger.info(f"[{request_id}] Adding request to queue for endpoint {endpoint} (queue size: {queue_size})")
        
        # Update the last request time
        app.state.last_request_time = time.time()
        
        # Check if we need to reload the CryptoModels server
        await RequestProcessor._ensure_server_running(request_id)
        
        start_wait_time = time.time()
        future = asyncio.Future()
        await RequestProcessor.queue.put((endpoint, request_data, future, request_id, start_wait_time))
        
        logger.info(f"[{request_id}] Waiting for result from endpoint {endpoint}")
        result = await future
        
        total_time = time.time() - start_wait_time
        logger.info(f"[{request_id}] Request completed for endpoint {endpoint} (total time: {total_time:.2f}s)")
        
        return result
    
    @staticmethod
    async def process_direct(endpoint: str, request_data: Dict[str, Any]):
        """Process a request directly without queueing for administrative endpoints."""
        request_id = generate_request_id()
        logger.info(f"[{request_id}] Processing direct request for endpoint {endpoint}")
        
        app.state.last_request_time = time.time()
        await RequestProcessor._ensure_server_running(request_id)
        
        start_time = time.time()
        if endpoint in RequestProcessor.MODEL_ENDPOINTS:
            model_cls, handler = RequestProcessor.MODEL_ENDPOINTS[endpoint]
            request_obj = model_cls(**request_data)
            result = await handler(request_obj)
            
            process_time = time.time() - start_time
            logger.info(f"[{request_id}] Direct request completed for endpoint {endpoint} (time: {process_time:.2f}s)")
            
            return result
        else:
            logger.error(f"[{request_id}] Endpoint not found: {endpoint}")
            raise HTTPException(status_code=404, detail="Endpoint not found")
    
    @staticmethod
    async def worker():
        """Worker function to process requests from the queue sequentially."""
        logger.info("Request processor worker started")
        processed_count = 0
        
        while True:
            try:
                endpoint, request_data, future, request_id, start_wait_time = await RequestProcessor.queue.get()
                
                wait_time = time.time() - start_wait_time
                queue_size = RequestProcessor.queue.qsize()
                processed_count += 1
                
                logger.info(f"[{request_id}] Processing request from queue for endpoint {endpoint} "
                           f"(wait time: {wait_time:.2f}s, queue size: {queue_size}, processed: {processed_count})")
                
                async with RequestProcessor.processing_lock:
                    processing_start = time.time()
                    
                    if endpoint in RequestProcessor.MODEL_ENDPOINTS:
                        model_cls, handler = RequestProcessor.MODEL_ENDPOINTS[endpoint]
                        try:
                            request_obj = model_cls(**request_data)
                            result = await handler(request_obj)
                            future.set_result(result)
                            
                            processing_time = time.time() - processing_start
                            total_time = time.time() - start_wait_time
                            
                            logger.info(f"[{request_id}] Completed request for endpoint {endpoint} "
                                       f"(processing: {processing_time:.2f}s, total: {total_time:.2f}s)")
                        except Exception as e:
                            logger.error(f"[{request_id}] Handler error for {endpoint}: {str(e)}")
                            future.set_exception(e)
                    else:
                        logger.error(f"[{request_id}] Endpoint not found: {endpoint}")
                        future.set_exception(HTTPException(status_code=404, detail="Endpoint not found"))
                
                RequestProcessor.queue.task_done()
                
                # Log periodic status about queue health
                if processed_count % 10 == 0:
                    logger.info(f"Queue status: current size={queue_size}, processed={processed_count}")
                
            except asyncio.CancelledError:
                logger.info("Worker task cancelled, exiting")
                break
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")

# Background Tasks
async def unload_checker():
    """
    Optimized unload checker that periodically checks if the CryptoModels server has been idle 
    for too long and unloads it if needed.
    """
    logger.info("Unload checker task started")
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    while True:
        try:
            await asyncio.sleep(UNLOAD_CHECK_INTERVAL)
            
            try:
                service_info = get_service_info()
                
                # Check if we have both PID and last request time
                if "pid" not in service_info:
                    logger.debug("No PID found in service info, skipping unload check")
                    consecutive_errors = 0  # Reset error counter
                    continue
                
                if not hasattr(app.state, "last_request_time"):
                    logger.debug("No last request time available, skipping unload check")
                    consecutive_errors = 0
                    continue
                
                # Calculate idle time
                current_time = time.time()
                idle_time = current_time - app.state.last_request_time
                
                # Log idle status periodically (every 5 minutes)
                if int(idle_time) % 300 == 0:
                    logger.info(f"CryptoModels server idle time: {idle_time:.1f}s (threshold: {IDLE_TIMEOUT}s)")
                
                # Check if server should be unloaded
                if idle_time > IDLE_TIMEOUT:
                    pid = service_info.get("pid")
                    logger.info(f"CryptoModels server (PID: {pid}) has been idle for {idle_time:.2f}s, initiating unload...")
                    
                    # Use the optimized kill method from CryptoModelsManager
                    unload_success = await crypto_models_manager.kill_ai_server()
                    
                    if unload_success:
                        logger.info("CryptoModels server unloaded successfully due to inactivity")
                        # Update last request time to prevent immediate re-unload attempts
                        app.state.last_request_time = current_time
                    else:
                        logger.warning("Failed to unload CryptoModels server")
                
                # Reset error counter on successful check
                consecutive_errors = 0
                
            except HTTPException as e:
                # Service info not available - this is expected when no service is running
                consecutive_errors = 0  # Don't count this as an error
                logger.debug(f"Service info not available (expected when no service running): {e}")
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error in unload checker (attempt {consecutive_errors}/{max_consecutive_errors}): {str(e)}")
                
                # If we have too many consecutive errors, wait longer before next attempt
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Too many consecutive errors in unload checker, extending sleep interval")
                    await asyncio.sleep(UNLOAD_CHECK_INTERVAL * 2)  # Double the sleep time
                    consecutive_errors = 0  # Reset counter after extended wait
            
        except asyncio.CancelledError:
            logger.info("Unload checker task cancelled")
            break
        except Exception as e:
            logger.error(f"Critical error in unload checker task: {str(e)}", exc_info=True)
            # Wait a bit longer before retrying on critical errors
            await asyncio.sleep(UNLOAD_CHECK_INTERVAL * 2)

# Performance monitoring middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware that adds a header with the processing time for the request."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Lifecycle Events
@app.on_event("startup")
async def startup_event():
    """Startup event handler: initialize the HTTP client and start the worker task."""
    # Create an asynchronous HTTP client with optimized connection pooling
    limits = httpx.Limits(
        max_connections=POOL_CONNECTIONS,
        max_keepalive_connections=POOL_CONNECTIONS,
        keepalive_expiry=POOL_KEEPALIVE
    )
    app.state.client = httpx.AsyncClient(
        limits=limits,
        timeout=HTTP_TIMEOUT,
        transport=httpx.AsyncHTTPTransport(
            retries=MAX_RETRIES,
            verify=False  # Disable SSL verification for local connections
        )
    )
    
    # Initialize the last request time
    app.state.last_request_time = time.time()
    
    # Start background tasks
    app.state.worker_task = asyncio.create_task(RequestProcessor.worker())
    app.state.unload_checker_task = asyncio.create_task(unload_checker())
    
    logger.info("Service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Optimized shutdown event with proper resource cleanup and error handling.
    """
    logger.info("Starting application shutdown...")
    shutdown_start_time = time.time()
    
    # Phase 1: Cancel background tasks gracefully
    tasks_to_cancel = []
    task_names = []
    
    for task_attr in ["worker_task", "unload_checker_task"]:
        if hasattr(app.state, task_attr):
            task = getattr(app.state, task_attr)
            if not task.done():
                task_names.append(task_attr)
                tasks_to_cancel.append(task)
                task.cancel()
    
    if tasks_to_cancel:
        logger.info(f"Cancelling background tasks: {', '.join(task_names)}")
        try:
            # Wait for tasks to complete cancellation with timeout
            await asyncio.wait_for(
                asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                timeout=10.0  # 10 second timeout for task cancellation
            )
            logger.info("Background tasks cancelled successfully")
        except asyncio.TimeoutError:
            logger.warning("Background task cancellation timed out, proceeding with shutdown")
        except Exception as e:
            logger.error(f"Error during background task cancellation: {str(e)}")
    
    # Phase 2: Clean up CryptoModels server
    try:
        service_info = get_service_info()
        if "pid" in service_info:
            pid = service_info.get("pid")
            logger.info(f"Terminating CryptoModels server (PID: {pid}) during shutdown...")
            
            # Use the optimized kill method with timeout
            kill_success = await asyncio.wait_for(
                crypto_models_manager.kill_ai_server(),
                timeout=15.0  # 15 second timeout for AI server termination
            )
            
            if kill_success:
                logger.info("CryptoModels server terminated successfully during shutdown")
            else:
                logger.warning("CryptoModels server termination failed during shutdown")
        else:
            logger.debug("No CryptoModels server PID found, skipping termination")
            
    except HTTPException:
        logger.debug("Service info not available during shutdown (expected)")
    except asyncio.TimeoutError:
        logger.error("CryptoModels server termination timed out during shutdown")
    except Exception as e:
        logger.error(f"Error terminating CryptoModels server during shutdown: {str(e)}")
    
    # Phase 3: Close HTTP client connections
    if hasattr(app.state, "client"):
        try:
            logger.info("Closing HTTP client connections...")
            await asyncio.wait_for(app.state.client.aclose(), timeout=5.0)
            logger.info("HTTP client closed successfully")
        except asyncio.TimeoutError:
            logger.warning("HTTP client close timed out")
        except Exception as e:
            logger.error(f"Error closing HTTP client: {str(e)}")
    
    # Phase 4: Clean up any remaining request queue
    if hasattr(RequestProcessor, 'queue'):
        try:
            queue_size = RequestProcessor.queue.qsize()
            if queue_size > 0:
                logger.warning(f"Request queue still has {queue_size} pending requests during shutdown")
                # Cancel any pending futures in the queue
                pending_requests = []
                while not RequestProcessor.queue.empty():
                    try:
                        _, _, future, request_id, _ = RequestProcessor.queue.get_nowait()
                        if not future.done():
                            future.cancel()
                            pending_requests.append(request_id)
                    except asyncio.QueueEmpty:
                        break
                
                if pending_requests:
                    logger.info(f"Cancelled {len(pending_requests)} pending requests")
        except Exception as e:
            logger.error(f"Error cleaning up request queue: {str(e)}")
    
    shutdown_duration = time.time() - shutdown_start_time
    logger.info(f"Application shutdown complete (duration: {shutdown_duration:.2f}s)")

# API Endpoints
@app.get("/health")
@app.get("/v1/health")
async def health():
    """Health check endpoint that bypasses the request queue for immediate response."""
    return {"status": "ok"}

@app.post("/update")
async def update(request: Dict[str, Any]):
    """Update the service information in the CryptoModelsManager."""
    if crypto_models_manager.update_service_info(request):
        return {"status": "ok", "message": "Service info updated successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to update service info")

# Model-based endpoints that use the request queue
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Endpoint for chat completion requests."""
    try:
        service_info = get_service_info()
        if service_info.get("task") == "embed":
            raise HTTPException(status_code=400, detail="Chat completion requests are not supported for embedding models")
    except HTTPException:
        pass  # Service info not available, continue with request
    request_dict = convert_request_to_dict(request)
    return await RequestProcessor.process_request("/chat/completions", request_dict)

@app.post("/embeddings")
async def embeddings(request: EmbeddingRequest):
    """Endpoint for embedding requests."""
    request_dict = convert_request_to_dict(request)
    return await RequestProcessor.process_request("/embeddings", request_dict)

@app.post("/v1/chat/completions")
async def v1_chat_completions(request: ChatCompletionRequest):
    """Endpoint for chat completion requests (v1 API)."""
    try:
        service_info = get_service_info()
        if service_info.get("task") == "embed":
            raise HTTPException(status_code=400, detail="Chat completion requests are not supported for embedding models")
    except HTTPException:
        pass  # Service info not available, continue with request
    request_dict = convert_request_to_dict(request)
    return await RequestProcessor.process_request("/v1/chat/completions", request_dict)

@app.post("/v1/embeddings")
async def v1_embeddings(request: EmbeddingRequest):
    """Endpoint for embedding requests (v1 API)."""
    request_dict = convert_request_to_dict(request)
    return await RequestProcessor.process_request("/v1/embeddings", request_dict)

@app.post("/v1/images/generations")
async def v1_image_generations(request: ImageGenerationRequest):
    """Endpoint for image generation requests (v1 API)."""
    try:
        service_info = get_service_info()
        if service_info.get("task") != "image-generation":
            raise HTTPException(status_code=400, detail="Image generation requests are only supported for image-generation models")
    except HTTPException:
        pass  # Service info not available, continue with request
    request_dict = convert_request_to_dict(request)
    return await RequestProcessor.process_request("/v1/images/generations", request_dict)

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """
    Provides a list of available models, compatible with OpenAI's /v1/models endpoint.
    Currently lists the single loaded model if available.
    """
    try:
        service_info = get_service_info()
    except HTTPException as e:
        # This pattern of handling 503 for missing service_info is consistent
        if e.status_code == 503:
            logger.info("/v1/models: Service information not available. No model loaded or /update not called.")
            return ModelList(data=[])
        logger.error(f"/v1/models: Unexpected HTTPException while fetching service_info: {e.detail}")
        raise # Re-raise other or unexpected HTTPExceptions

    model_hash = service_info.get("hash")
    folder_name_from_info = service_info.get("folder_name") # From Task 1

    if not model_hash:
        logger.warning("/v1/models: No model hash found in service_info, though service_info itself was present. Returning empty list.")
        return ModelList(data=[])

    # Prefer folder_name for user-facing ID, fallback to hash
    model_id = folder_name_from_info if folder_name_from_info else model_hash

    # Construct the ModelCard, similar to how other response objects are built
    # Ensure 'ram' from service_info is correctly parsed
    raw_ram_value = service_info.get("ram")
    parsed_ram_value = None
    if isinstance(raw_ram_value, (int, float)):
        parsed_ram_value = float(raw_ram_value)
    elif isinstance(raw_ram_value, str):
        try:
            # Attempt to parse if it's a string like "8GB", "8gb", or "8"
            parsed_ram_value = float(raw_ram_value.lower().replace("gb", "").strip())
        except ValueError:
            logger.warning(f"/v1/models: Could not parse RAM value '{raw_ram_value}' to float.")

    model_card = ModelCard(
        id=model_id,
        root=model_id, # Consistent with OpenAI for base models
        ram=parsed_ram_value,    # Use 'ram' field, populated with parsed value
        folder_name=folder_name_from_info  # From Task 1
    )

    logger.info(f"/v1/models: Returning information for model ID: {model_id}")
    return ModelList(data=[model_card])