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
# Import configuration settings
from json_repair import repair_json
from typing import Dict, Any, List, Optional, Tuple
from eternal_zoo.config import DEFAULT_CONFIG
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from eternal_zoo.manager import EternalZooManager, EternalZooServiceError

# Import schemas from schema.py
from eternal_zoo.schema import (
    Choice,
    Message,
    ModelCard,
    ModelList,
    LoraConfigRequest,
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

# Add CORS middleware to allow localhost only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1"],  # Keep explicit localhost
    allow_origin_regex=r"^https?://.*$",  # Allow any origin (enables remote MindsDB UI)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (preflight will respond 200)
    allow_headers=["*"],  # Allow all headers
)

# Initialize EternalZoo Manager for service management
eternal_zoo_manager = EternalZooManager()


# Performance constants from config
IDLE_TIMEOUT = DEFAULT_CONFIG.performance.IDLE_TIMEOUT
UNLOAD_CHECK_INTERVAL = DEFAULT_CONFIG.performance.UNLOAD_CHECK_INTERVAL
UNLOAD_LOG_INTERVAL = DEFAULT_CONFIG.performance.UNLOAD_LOG_INTERVAL
UNLOAD_MAX_CONSECUTIVE_ERRORS = DEFAULT_CONFIG.performance.UNLOAD_MAX_CONSECUTIVE_ERRORS
UNLOAD_ERROR_SLEEP_MULTIPLIER = DEFAULT_CONFIG.performance.UNLOAD_ERROR_SLEEP_MULTIPLIER
STREAM_CLEANUP_INTERVAL = DEFAULT_CONFIG.performance.STREAM_CLEANUP_INTERVAL
STREAM_CLEANUP_ERROR_SLEEP = DEFAULT_CONFIG.performance.STREAM_CLEANUP_ERROR_SLEEP
STREAM_STALE_TIMEOUT = DEFAULT_CONFIG.performance.STREAM_STALE_TIMEOUT
MODEL_SWITCH_VERIFICATION_DELAY = DEFAULT_CONFIG.performance.MODEL_SWITCH_VERIFICATION_DELAY
MODEL_SWITCH_MAX_RETRIES = DEFAULT_CONFIG.performance.MODEL_SWITCH_MAX_RETRIES
MODEL_SWITCH_STREAM_TIMEOUT = DEFAULT_CONFIG.performance.MODEL_SWITCH_STREAM_TIMEOUT
QUEUE_BACKPRESSURE_TIMEOUT = DEFAULT_CONFIG.performance.QUEUE_BACKPRESSURE_TIMEOUT
PROCESS_CHECK_INTERVAL = DEFAULT_CONFIG.performance.PROCESS_CHECK_INTERVAL
SHUTDOWN_TASK_TIMEOUT = DEFAULT_CONFIG.performance.SHUTDOWN_TASK_TIMEOUT
SHUTDOWN_SERVER_TIMEOUT = DEFAULT_CONFIG.performance.SHUTDOWN_SERVER_TIMEOUT
SHUTDOWN_CLIENT_TIMEOUT = DEFAULT_CONFIG.performance.SHUTDOWN_CLIENT_TIMEOUT
SERVICE_START_TIMEOUT = DEFAULT_CONFIG.performance.SERVICE_START_TIMEOUT
POOL_CONNECTIONS = DEFAULT_CONFIG.performance.POOL_CONNECTIONS
POOL_KEEPALIVE = DEFAULT_CONFIG.performance.POOL_KEEPALIVE
HTTP_TIMEOUT = DEFAULT_CONFIG.performance.HTTP_TIMEOUT
STREAM_TIMEOUT = DEFAULT_CONFIG.performance.STREAM_TIMEOUT
MAX_RETRIES = DEFAULT_CONFIG.performance.MAX_RETRIES
RETRY_DELAY = DEFAULT_CONFIG.performance.RETRY_DELAY
MAX_QUEUE_SIZE = DEFAULT_CONFIG.performance.MAX_QUEUE_SIZE
HEALTH_CHECK_INTERVAL = DEFAULT_CONFIG.performance.HEALTH_CHECK_INTERVAL
STREAM_CHUNK_SIZE = DEFAULT_CONFIG.performance.STREAM_CHUNK_SIZE

# Utility functions
def get_service_info() -> Dict[str, Any]:
    """Get service info from EternalZooManager with error handling."""
    try:
        return eternal_zoo_manager.get_service_info()
    except EternalZooServiceError as e:
        raise HTTPException(status_code=503, detail=str(e))

def convert_request_to_dict(request) -> Dict[str, Any]:
    """Convert request object to dictionary while dropping all None values (Pydantic v1/v2)."""
    if hasattr(request, "model_dump"):
        # Pydantic v2
        return request.model_dump(exclude_none=True)
    # Pydantic v1
    return request.dict(exclude_none=True)

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
        chat_models = eternal_zoo_manager.get_models_by_task(["chat"])            
            
        if len(chat_models) == 0:
            raise HTTPException(status_code=404, detail=f"No chat model found")
                
        model = None
        for chat_model in chat_models:
            if request.model == chat_model["model_id"]:
                model = chat_model
                break

        if model is None:
            model = chat_models[0]
            request.model = model["model_id"]

        port = model.get("port", None)
        if port is None:
            raise HTTPException(status_code=500, detail=f"Model {model.get('model_id', 'unknown')} has no port")
        
        host = model.get("host", "0.0.0.0")

        if request.is_vision_request():
            if not model.get("multimodal", False):
                content = "Unfortunately, I'm not equipped to interpret images at this time. Please provide a text description if possible."
                return ServiceHandler._create_vision_error_response(request, content)
                
        request.clean_messages()
        request.enhance_tool_messages()
        request_dict = convert_request_to_dict(request) 

        if request.stream:
            # For streaming requests, generate a stream ID 
            stream_id = generate_request_id()
            
            logger.debug(f"Creating streaming response for model {request.model} with stream ID {stream_id}")
            
            # Registration happens inside the generator to avoid race conditions
            return StreamingResponse(
                ServiceHandler._stream_generator(port, request_dict, stream_id),
                media_type="text/event-stream"
            )

        # Make a non-streaming API call
        logger.debug(f"Making non-streaming request for model {request.model}")
        response_data = await ServiceHandler._make_api_call(host, port, "/v1/chat/completions", request_dict)
        # reasoning_content = response_data.get("choices", [])[0].get("message", {}).get("reasoning_content", None)
        choices = response_data.get("choices", [])
        if len(choices) > 0:
            reasoning_content = choices[0].get("message", {}).get("reasoning_content", None)
            if reasoning_content:
                reasoning_content = "<think>\n\n" + reasoning_content + "</think>\n\n"

            final_content = None
            content = choices[0].get("message", {}).get("content", None)
            if reasoning_content:
                final_content = reasoning_content
            if content:
                if final_content:
                    final_content = final_content + content
                else:
                    final_content = content

            response_data["choices"][0]["message"]["content"] = final_content
            response_data["choices"][0]["message"]["reasoning_content"] = None

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
        
        embedding_models = eternal_zoo_manager.get_models_by_task(["chat", "embed"])
        if len(embedding_models) == 0:
            raise HTTPException(status_code=404, detail=f"No embedding model found")
        
        model = None
        for embedding_model in embedding_models:
            if request.model == embedding_model["model_id"]:
                model = embedding_model
                break
                
        if model is None:
            model = embedding_models[0]
            request.model = model["model_id"]
        
        port = model.get("port", None)
        if port is None:
            raise HTTPException(status_code=500, detail=f"Model {model.get('model_id', 'unknown')} has no port")
        
        host = model.get("host", "0.0.0.0")
        
        request_dict = convert_request_to_dict(request)
        response_data = await ServiceHandler._make_api_call(host, port, "/v1/embeddings", request_dict)
        return EmbeddingResponse(
            object=response_data.get("object", "list"),
            data=response_data.get("data", []),
            model=request.model
        )
    
    @staticmethod
    async def generate_image_response(request: ImageGenerationRequest):
        """Generate a response for image generation requests."""
        image_generation_models = eternal_zoo_manager.get_models_by_task(["image-generation"])
        if len(image_generation_models) == 0:
            raise HTTPException(status_code=404, detail=f"No image generation model found")
        
        model = None
        for image_generation_model in image_generation_models:
            if request.model == image_generation_model["model_id"]:
                model = image_generation_model
                break

        if model is None:
            model = image_generation_models[0]
            request.model = model["model_id"]
        
        port = model.get("port", None)
        if port is None:
            raise HTTPException(status_code=500, detail=f"Model {model.get('model_id', 'unknown')} has no port")
        
        host = model.get("host", "0.0.0.0")
        request_dict = convert_request_to_dict(request)
        response_data = await ServiceHandler._make_api_call(host, port, "/v1/images/generations", request_dict)
        return ImageGenerationResponse(
            created=response_data.get("created", int(time.time())),
            data=response_data.get("data", [])
        )
    
    @staticmethod
    async def _make_api_call(host: str, port: int, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a non-streaming API call to the specified endpoint and return the JSON response."""
        try:
            response = await app.state.client.post(
                f"http://{host}:{port}{endpoint}", 
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
    async def _stream_generator(port: int, data: Dict[str, Any], stream_id: str):
        """Generator for streaming responses from the service."""
        try:
            # Register stream at the start of actual streaming to avoid race conditions
            await RequestProcessor.register_stream(stream_id)
            logger.debug(f"Starting stream {stream_id}")
            
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
                
                if delta_tool_call.id:
                    tool_calls[tool_call_index]["id"] = delta_tool_call.id
                    
                function = delta_tool_call.function
                if function.name:
                    tool_calls[tool_call_index]["name"] = function.name
                if function.arguments:
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
                        logger.error(f"Failed to create tool call chunk in {stream_id}: {e}")
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
                        logger.error(f"Streaming error for {stream_id}: {response.status_code} - {error_text.decode()}")
                        try:
                            safe_payload = json.dumps(data)[:2000]
                        except Exception:
                            safe_payload = str(data)[:2000]
                        # Explicit stdout print for debug visibility
                        print(f"[STREAM 400] id={stream_id} status={response.status_code} payload={safe_payload} error={error_text.decode()[:500]}")
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
                                chunk_obj = ChatCompletionChunk.model_validate_json(json_str)
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
                                    yield f"data: {chunk_obj.model_dump_json()}\n\n"
                                        
                            except Exception as e:
                                logger.error(f"Failed to parse streaming chunk in {stream_id}: {e}")
                                # # Pass through unparseable data (except ping messages)
                                # if not line.strip().startswith(': ping'):
                                #     yield line
                                
                    # Process any remaining buffer content
                    if buffer.strip():
                        json_str = _extract_json_data(buffer)
                        if json_str and json_str != '[DONE]':
                            try:
                                chunk_obj = ChatCompletionChunk.model_validate_json(json_str)
                                yield f"data: {chunk_obj.model_dump_json()}\n\n"
                            except Exception as e:
                                logger.error(f"Failed to parse trailing chunk in {stream_id}: {e}")
                        elif json_str == '[DONE]':                        
                            yield 'data: [DONE]\n\n'
                            
            except Exception as e:
                logger.error(f"Streaming error for {stream_id}: {e}")
                yield f"data: {{\"error\":{{\"message\":\"{str(e)}\",\"code\":500}}}}\n\n"
                
        except Exception as e:
            logger.error(f"Critical error in stream generator {stream_id}: {e}")
            yield f"data: {{\"error\":{{\"message\":\"{str(e)}\",\"code\":500}}}}\n\n"
        finally:
            # Always unregister the stream when done
            try:
                await RequestProcessor.unregister_stream(stream_id)
                logger.debug(f"Stream {stream_id} completed and unregistered")
            except Exception as e:
                logger.error(f"Error unregistering stream {stream_id}: {e}")


# Request Processor
class RequestProcessor:
    """Process requests sequentially using a queue to accommodate single-threaded backends."""
    
    queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
    processing_lock = asyncio.Lock()
    
    # Track active streams to prevent model switching during streaming
    active_streams = set()
    active_streams_lock = asyncio.Lock()
    stream_timestamps = {}  # Track when streams were registered
    
    # Define which endpoints need to be processed sequentially
    MODEL_ENDPOINTS = {
        "/v1/chat/completions": (ChatCompletionRequest, ServiceHandler.generate_text_response),
        "/chat/completions": (ChatCompletionRequest, ServiceHandler.generate_text_response),
        "/v1/embeddings": (EmbeddingRequest, ServiceHandler.generate_embeddings_response),
        "/embeddings": (EmbeddingRequest, ServiceHandler.generate_embeddings_response),
        "/v1/images/generations": (ImageGenerationRequest, ServiceHandler.generate_image_response),
        "/images/generations": (ImageGenerationRequest, ServiceHandler.generate_image_response),
    }
    
    @staticmethod
    async def register_stream(stream_id: str):
        """Register an active stream to prevent model switching."""
        async with RequestProcessor.active_streams_lock:
            RequestProcessor.active_streams.add(stream_id)
            RequestProcessor.stream_timestamps[stream_id] = time.time()
            logger.debug(f"Registered active stream {stream_id}, total active: {len(RequestProcessor.active_streams)}")
    
    @staticmethod
    async def unregister_stream(stream_id: str):
        """Unregister a completed stream."""
        async with RequestProcessor.active_streams_lock:
            RequestProcessor.active_streams.discard(stream_id)
            RequestProcessor.stream_timestamps.pop(stream_id, None)
            logger.debug(f"Unregistered stream {stream_id}, total active: {len(RequestProcessor.active_streams)}")
    
    @staticmethod
    async def has_active_streams() -> bool:
        """Check if there are any active streams."""
        async with RequestProcessor.active_streams_lock:
            return len(RequestProcessor.active_streams) > 0
    
    @staticmethod
    async def terminate_active_streams():
        """Forcefully terminate all active streams."""
        async with RequestProcessor.active_streams_lock:
            terminated_count = len(RequestProcessor.active_streams)
            RequestProcessor.active_streams.clear()
            RequestProcessor.stream_timestamps.clear()
            logger.warning(f"Force terminated {terminated_count} active streams")
    
    @staticmethod
    async def wait_for_streams_to_complete(timeout: float = MODEL_SWITCH_STREAM_TIMEOUT, force_terminate: bool = False):
        """Wait for all active streams to complete before proceeding."""
        start_time = time.time()
        initial_count = len(RequestProcessor.active_streams)
        
        if initial_count > 0:
            logger.info(f"Waiting for {initial_count} active streams to complete (timeout: {timeout}s)")
        
        check_interval = 0.1
        last_log_time = start_time
        
        while await RequestProcessor.has_active_streams():
            current_time = time.time()
            elapsed = current_time - start_time
            
            if elapsed > timeout:
                remaining_streams = len(RequestProcessor.active_streams)
                if force_terminate:
                    logger.warning(f"Timeout waiting for streams to complete after {elapsed:.1f}s, "
                                  f"force terminating {remaining_streams} active streams")
                    # Log the active stream IDs for debugging
                    async with RequestProcessor.active_streams_lock:
                        active_stream_ids = list(RequestProcessor.active_streams)
                        logger.warning(f"Force terminating stream IDs: {active_stream_ids}")
                    
                    await RequestProcessor.terminate_active_streams()
                    break
                else:
                    logger.error(f"Timeout waiting for streams to complete after {elapsed:.1f}s, "
                                f"{remaining_streams} still active. Refusing to proceed without force_terminate=True")
                    # Log the active stream IDs for debugging
                    async with RequestProcessor.active_streams_lock:
                        active_stream_ids = list(RequestProcessor.active_streams)
                        logger.error(f"Active stream IDs: {active_stream_ids}")
                    return False
            
            # Log progress every 5 seconds
            if current_time - last_log_time >= 5.0:
                remaining_streams = len(RequestProcessor.active_streams)
                logger.info(f"Still waiting for {remaining_streams} streams to complete "
                           f"(elapsed: {elapsed:.1f}s/{timeout}s)")
                last_log_time = current_time
            
            await asyncio.sleep(check_interval)
        
        final_count = len(RequestProcessor.active_streams)
        if initial_count > 0:
            logger.info(f"Stream wait completed. Initial: {initial_count}, Final: {final_count}")
        
        return True
    
    @staticmethod
    async def cleanup_stale_streams():
        """Clean up any stale streams that might be stuck in the active list."""
        current_time = time.time()
        stale_streams = []
        
        async with RequestProcessor.active_streams_lock:
            for stream_id in list(RequestProcessor.active_streams):
                if stream_id in RequestProcessor.stream_timestamps:
                    if current_time - RequestProcessor.stream_timestamps[stream_id] > STREAM_STALE_TIMEOUT:
                        stale_streams.append(stream_id)
                else:
                    # Stream without timestamp is considered stale
                    stale_streams.append(stream_id)
            
            for stream_id in stale_streams:
                RequestProcessor.active_streams.discard(stream_id)
                RequestProcessor.stream_timestamps.pop(stream_id, None)
                logger.warning(f"Cleaned up stale stream {stream_id}")
            
            if stale_streams:
                logger.warning(f"Cleaned up {len(stale_streams)} stale streams")
            elif RequestProcessor.active_streams:
                logger.info(f"No stale streams found, {len(RequestProcessor.active_streams)} active streams are healthy")
    
    @staticmethod
    async def _add_to_queue_with_backpressure(item, timeout: float = QUEUE_BACKPRESSURE_TIMEOUT):
        """Add item to queue with timeout and backpressure handling."""
        try:
            await asyncio.wait_for(
                RequestProcessor.queue.put(item),
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            current_size = RequestProcessor.queue.qsize()
            logger.error(f"Queue full (size: {current_size}), request timed out after {timeout}s")
            raise HTTPException(
                status_code=503, 
                detail=f"Service overloaded. Queue size: {current_size}/{MAX_QUEUE_SIZE}"
            )
    
    @staticmethod
    async def _ensure_model_active_in_queue(model_requested: str, request_id: str) -> bool:
        """
        Ensure the requested model is active within the queue processing context.
        This method is called within the processing lock to ensure atomic model switching.
        
        Args:
            task (str): The task type (chat, embed, image-generation)
            model_requested (str): The model hash requested by the client
            request_id (str): The request ID for logging
            
        Returns:
            bool: True if the model is active or was successfully switched to
        """
        try:
            available_models = eternal_zoo_manager.get_available_models()
            target_model = None
            for model in available_models:
                if model["model_id"] == model_requested:
                    target_model = model
                    break
            
            if target_model is None:
                logger.error(f"[{request_id}] Requested model '{model_requested}' not found in available models")
                return False
            
            if target_model["active"]:
                return True
            
            # Wait for any active streams to complete before switching
            if await RequestProcessor.has_active_streams():
                stream_count = len(RequestProcessor.active_streams)
                logger.info(f"[{request_id}] Waiting for {stream_count} active streams to complete before model switch")
                if not await RequestProcessor.wait_for_streams_to_complete(timeout=MODEL_SWITCH_STREAM_TIMEOUT, force_terminate=True):
                    logger.error(f"[{request_id}] Failed to wait for streams to complete")
                    return False
                logger.info(f"[{request_id}] All streams completed, proceeding with model switch")
            
            # Perform the model switch
            logger.info(f"[{request_id}] Switching to model {model_requested}")
            switch_start_time = time.time()
            
            # Perform the model switch
            success = await eternal_zoo_manager.switch_model(model_requested)
            
            switch_duration = time.time() - switch_start_time
            
            if success:
                logger.info(f"[{request_id}] Successfully switched to {model_requested} "
                           f"(switch time: {switch_duration:.2f}s)")
                return True
            else:
                logger.error(f"[{request_id}] Failed to switch to model {model_requested} "
                           f"(attempted for {switch_duration:.2f}s)")
                return False
                
        except Exception as e:
            logger.error(f"[{request_id}] Error ensuring model active: {str(e)}", exc_info=True)
            return False
    
    @staticmethod
    async def process_request(endpoint: str, request_data: Dict[str, Any]):
        """Process a request by adding it to the queue and waiting for the result."""
        request_id = generate_request_id()
        queue_size = RequestProcessor.queue.qsize()
        
        logger.info(f"[{request_id}] Adding request to queue for endpoint {endpoint} (queue size: {queue_size})")
        
        start_wait_time = time.time()
        future = asyncio.Future()
        queue_item = (endpoint, request_data, future, request_id, start_wait_time)
        await RequestProcessor._add_to_queue_with_backpressure(queue_item)
        
        logger.info(f"[{request_id}] Waiting for result from endpoint {endpoint}")
        result = await future
        
        total_time = time.time() - start_wait_time
        logger.info(f"[{request_id}] Request completed for endpoint {endpoint} (total time: {total_time:.2f}s)")
        
        return result
    
    @staticmethod
    async def worker():
        """Enhanced worker function with better error recovery."""
        logger.info("Request processor worker started")
        processed_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while True:
            try:
                endpoint, request_data, future, request_id, start_wait_time = await RequestProcessor.queue.get()
                
                wait_time = time.time() - start_wait_time
                queue_size = RequestProcessor.queue.qsize()
                processed_count += 1
                tasks = []
                
                logger.info(f"[{request_id}] Processing request from queue for endpoint {endpoint} "
                           f"(wait time: {wait_time:.2f}s, queue size: {queue_size}, processed: {processed_count})")
                
                # Process the request within the lock to ensure sequential execution
                async with RequestProcessor.processing_lock:
                    processing_start = time.time()                    
                    if endpoint in RequestProcessor.MODEL_ENDPOINTS:
                        model_cls, handler = RequestProcessor.MODEL_ENDPOINTS[endpoint]

                        if endpoint == "/v1/chat/completions" or endpoint == "/chat/completions":
                            tasks = ["chat"]
                        elif endpoint == "/v1/embeddings" or endpoint == "/embeddings":
                            tasks = ["chat", "embed"]
                        elif endpoint == "/v1/images/generations" or endpoint == "/images/generations":
                            tasks = ["image-generation"]
                        else:
                            raise HTTPException(status_code=404, detail="Task not found")
                                                
                        try:
                            request_obj = model_cls(**request_data) 
                            # Check if this is a streaming request
                            is_streaming = hasattr(request_obj, 'stream') and request_obj.stream
                            if is_streaming:
                                logger.debug(f"[{request_id}] Processing streaming request for model {request_obj.model}")
                            
                            # Ensure model is active before processing (within the lock)
                            if hasattr(request_obj, 'model') and request_obj.model:
                                logger.debug(f"[{request_id}] Ensuring model {request_obj.model} is active")
                                
                                # Check current active streams before model switching
                                active_stream_count = len(RequestProcessor.active_streams)
                                if active_stream_count > 0:
                                    logger.info(f"[{request_id}] Found {active_stream_count} active streams before model check")
                                
                                success = await RequestProcessor._ensure_model_active_in_queue(request_obj.model, request_id)
                                if not success:
                                    switch_error = getattr(eternal_zoo_manager, 'last_switch_error', None)
                                    if switch_error:
                                        error_msg = f"{switch_error}"
                                    else:
                                        error_msg = f"Model {request_obj.model} is not available or failed to switch or is not a {tasks} model"
                                    try:
                                        safe_payload = json.dumps(request_data)[:2000]
                                    except Exception:
                                        safe_payload = str(request_data)[:2000]
                                    logger.error(f"[{request_id}] 400 due to model switch: {error_msg}. Payload: {safe_payload}")
                                    
                                    # Respond clearly to client
                                    error_body = {
                                        "error": {
                                            "message": error_msg,
                                            "type": "insufficient_resources",
                                            "code": "MODEL_OOM"
                                        }
                                    }
                                    if is_streaming:
                                        async def error_stream():
                                            yield f"data: {json.dumps(error_body)}\n\n"
                                            yield "data: [DONE]\n\n"
                                        future.set_result(StreamingResponse(error_stream(), media_type="text/event-stream"))
                                    else:
                                        future.set_exception(HTTPException(status_code=400, detail=error_body))
                                    continue
                                
                                # Refresh service info after potential model switch
                                logger.debug(f"[{request_id}] Model {request_obj.model} confirmed active, proceeding with request")
                            
                            # Process the request with the updated service info
                            result = await handler(request_obj)
                            future.set_result(result)
                            
                            processing_time = time.time() - processing_start
                            total_time = time.time() - start_wait_time
                            
                            logger.info(f"[{request_id}] Completed request for endpoint {endpoint} "
                                       f"(processing: {processing_time:.2f}s, total: {total_time:.2f}s)")
                        except Exception as e:
                            try:
                                safe_payload = json.dumps(request_data)[:2000]
                            except Exception:
                                safe_payload = str(request_data)[:2000]
                            logger.error(f"[{request_id}] Handler error for {endpoint}: {str(e)} | Payload: {safe_payload}")
                            print(f"[API ERROR] req={request_id} endpoint={endpoint} payload={safe_payload} error={str(e)[:500]}")
                            future.set_exception(e)
                    else:
                        logger.error(f"[{request_id}] Endpoint not found: {endpoint}")
                        future.set_exception(HTTPException(status_code=404, detail="Endpoint not found"))
                
                RequestProcessor.queue.task_done()
                
                # Reset consecutive errors on successful processing
                consecutive_errors = 0
                
                # Log periodic status about queue health
                if processed_count % 10 == 0:
                    active_stream_count = len(RequestProcessor.active_streams)
                    logger.info(f"Queue status: current size={queue_size}, processed={processed_count}, active streams={active_stream_count}")
                
            except asyncio.CancelledError:
                logger.info("Worker task cancelled, exiting")
                break
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Worker error (consecutive: {consecutive_errors}/{max_consecutive_errors}): {str(e)}")
                
                # If we have too many consecutive errors, pause before continuing
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(f"Too many consecutive worker errors, pausing for recovery")
                    await asyncio.sleep(5)  # Brief pause before continuing
                    consecutive_errors = 0  # Reset counter after recovery pause
                    
                    # Clean up any potential issues
                    try:
                        await RequestProcessor.cleanup_stale_streams()
                    except Exception as cleanup_error:
                        logger.error(f"Error during worker recovery cleanup: {cleanup_error}")
                
                # Mark the task as done to prevent queue from getting stuck
                try:
                    RequestProcessor.queue.task_done()
                except ValueError:
                    # task_done() called more times than items in queue
                    pass

async def stream_cleanup_task():
    """Periodic cleanup of stale streams."""
    logger.info("Stream cleanup task started")
    
    while True:
        try:
            await asyncio.sleep(STREAM_CLEANUP_INTERVAL)
            
            # Clean up stale streams
            await RequestProcessor.cleanup_stale_streams()
            
        except asyncio.CancelledError:
            logger.info("Stream cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in stream cleanup task: {str(e)}", exc_info=True)
            # Wait a bit longer before retrying on critical errors
            await asyncio.sleep(STREAM_CLEANUP_ERROR_SLEEP)

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
            verify = True # SSL verification for local connections
        )
    )
    
    # Initialize the last request time
    app.state.last_request_time = time.time()
    
    # Start background tasks
    app.state.worker_task = asyncio.create_task(RequestProcessor.worker())
    # TEMPORARILY DISABLED: Dynamic unload functionality
    # app.state.unload_checker_task = asyncio.create_task(unload_checker())
    app.state.stream_cleanup_task = asyncio.create_task(stream_cleanup_task())
    
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
    
    for task_attr in ["worker_task", "stream_cleanup_task"]:  # TEMPORARILY DISABLED: "unload_checker_task"
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
                timeout=SHUTDOWN_TASK_TIMEOUT
            )
            logger.info("Background tasks cancelled successfully")
        except asyncio.TimeoutError:
            logger.warning("Background task cancellation timed out, proceeding with shutdown")
        except Exception as e:
            logger.error(f"Error during background task cancellation: {str(e)}")
    
    # Phase 2: Clean up EternalZoo server
    try:
        service_info = get_service_info()
        if "pid" in service_info:
            pid = service_info.get("pid")
            logger.info(f"Terminating EternalZoo server (PID: {pid}) during shutdown...")
            
            # Use the optimized kill method with timeout
            kill_success = await asyncio.wait_for(
                eternal_zoo_manager.kill_ai_server(),
                timeout=SHUTDOWN_SERVER_TIMEOUT
            )
            
            if kill_success:
                logger.info("EternalZoo server terminated successfully during shutdown")
            else:
                logger.warning("EternalZoo server termination failed during shutdown")
        else:
            logger.debug("No EternalZoo server PID found, skipping termination")
            
    except HTTPException:
        logger.debug("Service info not available during shutdown (expected)")
    except asyncio.TimeoutError:
        logger.error("EternalZoo server termination timed out during shutdown")
    except Exception as e:
        logger.error(f"Error terminating EternalZoo server during shutdown: {str(e)}")
    
    # Phase 3: Close HTTP client connections
    if hasattr(app.state, "client"):
        try:
            logger.info("Closing HTTP client connections...")
            await asyncio.wait_for(app.state.client.aclose(), timeout=SHUTDOWN_CLIENT_TIMEOUT)
            logger.info("HTTP client closed successfully")
        except asyncio.TimeoutError:
            logger.warning("HTTP client close timed out")
        except Exception as e:
            logger.error(f"Error closing HTTP client: {str(e)}")
    
    # Phase 4: Clean up any remaining request queue and streams
    if hasattr(RequestProcessor, 'queue'):
        try:
            # Clean up any remaining active streams
            await RequestProcessor.cleanup_stale_streams()
            
            queue_size = RequestProcessor.queue.qsize()
            if queue_size > 0:
                logger.warning(f"Request queue still has {queue_size} pending requests during shutdown")
                # Cancel any pending futures in the queue
                pending_requests = []
                while not RequestProcessor.queue.empty():
                    try:
                        _, _, future, request_id, _, _ = RequestProcessor.queue.get_nowait()
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
    
@app.post("/update/lora")
async def update_lora(request: LoraConfigRequest):
    """Update the LoRA for a given model hash."""
    request_dict = convert_request_to_dict(request)
    if eternal_zoo_manager.update_lora(request_dict):
        return {"status": "ok", "message": "LoRA updated successfully"}
    else:
        return {"status": "error", "message": "Failed to update LoRA"}

@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Endpoint for chat completion requests (supports both /v1 and non-/v1)."""
    request_dict = convert_request_to_dict(request)
    return await RequestProcessor.process_request("/v1/chat/completions", request_dict)

@app.post("/embeddings")
@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    """Endpoint for embedding requests (supports both /v1 and non-/v1)."""
    request_dict = convert_request_to_dict(request)
    return await RequestProcessor.process_request("/v1/embeddings", request_dict)

@app.post("/images/generations")
@app.post("/v1/images/generations")
async def image_generations(request: ImageGenerationRequest):
    """Endpoint for image generation requests (supports both /v1 and non-/v1)."""
    request_dict = convert_request_to_dict(request)
    return await RequestProcessor.process_request("/v1/images/generations", request_dict)

@app.get("/models", response_model=ModelList)
@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """
    Provides a list of available models, compatible with OpenAI's /v1/models endpoint.
    Returns all models in multi-model service including main and on-demand models.
    """
    service_info = get_service_info()

    ai_services = service_info.get("ai_services", [])
    models = []

    for ai_service in ai_services:
        model_id = ai_service.get("model_id", "default-chat-model")
        task = ai_service.get("task", "chat")
        is_lora = ai_service.get("is_lora", False)
        multimodal = ai_service.get("multimodal", False)
        lora_config = ai_service.get("lora_config", None)
        context_length = ai_service.get("context_length", None)
        created = ai_service.get("created", int(time.time()))
        owned_by = ai_service.get("owned_by", "user")
        active = ai_service.get("active", False)    

        model_card = ModelCard(
            id=model_id,
            object="model",
            created=created,
            owned_by=owned_by,
            active=active,
            task=task,
            is_lora=bool(is_lora),
            multimodal=bool(multimodal),
            context_length=context_length,
            lora_config=lora_config
        )
        models.append(model_card)

    return ModelList(data=models)

@app.get("/models/test")
@app.get("/v1/models/test")
async def models_test():
    """Lightweight test endpoint that validates `/v1/models` availability.
    Returns 200 with a small payload if listing succeeds.
    """
    try:
        result = await list_models()
        return {"status": "ok", "count": len(result.data)}
    except HTTPException as e:
        # Surface the same error to clients
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))