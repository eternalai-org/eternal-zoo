"""
Schema definitions for API requests and responses following OpenAI's API standard.
"""

import re
from typing_extensions import Literal
from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Dict, Optional, Union, Any, ClassVar
import time
import uuid
from enum import Enum
# Import configuration settings
from crypto_models.config import config

# Precompile regex patterns for better performance
UNICODE_BOX_PATTERN = re.compile(r'\\u25[0-9a-fA-F]{2}')

# Common models used in both streaming and non-streaming contexts
class ImageUrl(BaseModel):
    """
    Represents an image URL in a message.
    """
    url: str = Field(..., description="URL of the image")

    @validator("url")
    def validate_url(cls, v: str) -> str:
        """Validate that the URL is properly formatted."""
        if not v.startswith(("http://", "https://", "data:")):
            raise ValueError("URL must start with http://, https://, or data:")
        return v

class VisionContentItem(BaseModel):
    """
    Represents a single content item in a message (text or image).
    """
    type: Literal["text", "image_url"] = Field(..., description="Type of content")
    text: Optional[str] = Field(None, description="Text content if type is text")
    image_url: Optional[ImageUrl] = Field(None, description="Image URL if type is image_url")
        

class FunctionCall(BaseModel):
    """
    Represents a function call in a message.
    """
    arguments: str = Field(..., description="JSON string of function arguments")
    name: str = Field(..., description="Name of the function to call")

    @validator("arguments")
    def validate_arguments(cls, v: str) -> str:
        """Validate that arguments is a valid JSON string."""
        try:
            import json
            # Don't store the parsed result, just validate it's valid JSON
            json.loads(v)
            return v
        except (json.JSONDecodeError, TypeError):
            raise ValueError("arguments must be a valid JSON string")

class ChatCompletionMessageToolCall(BaseModel):
    """
    Represents a tool call in a message.
    """
    id: str = Field(..., description="Unique identifier for the tool call")
    function: FunctionCall = Field(..., description="Function call details")
    type: Literal["function"] = Field("function", description="Type of tool call")

class Message(BaseModel):
    """
    Represents a message in a chat completion.
    """
    content: Optional[Union[str, List[VisionContentItem]]] = Field(None, description="Message content")
    refusal: Optional[str] = Field(None, description="Refusal message if any")
    role: Literal["system", "user", "assistant", "tool"] = Field(..., description="Role of the message sender")
    function_call: Optional[FunctionCall] = Field(None, description="Function call if any")
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = Field(None, description="Tool calls if any")
    
# Common request base for both streaming and non-streaming
class ChatCompletionRequestBase(BaseModel):
    """
    Base model for chat completion requests.
    """
    model: str = Field(config.model.DEFAULT_CHAT_MODEL, description="Model to use for completion")
    messages: List[Message] = Field(..., description="List of messages in the conversation")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="Available tools for the model")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Tool choice configuration")
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate")
    seed: Optional[int] = Field(None, description="Random seed for generation")
    
    @validator("messages")
    def check_messages_not_empty(cls, v: List[Message]) -> List[Message]:
        """Ensure that the messages list is not empty and validate message structure."""
        if not v:
            raise ValueError("messages cannot be empty")
        
        msg_count = len(v)
        if msg_count > 50:  # Local AI's limit is 50 messages
            raise ValueError("message history too long")
        
        # Pre-define set for O(1) lookup instead of creating it in loop    
        valid_roles = frozenset({"user", "assistant", "system", "tool"})
        
        # Use any() with generator expression for early exit on invalid role
        if any(msg.role not in valid_roles for msg in v):
            # Only find the specific invalid role if validation fails
            invalid_roles = [msg.role for msg in v if msg.role not in valid_roles]
            raise ValueError(f"invalid role(s): {invalid_roles[0]}")
                
        return v

    @validator("max_tokens")
    def validate_max_tokens(cls, v: Optional[int]) -> Optional[int]:
        """Validate max_tokens - convert -1 to None and ensure positive values."""
        if v is None or v == -1:
            return None
        if v < 1:
            raise ValueError("max_tokens must be greater than or equal to 1")
        return v

    def is_vision_request(self) -> bool:
        """
        Determines if the request is a vision request according to the OpenAI API standard.

        In the OpenAI API, vision requests are characterized by message content items that include
        a 'type' field set to 'image_url' and an accompanying 'image_url' field that may contain 
        either an external URL or a base64-encoded data URI.

        Returns:
            bool: True if the final message contains any valid vision content with type 'image_url',
                False otherwise.

        Edge cases handled:
            - Self.messages is None or empty
            - Final message's content is not a list
            - Items in content list missing 'type' or 'image_url'
            - Items with 'type' not equal to 'image_url'
            - Items with 'image_url' as None or empty
        """
        if not self.messages or not isinstance(self.messages, list):
            return False

        final_message = self.messages[-1]
        content = getattr(final_message, "content", None)

        if not isinstance(content, list):
            return False

        for item in content:
            if not isinstance(item, dict) and not hasattr(item, "__dict__"):
                continue
            item_type = getattr(item, "type", None)
            image_url = getattr(item, "image_url", None)

            if item_type == "image_url" and image_url:
                return True

        return False
    
class ChatTemplateKwargs(BaseModel):
    """
    Represents the arguments for a chat template.
    """
    enable_thinking: bool = Field(True, description="Whether to enable thinking mode")

# Non-streaming request and response
class ChatCompletionRequest(ChatCompletionRequestBase):
    """
    Model for non-streaming chat completion requests.
    """
    stream: bool = Field(False, description="Whether to stream the response")
    chat_template_kwargs: Optional[ChatTemplateKwargs] = Field(ChatTemplateKwargs(), description="Chat template kwargs")

    def clean_messages(self) -> None:
        """Fix the messages list to ensure proper formatting and ordering."""
        def clean_special_box_text(input_text: str) -> str:
            return UNICODE_BOX_PATTERN.sub('', input_text).strip()
        
        # Use a single pass with list comprehensions for better performance
        system_messages = []
        non_system_messages = []
        
        for message in self.messages:
            # Handle content cleaning
            if message.content is None:
                message.content = ""
            elif isinstance(message.content, str):
                message.content = clean_special_box_text(message.content)
            elif isinstance(message.content, list):
                # More efficient iteration with enumerate
                for item in message.content:
                    if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                        item["text"] = clean_special_box_text(item["text"])
            
            # Sort messages in the same loop to avoid second iteration
            if message.role == "system":
                system_messages.append(message)
            else:
                non_system_messages.append(message)
            
        self.messages = system_messages + non_system_messages

    def enhance_tool_messages(self) -> None:
        """
        Fixes tool messages by converting list content to merged string content.
        
        This ensures tool messages have proper string content for processing.
        """
        # In-place modification instead of creating new list
        for message in self.messages:
            if message.role == "tool" and isinstance(message.content, list):
                # Use list comprehension for better performance
                merged_content = [
                    content_item.text 
                    for content_item in message.content 
                    if hasattr(content_item, 'type') and content_item.type == "text" and hasattr(content_item, 'text') and content_item.text
                ]
                
                # Join all content with newlines for better formatting
                message.content = "\n".join(merged_content) if merged_content else ""

class Choice(BaseModel):
    """
    Represents a choice in a chat completion response.
    """
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"] = Field(..., description="Reason for completion")
    index: int = Field(..., ge=0, description="Index of the choice")
    message: Message = Field(..., description="Generated message")

class ChatCompletionResponse(BaseModel):
    """
    Represents a complete chat completion response.
    """
    id: str = Field(..., description="Unique identifier for the completion")
    object: Literal["chat.completion"] = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[Choice] = Field(..., description="Generated choices")

# Embedding models
class EmbeddingRequest(BaseModel):
    """
    Model for embedding requests.
    """
    model: str = Field(config.model.DEFAULT_EMBED_MODEL, description="Model to use for embedding")
    input: List[str] = Field(..., min_items=1, description="List of text inputs for embedding")

    @validator("input")
    def validate_input(cls, v: List[str]) -> List[str]:
        """Validate that input texts are not empty."""
        if not all(text.strip() for text in v):
            raise ValueError("Input texts cannot be empty")
        return v

class Embedding(BaseModel):
    """
    Represents an embedding object in an embedding response.
    """
    embedding: List[float] = Field(..., description="The embedding vector")
    index: int = Field(..., ge=0, description="The index of the embedding in the list")
    object: str = Field("embedding", description="The object type")

class EmbeddingResponse(BaseModel):
    """
    Represents an embedding response.
    """
    object: str = Field("list", description="Object type")
    data: List[Embedding] = Field(..., description="List of embeddings")
    model: str = Field(..., description="Model used for embedding")

class ChoiceDeltaFunctionCall(BaseModel):
    """
    Represents a function call delta in a streaming response.
    """
    arguments: Optional[str] = Field(None, description="Arguments for the function call delta.")
    name: Optional[str] = Field(None, description="Name of the function in the delta.")

class ChoiceDeltaToolCall(BaseModel):
    """
    Represents a tool call delta in a streaming response.
    """
    index: Optional[int] = Field(None, description="Index of the tool call delta.")
    id: Optional[str] = Field(None, description="ID of the tool call delta.")
    function: Optional[ChoiceDeltaFunctionCall] = Field(None, description="Function call details in the delta.")
    type: Optional[str] = Field(None, description="Type of the tool call delta.")

class Delta(BaseModel):
    """
    Represents a delta in a streaming response.
    """
    content: Optional[str] = Field(None, description="Content of the delta.")
    function_call: Optional[ChoiceDeltaFunctionCall] = Field(None, description="Function call delta, if any.")
    refusal: Optional[str] = Field(None, description="Refusal reason, if any.")
    role: Optional[Literal["system", "user", "assistant", "tool"]] = Field(None, description="Role in the delta.")
    tool_calls: Optional[List[ChoiceDeltaToolCall]] = Field(None, description="List of tool call deltas, if any.")
    reasoning_content: Optional[str] = Field(None, description="Reasoning content, if any.")

class StreamingChoice(BaseModel):
    """
    Represents a choice in a streaming response.
    """
    delta: Delta = Field(..., description="The delta for this streaming choice.")
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = Field(None, description="The reason for finishing, if any.")
    index: int = Field(..., description="The index of the streaming choice.")
    
class ChatCompletionChunk(BaseModel):
    """
    Represents a chunk in a streaming chat completion response.
    """
    id: str = Field(..., description="The chunk ID.")
    choices: List[StreamingChoice] = Field(..., description="List of streaming choices in the chunk.")
    created: int = Field(..., description="The creation timestamp of the chunk.")
    model: str = Field(..., description="The model used for the chunk.")
    object: Literal["chat.completion.chunk"] = Field(..., description="The object type, always 'chat.completion.chunk'.")
    system_fingerprint: Optional[str] = Field(None, description="System fingerprint for the completion")

# OpenAI-compatible /v1/models endpoint schemas
class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{uuid.uuid4().hex}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "user"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = Field(default_factory=lambda: [ModelPermission()])
    ram: Optional[float] = None
    folder_name: Optional[str] = None

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = [] 



class ImageSize(str, Enum):
    SMALL_SQUARE = "256x256"
    MEDIUM_SQUARE = "512x512"
    LARGE_SQUARE = "1024x1024"
    HORIZONTAL = "1792x1024"
    VERTICAL = "1024x1792"
    COSMOS_SIZE = "1280x704"

class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"

class ImageGenerationRequest(BaseModel):
    """Request schema for OpenAI-compatible image generation API"""
    prompt: str = Field(..., description="A text description of the desired image(s). The maximum length is 1000 characters.", max_length=1000)
    model: Optional[str] = Field(default="cosmos-predict2-2b-text2image", description="The model to use for image generation")
    size: Optional[ImageSize] = Field(default=ImageSize.COSMOS_SIZE, description="The size of the generated images")
    negative_prompt: Optional[str] = Field(None, description="The negative prompt to generate the image from")
    steps: Optional[int] = Field(default=50, ge=1, le=50, description="The number of inference steps (1-50)")
    priority: Optional[Priority] = Field(default=Priority.NORMAL, description="Task priority in queue")
    async_mode: Optional[bool] = Field(default=False, description="Whether to process asynchronously")

class ImageData(BaseModel):
    """Individual image data in the response"""
    url: Optional[str] = Field(None, description="The URL of the generated image, if response_format is url")
    b64_json: Optional[str] = Field(None, description="The base64-encoded JSON of the generated image, if response_format is b64_json")

class ImageGenerationResponse(BaseModel):
    """Response schema for OpenAI-compatible image generation API"""
    created: int = Field(..., description="The Unix timestamp (in seconds) when the image was created")
    data: List[ImageData] = Field(..., description="List of generated images")

class ImageGenerationError(BaseModel):
    """Error response schema"""
    code: str = Field(..., description="Error code (e.g., 'contentFilter', 'generation_error', 'queue_full')")
    message: str = Field(..., description="Human-readable error message")
    type: Optional[str] = Field(None, description="Error type")

class ImageGenerationErrorResponse(BaseModel):
    """Error response wrapper"""
    created: int = Field(..., description="The Unix timestamp (in seconds) when the error occurred")
    error: ImageGenerationError = Field(..., description="Error details")