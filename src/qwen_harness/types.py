"""Wire-format types: Gemini-style Content/Parts and the turn event stream.

qwen-code keeps Google's Gemini content format as its *internal* lingua
franca (history entries are `Content { role, parts[] }`), even though the
Qwen models are reached over an OpenAI-compatible API — a converter sits at
the edge (see content_generator.py). This mirrors turn.ts and the
@google/genai types it imports.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Content / Part (the Gemini format used for chat history)
# ---------------------------------------------------------------------------


@dataclass
class FunctionCall:
    """Model asks for a tool. `id` ties the eventual response back to it."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass
class FunctionResponse:
    """Tool result going back to the model (paired by id/name)."""

    name: str
    response: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass
class Part:
    """One part of a message. Exactly one field should be set.

    Upstream this is a union type; a dataclass with optional fields keeps
    the Python readable while staying isomorphic.
    """

    text: str | None = None
    function_call: FunctionCall | None = None
    function_response: FunctionResponse | None = None
    # `thought` parts carry model reasoning; they are shown to the user but
    # stripped from curated history sent back to the model (geminiChat.ts).
    thought: bool = False

    @staticmethod
    def from_text(text: str) -> "Part":
        return Part(text=text)


@dataclass
class Content:
    role: str  # 'user' | 'model'
    parts: list[Part] = field(default_factory=list)


def user_text(text: str) -> Content:
    return Content(role="user", parts=[Part(text=text)])


def model_text(text: str) -> Content:
    return Content(role="model", parts=[Part(text=text)])


def new_call_id() -> str:
    """Mint a tool-call id (upstream: `${name}-${Date.now()}-${randomUUID()}`)."""
    return f"call-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Turn events (ServerGeminiEventType in turn.ts)
# ---------------------------------------------------------------------------


class GeminiEventType(str, Enum):
    """Events yielded while a turn streams.

    Mirrors ServerGeminiEventType. The UI consumes Content/Thought for
    display; the driver loop consumes ToolCallRequest to schedule tools;
    Error/UserCancelled terminate the turn.
    """

    CONTENT = "content"  # incremental assistant text
    THOUGHT = "thought"  # reasoning summaries (subject/description)
    TOOL_CALL_REQUEST = "tool_call_request"
    TOOL_CALL_RESPONSE = "tool_call_response"
    TOOL_CALL_CONFIRMATION = "tool_call_confirmation"
    USER_CANCELLED = "user_cancelled"
    ERROR = "error"
    CHAT_COMPRESSED = "chat_compressed"
    FINISHED = "finished"
    LOOP_DETECTED = "loop_detected"
    MAX_SESSION_TURNS = "max_session_turns"


@dataclass
class ToolCallRequestInfo:
    """Payload of a TOOL_CALL_REQUEST event (turn.ts ToolCallRequestInfo)."""

    call_id: str
    name: str
    args: dict[str, Any]
    is_client_initiated: bool = False
    prompt_id: str = ""


@dataclass
class ToolCallResponseInfo:
    """What a finished tool call contributes back (turn.ts ToolCallResponseInfo)."""

    call_id: str
    response_parts: list[Part]
    result_display: str | None = None
    error: Exception | None = None


@dataclass
class GeminiEvent:
    type: GeminiEventType
    value: Any = None


# ---------------------------------------------------------------------------
# Usage metadata (subset of GenerateContentResponse.usageMetadata)
# ---------------------------------------------------------------------------


@dataclass
class UsageMetadata:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class ModelResponseChunk:
    """One streamed chunk, already converted from the OpenAI wire format.

    Equivalent to a GenerateContentResponse in the upstream stream: some
    combination of text delta, thought delta, and/or completed tool calls,
    plus usage on the final chunk.
    """

    text: str | None = None
    thought: str | None = None
    function_calls: list[FunctionCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: UsageMetadata | None = None
