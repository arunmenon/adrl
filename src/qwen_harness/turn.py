"""One model call = one Turn. Port of core/turn.ts.

Turn.run() adapts the raw model stream into typed harness events. The key
contract: a Turn NEVER executes tools. It only *announces* function calls
(ToolCallRequest events + pending_tool_calls); the outer driver (client +
scheduler) executes them and starts a new continuation with the results.
That separation is what makes approval flows and schedulers pluggable.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from .chat import GeminiChat
from .types import (GeminiEvent, GeminiEventType, Part, ToolCallRequestInfo,
                    UsageMetadata, new_call_id)


class Turn:
    def __init__(self, chat: GeminiChat, prompt_id: str):
        self.chat = chat
        self.prompt_id = prompt_id
        self.pending_tool_calls: list[ToolCallRequestInfo] = []
        self.finish_reason: str | None = None
        self.usage: UsageMetadata | None = None

    async def run(self, message_parts: list[Part],
                  cancelled: asyncio.Event | None = None) -> AsyncIterator[GeminiEvent]:
        try:
            async for chunk in self.chat.send_message_stream(message_parts):
                if cancelled is not None and cancelled.is_set():
                    yield GeminiEvent(GeminiEventType.USER_CANCELLED)
                    return
                if chunk.thought:
                    yield GeminiEvent(GeminiEventType.THOUGHT, chunk.thought)
                if chunk.text:
                    yield GeminiEvent(GeminiEventType.CONTENT, chunk.text)
                for call in chunk.function_calls:
                    request = ToolCallRequestInfo(
                        call_id=call.id or new_call_id(),
                        name=call.name or "undefined_tool_name",
                        args=call.args or {},
                        is_client_initiated=False,
                        prompt_id=self.prompt_id,
                    )
                    self.pending_tool_calls.append(request)
                    yield GeminiEvent(GeminiEventType.TOOL_CALL_REQUEST, request)
                if chunk.finish_reason:
                    self.finish_reason = chunk.finish_reason
                    self.usage = chunk.usage
                    yield GeminiEvent(GeminiEventType.FINISHED,
                                      {"reason": chunk.finish_reason,
                                       "usage": chunk.usage})
        except Exception as e:  # errors become events, not exceptions
            yield GeminiEvent(GeminiEventType.ERROR, {"message": str(e)})
