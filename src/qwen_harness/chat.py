"""Chat history + the single-request send path. Port of core/geminiChat.ts.

The design worth studying is the *two views of history*:

  comprehensive — every turn as it happened, including invalid/empty model
                  responses. This is what the UI replays and what
                  next-speaker inspects.
  curated       — what actually gets sent to the model. Computed on demand:
                  user turns always survive (consecutive ones merge); a run
                  of consecutive model turns survives only if EVERY turn in
                  it is valid, otherwise the whole run is dropped.

The model turn is recorded only after the stream fully completes, as ONE
consolidated Content (thoughts merged, adjacent text parts joined). An
empty/finish-less stream raises and is retried on a small budget; the
failed attempt still lands in comprehensive history, where curation will
skip it. The send path is serialized — one in-flight request per chat.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from .content_generator import ContentGenerator
from .types import Content, ModelResponseChunk, Part

INVALID_STREAM_MAX_ATTEMPTS = 3        # 1 try + 2 retries (upstream maxRetries: 2)
INVALID_STREAM_RETRY_DELAY_S = 2.0


class InvalidStreamError(RuntimeError):
    def __init__(self, kind: str):
        super().__init__(f"Invalid stream: {kind}")
        self.kind = kind  # 'NO_FINISH_REASON' | 'NO_RESPONSE_TEXT'


def is_valid_part(part: Part) -> bool:
    if part.function_call is not None or part.function_response is not None:
        return True
    if part.thought:
        return True
    return bool(part.text)  # non-thought text part must be non-empty


def is_valid_content(content: Content) -> bool:
    return bool(content.parts) and all(is_valid_part(p) for p in content.parts)


def extract_curated_history(comprehensive: list[Content]) -> list[Content]:
    curated: list[Content] = []
    i = 0
    while i < len(comprehensive):
        entry = comprehensive[i]
        if entry.role == "user":
            if curated and curated[-1].role == "user":
                curated[-1] = Content(role="user",
                                      parts=[*curated[-1].parts, *entry.parts])
            else:
                curated.append(Content(role="user", parts=list(entry.parts)))
            i += 1
            continue
        # a run of consecutive model turns lives or dies together
        run: list[Content] = []
        while i < len(comprehensive) and comprehensive[i].role == "model":
            run.append(comprehensive[i])
            i += 1
        if all(is_valid_content(c) for c in run):
            curated.extend(run)
        # else: drop the whole run; its user prompt stays and merges forward
    return curated


class GeminiChat:
    def __init__(self, generator: ContentGenerator, model: str,
                 system_instruction: str, history: list[Content] | None = None,
                 tools: list[dict] | None = None):
        self.generator = generator
        self.model = model
        self.system_instruction = system_instruction
        self.history: list[Content] = list(history or [])
        self.tools = tools or []
        self.last_prompt_token_count = 0
        self.last_output_token_count = 0
        self._send_lock = asyncio.Lock()

    # -- history views -------------------------------------------------------

    def get_history(self, curated: bool = False) -> list[Content]:
        source = extract_curated_history(self.history) if curated else self.history
        return [Content(role=c.role, parts=list(c.parts)) for c in source]

    def set_history(self, history: list[Content]) -> None:
        for c in history:
            if c.role not in ("user", "model"):
                raise ValueError(f"invalid history role: {c.role}")
        self.history = list(history)

    def add_history(self, content: Content) -> None:
        self.history.append(content)

    # -- orphan repair (repairOrphanedToolUseTurns) ---------------------------

    def repair_orphaned_tool_calls(self) -> None:
        """A model functionCall with no adjacent functionResponse (e.g. the
        session died mid-tool) gets a synthetic error response so the wire
        format stays legal — OpenAI rejects dangling tool_calls."""
        repaired: list[Content] = []
        for idx, entry in enumerate(self.history):
            repaired.append(entry)
            calls = [p.function_call for p in entry.parts if p.function_call]
            if entry.role != "model" or not calls:
                continue
            nxt = self.history[idx + 1] if idx + 1 < len(self.history) else None
            answered = {p.function_response.id for p in (nxt.parts if nxt else [])
                        if p.function_response}
            missing = [c for c in calls if c.id not in answered]
            if missing:
                from .types import FunctionResponse
                synthetic = Content(role="user", parts=[
                    Part(function_response=FunctionResponse(
                        name=c.name, id=c.id,
                        response={"error": "Tool call interrupted before a "
                                           "response was recorded."}))
                    for c in missing])
                if nxt is not None and answered:
                    nxt.parts.extend(synthetic.parts)
                else:
                    repaired.append(synthetic)
        self.history = repaired

    # -- the send path --------------------------------------------------------

    async def send_message_stream(self, message_parts: list[Part]
                                  ) -> AsyncIterator[ModelResponseChunk]:
        async with self._send_lock:
            user_content = Content(role="user", parts=message_parts)
            # pushed BEFORE the call; rolled back if the stream never lands
            self.history.append(user_content)
            self.repair_orphaned_tool_calls()

            attempt = 0
            while True:
                attempt += 1
                try:
                    async for chunk in self._process_stream():
                        yield chunk
                    return
                except InvalidStreamError:
                    if attempt >= INVALID_STREAM_MAX_ATTEMPTS:
                        raise
                    await asyncio.sleep(INVALID_STREAM_RETRY_DELAY_S)

    async def _process_stream(self) -> AsyncIterator[ModelResponseChunk]:
        request_history = extract_curated_history(self.history)
        text_acc: list[str] = []
        thought_acc: list[str] = []
        parts: list[Part] = []
        has_tool_call = False
        has_finish = False

        async for chunk in self.generator.generate_content_stream(
                model=self.model, contents=request_history,
                system_instruction=self.system_instruction,
                tools=self.tools):
            if chunk.text:
                text_acc.append(chunk.text)
            if chunk.thought:
                thought_acc.append(chunk.thought)
            for call in chunk.function_calls:
                has_tool_call = True
                parts.append(Part(function_call=call))
            if chunk.finish_reason:
                has_finish = True
            if chunk.usage:
                self.last_prompt_token_count = chunk.usage.prompt_tokens
                self.last_output_token_count = chunk.usage.completion_tokens
            yield chunk

        # validation gate (processStreamResponse): a stream with no tool call
        # must have both a finish reason and some content, or we retry
        if not has_tool_call:
            if not has_finish:
                raise InvalidStreamError("NO_FINISH_REASON")
            if not (text_acc or thought_acc):
                raise InvalidStreamError("NO_RESPONSE_TEXT")

        consolidated: list[Part] = []
        if thought_acc:
            consolidated.append(Part(text="".join(thought_acc), thought=True))
        if text_acc:
            consolidated.append(Part(text="".join(text_acc)))
        consolidated.extend(parts)  # function calls after text
        self.history.append(Content(role="model", parts=consolidated))
