"""OpenAI-compatible content generator. Port of core/openaiContentGenerator/
(pipeline.ts + converter.ts) and utils/retry.ts.

The architectural point to study: qwen-code is *Gemini-native inside* —
history, tools, and events all use Gemini Content/Part types (inherited
from gemini-cli) — and converts to the OpenAI chat.completions wire format
only here, at the very edge. Qwen models are served over OpenAI-compatible
endpoints (DashScope), so every request crosses this converter:

  Gemini Content            OpenAI message
  ----------------          ------------------------------------------
  systemInstruction    ->   {role: 'system', content}
  role 'model'         ->   {role: 'assistant', ...}
  anything else        ->   {role: 'user', ...}
  part.functionCall    ->   assistant tool_calls[{id, function:{name,
                            arguments: json string}}]
  part.functionResponse->   {role: 'tool', tool_call_id, content}
  part.thought         ->   'reasoning_content' (non-standard field)

Streaming tool calls arrive as fragments (index-keyed argument deltas);
StreamingToolCallParser reassembles them and they are only emitted when
finish_reason arrives.

Defaults mirror upstream: request timeout 120s, stream-idle watchdog 240s,
retry on 429/5xx with exponential backoff (init 1.5s, cap 30s, jitter 0.3,
Retry-After honored).
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import aiohttp

from .types import (Content, FunctionCall, ModelResponseChunk, UsageMetadata,
                    new_call_id)

DEFAULT_TIMEOUT_S = 120
DEFAULT_STREAM_IDLE_TIMEOUT_S = 240
RETRY_MAX_ATTEMPTS = 5          # upstream default is 7 at the generic layer
RETRY_INITIAL_DELAY_S = 1.5
RETRY_MAX_DELAY_S = 30.0
RETRY_JITTER = 0.3
RETRYABLE_STATUS = {429, 500, 502, 503, 504, 529}

# DashScope endpoints upstream defaults to (we default to a local LiteLLM
# rung instead — same wire format, observable in this repo's capture proxy).
DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_LOCAL_BASE_URL = "http://localhost:4001/v1"


class RetryableApiError(Exception):
    def __init__(self, status: int, message: str, retry_after: float | None = None):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# Gemini -> OpenAI request conversion (converter.ts, request direction)
# ---------------------------------------------------------------------------


def contents_to_openai_messages(contents: list[Content],
                                system_instruction: str | None = None) -> list[dict]:
    messages: list[dict] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})

    for content in contents:
        role = "assistant" if content.role == "model" else "user"
        texts: list[str] = []
        tool_calls: list[dict] = []
        tool_messages: list[dict] = []
        reasoning: list[str] = []

        for i, part in enumerate(content.parts):
            if part.function_call is not None:
                fc = part.function_call
                tool_calls.append({
                    "id": fc.id or f"call_{i}",
                    "type": "function",
                    "function": {"name": fc.name, "arguments": json.dumps(fc.args)},
                })
            elif part.function_response is not None:
                fr = part.function_response
                resp = fr.response or {}
                text = resp.get("output") or resp.get("error") or json.dumps(resp)
                # OpenAI requires every tool call to get a tool response,
                # even an empty one
                tool_messages.append({"role": "tool", "tool_call_id": fr.id or "",
                                      "content": text or ""})
            elif part.thought and part.text:
                reasoning.append(part.text)
            elif part.text is not None:
                texts.append(part.text)

        if tool_messages:
            messages.extend(tool_messages)
            if texts:  # rare: user text alongside tool results
                messages.append({"role": "user", "content": "\n".join(texts)})
            continue

        msg: dict[str, Any] = {"role": role}
        if tool_calls:
            msg["tool_calls"] = tool_calls
            msg["content"] = "\n".join(texts) if texts else None
        else:
            msg["content"] = "\n".join(texts)
        if reasoning and role == "assistant":
            msg["reasoning_content"] = "\n".join(reasoning)
            if msg.get("content") is None and not tool_calls:
                msg["content"] = ""  # some servers reject null content
        messages.append(msg)

    return _merge_consecutive_assistant(messages)


def _merge_consecutive_assistant(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        prev = out[-1] if out else None
        if (prev and m["role"] == "assistant" and prev["role"] == "assistant"
                and not prev.get("tool_calls") and not m.get("tool_calls")):
            prev["content"] = f"{prev.get('content') or ''}\n{m.get('content') or ''}"
            continue
        out.append(m)
    return out


def tools_to_openai(declarations: list[dict]) -> list[dict]:
    return [{"type": "function",
             "function": {"name": d["name"], "description": d.get("description", ""),
                          "parameters": d.get("parameters", {"type": "object", "properties": {}})}}
            for d in declarations]


# ---------------------------------------------------------------------------
# Streaming tool-call reassembly (StreamingToolCallParser)
# ---------------------------------------------------------------------------


class StreamingToolCallParser:
    """delta.tool_calls arrive as index-keyed fragments: the first fragment
    carries id+name, later ones append to `arguments`. Calls are emitted
    only once finish_reason arrives; truncated JSON downgrades the finish
    reason to 'length' so the caller can treat it as MAX_TOKENS."""

    def __init__(self):
        self._calls: dict[int, dict] = {}

    def add_delta(self, tool_call_deltas: list[dict]) -> None:
        for d in tool_call_deltas:
            slot = self._calls.setdefault(d.get("index", 0),
                                          {"id": None, "name": "", "arguments": ""})
            if d.get("id"):
                slot["id"] = d["id"]
            fn = d.get("function") or {}
            if fn.get("name"):
                slot["name"] += fn["name"]
            if fn.get("arguments"):
                slot["arguments"] += fn["arguments"]

    def has_incomplete_calls(self) -> bool:
        return any(not _json_parses(c["arguments"] or "{}") for c in self._calls.values())

    def emit(self) -> list[FunctionCall]:
        calls = []
        for _, c in sorted(self._calls.items()):
            args = _safe_json_parse(c["arguments"] or "{}")
            calls.append(FunctionCall(name=c["name"] or "undefined_tool_name",
                                      args=args, id=c["id"] or new_call_id()))
        self._calls.clear()
        return calls


def _json_parses(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _safe_json_parse(s: str) -> dict:
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------


@dataclass
class GeneratorConfig:
    base_url: str = field(default_factory=lambda: os.environ.get(
        "OPENAI_BASE_URL", DEFAULT_LOCAL_BASE_URL))
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", "sk-local"))
    timeout_s: float = DEFAULT_TIMEOUT_S
    stream_idle_timeout_s: float = DEFAULT_STREAM_IDLE_TIMEOUT_S
    max_retries: int = RETRY_MAX_ATTEMPTS
    extra_headers: dict[str, str] = field(default_factory=dict)


class ContentGenerator:
    """ContentGenerator interface (contentGenerator.ts), OpenAI flavor."""

    def __init__(self, config: GeneratorConfig | None = None):
        self.config = config or GeneratorConfig()

    # -- retry wrapper (utils/retry.ts) ------------------------------------

    async def _with_backoff(self, fn):
        delay = RETRY_INITIAL_DELAY_S
        for attempt in range(1, self.config.max_retries + 1):
            try:
                return await fn()
            except RetryableApiError as e:
                if attempt == self.config.max_retries:
                    raise
                wait = e.retry_after if e.retry_after is not None else delay
                wait *= 1 + random.uniform(-RETRY_JITTER, RETRY_JITTER)
                await asyncio.sleep(max(0.0, wait))
                if e.retry_after is None:
                    delay = min(delay * 2, RETRY_MAX_DELAY_S)
                else:
                    delay = RETRY_INITIAL_DELAY_S  # Retry-After resets the ladder

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json", **self.config.extra_headers}

    # -- streaming completions ---------------------------------------------

    async def generate_content_stream(
        self, model: str, contents: list[Content],
        system_instruction: str | None = None,
        tools: list[dict] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[ModelResponseChunk]:
        body: dict[str, Any] = {
            "model": model,
            "messages": contents_to_openai_messages(contents, system_instruction),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            body["tools"] = tools_to_openai(tools)  # never send tools: []
        if max_output_tokens:
            body["max_tokens"] = max_output_tokens
        if temperature is not None:
            body["temperature"] = temperature

        async def attempt() -> list[ModelResponseChunk] | None:
            # First chunk decides retryability; after that we stream through.
            return await self._open_stream(body)

        async for chunk in await self._with_backoff(attempt):
            yield chunk

    async def _open_stream(self, body: dict) -> AsyncIterator[ModelResponseChunk]:
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=self.config.timeout_s,
                                          sock_read=self.config.stream_idle_timeout_s))
        try:
            resp = await session.post(f"{self.config.base_url}/chat/completions",
                                      json=body, headers=self._headers())
        except Exception:
            await session.close()
            raise
        if resp.status in RETRYABLE_STATUS:
            retry_after = resp.headers.get("Retry-After")
            text = (await resp.text())[:512]
            await session.close()
            raise RetryableApiError(resp.status, text,
                                    float(retry_after) if retry_after else None)
        if resp.status >= 400:
            text = (await resp.text())[:512]
            await session.close()
            raise RuntimeError(f"[API Error: HTTP {resp.status}: {text}]")
        ctype = resp.headers.get("Content-Type", "")
        if "text/event-stream" not in ctype:
            # NonSSEResponseError: a gateway handed us an HTML block page
            text = (await resp.text())[:512]
            await session.close()
            raise RuntimeError(f"[API Error: expected SSE, got {ctype}: {text}]")
        return self._parse_sse(session, resp)

    async def _parse_sse(self, session: aiohttp.ClientSession,
                         resp: aiohttp.ClientResponse) -> AsyncIterator[ModelResponseChunk]:
        parser = StreamingToolCallParser()
        finish_reason: str | None = None
        usage: UsageMetadata | None = None
        try:
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if payload.get("usage"):
                    u = payload["usage"]
                    usage = UsageMetadata(
                        prompt_tokens=u.get("prompt_tokens", 0),
                        completion_tokens=u.get("completion_tokens", 0),
                        total_tokens=u.get("total_tokens", 0),
                        cached_tokens=(u.get("prompt_tokens_details") or {}).get(
                            "cached_tokens", 0),
                    )
                choices = payload.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}

                if delta.get("tool_calls"):
                    parser.add_delta(delta["tool_calls"])
                text = delta.get("content")
                thought = delta.get("reasoning_content") or delta.get("reasoning")
                if text or thought:
                    yield ModelResponseChunk(text=text or None, thought=thought or None)

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

            calls = parser.emit() if finish_reason else []
            if finish_reason and parser.has_incomplete_calls():
                finish_reason = "length"  # truncated mid-JSON -> MAX_TOKENS
            reason_map = {"stop": "STOP", "length": "MAX_TOKENS",
                          "content_filter": "SAFETY", "tool_calls": "STOP",
                          "function_call": "STOP"}
            yield ModelResponseChunk(
                function_calls=calls,
                finish_reason=reason_map.get(finish_reason or "stop", "STOP"),
                usage=usage,
            )
        finally:
            await session.close()

    # -- one-shot side queries (compression, next-speaker) ------------------

    async def generate_text(self, model: str, contents: list[Content],
                            system_instruction: str | None = None,
                            max_output_tokens: int | None = None) -> str:
        parts: list[str] = []
        async for chunk in self.generate_content_stream(
                model=model, contents=contents,
                system_instruction=system_instruction,
                max_output_tokens=max_output_tokens):
            if chunk.text:
                parts.append(chunk.text)
        return "".join(parts)

    async def generate_json(self, model: str, contents: list[Content],
                            schema: dict) -> dict:
        instruction = ("Respond ONLY with a JSON object matching this schema, "
                       "no prose, no code fences:\n" + json.dumps(schema))
        raw = await self.generate_text(model=model, contents=contents,
                                       system_instruction=instruction)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`\n")
            raw = raw[raw.index("{"):] if "{" in raw else raw
        return json.loads(raw)

    async def count_tokens(self, contents: list[Content]) -> int:
        """Never hits the network upstream either: char/4 estimate."""
        return sum(len(p.text or "") for c in contents for p in c.parts) // 4
