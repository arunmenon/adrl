"""Chat compression. Port of services/chatCompressionService.ts.

Historical note worth studying: gemini-cli (and early qwen-code) compressed
by *splitting* history at 70% of the window — summarize the first 70%, keep
the last 30% verbatim (COMPRESSION_TOKEN_THRESHOLD/COMPRESSION_PRESERVE_
THRESHOLD + findIndexAfterFraction). Current qwen-code replaced that with a
claude-code-style *full-history* compaction: the entire curated history is
summarized into one <state_snapshot> XML document, which reseeds history as

    [ user(summary + resume trailer),
      model('Got it. Thanks for the additional context!') ]

A three-tier threshold ladder (warn < auto < hard) decides when to fire,
and a circuit breaker stops retry storms after 3 consecutive failures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from ..prompts import COMPACT_ACK, RESUME_TRAILER, get_compression_prompt
from ..types import Content, Part

COMPACT_MAX_OUTPUT_TOKENS = 20_000
DEFAULT_PCT = 0.7            # analog of the old COMPRESSION_TOKEN_THRESHOLD
WARN_PCT_OFFSET = 0.1
SUMMARY_RESERVE = 20_000
AUTOCOMPACT_BUFFER = 13_000
WARN_BUFFER = 20_000
HARD_BUFFER = 3_000
MAX_CONSECUTIVE_FAILURES = 3
CHARS_PER_TOKEN = 4          # estimation fallback when no API-reported count


class CompressionStatus(Enum):
    COMPRESSED = 1
    COMPRESSION_FAILED_INFLATED_TOKEN_COUNT = 2
    COMPRESSION_FAILED_TOKEN_COUNT_ERROR = 3
    COMPRESSION_FAILED_EMPTY_SUMMARY = 4
    NOOP = 5


@dataclass
class Thresholds:
    warn: float
    auto: float
    hard: float


@dataclass
class CompressionInfo:
    original_token_count: int
    new_token_count: int
    status: CompressionStatus
    trigger_reason: str = "token_limit"  # 'token_limit' | 'manual'


def compute_thresholds(window: int, pct: float = DEFAULT_PCT) -> Thresholds:
    """Exact computeThresholds() math. `window` = context size minus the
    output-token reservation."""
    effective = max(0, window - SUMMARY_RESERVE)
    auto = max(pct * window, effective - AUTOCOMPACT_BUFFER)
    warn = max(0.0, max((pct - WARN_PCT_OFFSET) * window, auto - WARN_BUFFER))
    hard = min(window, max(effective - HARD_BUFFER, auto + HARD_BUFFER))
    return Thresholds(warn=warn, auto=auto, hard=hard)


def estimate_tokens(contents: list[Content]) -> int:
    """Character-count estimate (upstream falls back to len/4 when the
    provider gives no usage): crude, but it only gates *when* to compact."""
    chars = 0
    for c in contents:
        for p in c.parts:
            if p.text:
                chars += len(p.text)
            if p.function_call:
                chars += len(p.function_call.name) + len(str(p.function_call.args))
            if p.function_response:
                chars += len(str(p.function_response.response))
    return chars // CHARS_PER_TOKEN


_ANALYSIS_RE = re.compile(r"<analysis>.*?(?:</analysis>|$)", re.DOTALL)


def strip_analysis_block(text: str) -> str:
    """Remove the model's <analysis> scratchpad (handles unclosed tags)."""
    return _ANALYSIS_RE.sub("", text).strip()


class ChatCompressionService:
    def __init__(self):
        self.consecutive_failures = 0

    def should_compress(self, estimated_tokens: int, window: int, force: bool = False) -> bool:
        if force:
            return True
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            return False  # circuit breaker: stop retry storms
        return estimated_tokens >= compute_thresholds(window).auto

    async def compress(self, chat, generator, model: str,
                       force: bool = False) -> CompressionInfo:
        history = chat.get_history(curated=True)
        original = estimate_tokens(history)

        # Keep a trailing model functionCall out of the summary input but
        # glued to the new history, so a pending functionResponse still pairs.
        tail: list[Content] = []
        if history and history[-1].role == "model" and any(
                p.function_call for p in history[-1].parts):
            tail = [history[-1]]
            history = history[:-1]

        user_nudge = Content(role="user", parts=[Part(
            text="First, reason in your <analysis> block. Then, produce the "
                 "<state_snapshot> XML.")])
        try:
            summary = await generator.generate_text(
                model=model,
                system_instruction=get_compression_prompt(),
                contents=[*history, user_nudge],
                max_output_tokens=COMPACT_MAX_OUTPUT_TOKENS,
            )
        except Exception:
            self.consecutive_failures += 1
            return CompressionInfo(original, original,
                                   CompressionStatus.COMPRESSION_FAILED_TOKEN_COUNT_ERROR)

        summary = strip_analysis_block(summary or "")
        if not summary:
            self.consecutive_failures += 1
            return CompressionInfo(original, original,
                                   CompressionStatus.COMPRESSION_FAILED_EMPTY_SUMMARY)

        new_history = [
            Content(role="user", parts=[Part(text=f"{summary}\n\n{RESUME_TRAILER}")]),
            Content(role="model", parts=[Part(text=COMPACT_ACK)]),
            *tail,
        ]
        new_count = estimate_tokens(new_history)
        if new_count > original:
            self.consecutive_failures += 1
            return CompressionInfo(original, new_count,
                                   CompressionStatus.COMPRESSION_FAILED_INFLATED_TOKEN_COUNT)

        chat.set_history(new_history)
        self.consecutive_failures = 0
        return CompressionInfo(original, new_count, CompressionStatus.COMPRESSED,
                               trigger_reason="manual" if force else "token_limit")
