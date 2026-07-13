"""Production LLM difficulty classifier for the live routing hot path.

A drop-in companion to the regex classifier in ``features.py``. It sends ONLY
the isolated human instruction (never the surrounding context) to a small local
model — via the backend port (``backends.for_role("classifier")``; ollama by
default, swappable to lattice / llama.cpp / MLX by config) — and asks for a
one-line difficulty verdict.

**Fail-safe is paramount.** This function sits on the live routing hot path.
ANY failure — connection refused, timeout, non-200, empty or unparseable body,
or any exception whatsoever — returns ``None`` and never raises. The caller is
expected to fall back to the regex ``heuristic_score`` on ``None``. There is no
failure mode in which calling this function is more dangerous than not calling
it.

Tier -> score mapping keeps this drop-in with the T_EASY/T_HARD math in
policy.py (T_EASY=0.35, T_HARD=0.70): trivial=0.15 (clear-easy band),
standard=0.50 (ambiguous middle), hard=0.85 (clear-hard band).

Stdlib only (urllib) — no new dependencies on the hot path.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

DEFAULT_MODEL = "qwen2.5:3b-instruct"
DEFAULT_ENDPOINT = "http://localhost:11434/api/chat"
DEFAULT_TIMEOUT = 5.0
NUM_PREDICT = 60
KEEP_ALIVE = "10m"  # keep the model warm for the next hot-path call

VALID_TIERS = ("trivial", "standard", "hard")

# Tier -> difficulty score. Chosen so the values land cleanly inside the three
# policy bands: trivial below T_EASY, standard between the thresholds, hard
# above T_HARD.
TIER_SCORE: dict[str, float] = {
    "trivial": 0.15,
    "standard": 0.50,
    "hard": 0.85,
}

# The rubric is verbatim the DIFFICULTY RUBRIC used by the bake-off so the live
# path judges by exactly the standard the gold labels were built with.
SYSTEM_PROMPT = """You are a routing classifier for coding-agent instructions. \
Read ONLY the instruction the user gives you and judge how hard it is for a small \
local model to complete correctly. Do not attempt the task itself.

Use these exact tier definitions:
- "trivial": mechanical — commit message, rename, typo, format/lint.
- "standard": a single-file change or a clear bug fix. A competent small model \
can do it.
- "hard": multi-file / architectural / ambiguous / cross-cutting. Needs a \
frontier model.

needs_frontier is true if and only if the tier is "hard".

Reply with STRICT JSON on a single line and nothing else, in exactly this shape:
{"tier":"trivial|standard|hard","needs_frontier":true|false,"reason":"<=12 words"}"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.S)
_TIER_RE = re.compile(r'"?tier"?\s*[:=]\s*"?(trivial|standard|hard)"?', re.I)
_FRONTIER_RE = re.compile(r'"?needs_frontier"?\s*[:=]\s*"?(true|false)"?', re.I)


@dataclass
class LlmVerdict:
    """A single difficulty verdict from the LLM classifier."""

    tier: str
    needs_frontier: bool
    score: float
    reason: str


def _build_payload(text: str, model: str) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": NUM_PREDICT},
        "keep_alive": KEEP_ALIVE,
    }


def _http_post(endpoint: str, payload: dict, timeout: float) -> Optional[str]:
    """POST the payload and return the assistant message content, or None.

    Delegates the transport to the shared ``backends.http_post_json`` (the one
    HTTP implementation, WS0) and unwraps the ollama-native envelope. Kept as a
    small, monkeypatch-friendly seam so tests can inject responses without a
    live server.
    """
    from router.backends import http_post_json

    body = http_post_json(endpoint, payload, timeout)
    if body is None:
        return None
    content = (body.get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    return content


def parse_verdict(content: Optional[str]) -> Optional[LlmVerdict]:
    """Leniently turn a raw model reply into an LlmVerdict, or None.

    Tries strict JSON on the first ``{...}`` block first, then regex-recovers the
    tier and needs_frontier flag independently. If only the tier is recovered,
    needs_frontier is derived as ``tier == "hard"``. Returns None when no valid
    tier can be recovered at all.
    """
    if not content:
        return None

    tier: Optional[str] = None
    frontier: Optional[bool] = None
    reason = ""

    block = _JSON_BLOCK_RE.search(content)
    if block:
        try:
            obj = json.loads(block.group(0))
        except (json.JSONDecodeError, ValueError):
            obj = None
        if isinstance(obj, dict):
            candidate = str(obj.get("tier", "")).strip().lower()
            if candidate in VALID_TIERS:
                tier = candidate
            flag = obj.get("needs_frontier")
            if isinstance(flag, bool):
                frontier = flag
            elif isinstance(flag, str) and flag.strip().lower() in ("true", "false"):
                frontier = flag.strip().lower() == "true"
            raw_reason = obj.get("reason")
            if isinstance(raw_reason, str):
                reason = raw_reason.strip()

    if tier is None:
        match = _TIER_RE.search(content)
        if match:
            tier = match.group(1).lower()
    if frontier is None:
        match = _FRONTIER_RE.search(content)
        if match:
            frontier = match.group(1).lower() == "true"

    if tier is None:
        return None
    # The tier is the canonical verdict. Model output occasionally contradicts
    # its own schema (e.g. hard + false); deriving the boolean prevents that
    # malformed pair from silently routing a hard task locally.
    frontier = tier == "hard"

    return LlmVerdict(
        tier=tier,
        needs_frontier=frontier,
        score=TIER_SCORE[tier],
        reason=reason,
    )


def classify_intent_llm(
    text: str,
    *,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT,
    endpoint: str = DEFAULT_ENDPOINT,
    backend: Optional[object] = None,
    _sender: Optional[Callable[[str, dict, float], Optional[str]]] = None,
) -> Optional[LlmVerdict]:
    """Classify one isolated instruction's difficulty via the local LLM.

    Returns an LlmVerdict on success, or None on ANY failure. Never raises: this
    is safe to call on the live routing hot path, and the caller falls back to
    the regex score whenever it returns None.

    ``backend`` is the WS0 port path: pass any ``GenerationBackend`` (usually
    ``backends.for_role("classifier")``) and the framework choice comes from
    config, not code. The legacy ``endpoint``/``model`` kwargs keep the direct
    ollama path working unchanged; ``_sender`` is a test seam for injecting the
    HTTP transport.
    """
    try:
        if not text or not text.strip():
            return None
        if backend is not None and _sender is None:
            content = backend.chat(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": text}],
                {"temperature": 0, "num_predict": NUM_PREDICT,
                 "keep_alive": KEEP_ALIVE},
            )
            return parse_verdict(content)
        sender = _sender or _http_post
        payload = _build_payload(text, model)
        content = sender(endpoint, payload, timeout)
        return parse_verdict(content)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            OSError, ValueError, json.JSONDecodeError, Exception):
        # Total fail-safe: nothing that goes wrong here is worth crashing the
        # router for. Fall back to the regex path by returning None.
        return None
