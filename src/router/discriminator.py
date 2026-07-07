"""B7 (first component) — the call-type discriminator (design doc §5.1).

Labels every wire request with mechanical rules only — no ML, no embeddings,
microseconds. Fingerprints are copied from LIVE captures (B6 evidence,
2026-07-07, cc 2.1.201-202), not guessed:

  wire shape observed                                      -> label
  ------------------------------------------------------------------------
  path /v1/messages/count_tokens                           -> passthrough:count_tokens
  path not /v1/messages*                                   -> passthrough:non_api
  max_tokens <= 2 (cache prewarm ping)                     -> utility:prewarm
  no tools + small max_tokens (<=8192) + <=2 messages      -> utility:sidecar
     (auto-mode classifiers etc. — NOTE: observed on opus/sonnet with
      max_tokens=64, NOT haiku-class as the design assumed)
  system mentions conversation-summarization               -> utility:compaction
  last message carries tool_result block(s)                -> continuation
  anything else with human text at the tail                -> user_turn

Design rule honored: when unsure, label user_turn — the safe mistake
(one extra routing decision) vs. silently keeping a bad route alive.
"""

from __future__ import annotations

from typing import Any

SIDE_CAR_MAX_TOKENS = 8192   # observed: 64 and 8192 on classifier calls
SIDE_CAR_MAX_MSGS = 2
COMPACTION_MARKERS = (
    "summarizing conversations",
    "summarize this conversation",
)


def _system_head(body: dict, n: int = 200) -> str:
    sys = body.get("system", "")
    if isinstance(sys, list):
        return " ".join(
            b.get("text", "") for b in sys if isinstance(b, dict)
        )[:n]
    if isinstance(sys, str):
        return sys[:n]
    return ""


def _last_message(body: dict) -> dict:
    msgs = body.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[-1], dict):
        return msgs[-1]
    return {}


def _has_tool_result(msg: dict) -> bool:
    content = msg.get("content")
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        )
    return False


def classify(method: str, path: str, body: dict[str, Any] | None) -> str:
    """One wire request -> label. Pure function; must stay allocation-light —
    this runs on every request including the 80%+ passthrough flood."""
    if "/count_tokens" in path:
        return "passthrough:count_tokens"
    if not path.startswith("/v1/messages"):
        return "passthrough:non_api"
    if not isinstance(body, dict):
        return "user_turn"  # unparseable -> the safe default

    max_tokens = body.get("max_tokens") or 0
    if max_tokens <= 2:
        return "utility:prewarm"

    tools = body.get("tools") or []
    msgs = body.get("messages") or []
    if not tools and max_tokens <= SIDE_CAR_MAX_TOKENS and len(msgs) <= SIDE_CAR_MAX_MSGS:
        return "utility:sidecar"

    head = _system_head(body).lower()
    if any(m in head for m in COMPACTION_MARKERS):
        return "utility:compaction"

    if _has_tool_result(_last_message(body)):
        return "continuation"

    return "user_turn"


def session_key(body: dict[str, Any] | None) -> str | None:
    """B4 finding: metadata.user_id is JSON carrying a per-session session_id."""
    import json

    if not isinstance(body, dict):
        return None
    uid = (body.get("metadata") or {}).get("user_id")
    if not uid:
        return None
    try:
        return json.loads(uid).get("session_id")
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None


def entrypoint(body: dict[str, Any] | None) -> str:
    """cli = interactive session, sdk-* = headless (-p). Free provenance marker
    observed in the billing header — simulator traffic self-identifies."""
    if isinstance(body, dict):
        head = _system_head(body, 120)
        if "cc_entrypoint=cli" in head:
            return "cli"
        if "cc_entrypoint=sdk" in head:
            return "sdk"
    return "unknown"
