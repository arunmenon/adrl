"""P1-A — pre-call routing hook (the first thing that rewrites a real request).

Pure function: given a wire request, return the model name to route it to, or
None to leave it untouched. Phase 1 scope is deliberately tiny — only utility
housekeeping is rewritten, everything else passes through unchanged. This is
the seam that later grows to carry the full policy engine.

Fail-open contract: any error here must be caught by the caller and the request
passed through unrouted. A routing bug must never fail a user's request.
"""

from __future__ import annotations

from typing import Any

from .discriminator import classify

# Phase-1 rewrites, by discriminator label. Conservative on purpose:
#   utility:light      -> local-small   (titles, topic detection — throwaway)
#   utility:sidecar    -> local-small   (auto-mode classifiers, tiny budget)
# NOT rewritten in Phase 1:
#   utility:compaction -> (cloud)       quality-critical episode memory (§5.1)
#   passthrough:*      -> (unchanged)   count_tokens etc. are not completions
#   continuation       -> (sticky, later)
#   user_turn          -> (policy engine, Phase 2)
UTILITY_REWRITES = {
    "utility": "local-small",          # bare utility == light housekeeping
    "utility:light": "local-small",
    "utility:sidecar": "local-small",
}

# FEASIBILITY GATE (design §5.3, added after live telemetry caught 100% fallback):
# a utility call is labeled by SHAPE (no tools, tiny max_tokens, <=2 messages) but
# its INPUT can be huge — a title/summary sidecar on a long session carries the whole
# conversation (observed: 652KB / ~160k tokens). The small local model can't fit that,
# so LiteLLM's pre-call check rejects it and the call wastes ~45s before failing open.
# Only pin utility calls whose input actually fits the local rung. Byte proxy: ~4
# bytes/token; local-small practical budget ~12k tokens => ~48KB of request body.
LOCAL_INPUT_BYTE_BUDGET = 48 * 1024


def _input_bytes(body: dict[str, Any]) -> int:
    """Cheap size estimate of the request's input (system + messages)."""
    try:
        import json as _json
        return len(_json.dumps(body.get("system", "")).encode()) + \
            len(_json.dumps(body.get("messages", "")).encode())
    except (TypeError, ValueError):
        return 0


def _fits_local(body: dict[str, Any]) -> bool:
    return _input_bytes(body) <= LOCAL_INPUT_BYTE_BUDGET


def route_model(method: str, path: str, body: dict[str, Any] | None) -> str | None:
    """Return the model to route this request to, or None to leave it unchanged."""
    label = classify(method, path, body)
    target = UTILITY_REWRITES.get(label)
    if target is None or not isinstance(body, dict):
        return None
    if not _fits_local(body):
        return None   # too big for the local rung — leave on cloud (feasibility gate)
    return target


def apply(method: str, path: str, body: dict[str, Any] | None) -> tuple[dict | None, str | None]:
    """Rewrite `body['model']` if the hook fires AND the input fits the local rung.
    Returns (possibly-new body, label-or-None). Never mutates the caller's dict on
    the no-op path."""
    label = classify(method, path, body)
    target = UTILITY_REWRITES.get(label)
    if target is None or not isinstance(body, dict) or not _fits_local(body):
        return body, None
    new_body = dict(body)
    new_body["model"] = target
    return new_body, label
