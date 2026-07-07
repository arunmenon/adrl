"""Feasibility gate: utility calls too big for the local rung must NOT route.

Added after live telemetry caught 100% fallback — real utility:sidecar calls
carry the whole conversation as input (small output, huge input), which the
small local model can't fit. Routing them wastes ~45s before failing open.
"""

from router.hook import apply, route_model, LOCAL_INPUT_BYTE_BUDGET

UTIL_SYS = [{"type": "text", "text": "Generate a short title for this conversation"}]


def _util(body_extra):
    b = {"model": "claude-opus-4-8", "max_tokens": 64, "system": UTIL_SYS}
    b.update(body_extra)
    return b


def test_small_utility_routes_local():
    body = _util({"messages": [{"role": "user", "content": "fix the bug"}]})
    assert route_model("POST", "/v1/messages", body) == "local-small"
    new, label = apply("POST", "/v1/messages", body)
    assert label == "utility:sidecar" and new["model"] == "local-small"


def test_big_utility_stays_cloud():
    # a real long-session sidecar: tiny output budget, but the conversation
    # payload dwarfs the local context window.
    huge = "x" * (LOCAL_INPUT_BYTE_BUDGET + 10_000)
    body = _util({"messages": [{"role": "user", "content": huge}]})
    assert route_model("POST", "/v1/messages", body) is None
    new, label = apply("POST", "/v1/messages", body)
    assert label is None and new is body  # untouched -> cloud


def test_boundary():
    # just under the budget routes; just over does not
    under = _util({"messages": [{"role": "user", "content": "y" * 100}]})
    assert route_model("POST", "/v1/messages", under) == "local-small"
