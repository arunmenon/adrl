"""C-P1A — unit tests for the P1-A live-routing fail-open DECISION LOGIC.

The live flip broke because a utility call pinned to the local rung returned a
502 when LiteLLM was unreachable, instead of transparently falling back to
Anthropic. These tests pin the decision logic that guarantees fail-open, with
NO live server: the two pure seams in proxy.capture_proxy are exercised
directly —

  * ``plan_route``      — which upstream + which body, and the Anthropic
                          fallback body preserved for a locally-pinned call.
  * ``acquire_upstream``— given an *injected* ``send`` coroutine, which attempt
                          serves the client and whether we fell open. Local
                          failure is simulated by making ``send`` raise (conn
                          error / timeout) or return a 5xx — never a real socket.

Run: PYTHONPATH=src .venv/bin/python -m pytest tests/test_hook_failopen.py -q
"""

from __future__ import annotations

import asyncio
import json

import pytest

from proxy.capture_proxy import (
    AcquireResult,
    RoutePlan,
    acquire_upstream,
    attempts_for,
    plan_route,
)

ANTHROPIC = "https://api.anthropic.com"
LOCAL = "http://localhost:4001"
PATH = "/v1/messages"


# ── request fixtures (real wire shapes the discriminator labels) ─────────────
def _utility_body() -> dict:
    # no tools + tiny output budget + <=2 messages -> utility:sidecar (§5.1)
    return {
        "model": "claude-opus-4-20250514",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Pick a title for this chat."}],
    }


def _user_turn_body() -> dict:
    # tools present -> not a sidecar; human text at the tail -> user_turn
    return {
        "model": "claude-opus-4-20250514",
        "max_tokens": 4096,
        "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": "Refactor the auth module."}],
    }


def _bytes(body: dict) -> bytes:
    return json.dumps(body).encode()


def _run(coro):
    return asyncio.run(coro)


# ── send() doubles for acquire_upstream (all network I/O is injected here) ────
class _FakeResp:
    """Minimal stand-in for an aiohttp ClientResponse used by acquire_upstream."""

    def __init__(self, status: int):
        self.status = status
        self.released = False

    async def release(self):
        self.released = True


def _make_send(behaviour):
    """behaviour: dict upstream_base -> ('raise', exc) | ('status', int).
    Records the (upstream, body) of every attempt in ``calls``."""
    calls: list[tuple[str, bytes | None]] = []

    async def send(upstream: str, body):
        calls.append((upstream, body))
        kind, payload = behaviour[upstream]
        if kind == "raise":
            raise payload
        return _FakeResp(payload)

    return send, calls


# ════════════════════════════════════════════════════════════════════════════
# plan_route — the "which upstream + fallback body" decision
# ════════════════════════════════════════════════════════════════════════════
def test_utility_request_pins_local_with_anthropic_fallback_preserved():
    original = _bytes(_utility_body())
    plan = plan_route(
        "POST", PATH, original,
        route_utility=True, anthropic_upstream=ANTHROPIC, local_upstream=LOCAL,
    )
    # primary: the local rung, with the model rewritten to local-small
    assert plan.primary_upstream == LOCAL
    assert json.loads(plan.primary_body)["model"] == "local-small"
    assert plan.routed_label == "utility:sidecar"
    # fallback: the ORIGINAL Anthropic request, byte-for-byte, model intact
    assert plan.fallback_upstream == ANTHROPIC
    assert plan.fallback_body == original
    assert json.loads(plan.fallback_body)["model"] == "claude-opus-4-20250514"


def test_non_utility_request_goes_anthropic_with_no_rewrite_no_fallback():
    original = _bytes(_user_turn_body())
    plan = plan_route(
        "POST", PATH, original,
        route_utility=True, anthropic_upstream=ANTHROPIC, local_upstream=LOCAL,
    )
    assert plan.primary_upstream == ANTHROPIC
    assert plan.primary_body == original            # untouched — no rewrite
    assert plan.fallback_upstream is None            # nothing to fall open to
    assert plan.fallback_body is None
    assert plan.routed_label is None


def test_route_utility_disabled_never_rewrites_even_a_utility_body():
    original = _bytes(_utility_body())
    plan = plan_route(
        "POST", PATH, original,
        route_utility=False, anthropic_upstream=ANTHROPIC, local_upstream=LOCAL,
    )
    assert plan.primary_upstream == ANTHROPIC
    assert plan.primary_body == original
    assert plan.fallback_upstream is None
    assert plan.routed_label is None


def test_empty_body_passes_through():
    plan = plan_route(
        "GET", "/v1/models", b"",
        route_utility=True, anthropic_upstream=ANTHROPIC, local_upstream=LOCAL,
    )
    assert plan.primary_upstream == ANTHROPIC
    assert plan.fallback_upstream is None
    assert plan.routed_label is None


def test_hook_or_parse_error_fails_open_to_anthropic():
    logged: list[str] = []
    plan = plan_route(
        "POST", PATH, b"{not valid json",
        route_utility=True, anthropic_upstream=ANTHROPIC, local_upstream=LOCAL,
        log=logged.append,
    )
    # a parse error must degrade to the plain relay, never raise
    assert plan.primary_upstream == ANTHROPIC
    assert plan.primary_body == b"{not valid json"
    assert plan.fallback_upstream is None
    assert plan.routed_label is None
    assert len(logged) == 1 and "passing through unrouted" in logged[0]


def test_hook_exception_is_swallowed_and_fails_open():
    def _boom(method, path, body):
        raise RuntimeError("hook blew up")

    plan = plan_route(
        "POST", PATH, _bytes(_utility_body()),
        route_utility=True, anthropic_upstream=ANTHROPIC, local_upstream=LOCAL,
        hook_apply=_boom,
    )
    assert plan.primary_upstream == ANTHROPIC
    assert plan.fallback_upstream is None
    assert plan.routed_label is None


# ── attempts_for — the ordered attempt list handle() actually runs ───────────
def test_attempts_for_routed_is_local_then_anthropic():
    original = _bytes(_utility_body())
    plan = plan_route(
        "POST", PATH, original,
        route_utility=True, anthropic_upstream=ANTHROPIC, local_upstream=LOCAL,
    )
    attempts = attempts_for(plan)
    assert [u for u, _ in attempts] == [LOCAL, ANTHROPIC]
    assert attempts[1][1] == original  # fallback carries the original body


def test_attempts_for_unrouted_is_single_anthropic():
    plan = plan_route(
        "POST", PATH, _bytes(_user_turn_body()),
        route_utility=True, anthropic_upstream=ANTHROPIC, local_upstream=LOCAL,
    )
    attempts = attempts_for(plan)
    assert [u for u, _ in attempts] == [ANTHROPIC]


# ════════════════════════════════════════════════════════════════════════════
# acquire_upstream — the fail-open retry decision (send() injected, no server)
# ════════════════════════════════════════════════════════════════════════════
def test_local_success_serves_local_no_fallback():
    send, calls = _make_send({LOCAL: ("status", 200), ANTHROPIC: ("status", 200)})
    plan = plan_route(
        "POST", PATH, _bytes(_utility_body()),
        route_utility=True, anthropic_upstream=ANTHROPIC, local_upstream=LOCAL,
    )
    result: AcquireResult = _run(acquire_upstream(attempts_for(plan), send))
    assert result.response.status == 200
    assert result.used_fallback is False
    assert result.sent_body == plan.primary_body      # the rewritten local body
    assert calls == [(LOCAL, plan.primary_body)]       # Anthropic never touched


def test_local_connection_error_falls_open_to_anthropic_with_original_body():
    original = _bytes(_utility_body())
    plan = plan_route(
        "POST", PATH, original,
        route_utility=True, anthropic_upstream=ANTHROPIC, local_upstream=LOCAL,
    )
    send, calls = _make_send({
        LOCAL: ("raise", ConnectionRefusedError("litellm down")),
        ANTHROPIC: ("status", 200),
    })
    result = _run(acquire_upstream(attempts_for(plan), send))
    # served by Anthropic, with the ORIGINAL body+model — user gets a normal 200
    assert result.response.status == 200
    assert result.used_fallback is True
    assert result.sent_body == original
    assert json.loads(result.sent_body)["model"] == "claude-opus-4-20250514"
    assert [u for u, _ in calls] == [LOCAL, ANTHROPIC]


def test_local_timeout_falls_open_to_anthropic():
    original = _bytes(_utility_body())
    plan = plan_route(
        "POST", PATH, original,
        route_utility=True, anthropic_upstream=ANTHROPIC, local_upstream=LOCAL,
    )
    send, _ = _make_send({
        LOCAL: ("raise", asyncio.TimeoutError()),
        ANTHROPIC: ("status", 200),
    })
    result = _run(acquire_upstream(attempts_for(plan), send))
    assert result.response.status == 200
    assert result.used_fallback is True
    assert result.sent_body == original


def test_local_5xx_releases_and_falls_open_to_anthropic():
    original = _bytes(_utility_body())
    plan = plan_route(
        "POST", PATH, original,
        route_utility=True, anthropic_upstream=ANTHROPIC, local_upstream=LOCAL,
    )
    captured: dict[str, _FakeResp] = {}

    async def send(upstream, body):
        resp = _FakeResp(503 if upstream == LOCAL else 200)
        captured[upstream] = resp
        return resp

    result = _run(acquire_upstream(attempts_for(plan), send))
    assert result.response.status == 200               # Anthropic served it
    assert result.used_fallback is True
    assert result.sent_body == original
    assert captured[LOCAL].released is True             # the 503 was released


def test_fallback_5xx_is_served_not_looped():
    # local 5xx -> Anthropic, and Anthropic ALSO 5xx: relay Anthropic's honest
    # response rather than loop or 502. used_fallback stays True.
    original = _bytes(_utility_body())
    plan = plan_route(
        "POST", PATH, original,
        route_utility=True, anthropic_upstream=ANTHROPIC, local_upstream=LOCAL,
    )
    send, calls = _make_send({LOCAL: ("status", 500), ANTHROPIC: ("status", 500)})
    result = _run(acquire_upstream(attempts_for(plan), send))
    assert result.response.status == 500
    assert result.used_fallback is True
    assert result.sent_body == original
    assert len(calls) == 2                              # exactly one fallback, no loop


def test_non_utility_single_attempt_success_no_fallback():
    original = _bytes(_user_turn_body())
    plan = plan_route(
        "POST", PATH, original,
        route_utility=True, anthropic_upstream=ANTHROPIC, local_upstream=LOCAL,
    )
    send, calls = _make_send({ANTHROPIC: ("status", 200)})
    result = _run(acquire_upstream(attempts_for(plan), send))
    assert result.response.status == 200
    assert result.used_fallback is False
    assert result.sent_body == original
    assert calls == [(ANTHROPIC, original)]


def test_non_utility_anthropic_connection_error_yields_502_semantics():
    # unrouted path: a real Anthropic outage has no fallback -> None response +
    # error, which handle() renders as the 502 (byte-identical to the old proxy).
    original = _bytes(_user_turn_body())
    plan = plan_route(
        "POST", PATH, original,
        route_utility=True, anthropic_upstream=ANTHROPIC, local_upstream=LOCAL,
    )
    boom = ConnectionResetError("anthropic unreachable")
    send, calls = _make_send({ANTHROPIC: ("raise", boom)})
    result = _run(acquire_upstream(attempts_for(plan), send))
    assert result.response is None
    assert result.used_fallback is False
    assert result.error is boom
    assert len(calls) == 1                              # no phantom retry


def test_routed_both_upstreams_unreachable_yields_502_semantics():
    # local down AND Anthropic down: no response, error surfaced -> 502. The
    # fallback was attempted (used_fallback True) before giving up.
    original = _bytes(_utility_body())
    plan = plan_route(
        "POST", PATH, original,
        route_utility=True, anthropic_upstream=ANTHROPIC, local_upstream=LOCAL,
    )
    last = ConnectionRefusedError("anthropic down too")
    send, calls = _make_send({
        LOCAL: ("raise", ConnectionRefusedError("litellm down")),
        ANTHROPIC: ("raise", last),
    })
    result = _run(acquire_upstream(attempts_for(plan), send))
    assert result.response is None
    assert result.used_fallback is True
    assert result.error is last
    assert [u for u, _ in calls] == [LOCAL, ANTHROPIC]
