"""WS2 UNIT A — the LiveRouter pure decision layer.

Injects a real DictSessionStore plus fake memory/classifier so no live service
is touched. Verifies the DISPATCH MAP end to end, sticky continuations, the
fail-safe passthrough on any internal error, and the memory recording seam.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from router.live_router import (
    CHEAP_CLOUD_MODEL,
    LOCAL_CODE_MODEL,
    LOCAL_SMALL_MODEL,
    LiveRouter,
    RoutePlan,
)
from router.state import DictSessionStore


# ── helpers / fakes ──────────────────────────────────────────────────────────

ANTHROPIC = "https://api.anthropic.com"
LITELLM = "http://localhost:4001"


def _user_body(text: str, *, session_id: str = "s1", model: str = "claude-opus-4",
               extra_messages=None):
    """A minimal Anthropic /v1/messages body the discriminator labels user_turn:
    has tools, a real max_tokens, and human text at the tail."""
    messages = list(extra_messages or [])
    messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
    return {
        "model": model,
        "max_tokens": 4096,
        "tools": [{"name": "Edit"}],
        "metadata": {"user_id": json.dumps({"session_id": session_id})},
        "messages": messages,
    }


def _continuation_body(session_id: str = "s1"):
    """Last message carries a tool_result -> discriminator labels it continuation."""
    return {
        "model": "claude-opus-4",
        "max_tokens": 4096,
        "tools": [{"name": "Edit"}],
        "metadata": {"user_id": json.dumps({"session_id": session_id})},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "refactor everything"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1",
                                               "name": "Edit", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                          "content": "ok"}]},
        ],
    }


def _utility_body():
    """No tools, small max_tokens, <=2 messages -> utility:sidecar."""
    return {
        "model": "claude-opus-4",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "classify this"}],
    }


@dataclass
class _Verdict:
    tier: str
    needs_frontier: bool


def _classifier_frontier(text):
    return _Verdict(tier="hard", needs_frontier=True)


def _classifier_local(text):
    return _Verdict(tier="standard", needs_frontier=False)


class _FakeRecorder:
    """Stands in for RoutingRecorder: records calls, hands back a fixed route_id."""

    def __init__(self, route_id="rid-123"):
        self.route_id = route_id
        self.new_turns = []
        self.records = []

    def new_turn(self, session_id, **kwargs):
        self.new_turns.append(session_id)
        return 0

    def record(self, instruction_text, features, route, **kwargs):
        self.records.append((instruction_text, features, route, kwargs))
        return self.route_id


# ── user_turn dispatch ───────────────────────────────────────────────────────

def test_user_turn_easy_routes_local_with_anthropic_fallback():
    router = LiveRouter(store=DictSessionStore())
    # "fix a typo" -> trivial verb class -> clear-easy -> rung local
    plan = router.plan("POST", "/v1/messages", _user_body("fix the typo in the readme"))
    assert plan.primary_model == LOCAL_CODE_MODEL
    assert plan.primary_upstream == LITELLM
    assert plan.fallback_model is None          # original body on fail-open
    assert plan.fallback_upstream == ANTHROPIC
    assert router._normalize_rung(plan.rung) == "local"


def test_user_turn_hard_routes_frontier_passthrough():
    router = LiveRouter(store=DictSessionStore())
    # "refactor" -> hard verb class -> clear-hard -> rung frontier
    plan = router.plan("POST", "/v1/messages",
                       _user_body("refactor the entire authentication architecture"))
    assert plan.rung == "frontier"
    assert plan.primary_model is None           # original model, no rewrite
    assert plan.primary_upstream == ANTHROPIC
    assert plan.fallback_upstream is None        # no fail-open needed on Anthropic


def test_user_turn_middle_classifier_frontier_routes_cheap_is_not_used():
    # The middle band with a classifier voting frontier -> frontier passthrough.
    router = LiveRouter(store=DictSessionStore(), classifier=_classifier_frontier)
    plan = router.plan("POST", "/v1/messages", _user_body("wire up the new endpoint"))
    assert plan.rung == "frontier"
    assert plan.layer == "classifier"
    assert plan.primary_upstream == ANTHROPIC


def test_user_turn_middle_classifier_local_routes_local():
    router = LiveRouter(store=DictSessionStore(), classifier=_classifier_local)
    plan = router.plan("POST", "/v1/messages", _user_body("wire up the new endpoint"))
    assert router._normalize_rung(plan.rung) == "local"
    assert plan.primary_model == LOCAL_CODE_MODEL
    assert plan.primary_upstream == LITELLM
    assert plan.layer == "classifier"


def test_cheap_cloud_rung_rewrites_to_haiku_on_anthropic():
    # Exercise the dispatch for a cheap_cloud rung directly (policy rarely emits
    # it unassisted, but the escalation store can carry it on a continuation).
    router = LiveRouter(store=DictSessionStore())
    plan = router._dispatch("cheap_cloud", label="user_turn", layer="test")
    assert plan.primary_model == CHEAP_CLOUD_MODEL
    assert plan.primary_upstream == ANTHROPIC     # subscription auth, NOT LiteLLM
    # fail-open to the ORIGINAL model on Anthropic (review finding): a haiku
    # rewrite the subscription can't access retries the un-routed request.
    assert plan.fallback_upstream == ANTHROPIC and plan.fallback_model is None


# ── continuation sticks to the session route ─────────────────────────────────

def test_continuation_sticks_to_escalated_session_route():
    store = DictSessionStore()
    # Session already escalated to frontier (as the escalation controller would).
    store.set_route("s1", "frontier")
    router = LiveRouter(store=store)
    plan = router.plan("POST", "/v1/messages", _continuation_body("s1"))
    assert plan.label == "continuation"
    assert plan.rung == "frontier"
    assert plan.primary_upstream == ANTHROPIC
    assert plan.primary_model is None


def test_continuation_default_local_route_dispatches_local():
    store = DictSessionStore()  # fresh session -> default route "local"
    router = LiveRouter(store=store)
    plan = router.plan("POST", "/v1/messages", _continuation_body("s-new"))
    assert plan.label == "continuation"
    assert plan.primary_model == LOCAL_CODE_MODEL
    assert plan.primary_upstream == LITELLM
    assert plan.fallback_upstream == ANTHROPIC


def test_continuation_does_not_rerun_route_turn():
    # A continuation must not call the classifier (route_turn is skipped).
    calls = []

    def _spy_classifier(text):
        calls.append(text)
        return _Verdict(tier="hard", needs_frontier=True)

    store = DictSessionStore()
    store.set_route("s1", "cheap-cloud")
    router = LiveRouter(store=store, classifier=_spy_classifier)
    plan = router.plan("POST", "/v1/messages", _continuation_body("s1"))
    assert calls == []                            # classifier never consulted
    assert plan.primary_model == CHEAP_CLOUD_MODEL


# ── utility + passthrough ────────────────────────────────────────────────────

def test_utility_routes_local_small_with_fallback():
    router = LiveRouter(store=DictSessionStore())
    plan = router.plan("POST", "/v1/messages", _utility_body())
    assert plan.label.startswith("utility")
    assert plan.primary_model == LOCAL_SMALL_MODEL
    assert plan.primary_upstream == LITELLM
    assert plan.fallback_upstream == ANTHROPIC


def test_passthrough_label_is_untouched():
    router = LiveRouter(store=DictSessionStore())
    plan = router.plan("POST", "/v1/messages/count_tokens", {"messages": []})
    assert plan.rung == "passthrough"
    assert plan.primary_model is None
    assert plan.primary_upstream == ANTHROPIC
    assert plan.fallback_upstream is None


def test_non_api_path_passthrough():
    router = LiveRouter(store=DictSessionStore())
    plan = router.plan("GET", "/health", {})
    assert plan.rung == "passthrough"
    assert plan.primary_upstream == ANTHROPIC


# ── fail-safe: a raising classifier / route still yields a passthrough plan ───

def test_raising_classifier_yields_passthrough_plan():
    def _boom(text):
        raise RuntimeError("classifier exploded")

    router = LiveRouter(store=DictSessionStore(), classifier=_boom)
    # route_turn itself swallows classifier None-vs-raise? No — the classifier
    # here raises, so _plan_user_turn raises, and plan() must fail-safe.
    plan = router.plan("POST", "/v1/messages", _user_body("wire up the new endpoint"))
    assert isinstance(plan, RoutePlan)
    assert plan.primary_upstream == ANTHROPIC
    assert plan.primary_model is None
    assert plan.rung == "passthrough"


def test_broken_store_yields_passthrough_plan():
    class _BrokenStore:
        def get_session(self, sid):
            raise RuntimeError("store down")

    router = LiveRouter(store=_BrokenStore())
    plan = router.plan("POST", "/v1/messages", _user_body("fix the typo"))
    assert plan.rung == "passthrough"
    assert plan.primary_upstream == ANTHROPIC


# ── recording seam ───────────────────────────────────────────────────────────

def test_user_turn_records_decision_with_route_id():
    recorder = _FakeRecorder(route_id="rid-abc")
    router = LiveRouter(store=DictSessionStore(), memory=recorder)
    plan = router.plan("POST", "/v1/messages", _user_body("fix the typo", session_id="sX"))
    assert plan.route_id == "rid-abc"
    assert recorder.new_turns == ["sX"]
    assert len(recorder.records) == 1
    text, features, route, kwargs = recorder.records[0]
    assert text == "fix the typo"                 # RAW text handed to the recorder
    assert kwargs["session_id"] == "sX"
    assert "decision_ms" in kwargs


def test_recorder_failure_does_not_break_plan():
    class _BadRecorder:
        def new_turn(self, sid, **kw):
            return 0

        def record(self, *a, **kw):
            raise RuntimeError("memory down")

    router = LiveRouter(store=DictSessionStore(), memory=_BadRecorder())
    plan = router.plan("POST", "/v1/messages", _user_body("fix the typo"))
    # routing still succeeds; only the route_id is missing
    assert plan.primary_model == LOCAL_CODE_MODEL
    assert plan.route_id is None


def test_continuation_does_not_record():
    recorder = _FakeRecorder()
    store = DictSessionStore()
    store.set_route("s1", "frontier")
    router = LiveRouter(store=store, memory=recorder)
    router.plan("POST", "/v1/messages", _continuation_body("s1"))
    assert recorder.records == []                 # only user_turns are recorded
    assert recorder.new_turns == []


# ── build_forward_body ───────────────────────────────────────────────────────

def test_build_forward_body_rewrites_model():
    router = LiveRouter(store=DictSessionStore())
    body = {"model": "claude-opus-4", "max_tokens": 10}
    out = json.loads(router.build_forward_body(body, "local-code"))
    assert out["model"] == "local-code"
    assert out["max_tokens"] == 10
    assert body["model"] == "claude-opus-4"       # original not mutated


def test_build_forward_body_none_leaves_body_unchanged():
    router = LiveRouter(store=DictSessionStore())
    body = {"model": "claude-opus-4", "max_tokens": 10}
    out = json.loads(router.build_forward_body(body, None))
    assert out == body


# ── session key fallback ─────────────────────────────────────────────────────

def test_missing_session_key_uses_stable_fallback():
    router = LiveRouter(store=DictSessionStore())
    body = _user_body("fix the typo")
    body.pop("metadata")                          # no metadata.user_id
    body["system"] = "you are a coding agent working on repo X"
    plan_one = router.plan("POST", "/v1/messages", body)
    plan_two = router.plan("POST", "/v1/messages", body)
    # deterministic dispatch across identical bodies (same fallback sid)
    assert plan_one.rung == plan_two.rung
    assert plan_one.primary_model == LOCAL_CODE_MODEL


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_user_turn_persists_sticky_route_for_continuation():
    """Regression (review critical finding): a hard user_turn -> frontier must
    PERSIST its route so the follow-up continuation sticks to frontier — not the
    default local. No manual store.set_route pre-seed, unlike the other tests."""
    store = DictSessionStore()
    router = LiveRouter(store=store)                       # heuristic decides
    up = router.plan("POST", "/v1/messages",
                     _user_body("refactor the auth module across the entire codebase",
                                session_id="chain"))
    assert up.rung == "frontier"                           # hard -> frontier
    assert store.get_session("chain").route == "frontier"  # <-- the fix: persisted
    # the continuation reads that sticky route and dispatches frontier too
    cp = router.plan("POST", "/v1/messages", _continuation_body("chain"))
    assert cp.layer == "continuation" and cp.rung == "frontier"


def test_user_turn_increments_turn_counter():
    """Regression (review major): the live path advances turn_count so ledger
    turn_index orders within a session (was permanently 0)."""
    store = DictSessionStore()
    router = LiveRouter(store=store)
    assert store.get_session("t").turn_count == 0
    router.plan("POST", "/v1/messages", _user_body("fix the bug", session_id="t"))
    assert store.get_session("t").turn_count == 1
    router.plan("POST", "/v1/messages", _user_body("add a test", session_id="t"))
    assert store.get_session("t").turn_count == 2
