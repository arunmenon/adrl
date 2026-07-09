"""Unit tests for router.memory_facade (WS1, Unit A).

Covers the facade's three core promises:
  * PRIVACY GATE — pinned/secret turns get sha None, no embedding allowed,
    and instruction text never appears anywhere in the produced event;
  * CHAIN FAILOVER — a hostile provider that raises from every method is
    absorbed; NullProvider (terminal) answers and the router never sees an
    exception;
  * REPORT — build() is pure and renders provider chain + stats.

Plus the RoutingRecorder three-call flow (new_turn / record / attach) and the
mandatory nomic embedding prefixes.
"""

from __future__ import annotations

import hashlib

from router.features import extract
from router.memory_facade import (
    DOCUMENT_PREFIX,
    QUERY_PREFIX,
    RouterMemory,
    RoutingRecorder,
    build,
)
from router.memory_null import NullProvider
from router.memory_ports import OutcomeEvent
from router.policy import Route


# ── test doubles ──────────────────────────────────────────────────────────────


class SpyProvider:
    """Records every call; always succeeds. Duck-typed against the port."""

    def __init__(self):
        self.decisions = []          # (DecisionEvent, embedding)
        self.outcomes = []           # (route_id, OutcomeEvent)
        self.finalized = []          # (session_id, prev_interrupted, prev_retried)

    def record_decision(self, decision, embedding=None):
        self.decisions.append((decision, embedding))
        return decision.route_id

    def attach_outcome(self, route_id, outcome):
        self.outcomes.append((route_id, outcome))
        return True

    def finalize_turn(self, session_id, *, prev_interrupted, prev_retried):
        self.finalized.append((session_id, prev_interrupted, prev_retried))
        return 1

    def similar_turns(self, embedding, k=12):
        return []

    def stats(self):
        return {"provider": "spy", "decisions": len(self.decisions)}

    def health(self):
        return True


class HostileProvider:
    """Raises from every port method — the failover fixture."""

    def _boom(self, *args, **kwargs):
        raise RuntimeError("provider is on fire")

    record_decision = _boom
    attach_outcome = _boom
    finalize_turn = _boom
    similar_turns = _boom
    stats = _boom
    health = _boom


class StubEmbedder:
    """WS0 EmbeddingBackend double capturing the texts it was asked to embed."""

    def __init__(self):
        self.seen = []

    def embed(self, texts):
        self.seen.extend(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]


def _route(**overrides) -> Route:
    defaults = dict(rung="local", cascade=True, layer="heuristic",
                    score=0.3, reason="easy")
    defaults.update(overrides)
    return Route(**defaults)


# ── privacy gate ──────────────────────────────────────────────────────────────


def test_gate_hashes_normal_turns_and_allows_embedding():
    memory = RouterMemory([NullProvider()])
    text = "write a parser for the config format"
    features = extract(text, context_tokens=3000)
    event, embedding_allowed = memory.make_decision_event(
        text, features, _route(), session_id="s-1", turn_index=2)
    assert embedding_allowed is True
    assert event.instr_sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert event.route_id  # minted
    assert event.rung == "local" and event.layer == "heuristic"
    assert event.propensity == "heuristic"  # defaults to the deciding layer


def test_gate_pinned_turn_gets_no_sha_and_no_embedding():
    memory = RouterMemory([NullProvider()])
    text = "summarize the medical records in this repo"
    features = extract(text, privacy_pinned=True)
    route = _route(layer="gate:privacy", pinned=True)
    event, embedding_allowed = memory.make_decision_event(
        text, features, route, session_id="s-2")
    assert embedding_allowed is False
    assert event.instr_sha256 is None


def test_gate_secret_extra_flag_is_private_too():
    memory = RouterMemory([NullProvider()])
    text = "rotate the API keys"
    features = extract(text)
    features.extra["secret"] = True
    event, embedding_allowed = memory.make_decision_event(
        text, features, _route(), session_id="s-3")
    assert embedding_allowed is False
    assert event.instr_sha256 is None


def test_features_json_never_contains_instruction_text():
    memory = RouterMemory([NullProvider()])
    text = "a very distinctive secret-sauce instruction xyzzy42"
    features = extract(text)
    event, _ = memory.make_decision_event(
        text, features, _route(), session_id="s-4")
    assert "xyzzy42" not in event.features_json
    assert "instruction_text" not in event.features_json
    assert '"verb_class"' in event.features_json  # features themselves survive


def test_record_drops_embedding_for_sha_less_decisions():
    spy = SpyProvider()
    memory = RouterMemory([spy])
    features = extract("private thing", privacy_pinned=True)
    event, _ = memory.make_decision_event(
        "private thing", features, _route(pinned=True), session_id="s-5")
    # A hostile/buggy caller passes a vector anyway — the facade drops it.
    memory.record_decision(event, [0.5] * 4)
    assert spy.decisions[0][1] is None


# ── chain failover ────────────────────────────────────────────────────────────


def test_hostile_provider_is_absorbed_by_null_terminal():
    memory = RouterMemory([HostileProvider(), NullProvider()])
    features = extract("fix the bug")
    event, _ = memory.make_decision_event(
        "fix the bug", features, _route(), session_id="s-6")

    assert memory.record_decision(event) == event.route_id
    assert memory.attach_outcome(event.route_id, OutcomeEvent()) is True
    assert memory.finalize_turn("s-6") == 0
    assert memory.similar_turns([0.0] * 8) == []
    assert memory.stats() == {"provider": "null"}
    assert memory.health() is True
    assert memory.active_provider() == "NullProvider"


def test_facade_methods_survive_garbage_without_raising():
    memory = RouterMemory([HostileProvider(), NullProvider()])
    assert memory.attach_outcome("", OutcomeEvent()) is False
    assert memory.finalize_turn(None) == 0
    assert memory.similar_turns(None) == []
    assert memory.record_decision(None) is None


def test_default_chain_ends_in_null_and_is_healthy():
    memory = RouterMemory()   # default: [Sqlite-if-importable, Null]
    assert isinstance(memory.providers[-1], NullProvider)
    assert memory.health() is True


# ── embedding prefixes (nomic contract) ──────────────────────────────────────


def test_embedding_prefixes_are_mandatory():
    embedder = StubEmbedder()
    memory = RouterMemory([NullProvider()], embedder=embedder)
    assert memory.embed_document("hello") == [0.1, 0.2, 0.3]
    assert memory.embed_query("hello") == [0.1, 0.2, 0.3]
    assert embedder.seen == [DOCUMENT_PREFIX + "hello", QUERY_PREFIX + "hello"]


def test_embedding_fail_safe_when_backend_dead():
    class DeadEmbedder:
        def embed(self, texts):
            return None

    memory = RouterMemory([NullProvider()], embedder=DeadEmbedder())
    assert memory.embed_document("hello") is None


# ── RoutingRecorder (the WS2 three-call flow) ────────────────────────────────


def test_recorder_three_call_flow():
    spy = SpyProvider()
    recorder = RoutingRecorder(RouterMemory([spy], embedder=StubEmbedder()))
    text = "add a retry to the fetch helper"
    features = extract(text)

    # turn N: finalize previous (none yet), record, attach
    assert recorder.new_turn("sess-A") == 1  # spy finalizes 1 trivially
    route_id = recorder.record(text, features, _route(), session_id="sess-A",
                               turn_index=0, source="simulator")
    assert route_id == recorder.active_route_id("sess-A")
    assert recorder.attach("sess-A", OutcomeEvent(status="closed_turn")) is True
    assert spy.outcomes[0][0] == route_id

    # turn N+1 in the same session: a NEW route_id is minted and remembered
    route_id_2 = recorder.record("next thing", extract("next thing"),
                                 _route(), session_id="sess-A", turn_index=1)
    assert route_id_2 != route_id
    assert recorder.active_route_id("sess-A") == route_id_2

    # embedding was computed (spy saw a vector) for these unpinned turns
    assert spy.decisions[0][1] == [0.1, 0.2, 0.3]


def test_recorder_pinned_turn_records_without_embedding():
    spy = SpyProvider()
    recorder = RoutingRecorder(RouterMemory([spy], embedder=StubEmbedder()))
    features = extract("private", privacy_pinned=True)
    recorder.record("private", features, _route(pinned=True),
                    session_id="sess-B")
    decision, embedding = spy.decisions[0]
    assert decision.instr_sha256 is None
    assert embedding is None


def test_recorder_attach_without_active_route_is_false():
    recorder = RoutingRecorder(RouterMemory([NullProvider()]))
    assert recorder.attach("never-seen-session", OutcomeEvent()) is False


# ── report ────────────────────────────────────────────────────────────────────


def test_report_builds_markdown():
    memory = RouterMemory([HostileProvider(), NullProvider()])
    report = build(memory)
    assert report.startswith("# Router memory — facade report")
    assert "NullProvider" in report and "HostileProvider" in report
    assert "| 2 | NullProvider | OK |" in report
    assert "provider" in report  # stats table renders the null stats
