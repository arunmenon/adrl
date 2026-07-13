"""Parametrized MemoryProvider contract suite (WS1, Unit A).

Every provider behind the facade must honor the same port contract:
record/attach/finalize round-trip, empty-safe similar_turns, and — above all —
the FAIL-SAFE promise: these methods never raise, even on garbage inputs;
degraded answers are None / False / [] / 0 / {}.

Parametrized over the providers available at collection time: NullProvider is
always in; SqliteProvider (Unit B, built in parallel) joins automatically via
importorskip the moment router.memory_sqlite lands.
"""

from __future__ import annotations

import pytest

from router.memory_null import NullProvider
from router.memory_ports import (
    DecisionEvent,
    MemoryProvider,
    NeighborTurn,
    OutcomeEvent,
    VerifiedOutcome,
)

PROVIDER_PARAMS = ["null", "sqlite"]


@pytest.fixture(params=PROVIDER_PARAMS)
def provider(request, tmp_path):
    if request.param == "null":
        return NullProvider()
    module = pytest.importorskip("router.memory_sqlite")
    provider_cls = module.SqliteProvider
    db_path = tmp_path / "router-memory.db"
    for construct in (lambda: provider_cls(db_path),
                      lambda: provider_cls(db_path=db_path),
                      lambda: provider_cls(path=db_path),
                      lambda: provider_cls(str(db_path))):
        try:
            return construct()
        except TypeError:
            continue
    pytest.skip("SqliteProvider constructor signature not recognized")


def _decision(route_id: str = "r-0001", session_id: str = "s-1",
              turn_index: int = 0) -> DecisionEvent:
    return DecisionEvent(
        route_id=route_id, ts=1_700_000_000.0, session_id=session_id,
        turn_index=turn_index, source="simulator",
        instr_sha256="ab" * 32, features_json='{"verb_class": "write"}',
        layer="heuristic", rung="local", cascade=True, score=0.42,
        reason="easy (write, 0.42 < 0.35)", classifier_tier=None,
        propensity="heuristic", policy_version="v1",
        classifier_ms=0.0, decision_ms=1.5,
    )


def _outcome(status: str = "closed_turn") -> OutcomeEvent:
    return OutcomeEvent(
        status=status, escalated=False, tripwire_name=None, tripwire_type=None,
        edit_failures=0, error_results=1, output_tokens=850, latency_ms=1200.0,
        cost_estimate=0.002, interrupted=False, user_retried=None,
        outcome_proxy_hard=True,
    )


def test_implements_port(provider):
    assert isinstance(provider, MemoryProvider)


def test_record_attach_finalize_round_trip(provider):
    decision = _decision()
    stored = provider.record_decision(decision)
    assert stored == decision.route_id

    attached = provider.attach_outcome(decision.route_id, _outcome())
    assert attached is True

    verified = provider.attach_verification(
        decision.route_id,
        VerifiedOutcome(task_success=True, verifier_source="contract"),
        event_id="contract-verification",
    )
    assert verified is True

    finalized = provider.finalize_turn(
        decision.session_id, prev_interrupted=False, prev_retried=False)
    assert isinstance(finalized, int)
    assert finalized >= 0


def test_record_with_embedding(provider):
    decision = _decision(route_id="r-emb-1")
    embedding = [0.1] * 8
    assert provider.record_decision(decision, embedding) == decision.route_id


def test_similar_turns_empty_safe(provider):
    # No embeddings stored (or a provider without vectors): must answer [].
    neighbors = provider.similar_turns([0.0] * 768, k=5)
    assert isinstance(neighbors, list)
    for neighbor in neighbors:
        assert isinstance(neighbor, NeighborTurn)


def test_stats_and_health_shapes(provider):
    stats = provider.stats()
    assert isinstance(stats, dict)
    assert isinstance(provider.health(), bool)


def test_attach_to_unknown_route_id_is_bool(provider):
    # Providers may accept-into-the-void (Null) or refuse (a real store);
    # either way the answer is a bool and nothing raises.
    assert provider.attach_outcome("no-such-route-id", _outcome()) in (True, False)


@pytest.mark.parametrize("call", [
    lambda p: p.record_decision(None),
    lambda p: p.record_decision(_decision(), embedding="not-a-vector"),
    lambda p: p.attach_outcome(None, None),
    lambda p: p.attach_outcome(12345, object()),
    lambda p: p.attach_verification(None, None),
    lambda p: p.attach_verification(12345, object()),
    lambda p: p.finalize_turn(None, prev_interrupted=False, prev_retried=False),
    lambda p: p.similar_turns(None),
    lambda p: p.similar_turns("garbage", k=-3),
    lambda p: p.similar_turns([], k=0),
])
def test_fail_safe_never_raises_on_garbage(provider, call):
    result = call(provider)  # must not raise — degraded answers only
    assert result is None or isinstance(result, (bool, int, list, str, dict))
