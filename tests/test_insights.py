"""Insights layer: findings derived from a synthetic memory."""

import json
import time

import pytest

from router.insights import build, generate
from router.memory_ports import DecisionEvent, OutcomeEvent
from router.memory_sqlite import SqliteProvider


def _decision(rid, *, layer, rung, session="s1", turn=0, sha="ab" * 32):
    return DecisionEvent(
        route_id=rid, ts=time.time(), session_id=session, turn_index=turn,
        source="organic", instr_sha256=sha,
        features_json=json.dumps({"context_tokens": 5000, "fired_rules": []}),
        layer=layer, rung=rung, cascade=(rung == "local"), score=0.5,
        reason="test", classifier_tier=None, propensity="heuristic",
        policy_version="v1", classifier_ms=0.0, decision_ms=0.1)


def _outcome(*, hard, escalated=False):
    return OutcomeEvent(
        status="closed_final", escalated=escalated, tripwire_name=None,
        tripwire_type=None, edit_failures=1 if hard else 0, error_results=0,
        output_tokens=100, latency_ms=10.0, cost_estimate=0.0,
        interrupted=False, user_retried=False, outcome_proxy_hard=hard)


@pytest.fixture
def seeded(tmp_path):
    db = tmp_path / "mem.db"
    p = SqliteProvider(db, embedder=None)
    # a corpus that should trigger the gate-dominance + forced-frontier findings
    plan = [
        ("g1", "gate:feasibility", "frontier", None),   # forced frontier, secret (no sha)
        ("g2", "gate:feasibility", "frontier", None),
        ("g3", "gate:privacy", "local", None),
        ("m1", "middle_default", "local", "ab" * 32),   # middle kept local, went hard
        ("m2", "middle_default", "local", "ab" * 32),
        ("h1", "heuristic", "local", "cd" * 32),        # the lone heuristic fire
    ]
    for i, (rid, layer, rung, sha) in enumerate(plan):
        d = _decision(rid, layer=layer, rung=rung, turn=i,
                      sha=("ab" * 32 if sha else None))
        # override sha explicitly (None => privacy-excluded)
        d.instr_sha256 = sha
        p.record_decision(d)
        p.attach_outcome(rid, _outcome(hard=(rid == "m1")))
    return db


def test_generates_ranked_insights(seeded):
    ins = generate(seeded)
    assert ins, "expected findings"
    # ranked descending by magnitude
    mags = [i.magnitude for i in ins]
    assert mags == sorted(mags, reverse=True)
    kinds = {i.kind for i in ins}
    assert {"economics", "heuristic", "context", "privacy"} <= kinds


def test_forced_frontier_finding(seeded):
    ctx = next(i for i in generate(seeded) if i.kind == "context")
    # both frontier turns were sent by a gate -> 100% forced
    assert ctx.evidence["frontier"] == 2 and ctx.evidence["forced_by_gate"] == 2


def test_privacy_exclusion_counted(seeded):
    priv = next(i for i in generate(seeded) if i.kind == "privacy")
    assert priv.evidence["excluded"] == 3          # three sha-less rows


def test_rule_health_middle_contradiction(seeded):
    rh = next((i for i in generate(seeded) if i.kind == "rule_health"), None)
    assert rh is not None
    assert rh.evidence["middle_local"] == 2 and rh.evidence["went_hard"] == 1


def test_build_renders_markdown(seeded):
    md = build(seeded)
    assert "# Router insights" in md and "forced" in md.lower()


def test_empty_db_is_safe(tmp_path):
    assert generate(tmp_path / "nope.db") == []
    assert "No memory" in build(tmp_path / "nope.db")
