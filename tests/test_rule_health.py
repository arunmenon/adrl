"""Tests for WS3 heuristic-health (rule_health.py).

Seeds a real SQLite ledger (via the WS1 SqliteProvider schema) with crafted
decisions + outcomes, then asserts the rule audit computes the right rates,
lifts, and demote verdicts — and degrades honestly on a missing DB.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from router import rule_health
from router.memory_sqlite import SqliteProvider


def _seed(db_path: Path, rows: list[dict]) -> None:
    """Create the ledger schema and insert decision+outcome rows.

    Each row: {features: dict, hard: bool, source: str}. The outcome's
    ``outcome_proxy_hard`` carries the hardness so the pool reflects it.
    """
    SqliteProvider(db_path=db_path)  # runs _SCHEMA (idempotent)
    conn = sqlite3.connect(str(db_path))
    for i, row in enumerate(rows):
        route_id = f"r{i}"
        conn.execute(
            "INSERT INTO decisions (route_id, ts, session_id, turn_index, source, "
            "instr_sha256, features_json, layer, rung) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (route_id, float(i), "s", i, row.get("source", "organic"),
             "sha", json.dumps(row["features"]), "heuristic", "local"),
        )
        conn.execute(
            "INSERT INTO outcomes (route_id, status, escalated, user_retried, "
            "outcome_proxy_hard) VALUES (?, 'closed_final', 0, 0, ?)",
            (route_id, 1 if row["hard"] else 0),
        )
    conn.commit()
    conn.close()


def test_rule_predicates_fire():
    verdicts = {r.name: r for r in rule_health.RULES}
    assert verdicts["verb:trivial"].fired({"verb_class": "trivial"})
    assert not verdicts["verb:trivial"].fired({"verb_class": "fix"})
    assert verdicts["scope:broad"].fired({"broad_scope": True})
    assert verdicts["context:big"].fired({"context_tokens": 50_000})
    assert not verdicts["context:big"].fired({"context_tokens": 100})
    assert verdicts["traj:edit_failures"].fired({"recent_edit_failures": 2})
    assert not verdicts["traj:edit_failures"].fired({"recent_edit_failures": 0})
    assert verdicts["retry_signal"].fired({"prev_turn_interrupted": True})


def test_went_hard_signal():
    assert rule_health._went_hard(1, 0, 0)      # escalated
    assert rule_health._went_hard(0, True, 0)   # user retried
    assert rule_health._went_hard(0, 0, 1)      # proxy hard
    assert not rule_health._went_hard(0, 0, 0)
    assert not rule_health._went_hard(0, None, None)


def test_base_rate_and_lift(tmp_path):
    db = tmp_path / "m.db"
    # 100 turns, 20 hard -> base rate 20%. verb:fix fires on 30, of which 18 hard.
    rows = []
    for i in range(30):
        rows.append({"features": {"verb_class": "fix"}, "hard": i < 18})
    for i in range(70):
        rows.append({"features": {"verb_class": "unknown"}, "hard": i < 2})
    _seed(db, rows)
    result = rule_health.analyse(db, min_sample=10)
    assert result["total"] == 100
    assert result["hard_total"] == 20
    assert abs(result["base_rate"] - 0.20) < 1e-9
    fix = next(v for v in result["rules"] if v.rule == "verb:fix")
    assert fix.fired == 30
    assert abs(fix.hard_rate - 0.60) < 1e-9
    assert abs(fix.lift - 0.40) < 1e-9
    assert fix.verdict == "OK"  # hard-leaning with strong positive lift


def test_easy_rule_anti_signal_is_demote_candidate(tmp_path):
    db = tmp_path / "m.db"
    # verb:trivial (easy-leaning) fires on 40 turns, 30 of them hard (75%),
    # while the pool base rate is much lower -> anti-signal.
    rows = [{"features": {"verb_class": "trivial"}, "hard": i < 30} for i in range(40)]
    rows += [{"features": {"verb_class": "unknown"}, "hard": False} for _ in range(60)]
    _seed(db, rows)
    result = rule_health.analyse(db, min_sample=10)
    trivial = next(v for v in result["rules"] if v.rule == "verb:trivial")
    assert trivial.hard_rate > result["base_rate"]
    assert trivial.verdict == "DEMOTE-CANDIDATE"


def test_hard_rule_over_routing_is_demote_candidate(tmp_path):
    db = tmp_path / "m.db"
    # scope:broad (hard-leaning) fires on 40 turns but they are mostly EASY
    # (5 hard) while the rest of the pool is hard -> over-routing.
    rows = [{"features": {"broad_scope": True}, "hard": i < 5} for i in range(40)]
    rows += [{"features": {"verb_class": "unknown"}, "hard": True} for _ in range(60)]
    _seed(db, rows)
    result = rule_health.analyse(db, min_sample=10)
    broad = next(v for v in result["rules"] if v.rule == "scope:broad")
    assert broad.hard_rate < result["base_rate"]
    assert broad.verdict == "DEMOTE-CANDIDATE"


def test_insufficient_sample(tmp_path):
    db = tmp_path / "m.db"
    rows = [{"features": {"verb_class": "hard"}, "hard": True} for _ in range(3)]
    rows += [{"features": {"verb_class": "unknown"}, "hard": False} for _ in range(50)]
    _seed(db, rows)
    result = rule_health.analyse(db, min_sample=20)
    hard = next(v for v in result["rules"] if v.rule == "verb:hard")
    assert hard.fired == 3
    assert hard.verdict == "INSUFFICIENT"


def test_source_filter(tmp_path):
    db = tmp_path / "m.db"
    rows = [{"features": {"verb_class": "fix"}, "hard": True, "source": "simulator"}
            for _ in range(10)]
    rows += [{"features": {"verb_class": "explain"}, "hard": False, "source": "organic"}
             for _ in range(10)]
    _seed(db, rows)
    organic = rule_health.analyse(db, source="organic")
    simulator = rule_health.analyse(db, source="simulator")
    assert organic["total"] == 10
    assert simulator["total"] == 10
    assert organic["hard_total"] == 0
    assert simulator["hard_total"] == 10


def test_missing_db_degrades_not_crashes(tmp_path):
    db = tmp_path / "does-not-exist.db"
    result = rule_health.analyse(db)
    assert result["total"] == 0
    report = rule_health.build(db)
    assert "degraded" in report or "no closed outcomes" in report


def test_uninitialized_db_degrades(tmp_path):
    db = tmp_path / "empty.db"
    sqlite3.connect(str(db)).close()  # exists but has no tables
    result = rule_health.analyse(db)
    assert result["total"] == 0  # OperationalError swallowed


def test_build_markdown_smoke(tmp_path):
    db = tmp_path / "m.db"
    rows = [{"features": {"verb_class": "explain"}, "hard": False} for _ in range(25)]
    rows += [{"features": {"verb_class": "fix"}, "hard": True} for _ in range(25)]
    _seed(db, rows)
    report = rule_health.build(db, min_sample=20)
    assert "# rule health" in report
    assert "base hard-rate" in report
    assert "verb:explain" in report


def test_json_payload_serializable(tmp_path):
    db = tmp_path / "m.db"
    _seed(db, [{"features": {"verb_class": "fix"}, "hard": True} for _ in range(5)])
    payload = rule_health._json_payload(db, None, 20)
    # must round-trip through json (dataclasses converted to dicts)
    json.dumps(payload)
    assert payload["rules"][0]["rule"]
