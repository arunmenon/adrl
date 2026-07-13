"""Tests for router.memory_sqlite — the Engram-lite SQLite provider.

All databases live in tmp_path; the embedder is a deterministic fake (no live
server). Mirrors the module's fail-safe contract: degraded answers, never
exceptions.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from router.memory_sqlite import (
    DecisionEvent,
    NeighborTurn,
    OutcomeEvent,
    QUERY_PREFIX,
    SCHEMA_VERSION,
    STORAGE_PREFIX,
    SqliteProvider,
    VerifiedOutcome,
    build,
)


class FakeEmbedder:
    """Deterministic embedder that records exactly what it was asked to embed."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            seed = sum(ord(ch) for ch in text) % 97
            rng = np.random.default_rng(seed)
            vectors.append(rng.normal(size=self.dim).astype(np.float32).tolist())
        return vectors


def make_decision(route_id: str, *, session_id: str = "sess-1", turn_index: int = 0,
                  source: str = "simulator", rung: str = "local_small",
                  ts: float = 1000.0) -> DecisionEvent:
    return DecisionEvent(
        route_id=route_id,
        ts=ts,
        session_id=session_id,
        turn_index=turn_index,
        source=source,
        instr_sha256="ab" * 32,
        features_json='{"fired_rules": []}',
        layer="heuristic",
        rung=rung,
        cascade=False,
        score=0.42,
        reason="test",
        classifier_tier=None,
        propensity="heuristic",
        policy_version="v1",
        classifier_ms=0.0,
        decision_ms=1.5,
    )


def make_outcome(status: str = "closed_turn", *, escalated: bool = False,
                 outcome_proxy_hard: bool | None = False) -> OutcomeEvent:
    return OutcomeEvent(
        status=status,
        escalated=escalated,
        tripwire_name="edit_fail" if escalated else None,
        tripwire_type="dialect" if escalated else None,
        edit_failures=2 if escalated else 0,
        error_results=0,
        output_tokens=120,
        latency_ms=850.0,
        cost_estimate=0.003,
        interrupted=False,
        user_retried=None,
        outcome_proxy_hard=outcome_proxy_hard,
    )


@pytest.fixture
def provider(tmp_path):
    return SqliteProvider(db_path=tmp_path / "memory.db", embedder=FakeEmbedder())


# ── round-trip ───────────────────────────────────────────────────────────────


def test_record_decision_round_trip(provider, tmp_path):
    route_id = provider.record_decision(make_decision("r1"), embedding=[1.0, 0.0, 0.0])
    assert route_id == "r1"

    stats = provider.stats()
    assert stats["decisions"] == 1
    assert stats["by_source"] == {"simulator": 1}
    assert stats["outcomes_by_status"] == {"pending": 1}
    assert stats["embeddings"] == 1
    assert stats["embedding_coverage"] == 1.0

    assert provider.attach_outcome("r1", make_outcome("closed_turn")) is True
    stats = provider.stats()
    assert stats["outcomes_by_status"] == {"closed_turn": 1}

    connection = sqlite3.connect(tmp_path / "memory.db")
    decision_row = connection.execute(
        "SELECT session_id, rung, cascade, score FROM decisions WHERE route_id='r1'"
    ).fetchone()
    assert decision_row == ("sess-1", "local_small", 0, 0.42)
    outcome_row = connection.execute(
        "SELECT status, escalated, output_tokens, outcome_proxy_hard "
        "FROM outcomes WHERE route_id='r1'"
    ).fetchone()
    assert outcome_row == ("closed_turn", 0, 120, 0)
    connection.close()


def test_bad_dim_row_does_not_sink_projection(provider):
    # A valid closed decision with an embedding...
    provider.record_decision(make_decision("good", ts=1.0),
                             embedding=[1.0, 0.0, 0.0, 0.0])
    provider.attach_outcome("good", make_outcome(status="closed_final"))
    # ...plus a corrupt row whose dim column is NON-NUMERIC text (SQLite allows
    # any type in an INTEGER column). int('bad-dim') used to raise and sink the
    # entire projection via similar_turns' outer except (review finding).
    conn = provider._conn
    conn.execute(
        "INSERT INTO decisions (route_id, ts, session_id, turn_index, source, "
        "features_json, layer, rung) VALUES "
        "('bad', 2.0, 's2', 0, 'simulator', '{}', 'heuristic', 'local')")
    conn.execute(
        "INSERT INTO outcomes (route_id, status, escalated) "
        "VALUES ('bad', 'closed_final', 0)")
    conn.execute(
        "INSERT INTO embeddings (route_id, dim, vec) VALUES ('bad', 'bad-dim', ?)",
        (np.asarray([1, 0, 0, 0], dtype="<f4").tobytes(),))
    conn.commit()
    neighbors = provider.similar_turns([1.0, 0.0, 0.0, 0.0], k=5)
    ids = {n.route_id for n in neighbors}
    assert "good" in ids        # the valid neighbor survives...
    assert "bad" not in ids     # ...and the corrupt row is skipped, not fatal


def test_record_without_embedding_stores_no_embedding_row(provider):
    provider.record_decision(make_decision("r-pinned"))
    stats = provider.stats()
    assert stats["embeddings"] == 0
    assert stats["decisions"] == 1


def test_attach_outcome_unknown_route_returns_false(provider):
    assert provider.attach_outcome("nope", make_outcome()) is False


# ── idempotent ingestion ─────────────────────────────────────────────────────


def test_duplicate_route_id_is_idempotent(provider):
    first = provider.record_decision(make_decision("dup"), embedding=[1.0, 2.0])
    provider.attach_outcome("dup", make_outcome("closed_turn"))
    second = provider.record_decision(
        make_decision("dup", rung="frontier"), embedding=[9.0, 9.0]
    )
    assert first == second == "dup"
    stats = provider.stats()
    assert stats["decisions"] == 1
    assert stats["embeddings"] == 1
    # the original facts survive the re-ingestion; outcome not reset to pending
    assert stats["outcomes_by_status"] == {"closed_turn": 1}
    connection = sqlite3.connect(provider.db_path)
    assert connection.execute(
        "SELECT rung FROM decisions WHERE route_id='dup'"
    ).fetchone() == ("local_small",)
    original = np.frombuffer(
        connection.execute(
            "SELECT vec FROM embeddings WHERE route_id='dup'"
        ).fetchone()[0],
        dtype="<f4",
    )
    connection.close()
    assert original.tolist() == [1.0, 2.0]


# ── lifecycle forward-only ───────────────────────────────────────────────────


def test_lifecycle_forward_and_same_rank(provider):
    provider.record_decision(make_decision("life"))
    assert provider.attach_outcome("life", make_outcome("closed_turn")) is True
    # backward transitions rejected
    assert provider.attach_outcome("life", make_outcome("pending")) is False
    # SAME-rank re-update on a non-terminal row is allowed (last-write-wins) so a
    # turn's continuations can each refresh the working outcome (review finding)
    assert provider.attach_outcome(
        "life", make_outcome("closed_turn", escalated=True)) is True
    assert provider.attach_outcome("life", make_outcome("closed_final")) is True
    # closed_final is terminal — no re-write, no backward
    assert provider.attach_outcome("life", make_outcome("closed_turn")) is False
    assert provider.attach_outcome("life", make_outcome("closed_final")) is False
    assert provider.attach_outcome("life", make_outcome("bogus_status")) is False
    assert provider.stats()["outcomes_by_status"] == {"closed_final": 1}


def test_lifecycle_can_skip_to_closed_final(provider):
    provider.record_decision(make_decision("skip"))
    assert provider.attach_outcome("skip", make_outcome("closed_final")) is True
    assert provider.attach_outcome("skip", make_outcome("closed_turn")) is False


# ── finalize_turn ────────────────────────────────────────────────────────────


def test_finalize_turn_promotes_most_recent_closed_turn(provider):
    provider.record_decision(make_decision("t0", turn_index=0, ts=100.0))
    provider.record_decision(make_decision("t1", turn_index=1, ts=200.0))
    provider.attach_outcome("t0", make_outcome("closed_turn"))
    provider.attach_outcome("t1", make_outcome("closed_turn"))

    changed = provider.finalize_turn("sess-1", prev_interrupted=True, prev_retried=True)
    assert changed == 1

    connection = sqlite3.connect(provider.db_path)
    assert connection.execute(
        "SELECT status, interrupted, user_retried FROM outcomes WHERE route_id='t1'"
    ).fetchone() == ("closed_final", 1, 1)
    # the older turn is untouched
    assert connection.execute(
        "SELECT status FROM outcomes WHERE route_id='t0'"
    ).fetchone() == ("closed_turn",)
    connection.close()


def test_finalize_turn_promotes_pending(provider):
    # Corrected contract: a pending outcome (turn got no per-call outcome) is a
    # finalize candidate — turn N+1's retry signal still closes turn N.
    provider.record_decision(make_decision("pend"))
    assert provider.finalize_turn(
        "sess-1", prev_interrupted=True, prev_retried=False
    ) == 1
    row = provider.stats()  # sanity: it is now closed
    assert row["outcomes_by_status"].get("closed_final", 0) == 1


def test_finalize_turn_no_candidates(provider):
    # Genuinely nothing to finalize: unknown session, and an already-final one.
    assert provider.finalize_turn(
        "no-such-session", prev_interrupted=False, prev_retried=False
    ) == 0
    provider.record_decision(make_decision("done"))
    provider.finalize_turn("sess-1", prev_interrupted=False, prev_retried=False)
    # second finalize on the same session: latest is closed_final -> 0
    assert provider.finalize_turn(
        "sess-1", prev_interrupted=False, prev_retried=False
    ) == 0


# ── kNN retrieval ────────────────────────────────────────────────────────────


def _seed_closed(provider, route_id, embedding, **kwargs):
    provider.record_decision(make_decision(route_id, **kwargs), embedding=embedding)
    provider.attach_outcome(route_id, make_outcome("closed_turn"))


def test_similar_turns_orders_by_cosine(provider):
    _seed_closed(provider, "exact", [1.0, 0.0, 0.0])
    _seed_closed(provider, "near", [0.9, 0.1, 0.0])
    _seed_closed(provider, "orthogonal", [0.0, 0.0, 1.0])
    _seed_closed(provider, "opposite", [-1.0, 0.0, 0.0])

    neighbors = provider.similar_turns([1.0, 0.0, 0.0], k=4)
    assert [n.route_id for n in neighbors] == ["exact", "near", "orthogonal", "opposite"]
    assert neighbors[0].similarity == pytest.approx(1.0, abs=1e-6)
    assert neighbors[1].similarity == pytest.approx(
        0.9 / np.sqrt(0.9**2 + 0.1**2), abs=1e-6
    )
    assert neighbors[2].similarity == pytest.approx(0.0, abs=1e-6)
    assert neighbors[3].similarity == pytest.approx(-1.0, abs=1e-6)
    # scaling the stored vector must not change cosine ordering
    assert isinstance(neighbors[0], NeighborTurn) or hasattr(neighbors[0], "route_id")


def test_similar_turns_respects_k(provider):
    for index in range(5):
        _seed_closed(provider, f"k{index}", [1.0, float(index), 0.0])
    assert len(provider.similar_turns([1.0, 0.0, 0.0], k=3)) == 3


def test_similar_turns_carries_outcome_metadata(provider):
    provider.record_decision(
        make_decision("meta", source="organic", rung="frontier", ts=777.0),
        embedding=[0.0, 1.0, 0.0],
    )
    provider.attach_outcome(
        "meta", make_outcome("closed_turn", escalated=True, outcome_proxy_hard=True)
    )
    neighbor = provider.similar_turns([0.0, 1.0, 0.0], k=1)[0]
    assert neighbor.route_id == "meta"
    assert neighbor.rung == "frontier"
    assert neighbor.escalated is True
    assert neighbor.outcome_proxy_hard is True
    assert neighbor.source == "organic"
    assert neighbor.ts == 777.0


def test_pending_rows_invisible_to_knn(provider):
    _seed_closed(provider, "closed", [0.5, 0.5, 0.0])
    provider.record_decision(make_decision("still-pending"), embedding=[1.0, 0.0, 0.0])

    neighbors = provider.similar_turns([1.0, 0.0, 0.0], k=10)
    assert [n.route_id for n in neighbors] == ["closed"]

    # promoting the pending row makes it visible (status change, no new rows)
    provider.attach_outcome("still-pending", make_outcome("closed_turn"))
    neighbors = provider.similar_turns([1.0, 0.0, 0.0], k=10)
    assert [n.route_id for n in neighbors] == ["still-pending", "closed"]


def test_matrix_refreshes_on_new_rows(provider):
    _seed_closed(provider, "old", [1.0, 0.0, 0.0])
    assert [n.route_id for n in provider.similar_turns([0.0, 1.0, 0.0], k=5)] == ["old"]

    _seed_closed(provider, "new", [0.0, 1.0, 0.0])
    neighbors = provider.similar_turns([0.0, 1.0, 0.0], k=5)
    assert [n.route_id for n in neighbors] == ["new", "old"]


def test_similar_turns_empty_and_mismatched(provider):
    assert provider.similar_turns([1.0, 0.0, 0.0]) == []
    _seed_closed(provider, "d3", [1.0, 0.0, 0.0])
    # query dim mismatch degrades to []
    assert provider.similar_turns([1.0, 0.0], k=3) == []
    # zero query vector degrades to []
    assert provider.similar_turns([0.0, 0.0, 0.0], k=3) == []


# ── float32 BLOB fidelity ────────────────────────────────────────────────────


def test_float32_blob_round_trip(provider):
    original = [0.1, -2.5, 3.14159265, 1e-7, 12345.678]
    provider.record_decision(make_decision("blob"), embedding=original)
    connection = sqlite3.connect(provider.db_path)
    dim, blob = connection.execute(
        "SELECT dim, vec FROM embeddings WHERE route_id='blob'"
    ).fetchone()
    connection.close()
    assert dim == 5
    assert len(blob) == 5 * 4  # float32 little-endian
    recovered = np.frombuffer(blob, dtype="<f4")
    np.testing.assert_allclose(
        recovered, np.asarray(original, dtype=np.float32), rtol=0, atol=0
    )


# ── fail-safe ────────────────────────────────────────────────────────────────


def test_fail_safe_on_unusable_db_path(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("plain file, not a directory")
    broken = SqliteProvider(db_path=blocker / "memory.db", embedder=FakeEmbedder())

    assert broken.health() is False
    assert broken.record_decision(make_decision("x"), embedding=[1.0]) is None
    assert broken.attach_outcome("x", make_outcome()) is False
    assert broken.finalize_turn("s", prev_interrupted=False, prev_retried=False) == 0
    assert broken.similar_turns([1.0, 0.0]) == []
    assert broken.stats() == {}


def test_fail_safe_on_corrupted_db_file(tmp_path):
    corrupt = tmp_path / "memory.db"
    corrupt.write_bytes(b"this is not a sqlite database at all" * 40)
    broken = SqliteProvider(db_path=corrupt, embedder=FakeEmbedder())
    assert broken.record_decision(make_decision("x")) is None
    assert broken.similar_turns([1.0, 0.0]) == []
    assert broken.stats() == {}
    assert broken.health() is False


def test_health_true_on_working_db(provider):
    assert provider.health() is True


# ── embedding prefixes ───────────────────────────────────────────────────────


def test_embed_prefixes(provider):
    fake = provider._embedder
    stored = provider.embed_for_storage("refactor the auth module")
    queried = provider.embed_for_query("refactor the auth module")
    assert stored is not None and len(stored) == fake.dim
    assert queried is not None and len(queried) == fake.dim
    assert fake.calls == [
        [STORAGE_PREFIX + "refactor the auth module"],
        [QUERY_PREFIX + "refactor the auth module"],
    ]


def test_embed_helpers_fail_safe():
    class ExplodingEmbedder:
        def embed(self, texts):
            raise RuntimeError("server down")

    provider = SqliteProvider(
        db_path=Path("/dev/null/never-created.db"), embedder=ExplodingEmbedder()
    )
    assert provider.embed_for_storage("text") is None
    assert provider.embed_for_query("text") is None


# ── reporter ─────────────────────────────────────────────────────────────────


def test_build_report(tmp_path):
    db_path = tmp_path / "memory.db"
    seeded = SqliteProvider(db_path=db_path, embedder=FakeEmbedder())
    seeded.record_decision(make_decision("r1"), embedding=[1.0, 0.0])
    seeded.attach_outcome("r1", make_outcome("closed_turn"))
    seeded.record_decision(make_decision("r2", source="organic"))

    report = build(db_path)
    assert report.startswith("# router-memory")
    assert "decisions: 2" in report
    assert "simulator: 1" in report
    assert "organic: 1" in report
    assert "closed_turn: 1" in report
    assert "pending: 1" in report
    assert "healthy: yes" in report


# ── review-fix regressions ─────────────────────────────────────────────────────

def test_bad_embedding_leaves_no_partial_decision(provider):
    """Finding #1: a bad embedding must not persist a half-written decision."""
    assert provider.record_decision(
        make_decision("bad"), embedding=["not", "a", "number"]) is None
    assert provider.stats()["decisions"] == 0          # rolled back, no orphan
    # and a subsequent good write commits cleanly (would flush the orphan if any)
    assert provider.record_decision(make_decision("good")) == "good"
    assert provider.stats()["decisions"] == 1


def test_outcome_update_refreshes_knn_projection(provider):
    """Finding #5: an in-place outcome change (same row counts) must invalidate
    the cached kNN projection."""
    vec = [1.0, 0.0, 0.0, 0.0]
    provider.record_decision(make_decision("e1"), embedding=vec)
    provider.attach_outcome("e1", make_outcome("closed_turn", outcome_proxy_hard=False))
    first = provider.similar_turns(vec, k=1)
    assert first and first[0].outcome_proxy_hard is False
    # promote to closed_final flipping outcome_proxy_hard — row counts unchanged
    provider.attach_outcome("e1", make_outcome("closed_final", outcome_proxy_hard=True))
    second = provider.similar_turns(vec, k=1)
    assert second and second[0].outcome_proxy_hard is True   # projection refreshed


def test_pre_v2_database_migrates_in_place(tmp_path):
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE decisions (
            route_id TEXT PRIMARY KEY, ts REAL, session_id TEXT, turn_index INT,
            source TEXT, instr_sha256 TEXT, features_json TEXT, layer TEXT,
            rung TEXT, cascade INT, score REAL, reason TEXT,
            classifier_tier TEXT, propensity TEXT, policy_version TEXT,
            classifier_ms REAL, decision_ms REAL
        );
        CREATE TABLE outcomes (
            route_id TEXT PRIMARY KEY, status TEXT, escalated INT,
            tripwire_name TEXT, tripwire_type TEXT, edit_failures INT,
            error_results INT, output_tokens INT, latency_ms REAL,
            cost_estimate REAL, interrupted INT, user_retried INT,
            outcome_proxy_hard INT
        );
        CREATE TABLE embeddings (route_id TEXT PRIMARY KEY, dim INT, vec BLOB);
        INSERT INTO decisions VALUES (
            'legacy', 1.0, 's', 0, 'organic', 'sha', '{}', 'classifier',
            'local', 1, 0.5, 'old', NULL, 'classifier', 'v1', 1.0, 2.0
        );
        INSERT INTO outcomes VALUES (
            'legacy', 'closed_final', 1, 'edit_apply',
            'TripwireType.DIALECT', 2, 2, 10, 20.0, 0.0, 0, 0, 1
        );
    """)
    conn.commit()
    conn.close()

    migrated = SqliteProvider(db_path=db, embedder=FakeEmbedder())
    assert migrated.health()
    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    decision_columns = {r[1] for r in conn.execute("PRAGMA table_info(decisions)")}
    outcome_columns = {r[1] for r in conn.execute("PRAGMA table_info(outcomes)")}
    assert "trace_json" in decision_columns
    assert {"continuation_count", "failure_cause", "task_signal_hard",
            "verified_success", "verification_failure_cause"} <= outcome_columns
    # The dialect escalation remains operational friction but is explicitly not
    # a task-difficulty signal after migration.
    assert conn.execute(
        "SELECT outcome_proxy_hard, task_signal_hard FROM outcomes "
        "WHERE route_id='legacy'").fetchone() == (1, 0)
    assert conn.execute(
        "SELECT COUNT(*) FROM outcome_events WHERE route_id='legacy'").fetchone()[0] == 1
    conn.close()
    migrated._conn.close()

    # Reopening at the current version must not manufacture another event.
    reopened = SqliteProvider(db_path=db, embedder=FakeEmbedder())
    conn = sqlite3.connect(db)
    assert conn.execute(
        "SELECT COUNT(*) FROM outcome_events WHERE route_id='legacy'").fetchone()[0] == 1
    conn.close()
    reopened._conn.close()


def test_newer_schema_fails_safe_instead_of_downgrading(tmp_path):
    db = tmp_path / "future.db"
    conn = sqlite3.connect(db)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.commit()
    conn.close()
    provider = SqliteProvider(db_path=db, embedder=FakeEmbedder())
    assert provider.health() is False
    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION + 1
    conn.close()


def test_verified_outcome_and_immutable_event_round_trip(provider):
    provider.record_decision(make_decision("verified"))
    event = OutcomeEvent(
        status="closed_turn",
        task_signal_hard=False,
        continuation_count=2,
        verification=VerifiedOutcome(
            task_success=True,
            quality_score=0.95,
            verifier_source="pytest",
            confidence=1.0,
            verified_at=1234.0,
            failure_cause=None,
        ),
    )
    assert provider.attach_outcome("verified", event)
    # Replaying the same immutable event is idempotent.
    assert provider.attach_outcome("verified", event)
    conn = sqlite3.connect(provider.db_path)
    assert conn.execute(
        "SELECT continuation_count, task_signal_hard, verified_success, "
        "quality_score, verifier_source, verifier_confidence, verified_at "
        "FROM outcomes WHERE route_id='verified'").fetchone() == (
            2, 0, 1, 0.95, "pytest", 1.0, 1234.0)
    # pending-at-decision + one accepted closed_turn event; replay adds nothing.
    assert conn.execute(
        "SELECT COUNT(*) FROM outcome_events WHERE route_id='verified'").fetchone()[0] == 2
    conn.close()


def test_late_verification_enriches_finalized_outcome_without_erasing_it(provider):
    vector = [1.0, 0.0, 0.0, 0.0]
    provider.record_decision(
        make_decision("late", rung="local-code"), embedding=vector)
    outcome = OutcomeEvent(
        status="closed_final",
        escalated=True,
        tripwire_name="edit_apply",
        tripwire_type="dialect",
        edit_failures=3,
        error_results=2,
        output_tokens=450,
        outcome_proxy_hard=True,
        continuation_count=4,
        failure_cause="harness_dialect",
        task_signal_hard=True,
    )
    assert provider.attach_outcome("late", outcome)
    verification = VerifiedOutcome(
        task_success=True,
        quality_score=0.98,
        verifier_source="deterministic:v1",
        confidence=1.0,
        verified_at=2000.0,
    )
    assert provider.attach_verification(
        "late", verification, event_id="verify-late", observed_at=2001.0)
    # Replaying an immutable event is a no-op, even with different payload data.
    assert provider.attach_verification(
        "late", VerifiedOutcome(task_success=False),
        event_id="verify-late", observed_at=3000.0)

    conn = sqlite3.connect(provider.db_path)
    assert conn.execute(
        "SELECT status, escalated, edit_failures, error_results, "
        "continuation_count, failure_cause, task_signal_hard, "
        "verified_success, quality_score, verifier_source, verified_at "
        "FROM outcomes WHERE route_id='late'"
    ).fetchone() == (
        "closed_final", 1, 3, 2, 4, "harness_dialect", 1,
        1, 0.98, "deterministic:v1", 2000.0,
    )
    events = conn.execute(
        "SELECT payload_json FROM outcome_events WHERE route_id='late' "
        "ORDER BY observed_at"
    ).fetchall()
    assert len(events) == 3  # pending + closed_final + one verification
    assert sum('"source": "verification"' in row[0] for row in events) == 1
    conn.close()

    neighbor = provider.similar_turns(vector, k=1)[0]
    assert neighbor.verified_success is True
    assert neighbor.verification_failure_cause is None


def test_cross_connection_verification_invalidates_warm_projection(provider):
    vector = [1.0, 0.0, 0.0, 0.0]
    provider.record_decision(make_decision("cross-process"), embedding=vector)
    assert provider.attach_outcome(
        "cross-process", make_outcome(status="closed_final"))
    assert provider.similar_turns(vector, k=1)[0].verified_success is None

    verifier_process = SqliteProvider(provider.db_path, embedder=FakeEmbedder())
    assert verifier_process.attach_verification(
        "cross-process",
        VerifiedOutcome(task_success=True, verifier_source="deterministic:v1"),
        event_id="cross-process-verification",
    )

    # PRAGMA data_version changes on commits made by another connection, so the
    # already-warm provider must rebuild without a local outcome write/restart.
    assert provider.similar_turns(vector, k=1)[0].verified_success is True


def test_lifecycle_update_preserves_verification_attached_while_pending(provider):
    provider.record_decision(make_decision("early"))
    verification = VerifiedOutcome(
        task_success=False,
        quality_score=0.2,
        verifier_source="deterministic:v1",
        confidence=1.0,
        verified_at=50.0,
        failure_cause="task_capability",
    )
    assert provider.attach_verification(
        "early", verification, event_id="verify-early")
    assert provider.attach_outcome("early", make_outcome("closed_turn"))

    conn = sqlite3.connect(provider.db_path)
    assert conn.execute(
        "SELECT status, verified_success, quality_score, verifier_source, "
        "verification_failure_cause FROM outcomes WHERE route_id='early'"
    ).fetchone() == (
        "closed_turn", 0, 0.2, "deterministic:v1", "task_capability")
    conn.close()


def test_attach_verification_unknown_route_returns_false(provider):
    assert not provider.attach_verification(
        "missing", VerifiedOutcome(task_success=True), event_id="missing-v")


@pytest.mark.parametrize("verification", [
    VerifiedOutcome(task_success=True, verifier_source=""),
    VerifiedOutcome(
        task_success=True, verifier_source="bad", quality_score=1.5),
    VerifiedOutcome(
        task_success=True, verifier_source="bad", confidence=float("nan")),
    VerifiedOutcome(
        task_success=True, verifier_source="bad", failure_cause="task_capability"),
])
def test_malformed_verification_is_rejected_without_an_audit_event(
    provider, verification,
):
    provider.record_decision(make_decision("bad-verification"))
    assert not provider.attach_verification(
        "bad-verification", verification, event_id="invalid-verification")
    conn = sqlite3.connect(provider.db_path)
    assert conn.execute(
        "SELECT verified_success, verifier_source FROM outcomes "
        "WHERE route_id='bad-verification'").fetchone() == (None, None)
    assert conn.execute(
        "SELECT COUNT(*) FROM outcome_events "
        "WHERE route_id='bad-verification'").fetchone()[0] == 1
    conn.close()


def test_verification_timestamp_defaults_to_observation_time(provider):
    provider.record_decision(make_decision("verification-time"))
    assert provider.attach_verification(
        "verification-time",
        VerifiedOutcome(task_success=None, verifier_source="deterministic:v1",
                        failure_cause="unverifiable"),
        event_id="verification-time-event",
        observed_at=9876.5,
    )
    conn = sqlite3.connect(provider.db_path)
    assert conn.execute(
        "SELECT verified_at FROM outcomes WHERE route_id='verification-time'"
    ).fetchone()[0] == 9876.5
    conn.close()
