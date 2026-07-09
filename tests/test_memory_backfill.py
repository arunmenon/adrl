"""Tests for router.memory_backfill — corpus backfill into the memory ledger.

Runs against a small synthetic parquet + stub embedder (never the live
embedding backend): decisions/outcomes/embeddings/graph rows land, secret
sessions are excluded from hashing/embedding, and a re-run is idempotent.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from router.memory_backfill import (
    BACKFILL_PROPENSITY,
    build,
    load_rows,
    load_secret_session_ids,
    load_sim_session_ids,
    run_backfill,
)
from router.memory_sqlite import STORAGE_PREFIX

SIM_SESSION = "sim-session-1"
ORGANIC_SESSION = "organic-session-1"
SECRET_SESSION = "secret-session-1"


class StubEmbedder:
    """Deterministic embedder; records every text it was asked to embed."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.seen: list[str] = []

    def embed(self, texts):
        self.seen.extend(texts)
        return [[float((len(t) + i) % 7) + 0.5 for i in range(self.dim)]
                for t in texts]


def _row(session_id, ts, text, *, source_kind="main", label="user_turn",
         edit_failures=0, error_results=0, interrupted=False, continuations=0):
    return {
        "session_id": session_id,
        "source_kind": source_kind,
        "label": label,
        "instruction_text": text,
        "ts": ts,
        "source_path": f"{session_id}.jsonl",
        "input_tokens": 120,
        "cache_read_tokens": 4000,
        "n_assistant_msgs": 2,
        "output_tokens": 300,
        "duration_ms": 1500,
        "n_error_results": error_results,
        "n_edit_failures": edit_failures,
        "n_continuations": continuations,
        "interrupted": interrupted,
    }


@pytest.fixture()
def corpus(tmp_path: Path):
    """Synthetic parquet + sim ledger + secrets scan in tmp_path."""
    rows = [
        _row(ORGANIC_SESSION, "2026-07-01T10:00:00+00:00", "fix the failing test"),
        _row(ORGANIC_SESSION, "2026-07-01T10:05:00+00:00",
             "now refactor the whole module", edit_failures=1, interrupted=True),
        _row(SIM_SESSION, "2026-07-02T09:00:00+00:00",
             "give me a quick overview of this codebase"),
        _row(SECRET_SESSION, "2026-07-03T08:00:00+00:00",
             "rotate the API key in the env file", error_results=2),
        # Ineligible rows — must be filtered out:
        _row(ORGANIC_SESSION, "2026-07-01T10:10:00+00:00", "subagent work",
             source_kind="subagent"),
        _row(ORGANIC_SESSION, "2026-07-01T10:11:00+00:00", "not a user turn",
             label="assistant_turn"),
        _row(ORGANIC_SESSION, "2026-07-01T10:12:00+00:00", ""),
    ]
    columns = {key: [r[key] for r in rows] for key in rows[0]}
    parquet_path = tmp_path / "turns.parquet"
    pq.write_table(pa.table(columns), parquet_path)

    sim_ledger_path = tmp_path / "sim-ledger.jsonl"
    sim_ledger_path.write_text(
        json.dumps({"source": "simulator", "session_id": SIM_SESSION}) + "\n"
        + json.dumps({"session_ids": ["multi-a"],
                      "turns": [{"step": 1, "session_id": "multi-b"}]}) + "\n")

    secrets_path = tmp_path / "secrets-scan.json"
    secrets_path.write_text(json.dumps(
        {"would_have_pinned": [[SECRET_SESSION, {"n_hits": 3}]]}))

    return {"parquet": parquet_path, "sim_ledger": sim_ledger_path,
            "secrets": secrets_path, "db": tmp_path / "memory.db"}


def _run(corpus, embedder=None, **kwargs):
    return run_backfill(
        parquet_path=corpus["parquet"],
        sim_ledger_path=corpus["sim_ledger"],
        secrets_path=corpus["secrets"],
        db_path=corpus["db"],
        embedder=embedder if embedder is not None else StubEmbedder(),
        **kwargs,
    )


def test_loaders(corpus):
    assert load_sim_session_ids(corpus["sim_ledger"]) == {
        SIM_SESSION, "multi-a", "multi-b"}
    assert load_secret_session_ids(corpus["secrets"]) == {SECRET_SESSION}
    rows = load_rows(corpus["parquet"])
    assert len(rows) == 4  # ineligible rows filtered
    # Deterministic order: sorted by (session_id, ts).
    assert [r["session_id"] for r in rows] == [
        ORGANIC_SESSION, ORGANIC_SESSION, SECRET_SESSION, SIM_SESSION]


def test_backfill_lands_everything(corpus):
    embedder = StubEmbedder()
    stats = _run(corpus, embedder=embedder)

    assert stats["decisions_written"] == 4
    assert stats["by_source"] == {"organic": 3, "simulator": 1}
    assert stats["outcomes_attached"] == 4
    # Secret session excluded from embedding; the other 3 stored.
    assert stats["embeddings_stored"] == 3
    assert stats["secret_turns_excluded_from_embedding"] == 1
    # Every embed call used the mandatory storage prefix, and the secret
    # session's text never reached the embedder.
    assert embedder.seen and all(t.startswith(STORAGE_PREFIX)
                                 for t in embedder.seen)
    assert not any("rotate the API key" in t for t in embedder.seen)

    conn = sqlite3.connect(corpus["db"])
    try:
        assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 4
        assert conn.execute(
            "SELECT COUNT(*) FROM outcomes WHERE status='closed_final'"
        ).fetchone()[0] == 4
        assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 3

        # Secret session: sha NULL, no embeddings row.
        sha, = conn.execute(
            "SELECT instr_sha256 FROM decisions WHERE session_id=?",
            (SECRET_SESSION,)).fetchone()
        assert sha is None
        assert conn.execute(
            "SELECT COUNT(*) FROM embeddings e JOIN decisions d "
            "ON d.route_id=e.route_id WHERE d.session_id=?",
            (SECRET_SESSION,)).fetchone()[0] == 0

        # Non-secret decisions carry a sha and no raw text anywhere.
        shas = [r[0] for r in conn.execute(
            "SELECT instr_sha256 FROM decisions WHERE session_id!=?",
            (SECRET_SESSION,))]
        assert all(isinstance(s, str) and len(s) == 64 for s in shas)
        for features_json, in conn.execute(
                "SELECT features_json FROM decisions"):
            assert "instruction_text" not in features_json
            assert "rotate the API key" not in features_json

        # Provenance + propensity recorded.
        assert conn.execute(
            "SELECT source FROM decisions WHERE session_id=?",
            (SIM_SESSION,)).fetchone()[0] == "simulator"
        propensities = {r[0] for r in conn.execute(
            "SELECT propensity FROM decisions")}
        assert propensities == {BACKFILL_PROPENSITY}

        # Real outcome proxies landed (row 2 had an edit failure + interrupt).
        hard, = conn.execute(
            "SELECT outcome_proxy_hard FROM outcomes o JOIN decisions d "
            "ON d.route_id=o.route_id WHERE d.session_id=? AND d.turn_index=1",
            (ORGANIC_SESSION,)).fetchone()
        assert hard == 1

        # Graph projection landed in the same file.
        assert conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] > 0
        assert conn.execute(
            "SELECT COUNT(*) FROM edges WHERE kind='in_session'"
        ).fetchone()[0] == 4
    finally:
        conn.close()

    assert stats["graph_stats"]["decisions_projected"] == 4
    assert stats["graph_stats"]["closed_outcomes_projected"] == 4


def test_idempotent_rerun(corpus):
    first = _run(corpus)
    second = _run(corpus)
    for key in ("decisions", "embeddings"):
        assert second["provider_stats"][key] == first["provider_stats"][key]
    assert (second["provider_stats"]["outcomes_by_status"]
            == first["provider_stats"]["outcomes_by_status"])
    # Re-attach of an already-final outcome is refused, not duplicated.
    assert second["outcomes_attached"] == 0
    assert (second["graph_stats"]["decisions_projected"]
            == first["graph_stats"]["decisions_projected"])
    conn = sqlite3.connect(corpus["db"])
    try:
        assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 4
        # Projection rebuild is delete+rebuild — edge counts stay stable.
        assert conn.execute(
            "SELECT COUNT(*) FROM edges WHERE kind='in_session'"
        ).fetchone()[0] == 4
    finally:
        conn.close()


def test_limit_and_report(corpus):
    stats = _run(corpus, limit=2)
    assert stats["decisions_written"] == 2
    report = build(stats)
    assert report.startswith("# Memory backfill report")
    assert "## Decisions by source" in report
    assert "## Graph projection" in report
    # PRIVACY: no instruction text in the report, ever.
    for fragment in ("fix the failing test", "rotate the API key",
                     "quick overview"):
        assert fragment not in report
