"""SqliteProvider — the in-house Engram-lite memory provider (WS1).

The append-only routing ledger on a ``route_id`` spine, stored in a single
SQLite file (WAL mode, single-writer) at ``data/router-memory.db``:

    decisions   — immutable facts written at decision time (INSERT OR IGNORE:
                  idempotent ingestion, Engram ADR 0004)
    outcomes    — one companion row per decision; lifecycle is forward-only
                  (pending -> closed_turn -> closed_final, never backwards)
    embeddings  — 768-dim float32 little-endian BLOBs, own table/cadence;
                  absent entirely for privacy-pinned turns

Retrieval is a **projection** (Engram ADR 0005): an in-RAM normalized numpy
matrix + aligned metadata, rebuilt only when the underlying rowcounts change
(cheap check), never a per-query SELECT-all. Only ``closed_*`` outcomes are
visible to kNN.

Privacy: instruction TEXT never enters this database — only ``instr_sha256``,
features JSON, and the embedding BLOB. The ``search_document:`` /
``search_query:`` prefixes required by nomic-embed-text live here and ONLY
here (``embed_for_storage`` / ``embed_for_query``).

Fail-safe contract: no public method ever raises; degraded answers are
``None`` / ``False`` / ``[]`` / ``0`` / ``{}``.

CLI:
    PYTHONPATH=src python -m router.memory_sqlite --report [--db PATH]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# memory_ports is the single home of the shared contract (and, post-review, of
# the mandatory nomic prefixes). The parallel-build duck-type fallback that
# lived here was deleted once Unit A landed — dead code is a maintenance trap.
from router.memory_ports import (
    DOCUMENT_PREFIX,
    QUERY_PREFIX,
    DecisionEvent,
    MemoryProvider,
    NeighborTurn,
    OutcomeEvent,
)

_ProviderBase = MemoryProvider

DEFAULT_DB_PATH = Path("data/router-memory.db")

STORAGE_PREFIX = DOCUMENT_PREFIX  # backwards-compatible alias

_LIFECYCLE_RANK = {"pending": 0, "closed_turn": 1, "closed_final": 2}
_CLOSED_STATUSES = ("closed_turn", "closed_final")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    route_id        TEXT PRIMARY KEY,
    ts              REAL,
    session_id      TEXT,
    turn_index      INT,
    source          TEXT,
    instr_sha256    TEXT,
    features_json   TEXT,
    layer           TEXT,
    rung             TEXT,
    cascade         INT,
    score           REAL,
    reason          TEXT,
    classifier_tier TEXT,
    propensity      TEXT,
    policy_version  TEXT,
    classifier_ms   REAL,
    decision_ms     REAL
);
CREATE TABLE IF NOT EXISTS outcomes (
    route_id           TEXT PRIMARY KEY REFERENCES decisions(route_id),
    status             TEXT,
    escalated          INT,
    tripwire_name      TEXT,
    tripwire_type      TEXT,
    edit_failures      INT,
    error_results      INT,
    output_tokens      INT,
    latency_ms         REAL,
    cost_estimate      REAL,
    interrupted        INT,
    user_retried       INT,
    outcome_proxy_hard INT
);
CREATE TABLE IF NOT EXISTS embeddings (
    route_id TEXT PRIMARY KEY REFERENCES decisions(route_id),
    dim      INT,
    vec      BLOB
);
CREATE INDEX IF NOT EXISTS idx_decisions_session ON decisions(session_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_outcomes_status ON outcomes(status);
"""


def _as_nullable_int(value: Optional[bool]) -> Optional[int]:
    """None stays NULL; booleans become 0/1 for SQLite storage."""
    return None if value is None else int(bool(value))


def _as_nullable_bool(value: Optional[int]) -> Optional[bool]:
    return None if value is None else bool(value)


class SqliteProvider(_ProviderBase):
    """MemoryProvider backed by a local SQLite ledger + in-RAM kNN projection."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH, embedder=None):
        self.db_path = Path(db_path)
        self._embedder = embedder  # lazy default: backends.for_role('embedder')
        self._conn: Optional[sqlite3.Connection] = None
        # kNN projection cache: rebuilt when the fingerprint changes.
        self._projection_fingerprint: Optional[tuple] = None
        self._projection_matrix: Optional[np.ndarray] = None
        self._projection_meta: list[dict] = []
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(_SCHEMA)
            conn.commit()
            self._conn = conn
        except Exception:
            self._conn = None  # degraded: every method answers fail-safe

    # ── writes ───────────────────────────────────────────────────────────

    def record_decision(
        self, decision: DecisionEvent, embedding: Optional[list[float]] = None
    ) -> Optional[str]:
        """INSERT the immutable decision + a pending outcomes companion row.

        Idempotent on duplicate route_id (INSERT OR IGNORE). Returns the
        route_id on success (including duplicate re-ingestion), None on failure.
        """
        if self._conn is None:
            return None
        try:
            if decision.instr_sha256 is None:
                # Defense-in-depth (review finding): a pinned/secret turn must
                # never persist an embedding, even if a caller bypasses the
                # facade's gate. Inside the try so garbage input stays fail-safe.
                embedding = None
            self._conn.execute(
                "INSERT OR IGNORE INTO decisions VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    decision.route_id,
                    decision.ts,
                    decision.session_id,
                    decision.turn_index,
                    decision.source,
                    decision.instr_sha256,
                    decision.features_json,
                    decision.layer,
                    decision.rung,
                    int(bool(decision.cascade)),
                    decision.score,
                    decision.reason,
                    decision.classifier_tier,
                    decision.propensity,
                    decision.policy_version,
                    decision.classifier_ms,
                    decision.decision_ms,
                ),
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO outcomes (route_id, status, escalated, "
                "edit_failures, error_results, output_tokens, latency_ms, "
                "cost_estimate, interrupted) "
                "VALUES (?, 'pending', 0, 0, 0, 0, 0.0, 0.0, 0)",
                (decision.route_id,),
            )
            if embedding is not None:
                vector = np.asarray(embedding, dtype="<f4")
                self._conn.execute(
                    "INSERT OR IGNORE INTO embeddings (route_id, dim, vec) "
                    "VALUES (?,?,?)",
                    (decision.route_id, int(vector.shape[0]), vector.tobytes()),
                )
            self._conn.commit()
            return decision.route_id
        except Exception:
            return None

    def attach_outcome(self, route_id: str, outcome: OutcomeEvent) -> bool:
        """Overwrite the outcomes row with the event's fields; lifecycle moves
        forward only (pending -> closed_turn -> closed_final)."""
        if self._conn is None:
            return False
        try:
            new_rank = _LIFECYCLE_RANK.get(outcome.status)
            if new_rank is None:
                return False
            row = self._conn.execute(
                "SELECT status FROM outcomes WHERE route_id = ?", (route_id,)
            ).fetchone()
            if row is None:
                return False
            current_rank = _LIFECYCLE_RANK.get(row[0], 0)
            if new_rank <= current_rank:
                return False  # backwards or same-state transitions rejected
            self._conn.execute(
                "UPDATE outcomes SET status=?, escalated=?, tripwire_name=?, "
                "tripwire_type=?, edit_failures=?, error_results=?, "
                "output_tokens=?, latency_ms=?, cost_estimate=?, interrupted=?, "
                "user_retried=?, outcome_proxy_hard=? WHERE route_id=?",
                (
                    outcome.status,
                    int(bool(outcome.escalated)),
                    outcome.tripwire_name,
                    outcome.tripwire_type,
                    outcome.edit_failures,
                    outcome.error_results,
                    outcome.output_tokens,
                    outcome.latency_ms,
                    outcome.cost_estimate,
                    int(bool(outcome.interrupted)),
                    _as_nullable_int(outcome.user_retried),
                    _as_nullable_int(outcome.outcome_proxy_hard),
                    route_id,
                ),
            )
            self._conn.commit()
            return True
        except Exception:
            return False

    def finalize_turn(
        self, session_id: str, *, prev_interrupted: bool, prev_retried: bool
    ) -> int:
        """Promote the session's most recent open outcome (pending OR
        closed_turn — per the port contract) to closed_final, attaching turn
        N+1's interrupt/retry signal to turn N. Including pending means a turn
        whose per-call outcome never got attached still receives its retry
        signal instead of being stranded. Returns rows finalized (0 or 1)."""
        if self._conn is None:
            return 0
        try:
            row = self._conn.execute(
                "SELECT o.route_id FROM outcomes o "
                "JOIN decisions d ON d.route_id = o.route_id "
                "WHERE d.session_id = ? AND o.status IN ('pending', 'closed_turn') "
                "ORDER BY d.turn_index DESC, d.ts DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            if row is None:
                return 0
            cursor = self._conn.execute(
                "UPDATE outcomes SET status='closed_final', interrupted=?, "
                "user_retried=? WHERE route_id=?",
                (int(bool(prev_interrupted)), int(bool(prev_retried)), row[0]),
            )
            self._conn.commit()
            return cursor.rowcount
        except Exception:
            return 0

    # ── retrieval projection ─────────────────────────────────────────────

    def _projection_current_fingerprint(self) -> Optional[tuple]:
        """Cheap staleness check: (embeddings rowcount, closed-outcome rowcount).

        Catches both new embedding rows and pending->closed promotions without
        scanning any BLOBs."""
        row = self._conn.execute(
            "SELECT (SELECT COUNT(*) FROM embeddings), "
            "(SELECT COUNT(*) FROM outcomes WHERE status IN (?, ?))",
            _CLOSED_STATUSES,
        ).fetchone()
        return (row[0], row[1])

    def _refresh_projection(self) -> None:
        """Rebuild the normalized in-RAM matrix + aligned metadata from the
        ledger (a projection per Engram ADR 0005 — always rebuildable)."""
        rows = self._conn.execute(
            "SELECT e.route_id, e.dim, e.vec, d.rung, d.source, d.ts, "
            "o.escalated, o.outcome_proxy_hard "
            "FROM embeddings e "
            "JOIN decisions d ON d.route_id = e.route_id "
            "JOIN outcomes o ON o.route_id = e.route_id "
            "WHERE o.status IN (?, ?)",
            _CLOSED_STATUSES,
        ).fetchall()
        vectors: list[np.ndarray] = []
        metadata: list[dict] = []
        for route_id, dim, blob, rung, source, ts, escalated, proxy_hard in rows:
            if not isinstance(blob, (bytes, memoryview)) or len(blob) % 4 != 0:
                continue  # truncated/corrupt blob — skip BEFORE frombuffer raises
            vector = np.frombuffer(blob, dtype="<f4")
            if dim and vector.shape[0] != dim:
                continue  # malformed blob — skip, never raise
            if vectors and vector.shape[0] != vectors[0].shape[0]:
                continue  # heterogeneous dim (embedder swap) — can't join projection
            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                continue
            vectors.append(vector / norm)
            metadata.append(
                {
                    "route_id": route_id,
                    "rung": rung,
                    "escalated": bool(escalated),
                    "outcome_proxy_hard": _as_nullable_bool(proxy_hard),
                    "source": source,
                    "ts": ts,
                }
            )
        self._projection_matrix = np.vstack(vectors) if vectors else None
        self._projection_meta = metadata

    def similar_turns(self, embedding: list[float], k: int = 12) -> list[NeighborTurn]:
        """Cosine top-k over closed_* rows via the cached in-RAM projection."""
        if self._conn is None:
            return []
        try:
            fingerprint = self._projection_current_fingerprint()
            if fingerprint != self._projection_fingerprint:
                self._refresh_projection()
                self._projection_fingerprint = fingerprint
            if self._projection_matrix is None:
                return []
            query = np.asarray(embedding, dtype="<f4")
            if query.shape[0] != self._projection_matrix.shape[1]:
                return []  # dimension mismatch — degraded, not an error
            norm = float(np.linalg.norm(query))
            if norm == 0.0:
                return []
            similarities = self._projection_matrix @ (query / norm)
            top = np.argsort(-similarities)[: max(0, int(k))]
            return [
                NeighborTurn(
                    route_id=self._projection_meta[i]["route_id"],
                    similarity=float(similarities[i]),
                    rung=self._projection_meta[i]["rung"],
                    escalated=self._projection_meta[i]["escalated"],
                    outcome_proxy_hard=self._projection_meta[i]["outcome_proxy_hard"],
                    source=self._projection_meta[i]["source"],
                    ts=self._projection_meta[i]["ts"],
                )
                for i in top
            ]
        except Exception:
            return []

    # ── introspection ────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Counts by source, outcome lifecycle coverage, embedding coverage."""
        if self._conn is None:
            return {}
        try:
            by_source = dict(
                self._conn.execute(
                    "SELECT source, COUNT(*) FROM decisions GROUP BY source"
                ).fetchall()
            )
            by_status = dict(
                self._conn.execute(
                    "SELECT status, COUNT(*) FROM outcomes GROUP BY status"
                ).fetchall()
            )
            decision_count = sum(by_source.values())
            embedding_count = self._conn.execute(
                "SELECT COUNT(*) FROM embeddings"
            ).fetchone()[0]
            return {
                "db_path": str(self.db_path),
                "decisions": decision_count,
                "by_source": by_source,
                "outcomes_by_status": by_status,
                "embeddings": embedding_count,
                "embedding_coverage": (
                    embedding_count / decision_count if decision_count else 0.0
                ),
            }
        except Exception:
            return {}

    def health(self) -> bool:
        """True when the database is open and answers a trivial query."""
        if self._conn is None:
            return False
        try:
            self._conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    # ── embedding helpers (the ONLY place the nomic prefixes live) ───────

    def _resolve_embedder(self):
        if self._embedder is None:
            from router import backends

            self._embedder = backends.for_role("embedder")
        return self._embedder

    def _embed_with_prefix(self, prefix: str, text: str) -> Optional[list[float]]:
        try:
            vectors = self._resolve_embedder().embed([prefix + text])
            if not vectors:
                return None
            return list(vectors[0])
        except Exception:
            return None

    def embed_for_storage(self, text: str) -> Optional[list[float]]:
        """Embed instruction text for the ledger ('search_document: ' prefix)."""
        return self._embed_with_prefix(STORAGE_PREFIX, text)

    def embed_for_query(self, text: str) -> Optional[list[float]]:
        """Embed a lookup query ('search_query: ' prefix)."""
        return self._embed_with_prefix(QUERY_PREFIX, text)


# ── reporter ─────────────────────────────────────────────────────────────────


def build(db_path: Path = DEFAULT_DB_PATH) -> str:
    """Markdown status report for the SQLite memory ledger (pure)."""
    provider = SqliteProvider(db_path=db_path)
    statistics = provider.stats()
    lines = [
        "# router-memory — sqlite provider",
        "",
        f"- db: `{db_path}`",
        f"- healthy: {'yes' if provider.health() else 'NO'}",
    ]
    if not statistics:
        lines.append("- stats: unavailable (degraded)")
        return "\n".join(lines) + "\n"
    lines.append(f"- decisions: {statistics['decisions']}")
    for source, count in sorted(statistics["by_source"].items()):
        lines.append(f"  - {source}: {count}")
    lines.append("- outcome lifecycle:")
    for status in ("pending", "closed_turn", "closed_final"):
        lines.append(
            f"  - {status}: {statistics['outcomes_by_status'].get(status, 0)}"
        )
    lines.append(
        f"- embeddings: {statistics['embeddings']} "
        f"(coverage {statistics['embedding_coverage']:.1%})"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB_PATH, help="ledger path"
    )
    parser.add_argument(
        "--report", action="store_true", help="print the markdown status report"
    )
    parser.add_argument(
        "--stats-json", action="store_true", help="print stats() as JSON"
    )
    args = parser.parse_args()
    if args.stats_json:
        print(json.dumps(SqliteProvider(db_path=args.db).stats(), indent=2))
    else:
        print(build(args.db), end="")


if __name__ == "__main__":
    main()
