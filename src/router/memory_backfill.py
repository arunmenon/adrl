"""One-time corpus backfill: seed the router memory from data/turns.parquet (WS1, Unit D).

Replays every captured main-session user turn (source_kind=='main',
label=='user_turn', non-empty instruction_text; ~750 rows) through the real
feature extractor and the heuristics-only policy (``route_turn`` with
``classifier=None`` — deterministic and fast; recorded with
``propensity='backfill_heuristic'`` so WS4 can distinguish these from live
classifier-consulted decisions), then writes the resulting DecisionEvent +
closed_final OutcomeEvent (with the row's REAL outcome proxies) into the
SQLite ledger via the RouterMemory facade, and finally rebuilds the
graph-lite projection.

Session provenance: session_ids found in data/sim-ledger.jsonl are recorded
as source='simulator'; everything else is 'organic'.

Privacy (absolute): instruction text is transient — it is embedded in RAM
(``search_document:`` prefix, batched) and hashed, but NEVER stored and NEVER
printed in the report. Sessions flagged by data/secrets-scan.json get
``instr_sha256`` NULL and NO embedding row at all.

Idempotent: route_ids are deterministic (sha256 of session_id + turn index),
decisions/embeddings are INSERT OR IGNORE, and closed_final outcomes refuse
re-attachment — a re-run leaves every count unchanged.

Usage:
  PYTHONPATH=src .venv/bin/python -m router.memory_backfill [--limit N] [--db PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from router.features import extract
from router.graph_lite import project_from_ledger
from router.memory_facade import RouterMemory
from router.memory_ports import OutcomeEvent
from router.memory_sqlite import STORAGE_PREFIX, SqliteProvider
from router.outcomes import FailureCause, outcome_proxy_hard, task_signal_hard
from router.policy import SessionState, route_turn

DEFAULT_PARQUET_PATH = Path("data/turns.parquet")
DEFAULT_SIM_LEDGER_PATH = Path("data/sim-ledger.jsonl")
DEFAULT_SECRETS_PATH = Path("data/secrets-scan.json")
DEFAULT_DB_PATH = Path("data/router-memory.db")
DEFAULT_EMBED_BATCH_SIZE = 32

BACKFILL_PROPENSITY = "backfill_heuristic"


# ── corpus loading ───────────────────────────────────────────────────────────


def load_sim_session_ids(sim_ledger_path: Path) -> set[str]:
    """session_ids that belong to simulator runs (single- and multi-turn
    ledger record shapes: ``session_id``, ``session_ids``, ``turns[].session_id``)."""
    session_ids: set[str] = set()
    if not sim_ledger_path.exists():
        return session_ids
    with open(sim_ledger_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("session_id"):
                session_ids.add(str(record["session_id"]))
            for sid in record.get("session_ids") or []:
                session_ids.add(str(sid))
            for turn in record.get("turns") or []:
                if isinstance(turn, dict) and turn.get("session_id"):
                    session_ids.add(str(turn["session_id"]))
    return session_ids


def load_secret_session_ids(secrets_path: Path) -> set[str]:
    """session_ids the secrets scan would have privacy-pinned — these turns
    get NO instruction hash and NO embedding."""
    if not secrets_path.exists():
        return set()
    try:
        scan = json.loads(secrets_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()
    pinned = scan.get("would_have_pinned") or []
    session_ids: set[str] = set()
    for entry in pinned:
        if isinstance(entry, (list, tuple)) and entry:
            session_ids.add(str(entry[0]))
        elif isinstance(entry, str):
            session_ids.add(entry)
    return session_ids


def _parse_ts(value: Any) -> float:
    """ISO-8601 string (parquet ``ts``) -> epoch seconds; 0.0 when unparseable."""
    if isinstance(value, (int, float)):
        return float(value)
    if not value:
        return 0.0
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        return 0.0


def load_rows(parquet_path: Path, limit: Optional[int] = None) -> list[dict]:
    """Eligible corpus rows in deterministic order (session_id, ts).

    Eligibility: source_kind=='main', label=='user_turn', non-empty
    instruction_text. ``--limit N`` truncates AFTER the deterministic sort so
    smoke runs are reproducible.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    row_count = table.num_rows
    rows: list[dict] = []
    for i in range(row_count):
        if columns["source_kind"][i] != "main":
            continue
        if columns["label"][i] != "user_turn":
            continue
        text = columns["instruction_text"][i]
        if not text:
            continue
        rows.append({name: columns[name][i] for name in columns})
    rows.sort(key=lambda r: (str(r["session_id"]), str(r.get("ts") or ""),
                             str(r.get("source_path") or "")))
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return rows


# ── replay ───────────────────────────────────────────────────────────────────


def _context_estimate(row: dict) -> int:
    """Per-request context estimate: (cache_read + input) / n_assistant_msgs.

    The parquet aggregates token counts over the whole turn; dividing by the
    number of assistant messages approximates the context of one request."""
    total = (row.get("cache_read_tokens") or 0) + (row.get("input_tokens") or 0)
    messages = row.get("n_assistant_msgs") or 0
    return int(total / messages) if messages else int(total)


def _route_id_for(session_id: str, turn_index: int) -> str:
    """Deterministic route_id — the idempotency key for re-runs."""
    return hashlib.sha256(
        f"backfill:{session_id}:{turn_index}".encode("utf-8")).hexdigest()[:32]


def _outcome_for(row: dict) -> OutcomeEvent:
    """closed_final outcome from the row's REAL execution proxies."""
    proxy_row = {
        "n_edit_failures": row.get("n_edit_failures") or 0,
        "n_error_results": row.get("n_error_results") or 0,
        "interrupted": bool(row.get("interrupted")),
        "n_continuations": row.get("n_continuations") or 0,
    }
    continuation_count = int(proxy_row["n_continuations"])
    if proxy_row["interrupted"]:
        failure_cause = FailureCause.USER_ABORT.value
    elif continuation_count >= 10:
        failure_cause = FailureCause.TASK_CAPABILITY.value
    elif proxy_row["n_edit_failures"] or proxy_row["n_error_results"]:
        # Historical rows cannot distinguish ordinary command errors from
        # harness dialect failures. Preserve friction but do not invent a task
        # capability label.
        failure_cause = FailureCause.UNVERIFIABLE.value
    else:
        failure_cause = None
    return OutcomeEvent(
        status="closed_final",
        escalated=False,
        tripwire_name=None,
        tripwire_type=None,
        edit_failures=int(proxy_row["n_edit_failures"]),
        error_results=int(proxy_row["n_error_results"]),
        output_tokens=int(row.get("output_tokens") or 0),
        latency_ms=float(row.get("duration_ms") or 0.0),
        cost_estimate=0.0,
        interrupted=bool(proxy_row["interrupted"]),
        user_retried=None,
        outcome_proxy_hard=outcome_proxy_hard(proxy_row),
        continuation_count=continuation_count,
        failure_cause=failure_cause,
        task_signal_hard=task_signal_hard(
            tripwire_type="quality" if proxy_row["interrupted"] else None,
            continuation_count=continuation_count,
        ),
    )


def _embed_batch(embedder: Any, texts: list[str]) -> list[Optional[list[float]]]:
    """Batch-embed TRANSIENT instruction texts for storage (the mandatory
    'search_document: ' prefix, imported from its single home in
    memory_sqlite). Fail-safe: any failure degrades to per-text None."""
    if not texts:
        return []
    try:
        vectors = embedder.embed([STORAGE_PREFIX + text for text in texts])
    except Exception:
        vectors = None
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        return [None] * len(texts)
    return [list(v) if isinstance(v, (list, tuple)) and len(v) else None
            for v in vectors]


def run_backfill(
    parquet_path: Path = DEFAULT_PARQUET_PATH,
    sim_ledger_path: Path = DEFAULT_SIM_LEDGER_PATH,
    secrets_path: Path = DEFAULT_SECRETS_PATH,
    db_path: Path = DEFAULT_DB_PATH,
    *,
    limit: Optional[int] = None,
    embedder: Any = None,
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> dict:
    """Replay the corpus into the memory ledger; return a stats dict.

    ``embedder`` is injectable for tests; the default is the WS0 embedding
    port, ``backends.for_role('embedder')``.
    """
    started = time.monotonic()
    if embedder is None:
        from router.backends import for_role
        embedder = for_role("embedder")

    sim_sessions = load_sim_session_ids(sim_ledger_path)
    secret_sessions = load_secret_session_ids(secrets_path)
    rows = load_rows(parquet_path, limit=limit)

    provider = SqliteProvider(db_path=db_path)
    memory = RouterMemory(providers=[provider])

    # Pass 1 — build privacy-gated events (per-session turn indexes + previous
    # turn's trajectory signals feed the feature extractor).
    events: list[tuple[Any, bool, str, dict]] = []  # (event, embed_allowed, text, row)
    turn_index_by_session: dict[str, int] = {}
    previous_row_by_session: dict[str, dict] = {}
    for row in rows:
        session_id = str(row["session_id"])
        turn_index = turn_index_by_session.get(session_id, 0)
        turn_index_by_session[session_id] = turn_index + 1
        previous = previous_row_by_session.get(session_id)
        text = str(row["instruction_text"])
        is_secret = session_id in secret_sessions
        features = extract(
            text,
            context_tokens=_context_estimate(row),
            turn_index=turn_index,
            recent_errors=int((previous or {}).get("n_error_results") or 0),
            recent_edit_failures=int((previous or {}).get("n_edit_failures") or 0),
            prev_turn_interrupted=bool((previous or {}).get("interrupted")),
            privacy_pinned=is_secret,
        )
        route = route_turn(features, SessionState(session_id=session_id),
                           classifier=None)  # heuristics-only: deterministic
        event, embedding_allowed = memory.make_decision_event(
            text, features, route,
            session_id=session_id,
            turn_index=turn_index,
            source="simulator" if session_id in sim_sessions else "organic",
            route_id=_route_id_for(session_id, turn_index),
            propensity=BACKFILL_PROPENSITY,
            ts=_parse_ts(row.get("ts")),
        )
        events.append((event, embedding_allowed, text, row))
        previous_row_by_session[session_id] = row

    # Pass 2 — batch embeddings for the allowed (non-secret) events only.
    # Instruction text is TRANSIENT here: embedded and discarded, never stored.
    embeddings: dict[int, Optional[list[float]]] = {}
    allowed_indices = [i for i, (_, allowed, _, _) in enumerate(events) if allowed]
    for batch_start in range(0, len(allowed_indices), max(1, embed_batch_size)):
        batch = allowed_indices[batch_start:batch_start + max(1, embed_batch_size)]
        vectors = _embed_batch(embedder, [events[i][2] for i in batch])
        for i, vector in zip(batch, vectors):
            embeddings[i] = vector

    # Pass 3 — write decisions + closed_final outcomes through the facade.
    decisions_written = 0
    outcomes_attached = 0
    embeddings_stored = 0
    by_source: dict[str, int] = {}
    secret_turns = 0
    for i, (event, embedding_allowed, _text, row) in enumerate(events):
        if not embedding_allowed:
            secret_turns += 1
        embedding = embeddings.get(i)
        if memory.record_decision(event, embedding) is not None:
            decisions_written += 1
            by_source[event.source] = by_source.get(event.source, 0) + 1
            if embedding is not None and embedding_allowed:
                embeddings_stored += 1
        if memory.attach_outcome(event.route_id, _outcome_for(row)):
            outcomes_attached += 1

    try:
        graph_stats = project_from_ledger(db_path)
    except Exception:
        # The graph is a rebuildable projection — a projection failure must not
        # fail the backfill (review finding). Re-run graph_lite standalone later.
        graph_stats = {}
    provider_stats = provider.stats()

    return {
        "rows_eligible": len(rows),
        "decisions_written": decisions_written,
        "by_source": by_source,
        "outcomes_attached": outcomes_attached,
        "embeddings_stored": embeddings_stored,
        "secret_turns_excluded_from_embedding": secret_turns,
        "sim_sessions_known": len(sim_sessions),
        "secret_sessions_known": len(secret_sessions),
        "provider_stats": provider_stats,
        "graph_stats": graph_stats,
        "elapsed_s": round(time.monotonic() - started, 2),
        "db_path": str(db_path),
    }


# ── report (pure; counts only — NEVER any instruction text) ──────────────────


def build(stats: dict) -> str:
    """Markdown backfill report. Counts and coverage only."""
    provider_stats = stats.get("provider_stats") or {}
    graph_stats = stats.get("graph_stats") or {}
    outcomes_by_status = provider_stats.get("outcomes_by_status") or {}
    decisions_total = provider_stats.get("decisions") or 0
    lines = [
        "# Memory backfill report",
        "",
        f"- db: `{stats.get('db_path', '')}`",
        f"- elapsed: {stats.get('elapsed_s', 0.0)}s",
        f"- eligible corpus rows: {stats.get('rows_eligible', 0)}",
        f"- decisions written this run: {stats.get('decisions_written', 0)}",
        "",
        "## Decisions by source (this run)",
        "",
        "| source | count |",
        "|---|---|",
    ]
    for source in sorted(stats.get("by_source") or {}):
        lines.append(f"| {source} | {stats['by_source'][source]} |")
    lines += [
        "",
        "## Outcome coverage (ledger)",
        "",
        "| status | count |",
        "|---|---|",
    ]
    for status in ("pending", "closed_turn", "closed_final"):
        lines.append(f"| {status} | {outcomes_by_status.get(status, 0)} |")
    embedding_count = provider_stats.get("embeddings", 0)
    coverage = provider_stats.get("embedding_coverage", 0.0)
    lines += [
        "",
        "## Embedding coverage (ledger)",
        "",
        f"- decisions: {decisions_total}",
        f"- embeddings: {embedding_count} (coverage {coverage:.1%})",
        f"- secret turns (no hash, no embedding): "
        f"{stats.get('secret_turns_excluded_from_embedding', 0)}",
        "",
        "## Graph projection",
        "",
        f"- decisions projected: {graph_stats.get('decisions_projected', 0)}",
        f"- closed outcomes projected: "
        f"{graph_stats.get('closed_outcomes_projected', 0)}",
        "",
        "| kind | nodes |",
        "|---|---|",
    ]
    for kind in sorted(graph_stats.get("nodes") or {}):
        lines.append(f"| {kind} | {graph_stats['nodes'][kind]} |")
    lines += ["", "| kind | edges |", "|---|---|"]
    for kind in sorted(graph_stats.get("edges") or {}):
        lines.append(f"| {kind} | {graph_stats['edges'][kind]} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N eligible rows (smoke)")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH,
                        help="ledger database path")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET_PATH)
    parser.add_argument("--sim-ledger", type=Path, default=DEFAULT_SIM_LEDGER_PATH)
    parser.add_argument("--secrets", type=Path, default=DEFAULT_SECRETS_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_EMBED_BATCH_SIZE,
                        help="embedding batch size")
    args = parser.parse_args()
    stats = run_backfill(
        parquet_path=args.parquet,
        sim_ledger_path=args.sim_ledger,
        secrets_path=args.secrets,
        db_path=args.db,
        limit=args.limit,
        embed_batch_size=args.batch_size,
    )
    print(build(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
