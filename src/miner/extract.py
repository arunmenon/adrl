"""A4 — walk the corpus snapshot and emit one flat row per turn.

Usage:
    PYTHONPATH=src .venv/bin/python -m miner.extract \
        --corpus data/corpus --out data/turns.parquet

Writes parquet via pyarrow; falls back to CSV if pyarrow is unavailable.
Also writes data/parse-stats.json for the A1 acceptance check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .labels import label_turn, turn_flags
from .parser import ParseStats, iter_records, iter_source_files
from .turns import Turn, assemble_turns


def turn_row(turn: Turn) -> dict[str, Any]:
    flags = turn_flags(turn)
    return {
        "session_id": turn.session_id,
        "project": turn.project,
        "source_kind": turn.source_kind,
        "agent_id": turn.agent_id,
        "workflow_id": turn.workflow_id,
        "prompt_id": turn.prompt_id,
        "ts": turn.ts,
        "version": turn.version,
        "git_branch": turn.git_branch,
        "label": label_turn(turn),
        "instruction_text": turn.instruction_text,
        "instruction_chars": len(turn.instruction_text),
        "n_continuations": turn.n_continuations,
        "n_assistant_msgs": turn.n_assistant_msgs,
        "n_tool_uses": turn.n_tool_uses,
        "tools_used": json.dumps(turn.tools_used, sort_keys=True),
        "parallel_tool_use_msgs": turn.parallel_tool_use_msgs,
        "models": ",".join(sorted(turn.models)),
        "input_tokens": turn.input_tokens,
        "output_tokens": turn.output_tokens,
        "cache_read_tokens": turn.cache_read_tokens,
        "cache_creation_tokens": turn.cache_creation_tokens,
        "n_error_results": turn.n_error_results,
        "n_edit_failures": turn.n_edit_failures,
        "final_stop_reason": turn.final_stop_reason,
        "duration_ms": turn.duration_ms,
        "message_count": turn.message_count,
        **flags,
    }


def extract(corpus: Path, out: Path, force_csv: bool = False) -> dict[str, Any]:
    stats = ParseStats()
    rows: list[dict[str, Any]] = []
    seen_uuids: set[str] = set()
    t0 = time.time()

    for source in iter_source_files(corpus):
        records: list[dict[str, Any]] = []
        for rec in iter_records(source, stats):
            u = rec.get("uuid")
            if isinstance(u, str):
                if u in seen_uuids:
                    stats.duplicate_uuids += 1
                    continue
                seen_uuids.add(u)
            records.append(rec)
        for turn in assemble_turns(source, records):
            rows.append(turn_row(turn))

    elapsed = time.time() - t0
    summary = {
        "parse_stats": stats.as_dict(),
        "turns": len(rows),
        "elapsed_s": round(elapsed, 1),
        "labels": {},
        "requests_transcript_visible": {},
    }
    label_counts: dict[str, int] = {}
    cont_by_label: dict[str, int] = {}
    for r in rows:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
        cont_by_label[r["label"]] = cont_by_label.get(r["label"], 0) + r["n_continuations"]
    summary["labels"] = label_counts
    # Request-level view: each turn is 1 initiating request + N continuations.
    summary["requests_transcript_visible"] = {
        lbl: {"turns": n, "continuations": cont_by_label.get(lbl, 0)}
        for lbl, n in sorted(label_counts.items())
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    wrote = _write(rows, out, force_csv)
    summary["output"] = str(wrote)

    stats_path = out.parent / "parse-stats.json"
    stats_path.write_text(json.dumps(summary, indent=2))
    return summary


def _write(rows: list[dict[str, Any]], out: Path, force_csv: bool) -> Path:
    if not force_csv:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pylist(rows)
            pq.write_table(table, out)
            return out
        except ImportError:
            pass
    import csv

    csv_out = out.with_suffix(".csv")
    if rows:
        with csv_out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return csv_out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--out", type=Path, default=Path("data/turns.parquet"))
    ap.add_argument("--csv", action="store_true", help="force CSV output")
    args = ap.parse_args()

    if not args.corpus.is_dir():
        print(f"corpus not found: {args.corpus}", file=sys.stderr)
        return 1

    summary = extract(args.corpus, args.out, force_csv=args.csv)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
