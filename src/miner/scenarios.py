"""A5 — scenario fingerprint matchers over the corpus (scenarios doc S1-S15).

Validation bar (scenarios doc): >=3 real traces per scenario, or an explicit
verdict that the corpus cannot answer it (needs wire capture / simulator) or
that it is FALSIFIED for this setup.

Outputs:
  reports/scenario-validation.md        committed summary (counts + verdicts, no payloads)
  data/scenario-matches/S<nn>/*.jsonl   raw records of matched turns (gitignored)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from .parser import ParseStats, iter_records, iter_source_files, tool_use_blocks

EASY_INTENT = re.compile(
    r"\b(commit message|summar|rename|readme|title|typo|explain|what does|list the)\b",
    re.IGNORECASE,
)
HARD_INTENT = re.compile(
    r"\b(refactor|migrat|redesign|architect|across (the )?(codebase|services|modules))\b",
    re.IGNORECASE,
)
COMPLETION_PHRASE = re.compile(
    r"\b(that'?s done|great,? now|perfect,? now|ok(ay)? now|next,? (let'?s|please))\b",
    re.IGNORECASE,
)

TRACE_CAP = 5  # raw traces extracted per scenario


from router.canon import canonical_call  # single source of truth (dependency-light)


def detect_loops(corpus: Path) -> list[dict[str, Any]]:
    """Record-level pass: >=3 identical canonicalized tool calls within one turn."""
    stats = ParseStats()
    loops: list[dict[str, Any]] = []
    for source in iter_source_files(corpus):
        prompt_calls: dict[str, list[str]] = defaultdict(list)
        current_prompt = "_none"
        for rec in iter_records(source, stats):
            if rec.get("type") == "user" and rec.get("promptId"):
                current_prompt = rec["promptId"]
            elif rec.get("type") == "assistant":
                for blk in tool_use_blocks(rec.get("message")):
                    prompt_calls[current_prompt].append(
                        canonical_call(blk.get("name", "?"), blk.get("input"))
                    )
        for pid, calls in prompt_calls.items():
            counts: dict[str, int] = {}
            for i, h in enumerate(calls):
                # same call >=3 times within a sliding window of 6 actions
                window = calls[max(0, i - 5) : i + 1]
                if window.count(h) >= 3:
                    counts[h] = counts.get(h, 0) + 1
            if counts:
                loops.append(
                    {
                        "source_path": str(source.path.relative_to(corpus)),
                        "session_id": source.session_id,
                        "prompt_id": pid,
                        "n_loopy_calls": sum(counts.values()),
                    }
                )
                break  # one entry per file is enough for the report
    return loops


def match_scenarios(rows: list[dict[str, Any]], corpus: Path, secrets_path: Path) -> dict[str, dict]:
    by_session: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_session[r["session_id"]].append(r)
    for turns in by_session.values():
        turns.sort(key=lambda r: r["ts"] or "")

    main = [r for r in rows if r["source_kind"] == "main"]
    user_turns = [r for r in main if r["label"] == "user_turn"]

    def ctx_estimate(r: dict) -> float:
        n = max(r["n_assistant_msgs"], 1)
        return (r["cache_read_tokens"] + r["input_tokens"]) / n

    s: dict[str, dict] = {}

    s["S2"] = {
        "name": "one turn, many requests (sticky hot path)",
        "matches": [r for r in user_turns if r["n_continuations"] >= 5
                    and len(json.loads(r["tools_used"])) >= 2
                    and r["final_stop_reason"] == "end_turn"],
    }
    s["S3"] = {
        "name": "edit-apply trip-wire",
        "matches": [r for r in rows if r["n_edit_failures"] >= 1],
        "note": "2-strike (would-have-escalated): "
                f"{sum(1 for r in rows if r['n_edit_failures'] >= 2)} turns",
    }
    s["S4"] = {
        "name": "identical-call loops",
        "matches": detect_loops(corpus),
        "raw_records": False,
    }
    interrupted_then_retry = []
    for turns in by_session.values():
        for i, r in enumerate(turns):
            if r["interrupted"] and i + 1 < len(turns) and turns[i + 1]["label"] == "user_turn":
                interrupted_then_retry.append(turns[i + 1])
    s["S6"] = {"name": "interrupt-then-rephrase", "matches": interrupted_then_retry}
    s["S8"] = {
        "name": "easy intent inside huge context",
        "matches": [r for r in user_turns
                    if EASY_INTENT.search(r["instruction_text"]) and ctx_estimate(r) > 50_000],
    }
    secrets = json.loads(secrets_path.read_text()) if secrets_path.exists() else {}
    s["S9"] = {
        "name": "secret exposure (privacy pin)",
        "matches": [{"session_id": k, **v} for k, v in secrets.get("would_have_pinned", [])],
        "raw_records": False,
    }
    s["S10"] = {
        "name": "hard direct-to-frontier intents",
        "matches": [r for r in user_turns if HARD_INTENT.search(r["instruction_text"])],
    }
    s["S12"] = {
        "name": "parallel tool calls",
        "matches": [r for r in rows if r["parallel_tool_use_msgs"] > 0],
    }
    s["S13"] = {
        "name": "subagent spawn (Task tool in main session)",
        "matches": [r for r in main if "Task" in json.loads(r["tools_used"])
                    or "Agent" in json.loads(r["tools_used"])],
    }
    s["S14"] = {
        "name": "auto-compaction",
        "matches": [r for r in rows if r["label"] == "utility:compaction"],
    }
    s["S15b"] = {
        "name": "mid-turn model switch (invariant: must be ZERO)",
        "matches": [r for r in rows
                    if len([m for m in r["models"].split(",") if m and m != "<synthetic>"]) > 1],
        "invariant": True,
    }
    boundary = []
    for turns in by_session.values():
        for r in turns:
            if r["label"] == "user_turn" and COMPLETION_PHRASE.search(r["instruction_text"]):
                boundary.append(r)
    s["S15a"] = {"name": "episode boundary phrases", "matches": boundary}
    s["S11"] = {
        "name": "Codex dialect (absence check)",
        "matches": [],
        "verdict": "FALSIFIED for this setup — corpus is 100% Claude Code; defer per plan D4",
    }
    s["S1"] = {"name": "sidecar utility burst", "matches": [],
               "verdict": "wire-only — needs workstream B captures"}
    s["S5"] = {"name": "malformed tool call (local model)", "matches": [],
               "verdict": "needs workstream C (Ollama traffic via B proxy)"}
    s["S7"] = {"name": "infra fallback", "matches": [],
               "verdict": "needs workstream C (induced endpoint failure)"}
    return s


def extract_traces(scenario_id: str, matches: list[dict], corpus: Path, out_root: Path) -> int:
    """Dump raw records of up to TRACE_CAP matched turns (raw, local-only per D5)."""
    written = 0
    out_dir = out_root / scenario_id
    for m in matches[:TRACE_CAP]:
        src, pid = m.get("source_path"), m.get("prompt_id")
        if not src or not pid:
            continue
        path = corpus / src
        if not path.exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / f"trace{written + 1}.jsonl").open("w") as out:
            active = False
            for line in path.open():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("type") == "user" and rec.get("promptId"):
                    active = rec["promptId"] == pid
                if active and rec.get("type") in ("user", "assistant", "system"):
                    out.write(line)
        written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--turns", type=Path, default=Path("data/turns.parquet"))
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--secrets", type=Path, default=Path("data/secrets-scan.json"))
    ap.add_argument("--traces-out", type=Path, default=Path("data/scenario-matches"))
    ap.add_argument("--report", type=Path, default=Path("reports/scenario-validation.md"))
    args = ap.parse_args()

    import pyarrow.parquet as pq

    rows = pq.read_table(args.turns).to_pylist()
    scenarios = match_scenarios(rows, args.corpus, args.secrets)

    lines = [
        "# Scenario Validation — corpus pass (A5)",
        "",
        "Bar: >=3 real traces, or an explicit verdict. Raw traces (secrets included)",
        f"live under `{args.traces_out}/` — gitignored, local only (plan D5).",
        "",
        "| Scenario | Matches | Status | Notes |",
        "|---|---|---|---|",
    ]
    for sid in sorted(scenarios, key=lambda x: (len(x), x)):
        sc = scenarios[sid]
        n = len(sc["matches"])
        traces = 0
        if n and sc.get("raw_records", True) and "prompt_id" in (sc["matches"][0] or {}):
            traces = extract_traces(sid, sc["matches"], args.corpus, args.traces_out)
        if "verdict" in sc:
            status = sc["verdict"]
        elif sc.get("invariant"):
            status = "INVARIANT HOLDS (0 violations)" if n == 0 else f"VIOLATED x{n} — investigate"
        elif n >= 3:
            status = f"VALIDATED ({traces} traces extracted)"
        elif n > 0:
            status = f"WEAK ({n} < 3) — top up via B/C"
        else:
            status = "NO MATCHES — falsify or top up via B/C"
        lines.append(f"| {sid} | {n} | {status} | {sc['name']} |")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
