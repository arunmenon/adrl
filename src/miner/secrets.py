"""A8 — secrets scan over tool_result contents (scenario S9, privacy-pin stats).

Answers: how often would the router's one-way privacy pin have fired, and at
what turn index — i.e. how real is the pin-vs-context-growth collision (design
doc §5.8)?

Runs its own streaming pass (record-level detail is not in turns.parquet).
Output stays in data/ — it literally locates secrets; never commit it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from .parser import ParseStats, iter_records, iter_source_files, tool_result_blocks

# Cheap regex layer (design doc §13.4 "both cheap variants" — this is variant 1).
SECRET_PATTERNS: dict[str, re.Pattern] = {
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "connection_string_cred": re.compile(
        r"\b(?:postgres|postgresql|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s/:@]+:[^\s@]+@"
    ),
    "bearer_ish_token": re.compile(
        r"\b(?:sk-ant-[a-zA-Z0-9_-]{20,}|sk-[a-zA-Z0-9]{32,}|ghp_[a-zA-Z0-9]{36}|gho_[a-zA-Z0-9]{36}|xox[bp]-[a-zA-Z0-9-]{20,})"
    ),
    "env_assignment": re.compile(
        r"\b(?:API_KEY|SECRET|PASSWORD|TOKEN|PRIVATE_KEY|ACCESS_KEY)[A-Z_]*\s*[=:]\s*['\"]?[A-Za-z0-9+/_-]{16,}",
        re.IGNORECASE,
    ),
}


def scan_text(text: str) -> list[str]:
    return [name for name, pat in SECRET_PATTERNS.items() if pat.search(text)]


def scan(corpus: Path, out: Path) -> dict[str, Any]:
    stats = ParseStats()
    sessions: dict[str, dict[str, Any]] = {}

    for source in iter_source_files(corpus):
        turn_index = 0
        current_prompt = None
        for rec in iter_records(source, stats):
            if rec.get("type") != "user":
                continue
            pid = rec.get("promptId")
            if pid and pid != current_prompt:
                current_prompt = pid
                turn_index += 1
            for blk in tool_result_blocks(rec.get("message")):
                hits = scan_text(str(blk.get("content", "")))
                if not hits:
                    continue
                sid = rec.get("sessionId") or source.session_id
                entry = sessions.setdefault(
                    sid,
                    {
                        "project": source.project,
                        "kind": source.kind,
                        "source_path": str(source.path.relative_to(corpus)),
                        "first_hit_turn_index": turn_index,
                        "pattern_hits": {},
                        "n_hits": 0,
                    },
                )
                entry["n_hits"] += 1
                for h in hits:
                    entry["pattern_hits"][h] = entry["pattern_hits"].get(h, 0) + 1

    summary = {
        "sessions_scanned_files": stats.files,
        "sessions_with_secrets": len(sessions),
        "would_have_pinned": sorted(
            sessions.items(), key=lambda kv: -kv[1]["n_hits"]
        ),
        "pattern_totals": {},
        "pin_turn_indices": [s["first_hit_turn_index"] for s in sessions.values()],
    }
    totals: dict[str, int] = {}
    for s in sessions.values():
        for k, v in s["pattern_hits"].items():
            totals[k] = totals.get(k, 0) + v
    summary["pattern_totals"] = totals

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--out", type=Path, default=Path("data/secrets-scan.json"))
    args = ap.parse_args()
    s = scan(args.corpus, args.out)
    print(
        f"sessions with secret-bearing tool_results: {s['sessions_with_secrets']} "
        f"(pattern totals: {s['pattern_totals']})"
    )
    print(f"detail -> {args.out} (gitignored; contains secret locations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
