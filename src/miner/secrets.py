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
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .parser import ParseStats, iter_records, iter_source_files, tool_result_blocks

# ── v2 (2026-07-07, P1-B): precision-tuned ──────────────────────────────────
# Phase-0 audit found env_assignment matched the word "tokens:" in prose (229
# hits, mostly false). v2 requires: (a) real ALL-CAPS env-var key shape,
# case-SENSITIVE (kills "tokens:"), and (b) a high-entropy value. Connection
# strings additionally reject placeholder creds (user:pass, foo:bar, localhost).

# High-confidence literals — these shapes are secrets by construction, no entropy gate.
HIGH_CONFIDENCE: dict[str, re.Pattern] = {
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "bearer_token": re.compile(
        r"\b(?:sk-ant-[a-zA-Z0-9_-]{20,}|sk-[a-zA-Z0-9]{32,}|ghp_[a-zA-Z0-9]{36}|gho_[a-zA-Z0-9]{36}|xox[bp]-[a-zA-Z0-9-]{20,})"
    ),
}

# Candidate shapes that need an entropy check on the captured value (group "val").
_ENV_KEY = r"[A-Z][A-Z0-9]*_?(?:API_?KEY|SECRET(?:_?KEY)?|PASSWORD|PASSWD|ACCESS_?KEY|AUTH_?TOKEN|PRIVATE_?KEY|CLIENT_?SECRET)[A-Z0-9_]*"
ENTROPY_CANDIDATES: dict[str, re.Pattern] = {
    # Case-SENSITIVE: only ALL-CAPS env-style keys, not the prose word "tokens".
    "env_assignment": re.compile(
        rf"\b{_ENV_KEY}\s*[=:]\s*['\"]?(?P<val>[A-Za-z0-9+/=_-]{{16,}})['\"]?"
    ),
    "connection_string_cred": re.compile(
        r"\b(?:postgres|postgresql|mysql|mongodb(?:\+srv)?|redis|amqp)://"
        r"[^\s/:@]+:(?P<val>[^\s@]+)@(?P<host>[^\s/:]+)"
    ),
}

PLACEHOLDER_CREDS = {
    "pass", "password", "passwd", "changeme", "secret", "example", "test",
    "user", "username", "foo", "bar", "xxx", "your_password", "placeholder",
    "admin", "root", "123456", "postgres", "mysql", "redis",
}
PLACEHOLDER_HOSTS = {"localhost", "127.0.0.1", "example.com", "host", "db", "database"}
ENTROPY_MIN = 3.0   # bits/char; real keys ~4-6, dictionary words ~2-3


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def scan_text(text: str) -> list[str]:
    hits: list[str] = []
    for name, pat in HIGH_CONFIDENCE.items():
        if pat.search(text):
            hits.append(name)
    for name, pat in ENTROPY_CANDIDATES.items():
        for m in pat.finditer(text):
            val = m.group("val")
            if name == "connection_string_cred":
                host = (m.groupdict().get("host") or "").lower()
                if val.lower() in PLACEHOLDER_CREDS or host in PLACEHOLDER_HOSTS:
                    continue  # doc example, not a live credential
            if shannon_entropy(val) >= ENTROPY_MIN and len(val) >= 12:
                hits.append(name)
                break
    return hits


# Back-compat alias for callers/tests that referenced the old name.
SECRET_PATTERNS = {**HIGH_CONFIDENCE, **ENTROPY_CANDIDATES}


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
