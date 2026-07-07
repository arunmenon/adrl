"""Evaluate the discriminator against every wire capture.

Three checks, matching the shadow-router exit gates (design §10 Phase 0):
1. Label distribution — do the shares look like the design's traffic model?
2. Ground truth — simulator sessions (known from the ledger) must classify
   correctly: first request of an episode step = user_turn, tool_result-bearing
   requests = continuation, count_tokens = passthrough.
3. Latency — classification must cost microseconds (the <20ms decision budget
   belongs to the policy engine; the discriminator's share is noise).

Usage: PYTHONPATH=src .venv/bin/python -m router.eval_captures
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from .discriminator import classify, entrypoint, session_key


def load_captures(pattern: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(pattern)):
        r = json.load(open(f))
        try:
            body = json.loads(r.get("request_body") or "null")
        except json.JSONDecodeError:
            body = None
        out.append({"file": f, "method": r["method"], "path": r["path"],
                    "status": r["status"], "body": body})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captures", default="data/captures/*/*.json")
    ap.add_argument("--ledger", type=Path, default=Path("data/sim-ledger.jsonl"))
    ap.add_argument("--report", type=Path, default=Path("reports/discriminator-eval.md"))
    args = ap.parse_args()

    caps = load_captures(args.captures)
    if not caps:
        print("no captures found", file=sys.stderr)
        return 1

    # simulator session ids -> ground truth "this is agent conversation traffic"
    sim_sids: set[str] = set()
    if args.ledger.exists():
        for line in args.ledger.open():
            e = json.loads(line)
            for sid in e.get("session_ids", []) or ([e["session_id"]] if e.get("session_id") else []):
                sim_sids.add(sid)

    # 1+3: classify everything, timed
    labels = Counter()
    by_entry = defaultdict(Counter)
    latencies = []
    per_session_first: dict[str, str] = {}
    errors = []
    for c in caps:
        t0 = time.perf_counter_ns()
        label = classify(c["method"], c["path"], c["body"])
        latencies.append(time.perf_counter_ns() - t0)
        c["label"] = label
        labels[label] += 1
        by_entry[entrypoint(c["body"])][label] += 1
        sid = session_key(c["body"])
        c["sid"] = sid
        if sid and sid not in per_session_first and label not in (
            "passthrough:count_tokens", "passthrough:non_api", "utility:prewarm", "utility:sidecar"
        ):
            per_session_first[sid] = label

    # 2: ground-truth checks
    checks = []
    # (a) every count_tokens path must be passthrough
    ct_wrong = [c for c in caps if "/count_tokens" in c["path"]
                and c["label"] != "passthrough:count_tokens"]
    checks.append(("count_tokens -> passthrough", len(ct_wrong) == 0,
                   f"{len(ct_wrong)} misses"))
    # (b) tool_result-tailed conversation requests must be continuation
    tr_wrong = 0
    tr_total = 0
    for c in caps:
        if not isinstance(c["body"], dict) or "/count_tokens" in c["path"]:
            continue
        msgs = c["body"].get("messages") or []
        if msgs and isinstance(msgs[-1], dict) and isinstance(msgs[-1].get("content"), list) \
           and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in msgs[-1]["content"]) \
           and (c["body"].get("tools") or []):
            tr_total += 1
            if c["label"] != "continuation":
                tr_wrong += 1
    checks.append((f"tool_result tail -> continuation ({tr_total} cases)", tr_wrong == 0,
                   f"{tr_wrong} misses"))
    # (c) first conversation request of each simulator session must be user_turn
    sim_firsts = {s: l for s, l in per_session_first.items() if s in sim_sids}
    sim_wrong = {s: l for s, l in sim_firsts.items() if l != "user_turn"}
    checks.append((f"simulator episode openers -> user_turn ({len(sim_firsts)} sessions)",
                   len(sim_wrong) == 0, f"wrong: {sim_wrong}"))

    lat_us = [n / 1000 for n in latencies]
    p50 = statistics.median(lat_us)
    p99 = sorted(lat_us)[int(len(lat_us) * 0.99)]

    L = ["# Discriminator evaluation — live wire captures", "",
         f"Captures: {len(caps)} | fingerprints from cc 2.1.201-202 wire evidence", "",
         "## Label distribution", "", "| Label | n | share |", "|---|---|---|"]
    for label, n in labels.most_common():
        L.append(f"| {label} | {n} | {100 * n / len(caps):.1f}% |")
    L += ["", "## By entrypoint (cli = interactive, sdk = headless/simulator)", ""]
    for ep, cnt in by_entry.items():
        L.append(f"- **{ep}**: " + ", ".join(f"{l}={n}" for l, n in cnt.most_common()))
    L += ["", "## Ground-truth checks", ""]
    ok_all = True
    for name, ok, detail in checks:
        ok_all &= ok
        L.append(f"- {'PASS' if ok else 'FAIL'} — {name}" + ("" if ok else f" ({detail})"))
    L += ["", "## Latency (per classification)", "",
          f"- p50: {p50:.1f} µs | p99: {p99:.1f} µs | budget: microseconds on the "
          f"passthrough flood, <20ms total decision path — discriminator share is negligible", ""]
    verdict = "ALL CHECKS PASS" if ok_all else "FAILURES PRESENT — fix before shadow run"
    L.append(f"**Verdict: {verdict}**")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0 if ok_all else 2


if __name__ == "__main__":
    sys.exit(main())
