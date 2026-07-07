"""B9-lite — replay the policy engine against the mined historical corpus.

For every main-session user_turn in turns.parquet: build features (incl.
per-session trajectory state walked in ts order), run route_turn, and check
the decisions against everything we have ground truth for:

  1. every turn of a secret-flagged session (A8 scan) routes local & pinned
  2. huge-context turns never route local (feasibility gate)
  3. hard-intent turns (S10 matcher lexicon) go direct to frontier
  4. turns following an interrupt escalate (retry rule)
  5. decision latency p50 < 20ms (design §11; expect microseconds)

Plus the numbers the shadow run reports: predicted local share, band mix,
and what fraction the Phase-3 learned router would actually own.

Usage: PYTHONPATH=src .venv/bin/python -m router.eval_policy
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from .features import extract
from .policy import REGISTRY, CONTEXT_HEADROOM, SessionState, route_turn


def ctx_estimate(r: dict) -> int:
    n = max(r["n_assistant_msgs"], 1)
    return int((r["cache_read_tokens"] + r["input_tokens"]) / n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--turns", type=Path, default=Path("data/turns.parquet"))
    ap.add_argument("--secrets", type=Path, default=Path("data/secrets-scan.json"))
    ap.add_argument("--report", type=Path, default=Path("reports/policy-replay.md"))
    args = ap.parse_args()

    import pyarrow.parquet as pq

    rows = pq.read_table(args.turns).to_pylist()
    pinned_sessions: set[str] = set()
    if args.secrets.exists():
        for sid, _ in json.loads(args.secrets.read_text()).get("would_have_pinned", []):
            pinned_sessions.add(sid)

    by_session: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["source_kind"] == "main":
            by_session[r["session_id"]].append(r)
    for turns in by_session.values():
        turns.sort(key=lambda r: r["ts"] or "")

    decisions = []
    latencies = []
    for sid, turns in by_session.items():
        state = SessionState(session_id=sid, privacy_pinned=sid in pinned_sessions)
        recent_errors = recent_edit_failures = 0
        prev_interrupted = False
        for idx, r in enumerate(turns):
            if r["label"] != "user_turn":
                # non-decision turns still update trajectory
                recent_errors = r["n_error_results"]
                recent_edit_failures = r["n_edit_failures"]
                prev_interrupted = bool(r["interrupted"])
                continue
            f = extract(
                r["instruction_text"],
                context_tokens=ctx_estimate(r),
                turn_index=idx,
                recent_errors=recent_errors,
                recent_edit_failures=recent_edit_failures,
                prev_turn_interrupted=prev_interrupted,
            )
            t0 = time.perf_counter_ns()
            route = route_turn(f, state)
            latencies.append(time.perf_counter_ns() - t0)
            decisions.append({"row": r, "features": f, "route": route, "sid": sid})
            # trajectory update from what actually happened in this turn
            recent_errors = r["n_error_results"]
            recent_edit_failures = r["n_edit_failures"]
            prev_interrupted = bool(r["interrupted"])
            state.turn_count += 1

    n = len(decisions)
    if not n:
        print("no user_turns to replay", file=sys.stderr)
        return 1

    # ── checks ──
    checks = []
    pin_wrong = [d for d in decisions if d["sid"] in pinned_sessions
                 and not d["route"].pinned]
    n_pin = sum(1 for d in decisions if d["sid"] in pinned_sessions)
    checks.append((f"secret-flagged sessions always pinned ({n_pin} turns)",
                   not pin_wrong, f"{len(pin_wrong)} leaks"))
    feas_wrong = [d for d in decisions
                  if d["features"].context_tokens > CONTEXT_HEADROOM * REGISTRY["local"]["max_context"]
                  and d["route"].rung == "local" and not d["route"].pinned]
    n_big = sum(1 for d in decisions
                if d["features"].context_tokens > CONTEXT_HEADROOM * REGISTRY["local"]["max_context"])
    checks.append((f"oversized-context turns never local ({n_big} turns)",
                   not feas_wrong, f"{len(feas_wrong)} misses"))
    hard_wrong = [d for d in decisions
                  if d["features"].verb_class == "hard" and not d["features"].privacy_pinned
                  and d["sid"] not in pinned_sessions
                  and d["features"].context_tokens <= CONTEXT_HEADROOM * REGISTRY["local"]["max_context"]
                  and d["route"].rung != "frontier"]
    n_hard = sum(1 for d in decisions if d["features"].verb_class == "hard")
    checks.append((f"hard-verb turns -> frontier ({n_hard} turns)",
                   not hard_wrong, f"{len(hard_wrong)} misses"))
    retry_wrong = [d for d in decisions if d["features"].prev_turn_interrupted
                   and d["sid"] not in pinned_sessions
                   and d["features"].context_tokens <= CONTEXT_HEADROOM * REGISTRY["local"]["max_context"]
                   and d["route"].rung == "local"]
    n_retry = sum(1 for d in decisions if d["features"].prev_turn_interrupted)
    checks.append((f"post-interrupt turns escalate ({n_retry} turns)",
                   not retry_wrong, f"{len(retry_wrong)} misses"))

    lat_us = sorted(x / 1000 for x in latencies)
    p50, p99 = statistics.median(lat_us), lat_us[int(len(lat_us) * 0.99)]
    checks.append(("decision latency p50 < 20ms", p50 < 20_000, f"p50={p50:.1f}us"))

    # ── shares ──
    rungs = Counter(d["route"].rung for d in decisions)
    layers = Counter(d["route"].layer for d in decisions)
    conflicts = sum(1 for d in decisions if d["route"].conflict)

    L = ["# Policy replay — historical corpus (B9-lite)", "",
         f"user_turns replayed: {n} (main sessions, trajectory state walked in order)", "",
         "## Predicted routing", "", "| Rung | n | share |", "|---|---|---|"]
    for rung, c in rungs.most_common():
        L.append(f"| {rung} | {c} | {100 * c / n:.1f}% |")
    L += ["", "## Decided by", "", "| Layer | n | share |", "|---|---|---|"]
    for layer, c in layers.most_common():
        L.append(f"| {layer} | {c} | {100 * c / n:.1f}% |")
    L += ["",
          f"Phase-3 learned router would own only the `middle_default` band — "
          f"{layers.get('middle_default', 0)} turns ({100 * layers.get('middle_default', 0) / n:.1f}%). "
          f"Pin-context conflicts surfaced (§5.8): {conflicts}.",
          "", "## Ground-truth checks", ""]
    ok_all = True
    for name, ok, detail in checks:
        ok_all &= ok
        L.append(f"- {'PASS' if ok else 'FAIL'} — {name}" + ("" if ok else f" ({detail})"))
    L += ["", f"Latency: p50 {p50:.1f}us, p99 {p99:.1f}us (budget: <20ms; <30ms decision path §11)", "",
          f"**Verdict: {'ALL CHECKS PASS' if ok_all else 'FAILURES PRESENT'}**"]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0 if ok_all else 2


if __name__ == "__main__":
    sys.exit(main())
