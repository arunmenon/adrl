"""A6 + A7 — corpus metrics report and best-single-model baseline.

Produces reports/corpus-metrics.md: traffic shares vs design doc §4, trip-wire
frequencies (§5.5), per-intent medians, token economics, and the A7 baseline —
what this traffic costs if everything ran on claude-opus-4-8 (decision D1),
which is the number the Phase-3 learned router must beat.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Anthropic price table, $/MTok, pinned 2026-07-06 (claude-api skill, cached 2026-06-24).
# Cache reads ~0.1x input; cache writes ~1.25x input (5m TTL).
PRICES = {
    "claude-opus-4-8": {"in": 5.00, "out": 25.00},
    "claude-opus-4-7": {"in": 5.00, "out": 25.00},
    "claude-fable-5": {"in": 10.00, "out": 50.00},
    "claude-haiku-4-5": {"in": 1.00, "out": 5.00},
    "claude-sonnet-5": {"in": 3.00, "out": 15.00},
}
CACHE_READ_MULT = 0.1
CACHE_WRITE_MULT = 1.25
BASELINE_MODEL = "claude-opus-4-8"  # decision D1

INTENT_BUCKETS = [
    ("explain", re.compile(r"^(what|why|how|explain|describe|tell me|can you (tell|explain))", re.I)),
    ("fix", re.compile(r"^(fix|debug|resolve|why (is|does).*(fail|crash|error))", re.I)),
    ("write", re.compile(r"^(write|create|add|implement|build|generate|make)", re.I)),
    ("edit", re.compile(r"^(rename|update|change|modify|remove|delete|move)", re.I)),
    ("run", re.compile(r"^(run|execute|test|try|check|verify)", re.I)),
]


def intent_of(text: str) -> str:
    t = text.strip()
    for name, pat in INTENT_BUCKETS:
        if pat.search(t):
            return name
    return "other"


def price_model(model: str) -> dict:
    for known, p in PRICES.items():
        if model.startswith(known):
            return p
    return PRICES[BASELINE_MODEL]  # unknown -> priced as baseline, noted in report


def cost_usd(r: dict, prices: dict) -> float:
    return (
        r["input_tokens"] / 1e6 * prices["in"]
        + r["cache_read_tokens"] / 1e6 * prices["in"] * CACHE_READ_MULT
        + r["cache_creation_tokens"] / 1e6 * prices["in"] * CACHE_WRITE_MULT
        + r["output_tokens"] / 1e6 * prices["out"]
    )


def pct(a: float, b: float) -> str:
    return f"{100 * a / b:.1f}%" if b else "n/a"


def median(vals: list) -> float:
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else 0


def build(rows: list[dict]) -> str:
    main = [r for r in rows if r["source_kind"] == "main"]
    sub = [r for r in rows if r["source_kind"] == "subagent"]
    user_turns = [r for r in main if r["label"] == "user_turn"]

    L: list[str] = ["# Corpus Metrics — Phase 0 (A6/A7)", ""]
    L.append(f"Turns: {len(rows)} total ({len(main)} main-session, {len(sub)} subagent). "
             "All request counts are transcript-visible (lower bound — sidecar calls invisible; wire truth comes from workstream B).")
    L.append("")

    # ── 1. traffic shares vs design §4 ──
    L.append("## 1. Traffic shares (main sessions) vs design doc §4")
    L.append("")
    label_counts = Counter(r["label"] for r in main)
    conts = defaultdict(int)
    for r in main:
        conts[r["label"]] += r["n_continuations"]
    total_reqs = sum(label_counts.values()) + sum(conts.values())
    L.append("| Request kind | Design §4 | Measured |")
    L.append("|---|---|---|")
    L.append(f"| Tool continuations | 70-90% | {pct(sum(conts.values()), total_reqs)} |")
    L.append(f"| Utility/housekeeping (initiations) | 5-15% | "
             f"{pct(label_counts['utility'] + label_counts.get('utility:compaction', 0), total_reqs)} |")
    L.append(f"| New user turns (initiations) | 5-15% | {pct(label_counts['user_turn'], total_reqs)} |")
    L.append(f"| Subagent turns (separate files) | occasional | {len(sub)} turns, "
             f"{sum(r['n_continuations'] for r in sub)} continuations |")
    L.append("")

    # ── 2. trip-wire frequencies (§5.5) ──
    L.append("## 2. Trip-wire signal frequencies (design §5.5), per user_turn")
    L.append("")
    n_ut = len(user_turns) or 1
    edit1 = sum(1 for r in user_turns if r["n_edit_failures"] >= 1)
    edit2 = sum(1 for r in user_turns if r["n_edit_failures"] >= 2)
    L.append(f"- Edit-apply failure (>=1): {edit1} ({pct(edit1, n_ut)}); 2-strike trip-wire: {edit2} ({pct(edit2, n_ut)})")
    err = sum(1 for r in user_turns if r["had_errors"])
    L.append(f"- Any is_error tool_result: {err} ({pct(err, n_ut)})")
    intr = sum(1 for r in main if r["interrupted"])
    L.append(f"- User interrupts (main sessions): {intr}")
    rej = sum(1 for r in main if r["tool_rejected"])
    L.append(f"- Tool-use rejections: {rej}")
    par = sum(1 for r in rows if r["parallel_tool_use_msgs"] > 0)
    L.append(f"- Turns with parallel tool calls (S12, per-rung registry flag): {par} ({pct(par, len(rows))} of all turns)")
    L.append("")

    # ── 3. per-intent medians (sets §5.5 turn budgets) ──
    L.append("## 3. Per-intent medians, user_turns (turn-budget thresholds, §5.5)")
    L.append("")
    L.append("| Intent | n | median out-tokens | median duration (s) | median continuations |")
    L.append("|---|---|---|---|---|")
    by_intent = defaultdict(list)
    for r in user_turns:
        by_intent[intent_of(r["instruction_text"])].append(r)
    for name, rs in sorted(by_intent.items(), key=lambda kv: -len(kv[1])):
        dur = median([r["duration_ms"] for r in rs])
        L.append(f"| {name} | {len(rs)} | {median([r['output_tokens'] for r in rs]):.0f} "
                 f"| {dur / 1000:.0f} | {median([r['n_continuations'] for r in rs]):.0f} |")
    L.append("")
    L.append("Design §5.5 turn-budget rule '2x median tokens or 90s per intent class' can now be instantiated from this table.")
    L.append("")

    # ── 4. token economics ──
    L.append("## 4. Token economics (observed)")
    L.append("")
    by_model: dict[str, dict] = defaultdict(lambda: {"in": 0, "out": 0, "cr": 0, "cw": 0, "cost": 0.0, "turns": 0})
    for r in rows:
        model = next((m for m in r["models"].split(",") if m and m != "<synthetic>"), "unknown")
        b = by_model[model]
        b["in"] += r["input_tokens"]; b["out"] += r["output_tokens"]
        b["cr"] += r["cache_read_tokens"]; b["cw"] += r["cache_creation_tokens"]
        b["cost"] += cost_usd(r, price_model(model)); b["turns"] += 1
    L.append("| Model | turns | input | output | cache read | cache write | est. cost |")
    L.append("|---|---|---|---|---|---|---|")
    total_cost = 0.0
    for model, b in sorted(by_model.items(), key=lambda kv: -kv[1]["cost"]):
        total_cost += b["cost"]
        L.append(f"| {model} | {b['turns']} | {b['in']:,} | {b['out']:,} | {b['cr']:,} | {b['cw']:,} | ${b['cost']:.2f} |")
    all_in = sum(b["in"] for b in by_model.values())
    all_cr = sum(b["cr"] for b in by_model.values())
    L.append("")
    L.append(f"**Total estimated spend in corpus window: ${total_cost:.2f}.** "
             f"Cache-hit ratio: {pct(all_cr, all_cr + all_in)} of prompt tokens served from cache "
             "— this is the asset mid-turn model switching would destroy (design §2).")
    L.append("")

    # ── 5. A7 baseline ──
    L.append("## 5. Best-single-model baseline (A7, decision D1)")
    L.append("")
    baseline_cost = sum(cost_usd(r, PRICES[BASELINE_MODEL]) for r in rows)
    L.append(f"- Same token volumes re-priced entirely at **{BASELINE_MODEL}** "
             f"(${PRICES[BASELINE_MODEL]['in']}/MTok in, ${PRICES[BASELINE_MODEL]['out']}/MTok out, "
             f"cache read {CACHE_READ_MULT}x, write {CACHE_WRITE_MULT}x): **${baseline_cost:.2f}**")
    easy = [r for r in user_turns if r["n_continuations"] <= 3 and not r["had_errors"]
            and r["output_tokens"] < 2000]
    easy_cost = sum(cost_usd(r, PRICES[BASELINE_MODEL]) for r in easy)
    L.append(f"- Naive local-routable candidates (user_turns with <=3 continuations, no errors, <2k out-tokens): "
             f"{len(easy)}/{len(user_turns)} turns ({pct(len(easy), len(user_turns))}), "
             f"worth ${easy_cost:.2f} at baseline prices ({pct(easy_cost, baseline_cost)} of baseline spend)")
    L.append("")
    L.append("The Phase-3 learned router must beat this baseline on replayed traffic (design §10). "
             "The 'easy candidates' row is the ceiling on savings from the heuristic layer alone — "
             "a deliberately naive filter; the real policy engine should do better.")
    L.append("")
    L.append("*Caveats: single-user corpus, workflow-skewed; transcripts under-count requests; "
             "unknown models priced as baseline; prices pinned 2026-07-06.*")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--turns", type=Path, default=Path("data/turns.parquet"))
    ap.add_argument("--out", type=Path, default=Path("reports/corpus-metrics.md"))
    args = ap.parse_args()

    import pyarrow.parquet as pq

    rows = pq.read_table(args.turns).to_pylist()
    md = build(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
