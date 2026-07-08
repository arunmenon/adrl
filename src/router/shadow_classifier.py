"""UNIT B — LLM-classifier shadow harness (offline, never touches live routing).

Question: over the REAL corpus, how much does the local LLM classifier resolve
the regex's ~90% "middle" abstention — and are its frontier calls actually
better than the coin-flip the `middle_default` band is today?

We replay the regex intent path over every main-session user_turn, isolate the
turns the regex is UNCERTAIN about (verb_class=='unknown' OR intent-only score
in the [T_EASY, T_HARD] band), and for a stratified sample of those we ask the
local classifier (`router.llm_classifier.classify_intent_llm`, a sibling module)
to adjudicate. We record:

  * reclassification breakdown — uncertain -> trivial/standard/hard, i.e. how
    many the classifier pulls to a confident LOCAL call vs a FRONTIER call;
  * the None/fallback rate (classifier unavailable / unparseable);
  * outcome-proxy correlation — does the classifier's needs_frontier=True set
    have a higher REAL hard-rate (n_edit_failures>=1 OR n_error_results>=1 OR
    interrupted OR n_continuations>=10) than its needs_frontier=False set?
    This is the "is it actually better than the middle coin-flip" test.

Writes reports/classifier-shadow.md (metrics only, no raw instructions).

This module NEVER imports or calls policy.route_turn / hook — it is a
measurement harness only. Live-path wiring is done separately, by hand.

Usage:
  PYTHONPATH=src .venv/bin/python -m router.shadow_classifier --limit 150
  PYTHONPATH=src .venv/bin/python -m router.shadow_classifier --all
  PYTHONPATH=src .venv/bin/python -m router.shadow_classifier --limit 3   # smoke
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .features import SCOPE_BROAD, SCOPE_NARROW, classify_intent
from .policy import T_EASY, T_HARD

# Sibling module (written concurrently). Contract:
#   classify_intent_llm(text) -> Optional[LlmVerdict]
#   LlmVerdict has .tier ("trivial"|"standard"|"hard"), .needs_frontier (bool),
#   .score (float). Returns None on any failure (model down, bad JSON, timeout).
# Imported lazily so this harness loads (and its unit test runs with a stub)
# even before the sibling module has landed. `classify_intent_llm` here is a
# module-level name tests may monkeypatch.
try:  # pragma: no cover - trivial import shim
    from .llm_classifier import classify_intent_llm
except Exception:  # sibling not written yet / import-time failure
    classify_intent_llm = None

CONTEXT_TOKEN_THRESHOLD = 20_000
STRATIFY_SEED = 1729


def ctx_estimate(row: dict) -> int:
    """Same per-turn context estimate the policy replay uses (eval_policy.py)."""
    n_assistant = max(row["n_assistant_msgs"], 1)
    return int((row["cache_read_tokens"] + row["input_tokens"]) / n_assistant)


def intent_only_score(instruction_text: str, context_tokens: int) -> float:
    """The intent-only difficulty score used to define regex certainty.

    Replicates the *non-trajectory* part of features.heuristic_score:
    verb base score + scope adjustments + a big-context nudge. Trajectory
    signals (edit failures, recent errors, prior interrupt) are deliberately
    excluded — "regex-uncertain" is about whether the INTENT alone confidently
    places the turn, which is exactly the gap the classifier is meant to fill.
    """
    text = instruction_text or ""
    _, base = classify_intent(text)
    score = base
    if SCOPE_BROAD.search(text):
        score += 0.20
    if SCOPE_NARROW.search(text):
        score -= 0.10
    if context_tokens > CONTEXT_TOKEN_THRESHOLD:
        score += 0.10
    return max(0.0, min(1.0, score))


def is_regex_uncertain(verb_class: str, intent_score: float) -> bool:
    """The regex abstains: no recognized verb, or the score lands in the band
    the heuristic layer can't confidently cut (T_EASY <= s <= T_HARD)."""
    return verb_class == "unknown" or (T_EASY <= intent_score <= T_HARD)


def outcome_proxy_hard(row: dict) -> bool:
    """Weak execution-friction proxy for 'this turn really was hard'.

    Same definition the bake-off used: any edit-apply failure, any errored tool
    result, a user interrupt, or a runaway (>=10 continuations). Non-validating
    (see classifier-bakeoff.md caveats) but the only ground-truth-ish signal we
    have for whether frontier help was warranted.
    """
    return (
        (row.get("n_edit_failures") or 0) >= 1
        or (row.get("n_error_results") or 0) >= 1
        or bool(row.get("interrupted"))
        or (row.get("n_continuations") or 0) >= 10
    )


def load_main_user_turns(turns_path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(turns_path).to_pylist()
    return [
        r for r in rows
        if r["source_kind"] == "main"
        and r["label"] == "user_turn"
        and (r.get("instruction_text") or "").strip()
    ]


def stratified_sample(uncertain: list[dict], limit: int, seed: int) -> list[dict]:
    """Proportional stratified sample by verb_class, so the sample keeps the
    same intent mix as the full uncertain population. Deterministic under seed."""
    if limit >= len(uncertain):
        return list(uncertain)
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for u in uncertain:
        by_class[u["_verb_class"]].append(u)

    total = len(uncertain)
    sample: list[dict] = []
    for verb_class, group in by_class.items():
        take = max(1, round(limit * len(group) / total))
        take = min(take, len(group))
        sample.extend(rng.sample(group, take))
    # Trim/pad to exactly `limit` (rounding can over/undershoot).
    rng.shuffle(sample)
    if len(sample) > limit:
        sample = sample[:limit]
    elif len(sample) < limit:
        chosen = {id(s) for s in sample}
        remainder = [u for u in uncertain if id(u) not in chosen]
        rng.shuffle(remainder)
        sample.extend(remainder[: limit - len(sample)])
    return sample


def run(turns_path: Path, report_path: Path, limit: int | None, use_all: bool,
        model: str | None, seed: int = STRATIFY_SEED) -> int:
    turns = load_main_user_turns(turns_path)
    if not turns:
        print("no main-session user_turns with text found", file=sys.stderr)
        return 1

    # ── regex pass over the full corpus ──
    uncertain: list[dict] = []
    verb_counts: Counter = Counter()
    for r in turns:
        text = r["instruction_text"]
        ctx = ctx_estimate(r)
        verb_class, _ = classify_intent(text)
        score = intent_only_score(text, ctx)
        verb_counts[verb_class] += 1
        if is_regex_uncertain(verb_class, score):
            uncertain.append({"_row": r, "_verb_class": verb_class, "_score": score})

    n_total = len(turns)
    n_uncertain = len(uncertain)
    uncertain_share = 100 * n_uncertain / n_total if n_total else 0.0

    # ── choose which uncertain turns to adjudicate ──
    if use_all:
        sample = list(uncertain)
    else:
        sample = stratified_sample(uncertain, limit or 150, seed)
    n_sample = len(sample)

    # ── ask the classifier ──
    reclass: Counter = Counter()          # tier -> n (confident resolutions)
    n_none = 0                            # classifier unavailable / unparseable
    frontier_true: list[bool] = []        # outcome proxy for needs_frontier=True
    frontier_false: list[bool] = []       # outcome proxy for needs_frontier=False
    verb_of_frontier: Counter = Counter()  # original regex verb_class of frontier picks

    for item in sample:
        row = item["_row"]
        verdict = _classify(item["_row"]["instruction_text"], model)
        if verdict is None:
            n_none += 1
            continue
        tier = getattr(verdict, "tier", None) or "unknown"
        reclass[tier] += 1
        proxy = outcome_proxy_hard(row)
        if getattr(verdict, "needs_frontier", False):
            frontier_true.append(proxy)
            verb_of_frontier[item["_verb_class"]] += 1
        else:
            frontier_false.append(proxy)

    n_resolved = n_sample - n_none
    resolve_rate = 100 * n_resolved / n_sample if n_sample else 0.0
    n_frontier = len(frontier_true)
    n_local = len(frontier_false)

    def hard_rate(bucket: list[bool]) -> float:
        return 100 * sum(bucket) / len(bucket) if bucket else 0.0

    hr_frontier = hard_rate(frontier_true)
    hr_local = hard_rate(frontier_false)
    # baseline: real hard-rate over ALL sampled uncertain turns (the coin-flip
    # the middle_default band pays for today).
    all_proxy = [outcome_proxy_hard(it["_row"]) for it in sample]
    hr_baseline = hard_rate(all_proxy)

    resolves_enough = resolve_rate >= 50.0
    tracks_higher = (n_frontier > 0 and n_local > 0 and hr_frontier > hr_local)
    verdict_pass = resolves_enough and tracks_higher

    # ── report (metrics only, scrubbed) ──
    L: list[str] = []
    L.append("# Classifier shadow — resolving the regex middle (UNIT B)")
    L.append("")
    L.append("Offline harness. Does NOT touch live routing (no policy/hook import).")
    L.append(f"Classifier model: `{model or 'llm_classifier default'}`")
    L.append("")
    L.append("## Corpus & regex abstention")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| main-session user_turns (with text) | {n_total} |")
    L.append(f"| regex-uncertain (unknown verb OR score in "
             f"[{T_EASY}, {T_HARD}]) | {n_uncertain} ({uncertain_share:.1f}%) |")
    L.append(f"| adjudicated this run ({'ALL' if use_all else f'sample, seed={seed}'}) | {n_sample} |")
    L.append("")
    L.append("Regex verb-class mix over the full corpus:")
    L.append("")
    L.append("| verb_class | n | share |")
    L.append("|---|---|---|")
    for vc, c in verb_counts.most_common():
        L.append(f"| {vc} | {c} | {100 * c / n_total:.1f}% |")
    L.append("")

    L.append("## Reclassification of regex-uncertain turns")
    L.append("")
    L.append(f"Fallback / unavailable (classifier returned None): "
             f"{n_none}/{n_sample} ({100 * n_none / n_sample if n_sample else 0:.1f}%)")
    L.append(f"Confidently resolved: {n_resolved}/{n_sample} ({resolve_rate:.1f}%)")
    L.append("")
    L.append("| classifier tier | n | share of resolved | routes to |")
    L.append("|---|---|---|---|")
    for tier in ("trivial", "standard", "hard"):
        c = reclass.get(tier, 0)
        share = 100 * c / n_resolved if n_resolved else 0.0
        dest = "frontier" if tier == "hard" else "local"
        L.append(f"| {tier} | {c} | {share:.1f}% | {dest} |")
    # any unexpected tiers
    for tier, c in reclass.items():
        if tier not in ("trivial", "standard", "hard"):
            L.append(f"| {tier} (unexpected) | {c} | "
                     f"{100 * c / n_resolved if n_resolved else 0:.1f}% | ? |")
    L.append("")
    L.append(f"Pulled to a confident LOCAL call: {n_local} "
             f"({100 * n_local / n_resolved if n_resolved else 0:.1f}% of resolved). "
             f"Escalated to FRONTIER: {n_frontier} "
             f"({100 * n_frontier / n_resolved if n_resolved else 0:.1f}% of resolved).")
    L.append("")

    L.append("## Outcome-proxy correlation (is it better than the middle coin-flip?)")
    L.append("")
    L.append("Real hard-rate = share of turns with n_edit_failures>=1 OR "
             "n_error_results>=1 OR interrupted OR n_continuations>=10.")
    L.append("")
    L.append("| Bucket | n | real hard-rate |")
    L.append("|---|---|---|")
    L.append(f"| classifier needs_frontier=True | {n_frontier} | {hr_frontier:.1f}% |")
    L.append(f"| classifier needs_frontier=False | {n_local} | {hr_local:.1f}% |")
    L.append(f"| all sampled uncertain (today's middle) | {len(all_proxy)} | {hr_baseline:.1f}% |")
    L.append("")
    if n_frontier and n_local:
        lift = hr_frontier - hr_local
        L.append(f"Separation (frontier hard-rate - local hard-rate): "
                 f"{lift:+.1f} pts. A positive lift means the classifier's frontier "
                 f"calls concentrate the genuinely-hard turns better than the "
                 f"undifferentiated middle does.")
    else:
        L.append("Insufficient split to measure separation (one bucket empty).")
    L.append("")

    L.append("## Verdict")
    L.append("")
    L.append(f"- Resolves >=50% of regex-uncertain: "
             f"{'PASS' if resolves_enough else 'FAIL'} ({resolve_rate:.1f}% resolved)")
    L.append(f"- Frontier calls track higher real hard-rate: "
             f"{'PASS' if tracks_higher else 'FAIL'} "
             f"({hr_frontier:.1f}% vs {hr_local:.1f}%)")
    L.append("")
    L.append(f"**Verdict: {'PASS' if verdict_pass else 'FAIL'}** — "
             f"{'the classifier confidently resolves the regex middle and its frontier calls are better than a coin-flip.' if verdict_pass else 'the classifier does not yet clear both bars on this sample.'}")
    L.append("")
    L.append("_Caveats: outcome_proxy is execution-friction, not intrinsic "
             "difficulty (non-validating; see classifier-bakeoff.md). Single-user, "
             "all-frontier source corpus. Sample is stratified by verb_class under "
             "a fixed seed; re-run with --all for the full population._")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


def _classify(text: str, model: str | None):
    """Call the sibling classifier, tolerating either a (text) or (text, model)
    signature so a --model override works if the sibling supports it."""
    fn = classify_intent_llm
    if fn is None:  # sibling module absent — lazy retry in case it landed since import
        from importlib import import_module
        fn = getattr(import_module("router.llm_classifier"), "classify_intent_llm")
    if model is not None:
        try:
            return fn(text, model=model)
        except TypeError:
            pass
    return fn(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--turns", type=Path, default=Path("data/turns.parquet"))
    ap.add_argument("--report", type=Path, default=Path("reports/classifier-shadow.md"))
    ap.add_argument("--limit", type=int, default=150,
                    help="stratified sample size of regex-uncertain turns (default 150)")
    ap.add_argument("--all", action="store_true",
                    help="adjudicate every regex-uncertain turn (ignores --limit)")
    ap.add_argument("--model", type=str, default=None,
                    help="override classifier model (passed to llm_classifier if supported)")
    ap.add_argument("--seed", type=int, default=STRATIFY_SEED)
    args = ap.parse_args()
    return run(args.turns, args.report, args.limit, args.all, args.model, args.seed)


if __name__ == "__main__":
    sys.exit(main())
