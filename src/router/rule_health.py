"""WS3 - heuristic health: does each hand-written rule earn its keep?

The router's difficulty signal is a pile of hand-authored predicates: the verb
lexicon (``features.VERB_CLASSES``), the scope modifiers, the terse-continuation
guard, the big-context nudge, and the trajectory boosts (edit failures, recent
errors, escalate-on-retry). They accreted by intuition. This module audits each
one against recorded outcomes so none is trusted forever on faith - the direct
answer to "these heuristics won't scale".

HOW IT WORKS (read-only over the WS1 ledger):

  * Every decision stores its full ``features_json`` - and ``extract()`` runs
    BEFORE the hard gates, so a rule's predicate is recorded on every turn even
    when a gate (context feasibility, privacy) actually made the routing call.
    That means we can measure a rule's *predictive power* on all turns where it
    fired, not just the handful that reached the heuristic layer.
  * For each rule we compute ``hard_rate`` = P(turn went hard | rule fired) and
    compare it to the pool ``base_rate`` = P(turn went hard). The signed gap is
    the rule's **lift**.
      - an EASY-leaning rule (trivial, explain, narrow scope, terse approval)
        pulls its weight when its hard_rate sits clearly BELOW the base rate;
        clearly ABOVE and it is anti-signal -> DEMOTE-CANDIDATE.
      - a HARD-leaning rule (broad scope, big context, the trajectory boosts)
        earns its keep clearly ABOVE base; clearly BELOW and it is over-routing
        -> DEMOTE-CANDIDATE.
      - "clearly" means beyond WEAK_BAND (5%) of the base rate. Within the band
        (including a rule exactly AT base) is WEAK-SIGNAL: no clear signal.
      - leaning is judged by the policy's routing bands, so the middle-band verbs
        (small_edit, write, fix, unknown) are NEUTRAL and only reported, never
        flagged: the heuristic never routes them easy or hard in the first place.
  * Hardness uses ``outcomes.effective_task_hard``: v2's cause-clean task signal
    where present, with the legacy aggregate only for old rows. It is the SAME
    label WS4 and the shadow harness use. Only ``closed_*`` outcomes are counted.

Demotion is a HUMAN decision in v1: this reporter only flags candidates (a rule
firing rarely, or with the wrong-signed lift past a threshold on a large enough
sample). Auto-demotion is a WS5 job once these verdicts are trusted.

Provenance matters: synthetic (simulator) and organic outcomes are separable by
``--source`` so synthetic friction never silently contaminates a verdict about
real traffic. Default is every source, with the pool composition printed up top.

Fail-safe: an uninitialized/locked DB yields an honest degraded report, never a
crash (precedent: insights.py).

CLI:
    PYTHONPATH=src python -m router.rule_health --report [--db PATH] [--source organic]
    PYTHONPATH=src python -m router.rule_health --json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from router.features import CONTEXT_TOKEN_THRESHOLD
from router.outcomes import effective_task_hard, went_hard

DEFAULT_DB_PATH = Path("data/router-memory.db")

# A rule is worth flagging only when it has fired enough to trust the rate.
MIN_SAMPLE = 20
# A lift within this band of the base rate (including exactly 0) is treated as no
# clear signal -> WEAK-SIGNAL. Beyond the band, the sign vs the rule's leaning
# decides OK (favored direction) or DEMOTE-CANDIDATE (wrong direction).
WEAK_BAND = 0.05


@dataclass
class Rule:
    """A named predicate over the stored features dict, with a difficulty leaning.

    ``leaning`` is the direction the rule pushes the difficulty score:
    ``easy`` rules should select turns that go hard LESS than the base rate;
    ``hard`` rules should select turns that go hard MORE than the base rate;
    ``neutral`` rules make no strong claim (reported, never flagged).
    """

    name: str
    leaning: str  # "easy" | "hard" | "neutral"
    fired: Callable[[dict], bool]


def _verb_is(verb_class: str) -> Callable[[dict], bool]:
    return lambda features: features.get("verb_class") == verb_class


def _flag(field: str) -> Callable[[dict], bool]:
    return lambda features: bool(features.get(field))


# The rule catalogue mirrors features.py / policy.py. Every predicate the live
# path can act on appears here so WS3 audits the whole surface (the process fix
# for accretion: a new heuristic lands with its row here or it is invisible).
#
# `leaning` is judged against the policy's OWN routing bands (T_EASY=0.35 strict,
# T_HARD=0.70): a verb is 'easy' only if its base score routes it local, 'hard'
# only if it routes frontier, else 'neutral'. So small_edit (0.35, NOT < T_EASY)
# and fix (0.55) are NEUTRAL: the heuristic sends them to the middle band, so
# judging them by an easy/hard leaning the policy never implements would produce
# a spurious demote verdict.
RULES: list[Rule] = [
    # verb lexicon (features.VERB_CLASSES), leaning by the band the score routes to
    Rule("verb:trivial", "easy", _verb_is("trivial")),        # 0.10 -> local
    Rule("verb:explain", "easy", _verb_is("explain")),        # 0.20 -> local
    Rule("verb:small_edit", "neutral", _verb_is("small_edit")),  # 0.35 -> middle
    Rule("verb:write", "neutral", _verb_is("write")),         # 0.45 -> middle
    Rule("verb:fix", "neutral", _verb_is("fix")),             # 0.55 -> middle
    Rule("verb:hard", "hard", _verb_is("hard")),              # 0.85 -> frontier
    Rule("verb:unknown", "neutral", _verb_is("unknown")),     # 0.50 -> middle
    # scope modifiers (push the score up/down)
    Rule("scope:broad", "hard", _flag("broad_scope")),        # +0.20
    Rule("scope:narrow", "easy", _flag("narrow_scope")),      # -0.10
    # the terse-approval guard is STICKY, not easy: policy.py keeps the session's
    # CURRENT route (which may be frontier), so it is neutral - judging it easy
    # would spuriously DEMOTE it when "go ahead" follows a hard, frontier-stuck
    # task (review finding).
    Rule("terse_continuation", "neutral", _flag("is_terse_continuation")),
    # context nudge: import the threshold so this can never drift from features.py
    Rule("context:big", "hard",
         lambda features: (features.get("context_tokens") or 0) > CONTEXT_TOKEN_THRESHOLD),
    # trajectory boosts (heuristic_score); each forces the score up
    Rule("traj:edit_failures", "hard",
         lambda features: (features.get("recent_edit_failures") or 0) >= 1),
    Rule("traj:recent_errors", "hard",
         lambda features: (features.get("recent_errors") or 0) >= 3),
    Rule("retry_signal", "hard", _flag("prev_turn_interrupted")),
]


@dataclass
class RuleVerdict:
    """One rule's measured health over the analysed pool."""

    rule: str
    leaning: str
    fired: int
    fire_rate: float
    hard_rate: float           # P(went hard | fired)
    lift: float                # hard_rate - base_rate (signed)
    verdict: str               # OK | WEAK-SIGNAL | DEMOTE-CANDIDATE | INSUFFICIENT
    note: str


def _load_pool(db_path: Path, source: Optional[str]) -> list[tuple[dict, bool]]:
    """Return [(features_dict, went_hard)] over closed outcomes, honouring the
    optional source filter. Degrades to [] on any DB problem (never raises)."""
    try:
        conn = sqlite3.connect(str(db_path))
    except Exception:
        return []
    try:
        outcome_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(outcomes)")
        }
        task_signal_expr = (
            "o.task_signal_hard" if "task_signal_hard" in outcome_columns
            else "NULL"
        )
        verified_expr = (
            "o.verified_success" if "verified_success" in outcome_columns
            else "NULL"
        )
        verification_cause_expr = (
            "o.verification_failure_cause"
            if "verification_failure_cause" in outcome_columns else "NULL"
        )
        query = (
            "SELECT d.features_json, d.rung, o.escalated, o.user_retried, "
            f"o.outcome_proxy_hard, {task_signal_expr}, {verified_expr}, "
            f"{verification_cause_expr} "
            "FROM decisions d JOIN outcomes o ON o.route_id = d.route_id "
            "WHERE o.status IN ('closed_turn', 'closed_final')"
        )
        params: tuple = ()
        if source:
            query += " AND d.source = ?"
            params = (source,)
        rows = conn.execute(query, params).fetchall()
    except sqlite3.OperationalError:
        return []  # tables not created yet - uninitialized DB
    except Exception:
        return []
    finally:
        conn.close()

    pool: list[tuple[dict, bool]] = []
    for (features_json, rung, escalated, user_retried, proxy_hard, task_hard,
         verified_success, verification_failure_cause) in rows:
        try:
            features = json.loads(features_json) if features_json else {}
            if not isinstance(features, dict):
                features = {}
        except Exception:
            features = {}
        pool.append((features, effective_task_hard(
            task_hard, escalated, user_retried, proxy_hard,
            verified_success, rung, verification_failure_cause)))
    return pool


def _classify(rule: Rule, fired: int, hard_rate: float, base_rate: float,
              min_sample: int) -> tuple[str, str]:
    """Turn a rule's measured lift into a verdict + one-line note.

    Bands (symmetric, single WEAK_BAND constant):
      - fired < min_sample                     -> INSUFFICIENT (no verdict yet)
      - neutral leaning                        -> OK (reported, never flagged)
      - |lift| <= WEAK_BAND (incl. exactly 0)  -> WEAK-SIGNAL (no clear signal)
      - beyond the band, favored direction     -> OK
      - beyond the band, wrong direction       -> DEMOTE-CANDIDATE
    'Favored direction' is negative lift for an easy rule (selects easier turns),
    positive lift for a hard rule (selects harder turns)."""
    lift = hard_rate - base_rate
    if fired < min_sample:
        return "INSUFFICIENT", f"only {fired} fires (< {min_sample}) - no verdict yet"
    if rule.leaning == "neutral":
        return "OK", f"neutral rule, hard-rate {hard_rate:.0%} vs base {base_rate:.0%}"
    if abs(lift) <= WEAK_BAND:
        return "WEAK-SIGNAL", (
            f"within {WEAK_BAND:.0%} of base (lift {lift:+.0%}) - no clear signal")
    favored = lift < 0 if rule.leaning == "easy" else lift > 0
    if favored:
        direction = "easier" if rule.leaning == "easy" else "harder"
        return "OK", f"selects {direction} turns (lift {lift:+.0%})"
    if rule.leaning == "easy":
        return "DEMOTE-CANDIDATE", (
            f"easy-leaning but hard-rate {hard_rate:.0%} exceeds base {base_rate:.0%} "
            f"(lift {lift:+.0%}) - anti-signal")
    return "DEMOTE-CANDIDATE", (
        f"hard-leaning but hard-rate {hard_rate:.0%} below base {base_rate:.0%} "
        f"(lift {lift:+.0%}) - over-routing")


def analyse(db_path: Path = DEFAULT_DB_PATH, *, source: Optional[str] = None,
            min_sample: int = MIN_SAMPLE) -> dict:
    """Pure analysis: pool composition + a RuleVerdict per rule. No I/O beyond
    the read. Returns {} shape with 'pool' and 'rules' keys."""
    pool = _load_pool(db_path, source)
    total = len(pool)
    hard_total = sum(1 for _, hard in pool if hard)
    base_rate = (hard_total / total) if total else 0.0

    verdicts: list[RuleVerdict] = []
    for rule in RULES:
        fired_rows = [hard for features, hard in pool if _safe_fired(rule, features)]
        fired = len(fired_rows)
        hard_fired = sum(1 for hard in fired_rows if hard)
        hard_rate = (hard_fired / fired) if fired else 0.0
        verdict, note = _classify(rule, fired, hard_rate, base_rate, min_sample)
        verdicts.append(RuleVerdict(
            rule=rule.name, leaning=rule.leaning, fired=fired,
            fire_rate=(fired / total) if total else 0.0,
            hard_rate=hard_rate, lift=hard_rate - base_rate,
            verdict=verdict, note=note))
    return {
        "source": source or "all",
        "total": total,
        "hard_total": hard_total,
        "base_rate": base_rate,
        "rules": verdicts,
    }


def _safe_fired(rule: Rule, features: dict) -> bool:
    try:
        return bool(rule.fired(features))
    except Exception:
        return False


# ── reporter ───────────────────────────────────────────────────────────────


_ORDER = {"DEMOTE-CANDIDATE": 0, "WEAK-SIGNAL": 1, "OK": 2, "INSUFFICIENT": 3}


def build(db_path: Path = DEFAULT_DB_PATH, *, source: Optional[str] = None,
          min_sample: int = MIN_SAMPLE) -> str:
    """Markdown rule-health report (pure)."""
    result = analyse(db_path, source=source, min_sample=min_sample)
    lines = [
        "# rule health - do the hand-heuristics earn their keep?",
        "",
        f"- db: `{db_path}`",
        f"- source pool: **{result['source']}**",
        f"- closed turns analysed: {result['total']}",
    ]
    if not result["total"]:
        lines.append("- verdict: no closed outcomes yet (degraded/empty) - "
                     "run the flywheel or backfill first")
        return "\n".join(lines) + "\n"
    lines.append(
        f"- base hard-rate: {result['base_rate']:.1%} "
        f"({result['hard_total']}/{result['total']} turns went hard)")
    lines.append("")
    lines.append("Lift = P(hard | rule fired) - base rate. Easy-leaning rules want "
                 "negative lift; hard-leaning want positive.")
    lines.append("")
    lines.append("| rule | leaning | fired | fire% | hard-rate | lift | verdict |")
    lines.append("|------|---------|------:|------:|----------:|-----:|---------|")
    ranked = sorted(result["rules"], key=lambda v: (_ORDER.get(v.verdict, 9), -v.fired))
    for v in ranked:
        lines.append(
            f"| {v.rule} | {v.leaning} | {v.fired} | {v.fire_rate:.1%} | "
            f"{v.hard_rate:.0%} | {v.lift:+.0%} | {v.verdict} |")
    lines.append("")

    flagged = [v for v in ranked if v.verdict == "DEMOTE-CANDIDATE"]
    if flagged:
        lines.append("## demote candidates (human review - no auto-demote in v1)")
        for v in flagged:
            lines.append(f"- **{v.rule}** - {v.note}")
    else:
        lines.append("## demote candidates")
        lines.append("- none: every rule with a sufficient sample carries its "
                     "leaning's sign.")
    insufficient = [v.rule for v in ranked if v.verdict == "INSUFFICIENT"]
    if insufficient:
        lines.append("")
        lines.append(f"_Insufficient sample (< {min_sample} fires), no verdict yet: "
                     f"{', '.join(insufficient)}._")
    return "\n".join(lines) + "\n"


def _json_payload(db_path: Path, source: Optional[str], min_sample: int) -> dict:
    result = analyse(db_path, source=source, min_sample=min_sample)
    result["rules"] = [asdict(v) for v in result["rules"]]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="ledger path")
    parser.add_argument("--source", default=None,
                        choices=["organic", "simulator"],
                        help="restrict the pool to one provenance (default: all)")
    parser.add_argument("--min-sample", type=int, default=MIN_SAMPLE,
                        help="minimum fires before a rule gets a verdict")
    parser.add_argument("--report", action="store_true",
                        help="print the markdown report (the default action)")
    parser.add_argument("--json", action="store_true",
                        help="emit the analysis as JSON instead of markdown")
    args = parser.parse_args()
    if args.json:
        print(json.dumps(_json_payload(args.db, args.source, args.min_sample),
                         indent=2))
    else:
        print(build(args.db, source=args.source, min_sample=args.min_sample), end="")


if __name__ == "__main__":
    main()
