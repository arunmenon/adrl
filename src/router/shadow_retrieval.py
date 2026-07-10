"""WS4 shadow harness - would the retrieval router beat today's middle?

Retrospective, read-only evaluation of the retrieval router over the ledger's
ambiguous-middle decisions (``layer in {middle_default, classifier}``), the only
turns the retrieval layer would ever touch. For each such turn it asks the
memory, using LEAVE-ONE-OUT, "how did turns like this one go?" and compares the
vote to what actually happened.

Crucially it drives the REAL memory pipeline rather than a hand-rolled copy:

  * the neighbor set comes from ``SqliteProvider.similar_turns`` (the same
    normalized-projection kNN the live resolver queries), so the blob decode,
    normalization, and top-K-by-cosine ordering are shared, not re-implemented;
  * leave-one-out drops the target and its own session from that ranked list,
    then takes the top-K, so a filtered/firewalled neighbor consumes a K slot
    exactly as it does live (live filters AFTER the top-K cut - the previous
    hand-rolled walk filtered DURING the walk and evaluated turns live abstains
    on);
  * the filter, the neighbor-count gate, and the vote all run through
    ``retrieval_router.evaluate_neighbors`` - the identical decision core the
    live resolver uses;
  * the global ``MIN_FINALIZED`` cold-start gate is enforced, so no metric is
    reported for a memory regime where the live resolver abstains on everything;
  * the ground-truth "went hard" label is ``outcomes.went_hard`` - the SAME
    three-signal definition the neighbor vote now uses (NeighborTurn carries
    user_retried), so the vote is never scored against a stricter label than it
    votes on.

GRADUATION GATE (plan): the retrieval layer is wired live only when, over a
sufficient number of ACTUALLY-EVALUATED (non-abstained) middle-band decisions,
it beats both the LLM classifier and the best-single-model baseline on a
cost-aware objective, with frontier recall never below the classifier's. The
gate keys on ``evaluated``, not the raw middle count, so a run where almost
everything abstains can never be declared sufficient.

Known approximation: leave-one-out queries with each turn's STORED document
embedding (the raw text is never stored, by design), skipping the live
``search_query:`` prefix swap. Rankings are close; absolute numbers on a tiny
sample are indicative, not final - another reason the gate needs volume.

CLI:
    PYTHONPATH=src python -m router.shadow_retrieval --report [--db PATH]
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from router.memory_sqlite import SqliteProvider
from router.outcomes import went_hard
from router.retrieval_router import (
    GRADUATION_MIN_MIDDLE,
    K,
    MIN_FINALIZED,
    evaluate_neighbors,
)

DEFAULT_DB_PATH = Path("data/router-memory.db")
MIDDLE_LAYERS = ("middle_default", "classifier")
_CLOSED = ("closed_turn", "closed_final")


@dataclass
class _Target:
    """One middle-band decision to evaluate leave-one-out."""

    route_id: str
    session_id: str
    source: str
    ts: float
    escalated: bool
    user_retried: Optional[bool]
    proxy_hard: Optional[bool]

    @property
    def actual_hard(self) -> bool:
        return went_hard(self.escalated, self.user_retried, self.proxy_hard)


def _load_targets(conn: sqlite3.Connection) -> list[_Target]:
    """Middle-band, closed decisions that carry an embedding. [] on any error."""
    try:
        rows = conn.execute(
            "SELECT d.route_id, d.session_id, d.source, d.ts, "
            "o.escalated, o.user_retried, o.outcome_proxy_hard "
            "FROM decisions d "
            "JOIN outcomes o ON o.route_id = d.route_id "
            "JOIN embeddings e ON e.route_id = d.route_id "
            "WHERE o.status IN (?, ?) AND d.layer IN (?, ?)",
            (*_CLOSED, *MIDDLE_LAYERS),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    except Exception:
        return []
    targets: list[_Target] = []
    for route_id, session_id, source, ts, escalated, user_retried, proxy_hard in rows:
        targets.append(_Target(
            route_id=route_id, session_id=session_id, source=source,
            ts=float(ts or 0.0),
            escalated=bool(escalated),
            user_retried=None if user_retried is None else bool(user_retried),
            proxy_hard=None if proxy_hard is None else bool(proxy_hard)))
    return targets


@dataclass
class ShadowResult:
    middle_total: int
    evaluated: int              # non-abstained
    abstained: int
    tp: int                     # predicted frontier & actual hard
    fp: int
    fn: int
    tn: int
    now: float
    finalized: int

    @property
    def coverage(self) -> float:
        return self.evaluated / self.middle_total if self.middle_total else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.evaluated if self.evaluated else 0.0

    @property
    def frontier_recall(self) -> float:
        hard = self.tp + self.fn
        return self.tp / hard if hard else 0.0

    @property
    def frontier_send_rate(self) -> float:
        return (self.tp + self.fp) / self.evaluated if self.evaluated else 0.0

    @property
    def actual_hard_rate(self) -> float:
        return (self.tp + self.fn) / self.evaluated if self.evaluated else 0.0


def evaluate(db_path: Path = DEFAULT_DB_PATH, *, now: Optional[float] = None) -> ShadowResult:
    """Run the leave-one-out shadow evaluation against the real provider.

    ``now`` is injectable for deterministic tests; it defaults to the newest
    ledger ts (not wall-clock), so the recency weighting is reproducible from
    the data alone."""
    provider = SqliteProvider(db_path=db_path)
    if not provider.health():
        return ShadowResult(0, 0, 0, 0, 0, 0, 0, now or 0.0, 0)
    conn = sqlite3.connect(str(db_path))
    try:
        targets = _load_targets(conn)
        if now is None:
            row = conn.execute("SELECT MAX(ts) FROM decisions").fetchone()
            now = float(row[0]) if row and row[0] is not None else 0.0
    finally:
        conn.close()

    stats = provider.stats()
    by_status = stats.get("outcomes_by_status", {}) if isinstance(stats, dict) else {}
    finalized = int((by_status.get("closed_final", 0) or 0)
                    + (by_status.get("closed_turn", 0) or 0))

    tp = fp = fn = tn = evaluated = 0
    for target in targets:
        embedding = provider.embedding_for(target.route_id)
        if not embedding:
            continue  # abstain (no vector)
        # TEMPORAL leave-one-out: a neighbor is only eligible if it existed in
        # memory when the target was routed live - i.e. its decision ts is
        # strictly before the target's. This drops the target itself AND every
        # FUTURE outcome (a target can no longer be scored using turns that came
        # after it), while correctly KEEPING same-session PAST turns (those were
        # genuinely in live memory - the trajectory signal the router uses).
        # Approximated by decision ts; the outcome may close slightly later, but
        # this is far more faithful than the old blanket same-session exclusion
        # (which dropped legitimate history) or using future outcomes.
        ranked = provider.similar_turns(embedding, None)   # full ranked pool
        past = [n for n in ranked
                if n.route_id != target.route_id and n.ts < target.ts]
        # Cold-start gate, temporally correct: if too few outcomes existed BEFORE
        # this turn, the live resolver would have cold-started, so shadow abstains.
        if len(past) < MIN_FINALIZED:
            continue
        # Recency is measured as of the TARGET's routing time (its ts), so a
        # neighbor's age matches what the live resolver would have seen.
        verdict, _ = evaluate_neighbors(past, now=target.ts, k=K,
                                        query_source=target.source)
        if verdict is None:
            continue  # abstain
        evaluated += 1
        if verdict.needs_frontier and target.actual_hard:
            tp += 1
        elif verdict.needs_frontier and not target.actual_hard:
            fp += 1
        elif not verdict.needs_frontier and target.actual_hard:
            fn += 1
        else:
            tn += 1

    return ShadowResult(
        middle_total=len(targets), evaluated=evaluated,
        abstained=len(targets) - evaluated, tp=tp, fp=fp, fn=fn, tn=tn,
        now=now, finalized=finalized)


# ── reporter ───────────────────────────────────────────────────────────────


def build(db_path: Path = DEFAULT_DB_PATH, *, now: Optional[float] = None) -> str:
    """Markdown graduation report (pure)."""
    result = evaluate(db_path, now=now)
    lines = [
        "# retrieval router - shadow evaluation (WS4)",
        "",
        f"- db: `{db_path}`",
        f"- middle-band turns with embeddings: {result.middle_total}",
        f"- finalized outcomes in memory: {result.finalized} "
        f"(cold-start gate: {MIN_FINALIZED})",
        f"- graduation needs: >= {GRADUATION_MIN_MIDDLE} EVALUATED "
        f"(non-abstained) middle-band decisions",
    ]
    if result.middle_total == 0:
        lines.append("- verdict: **INSUFFICIENT DATA** - no middle-band turns "
                     "with embeddings yet (run the flywheel / backfill).")
        return "\n".join(lines) + "\n"

    lines += [
        f"- evaluated (non-abstained): {result.evaluated}  "
        f"(coverage {result.coverage:.0%}, abstained {result.abstained})",
        "",
        "## preview confusion matrix (leave-one-out)",
        "",
        "| | actual hard | actual easy |",
        "|--|--:|--:|",
        f"| **predicted frontier** | {result.tp} (TP) | {result.fp} (FP) |",
        f"| **predicted local** | {result.fn} (FN) | {result.tn} (TN) |",
        "",
    ]
    if result.evaluated:
        lines += [
            f"- accuracy: {result.accuracy:.0%}",
            f"- frontier recall: {result.frontier_recall:.0%} "
            f"(of turns that really went hard, how many it would send to frontier)",
            f"- frontier send-rate: {result.frontier_send_rate:.0%}  "
            f"(baseline always-local = 0%, always-frontier = 100%)",
            f"- actual hard-rate in this band: {result.actual_hard_rate:.0%}",
            "",
            "**Baselines on the same evaluated set** (cost-aware graduation target):",
            "- always-local: recall 0%, sends 0% to frontier "
            "(cheapest, misses every hard turn)",
            "- always-frontier: recall 100%, sends 100% to frontier "
            "(safest, most expensive)",
            "- the retrieval router must beat always-local on accuracy WITHOUT "
            "dropping frontier recall below the LLM classifier's "
            "(classifier verdicts not yet stamped in the ledger; that comparison "
            "lands once classifier_tier is populated live).",
            "",
        ]

    if result.evaluated < GRADUATION_MIN_MIDDLE:
        need = GRADUATION_MIN_MIDDLE - result.evaluated
        lines.append(
            f"## verdict: INSUFFICIENT DATA - {need} more EVALUATED middle-band "
            "decisions needed")
        lines.append(
            "The matrix above is a preview, not a graduation decision. Keep the "
            "flywheel running; re-run this report as the evaluated set grows.")
    else:
        lines.append("## verdict: SUFFICIENT SAMPLE - apply the graduation criteria")
        lines.append(
            f"Compare accuracy {result.accuracy:.0%} and recall "
            f"{result.frontier_recall:.0%} against the classifier + best-single "
            "baseline; wire live only on a clear win.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="ledger path")
    parser.add_argument("--report", action="store_true",
                        help="print the markdown report (default action)")
    args = parser.parse_args()
    print(build(args.db), end="")


if __name__ == "__main__":
    main()
