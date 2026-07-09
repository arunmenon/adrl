"""WS4 shadow harness — would the retrieval router beat today's middle?

Retrospective, read-only evaluation of ``retrieval_router`` over the ledger's
ambiguous-middle decisions (``layer in {middle_default, classifier}``) — the
only turns the retrieval layer would ever touch. For each such turn we ask the
memory, using LEAVE-ONE-OUT, "how did turns like this one go?" and compare the
vote to what actually happened.

  for each middle-band turn t with an embedding:
    neighbors = cosine top-K over ALL OTHER closed turns
                (exclude t and t's own session -> no leakage;
                 similarity floor + is_sim firewall, same as live)
    if too few confident neighbors -> ABSTAIN (t falls through to the classifier)
    else -> decide() (the SAME vote the live resolver uses)
    actual = did t really go hard? (escalated | user_retried | outcome_proxy_hard)

The vote math is imported from ``retrieval_router`` (``decide``), never
re-implemented, so shadow and live cannot drift.

GRADUATION GATE (plan): the retrieval layer is wired live only when, over at
least ``GRADUATION_MIN_MIDDLE`` middle-band decisions, it beats BOTH the LLM
classifier and the best-single-model baseline on a cost-aware objective, with
frontier recall never below the classifier's. Until that many middle-band turns
exist this report returns INSUFFICIENT DATA with the current confusion matrix as
a *preview only* — never a spurious pass on a handful of rows.

Known approximation: leave-one-out uses each turn's STORED document embedding as
the query vector (the raw text is never stored, by design), so it skips the
live ``search_query:`` prefix swap. Rankings are close; absolute numbers on a
tiny sample are indicative, not final — another reason the gate needs volume.

CLI:
    PYTHONPATH=src python -m router.shadow_retrieval --report [--db PATH]
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from router.memory_ports import NeighborTurn
from router.retrieval_router import (
    GRADUATION_MIN_MIDDLE,
    K,
    MIN_NEIGHBORS,
    MIN_SIMILARITY,
    RECENCY_HALFLIFE_S,
    decide,
)

DEFAULT_DB_PATH = Path("data/router-memory.db")
MIDDLE_LAYERS = ("middle_default", "classifier")
_CLOSED = ("closed_turn", "closed_final")


@dataclass
class _Row:
    route_id: str
    session_id: str
    layer: str
    source: str
    ts: float
    escalated: bool
    proxy_hard: Optional[bool]
    user_retried: Optional[bool]
    rung: str
    vector: np.ndarray          # L2-normalized


def _went_hard(row: _Row) -> bool:
    return bool(row.escalated) or bool(row.user_retried) or bool(row.proxy_hard)


def _load(db_path: Path) -> list[_Row]:
    """All closed decisions that carry an embedding, with normalized vectors.
    Degrades to [] on any DB problem."""
    try:
        conn = sqlite3.connect(str(db_path))
    except Exception:
        return []
    try:
        raw = conn.execute(
            "SELECT d.route_id, d.session_id, d.layer, d.source, d.ts, "
            "o.escalated, o.outcome_proxy_hard, o.user_retried, d.rung, "
            "e.dim, e.vec "
            "FROM decisions d "
            "JOIN outcomes o ON o.route_id = d.route_id "
            "JOIN embeddings e ON e.route_id = d.route_id "
            "WHERE o.status IN (?, ?)",
            _CLOSED,
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    except Exception:
        return []
    finally:
        conn.close()

    rows: list[_Row] = []
    dim0: Optional[int] = None
    for (route_id, session_id, layer, source, ts, escalated, proxy_hard,
         user_retried, rung, dim, blob) in raw:
        if not isinstance(blob, (bytes, memoryview)) or len(blob) % 4 != 0:
            continue
        vector = np.frombuffer(blob, dtype="<f4")
        if dim and vector.shape[0] != dim:
            continue
        if dim0 is None:
            dim0 = vector.shape[0]
        elif vector.shape[0] != dim0:
            continue  # heterogeneous dim (embedder swap) — cannot compare
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            continue
        rows.append(_Row(
            route_id=route_id, session_id=session_id, layer=layer,
            source=source, ts=float(ts or 0.0),
            escalated=bool(escalated),
            proxy_hard=None if proxy_hard is None else bool(proxy_hard),
            user_retried=None if user_retried is None else bool(user_retried),
            rung=rung, vector=vector / norm))
    return rows


def _neighbors_for(target: _Row, rows: list[_Row], matrix: np.ndarray,
                   index: dict[str, int]) -> list[NeighborTurn]:
    """Leave-one-out kNN: cosine of target against all rows, minus itself and
    its own session, with the live similarity floor + is_sim firewall applied."""
    sims = matrix @ target.vector
    order = np.argsort(-sims)
    kept: list[NeighborTurn] = []
    for j in order:
        other = rows[j]
        if other.route_id == target.route_id:
            continue
        if other.session_id == target.session_id:
            continue  # no same-session leakage
        similarity = float(sims[j])
        if similarity < MIN_SIMILARITY:
            break  # sorted desc — nothing else qualifies
        if target.source == "organic" and other.source == "simulator":
            continue  # is_sim firewall
        kept.append(NeighborTurn(
            route_id=other.route_id, similarity=similarity, rung=other.rung,
            escalated=other.escalated, outcome_proxy_hard=other.proxy_hard,
            source=other.source, ts=other.ts))
        if len(kept) >= K:
            break
    return kept


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
    """Run the leave-one-out shadow evaluation. ``now`` is injectable so tests
    are deterministic; defaults to the newest ledger ts (not wall-clock) so the
    recency weighting is reproducible from the data alone."""
    rows = _load(db_path)
    if not rows:
        return ShadowResult(0, 0, 0, 0, 0, 0, 0, now or 0.0)
    if now is None:
        now = max(r.ts for r in rows)
    matrix = np.vstack([r.vector for r in rows])
    index = {r.route_id: i for i, r in enumerate(rows)}
    middle = [r for r in rows if r.layer in MIDDLE_LAYERS]

    tp = fp = fn = tn = 0
    evaluated = 0
    for target in middle:
        neighbors = _neighbors_for(target, rows, matrix, index)
        if len(neighbors) < MIN_NEIGHBORS:
            continue  # abstain
        verdict = decide(neighbors, now=now)
        evaluated += 1
        actual_hard = _went_hard(target)
        if verdict.needs_frontier and actual_hard:
            tp += 1
        elif verdict.needs_frontier and not actual_hard:
            fp += 1
        elif not verdict.needs_frontier and actual_hard:
            fn += 1
        else:
            tn += 1
    return ShadowResult(
        middle_total=len(middle), evaluated=evaluated,
        abstained=len(middle) - evaluated, tp=tp, fp=fp, fn=fn, tn=tn, now=now)


# ── reporter ───────────────────────────────────────────────────────────────


def build(db_path: Path = DEFAULT_DB_PATH, *, now: Optional[float] = None) -> str:
    """Markdown graduation report (pure)."""
    result = evaluate(db_path, now=now)
    lines = [
        "# retrieval router — shadow evaluation (WS4)",
        "",
        f"- db: `{db_path}`",
        f"- middle-band turns with embeddings: {result.middle_total}",
        f"- graduation needs: >= {GRADUATION_MIN_MIDDLE} middle-band decisions",
    ]
    if result.middle_total == 0:
        lines.append("- verdict: **INSUFFICIENT DATA** — no middle-band turns "
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
            f"- always-local: recall 0%, sends 0% to frontier "
            f"(cheapest, misses every hard turn)",
            f"- always-frontier: recall 100%, sends 100% to frontier "
            f"(safest, most expensive)",
            "- the retrieval router must beat always-local on accuracy WITHOUT "
            "dropping frontier recall below the LLM classifier's "
            "(classifier verdicts not yet stamped in the ledger — that comparison "
            "lands once classifier_tier is populated live).",
            "",
        ]

    if result.middle_total < GRADUATION_MIN_MIDDLE:
        need = GRADUATION_MIN_MIDDLE - result.middle_total
        lines.append(
            f"## verdict: INSUFFICIENT DATA — {need} more middle-band decisions needed")
        lines.append(
            "The matrix above is a preview, not a graduation decision. Keep the "
            "flywheel running; re-run this report as the middle band grows.")
    else:
        lines.append("## verdict: SUFFICIENT SAMPLE — apply the graduation criteria")
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
