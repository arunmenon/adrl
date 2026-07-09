"""WS4 — the retrieval router: route the ambiguous middle by experience.

The three-layer policy resolves clear-easy and clear-hard with hand rules and
leaves the ambiguous middle to a stub (``middle_default``) or the LLM classifier.
This module offers a third option for that middle band: instead of a rule or a
cold classifier call, *look up how turns like this one actually went* and vote.

  embed the instruction (search_query: prefix)
    -> cosine top-K over the in-RAM projection of past decisions (WS1 memory)
    -> keep only confident, non-contaminating neighbors
    -> outcome-weighted + recency-weighted vote: did similar turns need frontier?
    -> a binary verdict (local-cascade vs frontier) OR abstain.

It plugs into the SAME injected ``classifier`` slot ``route_turn`` already
consults (``resolver(text) -> verdict|None`` with ``.needs_frontier`` / ``.tier``),
so wiring it live is a one-line injection — but it is SHADOW-FIRST: not wired
into the live policy until ``shadow_retrieval`` shows it beats both the LLM
classifier and the best-single-model baseline on a cost-aware objective without
regressing frontier recall (the graduation gate).

Why this generalizes where regex does not: it needs no lexicon and no retraining
— every new finalized outcome immediately sharpens the next similar decision.

SELECTION-BIAS GUARDS built in:
  * cold-start abstention — needs >= MIN_NEIGHBORS confident neighbors AND a
    minimum of finalized outcomes in memory, else abstains (falls through to the
    classifier / middle_default). Early behavior is therefore identical to today.
  * is_sim firewall — when serving ORGANIC traffic, synthetic (simulator)
    neighbors are dropped so fuel runs can never flood a real-traffic vote. A
    simulator query may use all neighbors (that is what shadow/fuel wants).
  * recency weighting — older outcomes decay (policy/model regimes drift), so a
    stale cluster cannot outvote fresh evidence.

NOT yet fully wired (documented, not faked): per-session neighbor caps and
instr_sha256 dedup need ``session_id`` / ``instr_sha256`` on ``NeighborTurn`` +
the projection; until the projection carries them a single chatty session could
over-weight the vote. Tracked for the projection-enrichment follow-up; the
firewall + similarity floor bound the risk meanwhile.

FAIL-SAFE: any failure -> abstain (``None``). A memory outage never blocks a
route; it just removes this layer's opinion.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from .memory_ports import NeighborTurn

# ── shared knobs (shadow harness imports these so live == evaluated) ──────────
K = 12
MIN_NEIGHBORS = 5
MIN_SIMILARITY = 0.75
MIN_FINALIZED = 8               # global data-sufficiency gate before voting
HARD_VOTE_THRESHOLD = 0.5       # hard-weight fraction at/above this -> frontier
RECENCY_HALFLIFE_S = 30 * 86_400.0   # a 30-day-old outcome counts half
GRADUATION_MIN_MIDDLE = 300     # middle-band decisions needed to judge graduation


@dataclass
class RetrievalVerdict:
    """The classifier-slot-compatible verdict (``.needs_frontier`` / ``.tier``)."""

    needs_frontier: bool
    tier: str                   # "retrieval:frontier" | "retrieval:local"
    confidence: float           # 0..1, distance of the hard-fraction from the threshold
    n_neighbors: int
    hard_weight: float
    total_weight: float
    reason: str


@dataclass
class ResolveTrace:
    """Provenance of the most recent ``resolve`` — exposed as ``resolver.last``
    for the shadow harness and live debugging. Never part of the routing path."""

    abstained: bool
    reason: str
    n_raw: int = 0
    n_kept: int = 0
    finalized_available: int = 0
    verdict: Optional[RetrievalVerdict] = None


def recency_weight(ts: float, now: float, halflife_s: float = RECENCY_HALFLIFE_S) -> float:
    """Exponential decay by age; 1.0 for a just-recorded outcome, 0.5 at one
    half-life. Clamped so a clock skew (future ts) never exceeds 1.0."""
    if halflife_s <= 0:
        return 1.0
    age = max(0.0, now - float(ts or 0.0))
    return math.pow(0.5, age / halflife_s)


def neighbor_went_hard(neighbor: NeighborTurn) -> bool:
    """A neighbor 'needed frontier' if it escalated or its outcome proxy is hard."""
    return bool(neighbor.escalated) or bool(neighbor.outcome_proxy_hard)


def tally(neighbors: Sequence[NeighborTurn], *, now: float,
          halflife_s: float = RECENCY_HALFLIFE_S) -> tuple[float, float]:
    """Outcome-weighted + recency-weighted vote. Returns (hard_weight,
    total_weight); each neighbor contributes ``similarity * recency`` to the
    total and, if it went hard, to the hard weight. Shared by the live resolver
    and the shadow harness so the vote can never diverge between them."""
    hard_weight = 0.0
    total_weight = 0.0
    for neighbor in neighbors:
        weight = float(neighbor.similarity) * recency_weight(neighbor.ts, now, halflife_s)
        if weight <= 0.0:
            continue
        total_weight += weight
        if neighbor_went_hard(neighbor):
            hard_weight += weight
    return hard_weight, total_weight


def decide(neighbors: Sequence[NeighborTurn], *, now: float,
           hard_vote_threshold: float = HARD_VOTE_THRESHOLD,
           halflife_s: float = RECENCY_HALFLIFE_S) -> RetrievalVerdict:
    """Turn a set of kept neighbors into a verdict (assumes the caller already
    enforced the neighbor-count / similarity gates)."""
    hard_weight, total_weight = tally(neighbors, now=now, halflife_s=halflife_s)
    hard_fraction = (hard_weight / total_weight) if total_weight else 0.0
    needs_frontier = hard_fraction >= hard_vote_threshold
    # confidence: how far the hard-fraction sits from the decision threshold,
    # normalized into 0..1 by the larger side of the split.
    span = max(hard_vote_threshold, 1.0 - hard_vote_threshold) or 1.0
    confidence = min(1.0, abs(hard_fraction - hard_vote_threshold) / span)
    return RetrievalVerdict(
        needs_frontier=needs_frontier,
        tier="retrieval:frontier" if needs_frontier else "retrieval:local",
        confidence=confidence,
        n_neighbors=len(neighbors),
        hard_weight=hard_weight,
        total_weight=total_weight,
        reason=(f"{len(neighbors)} neighbors, hard-weight fraction "
                f"{hard_fraction:.0%} vs threshold {hard_vote_threshold:.0%}"),
    )


class RetrievalResolver:
    """Callable that resolves the ambiguous middle from memory, or abstains.

    Constructed with the WS1 ``RouterMemory`` facade (``embed_query`` +
    ``similar_turns`` + ``stats``). ``query_source`` names the traffic this
    instance serves so the is_sim firewall knows whether to drop synthetic
    neighbors.
    """

    def __init__(self, memory: Any, *, k: int = K,
                 min_neighbors: int = MIN_NEIGHBORS,
                 min_similarity: float = MIN_SIMILARITY,
                 min_finalized: int = MIN_FINALIZED,
                 hard_vote_threshold: float = HARD_VOTE_THRESHOLD,
                 halflife_s: float = RECENCY_HALFLIFE_S,
                 query_source: str = "organic",
                 now_fn: Optional[Callable[[], float]] = None):
        self.memory = memory
        self.k = k
        self.min_neighbors = min_neighbors
        self.min_similarity = min_similarity
        self.min_finalized = min_finalized
        self.hard_vote_threshold = hard_vote_threshold
        self.halflife_s = halflife_s
        self.query_source = query_source
        self._now_fn = now_fn or time.time
        self.last: Optional[ResolveTrace] = None

    # ── firewall + gates ─────────────────────────────────────────────────────

    def _finalized_available(self) -> int:
        stats = self.memory.stats()
        by_status = stats.get("outcomes_by_status", {}) if isinstance(stats, dict) else {}
        return int((by_status.get("closed_final", 0) or 0)
                   + (by_status.get("closed_turn", 0) or 0))

    def _keep(self, neighbor: NeighborTurn) -> bool:
        """Confident enough, and not a synthetic vote leaking into organic."""
        if float(neighbor.similarity) < self.min_similarity:
            return False
        if self.query_source == "organic" and neighbor.source == "simulator":
            return False   # is_sim firewall
        return True

    # ── the decision ─────────────────────────────────────────────────────────

    def resolve(self, text: str) -> Optional[RetrievalVerdict]:
        """Return a verdict, or None to abstain (fall through to the classifier).
        Never raises."""
        try:
            return self._resolve(text)
        except Exception:
            self.last = ResolveTrace(abstained=True, reason="error")
            return None

    __call__ = resolve

    def _abstain(self, reason: str, **trace) -> None:
        self.last = ResolveTrace(abstained=True, reason=reason, **trace)
        return None

    def _resolve(self, text: str) -> Optional[RetrievalVerdict]:
        if not text:
            return self._abstain("no instruction text")
        finalized = self._finalized_available()
        if finalized < self.min_finalized:
            return self._abstain(
                f"cold-start: only {finalized} finalized outcomes "
                f"(< {self.min_finalized})", finalized_available=finalized)
        embedding = self.memory.embed_query(text)
        if not embedding:
            return self._abstain("no embedding (embedder unavailable)",
                                 finalized_available=finalized)
        raw = self.memory.similar_turns(embedding, self.k)
        kept = [n for n in raw if self._keep(n)]
        if len(kept) < self.min_neighbors:
            return self._abstain(
                f"only {len(kept)} confident neighbors (< {self.min_neighbors})",
                n_raw=len(raw), n_kept=len(kept), finalized_available=finalized)
        verdict = decide(kept, now=self._now_fn(),
                         hard_vote_threshold=self.hard_vote_threshold,
                         halflife_s=self.halflife_s)
        self.last = ResolveTrace(
            abstained=False, reason=verdict.reason, n_raw=len(raw),
            n_kept=len(kept), finalized_available=finalized, verdict=verdict)
        return verdict
