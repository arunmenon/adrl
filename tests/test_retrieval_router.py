"""Tests for WS4 — retrieval router (retrieval_router.py) + shadow harness.

Covers the shared vote math, the resolver's cold-start / firewall / fail-safe
gates against a fake memory, and the shadow harness's leave-one-out plumbing
against a real seeded ledger.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from router import retrieval_router as rr
from router import shadow_retrieval as sr
from router.memory_ports import NeighborTurn
from router.memory_sqlite import SqliteProvider


def _n(sim, *, hard=False, source="organic", ts=0.0) -> NeighborTurn:
    return NeighborTurn(route_id="x", similarity=sim, rung="local",
                        escalated=hard, outcome_proxy_hard=hard,
                        source=source, ts=ts)


# ── vote math ────────────────────────────────────────────────────────────────

def test_recency_weight_decay():
    assert rr.recency_weight(100.0, 100.0, halflife_s=10.0) == 1.0     # age 0
    assert abs(rr.recency_weight(90.0, 100.0, halflife_s=10.0) - 0.5) < 1e-9  # 1 half-life
    # future ts (clock skew) is clamped to weight 1.0, never > 1
    assert rr.recency_weight(200.0, 100.0, halflife_s=10.0) == 1.0


def test_neighbor_went_hard():
    assert rr.neighbor_went_hard(_n(0.9, hard=True))
    assert not rr.neighbor_went_hard(_n(0.9, hard=False))


def test_decide_frontier_when_neighbors_mostly_hard():
    # 4 hard + 1 easy, all sim 0.9, recency 1.0 (ts==now) -> hard_weight = 4*0.9,
    # total = 5*0.9, fraction 0.8. Assert the weighting math, not just > 0.
    neighbors = [_n(0.9, hard=True) for _ in range(4)] + [_n(0.9, hard=False)]
    verdict = rr.decide(neighbors, now=0.0)
    assert verdict.needs_frontier
    assert verdict.tier == "retrieval:frontier"
    assert abs(verdict.hard_weight - 4 * 0.9) < 1e-6
    assert abs(verdict.total_weight - 5 * 0.9) < 1e-6


def test_decide_local_when_neighbors_mostly_easy():
    neighbors = [_n(0.9, hard=False) for _ in range(4)] + [_n(0.9, hard=True)]
    verdict = rr.decide(neighbors, now=0.0)
    assert not verdict.needs_frontier
    assert verdict.tier == "retrieval:local"
    assert abs(verdict.hard_weight - 1 * 0.9) < 1e-6


def test_decide_threshold_is_inclusive_at_half():
    # exactly 50% hard-weight -> frontier (>= threshold: a tie escalates for safety)
    neighbors = [_n(0.9, hard=True), _n(0.9, hard=False)]
    assert rr.decide(neighbors, now=0.0, hard_vote_threshold=0.5).needs_frontier


def test_decide_threshold_actually_matters():
    # fraction 1/3 hard: local at the 0.5 default, frontier at a 0.3 threshold.
    # Proves HARD_VOTE_THRESHOLD is consulted, not a hardcoded 0.5 tipping point.
    neighbors = [_n(0.9, hard=True), _n(0.9, hard=False), _n(0.9, hard=False)]
    assert not rr.decide(neighbors, now=0.0, hard_vote_threshold=0.5).needs_frontier
    assert rr.decide(neighbors, now=0.0, hard_vote_threshold=0.3).needs_frontier


def test_recency_downweights_stale_hard_cluster():
    # one FRESH easy neighbor vs three STALE hard neighbors; with a short
    # half-life the stale hard cluster decays below the fresh easy vote.
    now = 1_000_000.0
    stale = 0.0
    neighbors = [_n(0.9, hard=True, ts=stale) for _ in range(3)] + \
                [_n(0.9, hard=False, ts=now)]
    hard_weight, total = rr.tally(neighbors, now=now, halflife_s=1.0)
    assert hard_weight < total - hard_weight  # easy weight dominates


# ── resolver gates against a fake memory ─────────────────────────────────────

class FakeMemory:
    def __init__(self, neighbors, *, finalized=100, embedding=(0.1, 0.2, 0.3),
                 raise_on_embed=False):
        self._neighbors = neighbors
        self._finalized = finalized
        self._embedding = list(embedding)
        self._raise_on_embed = raise_on_embed

    def stats(self):
        return {"outcomes_by_status": {"closed_final": self._finalized}}

    def embed_query(self, text):
        if self._raise_on_embed:
            raise RuntimeError("embedder down")
        return self._embedding

    def similar_turns(self, embedding, k):
        return self._neighbors[:k]


def test_resolver_frontier_vote():
    mem = FakeMemory([_n(0.9, hard=True) for _ in range(6)])
    resolver = rr.RetrievalResolver(mem, now_fn=lambda: 0.0)
    verdict = resolver("refactor the whole auth module")
    assert verdict is not None and verdict.needs_frontier
    assert resolver.last is not None and not resolver.last.abstained


def test_resolver_cold_start_abstains_on_few_finalized():
    mem = FakeMemory([_n(0.9, hard=True) for _ in range(6)], finalized=3)
    resolver = rr.RetrievalResolver(mem)
    assert resolver("do something") is None
    assert resolver.last.abstained and "cold-start" in resolver.last.reason


def test_resolver_abstains_on_too_few_confident_neighbors():
    # neighbors below the similarity floor are dropped -> abstain
    mem = FakeMemory([_n(0.5, hard=True) for _ in range(6)])
    resolver = rr.RetrievalResolver(mem)
    assert resolver("something ambiguous") is None
    assert resolver.last.abstained and "confident neighbors" in resolver.last.reason


def test_resolver_is_sim_firewall_drops_simulator_for_organic():
    # 6 confident neighbors, but all simulator-sourced; an organic query drops
    # them all and abstains, while a simulator query keeps them and votes.
    sim_neighbors = [_n(0.9, hard=True, source="simulator") for _ in range(6)]
    organic = rr.RetrievalResolver(FakeMemory(sim_neighbors), query_source="organic")
    assert organic("x") is None
    assert organic.last.n_kept == 0

    simq = rr.RetrievalResolver(FakeMemory(sim_neighbors), query_source="simulator",
                                now_fn=lambda: 0.0)
    assert simq("x") is not None


def test_resolver_never_raises_on_memory_failure():
    mem = FakeMemory([_n(0.9, hard=True) for _ in range(6)], raise_on_embed=True)
    resolver = rr.RetrievalResolver(mem)
    assert resolver("x") is None       # swallowed -> abstain, no exception
    assert resolver.last.abstained


def test_resolver_abstains_on_empty_text():
    resolver = rr.RetrievalResolver(FakeMemory([]))
    assert resolver("") is None
    assert "no instruction text" in resolver.last.reason


# ── shadow harness against a real seeded ledger ──────────────────────────────

def _seed(db_path: Path, rows: list[dict]) -> None:
    SqliteProvider(db_path=db_path)  # schema
    conn = sqlite3.connect(str(db_path))
    for i, row in enumerate(rows):
        rid = f"r{i}"
        conn.execute(
            "INSERT INTO decisions (route_id, ts, session_id, turn_index, source, "
            "instr_sha256, features_json, layer, rung) VALUES (?,?,?,?,?,?,?,?,?)",
            (rid, float(row.get("ts", i)), row.get("session", f"s{i}"), i,
             row.get("source", "organic"), "sha", "{}",
             row.get("layer", "middle_default"), "local"),
        )
        conn.execute(
            "INSERT INTO outcomes (route_id, status, escalated, user_retried, "
            "outcome_proxy_hard) VALUES (?, 'closed_final', ?, 0, ?)",
            (rid, 1 if row.get("hard") else 0, 1 if row.get("hard") else 0),
        )
        vec = np.asarray(row["vec"], dtype="<f4")
        conn.execute(
            "INSERT INTO embeddings (route_id, dim, vec) VALUES (?,?,?)",
            (rid, int(vec.shape[0]), vec.tobytes()),
        )
    conn.commit()
    conn.close()


_A = [1.0, 0.0, 0.0, 0.0]   # cluster A direction
_B = [0.0, 1.0, 0.0, 0.0]   # cluster B direction (orthogonal -> cosine 0, below floor)

# _seed assigns ts = insertion index, so a row's POSITION is its time. Under
# temporal leave-one-out a neighbor only counts if it precedes the target, so
# neighbors are seeded BEFORE the target row in every test below.


def _neighbor(vec=None, *, hard=True, source="organic", session):
    return {"vec": vec or _A, "layer": "heuristic", "hard": hard,
            "source": source, "session": session}


def _target(vec=None, *, layer="middle_default", hard=True, source="organic",
            session="target"):
    return {"vec": vec or _A, "layer": layer, "hard": hard, "source": source,
            "session": session}


def _past(n: int, *, hard: bool = True, source: str = "organic", vec=None) -> list[dict]:
    """n distinct-session neighbor rows, seeded (and so timestamped) before the
    target that follows them."""
    return [_neighbor(vec, hard=hard, source=source, session=f"n{i}")
            for i in range(n)]


def test_shadow_empty_db_insufficient(tmp_path):
    report = sr.build(tmp_path / "none.db")
    assert "INSUFFICIENT DATA" in report


def test_shadow_cold_start_gate_abstains(tmp_path):
    # Only 6 PAST neighbors (< MIN_FINALIZED=8): too little memory existed when the
    # target was routed, so live would cold-start and shadow abstains - even though
    # 6 >= MIN_NEIGHBORS would otherwise vote.
    db = tmp_path / "m.db"
    _seed(db, _past(6) + [_target()])
    result = sr.evaluate(db)
    assert result.middle_total == 1
    assert result.evaluated == 0


def test_shadow_excludes_future_neighbors(tmp_path):
    # The target is seeded FIRST, its neighbors AFTER it: none existed in memory
    # when it was routed, so it must abstain - never scored on future outcomes (#1).
    db = tmp_path / "m.db"
    _seed(db, [_target(session="t")] + _past(10))
    result = sr.evaluate(db)
    assert result.middle_total == 1
    assert result.evaluated == 0


def test_shadow_firewall_does_not_starve_organic(tmp_path):
    # #4 regression (filter-then-cap): 12 SIMULATOR neighbors rank ABOVE 6 organic
    # ones (all above floor, all past). Filtering BEFORE the top-K cut reaches the
    # 6 organics and votes; the old top-K-then-filter let the 12 nearer sims fill K,
    # get firewalled, and starve the organic vote into a wrong abstention.
    db = tmp_path / "m.db"

    def near(eps):        # cosine to _A decreases as eps grows; all stay > 0.75
        return [1.0, eps, 0.0, 0.0]

    rows = [_neighbor(near(0.10 + 0.01 * i), source="simulator", session=f"s{i}")
            for i in range(12)]
    rows += [_neighbor(near(0.60), source="organic", session=f"o{i}")
             for i in range(6)]
    rows += [_target(near(0.0), source="organic")]
    _seed(db, rows)
    result = sr.evaluate(db)
    assert result.middle_total == 1
    assert result.evaluated == 1        # organic neighbors found despite nearer sims


def test_shadow_true_positive_on_hard_cluster(tmp_path):
    # 8 hard past neighbors clear cold-start, vote frontier; actual hard -> TP.
    db = tmp_path / "m.db"
    _seed(db, _past(8, hard=True) + [_target(hard=True)])
    result = sr.evaluate(db)
    assert result.middle_total == 1
    assert result.evaluated == 1
    assert result.tp == 1 and result.fn == 0


def test_shadow_true_negative_on_easy_cluster(tmp_path):
    # 8 easy past neighbors -> vote local, actual easy -> TN.
    db = tmp_path / "m.db"
    _seed(db, _past(8, hard=False) + [_target(hard=False)])
    result = sr.evaluate(db)
    assert result.evaluated == 1
    assert result.tn == 1 and result.fp == 0


def test_shadow_similarity_floor_excludes_below_threshold(tmp_path):
    # 8 past neighbors clear cold-start, but all are cluster-B (cosine 0 < 0.75
    # floor) -> filtered out -> abstain. Exercises the floor.
    db = tmp_path / "m.db"
    _seed(db, _past(8, vec=_B) + [_target(vec=_A)])
    result = sr.evaluate(db)
    assert result.middle_total == 1
    assert result.evaluated == 0


def test_shadow_includes_same_session_past_neighbors(tmp_path):
    # #5: same-session PAST turns WERE in live memory (the trajectory signal), so
    # they must count. 5 same-session past cluster-A neighbors + below-floor filler
    # to clear cold-start -> evaluates using the same-session history. The old
    # blanket same-session exclusion wrongly abstained here.
    db = tmp_path / "m.db"
    rows = [_neighbor(session="shared") for _ in range(5)]   # same session, past
    rows += _past(8, vec=_B)                                 # below-floor filler
    rows += [_target(session="shared")]
    _seed(db, rows)
    assert sr.evaluate(db).evaluated == 1


def test_shadow_firewall_excludes_simulator_from_organic_target(tmp_path):
    # An organic target whose ONLY neighbors are simulator -> all firewalled.
    db = tmp_path / "m.db"
    _seed(db, [_neighbor(source="simulator", session=f"s{i}") for i in range(8)]
              + [_target(source="organic")])
    result = sr.evaluate(db)
    assert result.middle_total == 1   # target IS detected as middle-band...
    assert result.evaluated == 0      # ...but every neighbor is firewalled

    # control: a simulator target keeps the simulator neighbors and evaluates,
    # proving the abstention above is the firewall, not neighbor scarcity.
    db2 = tmp_path / "m2.db"
    _seed(db2, [_neighbor(source="simulator", session=f"s{i}") for i in range(8)]
               + [_target(source="simulator")])
    assert sr.evaluate(db2).evaluated == 1


def test_shadow_classifier_layer_counts_as_middle(tmp_path):
    db = tmp_path / "m.db"
    _seed(db, _past(8, hard=False) + [_target(layer="classifier", hard=False)])
    assert sr.evaluate(db).middle_total == 1


def test_shadow_only_middle_band_turns_are_scored(tmp_path):
    db = tmp_path / "m.db"
    _seed(db, _past(8))   # all heuristic-layer, no middle-band target
    assert sr.evaluate(db).middle_total == 0


def test_shadow_graduation_gate_keys_on_evaluated_not_middle_total(tmp_path):
    # GRADUATION_MIN_MIDDLE cluster-A middle targets in distinct sessions, seeded in
    # time order: target i has i past neighbors, so the first MIN_FINALIZED cold-
    # start -> evaluated = N - MIN_FINALIZED < N. middle_total >= the threshold but
    # evaluated < it, so the report must say INSUFFICIENT (proving the gate keys on
    # evaluated, not middle_total - if it keyed on middle_total it would pass).
    db = tmp_path / "m.db"
    n = rr.GRADUATION_MIN_MIDDLE
    _seed(db, [_target(hard=False, session=f"t{i}") for i in range(n)])
    result = sr.evaluate(db)
    assert result.middle_total >= rr.GRADUATION_MIN_MIDDLE
    assert result.evaluated < rr.GRADUATION_MIN_MIDDLE
    report = sr.build(db)
    assert "INSUFFICIENT DATA" in report
    assert "SUFFICIENT SAMPLE" not in report
