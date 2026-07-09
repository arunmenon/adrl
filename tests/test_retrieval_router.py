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


def _filler(n: int = 8, *, hard: bool = False) -> list[dict]:
    """Cluster-B closed rows in distinct sessions, purely to clear the
    MIN_FINALIZED cold-start gate. Being orthogonal to _A they never become
    above-floor neighbors of an _A target, so they don't perturb the vote."""
    return [{"vec": _B, "layer": "heuristic", "hard": hard, "session": f"f{i}"}
            for i in range(n)]


def test_shadow_empty_db_insufficient(tmp_path):
    report = sr.build(tmp_path / "none.db")
    assert "INSUFFICIENT DATA" in report


def test_shadow_cold_start_gate_abstains(tmp_path):
    # 7 finalized (< MIN_FINALIZED=8): the live resolver abstains on every query,
    # so shadow must report zero evaluated even with a clean hard cluster.
    db = tmp_path / "m.db"
    rows = [{"vec": _A, "layer": "middle_default", "hard": True, "session": "t"}]
    rows += [{"vec": _A, "layer": "heuristic", "hard": True, "session": f"n{i}"}
             for i in range(6)]
    _seed(db, rows)  # 7 finalized total
    result = sr.evaluate(db, now=100.0)
    assert result.finalized == 7
    assert result.evaluated == 0        # cold-start gate, not neighbor scarcity


def test_shadow_cold_start_excludes_target_from_finalized_count(tmp_path):
    # Exactly MIN_FINALIZED finalized outcomes, the target being one of them. Each
    # target excludes ITSELF (it was not in memory when routed live), so
    # finalized-1 = 7 < 8 -> abstain, even with 7 confident neighbors that would
    # otherwise vote. Pairs with the TP test (9 finalized -> evaluates) to pin the
    # boundary. Guards the self-count inflation the completeness critic found.
    db = tmp_path / "m.db"
    rows = [{"vec": _A, "layer": "middle_default", "hard": True, "session": "t"}]
    rows += [{"vec": _A, "layer": "heuristic", "hard": True, "session": f"n{i}"}
             for i in range(7)]   # 8 finalized total, 7 confident neighbors
    _seed(db, rows)
    result = sr.evaluate(db, now=100.0)
    assert result.finalized == 8
    assert result.evaluated == 0        # finalized-1 = 7 < MIN_FINALIZED=8


def test_shadow_top_k_then_filter_matches_live(tmp_path):
    # F1 regression: live takes top-K by cosine THEN filters. Organic target with
    # 12 SIMULATOR neighbors ranked ABOVE 6 organic neighbors (all above the 0.75
    # floor). Correct top-K-then-filter fills the top-K=12 with the sims, which the
    # firewall then drops -> abstain. The old filter-during-walk would skip the
    # firewalled sims and reach the 6 organics -> wrongly evaluate. Needs >K
    # neighbors to distinguish the two orderings.
    db = tmp_path / "m.db"

    def near(eps):        # cosine to _A decreases as eps grows; all stay > 0.75
        return [1.0, eps, 0.0, 0.0]

    rows = [{"vec": near(0.0), "layer": "middle_default", "hard": True,
             "source": "organic", "session": "t"}]
    rows += [{"vec": near(0.10 + 0.01 * i), "layer": "heuristic", "hard": True,
              "source": "simulator", "session": f"s{i}"} for i in range(12)]
    rows += [{"vec": near(0.60), "layer": "heuristic", "hard": True,
              "source": "organic", "session": f"o{i}"} for i in range(6)]
    _seed(db, rows)
    result = sr.evaluate(db, now=100.0)
    assert result.middle_total == 1
    assert result.evaluated == 0        # top-12 are all firewalled sims


def test_shadow_true_positive_on_hard_cluster(tmp_path):
    db = tmp_path / "m.db"
    # target + 8 hard cluster-A neighbors (distinct sessions) -> past cold-start,
    # vote frontier, actual hard -> TP. Would fail if decide() returned local.
    rows = [{"vec": _A, "layer": "middle_default", "hard": True, "session": "target"}]
    rows += [{"vec": _A, "layer": "heuristic", "hard": True, "session": f"n{i}"}
             for i in range(8)]
    _seed(db, rows)
    result = sr.evaluate(db, now=100.0)
    assert result.middle_total == 1
    assert result.evaluated == 1
    assert result.tp == 1 and result.fn == 0


def test_shadow_true_negative_on_easy_cluster(tmp_path):
    db = tmp_path / "m.db"
    # easy target + 8 easy neighbors -> vote local, actual easy -> TN (would fail
    # if the vote inverted or the hardness label were wrong).
    rows = [{"vec": _A, "layer": "middle_default", "hard": False, "session": "target"}]
    rows += [{"vec": _A, "layer": "heuristic", "hard": False, "session": f"n{i}"}
             for i in range(8)]
    _seed(db, rows)
    result = sr.evaluate(db, now=100.0)
    assert result.evaluated == 1
    assert result.tn == 1 and result.fp == 0


def test_shadow_similarity_floor_excludes_below_threshold(tmp_path):
    db = tmp_path / "m.db"
    # cluster-A target; 8 cluster-B neighbors are orthogonal (cosine 0 < 0.75
    # floor) -> all filtered by the floor -> abstain, though they clear the count.
    # Exercises the floor half of _neighbors selection (was uncovered).
    rows = [{"vec": _A, "layer": "middle_default", "hard": True, "session": "t"}]
    rows += [{"vec": _B, "layer": "heuristic", "hard": True, "session": f"n{i}"}
             for i in range(8)]
    _seed(db, rows)
    result = sr.evaluate(db, now=100.0)
    assert result.middle_total == 1
    assert result.evaluated == 0        # 8 neighbors, all below the 0.75 floor


def test_shadow_leave_one_out_excludes_same_session(tmp_path):
    db = tmp_path / "m.db"
    # target + a same-session twin + only 4 OTHER-session cluster-A neighbors,
    # plus filler to clear cold-start. Excluding the twin leaves 4 (< 5) -> abstain.
    rows = [{"vec": _A, "layer": "middle_default", "hard": True, "session": "shared"}]
    rows += [{"vec": _A, "layer": "heuristic", "hard": True, "session": "shared"}]  # twin
    rows += [{"vec": _A, "layer": "heuristic", "hard": True, "session": f"n{i}"}
             for i in range(4)]
    rows += _filler(8)
    _seed(db, rows)
    assert sr.evaluate(db, now=100.0).evaluated == 0    # twin excluded, 4 < 5

    # control: one more DISTINCT-session neighbor -> 5 -> evaluates (isolates the
    # same-session skip as the cause, not a count fluke).
    db2 = tmp_path / "m2.db"
    _seed(db2, rows + [{"vec": _A, "layer": "heuristic", "hard": True,
                        "session": "extra"}])
    assert sr.evaluate(db2, now=100.0).evaluated == 1


def test_shadow_firewall_excludes_simulator_from_organic_target(tmp_path):
    db = tmp_path / "m.db"
    # organic middle target + 8 SIMULATOR cluster-A neighbors -> all firewalled.
    rows = [{"vec": _A, "layer": "middle_default", "hard": True,
             "source": "organic", "session": "t"}]
    rows += [{"vec": _A, "layer": "heuristic", "hard": True,
              "source": "simulator", "session": f"n{i}"} for i in range(8)]
    _seed(db, rows)
    result = sr.evaluate(db, now=100.0)
    assert result.middle_total == 1    # target IS detected as middle-band...
    assert result.evaluated == 0       # ...but every neighbor is firewalled

    # control: a SIMULATOR target keeps the simulator neighbors and evaluates,
    # proving the abstention above is the firewall, not neighbor scarcity.
    db2 = tmp_path / "m2.db"
    rows2 = [{"vec": _A, "layer": "middle_default", "hard": True,
              "source": "simulator", "session": "t"}]
    rows2 += [{"vec": _A, "layer": "heuristic", "hard": True,
               "source": "simulator", "session": f"n{i}"} for i in range(8)]
    _seed(db2, rows2)
    assert sr.evaluate(db2, now=100.0).evaluated == 1


def test_shadow_classifier_layer_counts_as_middle(tmp_path):
    db = tmp_path / "m.db"
    rows = [{"vec": _A, "layer": "classifier", "hard": False, "session": "t"}]
    rows += [{"vec": _A, "layer": "heuristic", "hard": False, "session": f"n{i}"}
             for i in range(8)]
    _seed(db, rows)
    assert sr.evaluate(db, now=100.0).middle_total == 1


def test_shadow_only_middle_band_turns_are_scored(tmp_path):
    db = tmp_path / "m.db"
    rows = [{"vec": _A, "layer": "heuristic", "hard": True, "session": f"n{i}"}
            for i in range(8)]
    _seed(db, rows)
    assert sr.evaluate(db, now=100.0).middle_total == 0


def test_shadow_graduation_gate_keys_on_evaluated_not_middle_total(tmp_path):
    db = tmp_path / "m.db"
    # >= GRADUATION_MIN_MIDDLE middle targets, but ALL share one session so
    # leave-one-out excludes every neighbor -> evaluated == 0. If the gate keyed
    # on middle_total it would wrongly print SUFFICIENT SAMPLE.
    n = rr.GRADUATION_MIN_MIDDLE
    rows = [{"vec": _A, "layer": "middle_default", "hard": False, "session": "s"}
            for _ in range(n)]
    rows += _filler(8)   # distinct sessions, below floor: clear cold-start only
    _seed(db, rows)
    result = sr.evaluate(db, now=100.0)
    assert result.middle_total >= rr.GRADUATION_MIN_MIDDLE
    assert result.evaluated == 0
    report = sr.build(db)
    assert "INSUFFICIENT DATA" in report
    assert "SUFFICIENT SAMPLE" not in report
