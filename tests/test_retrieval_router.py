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
    neighbors = [_n(0.9, hard=True) for _ in range(4)] + [_n(0.9, hard=False)]
    verdict = rr.decide(neighbors, now=0.0)
    assert verdict.needs_frontier
    assert verdict.tier == "retrieval:frontier"
    assert verdict.hard_weight > 0


def test_decide_local_when_neighbors_mostly_easy():
    neighbors = [_n(0.9, hard=False) for _ in range(4)] + [_n(0.9, hard=True)]
    verdict = rr.decide(neighbors, now=0.0)
    assert not verdict.needs_frontier
    assert verdict.tier == "retrieval:local"


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
_B = [0.0, 1.0, 0.0, 0.0]   # cluster B direction (orthogonal -> cosine 0)


def test_shadow_empty_db_insufficient(tmp_path):
    report = sr.build(tmp_path / "none.db")
    assert "INSUFFICIENT DATA" in report


def test_shadow_true_positive_on_hard_cluster(tmp_path):
    db = tmp_path / "m.db"
    # 1 middle-band HARD turn surrounded by 6 hard cluster-A neighbors (each a
    # distinct session so none is excluded as same-session). Vote -> frontier,
    # actual hard -> TP.
    rows = [{"vec": _A, "layer": "middle_default", "hard": True, "session": "target"}]
    rows += [{"vec": _A, "layer": "heuristic", "hard": True, "session": f"n{i}"}
             for i in range(6)]
    _seed(db, rows)
    result = sr.evaluate(db, now=100.0)
    assert result.middle_total == 1
    assert result.evaluated == 1
    assert result.tp == 1 and result.fn == 0


def test_shadow_leave_one_out_excludes_same_session(tmp_path):
    db = tmp_path / "m.db"
    # target + a same-session twin + only 4 other cluster-A neighbors. Excluding
    # the twin leaves 4 (< MIN_NEIGHBORS=5) -> abstain. Proves same-session skip.
    rows = [{"vec": _A, "layer": "middle_default", "hard": True, "session": "shared"}]
    rows += [{"vec": _A, "layer": "heuristic", "hard": True, "session": "shared"}]  # twin
    rows += [{"vec": _A, "layer": "heuristic", "hard": True, "session": f"n{i}"}
             for i in range(4)]
    _seed(db, rows)
    result = sr.evaluate(db, now=100.0)
    assert result.middle_total == 1
    assert result.evaluated == 0        # abstained: twin excluded, 4 < 5
    assert result.abstained == 1


def test_shadow_firewall_excludes_simulator_from_organic_target(tmp_path):
    db = tmp_path / "m.db"
    # organic middle target with 6 SIMULATOR cluster-A neighbors -> all firewalled
    # -> abstain.
    rows = [{"vec": _A, "layer": "middle_default", "hard": True,
             "source": "organic", "session": "t"}]
    rows += [{"vec": _A, "layer": "heuristic", "hard": True,
              "source": "simulator", "session": f"n{i}"} for i in range(6)]
    _seed(db, rows)
    result = sr.evaluate(db, now=100.0)
    assert result.evaluated == 0

    report = sr.build(db)
    assert "INSUFFICIENT DATA" in report   # 1 middle turn << 300


def test_shadow_only_middle_band_turns_are_scored(tmp_path):
    db = tmp_path / "m.db"
    # a heuristic-layer turn is never a target even with plenty of neighbors
    rows = [{"vec": _A, "layer": "heuristic", "hard": True, "session": f"n{i}"}
            for i in range(8)]
    _seed(db, rows)
    result = sr.evaluate(db, now=100.0)
    assert result.middle_total == 0
