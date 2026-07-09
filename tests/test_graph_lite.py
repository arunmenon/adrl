"""Tests for the graph-lite projection (src/router/graph_lite.py).

Builds a small synthetic ledger (decisions/outcomes rows inserted directly
per the shared DDL), projects it, and exercises the adjacency-list graph:
upsert identity, typed neighbor filtering, cycle-safe k-hop, ancestry walks,
direction semantics, and reprojection idempotence.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from router.graph_lite import (
    EDGE_DECIDED_BY,
    EDGE_ESCALATED_TO,
    EDGE_IN_SESSION,
    EDGE_ROUTED_TO,
    EDGE_TRIPPED,
    NODE_MODEL,
    NODE_RULE,
    NODE_SESSION,
    NODE_TRIPWIRE,
    NODE_TURN,
    GraphLite,
    build,
    project_from_ledger,
)

LEDGER_DDL = """
CREATE TABLE decisions (
    route_id TEXT PRIMARY KEY, ts REAL, session_id TEXT, turn_index INT,
    source TEXT, instr_sha256 TEXT, features_json TEXT, layer TEXT,
    rung TEXT, cascade INT, score REAL, reason TEXT, classifier_tier TEXT,
    propensity TEXT, policy_version TEXT, classifier_ms REAL, decision_ms REAL
);
CREATE TABLE outcomes (
    route_id TEXT PRIMARY KEY REFERENCES decisions(route_id), status TEXT,
    escalated INT, tripwire_name TEXT, tripwire_type TEXT, edit_failures INT,
    error_results INT, output_tokens INT, latency_ms REAL, cost_estimate REAL,
    interrupted INT, user_retried INT, outcome_proxy_hard INT
);
"""


def _insert_decision(conn, route_id, session_id, turn_index, rung, layer,
                     fired_rules=None, ts=1000.0, source="simulator"):
    features = {"verb_class": "edit"}
    if fired_rules is not None:
        features["fired_rules"] = fired_rules
    conn.execute(
        "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (route_id, ts + turn_index, session_id, turn_index, source, None,
         json.dumps(features), layer, rung, 0, 0.4, "test", None, layer,
         "v1", 0.0, 1.0))


def _insert_outcome(conn, route_id, status="closed_turn", escalated=0,
                    tripwire_name=None, tripwire_type=None):
    conn.execute(
        "INSERT INTO outcomes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (route_id, status, escalated, tripwire_name, tripwire_type,
         0, 0, 100, 500.0, 0.001, 0, 0, 0))


@pytest.fixture()
def ledger_db(tmp_path):
    """A synthetic ledger: 3 turns in one session, 1 in another.

    t1: local via heuristic rule small_edit_local; closed, escalated,
        tripwire edit_fail_x2.
    t2: frontier via gate:hard_intent (no fired_rules -> layer is the rule);
        closed clean.
    t3: local via heuristic; outcome still PENDING (must not project edges).
    t4: other session, cheap_cloud; no outcome row at all.
    """
    db = tmp_path / "router-memory.db"
    conn = sqlite3.connect(db)
    conn.executescript(LEDGER_DDL)
    _insert_decision(conn, "t1", "sess-A", 0, "local", "heuristic",
                     fired_rules=["small_edit_local"])
    _insert_decision(conn, "t2", "sess-A", 1, "frontier", "gate:hard_intent")
    _insert_decision(conn, "t3", "sess-A", 2, "local", "heuristic",
                     fired_rules=["small_edit_local"])
    _insert_decision(conn, "t4", "sess-B", 0, "cheap_cloud",
                     "middle_default")
    _insert_outcome(conn, "t1", status="closed_turn", escalated=1,
                    tripwire_name="edit_fail_x2", tripwire_type="edit")
    _insert_outcome(conn, "t2", status="closed_final")
    _insert_outcome(conn, "t3", status="pending", escalated=1,
                    tripwire_name="tool_loop", tripwire_type="loop")
    conn.commit()
    conn.close()
    return db


# ------------------------------------------------------------- primitives --


def test_upsert_node_uniqueness(tmp_path):
    g = GraphLite(tmp_path / "g.db")
    id1 = g.upsert_node(NODE_MODEL, "local", {"a": 1})
    id2 = g.upsert_node(NODE_MODEL, "local", {"a": 2})
    assert id1 == id2
    rows = g._conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE kind=? AND key=?",
        (NODE_MODEL, "local")).fetchone()[0]
    assert rows == 1
    # props updated on conflict with explicit props, kept with props=None
    assert g.get_node(NODE_MODEL, "local")["props"] == {"a": 2}
    g.upsert_node(NODE_MODEL, "local")
    assert g.get_node(NODE_MODEL, "local")["props"] == {"a": 2}
    # same key under a different kind is a different node
    id3 = g.upsert_node(NODE_RULE, "local")
    assert id3 != id1
    g.close()


def test_neighbors_typed_filtering_and_direction(tmp_path):
    g = GraphLite(tmp_path / "g.db")
    turn = g.upsert_node(NODE_TURN, "t1")
    sess = g.upsert_node(NODE_SESSION, "s1")
    model = g.upsert_node(NODE_MODEL, "local")
    g.add_edge(turn, sess, EDGE_IN_SESSION)
    g.add_edge(turn, model, EDGE_ROUTED_TO)

    # out, unfiltered: both neighbors
    out = g.neighbors(turn, direction="out")
    assert {n["key"] for n, _ in out} == {"s1", "local"}
    # typed filter narrows to one
    only_model = g.neighbors(turn, edge_kinds=[EDGE_ROUTED_TO],
                             direction="out")
    assert [(n["kind"], n["key"]) for n, _ in only_model] == \
        [(NODE_MODEL, "local")]
    assert only_model[0][1]["kind"] == EDGE_ROUTED_TO
    # direction: turn has no in-edges; session sees the turn inbound
    assert g.neighbors(turn, direction="in") == []
    inbound = g.neighbors(sess, direction="in")
    assert [(n["kind"], n["key"]) for n, _ in inbound] == [(NODE_TURN, "t1")]
    # both = union of in and out
    assert len(g.neighbors(sess, direction="both")) == 1
    assert len(g.neighbors(turn, direction="both")) == 2
    with pytest.raises(ValueError):
        g.neighbors(turn, direction="sideways")
    g.close()


def test_k_hop_known_topology_with_cycle(tmp_path):
    g = GraphLite(tmp_path / "g.db")
    # a -> b -> c -> a (cycle), c -> d, d -> e
    ids = {k: g.upsert_node(NODE_TURN, k) for k in "abcde"}
    g.add_edge(ids["a"], ids["b"], "next")
    g.add_edge(ids["b"], ids["c"], "next")
    g.add_edge(ids["c"], ids["a"], "next")  # the cycle edge
    g.add_edge(ids["c"], ids["d"], "next")
    g.add_edge(ids["d"], ids["e"], "next")

    hops = g.k_hop(ids["a"], 2, direction="out")
    assert hops == {ids["a"]: 0, ids["b"]: 1, ids["c"]: 2}
    # deep k: terminates despite the cycle, minimum hop counts kept
    hops = g.k_hop(ids["a"], 10, direction="out")
    assert hops == {ids["a"]: 0, ids["b"]: 1, ids["c"]: 2,
                    ids["d"]: 3, ids["e"]: 4}
    # direction=both from c reaches b (in) and a/d (mixed) at hop 1
    hops = g.k_hop(ids["c"], 1, direction="both")
    assert hops == {ids["c"]: 0, ids["a"]: 1, ids["b"]: 1, ids["d"]: 1}
    # k=0 is just the start node
    assert g.k_hop(ids["a"], 0) == {ids["a"]: 0}
    g.close()


def test_ancestry_directed_walk(tmp_path):
    g = GraphLite(tmp_path / "g.db")
    ids = {k: g.upsert_node(NODE_RULE, k) for k in "abcd"}
    g.add_edge(ids["a"], ids["b"], "derived_from")
    g.add_edge(ids["b"], ids["c"], "derived_from")
    g.add_edge(ids["c"], ids["d"], "other_kind")   # wrong kind: not followed
    g.add_edge(ids["c"], ids["a"], "derived_from")  # cycle back: not revisited

    path = g.ancestry(ids["a"], ["derived_from"], max_depth=10)
    assert [n["key"] for n in path] == ["a", "b", "c"]
    # max_depth truncates the walk
    path = g.ancestry(ids["a"], ["derived_from"], max_depth=1)
    assert [n["key"] for n in path] == ["a", "b"]
    # unknown node -> empty path
    assert g.ancestry(99999, ["derived_from"]) == []
    g.close()


# ------------------------------------------------------------- projection --


def test_project_from_ledger_nodes_and_edges(ledger_db):
    stats = project_from_ledger(ledger_db)
    assert stats["decisions_projected"] == 4
    assert stats["closed_outcomes_projected"] == 2   # t3 is pending

    assert stats["nodes"][NODE_TURN] == 4            # one per route_id
    assert stats["nodes"][NODE_SESSION] == 2
    # models: local, frontier, cheap_cloud (decisions) — t1's escalation
    # from local targets cheap_cloud, already present
    assert stats["nodes"][NODE_MODEL] == 3
    # rules: small_edit_local + the layers that decided without fired_rules
    assert stats["nodes"][NODE_RULE] == 3
    assert stats["nodes"][NODE_TRIPWIRE] == 1        # t1 only; t3 pending

    assert stats["edges"][EDGE_IN_SESSION] == 4
    assert stats["edges"][EDGE_ROUTED_TO] == 4
    assert stats["edges"][EDGE_DECIDED_BY] == 4
    assert stats["edges"][EDGE_ESCALATED_TO] == 1
    assert stats["edges"][EDGE_TRIPPED] == 1


def test_projection_edge_semantics(ledger_db):
    project_from_ledger(ledger_db)
    g = GraphLite(ledger_db)

    t1 = g.get_node(NODE_TURN, "t1")
    # routed local, escalated one rung up to cheap_cloud
    routed = g.neighbors(t1["id"], edge_kinds=[EDGE_ROUTED_TO])
    assert [n["key"] for n, _ in routed] == ["local"]
    esc = g.neighbors(t1["id"], edge_kinds=[EDGE_ESCALATED_TO])
    assert [n["key"] for n, _ in esc] == ["cheap_cloud"]
    assert esc[0][1]["props"] == {"from_rung": "local"}
    tripped = g.neighbors(t1["id"], edge_kinds=[EDGE_TRIPPED])
    assert [(n["kind"], n["key"]) for n, _ in tripped] == \
        [(NODE_TRIPWIRE, "edit_fail_x2")]
    assert tripped[0][0]["props"] == {"type": "edit"}
    decided = g.neighbors(t1["id"], edge_kinds=[EDGE_DECIDED_BY])
    assert [n["key"] for n, _ in decided] == ["small_edit_local"]

    # t2: no fired_rules -> the deciding layer is the rule node
    t2 = g.get_node(NODE_TURN, "t2")
    decided = g.neighbors(t2["id"], edge_kinds=[EDGE_DECIDED_BY])
    assert [n["key"] for n, _ in decided] == ["gate:hard_intent"]
    # frontier is the top rung: escalation would have no target, and t2
    # did not escalate anyway
    assert g.neighbors(t2["id"], edge_kinds=[EDGE_ESCALATED_TO]) == []

    # t3: pending outcome must contribute NO outcome edges
    t3 = g.get_node(NODE_TURN, "t3")
    assert g.neighbors(t3["id"], edge_kinds=[EDGE_ESCALATED_TO,
                                             EDGE_TRIPPED]) == []

    # session fan-in: sess-A gathers its three turns
    sess = g.get_node(NODE_SESSION, "sess-A")
    inbound = g.neighbors(sess["id"], edge_kinds=[EDGE_IN_SESSION],
                          direction="in")
    assert {n["key"] for n, _ in inbound} == {"t1", "t2", "t3"}
    g.close()


def test_traversal_over_projection(ledger_db):
    """2-hop from a turn reaches its session siblings through the session."""
    project_from_ledger(ledger_db)
    g = GraphLite(ledger_db)
    t1 = g.get_node(NODE_TURN, "t1")
    hops = g.k_hop(t1["id"], 2, edge_kinds=[EDGE_IN_SESSION],
                   direction="both")
    keys = {g._conn.execute("SELECT key FROM nodes WHERE id=?",
                            (nid,)).fetchone()[0]
            for nid, hop in hops.items() if hop == 2}
    assert keys == {"t2", "t3"}
    g.close()


def test_reprojection_idempotence(ledger_db):
    first = project_from_ledger(ledger_db)
    second = project_from_ledger(ledger_db)
    assert first["nodes"] == second["nodes"]
    assert first["edges"] == second["edges"]
    # absolute row counts, not just per-kind (guards against duplication)
    conn = sqlite3.connect(ledger_db)
    assert conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == \
        sum(second["nodes"].values())
    assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == \
        sum(second["edges"].values())
    conn.close()


def test_project_accepts_open_connection(ledger_db):
    conn = sqlite3.connect(ledger_db)
    stats = project_from_ledger(conn)
    assert stats["nodes"][NODE_TURN] == 4
    # connection is not closed for us
    conn.execute("SELECT 1").fetchone()
    conn.close()


def test_project_missing_ledger_tables(tmp_path):
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    stats = project_from_ledger(empty)
    assert stats["decisions_projected"] == 0
    assert stats["nodes"] == {} and stats["edges"] == {}


def test_report_build_counts_only(ledger_db):
    stats = project_from_ledger(ledger_db)
    md = build(stats)
    assert "# Graph-lite projection report" in md
    assert f"| {NODE_TURN} | 4 |" in md
    assert f"| {EDGE_ESCALATED_TO} | 1 |" in md
    # counts only — no route ids, session ids, or rule payloads leak
    for payload in ("t1", "sess-A", "small_edit_local", "edit_fail_x2"):
        assert payload not in md
