"""Graph-lite projection over the router memory ledger (WS1, Unit C).

A production-grade adjacency-list graph, lite: typed ``nodes``/``edges``
tables in the SAME SQLite file as the event ledger (data/router-memory.db),
rebuildable at any time from the ledger tables (``decisions``/``outcomes``).
No Neo4j, no query language — traversals are pure SQL-backed adjacency
lookups over indexed (src, kind) / (dst, kind) columns, so nothing loads
the whole graph into RAM.

The projection is DISPOSABLE by design (Engram ADR 0005): delete + rebuild
is the idempotence strategy, and the ledger stays the single source of
truth. Privacy contract: only ledger metadata is projected — instruction
text never exists in the ledger, so it can never exist here.

Node kinds:  turn, session, model, rule, tripwire
             (entity is deferred — no extraction pipeline yet; see TODO)
Edge kinds:  in_session   turn -> session
             decided_by   turn -> rule   (features_json.fired_rules + layer)
             routed_to    turn -> model  (the decided rung)
             escalated_to turn -> model  (next rung up, when outcome escalated)
             tripped      turn -> tripwire

Usage:
    PYTHONPATH=src .venv/bin/python -m router.graph_lite --report
    PYTHONPATH=src .venv/bin/python -m router.graph_lite --db data/router-memory.db --report
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import deque
from pathlib import Path
from typing import Iterable, Optional, Union

from router.escalation_controller import NEXT_RUNG
from router.policy import REGISTRY

# ---------------------------------------------------------------- kinds ----

NODE_TURN = "turn"
NODE_SESSION = "session"
NODE_MODEL = "model"
NODE_RULE = "rule"
NODE_TRIPWIRE = "tripwire"
# TODO(entity): NODE_ENTITY = "entity" is deferred — there is no entity
# extraction pipeline yet (files/modules/error signatures). When it lands,
# entity nodes + `touches` edges become one more projection pass here.

NODE_KINDS = (NODE_TURN, NODE_SESSION, NODE_MODEL, NODE_RULE, NODE_TRIPWIRE)

EDGE_IN_SESSION = "in_session"
EDGE_DECIDED_BY = "decided_by"
EDGE_ROUTED_TO = "routed_to"
EDGE_ESCALATED_TO = "escalated_to"
EDGE_TRIPPED = "tripped"

EDGE_KINDS = (EDGE_IN_SESSION, EDGE_DECIDED_BY, EDGE_ROUTED_TO,
              EDGE_ESCALATED_TO, EDGE_TRIPPED)

# Escalation ladder: semantic ladder from the escalation controller
# (local-code/local-small -> cheap-cloud -> frontier) UNIONED with the
# policy-registry vocabulary (local -> cheap_cloud -> frontier, ordered by
# cost_rank) so both rung spellings found in ledger rows resolve. Top rungs
# have no entry: an escalation from the top produces no escalated_to edge.
def _policy_ladder() -> dict[str, str]:
    ordered = sorted(REGISTRY, key=lambda k: REGISTRY[k]["cost_rank"])
    return {a: b for a, b in zip(ordered, ordered[1:])}


ESCALATION_TARGET: dict[str, str] = {**_policy_ladder(), **NEXT_RUNG}

_DIRECTIONS = ("out", "in", "both")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id         INTEGER PRIMARY KEY,
    kind       TEXT,
    key        TEXT,
    props_json TEXT,
    UNIQUE(kind, key)
);
CREATE TABLE IF NOT EXISTS edges (
    id         INTEGER PRIMARY KEY,
    src        INT,
    dst        INT,
    kind       TEXT,
    props_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_edges_src_kind ON edges(src, kind);
CREATE INDEX IF NOT EXISTS idx_edges_dst_kind ON edges(dst, kind);
"""


def _row_to_node(row: sqlite3.Row) -> dict:
    return {"id": row["id"], "kind": row["kind"], "key": row["key"],
            "props": json.loads(row["props_json"]) if row["props_json"] else {}}


def _edge_dict(row: sqlite3.Row) -> dict:
    return {"id": row["eid"], "src": row["src"], "dst": row["dst"],
            "kind": row["ekind"],
            "props": json.loads(row["eprops"]) if row["eprops"] else {}}


class GraphLite:
    """Adjacency-list graph over SQLite ``nodes``/``edges`` tables.

    ``db_path`` may be a path (str/Path) or an already-open
    ``sqlite3.Connection`` (used by tests and by ``project_from_ledger``
    when handed a connection). Single-writer assumption, WAL mode.
    """

    def __init__(self, db_path: Union[str, Path, sqlite3.Connection]):
        if isinstance(db_path, sqlite3.Connection):
            self._conn = db_path
            self._owns_conn = False
        else:
            self._conn = sqlite3.connect(str(db_path))
            self._owns_conn = True
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self.ensure_schema()

    # ------------------------------------------------------------ schema --

    def ensure_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        if self._owns_conn:
            self._conn.close()

    # ------------------------------------------------------------ writes --

    def upsert_node(self, kind: str, key: str, props: Optional[dict] = None) -> int:
        """Insert or update a node; (kind, key) is the identity.

        A conflicting upsert with ``props=None`` keeps the existing props;
        with props given, they replace the stored ones. Returns the node id
        (stable across upserts).
        """
        props_json = json.dumps(props) if props is not None else None
        row = self._conn.execute(
            "INSERT INTO nodes(kind, key, props_json) VALUES (?, ?, ?) "
            "ON CONFLICT(kind, key) DO UPDATE SET "
            "  props_json = COALESCE(excluded.props_json, nodes.props_json) "
            "RETURNING id",
            (kind, key, props_json),
        ).fetchone()
        self._conn.commit()
        return int(row["id"])

    def add_edge(self, src_id: int, dst_id: int, kind: str,
                 props: Optional[dict] = None) -> int:
        props_json = json.dumps(props) if props is not None else None
        cur = self._conn.execute(
            "INSERT INTO edges(src, dst, kind, props_json) VALUES (?, ?, ?, ?)",
            (src_id, dst_id, kind, props_json),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    # ------------------------------------------------------------- reads --

    def get_node(self, kind: str, key: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT id, kind, key, props_json FROM nodes WHERE kind=? AND key=?",
            (kind, key)).fetchone()
        return _row_to_node(row) if row else None

    def neighbors(self, node_id: int, *,
                  edge_kinds: Optional[Iterable[str]] = None,
                  direction: str = "out") -> list[tuple[dict, dict]]:
        """Adjacent (node, edge) pairs via one indexed SQL lookup per direction."""
        if direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {_DIRECTIONS}")
        kinds = list(edge_kinds) if edge_kinds else None
        kind_clause, kind_params = "", []
        if kinds:
            kind_clause = f" AND e.kind IN ({','.join('?' * len(kinds))})"
            kind_params = kinds

        out: list[tuple[dict, dict]] = []
        if direction in ("out", "both"):
            rows = self._conn.execute(
                "SELECT n.id, n.kind, n.key, n.props_json, "
                "       e.id AS eid, e.src, e.dst, e.kind AS ekind, "
                "       e.props_json AS eprops "
                "FROM edges e JOIN nodes n ON n.id = e.dst "
                f"WHERE e.src = ?{kind_clause} ORDER BY e.id",
                [node_id, *kind_params]).fetchall()
            out.extend((_row_to_node(r), _edge_dict(r)) for r in rows)
        if direction in ("in", "both"):
            rows = self._conn.execute(
                "SELECT n.id, n.kind, n.key, n.props_json, "
                "       e.id AS eid, e.src, e.dst, e.kind AS ekind, "
                "       e.props_json AS eprops "
                "FROM edges e JOIN nodes n ON n.id = e.src "
                f"WHERE e.dst = ?{kind_clause} ORDER BY e.id",
                [node_id, *kind_params]).fetchall()
            out.extend((_row_to_node(r), _edge_dict(r)) for r in rows)
        return out

    def k_hop(self, node_id: int, k: int,
              edge_kinds: Optional[Iterable[str]] = None,
              direction: str = "both") -> dict[int, int]:
        """BFS out to ``k`` hops; returns {node_id: hop_count}, start at 0.

        Cycle-safe: a visited set guarantees termination and that each node
        keeps its MINIMUM hop distance. Frontier expansion is one indexed
        adjacency query per node — the full graph is never materialised.
        """
        kinds = list(edge_kinds) if edge_kinds else None
        visited: dict[int, int] = {node_id: 0}
        queue: deque[int] = deque([node_id])
        while queue:
            current = queue.popleft()
            hop = visited[current]
            if hop >= k:
                continue
            for node, _edge in self.neighbors(current, edge_kinds=kinds,
                                              direction=direction):
                if node["id"] not in visited:
                    visited[node["id"]] = hop + 1
                    queue.append(node["id"])
        return visited

    def ancestry(self, node_id: int, edge_kinds: Iterable[str],
                 max_depth: int = 10) -> list[dict]:
        """Directed walk along OUT edges of the given kinds.

        Returns the path as a list of node dicts starting at ``node_id``.
        At each step the lowest-edge-id match is followed (deterministic);
        the walk stops at ``max_depth``, at a dead end, or on a cycle
        (visited set — never loops forever).
        """
        kinds = list(edge_kinds)
        start = self._conn.execute(
            "SELECT id, kind, key, props_json FROM nodes WHERE id=?",
            (node_id,)).fetchone()
        if start is None:
            return []
        path = [_row_to_node(start)]
        seen = {node_id}
        current = node_id
        for _ in range(max_depth):
            nxt = self.neighbors(current, edge_kinds=kinds, direction="out")
            nxt = [(n, e) for n, e in nxt if n["id"] not in seen]
            if not nxt:
                break
            node, _edge = min(nxt, key=lambda pair: pair[1]["id"])
            path.append(node)
            seen.add(node["id"])
            current = node["id"]
        return path

    # ------------------------------------------------------------- stats --

    def counts(self) -> dict:
        nodes = {r["kind"]: r["c"] for r in self._conn.execute(
            "SELECT kind, COUNT(*) AS c FROM nodes GROUP BY kind ORDER BY kind")}
        edges = {r["kind"]: r["c"] for r in self._conn.execute(
            "SELECT kind, COUNT(*) AS c FROM edges GROUP BY kind ORDER BY kind")}
        return {"nodes": nodes, "edges": edges}


# ------------------------------------------------------------ projection --

_CLOSED_STATUSES = ("closed_turn", "closed_final")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,)).fetchone() is not None


def project_from_ledger(conn_or_path: Union[str, Path, sqlite3.Connection]) -> dict:
    """Idempotently (re)build the graph projection from the ledger.

    Reads ``decisions`` (all rows) and ``outcomes`` (closed only) from the
    same SQLite file, DELETES the projection tables, and rebuilds: one turn
    node per route_id, plus session/model/rule/tripwire nodes and the five
    typed edges. Projections are disposable — the ledger is the truth.

    Returns a stats dict: node/edge counts by kind plus rows consumed.
    Missing ledger tables are not an error (empty projection, zero stats).
    """
    graph = GraphLite(conn_or_path)
    conn = graph._conn

    conn.execute("DELETE FROM edges")
    conn.execute("DELETE FROM nodes")
    conn.commit()

    decisions_seen = 0
    outcomes_seen = 0

    if _table_exists(conn, "decisions"):
        # One pass over decisions: turn/session/model/rule nodes + edges.
        for d in conn.execute(
                "SELECT route_id, ts, session_id, turn_index, source, layer, "
                "       rung, features_json FROM decisions ORDER BY ts"
                ).fetchall():
            decisions_seen += 1
            turn_id = graph.upsert_node(NODE_TURN, d["route_id"], {
                "ts": d["ts"], "turn_index": d["turn_index"],
                "source": d["source"], "layer": d["layer"], "rung": d["rung"],
            })
            session_id = graph.upsert_node(NODE_SESSION, d["session_id"])
            graph.add_edge(turn_id, session_id, EDGE_IN_SESSION)

            model_id = graph.upsert_node(NODE_MODEL, d["rung"])
            graph.add_edge(turn_id, model_id, EDGE_ROUTED_TO,
                           {"layer": d["layer"]})

            # decided_by: fired rules from features_json; the deciding layer
            # itself is the rule when no explicit rules fired (gates, middle).
            fired = _parse_features(d["features_json"]).get("fired_rules") or []
            rule_keys = list(fired) if fired else [d["layer"]]
            for rule_key in rule_keys:
                rule_id = graph.upsert_node(NODE_RULE, str(rule_key))
                graph.add_edge(turn_id, rule_id, EDGE_DECIDED_BY,
                               {"layer": d["layer"]})

    if _table_exists(conn, "outcomes") and _table_exists(conn, "decisions"):
        # Closed outcomes only: escalated_to + tripped edges.
        placeholders = ",".join("?" * len(_CLOSED_STATUSES))
        for o in conn.execute(
                f"SELECT o.route_id, o.escalated, o.tripwire_name, "
                f"       o.tripwire_type, d.rung "
                f"FROM outcomes o JOIN decisions d ON d.route_id = o.route_id "
                f"WHERE o.status IN ({placeholders})",
                _CLOSED_STATUSES).fetchall():
            outcomes_seen += 1
            turn = graph.get_node(NODE_TURN, o["route_id"])
            if turn is None:  # defensive; join makes this unreachable
                continue
            if o["escalated"]:
                target = ESCALATION_TARGET.get(o["rung"])
                if target:  # top rung / unknown rung: nowhere up to point
                    model_id = graph.upsert_node(NODE_MODEL, target)
                    graph.add_edge(turn["id"], model_id, EDGE_ESCALATED_TO,
                                   {"from_rung": o["rung"]})
            if o["tripwire_name"]:
                trip_id = graph.upsert_node(
                    NODE_TRIPWIRE, o["tripwire_name"],
                    {"type": o["tripwire_type"]})
                graph.add_edge(turn["id"], trip_id, EDGE_TRIPPED,
                               {"type": o["tripwire_type"]})

    conn.commit()
    stats = graph.counts()
    stats["decisions_projected"] = decisions_seen
    stats["closed_outcomes_projected"] = outcomes_seen
    if not isinstance(conn_or_path, sqlite3.Connection):
        graph.close()
    return stats


def _parse_features(features_json: Optional[str]) -> dict:
    if not features_json:
        return {}
    try:
        parsed = json.loads(features_json)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


# ---------------------------------------------------------------- report --

def build(stats: dict) -> str:
    """Markdown report of node/edge counts by kind. Counts only, no payloads."""
    lines = ["# Graph-lite projection report", ""]
    lines.append(f"- decisions projected: {stats.get('decisions_projected', 0)}")
    lines.append(f"- closed outcomes projected: "
                 f"{stats.get('closed_outcomes_projected', 0)}")
    lines += ["", "## Nodes by kind", "", "| kind | count |", "|---|---|"]
    node_counts = stats.get("nodes", {})
    for kind in NODE_KINDS:
        lines.append(f"| {kind} | {node_counts.get(kind, 0)} |")
    lines += ["", "## Edges by kind", "", "| kind | count |", "|---|---|"]
    edge_counts = stats.get("edges", {})
    for kind in EDGE_KINDS:
        lines.append(f"| {kind} | {edge_counts.get(kind, 0)} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/router-memory.db"))
    parser.add_argument("--report", action="store_true",
                        help="rebuild the projection from the ledger and "
                             "print node/edge counts by kind")
    args = parser.parse_args()

    if not args.report:
        parser.print_help()
        return 0
    if not args.db.exists():
        print(f"no ledger database at {args.db} — nothing to project",
              file=sys.stderr)
        return 1
    stats = project_from_ledger(args.db)
    print(build(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
