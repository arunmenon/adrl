"""Insights layer — the router's memory explaining itself.

Reads the transaction memory (decisions + outcomes + the graph projection) and
emits a ranked list of human-readable findings: where routing decisions come
from, what forces the expensive rungs, which hand-rules actually matter, and
where outcomes contradict the rules that made them. This is the read-side that
WS3 (heuristic health) and WS5 (self-healing adjustments) act on — surfaced as
plain findings a human (or a later job) can act on.

Pure analytics over data/router-memory.db; never writes to the ledger. Follows
house convention: argparse(description=__doc__), a pure build()->markdown
reporter, and a --json mode for the graph artifact.

Usage:
  PYTHONPATH=src .venv/bin/python -m router.insights            # markdown report
  PYTHONPATH=src .venv/bin/python -m router.insights --json     # machine-readable
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_DB = Path("data/router-memory.db")
LOCAL_CONTEXT_CEILING = 26_214  # 0.8 * 32768 — the feasibility headroom (policy §5.3)


@dataclass
class Insight:
    """One ranked finding derived from the memory."""

    kind: str          # economics | context | privacy | heuristic | rule_health | shape | gap
    title: str
    finding: str       # the one-line headline
    detail: str        # the why / so-what
    magnitude: float   # 0..1 importance, for ranking + bar length
    stat: str          # the headline number, pre-formatted
    action: str        # the suggested lever
    evidence: dict = field(default_factory=dict)


def _rows(conn, q, *a):
    return conn.execute(q, a).fetchall()


def _pct(n, d):
    return (100.0 * n / d) if d else 0.0


def _feature(features_json: str, key: str, default=None):
    try:
        return json.loads(features_json).get(key, default)
    except (json.JSONDecodeError, TypeError):
        return default


def generate(db_path: Path = DEFAULT_DB) -> list[Insight]:
    """Compute all insights over the memory, ranked most-important first."""
    if not Path(db_path).is_file():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        return _generate(conn)
    finally:
        conn.close()


def _generate(conn) -> list[Insight]:
    total = _rows(conn, "SELECT COUNT(*) FROM decisions")[0][0]
    if not total:
        return []
    out: list[Insight] = []

    by_layer = dict(_rows(conn, "SELECT layer, COUNT(*) FROM decisions GROUP BY layer"))
    by_rung = dict(_rows(conn, "SELECT rung, COUNT(*) FROM decisions GROUP BY rung"))
    gate_total = sum(v for k, v in by_layer.items() if k.startswith("gate:"))
    heur_total = sum(v for k, v in by_layer.items() if k.startswith("heuristic"))
    middle = by_layer.get("middle_default", 0)

    # 1 — who actually decides: gates vs the "intelligent" layers
    out.append(Insight(
        kind="economics",
        title="Gates do the routing, not the classifier",
        finding=f"{_pct(gate_total, total):.0f}% of decisions were hard gates.",
        detail=(f"Hard gates (privacy, feasibility, context-pin) decided {gate_total} of "
                f"{total} turns. The classifier/retrieval layer only ever reaches the "
                f"ambiguous middle — {middle} turns ({_pct(middle, total):.0f}%). Effort on the "
                f"classifier caps out at that slice; the biggest levers are the gates themselves."),
        magnitude=_pct(gate_total, total) / 100,
        stat=f"{_pct(gate_total, total):.0f}% gates",
        action="Invest in gate accuracy (context-fit + secret detection) before the classifier.",
        evidence={"gate": gate_total, "middle": middle, "total": total,
                  "by_layer": by_layer},
    ))

    # 2 — the hand heuristics are nearly dead weight (validates the scaling fear)
    out.append(Insight(
        kind="heuristic",
        title="The regex lexicon barely fires",
        finding=f"The hand-tuned verb heuristics decided just {heur_total} of {total} turns "
                f"({_pct(heur_total, total):.1f}%).",
        detail=("The accreting hand-rules that were feared not to scale are, on this corpus, "
                "almost never the decider — gates catch the turns first, and the rest fall to "
                "the middle. Confirms the concern with data: the lexicon is low-leverage."),
        magnitude=0.9,
        stat=f"{heur_total} turns ({_pct(heur_total, total):.1f}%)",
        action="Stop growing the lexicon; route the middle via the retrieval router (WS4).",
        evidence={"heuristic": heur_total,
                  "rules": {k: v for k, v in by_layer.items() if k.startswith("heuristic")}},
    ))

    # 3 — what FORCES the expensive rungs: difficulty or physics?
    frontier = by_rung.get("frontier", 0)
    forced = _rows(conn,
                   "SELECT COUNT(*) FROM decisions WHERE rung='frontier' AND layer LIKE 'gate:%'")[0][0]
    if frontier:
        out.append(Insight(
            kind="context",
            title="Frontier routing is forced by context, not difficulty",
            finding=f"{_pct(forced, frontier):.0f}% of frontier turns were sent there by a gate, "
                    f"not because the task was hard.",
            detail=(f"{forced} of {frontier} frontier-routed turns hit a hard gate "
                    f"(the context didn't fit local, or the session was pinned). The frontier "
                    f"model is doing expensive work that a bigger local context window would reclaim "
                    f"— the bottleneck is local capacity, not model intelligence."),
            magnitude=_pct(forced, frontier) / 100,
            stat=f"{_pct(forced, frontier):.0f}% forced",
            action="A larger-context local rung (office M4 Max / bigger model) reclaims these turns.",
            evidence={"frontier": frontier, "forced_by_gate": forced},
        ))

    # 4 — privacy footprint
    pinned = by_layer.get("gate:privacy", 0)
    no_embed = _rows(conn,
                     "SELECT COUNT(*) FROM decisions WHERE instr_sha256 IS NULL")[0][0]
    out.append(Insight(
        kind="privacy",
        title="Secrets dominate the corpus, thinning the memory",
        finding=f"{_pct(no_embed, total):.0f}% of turns were privacy-excluded (no hash, no embedding).",
        detail=(f"{no_embed} of {total} turns live in secret-flagged sessions, so they carry no "
                f"embedding — the retrieval memory (WS4) sees only {total - no_embed} turns, not "
                f"{total}. Privacy is both a dominant router (gate:privacy fired {pinned}×) and a "
                f"hard limit on how much the kNN can learn from history."),
        magnitude=_pct(no_embed, total) / 100,
        stat=f"{_pct(no_embed, total):.0f}% excluded",
        action="Lean on simulator traffic (non-secret) to fill the retrieval memory.",
        evidence={"excluded": no_embed, "gate_privacy": pinned},
    ))

    # 5 — rule health seed (WS3): does middle_default->local hold up against outcomes?
    md_rows = _rows(conn,
                    "SELECT d.rung, o.outcome_proxy_hard FROM decisions d "
                    "JOIN outcomes o ON o.route_id=d.route_id "
                    "WHERE d.layer='middle_default' AND o.status LIKE 'closed%'")
    if md_rows:
        md_local = [r for r in md_rows if r[0] == "local"]
        md_hard = sum(1 for r in md_local if r[1] == 1)
        contradiction = _pct(md_hard, len(md_local)) if md_local else 0.0
        out.append(Insight(
            kind="rule_health",
            title="The middle_default->local guess, audited",
            finding=f"{contradiction:.0f}% of middle turns kept local actually went hard.",
            detail=(f"Of {len(md_local)} turns the middle_default rule sent to local, {md_hard} "
                    f"showed a hard outcome proxy (edit-fail / errors / interrupt / long loop). "
                    f"That is the exact slice the retrieval router should reclaim — where the "
                    f"coin-flip middle guessed cheap but the turn was not."),
            magnitude=min(1.0, contradiction / 100 + 0.3),
            stat=f"{contradiction:.0f}% contradicted",
            action="Target this slice first when grading the retrieval router (WS4).",
            evidence={"middle_local": len(md_local), "went_hard": md_hard},
        ))

    # 6 — session shape (power law)
    sess = _rows(conn, "SELECT session_id, COUNT(*) c FROM decisions GROUP BY session_id ORDER BY c DESC")
    if sess:
        top = sess[0][1]
        top3 = sum(c for _, c in sess[:3])
        out.append(Insight(
            kind="shape",
            title="A few sessions hold most of the traffic",
            finding=f"The top 3 of {len(sess)} sessions hold {_pct(top3, total):.0f}% of all turns.",
            detail=(f"Session lengths are heavy-tailed (longest {top} turns). Per-session routing "
                    f"state and stickiness matter disproportionately — a wrong sticky route on a "
                    f"long session costs many turns."),
            magnitude=0.5,
            stat=f"{_pct(top3, total):.0f}% in top 3",
            action="Weight evaluation by session length, not turn count.",
            evidence={"sessions": len(sess), "longest": top},
        ))

    # 7 — gap: no live escalations yet
    escalated = _rows(conn, "SELECT COUNT(*) FROM outcomes WHERE escalated=1")[0][0]
    if escalated == 0:
        out.append(Insight(
            kind="gap",
            title="No escalations recorded yet",
            finding="0 turns show a live escalation — the trip-wire loop is untested on real routing.",
            detail=("The memory was seeded by a heuristic-only backfill with no live model calls, so "
                    "no trip-wire has fired against a real cheap-model attempt. The escalated_to / "
                    "tripped graph edges are empty until routing goes live (WS2)."),
            magnitude=0.8,
            stat="0 escalations",
            action="WS2: route sim traffic live so the escalation loop produces real edges.",
            evidence={"escalated": 0},
        ))

    out.sort(key=lambda i: i.magnitude, reverse=True)
    return out


def build(db_path: Path = DEFAULT_DB) -> str:
    insights = generate(db_path)
    if not insights:
        return "# Router insights\n\nNo memory found (run the backfill first).\n"
    lines = ["# Router insights", "",
             f"Ranked findings derived from `{db_path}` — the memory explaining itself.", ""]
    for i, ins in enumerate(insights, 1):
        lines += [f"## {i}. {ins.title}  ·  `{ins.stat}`",
                  f"**{ins.finding}**", "", ins.detail, "",
                  f"→ *{ins.action}*", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--report", type=Path, default=None, help="also write markdown here")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    if args.json:
        print(json.dumps([asdict(i) for i in generate(args.db)], indent=2))
        return 0
    md = build(args.db)
    print(md)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
