"""B7 — the three-layer policy engine (design doc §5.3, v2.1 semantics).

Layer 0: hard gates — privacy pin (trip-wires stay armed, escalation target is
         the USER), pin+context conflict surfaced not silently mis-routed,
         health filter, context feasibility -> cheapest feasible rung,
         escalation hysteresis.
Layer 1: heuristics — decides the clear-easy and clear-hard ends.
Layer 2: learned router — NOT IMPLEMENTED (Phase 3, gated on beating the
         best-single-model baseline). The ambiguous middle currently routes
         local-with-cascade and is labeled `middle_default` so shadow reports
         can quantify exactly how much traffic Phase 3 would actually decide.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .features import TurnFeatures, heuristic_score

T_EASY = 0.35
T_HARD = 0.70

# Capability registry sketch (§5.4) — measured numbers replace these as the
# eval pack runs. Health flags are fed by LiteLLM checks in the live system.
REGISTRY: dict[str, dict] = {
    "local": {"max_context": 32_768, "healthy": True, "cost_rank": 0},
    "cheap_cloud": {"max_context": 200_000, "healthy": True, "cost_rank": 1},
    "frontier": {"max_context": 200_000, "healthy": True, "cost_rank": 2},
}
CONTEXT_HEADROOM = 0.8  # a rung must fit the turn with 20% headroom


@dataclass
class SessionState:
    """The minimum viable per-session memory (§5.6)."""

    session_id: str
    route: str = "local"
    privacy_pinned: bool = False
    escalated_this_episode: bool = False
    turn_count: int = 0
    strikes: dict = field(default_factory=dict)


@dataclass
class Route:
    rung: str
    cascade: bool
    layer: str            # which layer decided: gate:* | heuristic | middle_default
    score: float | None = None
    pinned: bool = False
    conflict: str | None = None   # e.g. "pin_context" -> surface to user (§5.8)
    reason: str = ""


def _healthy(registry: dict) -> dict[str, dict]:
    return {k: v for k, v in registry.items() if v.get("healthy", True)}


def _cheapest_feasible(rungs: dict[str, dict], context_tokens: int) -> str:
    fits = [
        (v["cost_rank"], k) for k, v in rungs.items()
        if context_tokens <= CONTEXT_HEADROOM * v["max_context"]
    ]
    if not fits:
        return "frontier"  # nothing fits with headroom — biggest window, no cascade
    return min(fits)[1]


def route_turn(f: TurnFeatures, session: SessionState,
               registry: dict[str, dict] = REGISTRY) -> Route:
    # ── Layer 0: HARD GATES — never overridden ──────────────────────────────
    if session.privacy_pinned or f.privacy_pinned:
        if f.context_tokens > CONTEXT_HEADROOM * registry["local"]["max_context"]:
            return Route("local", cascade=False, layer="gate:pin_context_conflict",
                         pinned=True, conflict="pin_context",
                         reason="pinned session outgrew local context — surface to user (§5.8)")
        return Route("local", cascade=True, layer="gate:privacy", pinned=True,
                     reason="privacy pin: trip-wires armed, escalation target = user")

    rungs = _healthy(registry)
    if "local" not in rungs or f.context_tokens > CONTEXT_HEADROOM * rungs.get(
            "local", {"max_context": 0})["max_context"]:
        rung = _cheapest_feasible(rungs, f.context_tokens)
        return Route(rung, cascade=False, layer="gate:feasibility",
                     reason=f"local unavailable/doesn't fit {f.context_tokens} tokens "
                            f"-> cheapest feasible = {rung}")

    if session.escalated_this_episode or f.escalated_this_episode:
        return Route(session.route if session.route != "local" else "frontier",
                     cascade=False, layer="gate:hysteresis",
                     reason="episode already escalated — stay up until boundary")

    # ── Layer 1: HEURISTICS — the clear ends ────────────────────────────────
    s = heuristic_score(f)
    if f.prev_turn_interrupted:
        # escalate-on-retry outranks difficulty, including "this looks easy" (§5.5/S6)
        return Route("frontier", cascade=False, layer="heuristic:retry_signal",
                     score=s, reason="user interrupted/rephrased — believe them")
    if s < T_EASY:
        return Route("local", cascade=True, layer="heuristic", score=s,
                     reason=f"easy ({f.verb_class}, {s:.2f} < {T_EASY})")
    if s > T_HARD:
        return Route("frontier", cascade=False, layer="heuristic", score=s,
                     reason=f"hard ({f.verb_class}, {s:.2f} > {T_HARD}) — don't burn a doomed local try")

    # ── Layer 2: LEARNED ROUTER — Phase 3 stub ──────────────────────────────
    return Route("local", cascade=True, layer="middle_default", score=s,
                 reason=f"ambiguous middle ({s:.2f}) — local w/ trip-wires until Phase 3 earns its keep")
