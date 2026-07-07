"""P1-D — the live escalation controller (the post-call path, wired for real time).

Turns the shadow post-call machinery into a per-session live decision: as
responses stream back from the local rung, feed them to the session's
trip-wire state; when a wire fires, escalate — flip the session's sticky route
to the next rung so subsequent requests in the failing turn/episode go to
cloud. This is the SEMANTIC escalation ladder (§5.5), distinct from LiteLLM's
infra fallback (§8.3): it fires on bad-but-valid local responses that a
transport proxy can't see.

Design choices for the live (per-request) proxy path:
- The proxy sees individual requests, not whole turns. Trip-wire strikes
  accumulate across a session's requests via the SessionStore; escalation flips
  the sticky route rather than rebuilding+reissuing the current request (the
  current request already has its response). The design's transcript rebuild
  (escalate.rebuild_for_escalation) applies when we later synthesize the
  higher-rung request; here we just redirect the next one.
- One-way within an episode (hysteresis): once escalated, stays escalated.
- Privacy pin blocks cloud escalation: a pinned session's escalation target is
  the USER, never a cloud rung (§5.3). The controller surfaces that instead.
- Fail-safe: any error observing a response leaves the route unchanged (never
  escalates spuriously, never crashes the proxy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from router.state import SessionStore, DictSessionStore
from router.tripwires import TripwireState, TripwireHit, routes_to_registry

# Rung ladder for semantic escalation.
NEXT_RUNG = {"local-code": "cheap-cloud", "local-small": "cheap-cloud",
             "cheap-cloud": "frontier"}


@dataclass
class EscalationDecision:
    escalate: bool
    from_rung: str
    to_rung: Optional[str]        # None when target is the user (pinned) or top rung
    tripwire: Optional[str]
    tripwire_type: Optional[str]
    to_user: bool = False         # pinned session — surface to user, don't route cloud
    reason: str = ""


class EscalationController:
    """Per-session live trip-wire evaluation + sticky route escalation."""

    def __init__(self, store: Optional[SessionStore] = None):
        self.store = store or DictSessionStore()
        self._tw: dict[str, TripwireState] = {}

    def _tw_for(self, session_id: str) -> TripwireState:
        tw = self._tw.get(session_id)
        if tw is None:
            tw = TripwireState(budget=None)
            self._tw[session_id] = tw
        return tw

    def new_turn(self, session_id: str) -> None:
        """Reset per-turn trip-wire strikes at a new user turn (§5.5)."""
        self._tw[session_id] = TripwireState(budget=None)
        try:
            self.store.reset_strikes(session_id)
        except Exception:
            pass

    def current_route(self, session_id: str, default: str = "local-code") -> str:
        try:
            return self.store.get_session(session_id).route or default
        except Exception:
            return default

    def observe_response(self, session_id: str, response_blocks: list[dict],
                         *, output_tokens: int = 0) -> EscalationDecision:
        return self._observe(session_id, lambda tw: tw.observe_response(
            response_blocks, output_tokens=output_tokens))

    def observe_tool_results(self, session_id: str,
                             user_content_blocks: list[dict]) -> EscalationDecision:
        return self._observe(session_id, lambda tw: tw.observe_tool_results(
            user_content_blocks))

    def _observe(self, session_id: str, feed) -> EscalationDecision:
        cur = self.current_route(session_id)
        try:
            tw = self._tw_for(session_id)
            hit: Optional[TripwireHit] = feed(tw)
        except Exception:
            # fail-safe: never escalate on an observation error
            return EscalationDecision(False, cur, None, None, None,
                                      reason="observe error — route unchanged")
        if hit is None:
            return EscalationDecision(False, cur, None, None, None)

        # a wire fired — mirror the strike into the store for the record
        try:
            self.store.incr_strike(session_id, hit.name)
        except Exception:
            pass

        # privacy pin blocks cloud escalation (§5.3): surface to user instead.
        try:
            pinned = self.store.get_session(session_id).privacy_pinned
        except Exception:
            pinned = False
        if pinned:
            return EscalationDecision(
                True, cur, None, hit.name, str(hit.type), to_user=True,
                reason=f"{hit.name} fired but session is privacy-pinned — surface to user")

        target = NEXT_RUNG.get(cur)
        if target is None:
            return EscalationDecision(
                True, cur, None, hit.name, str(hit.type),
                reason=f"{hit.name} fired at top rung {cur} — nowhere to escalate")

        # flip the sticky route (hysteresis: stays escalated)
        try:
            self.store.set_route(session_id, target)
            self.store.mark_escalated(session_id)
        except Exception:
            pass
        return EscalationDecision(
            True, cur, target, hit.name, str(hit.type),
            reason=f"{hit.name} ({hit.type}) -> escalate {cur} to {target}"
                   + (" [dialect: trains registry]" if routes_to_registry(hit.type) else ""))
