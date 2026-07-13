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


@dataclass(frozen=True)
class TurnEvidence:
    """Canonical cumulative evidence for one live user turn."""

    strikes: dict[str, int]
    edit_failures: int = 0
    error_results: int = 0
    output_tokens: int = 0
    actions: int = 0
    continuation_count: int = 0
    infrastructure_failures: int = 0
    tripwire_name: Optional[str] = None
    tripwire_type: Optional[str] = None


class EscalationController:
    """Per-session live trip-wire evaluation + sticky route escalation."""

    def __init__(self, store: Optional[SessionStore] = None):
        self.store = store or DictSessionStore()
        self._tw: dict[str, TripwireState] = {}
        self._infra_failures: dict[str, int] = {}
        # Response-side detectors fire after a streamed response. Preserve the
        # handoff so the next request can be rebuilt before crossing providers.
        self._pending: dict[str, EscalationDecision] = {}

    def _tw_for(self, session_id: str) -> TripwireState:
        tw = self._tw.get(session_id)
        if tw is None:
            tw = TripwireState(budget=None)
            self._tw[session_id] = tw
        return tw

    def new_turn(self, session_id: str) -> None:
        """Reset per-turn trip-wire strikes at a new user turn (§5.5)."""
        self._tw[session_id] = TripwireState(budget=None)
        self._infra_failures[session_id] = 0
        try:
            self.store.reset_strikes(session_id)
            self.store.reset_continuations(session_id)
        except Exception:
            pass

    def consume_pending(self, session_id: str) -> Optional[EscalationDecision]:
        """Consume one pending cross-rung handoff, if a detector fired."""
        return self._pending.pop(session_id, None)

    def sync_served_route(self, session_id: str, rung: str) -> None:
        """Persist the rung that actually served after transport fallback."""
        try:
            self.store.set_route(session_id, rung)
        except Exception:
            pass

    def mark_turn_clean(self, session_id: str, clean: bool) -> None:
        """Publish the previous-turn signal used by episode-boundary detection."""
        try:
            self.store.mark_turn_clean(session_id, clean)
        except Exception:
            pass

    def note_continuation(self, session_id: str) -> int:
        """Count one tool-result-bearing request in the active user turn."""
        try:
            return self.store.incr_continuation(session_id)
        except Exception:
            return 0

    def note_infrastructure_failure(self, session_id: str) -> int:
        """Record a within-rung transport/serving failure for this turn."""
        count = self._infra_failures.get(session_id, 0) + 1
        self._infra_failures[session_id] = count
        return count

    def snapshot(self, session_id: str) -> TurnEvidence:
        """Return one canonical view used by live outcome capture."""
        try:
            tw = self._tw_for(session_id)
            counts = tw.evidence
            hit = tw.fired()
            session = self.store.get_session(session_id)
            return TurnEvidence(
                strikes=dict(tw.strikes),
                edit_failures=int(counts.get("edit_failures", 0)),
                error_results=int(counts.get("error_results", 0)),
                output_tokens=int(counts.get("output_tokens", 0)),
                actions=int(counts.get("actions", 0)),
                continuation_count=int(getattr(session, "continuation_count", 0) or 0),
                infrastructure_failures=self._infra_failures.get(session_id, 0),
                tripwire_name=hit.name if hit else None,
                tripwire_type=hit.type.value if hit else None,
            )
        except Exception:
            return TurnEvidence(strikes={})

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
            already_fired = tw.fired() is not None
            hit: Optional[TripwireHit] = feed(tw)
            # Mirror the FULL evaluator snapshot on every observation. Partial
            # strikes matter to outcomes even before a threshold fires.
            try:
                self.store.set_strikes(session_id, tw.strikes)
            except Exception:
                pass
        except Exception:
            # fail-safe: never escalate on an observation error
            return EscalationDecision(False, cur, None, None, None,
                                      reason="observe error — route unchanged")
        if already_fired:
            # The evaluator latches its first hit for outcome provenance. A
            # later response/result observation returns that same hit, but must
            # not advance the escalation ladder a second time.
            return EscalationDecision(False, cur, None, None, None)
        if hit is None:
            return EscalationDecision(False, cur, None, None, None)

        # privacy pin blocks cloud escalation (§5.3): surface to user instead.
        try:
            pinned = self.store.get_session(session_id).privacy_pinned
        except Exception:
            pinned = False
        if pinned:
            decision = EscalationDecision(
                True, cur, None, hit.name, hit.type.value, to_user=True,
                reason=f"{hit.name} fired but session is privacy-pinned — surface to user")
            self._pending[session_id] = decision
            return decision

        target = NEXT_RUNG.get(cur)
        if target is None:
            return EscalationDecision(
                True, cur, None, hit.name, hit.type.value,
                reason=f"{hit.name} fired at top rung {cur} — nowhere to escalate")

        # flip the sticky route (hysteresis: stays escalated)
        try:
            self.store.set_route(session_id, target)
            self.store.mark_escalated(session_id)
        except Exception:
            pass
        decision = EscalationDecision(
            True, cur, target, hit.name, hit.type.value,
            reason=f"{hit.name} ({hit.type}) -> escalate {cur} to {target}"
                   + (" [dialect: trains registry]" if routes_to_registry(hit.type) else ""))
        self._pending[session_id] = decision
        return decision
