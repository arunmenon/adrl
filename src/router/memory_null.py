"""NullProvider — the terminal memory provider (WS1).

Always last in the facade's chain, so every memory call ALWAYS succeeds:
writes are accepted into the void, reads answer "no data". This makes
"no memory configured" a first-class, tested state — the router keeps
routing, it just stops learning.

Import-only module — no CLI.
"""

from __future__ import annotations

from typing import Optional

from .memory_ports import DecisionEvent, MemoryProvider, NeighborTurn, OutcomeEvent


class NullProvider(MemoryProvider):
    """Accepts writes into the void; answers reads with "no data"; never raises."""

    def record_decision(self, decision: DecisionEvent,
                        embedding: Optional[list[float]] = None) -> Optional[str]:
        try:
            route_id = getattr(decision, "route_id", None)
        except Exception:
            return None
        return route_id if isinstance(route_id, str) and route_id else None

    def attach_outcome(self, route_id: str, outcome: OutcomeEvent) -> bool:
        return True

    def finalize_turn(self, session_id: str, *, prev_interrupted: bool = False,
                      prev_retried: bool = False) -> int:
        return 0

    def similar_turns(self, embedding: list[float], k: int = 12) -> list[NeighborTurn]:
        return []

    def stats(self) -> dict:
        return {"provider": "null"}

    def health(self) -> bool:
        return True
