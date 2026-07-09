"""MemoryProvider port + event dataclasses — the WS1 memory contract (Engram-shaped).

The router's transaction memory is an append-only ledger on a ``route_id``
spine: a ``DecisionEvent`` is written once at decision time, an
``OutcomeEvent`` arrives late (lifecycle pending -> closed_turn ->
closed_final), and instruction embeddings have their own cadence. Providers
(SQLite Engram-lite, full Engram, Null) all implement the ``MemoryProvider``
port below; the router only ever talks to the facade (memory_facade.py),
which dispatches down a provider chain.

FAIL-SAFE contract: providers never raise from these methods; degraded
answers are None / False / [] / 0 / {}.

Import-only module — no CLI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

# The mandatory nomic-embed-text task prefixes — part of the shared contract
# (single home post-review; the facade and providers import from here).
DOCUMENT_PREFIX = "search_document: "   # when storing ledger embeddings
QUERY_PREFIX = "search_query: "         # when embedding a lookup query


@dataclass
class DecisionEvent:
    """Write-once fact recorded at decision time (immutable ledger entry).

    Privacy: ``instr_sha256`` is None for privacy-pinned/secret turns, and the
    raw instruction text is NEVER part of this event (``features_json`` is
    scrubbed of it by the facade before the event is built).
    """

    route_id: str = ""
    ts: float = 0.0
    session_id: str = ""
    turn_index: int = 0
    source: str = "organic"              # 'organic' | 'simulator'
    instr_sha256: Optional[str] = None   # None when privacy-pinned/secret
    features_json: str = "{}"            # TurnFeatures incl. fired rules, text-scrubbed
    layer: str = ""                      # which layer decided (Route.layer)
    rung: str = ""                       # Route.rung
    cascade: bool = False
    score: Optional[float] = None
    reason: str = ""
    classifier_tier: Optional[str] = None
    propensity: str = ""                 # which layer decided (selection-bias bookkeeping)
    policy_version: str = ""
    classifier_ms: float = 0.0
    decision_ms: float = 0.0


@dataclass
class OutcomeEvent:
    """Follow-up event keyed by route_id; lifecycle pending -> closed_turn ->
    closed_final (turn N+1's interrupt/retry signal finalizes turn N)."""

    status: str = "pending"              # 'pending' | 'closed_turn' | 'closed_final'
    escalated: bool = False
    tripwire_name: Optional[str] = None
    tripwire_type: Optional[str] = None
    edit_failures: int = 0
    error_results: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_estimate: float = 0.0
    interrupted: bool = False
    user_retried: Optional[bool] = None      # known only at closed_final
    outcome_proxy_hard: Optional[bool] = None  # router.outcomes.outcome_proxy_hard


@dataclass
class NeighborTurn:
    """One kNN neighbor: a past decision + its (possibly still-open) outcome."""

    route_id: str = ""
    similarity: float = 0.0
    rung: str = ""
    escalated: bool = False
    outcome_proxy_hard: Optional[bool] = None
    source: str = ""
    ts: float = 0.0


class MemoryProvider(ABC):
    """Storage/retrieval mechanics ONLY — privacy gating, fail-safety wrapping
    and provider selection are the facade's job. Implementations must still
    honor the fail-safe contract themselves (never raise; degrade to
    None / False / [] / 0 / {})."""

    @abstractmethod
    def record_decision(self, decision: DecisionEvent,
                        embedding: Optional[list[float]] = None) -> Optional[str]:
        """Persist a decision (and optional embedding). Return the route_id on
        success, None on any failure."""

    @abstractmethod
    def attach_outcome(self, route_id: str, outcome: OutcomeEvent) -> bool:
        """Attach/update the outcome for a decision. False on any failure."""

    @abstractmethod
    def finalize_turn(self, session_id: str, *, prev_interrupted: bool,
                      prev_retried: bool) -> int:
        """Close the session's previous open turn (pending/closed_turn ->
        closed_final) using the NEXT turn's interrupt/retry signals.
        Return the number of rows finalized (0 on none/failure)."""

    @abstractmethod
    def similar_turns(self, embedding: list[float], k: int = 12) -> list[NeighborTurn]:
        """kNN over stored embeddings. [] when none / no data / failure."""

    @abstractmethod
    def stats(self) -> dict:
        """Provider-shaped counters (rows, sources, lifecycle coverage). {} on failure."""

    @abstractmethod
    def health(self) -> bool:
        """True when the provider can currently serve reads and writes."""
