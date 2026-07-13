"""RouterMemory — the memory facade, the ONLY thing the router ever calls (WS1).

Responsibilities (strict split — providers own storage mechanics only):
  * PRIVACY GATE: ``make_decision_event`` turns raw instruction text +
    TurnFeatures + Route into a DecisionEvent, hashing the text to
    ``instr_sha256`` UNLESS the turn is privacy-pinned/secret-flagged (then the
    sha is None), and decides whether an embedding may be computed (never for
    pinned). Instruction TEXT never reaches a provider — ``features_json`` is
    scrubbed of it here.
  * PROVIDER CHAIN: config-ordered chain-of-responsibility via FallbackChain
    (default: SqliteProvider when importable, then NullProvider — the terminal
    member, so every call ALWAYS succeeds).
  * UNIFORM FAIL-SAFETY: provider errors never escape; degraded answers are
    None / False / [] / 0 / {}.
  * Embedding via the WS0 EmbeddingBackend port (role 'embedder',
    nomic-embed-text) with the MANDATORY nomic prefixes: 'search_document: '
    when storing, 'search_query: ' when querying.

``RoutingRecorder`` is the three-call convenience for WS2: ``new_turn`` (close
the session's previous turn), ``record`` (mint a uuid route_id, build the
gated event, embed if allowed, store), ``attach`` (attach an outcome to the
session's active route_id).

Usage:
  PYTHONPATH=src .venv/bin/python -m router.memory_facade --report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Optional, Sequence

from .chain import FallbackChain
from .memory_null import NullProvider
from .memory_ports import (
    DOCUMENT_PREFIX,   # single home of the mandatory nomic prefixes (post-review)
    QUERY_PREFIX,
    DecisionEvent,
    NeighborTurn,
    OutcomeEvent,
    VerifiedOutcome,
)

DEFAULT_POLICY_VERSION = "v1"
_TRACE_FIELDS = {
    "resolver", "called", "abstained", "tier", "needs_frontier", "score",
    "confidence", "n_neighbors", "candidate_pool", "model_version",
}


def default_providers() -> list:
    """Config-default chain: [SqliteProvider (when importable), NullProvider].

    SqliteProvider (memory_sqlite, built in parallel — Unit B) is imported
    lazily and fail-safe: if the module or its construction is unavailable,
    the chain simply starts at NullProvider.
    """
    providers: list = []
    try:
        from .memory_sqlite import SqliteProvider  # Unit B — may not exist yet
        providers.append(SqliteProvider())
    except Exception:
        pass
    providers.append(NullProvider())
    return providers


class RouterMemory:
    """The facade. Owns privacy gating, the provider chain, and fail-safety."""

    def __init__(self, providers: Optional[Sequence] = None, *,
                 embedder: Any = None,
                 policy_version: str = DEFAULT_POLICY_VERSION):
        self.providers = list(providers) if providers else default_providers()
        self._chain = FallbackChain(self.providers)
        self._embedder = embedder          # WS0 EmbeddingBackend; lazy default
        self.policy_version = policy_version

    # ── privacy gate ─────────────────────────────────────────────────────────

    @staticmethod
    def _is_private(features: Any, route: Any) -> bool:
        """Pinned/secret detection: TurnFeatures.privacy_pinned, Route.pinned,
        or a truthy 'secret' flag in TurnFeatures.extra."""
        if bool(getattr(features, "privacy_pinned", False)):
            return True
        if bool(getattr(route, "pinned", False)):
            return True
        extra = getattr(features, "extra", None)
        return isinstance(extra, dict) and bool(extra.get("secret"))

    @staticmethod
    def _features_json(features: Any) -> str:
        """Serialize TurnFeatures WITHOUT the raw instruction text (privacy:
        text never reaches a provider — only sha, features, embedding)."""
        try:
            data = asdict(features) if is_dataclass(features) else dict(vars(features))
        except Exception:
            data = {}
        data.pop("instruction_text", None)
        try:
            return json.dumps(data, default=str, sort_keys=True)
        except Exception:
            return "{}"

    @staticmethod
    def _trace_json(trace: Optional[dict]) -> str:
        """Serialize only approved resolver metadata, never free-form text."""
        if not isinstance(trace, dict):
            return "{}"
        scrubbed = {key: trace[key] for key in _TRACE_FIELDS if key in trace}
        try:
            return json.dumps(scrubbed, default=str, sort_keys=True)
        except Exception:
            return "{}"

    def make_decision_event(self, instruction_text: str, features: Any, route: Any,
                            *, session_id: str, turn_index: int = 0,
                            source: str = "organic",
                            route_id: Optional[str] = None,
                            classifier_tier: Optional[str] = None,
                            propensity: Optional[str] = None,
                            classifier_ms: float = 0.0,
                            decision_ms: float = 0.0,
                            decision_trace: Optional[dict] = None,
                            ts: Optional[float] = None) -> tuple[DecisionEvent, bool]:
        """The privacy gate. Returns ``(event, embedding_allowed)``.

        Non-private turns: text -> sha256, embedding allowed. Privacy-pinned or
        secret-flagged turns: ``instr_sha256`` is None and the embedding is NOT
        allowed (callers must not compute one; ``record_decision`` drops it
        anyway as a second line of defense).
        """
        private = self._is_private(features, route)
        text = instruction_text or ""
        sha = None if (private or not text) else hashlib.sha256(
            text.encode("utf-8")).hexdigest()
        layer = str(getattr(route, "layer", "") or "")
        score = getattr(route, "score", None)
        event = DecisionEvent(
            route_id=route_id or uuid.uuid4().hex,
            ts=time.time() if ts is None else float(ts),
            session_id=str(session_id),
            turn_index=int(turn_index),
            source=str(source),
            instr_sha256=sha,
            features_json=self._features_json(features),
            layer=layer,
            rung=str(getattr(route, "rung", "") or ""),
            cascade=bool(getattr(route, "cascade", False)),
            score=float(score) if isinstance(score, (int, float)) else None,
            reason=str(getattr(route, "reason", "") or ""),
            classifier_tier=classifier_tier,
            propensity=propensity if propensity is not None else layer,
            policy_version=self.policy_version,
            classifier_ms=float(classifier_ms),
            decision_ms=float(decision_ms),
            trace_json=self._trace_json(decision_trace),
        )
        return event, not private

    # ── embedding (WS0 port, mandatory nomic prefixes) ───────────────────────

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from .backends import for_role
                self._embedder = for_role("embedder")
            except Exception:
                self._embedder = False  # sentinel: construction failed for good
        return self._embedder or None

    def _embed(self, prefix: str, text: str) -> Optional[list[float]]:
        if not text:
            return None
        backend = self._get_embedder()
        if backend is None:
            return None
        try:
            vectors = backend.embed([prefix + text])
        except Exception:
            return None
        if not isinstance(vectors, list) or not vectors:
            return None
        vector = vectors[0]
        return vector if isinstance(vector, list) and vector else None

    def embed_document(self, text: str) -> Optional[list[float]]:
        """Embed instruction text for STORAGE ('search_document: ' prefix).
        Callers must only invoke this when the privacy gate allowed it."""
        return self._embed(DOCUMENT_PREFIX, text)

    def embed_query(self, text: str) -> Optional[list[float]]:
        """Embed instruction text for RETRIEVAL ('search_query: ' prefix)."""
        return self._embed(QUERY_PREFIX, text)

    # ── provider dispatch (uniform fail-safety) ──────────────────────────────

    def record_decision(self, decision: DecisionEvent,
                        embedding: Optional[list[float]] = None) -> Optional[str]:
        # Second line of privacy defense: a sha-less (pinned/secret) decision
        # never stores a vector, whatever the caller passed.
        if getattr(decision, "instr_sha256", None) is None:
            embedding = None
        try:
            result = self._chain.record_decision(decision, embedding)
        except Exception:
            return None
        return result if isinstance(result, str) else None

    def attach_outcome(self, route_id: str, outcome: OutcomeEvent) -> bool:
        if not route_id:
            return False
        try:
            return bool(self._chain.attach_outcome(route_id, outcome))
        except Exception:
            return False

    def attach_verification(
        self,
        route_id: str,
        verification: VerifiedOutcome,
        *,
        event_id: Optional[str] = None,
        observed_at: Optional[float] = None,
    ) -> bool:
        """Append verifier evidence without rewriting the operational outcome."""
        if not route_id:
            return False
        try:
            return bool(self._chain.attach_verification(
                route_id,
                verification,
                event_id=event_id,
                observed_at=observed_at,
            ))
        except Exception:
            return False

    def finalize_turn(self, session_id: str, *, prev_interrupted: bool = False,
                      prev_retried: bool = False) -> int:
        try:
            result = self._chain.finalize_turn(
                session_id, prev_interrupted=prev_interrupted,
                prev_retried=prev_retried)
        except Exception:
            return 0
        return result if isinstance(result, int) and not isinstance(result, bool) else 0

    def similar_turns(self, embedding: list[float], k: int = 12) -> list[NeighborTurn]:
        try:
            result = self._chain.similar_turns(embedding, k)
        except Exception:
            return []
        return result if isinstance(result, list) else []

    def stats(self) -> dict:
        try:
            result = self._chain.stats()
        except Exception:
            return {}
        return result if isinstance(result, dict) else {}

    def health(self) -> bool:
        try:
            return bool(self._chain.health())
        except Exception:
            return False

    def active_provider(self) -> str:
        """Name of the first healthy provider — where writes land right now."""
        for member in self.providers:
            try:
                if member.health():
                    return type(member).__name__
            except Exception:
                continue
        return "none"


class RoutingRecorder:
    """Per-session route_id bookkeeping over a RouterMemory — WS2 uses this
    with three calls per turn:

        recorder.new_turn(session_id, prev_interrupted=..., prev_retried=...)
        route_id = recorder.record(text, features, route, session_id=...)
        recorder.attach(session_id, outcome)          # at the turn boundary
    """

    def __init__(self, memory: Optional[RouterMemory] = None):
        self.memory = memory if memory is not None else RouterMemory()
        self._active_route: dict[str, str] = {}       # session_id -> route_id

    def new_turn(self, session_id: str, *, prev_interrupted: bool = False,
                 prev_retried: bool = False) -> int:
        """Finalize the session's PREVIOUS turn (its user_retried/interrupt
        signals only become known now). Call at the start of each user turn."""
        return self.memory.finalize_turn(
            session_id, prev_interrupted=prev_interrupted,
            prev_retried=prev_retried)

    def record(self, instruction_text: str, features: Any, route: Any, *,
               session_id: str, turn_index: int = 0, source: str = "organic",
               classifier_tier: Optional[str] = None,
               propensity: Optional[str] = None,
               classifier_ms: float = 0.0, decision_ms: float = 0.0,
               decision_trace: Optional[dict] = None,
               compute_embedding: bool = True) -> Optional[str]:
        """Mint a route_id, build the privacy-gated event, embed when allowed,
        store — and remember the route_id as this session's active turn."""
        event, embedding_allowed = self.memory.make_decision_event(
            instruction_text, features, route, session_id=session_id,
            turn_index=turn_index, source=source,
            classifier_tier=classifier_tier, propensity=propensity,
            classifier_ms=classifier_ms, decision_ms=decision_ms,
            decision_trace=decision_trace)
        embedding = (self.memory.embed_document(instruction_text)
                     if (embedding_allowed and compute_embedding) else None)
        stored = self.memory.record_decision(event, embedding)
        self._active_route[session_id] = event.route_id
        return stored

    def attach(self, session_id: str, outcome: OutcomeEvent) -> bool:
        """Attach an outcome to the session's active (most recent) route_id."""
        route_id = self._active_route.get(session_id)
        if not route_id:
            return False
        return self.memory.attach_outcome(route_id, outcome)

    def verify(
        self,
        session_id: str,
        verification: VerifiedOutcome,
        *,
        event_id: Optional[str] = None,
        observed_at: Optional[float] = None,
    ) -> bool:
        """Attach late verification to the session's active route."""
        route_id = self._active_route.get(session_id)
        if not route_id:
            return False
        return self.memory.attach_verification(
            route_id,
            verification,
            event_id=event_id,
            observed_at=observed_at,
        )

    def active_route_id(self, session_id: str) -> Optional[str]:
        return self._active_route.get(session_id)


# ── report ────────────────────────────────────────────────────────────────────


def build(memory: RouterMemory) -> str:
    """Pure markdown report: active provider, chain health, stats."""
    lines: list[str] = []
    lines.append("# Router memory — facade report")
    lines.append("")
    lines.append(f"Active provider: **{memory.active_provider()}** "
                 f"(first healthy member; writes land here)")
    lines.append("")
    lines.append("## Provider chain")
    lines.append("")
    lines.append("| # | provider | health |")
    lines.append("|---|---|---|")
    for position, member in enumerate(memory.providers, start=1):
        try:
            healthy = bool(member.health())
        except Exception:
            healthy = False
        lines.append(f"| {position} | {type(member).__name__} | "
                     f"{'OK' if healthy else 'DOWN'} |")
    lines.append("")
    lines.append("## Stats (first provider that answers)")
    lines.append("")
    stats = memory.stats()
    if stats:
        lines.append("| key | value |")
        lines.append("|---|---|")
        for key in sorted(stats):
            lines.append(f"| {key} | {stats[key]} |")
    else:
        lines.append("_no stats available_")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="store_true",
                        help="print active provider, chain health, stats")
    args = parser.parse_args()
    if not args.report:
        parser.print_help()
        return 0
    print(build(RouterMemory()), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
