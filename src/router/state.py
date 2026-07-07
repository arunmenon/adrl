"""C3 — the session state store (design §5.6, §6).

The router's "shared whiteboard": the per-session memory that turns a stream of
isolated HTTP requests into a coherent routing decision. Continuations do a
sticky lookup here (the hottest read); the escalation controller increments
strikes here; the privacy gate reads the one-way pin here.

Design decision, per §6: **the interface is the design; Redis is just the
default grown-up implementation.** We define `SessionStore` as an ABC now and
ship the single-process `DictSessionStore`. The day the proxy runs >1 worker,
a `RedisSessionStore` drops in behind the same ABC with no caller changes. Each
method's docstring names the Redis command it maps to so that swap is mechanical:

    get_session      HGETALL session:{id}         (+ auto-create)
    set_route        HSET    session:{id} route
    incr_strike      HINCRBY session:{id}:strikes {kind} 1     (atomic)
    reset_strikes    DEL     session:{id}:strikes             (per turn)
    pin_privacy      HSET    session:{id} privacy_pinned 1     (one-way)
    mark_escalated   HSET    session:{id} escalated 1
    incr_turn        HINCRBY session:{id} turn_count 1
    touch            EXPIRE  session:{id} 14400               (refresh TTL)
    expire_idle      (no-op — Redis TTL evicts automatically; see method doc)

We reuse `policy.SessionState` as the record type rather than duplicating its
fields — the store owns lifecycle and durability, the dataclass owns the shape.

Scope boundary: this store holds *live* session state only. The flywheel
outcome log (§5.7, the Redis `outcomes` STREAM / a real database) is a separate
sink and deliberately not part of this interface — §6 is explicit that durable
analytics do not belong in the live-state store.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod

from .policy import SessionState

# §6 keyspace: "all session:* keys get EXPIRE 14400, refreshed on touch."
IDLE_TTL_S = 4 * 60 * 60  # 14400 seconds = 4h idle


class SessionStore(ABC):
    """Redis-swappable interface for per-session router memory (§5.6/§6).

    Contract shared by every implementation:

    * **Auto-create.** `get_session` (and every mutator, which goes through it)
      creates a fresh `SessionState` for an unknown session id. There is no
      "does this session exist" question the caller must ask first — an unknown
      id is simply a session that has not acted yet. This mirrors §5.1's
      "when unsure, treat it as a live turn" bias: never lose state by guessing.
    * **One-way privacy pin.** `pin_privacy` is monotonic. Once pinned, the
      session never un-pins for its whole life; the *only* thing that clears it
      is a genuinely new session (`start_fresh_session`, or simply a new id).
      A pin violation is a security incident (§9), so the store re-asserts the
      pin on every read — a caller that flips the field back is overruled.
    * **Atomic strike increments.** `incr_strike` must not lose updates under
      concurrent continuations (§6.3). The dict impl uses a lock; the Redis
      impl uses `HINCRBY`.
    * **Per-turn strike reset.** `reset_strikes` is called at each new turn
      boundary; trip-wire counters are per-turn, not per-session (§5.5, §6).
    """

    @abstractmethod
    def get_session(self, sid: str) -> SessionState:
        """Return the live state for `sid`, auto-creating it if unknown."""

    @abstractmethod
    def set_route(self, sid: str, rung: str) -> SessionState:
        """Set the session's sticky route to `rung`; return the updated state."""

    @abstractmethod
    def incr_strike(self, sid: str, kind: str) -> int:
        """Atomically increment the `kind` trip-wire counter; return new count."""

    @abstractmethod
    def reset_strikes(self, sid: str) -> None:
        """Clear all trip-wire counters (called per turn boundary)."""

    @abstractmethod
    def pin_privacy(self, sid: str) -> None:
        """One-way: mark the session privacy-pinned. Never reversible in-life."""

    @abstractmethod
    def mark_escalated(self, sid: str) -> None:
        """Set `escalated_this_episode` (hysteresis gate, §5.3)."""

    @abstractmethod
    def incr_turn(self, sid: str) -> int:
        """Increment the session turn counter; return the new turn count."""

    @abstractmethod
    def touch(self, sid: str) -> None:
        """Refresh the idle TTL for `sid` (Redis: EXPIRE 14400). Auto-creates."""

    @abstractmethod
    def start_fresh_session(self, sid: str) -> SessionState:
        """Discard any prior state under `sid` and return a clean session.

        This is the ONLY sanctioned way a privacy pin clears (§5.6): a new
        session, not an un-pin. Present so a harness that recycles a session id
        can force a clean slate; in normal operation a new session is simply a
        new id and this is never needed.
        """

    @abstractmethod
    def expire_idle(self, now: float | None = None, ttl_s: int = IDLE_TTL_S) -> int:
        """Idle-eviction hook (§6). Documented no-op in the live impls — see
        `DictSessionStore.expire_idle`. Returns the number of sessions evicted."""


class DictSessionStore(SessionStore):
    """In-process, dict-backed store — the right choice for a single proxy
    process on one machine (§6 decision table). Thread-safe: a re-entrant lock
    guards every mutation so `incr_strike` keeps its atomic contract under the
    uvicorn threadpool, matching the `HINCRBY` guarantee the Redis impl gets
    for free.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        # sids that have EVER been pinned. Separate from the dataclass field so
        # the pin is durable against direct field mutation — the store's record
        # wins on every read, enforcing the one-way guarantee (§5.6/§9).
        self._pinned: set[str] = set()
        # last-activity wall-clock per sid; the substrate a real TTL sweep (or
        # the Redis EXPIRE refresh) would use. Tracked now, evicted never (below).
        self._last_touch: dict[str, float] = {}
        self._lock = threading.RLock()

    # ── reads ────────────────────────────────────────────────────────────────
    def get_session(self, sid: str) -> SessionState:
        with self._lock:
            state = self._sessions.get(sid)
            if state is None:
                state = SessionState(session_id=sid)
                self._sessions[sid] = state
                self._last_touch[sid] = time.time()
            # Re-assert the one-way pin: even if some caller flipped the field
            # back to False, a session that was ever pinned stays pinned.
            if sid in self._pinned:
                state.privacy_pinned = True
            return state

    # ── mutations ──────────────────────────────────────────────────────────
    def set_route(self, sid: str, rung: str) -> SessionState:
        with self._lock:
            state = self.get_session(sid)
            state.route = rung
            self._last_touch[sid] = time.time()
            return state

    def incr_strike(self, sid: str, kind: str) -> int:
        with self._lock:
            state = self.get_session(sid)
            state.strikes[kind] = state.strikes.get(kind, 0) + 1
            self._last_touch[sid] = time.time()
            return state.strikes[kind]

    def reset_strikes(self, sid: str) -> None:
        with self._lock:
            state = self.get_session(sid)
            state.strikes.clear()
            self._last_touch[sid] = time.time()

    def pin_privacy(self, sid: str) -> None:
        with self._lock:
            state = self.get_session(sid)
            state.privacy_pinned = True
            self._pinned.add(sid)  # durable record — this is what makes it one-way
            self._last_touch[sid] = time.time()

    def mark_escalated(self, sid: str) -> None:
        with self._lock:
            state = self.get_session(sid)
            state.escalated_this_episode = True
            self._last_touch[sid] = time.time()

    def incr_turn(self, sid: str) -> int:
        with self._lock:
            state = self.get_session(sid)
            state.turn_count += 1
            self._last_touch[sid] = time.time()
            return state.turn_count

    def touch(self, sid: str) -> None:
        with self._lock:
            self.get_session(sid)  # auto-create so a touched session exists
            self._last_touch[sid] = time.time()

    def start_fresh_session(self, sid: str) -> SessionState:
        with self._lock:
            self._pinned.discard(sid)  # the one sanctioned pin-clear (§5.6)
            state = SessionState(session_id=sid)
            self._sessions[sid] = state
            self._last_touch[sid] = time.time()
            return state

    def expire_idle(self, now: float | None = None, ttl_s: int = IDLE_TTL_S) -> int:
        """No-op-now hook (§6).

        In the Redis impl, per-key `EXPIRE 14400` (refreshed on every `touch`)
        evicts idle sessions automatically — there is nothing to sweep. The
        single-process dict impl deliberately does NOT actively evict: session
        lifetime is bounded by the process, and wiring a background sweeper here
        would be machinery §6 says belongs to the Redis swap, not before it.

        Kept as a real method (not just a comment) so the eviction seam and its
        ≥4h idle semantics are part of the interface today. `_last_touch` is
        maintained so a future sweep — or a diagnostic — has the data it needs.
        Returns 0: nothing is evicted now.
        """
        return 0

    # ── diagnostics (read-only; not part of the routing hot path) ──────────
    def idle_sessions(self, now: float | None = None, ttl_s: int = IDLE_TTL_S) -> list[str]:
        """Report which sessions are past the idle TTL, without evicting them.
        A read-only companion to `expire_idle` for dashboards/tests; the live
        no-op eviction policy above is unchanged by calling this."""
        now = time.time() if now is None else now
        with self._lock:
            return [sid for sid, ts in self._last_touch.items() if now - ts >= ttl_s]
