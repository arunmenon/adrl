"""Thread-safe runtime health registry and circuit breaker for model rungs."""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional

from .policy import REGISTRY


@dataclass(frozen=True)
class HealthState:
    healthy: bool
    consecutive_failures: int
    retry_after: float
    last_error: str | None = None


class RungHealthMonitor:
    """Runtime registry with immediate open-circuit and timed half-open retry."""

    def __init__(self, *, registry: Optional[dict[str, dict]] = None,
                 failure_threshold: int = 1, cooldown_s: float = 30.0,
                 clock: Callable[[], float] = time.monotonic):
        self._base = deepcopy(registry or REGISTRY)
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_s = max(0.0, float(cooldown_s))
        self._clock = clock
        self._states: dict[str, HealthState] = {
            rung: HealthState(bool(spec.get("healthy", True)), 0, 0.0)
            for rung, spec in self._base.items()
        }
        self._lock = threading.RLock()

    @staticmethod
    def normalize(rung: str) -> str:
        value = str(rung or "").replace("-", "_")
        if value in {"local", "local_code", "local_small"}:
            return "local"
        if value == "cheap_cloud":
            return "cheap_cloud"
        return value

    def record_success(self, rung: str) -> None:
        key = self.normalize(rung)
        with self._lock:
            if key in self._states:
                self._states[key] = HealthState(True, 0, 0.0)

    def record_failure(self, rung: str, error: str | None = None) -> None:
        key = self.normalize(rung)
        with self._lock:
            current = self._states.get(key)
            if current is None:
                return
            failures = current.consecutive_failures + 1
            unhealthy = failures >= self.failure_threshold
            retry_after = self._clock() + self.cooldown_s if unhealthy else 0.0
            self._states[key] = HealthState(
                not unhealthy, failures, retry_after, str(error) if error else None)

    def set_health(self, rung: str, healthy: bool,
                   error: str | None = None) -> None:
        """External probe hook; callers can feed active health-check results."""
        if healthy:
            self.record_success(rung)
        else:
            self.record_failure(rung, error)

    def state(self, rung: str) -> HealthState | None:
        key = self.normalize(rung)
        with self._lock:
            return self._effective_state(key, self._clock())

    def snapshot(self) -> dict[str, dict]:
        now = self._clock()
        with self._lock:
            result = deepcopy(self._base)
            for rung, spec in result.items():
                state = self._effective_state(rung, now)
                spec["healthy"] = bool(state.healthy) if state else False
            return result

    def _effective_state(self, rung: str, now: float) -> HealthState | None:
        current = self._states.get(rung)
        if current is None:
            return None
        if not current.healthy and now >= current.retry_after:
            # Half-open: allow one request. Success closes the circuit; another
            # failure opens it for a fresh cooldown.
            current = HealthState(True, current.consecutive_failures, 0.0,
                                  current.last_error)
            self._states[rung] = current
        return current
