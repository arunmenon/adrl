"""Shared outcome labels (WS1).

``outcome_proxy_hard`` preserves the original operational-friction metric.
``task_signal_hard`` removes known harness-dialect and infrastructure
contamination for learning, while ``effective_task_hard`` keeps old ledgers
readable by falling back only when a v2 signal is absent. These definitions are
shared by memory, rule health, and retrieval so evaluation cannot drift.

Import-only module — no CLI.
"""

from __future__ import annotations

from enum import Enum


class FailureCause(str, Enum):
    """Routing-relevant cause taxonomy for outcome-label integrity."""

    TASK_CAPABILITY = "task_capability"
    HARNESS_DIALECT = "harness_dialect"
    INFRASTRUCTURE = "infrastructure"
    POLICY_CONSTRAINT = "policy_constraint"
    USER_ABORT = "user_abort"
    UNVERIFIABLE = "unverifiable"


def _type_value(tripwire_type) -> str:
    value = getattr(tripwire_type, "value", tripwire_type)
    return str(value or "").lower()


def outcome_proxy_hard(row: dict) -> bool:
    """Weak operational proxy for a turn that experienced friction.

    Same definition the bake-off used: any edit-apply failure, any errored tool
    result, a user interrupt, or a runaway (>=10 continuations). Non-validating
    (see classifier-bakeoff.md caveats) but the only ground-truth-ish signal we
    have for whether frontier help was warranted.
    """
    return (
        (row.get("n_edit_failures") or 0) >= 1
        or (row.get("n_error_results") or 0) >= 1
        or bool(row.get("interrupted"))
        or (row.get("n_continuations") or 0) >= 10
    )


def failure_cause_for(tripwire_type=None, *, infrastructure_failure: bool = False):
    """Map mechanical evidence to the label owner that should learn from it."""
    kind = _type_value(tripwire_type)
    if kind == "dialect":
        return FailureCause.HARNESS_DIALECT.value
    if kind in ("difficulty", "cost"):
        return FailureCause.TASK_CAPABILITY.value
    if kind == "quality":
        return FailureCause.USER_ABORT.value
    if infrastructure_failure:
        return FailureCause.INFRASTRUCTURE.value
    return None


def task_signal_hard(*, tripwire_type=None, user_retried=False,
                     continuation_count: int = 0) -> bool:
    """Weak task-difficulty signal with dialect/infra contamination removed.

    This is deliberately separate from ``outcome_proxy_hard``. A malformed tool
    call is operational friction, but it says the model/harness pair is
    incompatible rather than that the underlying task required a stronger model.
    """
    kind = _type_value(tripwire_type)
    return (
        bool(user_retried)
        or int(continuation_count or 0) >= 10
        or kind in ("difficulty", "cost", "quality")
    )


def went_hard(escalated, user_retried, proxy_hard) -> bool:
    """Legacy three-signal label retained for pre-v2 rows and old reports.

    New routing evidence should use ``task_signal_hard``. This legacy label
    mixes operational friction with task difficulty because historical rows do
    not carry enough cause information to separate them.

    The label combines an outcome row's
    three signals: an escalation fired, the user re-tried the turn, or the
    friction proxy tripped. Single home (this module) so the retrieval vote
    (WS4), the shadow ground truth, and rule health (WS3) all score by the SAME
    definition rather than three subtly different copies. Any None counts False.
    """
    return bool(escalated) or bool(user_retried) or bool(proxy_hard)


def effective_task_hard(
    task_signal,
    escalated,
    user_retried,
    proxy_hard,
    verified_success=None,
    rung=None,
    verification_failure_cause=None,
) -> bool:
    """Prefer verifier evidence where it answers the routing question.

    A deterministic failure is hard evidence that the attempted rung did not
    complete the task. A successful non-frontier attempt proves frontier was
    unnecessary. A successful frontier attempt does *not* prove that cheaper
    rungs would have failed, so it leaves the cause-clean/legacy label intact.
    """
    verified = None
    if verified_success is not None:
        if isinstance(verified_success, str):
            verified = verified_success.strip().lower() in {"1", "true", "yes"}
        else:
            verified = bool(verified_success)
    verification_cause = getattr(
        verification_failure_cause, "value", verification_failure_cause)
    verification_cause = str(verification_cause or "").lower()
    if verified is False and verification_cause in {
        "", FailureCause.TASK_CAPABILITY.value,
    }:
        return True
    normalized_rung = str(rung or "").replace("_", "-").lower()
    if verified is True and normalized_rung in {
        "local", "local-code", "local-small", "cheap-cloud",
    }:
        return False
    if task_signal is not None:
        return bool(task_signal)
    return went_hard(escalated, user_retried, proxy_hard)
