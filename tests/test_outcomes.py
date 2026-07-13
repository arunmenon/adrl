"""Outcome-label integrity: friction and task difficulty stay distinct."""

from router.outcomes import (
    FailureCause,
    effective_task_hard,
    failure_cause_for,
    outcome_proxy_hard,
    task_signal_hard,
)


def test_dialect_failure_is_friction_not_task_difficulty():
    assert outcome_proxy_hard({"n_edit_failures": 1})
    assert not task_signal_hard(tripwire_type="dialect")
    assert failure_cause_for("dialect") == FailureCause.HARNESS_DIALECT.value


def test_task_tripwires_and_long_trajectory_are_hard_signals():
    assert task_signal_hard(tripwire_type="difficulty")
    assert task_signal_hard(tripwire_type="cost")
    assert task_signal_hard(continuation_count=10)
    assert not task_signal_hard(continuation_count=9)


def test_infrastructure_is_owned_by_health_not_difficulty():
    assert failure_cause_for(
        infrastructure_failure=True) == FailureCause.INFRASTRUCTURE.value
    assert not task_signal_hard()


def test_v2_task_signal_overrides_contaminated_legacy_proxy():
    # Explicit False is meaningful: this was known dialect/infra friction.
    assert not effective_task_hard(False, True, False, True)
    # NULL means a pre-v2 row, where only the legacy aggregate is available.
    assert effective_task_hard(None, True, False, False)


def test_verifier_evidence_only_overrides_when_it_answers_routing_question():
    # A successful cheap attempt proves frontier was unnecessary.
    assert not effective_task_hard(
        True, True, True, True, verified_success=True, rung="local-code")
    assert not effective_task_hard(
        True, True, True, True, verified_success=True, rung="cheap_cloud")
    # A deterministic failure is hard evidence at every rung.
    assert effective_task_hard(
        False, False, False, False, verified_success=False, rung="local")
    # A scope/policy failure is not evidence that the task needed a bigger model.
    assert not effective_task_hard(
        False, False, False, False, verified_success=False, rung="local",
        verification_failure_cause=FailureCause.POLICY_CONSTRAINT.value)
    # Frontier success alone says nothing about whether local would have worked.
    assert effective_task_hard(
        True, True, False, False, verified_success=True, rung="frontier")
