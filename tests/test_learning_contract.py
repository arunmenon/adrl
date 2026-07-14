"""Learning contract: predecision features, pairing, splits, and artifacts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from router.learning_contract import (
    LearningContractError,
    LearningEvidence,
    build_dataset_manifest,
    decision_eligibility_reasons,
    load_learning_contract,
    sanitize_predecision_features,
    split_learning_evidence,
    validate_artifact_manifest,
    validate_counterfactual_pairs,
    is_cause_clean_label,
)


def _features(**updates):
    values = {
        "verb_class": "fix",
        "verb_score": 0.55,
        "broad_scope": False,
        "narrow_scope": True,
        "context_tokens": 1200,
        "turn_index": 2,
        "recent_errors": 1,
        "recent_edit_failures": 0,
        "prev_turn_interrupted": False,
        "privacy_pinned": False,
        "escalated_this_episode": False,
        "is_terse_continuation": False,
        "extra": {},
    }
    values.update(updates)
    return values


def _evidence(identity: str, *, pair: str = "pair-1", rung: str = "local-code",
              decided_at: float = 10.0, repository: str = "a" * 64,
              task: str = "task-1") -> LearningEvidence:
    contract = load_learning_contract()
    return LearningEvidence(
        evidence_id=identity,
        pair_id=pair,
        task_id=task,
        repository_sha256=repository,
        snapshot_commit="b" * 40,
        verification_plan_sha256="c" * 64,
        features=sanitize_predecision_features(_features(), contract),
        decided_at=decided_at,
        source="counterfactual",
        rung=rung,
        model=f"model-{identity}",
        harness="codex",
        verified_success=True,
        quality_score=1.0,
        verifier_source="deterministic:v1",
        verifier_confidence=1.0,
        failure_cause=None,
        cost_estimate=0.1,
        latency_ms=100.0,
        user_retried=False,
    )


def test_checked_in_contract_has_stable_feature_and_artifact_requirements():
    contract = load_learning_contract()

    assert contract.contract_version == "adrl-learning-v1"
    assert contract.feature_schema_version == "predecision-turn-v1"
    assert len(contract.feature_schema_sha256) == 64
    assert "classifier_tier" in contract.forbidden_feature_fields
    assert "objective" in contract.artifact_schema["required_fields"]


def test_predecision_projection_drops_gates_and_rejects_leakage_or_schema_drift():
    projected = sanitize_predecision_features(_features())

    assert set(projected) == {
        "verb_class", "verb_score", "broad_scope", "narrow_scope",
        "context_tokens", "turn_index", "recent_errors",
        "recent_edit_failures", "prev_turn_interrupted",
    }
    with pytest.raises(LearningContractError, match="forbidden"):
        sanitize_predecision_features(_features(classifier_tier="hard"))
    with pytest.raises(LearningContractError, match="unversioned"):
        sanitize_predecision_features(_features(new_signal=1))
    with pytest.raises(LearningContractError, match="must be integer"):
        sanitize_predecision_features(_features(turn_index=True))


def test_eligibility_keeps_hard_gates_and_non_middle_routes_out_of_learning():
    assert decision_eligibility_reasons(
        source="organic", layer="classifier", features=_features()) == ()
    reasons = decision_eligibility_reasons(
        source="simulator", layer="gate:privacy",
        features=_features(privacy_pinned=True),
    )
    assert "ineligible decision source" in reasons
    assert "outside ambiguous routing layer" in reasons
    assert "hard/semantic gate active: privacy_pinned" in reasons


def test_counterfactual_pair_requires_same_snapshot_features_and_both_rungs():
    local = _evidence("local")
    frontier = _evidence("frontier", rung="frontier")

    assert set(validate_counterfactual_pairs((local, frontier))) == {"pair-1"}
    with pytest.raises(LearningContractError, match="no frontier"):
        validate_counterfactual_pairs((local,))
    with pytest.raises(LearningContractError, match="snapshot_commit"):
        validate_counterfactual_pairs((
            local, replace(frontier, snapshot_commit="d" * 40)))


def test_split_keeps_pairs_together_and_repository_holdout_unseen():
    train = (
        _evidence("train-local", pair="train", decided_at=10),
        _evidence("train-frontier", pair="train", rung="frontier", decided_at=10),
    )
    temporal = (
        _evidence("time-local", pair="time", decided_at=30),
        _evidence("time-frontier", pair="time", rung="frontier", decided_at=30),
    )
    heldout = (
        _evidence("repo-local", pair="repo", repository="d" * 64, decided_at=5,
                  task="task-heldout"),
        _evidence("repo-frontier", pair="repo", rung="frontier",
                  repository="d" * 64, decided_at=5, task="task-heldout"),
    )

    splits = split_learning_evidence(
        (*train, *temporal, *heldout), temporal_cutoff=20,
        heldout_repositories={"d" * 64},
    )

    assert {item.pair_id for item in splits.train} == {"train"}
    assert {item.pair_id for item in splits.temporal_holdout} == {"time"}
    assert {item.pair_id for item in splits.repository_task_holdout} == {"repo"}


def test_split_rejects_pair_that_crosses_temporal_boundary():
    with pytest.raises(LearningContractError, match="crosses the temporal cutoff"):
        split_learning_evidence(
            (
                _evidence("before", decided_at=10),
                _evidence("after", rung="frontier", decided_at=30),
            ),
            temporal_cutoff=20,
            heldout_repositories={"d" * 64},
        )


def test_split_and_clean_label_gate_reject_missing_quality_or_low_confidence():
    local = _evidence("local")
    missing_quality = replace(local, quality_score=None)
    assert is_cause_clean_label(missing_quality) is False
    with pytest.raises(LearningContractError, match="not a cause-clean label"):
        split_learning_evidence(
            (missing_quality,), temporal_cutoff=20,
            heldout_repositories={"d" * 64})
    with pytest.raises(LearningContractError, match="not a cause-clean label"):
        split_learning_evidence(
            (replace(local, verifier_confidence=0.5),), temporal_cutoff=20,
            heldout_repositories={"d" * 64})


def test_dataset_and_artifact_manifests_version_every_training_dependency():
    records = (
        _evidence("local"),
        _evidence("frontier", rung="frontier"),
        _evidence("heldout-local", pair="heldout", repository="d" * 64,
                  task="heldout"),
        _evidence("heldout-frontier", pair="heldout", rung="frontier",
                  repository="d" * 64, task="heldout"),
    )
    splits = split_learning_evidence(
        records, temporal_cutoff=20, heldout_repositories={"d" * 64})
    dataset = build_dataset_manifest(
        splits, temporal_cutoff=20, heldout_repositories={"d" * 64})
    contract = load_learning_contract()
    artifact = {
        "artifact_version": "router-2026-07-13.1",
        "contract_version": contract.contract_version,
        "feature_schema_version": contract.feature_schema_version,
        "feature_schema_sha256": contract.feature_schema_sha256,
        "data_snapshot_sha256": dataset["data_snapshot_sha256"],
        "objective": {"name": "marginal_frontier_components_v1"},
        "calibration": {"method": "isotonic", "split": "temporal_holdout"},
        "thresholds": {"abstain_below_confidence": 0.8},
        "policy_compatibility": ["v1"],
    }

    validate_artifact_manifest(artifact, contract)
    with pytest.raises(LearningContractError, match="missing: calibration"):
        validate_artifact_manifest({
            key: value for key, value in artifact.items() if key != "calibration"
        }, contract)
