"""Evidence-derived learning readiness, gates, and transparent ADR scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from router.features import extract
from router.learning_contract import (
    feature_payload_sha256,
    load_learning_contract,
    sanitize_predecision_features,
)
from router.learning_readiness import (
    ReadinessError,
    build_markdown,
    collect_readiness,
    load_score_baseline,
    load_score_contract,
    record_history_snapshot,
    taxonomy_evidence_index,
    validate_persisted_score_artifacts,
    wilson_lower_bound,
)
from router.memory_facade import RouterMemory
from router.memory_ports import OutcomeEvent, VerifiedOutcome
from router.memory_sqlite import SqliteProvider
from router.policy import Route


def _route(provider: SqliteProvider, route_id: str, *, rung: str = "local-code",
           success: bool = True, cause=None):
    memory = RouterMemory([provider], embedder=False)
    features = extract("Fix VALUE", turn_index=1)
    route = Route(rung, True, "classifier", score=0.5)
    event, _ = memory.make_decision_event(
        "Fix VALUE", features, route, session_id=f"session-{route_id}",
        turn_index=1, source="organic", route_id=route_id,
        classifier_tier="standard", decision_trace={
            "resolver": "llm_classifier", "called": True,
            "abstained": False, "tier": "standard",
        },
    )
    assert provider.record_decision(event) == route_id
    assert provider.attach_outcome(route_id, OutcomeEvent(
        status="closed_final", event_id=f"final:{route_id}", observed_at=2.0))
    evidence_id = f"live-verification:1700000000000-{route_id[-12:].rjust(12, '0')}"
    assert provider.attach_verification(
        route_id,
        VerifiedOutcome(
            task_success=success,
            quality_score=1.0 if success else 0.0,
            verifier_source="deterministic:v1",
            confidence=1.0,
            verified_at=3.0,
            failure_cause=cause,
        ),
        event_id=evidence_id,
        observed_at=3.0,
    )
    return event, evidence_id


def _context(event, evidence_id: str, *, pair: str, rung: str,
             success: bool = True, cause=None):
    contract = load_learning_contract()
    return {
        "schema_version": "organic-verification-context-v1",
        "evidence_id": evidence_id,
        "route_id": event.route_id,
        "task_id": "task-1",
        "pair_id": pair,
        "decision_source": "organic",
        "decision_ts": event.ts,
        "decision_layer": "classifier",
        "policy_version": event.policy_version,
        "selected_rung": rung,
        "served_rung": rung,
        "model": f"model-{rung}",
        "harness": "codex",
        "repository_sha256": "a" * 64,
        "snapshot_commit": "b" * 40,
        "learning_contract_version": contract.contract_version,
        "feature_schema_version": contract.feature_schema_version,
        "features_sha256": feature_payload_sha256(event.features_json),
        "verification_plan_version": "plan-v1",
        "verification_plan_sha256": "c" * 64,
        "classifier_provenance": {
            "tier": "standard",
            "trace": {"called": True, "abstained": False},
        },
        "verification": {
            "task_success": success,
            "quality_score": 1.0 if success else 0.0,
            "confidence": 1.0,
            "verifier_source": "deterministic:v1",
            "failure_cause": cause,
            "checks": [],
        },
    }


def _write_jsonl(path: Path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values),
                    encoding="utf-8")


def _collect_readiness(tmp_path: Path, **overrides):
    paths = {
        "evidence_path": tmp_path / "organic-evidence.jsonl",
        "counterfactual_path": tmp_path / "counterfactual-evidence.jsonl",
        "label_audit_path": tmp_path / "label-audit.jsonl",
    }
    paths.update(overrides)
    return collect_readiness(**paths)


def _counterfactual_record(identity: str, *, rung: str, source: str = "synthetic"):
    contract = load_learning_contract()
    values = vars(extract("Fix VALUE", turn_index=0)).copy()
    values.pop("instruction_text")
    features = sanitize_predecision_features(values, contract)
    return {
        "evidence_schema_version": "counterfactual-evidence-v1",
        "evidence_id": f"counterfactual:run-1:{identity}",
        "pair_id": "run-1",
        "learning_contract_version": contract.contract_version,
        "feature_schema_version": contract.feature_schema_version,
        "features": features,
        "features_sha256": feature_payload_sha256(features),
        "repository_sha256": "a" * 64,
        "verification_plan_version": "plan-v1",
        "verification_plan_sha256": "c" * 64,
        "decided_at": 1.0,
        "run_id": "run-1",
        "task_id": "task-1",
        "task_source": source,
        "prompt_sha256": "d" * 64,
        "prompt": None,
        "snapshot_commit": "b" * 40,
        "candidate": {"rung": rung, "model": f"model-{rung}", "harness": "codex"},
        "execution": {"duration_ms": 10.0, "cost_estimate": 0.1},
        "verification": {
            "task_success": True,
            "quality_score": 1.0,
            "confidence": 1.0,
            "verifier_source": "deterministic:v1",
            "failure_cause": None,
        },
    }


def test_readiness_counts_cause_clean_pair_but_keeps_precision_gate_blocked(tmp_path):
    db_path = tmp_path / "memory.db"
    provider = SqliteProvider(db_path)
    local, local_id = _route(provider, "local-route-01", rung="local-code")
    frontier, frontier_id = _route(provider, "frontier-001", rung="frontier")
    evidence_path = tmp_path / "evidence.jsonl"
    _write_jsonl(evidence_path, (
        _context(local, local_id, pair="pair-1", rung="local-code"),
        _context(frontier, frontier_id, pair="pair-1", rung="frontier"),
    ))
    audit_path = tmp_path / "label-audit.jsonl"
    _write_jsonl(audit_path, ({
        "schema_version": "label-review-v1",
        "evidence_id": local_id,
        "reviewer_role": "operator",
        "verifier_result_agrees": True,
        "failure_cause_agrees": True,
        "reviewed_at": 4.0,
    },))

    report = _collect_readiness(
        tmp_path, db_path=db_path, evidence_path=evidence_path,
        label_audit_path=audit_path, generated_at=1.0)

    assert report["counts"]["cause_clean_organic_labels"] == 2
    assert report["counts"]["usable_final_middle_labels"] == 2
    assert report["counts"]["valid_same_snapshot_pairs"] == 1
    assert report["gates"]["organic_verifier_labels"]["status"] == "pass"
    assert report["gates"]["classifier_provenance"]["status"] == "pass"
    assert report["gates"]["paired_counterfactual_integrity"]["status"] == "pass"
    assert report["gates"]["representative_pairs"]["status"] == "pass"
    assert report["gates"]["label_precision"]["status"] == "blocked"
    assert report["label_audit"]["valid_reviews"] == 1
    assert report["label_audit"]["strata"]["success"]["reviewed"] == 1
    assert report["production_readiness"] == "blocked"
    assert "Production readiness: **BLOCKED**" in build_markdown(report)
    assert report["baseline"]["score"] == pytest.approx(40.5)
    assert report["baseline"]["current_delta"] == pytest.approx(2.3)
    assert report["score_contract"]["version"] == "architecture-evidence-index-v1"


def test_tampered_context_is_counted_invalid_and_never_becomes_training_data(tmp_path):
    db_path = tmp_path / "memory.db"
    provider = SqliteProvider(db_path)
    event, evidence_id = _route(provider, "route-tamper")
    context = _context(event, evidence_id, pair="pair-x", rung="local-code")
    context["features_sha256"] = "d" * 64
    evidence_path = tmp_path / "evidence.jsonl"
    _write_jsonl(evidence_path, (context,))

    report = _collect_readiness(
        tmp_path, db_path=db_path, evidence_path=evidence_path,
        label_audit_path=tmp_path / "missing.jsonl", generated_at=1.0)

    assert report["counts"]["invalid_verification_contexts"] == 1
    assert report["counts"]["cause_clean_organic_labels"] == 0
    assert report["gates"]["organic_verifier_labels"]["status"] == "blocked"


def test_synthetic_counterfactual_pair_passes_integrity_not_representativeness(
    tmp_path,
):
    db_path = tmp_path / "memory.db"
    assert SqliteProvider(db_path).health()
    counterfactual_path = tmp_path / "counterfactual.jsonl"
    _write_jsonl(counterfactual_path, (
        _counterfactual_record("local", rung="local-code"),
        _counterfactual_record("frontier", rung="frontier"),
    ))

    report = _collect_readiness(
        tmp_path, db_path=db_path, counterfactual_path=counterfactual_path,
        generated_at=1.0)

    assert report["counts"]["valid_counterfactual_records"] == 2
    assert report["counts"]["valid_same_snapshot_pairs"] == 1
    assert report["counts"]["representative_same_snapshot_pairs"] == 0
    assert report["gates"]["paired_counterfactual_integrity"]["status"] == "pass"
    assert report["gates"]["representative_pairs"]["status"] == "blocked"


def test_missing_ledger_is_an_error_instead_of_an_empty_readiness_report(tmp_path):
    with pytest.raises(ReadinessError, match="does not exist"):
        _collect_readiness(
            tmp_path, db_path=tmp_path / "typo.db", generated_at=1.0)


def test_wilson_gate_requires_statistical_support_not_raw_perfect_rate():
    assert wilson_lower_bound(1, 1, 1.959963984540054) < 0.9
    assert wilson_lower_bound(40, 40, 1.959963984540054) > 0.9
    with pytest.raises(ReadinessError):
        wilson_lower_bound(2, 1, 1.96)


def test_taxonomy_index_exposes_weights_profiles_and_current_learning_progress():
    index = taxonomy_evidence_index()

    assert index["name"] == "architecture_evidence_index_not_production_readiness"
    assert index["buckets"]["LRN"]["score"] == pytest.approx(22.9)
    assert index["buckets"]["EVL"]["score"] == pytest.approx(35.6)
    assert index["buckets"]["LRN"]["weight"] == pytest.approx(1 / 9)
    assert index["buckets"]["LRN"]["evidence_profile"]["D0 Design"] == 3
    assert index["confidence"]["representation"] == (
        "per-bucket D-level evidence profile")


def test_frozen_score_contract_and_taxonomy_baseline_are_reproducible():
    contract = load_score_contract()
    baseline = load_score_baseline()
    index = taxonomy_evidence_index()

    assert contract["sha256"] == (
        "d26f5175f43a66aac99d80887c88de81ac6683d39127c53553a3d859df677bb4"
    )
    assert baseline["baseline_id"] == "taxonomy-v1-freeze-a46ca8c"
    assert baseline["source_git_commit"] == (
        "a46ca8ca5d0c293fb701fedfe2575a78d8d491d8"
    )
    assert baseline["architecture_evidence_index"]["score"] == pytest.approx(40.5)
    assert index["score"] == pytest.approx(42.8)
    assert index["score_unrounded"] - baseline["architecture_evidence_index"][
        "score_unrounded"
    ] == pytest.approx(2.257495590828924)


def test_contract_or_baseline_tampering_cannot_silently_move_goalposts(tmp_path):
    contract_path = Path("config/readiness-score-v1.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["index"]["bucket_weights"]["LRN"] = 2
    tampered_contract = tmp_path / "score-contract.json"
    tampered_contract.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ReadinessError, match="different score contract"):
        load_score_baseline(score_contract_path=tampered_contract)

    baseline_path = Path("reports/readiness-baseline-v1.json")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["architecture_evidence_index"]["score"] = 41.0
    tampered_baseline = tmp_path / "baseline.json"
    tampered_baseline.write_text(json.dumps(baseline), encoding="utf-8")

    with pytest.raises(ReadinessError, match="displayed score was modified"):
        load_score_baseline(tampered_baseline)

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["architecture_evidence_index"]["source"][
        "taxonomy_evidence_sha256"
    ] = "f" * 64
    forged_source = tmp_path / "forged-source-baseline.json"
    forged_source.write_text(json.dumps(baseline), encoding="utf-8")
    with pytest.raises(ReadinessError, match="does not match its source commit"):
        validate_persisted_score_artifacts(score_baseline_path=forged_source)


def test_history_records_each_evidence_state_once(tmp_path):
    db_path = tmp_path / "memory.db"
    provider = SqliteProvider(db_path)
    assert provider.health()
    first = _collect_readiness(tmp_path, db_path=db_path, generated_at=1.0)
    second = _collect_readiness(tmp_path, db_path=db_path, generated_at=2.0)
    history_path = tmp_path / "history.jsonl"

    assert first["snapshot_id"] == second["snapshot_id"]
    assert record_history_snapshot(first, history_path) is True
    assert record_history_snapshot(second, history_path) is False
    assert len(history_path.read_text(encoding="utf-8").splitlines()) == 1

    _route(provider, "history-route")
    third = _collect_readiness(tmp_path, db_path=db_path, generated_at=3.0)
    assert record_history_snapshot(third, history_path) is True
    entries = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
    ]
    assert entries[1]["previous_entry_sha256"] == entries[0]["entry_sha256"]

    entries[0]["counts"]["cause_clean_organic_labels"] += 1
    _write_jsonl(history_path, entries)
    with pytest.raises(ReadinessError, match="content hash mismatches"):
        record_history_snapshot(third, history_path)


def test_persisted_score_artifacts_are_cross_checked_without_private_data(tmp_path):
    db_path = tmp_path / "memory.db"
    assert SqliteProvider(db_path).health()
    report = _collect_readiness(tmp_path, db_path=db_path, generated_at=1.0)
    json_path = tmp_path / "learning-readiness.json"
    markdown_path = tmp_path / "learning-readiness.md"
    history_path = tmp_path / "history.jsonl"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    markdown_path.write_text(build_markdown(report), encoding="utf-8")
    assert record_history_snapshot(report, history_path)

    result = validate_persisted_score_artifacts(
        json_report_path=json_path,
        markdown_report_path=markdown_path,
        history_path=history_path,
    )

    assert result["baseline_score"] == pytest.approx(40.5)
    assert result["current_score"] == pytest.approx(42.8)
    assert result["current_delta"] == pytest.approx(2.3)

    markdown_path.write_text("stale\n", encoding="utf-8")
    with pytest.raises(ReadinessError, match="Markdown and JSON disagree"):
        validate_persisted_score_artifacts(
            json_report_path=json_path,
            markdown_report_path=markdown_path,
            history_path=history_path,
        )
