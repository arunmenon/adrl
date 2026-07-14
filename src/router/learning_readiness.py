"""Evidence-derived readiness and label-quality reporting for learned routing.

The report deliberately separates architecture implementation maturity from
observed ledger evidence and graduation gates. The arithmetic index is only a
transparent summary of ADR D-levels; it is not a production-readiness claim.
Hard evidence gates remain visible and cannot be averaged away.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .learning_contract import (
    DEFAULT_CONTRACT_PATH,
    LearningContract,
    LearningContractError,
    LearningEvidence,
    decision_eligibility_reasons,
    feature_payload_sha256,
    is_cause_clean_label,
    learning_evidence_from_counterfactual_record,
    load_learning_contract,
    sanitize_predecision_features,
    validate_counterfactual_pairs,
    validate_learning_evidence,
)
from .memory_sqlite import DEFAULT_DB_PATH, SqliteProvider


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE_PATH = Path("data/verified-evidence.jsonl")
DEFAULT_COUNTERFACTUAL_PATH = Path("data/counterfactual/records.jsonl")
DEFAULT_LABEL_AUDIT_PATH = Path("data/label-audit.jsonl")
DEFAULT_AUDIT_POLICY_PATH = ROOT / "config" / "label-audit-policy-v1.json"
DEFAULT_ADR_INDEX_PATH = ROOT / "docs" / "adr-index.md"
DEFAULT_SCORE_CONTRACT_PATH = ROOT / "config" / "readiness-score-v1.json"
DEFAULT_SCORE_BASELINE_PATH = ROOT / "reports" / "readiness-baseline-v1.json"
DEFAULT_MARKDOWN_REPORT_PATH = ROOT / "reports" / "learning-readiness.md"
DEFAULT_JSON_REPORT_PATH = ROOT / "reports" / "learning-readiness.json"
DEFAULT_HISTORY_PATH = ROOT / "reports" / "readiness-history.jsonl"
REPORT_SCHEMA_VERSION = "learning-readiness-v1"
SCORE_BASELINE_SCHEMA_VERSION = "readiness-baseline-v1"
HISTORY_ENTRY_SCHEMA_VERSION = "readiness-history-entry-v1"
FROZEN_BUCKETS = (
    "FND", "SEM", "SAF", "RTG", "CAS", "MEM", "LRN", "EVL", "OPS",
)
MATURITY_LEVELS = (
    "D0 Design", "D1 Code", "D2 Tested", "D3 Shadow", "D4 Pilot",
    "D5 Graduated",
)
GATE_ORDER = (
    "feature_contract",
    "organic_verifier_labels",
    "classifier_provenance",
    "label_precision",
    "paired_counterfactual_integrity",
    "representative_pairs",
    "learned_router_authority",
)
LABEL_STRATA_ORDER = ("success", "task_capability_failure")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


class ReadinessError(RuntimeError):
    """Readiness inputs are unavailable or internally inconsistent."""


@dataclass(frozen=True)
class Gate:
    status: str
    evidence: str


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReadinessError(f"unable to read {name}") from exc
    if not isinstance(value, Mapping):
        raise ReadinessError(f"{name} must be a JSON object")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _display_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_score_contract(
    path: Path = DEFAULT_SCORE_CONTRACT_PATH,
) -> dict[str, Any]:
    """Load and strictly validate the versioned architecture-score method."""
    raw = dict(_read_json(path, "readiness score contract"))
    if raw.get("schema_version") != "readiness-score-contract-v1":
        raise ReadinessError("unknown readiness score contract schema")
    if raw.get("contract_version") != "architecture-evidence-index-v1":
        raise ReadinessError("unknown readiness score contract version")
    if raw.get("taxonomy_version") != "1.0":
        raise ReadinessError("readiness score contract taxonomy version must be 1.0")

    index = raw.get("index")
    if not isinstance(index, Mapping):
        raise ReadinessError("readiness score contract has no index definition")
    if index.get("name") != "architecture_evidence_index_not_production_readiness":
        raise ReadinessError("readiness score contract has an unknown index name")
    if index.get("accepted_state") != "Accepted":
        raise ReadinessError("readiness score contract must score accepted decisions")
    if index.get("decision_weighting") != "equal_within_bucket":
        raise ReadinessError("readiness score contract decision weighting is unsupported")

    bucket_order = index.get("bucket_order")
    if not isinstance(bucket_order, list) or tuple(bucket_order) != FROZEN_BUCKETS:
        raise ReadinessError("readiness score contract changed the frozen buckets")
    bucket_weights = index.get("bucket_weights")
    if not isinstance(bucket_weights, Mapping) or set(bucket_weights) != set(
        FROZEN_BUCKETS
    ):
        raise ReadinessError("readiness score contract bucket weights are incomplete")
    for bucket in FROZEN_BUCKETS:
        weight = bucket_weights[bucket]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ReadinessError(f"readiness score weight for {bucket} is not numeric")
        if not math.isfinite(float(weight)) or float(weight) <= 0:
            raise ReadinessError(f"readiness score weight for {bucket} must be positive")

    maturity_points = index.get("maturity_points")
    if not isinstance(maturity_points, Mapping) or set(maturity_points) != set(
        MATURITY_LEVELS
    ):
        raise ReadinessError("readiness score maturity scale is incomplete")
    values: list[float] = []
    for level in MATURITY_LEVELS:
        value = maturity_points[level]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ReadinessError(f"readiness score point for {level} is not numeric")
        number = float(value)
        if not math.isfinite(number) or not 0 <= number <= 100:
            raise ReadinessError(f"readiness score point for {level} is outside 0..100")
        values.append(number)
    if values != sorted(set(values)):
        raise ReadinessError("readiness score maturity points must strictly increase")

    digits = index.get("rounding_digits")
    if isinstance(digits, bool) or not isinstance(digits, int) or not 0 <= digits <= 6:
        raise ReadinessError("readiness score rounding_digits must be an integer in 0..6")

    readiness = raw.get("production_readiness")
    if not isinstance(readiness, Mapping) or (
        readiness.get("strategy") != "hard_gate_verdict"
        or readiness.get("blocked_gate_status") != "blocked"
        or readiness.get("eligible_status") != "eligible_for_review"
    ):
        raise ReadinessError("readiness score contract hard-gate verdict is invalid")

    return {
        **raw,
        "path": _display_path(path),
        "sha256": _canonical_sha256(raw),
    }


def _score_contract_reference(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": contract["contract_version"],
        "sha256": contract["sha256"],
        "taxonomy_version": contract["taxonomy_version"],
    }


def _read_jsonl(path: Path, name: str) -> list[Mapping[str, Any]]:
    if not Path(path).exists():
        return []
    result: list[Mapping[str, Any]] = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ReadinessError(f"{name} contains a non-object record")
            result.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReadinessError(f"unable to read {name}") from exc
    return result


def wilson_lower_bound(successes: int, total: int, z: float) -> float:
    """Wilson score lower confidence bound for a binomial agreement rate."""
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           for value in (successes, total, z)):
        raise ReadinessError("Wilson inputs must be numeric")
    if successes < 0 or total < 0 or successes > total or z <= 0:
        raise ReadinessError("Wilson inputs are outside their valid range")
    if total == 0:
        return 0.0
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = rate + z * z / (2.0 * total)
    margin = z * math.sqrt(
        rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
    return max(0.0, (center - margin) / denominator)


def _score_profiles(
    profiles: Mapping[str, Mapping[str, int]],
    contract: Mapping[str, Any],
) -> tuple[float, dict[str, Any]]:
    index_contract = contract["index"]
    maturity_points = index_contract["maturity_points"]
    raw_weights = index_contract["bucket_weights"]
    total_weight = sum(float(raw_weights[bucket]) for bucket in FROZEN_BUCKETS)
    digits = int(index_contract["rounding_digits"])
    overall = 0.0
    bucket_results: dict[str, Any] = {}
    for bucket in FROZEN_BUCKETS:
        profile = profiles.get(bucket)
        if not isinstance(profile, Mapping):
            raise ReadinessError(f"readiness profile is missing bucket {bucket}")
        unknown = set(profile) - set(MATURITY_LEVELS)
        if unknown:
            raise ReadinessError(
                f"readiness profile for {bucket} has unknown maturity levels"
            )
        count = 0
        points = 0.0
        normalized_profile: dict[str, int] = {}
        for level in MATURITY_LEVELS:
            value = profile.get(level, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ReadinessError(
                    f"readiness profile count for {bucket}/{level} is invalid"
                )
            if value:
                normalized_profile[level] = value
            count += value
            points += float(maturity_points[level]) * value
        if count == 0:
            raise ReadinessError(f"readiness profile has no decisions in {bucket}")
        score = points / count
        weight = float(raw_weights[bucket]) / total_weight
        overall += score * weight
        bucket_results[bucket] = {
            "score": round(score, digits),
            "score_unrounded": score,
            "weight": weight,
            "decision_weight": 1.0 / count,
            "decision_count": count,
            "evidence_profile": normalized_profile,
        }
    return overall, bucket_results


def _taxonomy_evidence_index_from_text(
    text: str, source_path: str, contract: Mapping[str, Any],
) -> dict[str, Any]:
    profiles: dict[str, Counter[str]] = {
        bucket: Counter() for bucket in FROZEN_BUCKETS
    }
    source_rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("| `ADRL-"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            continue
        match = re.fullmatch(r"`(ADRL-([A-Z]{3})-\d{3})`", cells[0])
        if (
            match
            and match.group(2) in profiles
            and cells[2] == contract["index"]["accepted_state"]
            and cells[3] in contract["index"]["maturity_points"]
        ):
            profiles[match.group(2)][cells[3]] += 1
            source_rows.append({"id": match.group(1), "maturity": cells[3]})
    if any(not profiles[bucket] for bucket in FROZEN_BUCKETS):
        raise ReadinessError("ADR index is missing accepted decisions in a bucket")

    overall, bucket_results = _score_profiles(profiles, contract)
    digits = int(contract["index"]["rounding_digits"])
    return {
        "name": contract["index"]["name"],
        "score": round(overall, digits),
        "score_unrounded": overall,
        "scale": dict(contract["index"]["maturity_points"]),
        "bucket_weighting": "normalized versioned bucket weights",
        "decision_weighting": contract["index"]["decision_weighting"],
        "rounding_digits": digits,
        "score_contract": _score_contract_reference(contract),
        "source": {
            "path": source_path,
            "decision_count": len(source_rows),
            "taxonomy_evidence_sha256": _canonical_sha256(source_rows),
        },
        "confidence": {
            "representation": "per-bucket D-level evidence profile",
            "limitation": (
                "D-levels identify evidence stage; statistical confidence is "
                "reported separately for label precision"
            ),
        },
        "buckets": bucket_results,
    }


def taxonomy_evidence_index(
    index_path: Path = DEFAULT_ADR_INDEX_PATH,
    score_contract_path: Path = DEFAULT_SCORE_CONTRACT_PATH,
) -> dict[str, Any]:
    """Summarize accepted ADR evidence under a versioned scoring contract."""
    contract = load_score_contract(score_contract_path)
    try:
        text = Path(index_path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ReadinessError("unable to read ADR index") from exc
    return _taxonomy_evidence_index_from_text(
        text, _display_path(index_path), contract,
    )


def _same_number(left: Any, right: float) -> bool:
    return (
        not isinstance(left, bool)
        and isinstance(left, (int, float))
        and math.isfinite(float(left))
        and math.isclose(float(left), right, rel_tol=0.0, abs_tol=1e-9)
    )


def load_score_baseline(
    path: Path = DEFAULT_SCORE_BASELINE_PATH,
    score_contract_path: Path = DEFAULT_SCORE_CONTRACT_PATH,
    *,
    contract: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Load a fixed baseline and prove it was computed with this contract."""
    score_contract = (
        dict(contract) if contract is not None
        else load_score_contract(score_contract_path)
    )
    raw = dict(_read_json(path, "readiness score baseline"))
    if raw.get("schema_version") != SCORE_BASELINE_SCHEMA_VERSION:
        raise ReadinessError("unknown readiness score baseline schema")
    baseline_id = raw.get("baseline_id")
    if not isinstance(baseline_id, str) or not baseline_id.strip():
        raise ReadinessError("readiness score baseline has no baseline_id")
    if not isinstance(raw.get("recorded_at"), str) or not raw["recorded_at"]:
        raise ReadinessError("readiness score baseline has no recorded_at timestamp")
    if raw.get("score_contract") != _score_contract_reference(score_contract):
        raise ReadinessError(
            "readiness score baseline was produced by a different score contract"
        )
    source_commit = raw.get("source_git_commit")
    if not isinstance(source_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", source_commit
    ):
        raise ReadinessError("readiness score baseline has no exact source commit")

    saved_index = raw.get("architecture_evidence_index")
    if not isinstance(saved_index, Mapping):
        raise ReadinessError("readiness score baseline has no architecture index")
    if saved_index.get("name") != score_contract["index"]["name"]:
        raise ReadinessError("readiness score baseline index name mismatches")
    if saved_index.get("score_contract") != _score_contract_reference(score_contract):
        raise ReadinessError("readiness score baseline index contract mismatches")
    source = saved_index.get("source")
    if not isinstance(source, Mapping) or not _HEX_64.fullmatch(
        str(source.get("taxonomy_evidence_sha256") or "")
    ):
        raise ReadinessError("readiness score baseline source fingerprint is invalid")
    if source.get("path") != "docs/adr-index.md":
        raise ReadinessError("readiness score baseline source path is invalid")

    saved_buckets = saved_index.get("buckets")
    if not isinstance(saved_buckets, Mapping) or set(saved_buckets) != set(
        FROZEN_BUCKETS
    ):
        raise ReadinessError("readiness score baseline buckets are incomplete")
    profiles: dict[str, Mapping[str, int]] = {}
    for bucket in FROZEN_BUCKETS:
        item = saved_buckets[bucket]
        if not isinstance(item, Mapping) or not isinstance(
            item.get("evidence_profile"), Mapping
        ):
            raise ReadinessError(
                f"readiness score baseline bucket {bucket} has no evidence profile"
            )
        profiles[bucket] = item["evidence_profile"]

    overall, recomputed_buckets = _score_profiles(profiles, score_contract)
    digits = int(score_contract["index"]["rounding_digits"])
    if not _same_number(saved_index.get("score_unrounded"), overall):
        raise ReadinessError("readiness score baseline unrounded score was modified")
    if not _same_number(saved_index.get("score"), round(overall, digits)):
        raise ReadinessError("readiness score baseline displayed score was modified")
    decision_count = 0
    for bucket in FROZEN_BUCKETS:
        saved_bucket = saved_buckets[bucket]
        recomputed = recomputed_buckets[bucket]
        decision_count += recomputed["decision_count"]
        for field in ("score", "score_unrounded", "weight", "decision_weight"):
            if not _same_number(saved_bucket.get(field), recomputed[field]):
                raise ReadinessError(
                    f"readiness score baseline {bucket}/{field} was modified"
                )
        if saved_bucket.get("decision_count") != recomputed["decision_count"]:
            raise ReadinessError(
                f"readiness score baseline {bucket} decision count was modified"
            )
    if source.get("decision_count") != decision_count:
        raise ReadinessError("readiness score baseline total decision count mismatches")
    return raw


def _baseline_reference(
    baseline: Mapping[str, Any], current_index: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_index = baseline["architecture_evidence_index"]
    digits = int(current_index["rounding_digits"])
    delta = float(current_index["score_unrounded"]) - float(
        baseline_index["score_unrounded"]
    )
    return {
        "id": baseline["baseline_id"],
        "recorded_at": baseline["recorded_at"],
        "source_git_commit": baseline["source_git_commit"],
        "score": baseline_index["score"],
        "current_delta": round(delta, digits),
        "comparison": "comparable_same_contract",
    }


def _baseline_index_from_git(
    baseline: Mapping[str, Any], contract: Mapping[str, Any], repository: Path,
) -> dict[str, Any]:
    source = baseline["architecture_evidence_index"]["source"]
    revision = f"{baseline['source_git_commit']}:{source['path']}"
    try:
        result = subprocess.run(
            ["git", "show", revision], cwd=repository, capture_output=True,
            text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReadinessError("unable to reconstruct readiness baseline commit") from exc
    if result.returncode != 0:
        raise ReadinessError(
            "readiness baseline commit is unavailable; fetch repository history"
        )
    return _taxonomy_evidence_index_from_text(
        result.stdout, str(source["path"]), contract,
    )


def _ledger_rows(db_path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    if not Path(db_path).is_file():
        raise ReadinessError("memory database does not exist")
    provider = SqliteProvider(Path(db_path))
    if not provider.health():
        raise ReadinessError("memory database is unavailable")
    connection: Optional[sqlite3.Connection] = None
    try:
        connection = sqlite3.connect(str(db_path))
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute(
            "SELECT d.route_id, d.ts, d.source, d.features_json, d.layer, "
            "d.rung, d.policy_version, d.classifier_tier, d.trace_json, "
            "o.status, o.cost_estimate, o.latency_ms, o.user_retried, "
            "o.verified_success, o.quality_score, o.verifier_source, "
            "o.verifier_confidence, o.verified_at, "
            "o.verification_failure_cause "
            "FROM decisions d JOIN outcomes o ON o.route_id=d.route_id"
        )]
        events = {
            (str(row[0]), str(row[1])) for row in connection.execute(
                "SELECT event_id, route_id FROM outcome_events "
                "WHERE event_id LIKE 'live-verification:%'")
        }
        return rows, events
    except sqlite3.Error as exc:
        raise ReadinessError("unable to query memory database") from exc
    finally:
        if connection is not None:
            connection.close()


def _bool(value: Any) -> Optional[bool]:
    return None if value is None else bool(value)


def _trace(value: Any) -> Mapping[str, Any]:
    try:
        result = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return result if isinstance(result, Mapping) else {}


def _classifier_provenance_complete(row: Mapping[str, Any]) -> bool:
    trace = _trace(row.get("trace_json"))
    called = trace.get("called")
    if not isinstance(called, bool):
        return False
    if not called:
        return trace.get("resolver") in {"none", "llm_classifier"}
    if trace.get("abstained") is True:
        return True
    return (
        trace.get("abstained") is False
        and row.get("classifier_tier") in {"trivial", "standard", "hard"}
    )


def _context_to_evidence(
    context: Mapping[str, Any], row: Mapping[str, Any],
    contract: LearningContract,
) -> LearningEvidence:
    verification = context.get("verification")
    if not isinstance(verification, Mapping):
        raise LearningContractError("verification context has no result")
    features = sanitize_predecision_features(row["features_json"], contract)
    return LearningEvidence(
        evidence_id=str(context.get("evidence_id") or ""),
        pair_id=str(context.get("pair_id") or ""),
        task_id=str(context.get("task_id") or ""),
        repository_sha256=str(context.get("repository_sha256") or ""),
        snapshot_commit=str(context.get("snapshot_commit") or ""),
        verification_plan_sha256=str(
            context.get("verification_plan_sha256") or ""),
        features=features,
        decided_at=float(row["ts"]),
        source=str(row["source"]),
        rung=str(context.get("served_rung") or ""),
        model=str(context.get("model") or ""),
        harness=str(context.get("harness") or ""),
        verified_success=_bool(row.get("verified_success")),
        quality_score=row.get("quality_score"),
        verifier_source=str(row.get("verifier_source") or ""),
        verifier_confidence=row.get("verifier_confidence"),
        failure_cause=row.get("verification_failure_cause"),
        cost_estimate=float(row.get("cost_estimate") or 0.0),
        latency_ms=float(row.get("latency_ms") or 0.0),
        user_retried=_bool(row.get("user_retried")),
    )


def _validate_context(
    context: Mapping[str, Any], row: Mapping[str, Any],
    events: set[tuple[str, str]], contract: LearningContract,
) -> LearningEvidence:
    if context.get("schema_version") != "organic-verification-context-v1":
        raise LearningContractError("unknown verification context schema")
    evidence_id = str(context.get("evidence_id") or "")
    route_id = str(context.get("route_id") or "")
    if (evidence_id, route_id) not in events:
        raise LearningContractError("verification context has no matching ledger event")
    if route_id != row["route_id"] or context.get("decision_source") != row["source"]:
        raise LearningContractError("verification context route provenance mismatches")
    if (
        context.get("decision_ts") != row["ts"]
        or context.get("decision_layer") != row["layer"]
        or context.get("policy_version") != row["policy_version"]
        or context.get("selected_rung") != row["rung"]
    ):
        raise LearningContractError("verification context decision provenance mismatches")
    if context.get("learning_contract_version") != contract.contract_version:
        raise LearningContractError("verification context learning contract mismatches")
    if context.get("feature_schema_version") != contract.feature_schema_version:
        raise LearningContractError("verification context feature schema mismatches")
    if context.get("features_sha256") != feature_payload_sha256(
        row["features_json"], contract
    ):
        raise LearningContractError("verification context feature hash mismatches")
    for name in ("repository_sha256", "verification_plan_sha256"):
        if not _HEX_64.fullmatch(str(context.get(name) or "")):
            raise LearningContractError(f"verification context {name} is invalid")
    verification = context.get("verification")
    if not isinstance(verification, Mapping):
        raise LearningContractError("verification context has no result")
    if verification.get("task_success") != _bool(row.get("verified_success")):
        raise LearningContractError("verification result mismatches ledger")
    if verification.get("verifier_source") != row.get("verifier_source"):
        raise LearningContractError("verification source mismatches ledger")
    if verification.get("failure_cause") != row.get("verification_failure_cause"):
        raise LearningContractError("verification failure cause mismatches ledger")
    if (
        verification.get("quality_score") != row.get("quality_score")
        or verification.get("confidence") != row.get("verifier_confidence")
    ):
        raise LearningContractError("verification score or confidence mismatches ledger")
    evidence = _context_to_evidence(context, row, contract)
    return validate_learning_evidence(evidence, contract)


def _label_stratum(evidence: LearningEvidence) -> str:
    if evidence.verified_success is True:
        return "success"
    if (evidence.verified_success is False
            and evidence.failure_cause == "task_capability"):
        return "task_capability_failure"
    return "other_or_unverifiable"


def _label_audit(
    reviews: Sequence[Mapping[str, Any]],
    evidence_by_id: Mapping[str, LearningEvidence],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    expected_schema = policy.get("review_schema_version")
    required_strata = policy.get("required_strata")
    z = policy.get("wilson_z")
    threshold = policy.get("minimum_wilson_lower_bound")
    reviewer_roles = policy.get("reviewer_roles")
    if not isinstance(required_strata, list) or not required_strata:
        raise ReadinessError("label audit policy required_strata is invalid")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           for value in (z, threshold)):
        raise ReadinessError("label audit policy confidence settings are invalid")
    if not isinstance(reviewer_roles, list) or not reviewer_roles or any(
        not isinstance(role, str) or not role for role in reviewer_roles
    ):
        raise ReadinessError("label audit policy reviewer_roles is invalid")

    reviewed: dict[str, Mapping[str, Any]] = {}
    invalid = 0
    for review in reviews:
        evidence_id = str(review.get("evidence_id") or "")
        valid = (
            review.get("schema_version") == expected_schema
            and evidence_id in evidence_by_id
            and evidence_id not in reviewed
            and isinstance(review.get("verifier_result_agrees"), bool)
            and isinstance(review.get("failure_cause_agrees"), bool)
            and review.get("reviewer_role") in reviewer_roles
            and isinstance(review.get("reviewed_at"), (int, float))
            and not isinstance(review.get("reviewed_at"), bool)
            and math.isfinite(float(review.get("reviewed_at")))
        )
        if not valid:
            invalid += 1
            continue
        reviewed[evidence_id] = review

    strata: dict[str, dict[str, Any]] = {}
    for stratum in required_strata:
        items = [
            (evidence_id, review) for evidence_id, review in reviewed.items()
            if _label_stratum(evidence_by_id[evidence_id]) == stratum
        ]
        agreements = sum(
            bool(review["verifier_result_agrees"])
            and bool(review["failure_cause_agrees"])
            for _, review in items
        )
        lower = wilson_lower_bound(agreements, len(items), float(z))
        strata[stratum] = {
            "reviewed": len(items),
            "agreements": agreements,
            "agreement_rate": agreements / len(items) if items else 0.0,
            "wilson_lower_bound": lower,
            "passes": lower >= float(threshold),
        }
    return {
        "policy_version": policy.get("policy_version"),
        "confidence_level": policy.get("confidence_level"),
        "minimum_wilson_lower_bound": float(threshold),
        "valid_reviews": len(reviewed),
        "invalid_reviews": invalid,
        "strata": strata,
        "passes": invalid == 0 and all(item["passes"] for item in strata.values()),
    }


def _valid_pair_ids(
    evidence: Sequence[LearningEvidence], contract,
) -> set[str]:
    groups: dict[str, list[LearningEvidence]] = defaultdict(list)
    for item in evidence:
        groups[item.pair_id].append(item)
    local = set(contract.pair_schema["local_rungs"])
    frontier = set(contract.pair_schema["frontier_rungs"])
    valid: set[str] = set()
    for group in groups.values():
        if not any(item.rung in local for item in group):
            continue
        if not any(item.rung in frontier for item in group):
            continue
        try:
            validate_counterfactual_pairs(group, contract)
        except LearningContractError:
            continue
        valid.add(group[0].pair_id)
    return valid


def _snapshot_id(report: Mapping[str, Any]) -> str:
    state = {
        key: report[key]
        for key in (
            "schema_version",
            "score_contract",
            "baseline",
            "production_readiness",
            "blockers",
            "architecture_evidence_index",
            "counts",
            "label_audit",
            "gates",
        )
    }
    return f"readiness:{_canonical_sha256(state)}"


def collect_readiness(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    evidence_path: Path = DEFAULT_EVIDENCE_PATH,
    counterfactual_path: Path = DEFAULT_COUNTERFACTUAL_PATH,
    label_audit_path: Path = DEFAULT_LABEL_AUDIT_PATH,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    audit_policy_path: Path = DEFAULT_AUDIT_POLICY_PATH,
    adr_index_path: Path = DEFAULT_ADR_INDEX_PATH,
    score_contract_path: Path = DEFAULT_SCORE_CONTRACT_PATH,
    score_baseline_path: Path = DEFAULT_SCORE_BASELINE_PATH,
    generated_at: Optional[float] = None,
) -> dict[str, Any]:
    contract = load_learning_contract(contract_path)
    policy = _read_json(audit_policy_path, "label audit policy")
    score_contract = load_score_contract(score_contract_path)
    score_baseline = load_score_baseline(
        score_baseline_path, score_contract_path, contract=score_contract,
    )
    architecture_index = taxonomy_evidence_index(
        adr_index_path, score_contract_path,
    )
    rows, verification_events = _ledger_rows(Path(db_path))
    by_route = {row["route_id"]: row for row in rows}

    counts = Counter()
    eligible_routes: set[str] = set()
    for row in rows:
        counts["decisions"] += 1
        counts[f"source_{row['source']}"] += 1
        counts[f"outcome_{row['status']}"] += 1
        try:
            reasons = decision_eligibility_reasons(
                source=row["source"], layer=row["layer"],
                features=row["features_json"], contract=contract)
        except LearningContractError:
            reasons = ("invalid feature contract",)
        if any("feature" in reason for reason in reasons):
            counts["invalid_feature_rows"] += 1
        if not reasons:
            eligible_routes.add(row["route_id"])
            counts["eligible_middle_decisions"] += 1
            if _classifier_provenance_complete(row):
                counts["eligible_classifier_provenance_complete"] += 1
        if row.get("verifier_source"):
            counts["verified_rows"] += 1
            if str(row.get("verifier_source") or "").startswith("deterministic:"):
                counts["deterministic_verified_rows"] += 1

    contexts = _read_jsonl(Path(evidence_path), "verified evidence")
    context_ids: set[str] = set()
    valid_evidence: list[LearningEvidence] = []
    valid_context_by_id: dict[str, Mapping[str, Any]] = {}
    invalid_contexts = 0
    for context in contexts:
        evidence_id = str(context.get("evidence_id") or "")
        route_id = str(context.get("route_id") or "")
        if not evidence_id or evidence_id in context_ids or route_id not in by_route:
            invalid_contexts += 1
            continue
        context_ids.add(evidence_id)
        try:
            evidence = _validate_context(
                context, by_route[route_id], verification_events, contract)
        except (LearningContractError, TypeError, ValueError, OverflowError):
            invalid_contexts += 1
            continue
        valid_evidence.append(evidence)
        valid_context_by_id[evidence.evidence_id] = context

    cause_clean = [
        item for item in valid_evidence if is_cause_clean_label(item, contract)]
    usable = [
        item for item in cause_clean
        if (
            by_route[str(valid_context_by_id[item.evidence_id]["route_id"])]["status"]
            == "closed_final"
            and str(valid_context_by_id[item.evidence_id]["route_id"])
            in eligible_routes
        )
    ]
    complete_provenance = sum(
        _classifier_provenance_complete(
            by_route[str(valid_context_by_id[item.evidence_id]["route_id"])])
        for item in usable
    )
    organic_pair_ids = _valid_pair_ids(usable, contract)

    counterfactual_rows = _read_jsonl(
        Path(counterfactual_path), "counterfactual evidence")
    counterfactual_ids: set[str] = set()
    valid_counterfactual: list[LearningEvidence] = []
    counterfactual_task_source: dict[str, str] = {}
    invalid_counterfactual = 0
    for record in counterfactual_rows:
        evidence_id = str(record.get("evidence_id") or "")
        if not evidence_id or evidence_id in counterfactual_ids:
            invalid_counterfactual += 1
            continue
        counterfactual_ids.add(evidence_id)
        try:
            evidence = learning_evidence_from_counterfactual_record(
                record, contract)
        except (LearningContractError, TypeError, ValueError, OverflowError):
            invalid_counterfactual += 1
            continue
        valid_counterfactual.append(evidence)
        counterfactual_task_source[evidence.evidence_id] = str(
            record.get("task_source") or "")
    clean_counterfactual = [
        item for item in valid_counterfactual
        if is_cause_clean_label(item, contract)
    ]
    counterfactual_pair_ids = _valid_pair_ids(clean_counterfactual, contract)
    representative_counterfactual_pairs = {
        pair_id for pair_id in counterfactual_pair_ids
        if all(
            counterfactual_task_source[item.evidence_id] == "organic"
            for item in clean_counterfactual if item.pair_id == pair_id
        )
    }
    pair_count = len(organic_pair_ids) + len(counterfactual_pair_ids)
    representative_pair_count = (
        len(organic_pair_ids) + len(representative_counterfactual_pairs))
    reviews = _read_jsonl(Path(label_audit_path), "label audit")
    audit = _label_audit(
        reviews, {item.evidence_id: item for item in valid_evidence}, policy)

    counts.update({
        "verification_contexts": len(contexts),
        "valid_verification_contexts": len(valid_evidence),
        "invalid_verification_contexts": invalid_contexts,
        "cause_clean_organic_labels": len(cause_clean),
        "usable_final_middle_labels": len(usable),
        "usable_classifier_provenance_complete": complete_provenance,
        "valid_same_snapshot_pairs": pair_count,
        "representative_same_snapshot_pairs": representative_pair_count,
        "counterfactual_records": len(counterfactual_rows),
        "valid_counterfactual_records": len(valid_counterfactual),
        "invalid_counterfactual_records": invalid_counterfactual,
    })
    gates = {
        "feature_contract": Gate(
            "pass" if counts["invalid_feature_rows"] == 0 else "blocked",
            f"{counts['invalid_feature_rows']} ledger rows violate the versioned schema",
        ),
        "organic_verifier_labels": Gate(
            "pass" if usable else "blocked",
            f"{len(usable)} cause-clean, finalized, eligible organic labels",
        ),
        "classifier_provenance": Gate(
            ("not_evaluated" if not usable
             else "pass" if complete_provenance == len(usable) else "blocked"),
            f"{complete_provenance}/{len(usable)} usable labels have complete provenance",
        ),
        "label_precision": Gate(
            "pass" if audit["passes"] else "blocked",
            f"{audit['valid_reviews']} valid independent reviews under "
            f"{audit['policy_version']}",
        ),
        "paired_counterfactual_integrity": Gate(
            "pass" if pair_count else "blocked",
            f"{pair_count} valid local/frontier same-snapshot pairs",
        ),
        "representative_pairs": Gate(
            "pass" if representative_pair_count else "blocked",
            f"{representative_pair_count} pairs originate from organic tasks",
        ),
        "learned_router_authority": Gate(
            "blocked",
            "no calibrated artifact has beaten required baselines on clean holdouts",
        ),
    }
    blockers = [name for name, gate in gates.items() if gate.status == "blocked"]
    when = time.time() if generated_at is None else float(generated_at)
    readiness_contract = score_contract["production_readiness"]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "snapshot_id": "",
        "generated_at": datetime.fromtimestamp(when, timezone.utc).isoformat(),
        "score_contract": _score_contract_reference(score_contract),
        "baseline": _baseline_reference(score_baseline, architecture_index),
        "production_readiness": (
            readiness_contract["blocked_gate_status"]
            if blockers else readiness_contract["eligible_status"]
        ),
        "blockers": blockers,
        "architecture_evidence_index": architecture_index,
        "counts": dict(sorted(counts.items())),
        "label_audit": audit,
        "gates": {name: asdict(gate) for name, gate in gates.items()},
    }
    report["snapshot_id"] = _snapshot_id(report)
    return report


def build_markdown(report: Mapping[str, Any]) -> str:
    index = report["architecture_evidence_index"]
    score_contract = report["score_contract"]
    baseline = report["baseline"]
    maturity_scale = ", ".join(
        f"{level.split()[0]}={points:g}"
        for level, points in index["scale"].items()
    )
    delta = float(baseline["current_delta"])
    lines = [
        "# ADRL learning-readiness scorecard",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Snapshot: `{report['snapshot_id']}`",
        "",
        "## Verdict",
        "",
        f"- Production readiness: **{str(report['production_readiness']).upper()}**",
        f"- Architecture evidence index: **{index['score']:.1f}/100**",
        f"- Frozen baseline: **{float(baseline['score']):.1f}/100** "
        f"(`{baseline['id']}` at `{baseline['source_git_commit'][:8]}`)",
        f"- Change from baseline: **{delta:+.1f} points**",
        f"- Scoring contract: `{score_contract['version']}` "
        f"(`sha256:{score_contract['sha256']}`)",
        "- The index summarizes checked-in ADR D-levels; it is not a production-readiness score.",
        "- Scores without this exact contract version and hash are not part of this comparable series.",
        f"- Blocking gates: {', '.join(report['blockers']) or 'none'}",
        "",
        "## Method",
        "",
        f"- Maturity points: {maturity_scale}.",
        "- Points, bucket weights, decision weighting, and rounding come from the versioned scoring contract.",
        "- Under v1, each decision has equal weight inside its bucket and each frozen bucket has weight 1/9.",
        "- Confidence is exposed as each bucket's D-level evidence profile below.",
        "- Production authority is a hard-gate verdict and is never inferred from the arithmetic mean.",
        "- Synthetic pairs may pass integrity checks but cannot pass the organic representative-data gate.",
        "",
        "## Bucket evidence",
        "",
        "| Bucket | Index | Weight | Evidence profile |",
        "|---|---:|---:|---|",
    ]
    for bucket in FROZEN_BUCKETS:
        item = index["buckets"][bucket]
        profile = ", ".join(
            f"{level.split()[0]}={count}"
            for level, count in item["evidence_profile"].items())
        lines.append(
            f"| `{bucket}` | {item['score']:.1f} | {item['weight']:.3f} | {profile} |")
    lines.extend([
        "",
        "## Learning evidence",
        "",
        "| Measure | Count |",
        "|---|---:|",
    ])
    labels = {
        "source_organic": "Organic decisions",
        "outcome_closed_final": "Finalized outcomes",
        "eligible_middle_decisions": "Eligible ambiguous-middle decisions",
        "verified_rows": "Ledger rows with any verification",
        "valid_verification_contexts": "Valid organic verification contexts",
        "cause_clean_organic_labels": "Cause-clean organic verifier labels",
        "usable_final_middle_labels": "Usable finalized middle-band labels",
        "valid_same_snapshot_pairs": "Valid same-snapshot local/frontier pairs",
        "representative_same_snapshot_pairs": "Production-representative pairs",
        "valid_counterfactual_records": "Valid versioned counterfactual records",
        "invalid_counterfactual_records": "Invalid counterfactual records",
        "invalid_verification_contexts": "Invalid verification contexts",
    }
    counts = report["counts"]
    for key, label in labels.items():
        lines.append(f"| {label} | {counts.get(key, 0)} |")
    lines.extend([
        "",
        "## Gates",
        "",
        "| Gate | Status | Evidence |",
        "|---|---|---|",
    ])
    for name in GATE_ORDER:
        gate = report["gates"][name]
        lines.append(
            f"| `{name}` | **{str(gate['status']).upper()}** | {gate['evidence']} |")
    audit = report["label_audit"]
    lines.extend([
        "",
        "## Label precision",
        "",
        f"Policy: `{audit['policy_version']}`; {audit['confidence_level']:.0%} Wilson "
        f"lower bound must be at least {audit['minimum_wilson_lower_bound']:.0%} in every required stratum.",
        "",
        "| Stratum | Reviewed | Agreements | Rate | Lower bound | Pass |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for name in LABEL_STRATA_ORDER:
        item = audit["strata"][name]
        lines.append(
            f"| `{name}` | {item['reviewed']} | {item['agreements']} | "
            f"{item['agreement_rate']:.1%} | {item['wilson_lower_bound']:.1%} | "
            f"{'yes' if item['passes'] else 'NO'} |")
    lines.extend([
        "",
        "## Next measured action",
        "",
        "Run the exact-route `router.live_verification` begin/finish flow on eligible organic tasks, independently review both successful and task-capability-failure labels, and run `router.counterfactual` candidates from the same organic snapshot. Training remains blocked until precision, pair-integrity, and representative-data gates pass.",
        "",
    ])
    return "\n".join(lines)


def build_history_entry(
    report: Mapping[str, Any], previous_entry_sha256: Optional[str] = None,
) -> dict[str, Any]:
    """Build the append-only, aggregate milestone record for a score snapshot."""
    if previous_entry_sha256 is not None and not _HEX_64.fullmatch(
        previous_entry_sha256
    ):
        raise ReadinessError("readiness history predecessor hash is invalid")
    entry = {
        "schema_version": HISTORY_ENTRY_SCHEMA_VERSION,
        "snapshot_id": report["snapshot_id"],
        "generated_at": report["generated_at"],
        "previous_entry_sha256": previous_entry_sha256,
        "score_contract": report["score_contract"],
        "baseline": report["baseline"],
        "architecture_evidence_index": report["architecture_evidence_index"],
        "production_readiness": report["production_readiness"],
        "blockers": report["blockers"],
        "gate_statuses": {
            name: gate["status"] for name, gate in report["gates"].items()
        },
        "counts": report["counts"],
    }
    entry["entry_sha256"] = _canonical_sha256(entry)
    return entry


def _stable_history_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in entry.items()
        if key not in {"generated_at", "entry_sha256"}
    }


def _validate_history_entries(
    entries: Sequence[Mapping[str, Any]],
) -> set[str]:
    seen: set[str] = set()
    previous_hash: Optional[str] = None
    for entry in entries:
        if entry.get("schema_version") != HISTORY_ENTRY_SCHEMA_VERSION:
            raise ReadinessError("readiness history contains an unknown schema")
        snapshot_id = str(entry.get("snapshot_id") or "")
        if not snapshot_id or snapshot_id in seen:
            raise ReadinessError("readiness history has a missing or duplicate snapshot_id")
        seen.add(snapshot_id)
        if entry.get("previous_entry_sha256") != previous_hash:
            raise ReadinessError("readiness history hash chain is broken")
        entry_hash = str(entry.get("entry_sha256") or "")
        payload = {
            key: value for key, value in entry.items() if key != "entry_sha256"
        }
        if not _HEX_64.fullmatch(entry_hash) or entry_hash != _canonical_sha256(
            payload
        ):
            raise ReadinessError("readiness history entry content hash mismatches")
        previous_hash = entry_hash
    return seen


def record_history_snapshot(
    report: Mapping[str, Any], path: Path = DEFAULT_HISTORY_PATH,
) -> bool:
    """Append a new evidence state once; repeated measurements are idempotent."""
    history_path = Path(path)
    entries = _read_jsonl(history_path, "readiness history")
    _validate_history_entries(entries)
    for entry in entries:
        if entry["snapshot_id"] == report["snapshot_id"]:
            expected = build_history_entry(
                report, entry.get("previous_entry_sha256"),
            )
            if _stable_history_entry(entry) != _stable_history_entry(expected):
                raise ReadinessError(
                    "readiness history snapshot_id maps to different evidence"
                )
            return False
    previous_hash = str(entries[-1]["entry_sha256"]) if entries else None
    new_entry = build_history_entry(report, previous_hash)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(new_entry) + "\n")
    return True


def _write_if_changed(path: Path, text: str) -> None:
    output_path = Path(path)
    if output_path.is_file():
        try:
            if output_path.read_text(encoding="utf-8") == text:
                return
        except (OSError, UnicodeError) as exc:
            raise ReadinessError(f"unable to read output {output_path}") from exc
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ReadinessError(f"unable to write output {output_path}") from exc


def _reuse_existing_snapshot_time(
    report: dict[str, Any], path: Path,
) -> dict[str, Any]:
    if not Path(path).is_file():
        return report
    existing = _read_json(path, "persisted readiness report")
    if existing.get("snapshot_id") == report["snapshot_id"]:
        generated_at = existing.get("generated_at")
        if isinstance(generated_at, str) and generated_at:
            report["generated_at"] = generated_at
    return report


def validate_persisted_score_artifacts(
    *,
    score_contract_path: Path = DEFAULT_SCORE_CONTRACT_PATH,
    score_baseline_path: Path = DEFAULT_SCORE_BASELINE_PATH,
    adr_index_path: Path = DEFAULT_ADR_INDEX_PATH,
    json_report_path: Path = DEFAULT_JSON_REPORT_PATH,
    markdown_report_path: Path = DEFAULT_MARKDOWN_REPORT_PATH,
    history_path: Path = DEFAULT_HISTORY_PATH,
    repository_path: Path = ROOT,
) -> dict[str, Any]:
    """Validate persisted baseline/current/history without requiring private data."""
    contract = load_score_contract(score_contract_path)
    baseline = load_score_baseline(
        score_baseline_path, score_contract_path, contract=contract,
    )
    reconstructed_baseline = _baseline_index_from_git(
        baseline, contract, Path(repository_path),
    )
    if baseline["architecture_evidence_index"] != reconstructed_baseline:
        raise ReadinessError(
            "persisted readiness baseline does not match its source commit"
        )
    expected_index = taxonomy_evidence_index(adr_index_path, score_contract_path)
    current = dict(_read_json(json_report_path, "persisted readiness report"))
    if current.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ReadinessError("persisted readiness report schema mismatches")
    if current.get("score_contract") != _score_contract_reference(contract):
        raise ReadinessError("persisted readiness report score contract mismatches")
    if current.get("architecture_evidence_index") != expected_index:
        raise ReadinessError("persisted architecture score is stale or was modified")
    if current.get("baseline") != _baseline_reference(baseline, expected_index):
        raise ReadinessError("persisted readiness baseline comparison mismatches")
    try:
        expected_snapshot_id = _snapshot_id(current)
    except (KeyError, TypeError) as exc:
        raise ReadinessError("persisted readiness report is incomplete") from exc
    if current.get("snapshot_id") != expected_snapshot_id:
        raise ReadinessError("persisted readiness snapshot identity mismatches")

    try:
        markdown = Path(markdown_report_path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ReadinessError("unable to read persisted readiness Markdown") from exc
    expected_markdown = build_markdown(current)
    if markdown != expected_markdown:
        raise ReadinessError("persisted readiness Markdown and JSON disagree")

    entries = _read_jsonl(Path(history_path), "readiness history")
    _validate_history_entries(entries)
    matches = [
        entry for entry in entries
        if entry["snapshot_id"] == current["snapshot_id"]
    ]
    if len(matches) != 1:
        raise ReadinessError("current readiness snapshot is not recorded exactly once")
    current_entry = matches[0]
    expected_entry = build_history_entry(
        current, current_entry.get("previous_entry_sha256"),
    )
    if _stable_history_entry(current_entry) != _stable_history_entry(expected_entry):
        raise ReadinessError("current readiness history entry was modified")
    if entries[-1]["snapshot_id"] != current["snapshot_id"]:
        raise ReadinessError("persisted readiness report is not the latest history state")
    return {
        "contract_version": contract["contract_version"],
        "contract_sha256": contract["sha256"],
        "baseline_id": baseline["baseline_id"],
        "baseline_score": baseline["architecture_evidence_index"]["score"],
        "current_score": expected_index["score"],
        "current_delta": current["baseline"]["current_delta"],
        "snapshot_id": current["snapshot_id"],
        "history_entries": len(entries),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE_PATH)
    parser.add_argument(
        "--counterfactual", type=Path, default=DEFAULT_COUNTERFACTUAL_PATH)
    parser.add_argument("--label-audit", type=Path, default=DEFAULT_LABEL_AUDIT_PATH)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--audit-policy", type=Path, default=DEFAULT_AUDIT_POLICY_PATH)
    parser.add_argument("--adr-index", type=Path, default=DEFAULT_ADR_INDEX_PATH)
    parser.add_argument(
        "--score-contract", type=Path, default=DEFAULT_SCORE_CONTRACT_PATH)
    parser.add_argument(
        "--score-baseline", type=Path, default=DEFAULT_SCORE_BASELINE_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--record-history", type=Path)
    parser.add_argument(
        "--persist", action="store_true",
        help="refresh the canonical Markdown, JSON, and history artifacts",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.persist and any((
        args.output, args.json_output, args.record_history, args.json,
    )):
        parser.error(
            "--persist cannot be combined with --output, --json-output, "
            "--record-history, or --json"
        )
    try:
        report = collect_readiness(
            db_path=args.db, evidence_path=args.evidence,
            counterfactual_path=args.counterfactual,
            label_audit_path=args.label_audit, contract_path=args.contract,
            audit_policy_path=args.audit_policy, adr_index_path=args.adr_index,
            score_contract_path=args.score_contract,
            score_baseline_path=args.score_baseline,
        )
    except (ReadinessError, LearningContractError) as exc:
        parser.error(str(exc))
    if args.persist:
        report = _reuse_existing_snapshot_time(report, DEFAULT_JSON_REPORT_PATH)
        markdown_output = DEFAULT_MARKDOWN_REPORT_PATH
        json_output = DEFAULT_JSON_REPORT_PATH
        history_output = DEFAULT_HISTORY_PATH
    else:
        markdown_output = args.output
        json_output = args.json_output
        history_output = args.record_history

    rendered_json = json.dumps(report, indent=2, sort_keys=True) + "\n"
    rendered_markdown = build_markdown(report)
    if not rendered_markdown.endswith("\n"):
        rendered_markdown += "\n"
    try:
        if markdown_output:
            primary = rendered_json if args.json else rendered_markdown
            _write_if_changed(markdown_output, primary)
        if json_output:
            _write_if_changed(json_output, rendered_json)
        if history_output:
            record_history_snapshot(report, history_output)
    except ReadinessError as exc:
        parser.error(str(exc))

    if not any((markdown_output, json_output, history_output)):
        print(rendered_json if args.json else rendered_markdown, end="")
    elif args.persist:
        print(
            f"readiness snapshot {report['snapshot_id']} persisted: "
            f"{report['architecture_evidence_index']['score']:.1f}/100, "
            f"{report['baseline']['current_delta']:+.1f} from baseline, "
            f"production {report['production_readiness']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
