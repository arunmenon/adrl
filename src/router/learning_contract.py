"""Versioned, leakage-safe contracts for learned routing data and artifacts.

The learned router is allowed to consume only the predecision projection
defined in ``config/learning-contract-v1.json``. Operational outcomes,
existing route choices, classifier verdicts, identifiers, and raw text are
targets or provenance, never model inputs. Dataset splitting keeps paired
attempts together and provides distinct temporal and unseen-repository/task
holdouts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_PATH = ROOT / "config" / "learning-contract-v1.json"
COUNTERFACTUAL_EVIDENCE_SCHEMA_VERSION = "counterfactual-evidence-v1"
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{7,64}$")
_FIELD_TYPES = {"boolean", "integer", "number", "string"}


class LearningContractError(ValueError):
    """The contract or evidence violates a fail-closed learning invariant."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LearningContractError(f"{name} must be an object")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LearningContractError(f"{name} must be a non-empty string")
    return value.strip()


def _string_set(value: Any, name: str, *, allow_empty: bool = False) -> frozenset[str]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise LearningContractError(f"{name} must be a list of non-empty strings")
    if not allow_empty and not value:
        raise LearningContractError(f"{name} cannot be empty")
    return frozenset(value)


@dataclass(frozen=True)
class FeatureField:
    name: str
    kind: str
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    values: tuple[str, ...] = ()


@dataclass(frozen=True)
class LearningContract:
    contract_version: str
    feature_schema_version: str
    feature_fields: tuple[FeatureField, ...]
    dropped_feature_fields: frozenset[str]
    forbidden_feature_fields: frozenset[str]
    decision_sources: frozenset[str]
    decision_layers: frozenset[str]
    exclude_if_true: frozenset[str]
    label_schema: Mapping[str, Any]
    pair_schema: Mapping[str, Any]
    split_policy: Mapping[str, Any]
    artifact_schema: Mapping[str, Any]
    raw: Mapping[str, Any]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.feature_fields)

    @property
    def feature_schema_sha256(self) -> str:
        return _sha256(self.raw["feature_schema"])

    @property
    def contract_sha256(self) -> str:
        return _sha256(self.raw)


def load_learning_contract(
    path: Path = DEFAULT_CONTRACT_PATH,
) -> LearningContract:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LearningContractError(f"unable to load learning contract: {exc}") from exc
    root = _require_mapping(raw, "learning contract")
    contract_version = _require_string(
        root.get("contract_version"), "contract_version")

    feature_schema = _require_mapping(root.get("feature_schema"), "feature_schema")
    feature_version = _require_string(
        feature_schema.get("version"), "feature_schema.version")
    raw_fields = feature_schema.get("fields")
    if not isinstance(raw_fields, list) or not raw_fields:
        raise LearningContractError("feature_schema.fields must be a non-empty list")
    fields: list[FeatureField] = []
    names: set[str] = set()
    for index, item in enumerate(raw_fields):
        spec = _require_mapping(item, f"feature_schema.fields[{index}]")
        name = _require_string(spec.get("name"), f"feature field {index} name")
        kind = _require_string(spec.get("type"), f"feature field {name} type")
        if kind not in _FIELD_TYPES:
            raise LearningContractError(f"unsupported feature type for {name}: {kind}")
        if name in names:
            raise LearningContractError(f"duplicate feature field: {name}")
        names.add(name)
        values = spec.get("values", ())
        if not isinstance(values, (list, tuple)) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise LearningContractError(f"feature {name} values must be strings")
        if kind == "string" and not values:
            raise LearningContractError(f"string feature {name} requires an enum")
        minimum = spec.get("minimum")
        maximum = spec.get("maximum")
        for bound_name, bound in (("minimum", minimum), ("maximum", maximum)):
            if bound is not None and (
                isinstance(bound, bool) or not isinstance(bound, (int, float))
                or not math.isfinite(float(bound))
            ):
                raise LearningContractError(
                    f"feature {name} {bound_name} must be finite numeric")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise LearningContractError(f"feature {name} bounds are reversed")
        fields.append(FeatureField(
            name=name,
            kind=kind,
            minimum=minimum,
            maximum=maximum,
            values=tuple(values),
        ))

    dropped = _string_set(
        feature_schema.get("drop_before_training", ()),
        "feature_schema.drop_before_training", allow_empty=True)
    forbidden = _string_set(
        feature_schema.get("forbidden", ()), "feature_schema.forbidden")
    if names & dropped or names & forbidden or dropped & forbidden:
        raise LearningContractError("feature, dropped, and forbidden fields must be disjoint")

    eligibility = _require_mapping(root.get("eligibility"), "eligibility")
    sources = _string_set(
        eligibility.get("decision_sources", ()), "eligibility.decision_sources")
    layers = _string_set(
        eligibility.get("decision_layers", ()), "eligibility.decision_layers")
    exclude_if_true = _string_set(
        eligibility.get("exclude_if_true", ()), "eligibility.exclude_if_true",
        allow_empty=True)
    if not exclude_if_true.issubset(dropped):
        raise LearningContractError("eligibility exclusions must be dropped fields")

    label_schema = _require_mapping(root.get("label_schema"), "label_schema")
    pair_schema = _require_mapping(root.get("pair_schema"), "pair_schema")
    split_policy = _require_mapping(root.get("split_policy"), "split_policy")
    artifact_schema = _require_mapping(
        root.get("artifact_manifest"), "artifact_manifest")
    for name, section in (
        ("label_schema", label_schema),
        ("pair_schema", pair_schema),
        ("split_policy", split_policy),
        ("artifact_manifest", artifact_schema),
    ):
        _require_string(section.get("version"), f"{name}.version")

    prefixes = label_schema.get("verifier_source_prefixes")
    confidence = label_schema.get("minimum_confidence")
    if not isinstance(prefixes, list) or not prefixes or any(
        not isinstance(prefix, str) or not prefix for prefix in prefixes
    ):
        raise LearningContractError("label verifier prefixes must be non-empty")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise LearningContractError("minimum verifier confidence must be numeric")
    if not 0.0 <= float(confidence) <= 1.0:
        raise LearningContractError("minimum verifier confidence must be in [0,1]")

    required_manifest = artifact_schema.get("required_fields")
    if not isinstance(required_manifest, list) or not required_manifest:
        raise LearningContractError("artifact manifest required_fields cannot be empty")
    return LearningContract(
        contract_version=contract_version,
        feature_schema_version=feature_version,
        feature_fields=tuple(fields),
        dropped_feature_fields=dropped,
        forbidden_feature_fields=forbidden,
        decision_sources=sources,
        decision_layers=layers,
        exclude_if_true=exclude_if_true,
        label_schema=label_schema,
        pair_schema=pair_schema,
        split_policy=split_policy,
        artifact_schema=artifact_schema,
        raw=root,
    )


def _validate_feature_value(field: FeatureField, value: Any) -> Any:
    if field.kind == "boolean":
        if not isinstance(value, bool):
            raise LearningContractError(f"feature {field.name} must be boolean")
    elif field.kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise LearningContractError(f"feature {field.name} must be integer")
    elif field.kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LearningContractError(f"feature {field.name} must be numeric")
        value = float(value)
        if not math.isfinite(value):
            raise LearningContractError(f"feature {field.name} must be finite")
    elif field.kind == "string":
        if not isinstance(value, str) or value not in field.values:
            raise LearningContractError(
                f"feature {field.name} must be one of {', '.join(field.values)}")
    if field.minimum is not None and value < field.minimum:
        raise LearningContractError(f"feature {field.name} is below its minimum")
    if field.maximum is not None and value > field.maximum:
        raise LearningContractError(f"feature {field.name} is above its maximum")
    return value


def parse_feature_payload(value: str | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise LearningContractError("features_json is not valid JSON") from exc
    return _require_mapping(value, "feature payload")


def sanitize_predecision_features(
    value: str | Mapping[str, Any],
    contract: Optional[LearningContract] = None,
) -> dict[str, Any]:
    """Return only versioned predecision fields; reject drift and leakage."""
    contract = contract or load_learning_contract()
    payload = parse_feature_payload(value)
    present = set(payload)
    forbidden = present & contract.forbidden_feature_fields
    if forbidden:
        raise LearningContractError(
            "forbidden postdecision/private feature fields: "
            + ", ".join(sorted(forbidden)))
    allowed_input = set(contract.feature_names) | contract.dropped_feature_fields
    unknown = present - allowed_input
    if unknown:
        raise LearningContractError(
            "unversioned feature fields: " + ", ".join(sorted(unknown)))
    missing = set(contract.feature_names) - present
    if missing:
        raise LearningContractError(
            "missing required predecision features: " + ", ".join(sorted(missing)))
    return {
        field.name: _validate_feature_value(field, payload[field.name])
        for field in contract.feature_fields
    }


def feature_payload_sha256(
    value: str | Mapping[str, Any],
    contract: Optional[LearningContract] = None,
) -> str:
    return _sha256(sanitize_predecision_features(value, contract))


def decision_eligibility_reasons(
    *,
    source: str,
    layer: str,
    features: str | Mapping[str, Any],
    contract: Optional[LearningContract] = None,
) -> tuple[str, ...]:
    """Explain why a decision cannot enter learned-router training."""
    contract = contract or load_learning_contract()
    payload = parse_feature_payload(features)
    reasons: list[str] = []
    if source not in contract.decision_sources:
        reasons.append("ineligible decision source")
    if layer not in contract.decision_layers:
        reasons.append("outside ambiguous routing layer")
    for field in sorted(contract.exclude_if_true):
        if payload.get(field) is True:
            reasons.append(f"hard/semantic gate active: {field}")
    try:
        sanitize_predecision_features(payload, contract)
    except LearningContractError as exc:
        reasons.append(str(exc))
    return tuple(reasons)


@dataclass(frozen=True)
class LearningEvidence:
    evidence_id: str
    pair_id: str
    task_id: str
    repository_sha256: str
    snapshot_commit: str
    verification_plan_sha256: str
    features: Mapping[str, Any]
    decided_at: float
    source: str
    rung: str
    model: str
    harness: str
    verified_success: Optional[bool]
    quality_score: Optional[float]
    verifier_source: str
    verifier_confidence: Optional[float]
    failure_cause: Optional[str]
    cost_estimate: float = 0.0
    latency_ms: float = 0.0
    user_retried: Optional[bool] = None

    @property
    def features_sha256(self) -> str:
        return _sha256(self.features)


def validate_learning_evidence(
    evidence: LearningEvidence,
    contract: Optional[LearningContract] = None,
) -> LearningEvidence:
    contract = contract or load_learning_contract()
    for name in (
        "evidence_id", "pair_id", "task_id", "verification_plan_sha256",
        "source", "rung", "model", "harness", "verifier_source",
    ):
        _require_string(getattr(evidence, name), name)
    if evidence.source not in contract.decision_sources:
        raise LearningContractError("evidence source is not permitted by the contract")
    if not _HEX_64.fullmatch(evidence.repository_sha256):
        raise LearningContractError("repository_sha256 must be 64 lowercase hex chars")
    if not _HEX_64.fullmatch(evidence.verification_plan_sha256):
        raise LearningContractError(
            "verification_plan_sha256 must be 64 lowercase hex chars")
    if not _COMMIT.fullmatch(evidence.snapshot_commit):
        raise LearningContractError("snapshot_commit must be a git object id")
    features = sanitize_predecision_features(evidence.features, contract)
    if features != dict(evidence.features):
        raise LearningContractError("evidence features are not the canonical projection")
    if evidence.verified_success is not None and not isinstance(
        evidence.verified_success, bool
    ):
        raise LearningContractError("verified_success must be boolean or null")
    for name in (
        "decided_at", "cost_estimate", "latency_ms", "quality_score",
        "verifier_confidence",
    ):
        value = getattr(evidence, name)
        if value is None and name in {"quality_score", "verifier_confidence"}:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LearningContractError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise LearningContractError(f"{name} must be finite and non-negative")
    for name in ("quality_score", "verifier_confidence"):
        value = getattr(evidence, name)
        if value is not None and float(value) > 1.0:
            raise LearningContractError(f"{name} must be in [0,1]")
    if evidence.user_retried is not None and not isinstance(evidence.user_retried, bool):
        raise LearningContractError("user_retried must be boolean or null")
    if evidence.failure_cause is not None and not isinstance(
        evidence.failure_cause, str
    ):
        raise LearningContractError("failure_cause must be string or null")
    return evidence


def learning_evidence_from_counterfactual_record(
    value: Mapping[str, Any],
    contract: Optional[LearningContract] = None,
) -> LearningEvidence:
    """Adapt one versioned counterfactual JSONL row to the shared contract."""
    contract = contract or load_learning_contract()
    value = _require_mapping(value, "counterfactual record")
    if value.get("evidence_schema_version") != COUNTERFACTUAL_EVIDENCE_SCHEMA_VERSION:
        raise LearningContractError("unknown counterfactual evidence schema")
    if value.get("learning_contract_version") != contract.contract_version:
        raise LearningContractError("counterfactual learning contract mismatches")
    if value.get("feature_schema_version") != contract.feature_schema_version:
        raise LearningContractError("counterfactual feature schema mismatches")
    _require_string(
        value.get("verification_plan_version"),
        "counterfactual verification_plan_version")
    if value.get("task_source") not in {"organic", "synthetic"}:
        raise LearningContractError("counterfactual task_source is unknown")
    features = sanitize_predecision_features(
        _require_mapping(value.get("features"), "counterfactual features"), contract)
    if value.get("features_sha256") != _sha256(features):
        raise LearningContractError("counterfactual feature hash mismatches")
    candidate = _require_mapping(value.get("candidate"), "counterfactual candidate")
    execution = _require_mapping(value.get("execution"), "counterfactual execution")
    verification = _require_mapping(
        value.get("verification"), "counterfactual verification")
    evidence = LearningEvidence(
        evidence_id=str(value.get("evidence_id") or ""),
        pair_id=str(value.get("pair_id") or ""),
        task_id=str(value.get("task_id") or ""),
        repository_sha256=str(value.get("repository_sha256") or ""),
        snapshot_commit=str(value.get("snapshot_commit") or ""),
        verification_plan_sha256=str(
            value.get("verification_plan_sha256") or ""),
        features=features,
        decided_at=value.get("decided_at"),
        source="counterfactual",
        rung=str(candidate.get("rung") or ""),
        model=str(candidate.get("model") or ""),
        harness=str(candidate.get("harness") or ""),
        verified_success=verification.get("task_success"),
        quality_score=verification.get("quality_score"),
        verifier_source=str(verification.get("verifier_source") or ""),
        verifier_confidence=verification.get("confidence"),
        failure_cause=verification.get("failure_cause"),
        cost_estimate=execution.get("cost_estimate", 0.0),
        latency_ms=execution.get("duration_ms", 0.0),
        user_retried=None,
    )
    return validate_learning_evidence(evidence, contract)


def is_cause_clean_label(
    evidence: LearningEvidence,
    contract: Optional[LearningContract] = None,
) -> bool:
    contract = contract or load_learning_contract()
    prefixes = tuple(contract.label_schema["verifier_source_prefixes"])
    minimum = float(contract.label_schema["minimum_confidence"])
    if evidence.verified_success is None:
        return False
    if evidence.quality_score is None:
        return False
    if not evidence.verifier_source.startswith(prefixes):
        return False
    if evidence.verifier_confidence is None or evidence.verifier_confidence < minimum:
        return False
    if evidence.verified_success:
        return evidence.failure_cause in (None, "")
    clean_failures = set(contract.label_schema["cause_clean_failure_causes"])
    return evidence.failure_cause in clean_failures


def validate_counterfactual_pairs(
    records: Sequence[LearningEvidence],
    contract: Optional[LearningContract] = None,
) -> dict[str, tuple[LearningEvidence, ...]]:
    contract = contract or load_learning_contract()
    grouped: dict[str, list[LearningEvidence]] = {}
    for record in records:
        validate_learning_evidence(record, contract)
        grouped.setdefault(record.pair_id, []).append(record)
    local_rungs = set(contract.pair_schema["local_rungs"])
    frontier_rungs = set(contract.pair_schema["frontier_rungs"])
    result: dict[str, tuple[LearningEvidence, ...]] = {}
    for pair_id, group in grouped.items():
        identities = {(item.rung, item.model, item.harness) for item in group}
        if len(identities) != len(group):
            raise LearningContractError(f"pair {pair_id} repeats a candidate")
        if not any(item.rung in local_rungs for item in group):
            raise LearningContractError(f"pair {pair_id} has no local candidate")
        if not any(item.rung in frontier_rungs for item in group):
            raise LearningContractError(f"pair {pair_id} has no frontier candidate")
        comparisons = {
            "task_id": {item.task_id for item in group},
            "repository_sha256": {item.repository_sha256 for item in group},
            "snapshot_commit": {item.snapshot_commit for item in group},
            "verification_plan_sha256": {
                item.verification_plan_sha256 for item in group},
            "features_sha256": {item.features_sha256 for item in group},
        }
        mismatched = [name for name, values in comparisons.items() if len(values) != 1]
        if mismatched:
            raise LearningContractError(
                f"pair {pair_id} mismatches: {', '.join(mismatched)}")
        result[pair_id] = tuple(group)
    return result


@dataclass(frozen=True)
class EvidenceSplits:
    train: tuple[LearningEvidence, ...]
    temporal_holdout: tuple[LearningEvidence, ...]
    repository_task_holdout: tuple[LearningEvidence, ...]

    def as_dict(self) -> dict[str, tuple[LearningEvidence, ...]]:
        return {
            "train": self.train,
            "temporal_holdout": self.temporal_holdout,
            "repository_task_holdout": self.repository_task_holdout,
        }


def split_learning_evidence(
    records: Sequence[LearningEvidence],
    *,
    temporal_cutoff: float,
    heldout_repositories: Iterable[str] = (),
    heldout_tasks: Iterable[str] = (),
    contract: Optional[LearningContract] = None,
) -> EvidenceSplits:
    """Split pair groups without leaking held-out repositories or tasks."""
    contract = contract or load_learning_contract()
    if isinstance(temporal_cutoff, bool) or not isinstance(
        temporal_cutoff, (int, float)
    ) or not math.isfinite(float(temporal_cutoff)):
        raise LearningContractError("temporal_cutoff must be finite")
    repository_holdout = set(heldout_repositories)
    task_holdout = set(heldout_tasks)
    if not repository_holdout and not task_holdout:
        raise LearningContractError(
            "an explicit unseen repository or task holdout is required")

    grouped: dict[str, list[LearningEvidence]] = {}
    for record in records:
        validate_learning_evidence(record, contract)
        if not is_cause_clean_label(record, contract):
            raise LearningContractError(
                f"evidence {record.evidence_id} is not a cause-clean label")
        grouped.setdefault(record.pair_id or record.evidence_id, []).append(record)
    buckets: dict[str, list[LearningEvidence]] = {
        "train": [],
        "temporal_holdout": [],
        "repository_task_holdout": [],
    }
    for pair_id, group in grouped.items():
        repositories = {item.repository_sha256 for item in group}
        tasks = {item.task_id for item in group}
        if len(repositories) != 1 or len(tasks) != 1:
            raise LearningContractError(
                f"split group {pair_id} crosses repositories or tasks")
        repository = next(iter(repositories))
        task = next(iter(tasks))
        if repository in repository_holdout or task in task_holdout:
            split = "repository_task_holdout"
        else:
            before = [item.decided_at < temporal_cutoff for item in group]
            if any(before) and not all(before):
                raise LearningContractError(
                    f"pair {pair_id} crosses the temporal cutoff")
            split = "train" if all(before) else "temporal_holdout"
        buckets[split].extend(group)

    unseen = buckets["repository_task_holdout"]
    seen_repository = {
        item.repository_sha256
        for name in ("train", "temporal_holdout") for item in buckets[name]
    }
    seen_tasks = {
        item.task_id
        for name in ("train", "temporal_holdout") for item in buckets[name]
    }
    if seen_repository & {item.repository_sha256 for item in unseen}:
        raise LearningContractError("repository holdout leaked into a seen split")
    if seen_tasks & {item.task_id for item in unseen}:
        raise LearningContractError("task holdout leaked into a seen split")
    return EvidenceSplits(**{
        name: tuple(sorted(items, key=lambda item: item.evidence_id))
        for name, items in buckets.items()
    })


def dataset_snapshot_sha256(records: Sequence[LearningEvidence]) -> str:
    payload = [
        asdict(record) for record in sorted(records, key=lambda item: item.evidence_id)
    ]
    return _sha256(payload)


def build_dataset_manifest(
    splits: EvidenceSplits,
    *,
    temporal_cutoff: float,
    heldout_repositories: Iterable[str],
    heldout_tasks: Iterable[str] = (),
    contract: Optional[LearningContract] = None,
) -> dict[str, Any]:
    contract = contract or load_learning_contract()
    all_records = tuple(
        item for records in splits.as_dict().values() for item in records)
    return {
        "contract_version": contract.contract_version,
        "contract_sha256": contract.contract_sha256,
        "feature_schema_version": contract.feature_schema_version,
        "feature_schema_sha256": contract.feature_schema_sha256,
        "data_snapshot_sha256": dataset_snapshot_sha256(all_records),
        "split_policy_version": contract.split_policy["version"],
        "temporal_cutoff": float(temporal_cutoff),
        "heldout_repositories": sorted(set(heldout_repositories)),
        "heldout_tasks": sorted(set(heldout_tasks)),
        "counts": {
            name: len(records) for name, records in splits.as_dict().items()
        },
    }


def validate_artifact_manifest(
    manifest: Mapping[str, Any],
    contract: Optional[LearningContract] = None,
) -> None:
    contract = contract or load_learning_contract()
    manifest = _require_mapping(manifest, "artifact manifest")
    required = set(contract.artifact_schema["required_fields"])
    missing = required - set(manifest)
    if missing:
        raise LearningContractError(
            "artifact manifest is missing: " + ", ".join(sorted(missing)))
    if manifest["contract_version"] != contract.contract_version:
        raise LearningContractError("artifact contract_version is incompatible")
    if manifest["feature_schema_version"] != contract.feature_schema_version:
        raise LearningContractError("artifact feature_schema_version is incompatible")
    if manifest["feature_schema_sha256"] != contract.feature_schema_sha256:
        raise LearningContractError("artifact feature schema hash is incompatible")
    if not _HEX_64.fullmatch(str(manifest["data_snapshot_sha256"])):
        raise LearningContractError("artifact data snapshot hash is invalid")
    for name in ("objective", "calibration", "thresholds"):
        if not isinstance(manifest[name], Mapping) or not manifest[name]:
            raise LearningContractError(f"artifact {name} must be a non-empty object")
    compatibility = manifest["policy_compatibility"]
    if not isinstance(compatibility, (list, tuple)) or not compatibility or any(
        not isinstance(value, str) or not value for value in compatibility
    ):
        raise LearningContractError(
            "artifact policy_compatibility must be a non-empty list")
