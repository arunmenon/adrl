"""Bind deterministic repository verification to one organic routing outcome.

This is an explicit two-phase operator/harness bridge, not an HTTP post-call
hook. ``begin`` runs before tools edit a clean Git workspace and records the
baseline plus an argv-only verification plan. ``finish`` verifies the final
workspace and enriches exactly the bound ``route_id``. No command is accepted
from request/model content, and no route is inferred by session or time.

CLI:
    PYTHONPATH=src .venv/bin/python -m router.live_verification begin ...
    PYTHONPATH=src .venv/bin/python -m router.live_verification finish ...
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .learning_contract import (
    LearningContractError,
    feature_payload_sha256,
    load_learning_contract,
)
from .memory_ports import VerifiedOutcome
from .memory_sqlite import DEFAULT_DB_PATH, SqliteProvider
from .verifier import (
    CheckResult,
    CheckStatus,
    VerificationPlan,
    VerificationPlanError,
    run_command_checks,
    run_content_check,
    run_repository_content_check,
    repository_identity_sha256,
    verification_plan_from_mapping,
    verification_plan_to_mapping,
    verify_candidate,
)


JOB_SCHEMA_VERSION = "live-verification-job-v1"
EVIDENCE_SCHEMA_VERSION = "organic-verification-context-v1"
DEFAULT_JOBS_DIR = Path("data/verification-jobs")
DEFAULT_EVIDENCE_PATH = Path("data/verified-evidence.jsonl")
_OPAQUE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/+~-]{0,191}$")
_JOB_ID = re.compile(r"^[0-9]{13}-[0-9a-f]{12}$")


class LiveVerificationError(RuntimeError):
    """The live evidence job cannot be created or safely completed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _git(workspace: Path, *args: str) -> subprocess.CompletedProcess:
    command = ["git", *args]
    try:
        return subprocess.run(
            command, cwd=workspace, capture_output=True, text=True,
            timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return subprocess.CompletedProcess(command, 127, "", "")


def _repository_root(workspace: Path) -> tuple[Path, str]:
    workspace = Path(workspace).expanduser().resolve()
    if not workspace.is_dir():
        raise LiveVerificationError("workspace does not exist")
    root_result = _git(workspace, "rev-parse", "--show-toplevel")
    head_result = _git(workspace, "rev-parse", "HEAD")
    if root_result.returncode != 0 or head_result.returncode != 0:
        raise LiveVerificationError("workspace must be a Git repository with HEAD")
    root = Path(root_result.stdout.strip()).resolve()
    commit = head_result.stdout.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{7,64}", commit):
        raise LiveVerificationError("workspace HEAD is not a valid Git object id")
    return root, commit


def _require_clean(workspace: Path) -> None:
    result = _git(
        workspace, "status", "--porcelain=v1", "--untracked-files=all")
    if result.returncode != 0 or result.stdout.strip():
        raise LiveVerificationError(
            "verification begin requires a clean Git workspace")


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LiveVerificationError(f"unable to read {name}") from exc
    if not isinstance(value, Mapping):
        raise LiveVerificationError(f"{name} must contain a JSON object")
    return value


def _write_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        encoded = (_canonical_json(payload) + "\n").encode("utf-8")
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            if os.path.exists(temporary):
                os.unlink(temporary)
        except OSError:
            pass


def _read_evidence(path: Path, evidence_id: str) -> Optional[Mapping[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, Mapping) and item.get("evidence_id") == evidence_id:
                    return item
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LiveVerificationError("verified evidence file is unreadable") from exc
    return None


def _append_evidence_once(path: Path, payload: Mapping[str, Any]) -> None:
    evidence_id = str(payload.get("evidence_id") or "")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        with os.fdopen(descriptor, "r+b", closefd=False) as stream:
            try:
                existing_records = [
                    json.loads(line) for line in stream.read().decode("utf-8").splitlines()
                    if line.strip()
                ]
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise LiveVerificationError(
                    "verified evidence file is unreadable") from exc
            existing = next((
                item for item in existing_records
                if isinstance(item, Mapping) and item.get("evidence_id") == evidence_id
            ), None)
            if existing is not None:
                if _canonical_json(existing) != _canonical_json(payload):
                    raise LiveVerificationError(
                        "evidence ID already has different content")
                return
            stream.seek(0, os.SEEK_END)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise LiveVerificationError("unable to append verified evidence") from exc
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)


def _route_record(db_path: Path, route_id: str) -> dict[str, Any]:
    if not Path(db_path).is_file():
        raise LiveVerificationError("memory database does not exist")
    connection: Optional[sqlite3.Connection] = None
    try:
        connection = sqlite3.connect(str(db_path))
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT d.route_id, d.ts, d.source, d.instr_sha256, "
            "d.features_json, d.layer, d.rung, d.policy_version, "
            "d.classifier_tier, d.trace_json, o.status, o.verifier_source, "
            "o.verified_success, o.verification_failure_cause "
            "FROM decisions d JOIN outcomes o ON o.route_id=d.route_id "
            "WHERE d.route_id=?",
            (route_id,),
        ).fetchone()
    except sqlite3.Error as exc:
        raise LiveVerificationError("unable to read route from memory database") from exc
    finally:
        if connection is not None:
            connection.close()
    if row is None:
        raise LiveVerificationError("route_id does not exist in the memory ledger")
    return dict(row)


def _route_binding_sha256(route: Mapping[str, Any]) -> str:
    return _sha256({
        key: route.get(key) for key in (
            "route_id", "ts", "source", "instr_sha256", "features_json",
            "layer", "rung", "policy_version", "classifier_tier", "trace_json",
        )
    })


def _outcome_event_route(db_path: Path, event_id: str) -> Optional[str]:
    connection: Optional[sqlite3.Connection] = None
    try:
        connection = sqlite3.connect(str(db_path))
        row = connection.execute(
            "SELECT route_id FROM outcome_events WHERE event_id=?", (event_id,)
        ).fetchone()
        return None if row is None else str(row[0])
    except sqlite3.Error as exc:
        raise LiveVerificationError("unable to inspect verification event") from exc
    finally:
        if connection is not None:
            connection.close()


def _load_plan(path: Path) -> tuple[str, VerificationPlan, str]:
    raw = _read_json(Path(path), "verification plan")
    version = raw.get("plan_version")
    if not isinstance(version, str) or not version.strip():
        raise LiveVerificationError("verification plan requires plan_version")
    plan_payload = {key: value for key, value in raw.items() if key != "plan_version"}
    try:
        plan = verification_plan_from_mapping(plan_payload)
    except VerificationPlanError as exc:
        raise LiveVerificationError(str(exc)) from exc
    digest = _sha256({
        "plan_version": version,
        "plan": verification_plan_to_mapping(plan),
    })
    return version, plan, digest


def _baseline_results(workspace: Path, plan: VerificationPlan) -> tuple[CheckResult, ...]:
    return (
        *run_command_checks(workspace, plan.command_checks),
        *(run_content_check(workspace, check) for check in plan.content_checks),
        *(run_repository_content_check(workspace, check)
          for check in plan.repository_checks),
    )


def _assert_baseline_is_usable(results: Sequence[CheckResult]) -> None:
    unavailable = [
        result.name for result in results
        if result.required and result.status in {
            CheckStatus.UNAVAILABLE.value,
            CheckStatus.TIMEOUT.value,
            CheckStatus.ERROR.value,
        }
    ]
    if unavailable:
        raise LiveVerificationError(
            "required baseline checks were not executable: " + ", ".join(unavailable))


def _opaque(value: str, name: str) -> str:
    if not isinstance(value, str) or not _OPAQUE_ID.fullmatch(value):
        raise LiveVerificationError(
            f"{name} must be an opaque 1-192 character identifier")
    return value


def begin_verification_job(
    *,
    route_id: str,
    task_id: str,
    workspace: Path,
    plan_path: Path,
    served_rung: str,
    model: str,
    harness: str,
    db_path: Path = DEFAULT_DB_PATH,
    jobs_dir: Path = DEFAULT_JOBS_DIR,
    evidence_path: Path = DEFAULT_EVIDENCE_PATH,
    pair_id: Optional[str] = None,
) -> dict[str, Any]:
    """Create an exact-route verification job before repository edits begin."""
    route_id = _opaque(route_id, "route_id")
    task_id = _opaque(task_id, "task_id")
    pair_id = _opaque(pair_id, "pair_id") if pair_id else route_id
    served_rung = _opaque(served_rung, "served_rung")
    model = _opaque(model, "model")
    harness = _opaque(harness, "harness")
    db_path = Path(db_path).expanduser().resolve()
    jobs_dir = Path(jobs_dir).expanduser().resolve()
    evidence_path = Path(evidence_path).expanduser().resolve()

    provider = SqliteProvider(db_path)
    if not provider.health():
        raise LiveVerificationError("memory database is unavailable")
    route = _route_record(db_path, route_id)
    if route["source"] != "organic":
        raise LiveVerificationError("live verification accepts organic routes only")
    if route.get("verifier_source"):
        raise LiveVerificationError("route already has verifier evidence")
    try:
        features_sha256 = feature_payload_sha256(route["features_json"])
    except LearningContractError as exc:
        raise LiveVerificationError(
            "route features do not satisfy the learning contract") from exc

    root, baseline_commit = _repository_root(workspace)
    _require_clean(root)
    plan_version, plan, plan_sha256 = _load_plan(plan_path)
    baseline_results = _baseline_results(root, plan)
    _assert_baseline_is_usable(baseline_results)
    _require_clean(root)

    job_id = f"{int(time.time() * 1000):013d}-{uuid.uuid4().hex[:12]}"
    job = {
        "schema_version": JOB_SCHEMA_VERSION,
        "job_id": job_id,
        "status": "begun",
        "created_at": time.time(),
        "route_id": route_id,
        "route_binding_sha256": _route_binding_sha256(route),
        "task_id": task_id,
        "pair_id": pair_id,
        "workspace": str(root),
        "repository_sha256": repository_identity_sha256(root, baseline_commit),
        "baseline_commit": baseline_commit,
        "served_rung": served_rung,
        "selected_rung": route["rung"],
        "model": model,
        "harness": harness,
        "db_path": str(db_path),
        "evidence_path": str(evidence_path),
        "plan_version": plan_version,
        "plan_sha256": plan_sha256,
        "verification_plan": verification_plan_to_mapping(plan),
        "baseline_results": [asdict(result) for result in baseline_results],
        "feature_schema_version": load_learning_contract().feature_schema_version,
        "features_sha256": features_sha256,
    }
    job_path = jobs_dir / f"{job_id}.json"
    _write_private_json(job_path, job)
    return {**job, "job_path": str(job_path)}


def _job_path(jobs_dir: Path, job_id: str) -> Path:
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        raise LiveVerificationError("invalid verification job_id")
    return Path(jobs_dir).expanduser().resolve() / f"{job_id}.json"


def _check_result(value: Mapping[str, Any]) -> CheckResult:
    try:
        return CheckResult(**dict(value))
    except (TypeError, ValueError) as exc:
        raise LiveVerificationError("verification job has invalid baseline results") from exc


def _result_checks(result) -> list[dict[str, Any]]:
    return [
        {
            "name": check.name,
            "status": check.status,
            "required": check.required,
            "returncode": check.returncode,
            "output_sha256": check.output_sha256,
        }
        for check in (
            *result.command_results,
            *result.content_results,
            *result.repository_results,
        )
    ]


def _build_evidence(
    *,
    evidence_id: str,
    route: Mapping[str, Any],
    job: Mapping[str, Any],
    result,
) -> dict[str, Any]:
    try:
        trace = json.loads(route.get("trace_json") or "{}")
    except json.JSONDecodeError:
        trace = {}
    if not isinstance(trace, Mapping):
        trace = {}
    contract = load_learning_contract()
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "evidence_id": evidence_id,
        "route_id": route["route_id"],
        "task_id": job["task_id"],
        "pair_id": job["pair_id"],
        "decision_source": route["source"],
        "decision_ts": route["ts"],
        "decision_layer": route["layer"],
        "policy_version": route["policy_version"],
        "selected_rung": job["selected_rung"],
        "served_rung": job["served_rung"],
        "model": job["model"],
        "harness": job["harness"],
        "repository_sha256": job["repository_sha256"],
        "snapshot_commit": job["baseline_commit"],
        "learning_contract_version": contract.contract_version,
        "feature_schema_version": contract.feature_schema_version,
        "features_sha256": job["features_sha256"],
        "verification_plan_version": job["plan_version"],
        "verification_plan_sha256": job["plan_sha256"],
        "classifier_provenance": {
            "tier": route.get("classifier_tier"),
            "trace": dict(trace),
        },
        "verification": {
            "task_success": result.task_success,
            "quality_score": result.quality_score,
            "confidence": result.confidence,
            "verifier_source": result.verifier_source,
            "failure_cause": result.failure_cause,
            "checks": _result_checks(result),
            "changed_files": len(result.changes.files),
            "insertions": result.changes.insertions,
            "deletions": result.changes.deletions,
            "regressions": list(result.regressions),
            "improvements": list(result.improvements),
        },
    }


def _pending_result(job: Mapping[str, Any]) -> tuple[dict[str, Any], VerifiedOutcome]:
    evidence = job.get("pending_evidence")
    outcome = job.get("pending_verified_outcome")
    if not isinstance(evidence, Mapping) or not isinstance(outcome, Mapping):
        raise LiveVerificationError("verified job is missing its recovery journal")
    if _sha256(evidence) != job.get("pending_evidence_sha256"):
        raise LiveVerificationError("verification recovery journal changed")
    try:
        verified = VerifiedOutcome(**dict(outcome))
    except TypeError as exc:
        raise LiveVerificationError("verification recovery outcome is invalid") from exc
    return dict(evidence), verified


def finish_verification_job(
    *,
    job_id: str,
    jobs_dir: Path = DEFAULT_JOBS_DIR,
) -> dict[str, Any]:
    """Verify the bound workspace and append one idempotent ledger event."""
    path = _job_path(jobs_dir, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return _finish_verification_job_locked(path=path, job_id=job_id)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)


def _finish_verification_job_locked(
    *, path: Path, job_id: str,
) -> dict[str, Any]:
    job = dict(_read_json(path, "verification job"))
    if job.get("schema_version") != JOB_SCHEMA_VERSION or job.get("job_id") != job_id:
        raise LiveVerificationError("verification job schema or identity is invalid")
    evidence_id = f"live-verification:{job_id}"
    evidence_path = Path(str(job.get("evidence_path", ""))).resolve()
    existing = _read_evidence(evidence_path, evidence_id)
    if existing is not None:
        if job.get("status") != "complete":
            job["status"] = "complete"
            job["evidence_id"] = evidence_id
            _write_private_json(path, job)
        return dict(existing)

    workspace = Path(str(job.get("workspace", ""))).resolve()
    root, _ = _repository_root(workspace)
    if root != workspace:
        raise LiveVerificationError("verification workspace identity changed")
    baseline_commit = str(job.get("baseline_commit") or "")
    ancestor = _git(root, "merge-base", "--is-ancestor", baseline_commit, "HEAD")
    if ancestor.returncode != 0:
        raise LiveVerificationError("verification baseline is no longer an ancestor")

    try:
        plan = verification_plan_from_mapping(job.get("verification_plan", {}))
    except VerificationPlanError as exc:
        raise LiveVerificationError("verification job plan is invalid") from exc
    plan_sha256 = _sha256({
        "plan_version": job.get("plan_version"),
        "plan": verification_plan_to_mapping(plan),
    })
    if plan_sha256 != job.get("plan_sha256"):
        raise LiveVerificationError("verification plan changed after begin")

    db_path = Path(str(job.get("db_path", ""))).resolve()
    route = _route_record(db_path, str(job.get("route_id") or ""))
    if _route_binding_sha256(route) != job.get("route_binding_sha256"):
        raise LiveVerificationError("bound route provenance changed")
    if feature_payload_sha256(route["features_json"]) != job.get("features_sha256"):
        raise LiveVerificationError("bound route feature projection changed")
    event_route = _outcome_event_route(db_path, evidence_id)
    if route.get("verifier_source") and event_route != route["route_id"]:
        raise LiveVerificationError("route acquired different verifier evidence")
    if not route.get("verifier_source") and event_route is not None:
        raise LiveVerificationError("verification event projection is inconsistent")

    if job.get("status") in {"verified", "complete"}:
        evidence, verified_outcome = _pending_result(job)
    elif job.get("status") == "begun":
        if route.get("verifier_source") or event_route is not None:
            raise LiveVerificationError("route acquired verifier evidence before journaling")
        baseline_values = job.get("baseline_results")
        if not isinstance(baseline_values, list):
            raise LiveVerificationError("verification job has no baseline results")
        baseline_results = tuple(_check_result(item) for item in baseline_values)
        result = verify_candidate(
            root, baseline_commit, plan, baseline_results=baseline_results)
        verified_outcome = result.as_verified_outcome()
        evidence = _build_evidence(
            evidence_id=evidence_id, route=route, job=job, result=result)
        job["status"] = "verified"
        job["pending_evidence"] = evidence
        job["pending_evidence_sha256"] = _sha256(evidence)
        job["pending_verified_outcome"] = asdict(verified_outcome)
        _write_private_json(path, job)
    else:
        raise LiveVerificationError("verification job has an invalid lifecycle status")

    provider = SqliteProvider(db_path)
    if not provider.attach_verification(
        route["route_id"], verified_outcome,
        event_id=evidence_id, observed_at=verified_outcome.verified_at,
    ):
        raise LiveVerificationError("unable to attach verification to route ledger")
    _append_evidence_once(evidence_path, evidence)
    job["status"] = "complete"
    job["evidence_id"] = evidence_id
    job["completed_at"] = verified_outcome.verified_at
    _write_private_json(path, job)
    return evidence


def _public_begin(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "route_id": job["route_id"],
        "baseline_commit": job["baseline_commit"],
        "plan_version": job["plan_version"],
        "baseline_checks": [
            {"name": item["name"], "status": item["status"]}
            for item in job["baseline_results"]
        ],
        "job_path": job["job_path"],
    }


def _public_finish(evidence: Mapping[str, Any]) -> dict[str, Any]:
    verification = evidence["verification"]
    return {
        "evidence_id": evidence["evidence_id"],
        "route_id": evidence["route_id"],
        "task_success": verification["task_success"],
        "quality_score": verification["quality_score"],
        "failure_cause": verification["failure_cause"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    begin = subparsers.add_parser("begin", help="bind route, baseline, and plan")
    begin.add_argument("--route-id", required=True)
    begin.add_argument("--task-id", required=True)
    begin.add_argument("--pair-id")
    begin.add_argument("--workspace", type=Path, required=True)
    begin.add_argument("--plan", type=Path, required=True)
    begin.add_argument("--served-rung", required=True)
    begin.add_argument("--model", required=True)
    begin.add_argument("--harness", required=True)
    begin.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    begin.add_argument("--jobs-dir", type=Path, default=DEFAULT_JOBS_DIR)
    begin.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE_PATH)

    finish = subparsers.add_parser("finish", help="verify and attach exact route")
    finish.add_argument("--job-id", required=True)
    finish.add_argument("--jobs-dir", type=Path, default=DEFAULT_JOBS_DIR)
    args = parser.parse_args(argv)
    try:
        if args.command == "begin":
            job = begin_verification_job(
                route_id=args.route_id,
                task_id=args.task_id,
                pair_id=args.pair_id,
                workspace=args.workspace,
                plan_path=args.plan,
                served_rung=args.served_rung,
                model=args.model,
                harness=args.harness,
                db_path=args.db,
                jobs_dir=args.jobs_dir,
                evidence_path=args.evidence,
            )
            print(json.dumps(_public_begin(job), sort_keys=True))
        else:
            evidence = finish_verification_job(
                job_id=args.job_id, jobs_dir=args.jobs_dir)
            print(json.dumps(_public_finish(evidence), sort_keys=True))
    except (LiveVerificationError, LearningContractError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
