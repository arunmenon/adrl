"""Exact-route organic verification bridge and evidence provenance."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from router.features import extract
from router.live_verification import (
    LiveVerificationError,
    begin_verification_job,
    finish_verification_job,
)
from router.memory_facade import RouterMemory
from router.memory_sqlite import SqliteProvider
from router.policy import Route
from router.verifier import VerificationPlanError, verification_plan_from_mapping


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True, text=True,
        check=True,
    )
    return result.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "app.py").write_text("VALUE = 0\n", encoding="utf-8")
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "tests@example.invalid")
    _git(repository, "config", "user.name", "ADRL Tests")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "baseline")
    return repository, _git(repository, "rev-parse", "HEAD")


def _plan(tmp_path: Path) -> Path:
    script = (
        "from pathlib import Path; "
        "raise SystemExit(0 if Path('app.py').read_text() == "
        "'VALUE = 1\\n' else 1)"
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "plan_version": "value-change-v1",
        "command_checks": [{
            "name": "value",
            "argv": [sys.executable, "-c", script],
            "timeout_s": 30,
        }],
        "require_changes": True,
        "max_changed_files": 1,
    }), encoding="utf-8")
    return path


def _record_route(db_path: Path, route_id: str = "organic-route",
                  source: str = "organic") -> SqliteProvider:
    provider = SqliteProvider(db_path)
    memory = RouterMemory([provider], embedder=False)
    features = extract("Fix VALUE", turn_index=1)
    route = Route("local-code", True, "classifier", score=0.5)
    event, _ = memory.make_decision_event(
        "Fix VALUE", features, route,
        session_id="session", turn_index=1, source=source,
        route_id=route_id, classifier_tier="standard",
        decision_trace={
            "resolver": "llm_classifier", "called": True,
            "abstained": False, "tier": "standard",
        },
    )
    assert provider.record_decision(event) == route_id
    return provider


def test_begin_finish_attaches_exact_route_and_writes_private_context(tmp_path):
    repository, commit = _repository(tmp_path)
    db_path = tmp_path / "memory.db"
    _record_route(db_path)
    jobs = tmp_path / "jobs"
    evidence_path = tmp_path / "verified.jsonl"
    job = begin_verification_job(
        route_id="organic-route",
        task_id="value-fix-1",
        workspace=repository,
        plan_path=_plan(tmp_path),
        served_rung="local-code",
        model="local-model-v1",
        harness="codex",
        db_path=db_path,
        jobs_dir=jobs,
        evidence_path=evidence_path,
    )

    assert job["baseline_commit"] == commit
    assert job["baseline_results"][0]["status"] == "fail"
    (repository / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    evidence = finish_verification_job(job_id=job["job_id"], jobs_dir=jobs)

    assert evidence["route_id"] == "organic-route"
    assert evidence["verification"]["task_success"] is True
    assert evidence["classifier_provenance"]["tier"] == "standard"
    assert evidence["classifier_provenance"]["trace"]["called"] is True
    assert "workspace" not in evidence
    assert "prompt" not in json.dumps(evidence)
    assert evidence_path.stat().st_mode & 0o777 == 0o600

    connection = sqlite3.connect(db_path)
    assert connection.execute(
        "SELECT verified_success, verifier_source FROM outcomes WHERE route_id=?",
        ("organic-route",),
    ).fetchone() == (1, "deterministic:v1")
    assert connection.execute(
        "SELECT COUNT(*) FROM outcome_events WHERE event_id=?",
        (f"live-verification:{job['job_id']}",),
    ).fetchone()[0] == 1
    connection.close()

    repeated = finish_verification_job(job_id=job["job_id"], jobs_dir=jobs)
    assert repeated == evidence
    assert len(evidence_path.read_text().splitlines()) == 1


def test_finish_recovers_when_ledger_attach_precedes_sidecar_write(
    tmp_path, monkeypatch,
):
    repository, _ = _repository(tmp_path)
    db_path = tmp_path / "memory.db"
    _record_route(db_path, "recover-route")
    jobs = tmp_path / "jobs"
    evidence_path = tmp_path / "verified.jsonl"
    job = begin_verification_job(
        route_id="recover-route", task_id="recover-1", workspace=repository,
        plan_path=_plan(tmp_path), served_rung="local-code",
        model="provider/local-model-v1", harness="codex", db_path=db_path,
        jobs_dir=jobs, evidence_path=evidence_path,
    )
    (repository / "app.py").write_text("VALUE = 1\n", encoding="utf-8")

    from router import live_verification
    real_append = live_verification._append_evidence_once

    def fail_append(*args, **kwargs):
        raise LiveVerificationError("injected sidecar failure")

    monkeypatch.setattr(live_verification, "_append_evidence_once", fail_append)
    with pytest.raises(LiveVerificationError, match="injected sidecar failure"):
        finish_verification_job(job_id=job["job_id"], jobs_dir=jobs)

    connection = sqlite3.connect(db_path)
    assert connection.execute(
        "SELECT verified_success FROM outcomes WHERE route_id='recover-route'"
    ).fetchone() == (1,)
    connection.close()
    persisted = json.loads(Path(job["job_path"]).read_text())
    assert persisted["status"] == "verified"
    assert persisted["pending_evidence"]["verification"]["task_success"] is True

    # Recovery uses the journaled result. It must not re-run verification after
    # the workspace changes again.
    (repository / "app.py").write_text("VALUE = 0\n", encoding="utf-8")
    monkeypatch.setattr(live_verification, "_append_evidence_once", real_append)
    recovered = finish_verification_job(job_id=job["job_id"], jobs_dir=jobs)

    assert recovered["verification"]["task_success"] is True
    assert len(evidence_path.read_text().splitlines()) == 1
    connection = sqlite3.connect(db_path)
    assert connection.execute(
        "SELECT COUNT(*) FROM outcome_events WHERE event_id=?",
        (f"live-verification:{job['job_id']}",),
    ).fetchone()[0] == 1
    connection.close()


def test_begin_rejects_dirty_workspace_nonorganic_route_and_existing_label(tmp_path):
    repository, _ = _repository(tmp_path)
    plan = _plan(tmp_path)
    jobs = tmp_path / "jobs"
    evidence = tmp_path / "evidence.jsonl"

    dirty_db = tmp_path / "dirty.db"
    _record_route(dirty_db, "dirty-route")
    (repository / "scratch.py").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(LiveVerificationError, match="clean Git workspace"):
        begin_verification_job(
            route_id="dirty-route", task_id="dirty", workspace=repository,
            plan_path=plan, served_rung="local-code", model="local-v1",
            harness="codex", db_path=dirty_db, jobs_dir=jobs,
            evidence_path=evidence,
        )
    (repository / "scratch.py").unlink()

    simulator_db = tmp_path / "simulator.db"
    _record_route(simulator_db, "sim-route", source="simulator")
    with pytest.raises(LiveVerificationError, match="organic routes only"):
        begin_verification_job(
            route_id="sim-route", task_id="sim", workspace=repository,
            plan_path=plan, served_rung="local-code", model="local-v1",
            harness="codex", db_path=simulator_db, jobs_dir=jobs,
            evidence_path=evidence,
        )

    verified_db = tmp_path / "verified.db"
    provider = _record_route(verified_db, "verified-route")
    from router.memory_ports import VerifiedOutcome
    assert provider.attach_verification(
        "verified-route", VerifiedOutcome(
            task_success=True, quality_score=1.0,
            verifier_source="deterministic:v1", confidence=1.0,
        ))
    with pytest.raises(LiveVerificationError, match="already has verifier"):
        begin_verification_job(
            route_id="verified-route", task_id="verified", workspace=repository,
            plan_path=plan, served_rung="local-code", model="local-v1",
            harness="codex", db_path=verified_db, jobs_dir=jobs,
            evidence_path=evidence,
        )


def test_serialized_verification_plan_is_argv_only_strict_and_nonvacuous():
    with pytest.raises(VerificationPlanError, match="required functional check"):
        verification_plan_from_mapping({"require_changes": True})
    with pytest.raises(VerificationPlanError, match="argv"):
        verification_plan_from_mapping({
            "command_checks": [{"name": "tests", "argv": "pytest -q"}],
        })
    with pytest.raises(VerificationPlanError, match="inside the workspace"):
        verification_plan_from_mapping({
            "command_checks": [{
                "name": "tests", "argv": ["pytest"], "cwd": "../other",
            }],
        })
    with pytest.raises(VerificationPlanError, match="cannot remove"):
        verification_plan_from_mapping({
            "command_checks": [{"name": "tests", "argv": ["pytest"]}],
            "forbidden_path_globs": ["data/**"],
        })
