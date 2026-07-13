"""Counterfactual runner: exact snapshots, isolation, records, and fail-safety."""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from router.counterfactual import (
    Candidate,
    ClaudeCliExecutor,
    CounterfactualTask,
    ExecutionResult,
    run_counterfactual,
    verification_plan_from_sandbox,
)
from router.outcomes import FailureCause
from router.memory_sqlite import DecisionEvent, SqliteProvider
from router.verifier import CommandCheck, VerificationPlan
from router.verifier import (
    run_command_checks,
    run_repository_content_check,
    verify_candidate,
)
from simulator.sandbox import make_sandbox


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True, text=True,
        check=True,
    )
    return result.stdout.strip()


def _task(tmp_path: Path) -> tuple[CounterfactualTask, str]:
    repository = tmp_path / "source"
    repository.mkdir()
    (repository / "app.py").write_text("VALUE = 0\n", encoding="utf-8")
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "tests@example.invalid")
    _git(repository, "config", "user.name", "ADRL Tests")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "baseline")
    commit = _git(repository, "rev-parse", "HEAD")
    script = (
        "from pathlib import Path; "
        "raise SystemExit(0 if Path('app.py').read_text() == "
        "'VALUE = 1\\n' else 1)"
    )
    task = CounterfactualTask(
        task_id="fix-value",
        prompt="Change VALUE from zero to one.",
        repository=repository,
        verification=VerificationPlan(command_checks=(
            CommandCheck("value", (sys.executable, "-c", script)),
        )),
        metadata={"suite": "unit"},
    )
    return task, commit


class FakeExecutor:
    def execute(self, task, candidate, workspace):
        if candidate.rung != "cheap-cloud":
            (workspace / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
        return ExecutionResult(
            status="ok", duration_ms=10.0, cost_estimate=0.01,
            input_tokens=20, output_tokens=10,
        )

    def execute_with_budget(self, task, candidate, workspace, *, max_cost_usd):
        if max_cost_usd < 0.01:
            return ExecutionResult(
                status="skipped", duration_ms=0.0,
                failure_cause=FailureCause.POLICY_CONSTRAINT.value,
                detail="fake hard budget exhausted",
            )
        return self.execute(task, candidate, workspace)


def test_candidates_run_from_same_snapshot_in_isolated_clones(tmp_path):
    task, commit = _task(tmp_path)
    candidates = (
        Candidate("local-code", "local-model"),
        Candidate("cheap-cloud", "cheap-model"),
        Candidate("frontier", "frontier-model"),
    )

    run = run_counterfactual(
        task, candidates, FakeExecutor(), output_root=tmp_path / "runs")

    assert run.snapshot_commit == commit
    assert len(run.records) == 3
    assert run.total_cost_estimate == pytest.approx(0.03)
    assert [record.verification.task_success for record in run.records] == [
        True, False, True,
    ]
    assert [record.verification.quality_score for record in run.records] == [
        1.0, 0.0, 1.0,
    ]
    assert run.records[0].verification.improvements == ("value",)
    assert "repository is unchanged" in run.records[1].verification.changes.violations[0]
    assert task.repository.joinpath("app.py").read_text() == "VALUE = 0\n"
    assert _git(task.repository, "status", "--porcelain=v1") == ""

    workspaces = [Path(record.workspace) for record in run.records]
    assert len(set(workspaces)) == 3
    assert all(_git(workspace, "rev-parse", "HEAD") == commit
               for workspace in workspaces)

    rows = [json.loads(line) for line in Path(run.dataset_path).read_text().splitlines()]
    assert len(rows) == 3
    expected_hash = hashlib.sha256(task.prompt.encode()).hexdigest()
    assert {row["prompt_sha256"] for row in rows} == {expected_hash}
    assert all(row["prompt"] is None for row in rows)
    assert {row["snapshot_commit"] for row in rows} == {commit}
    assert stat.S_IMODE(Path(run.dataset_path).stat().st_mode) == 0o600


def test_verified_candidate_attaches_to_explicit_memory_route(tmp_path):
    task, _ = _task(tmp_path)
    provider = SqliteProvider(tmp_path / "memory.db")
    route_id = "live-route-to-verify"
    assert provider.record_decision(DecisionEvent(
        route_id=route_id, ts=1.0, session_id="live-session",
        turn_index=0, source="organic", instr_sha256="ab" * 32,
        features_json="{}", layer="heuristic", rung="local-code",
        cascade=True, reason="test", propensity="heuristic",
        policy_version="test",
    )) == route_id

    run = run_counterfactual(
        task,
        (Candidate("local-code", "local-model", route_id=route_id),),
        FakeExecutor(), output_root=tmp_path / "runs",
        verification_sink=provider,
    )

    assert run.records[0].verification.task_success is True
    assert run.records[0].verification_attached is True
    connection = sqlite3.connect(provider.db_path)
    row = connection.execute(
        "SELECT verified_success, verifier_source, verification_failure_cause "
        "FROM outcomes WHERE route_id=?", (route_id,),
    ).fetchone()
    assert row == (1, "deterministic:v1", None)
    assert connection.execute(
        "SELECT COUNT(*) FROM outcome_events WHERE route_id=? ", (route_id,),
    ).fetchone()[0] == 2
    connection.close()
    dataset_row = json.loads(Path(run.dataset_path).read_text().splitlines()[0])
    assert dataset_row["candidate"]["route_id"] == route_id
    assert dataset_row["verification_attached"] is True


class RaisingExecutor:
    def execute(self, task, candidate, workspace):
        if candidate.rung == "local-code":
            raise RuntimeError("simulated executor crash")
        (workspace / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
        return ExecutionResult(status="ok", duration_ms=1.0)


def test_executor_exception_is_recorded_and_other_candidates_continue(tmp_path):
    task, _ = _task(tmp_path)
    run = run_counterfactual(
        task,
        (Candidate("local-code", "local"), Candidate("frontier", "frontier")),
        RaisingExecutor(),
        output_root=tmp_path / "runs",
    )

    failed, succeeded = run.records
    assert failed.execution.status == "error"
    assert failed.execution.failure_cause == FailureCause.INFRASTRUCTURE.value
    assert failed.verification.task_success is None
    assert succeeded.verification.task_success is True


def test_dirty_source_is_rejected_before_creating_run(tmp_path):
    task, _ = _task(tmp_path)
    task.repository.joinpath("scratch.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be clean"):
        run_counterfactual(
            task, (Candidate("local-code", "local"),), FakeExecutor(),
            output_root=tmp_path / "runs",
        )
    assert not (tmp_path / "runs").exists()


def test_ephemeral_mode_removes_candidate_workspaces(tmp_path):
    task, _ = _task(tmp_path)
    run = run_counterfactual(
        task, (Candidate("local-code", "local"),), FakeExecutor(),
        output_root=tmp_path / "runs", retain_workspaces=False,
    )

    assert not Path(run.records[0].workspace).exists()
    assert Path(run.dataset_path).is_file()


def test_candidate_identity_and_required_fields_are_validated(tmp_path):
    task, _ = _task(tmp_path)
    with pytest.raises(ValueError, match="unique"):
        run_counterfactual(
            task,
            (Candidate("local", "same"), Candidate("local", "same")),
            FakeExecutor(), output_root=tmp_path / "runs-a",
        )
    with pytest.raises(ValueError, match="required"):
        run_counterfactual(
            task, (Candidate("local", " "),), FakeExecutor(),
            output_root=tmp_path / "runs-b",
        )


def test_cost_limit_records_skipped_candidates_instead_of_silently_dropping_them(
    tmp_path,
):
    task, _ = _task(tmp_path)
    run = run_counterfactual(
        task,
        (Candidate("local", "one"), Candidate("frontier", "two")),
        FakeExecutor(),
        output_root=tmp_path / "runs",
        max_total_cost_usd=0.01,
    )

    assert [record.execution.status for record in run.records] == ["ok", "skipped"]
    assert run.records[1].verification.task_success is None
    assert run.records[1].verification.failure_cause == "policy_constraint"
    assert run.total_cost_estimate == pytest.approx(0.01)


def test_cost_limit_skips_executor_that_cannot_enforce_hard_ceiling(tmp_path):
    class UnboundedExecutor:
        def execute(self, task, candidate, workspace):
            return ExecutionResult(
                status="ok", duration_ms=1.0, cost_estimate=10.0)

    task, _ = _task(tmp_path)
    run = run_counterfactual(
        task, (Candidate("frontier", "unbounded"),), UnboundedExecutor(),
        output_root=tmp_path / "runs", max_total_cost_usd=5.0,
    )
    assert run.records[0].execution.status == "skipped"
    assert "cannot enforce" in run.records[0].execution.detail
    assert run.total_cost_estimate == 0.0


def test_claude_executor_passes_remaining_budget_to_cli(tmp_path, monkeypatch):
    task, _ = _task(tmp_path)
    calls = []

    class Completed:
        returncode = 0
        stdout = json.dumps({"total_cost_usd": 0.25})
        stderr = ""

    def fake_run(command, **kwargs):
        calls.append(command)
        return Completed()

    monkeypatch.setattr("router.counterfactual.subprocess.run", fake_run)
    result = ClaudeCliExecutor(base_url="http://localhost:4001").execute_with_budget(
        task, Candidate("frontier", "opus"), task.repository,
        max_cost_usd=0.75,
    )
    assert result.status == "ok"
    index = calls[0].index("--max-budget-usd")
    assert calls[0][index + 1] == "0.75000000"


class InvalidMetricsExecutor:
    def execute(self, task, candidate, workspace):
        return ExecutionResult(status="ok", duration_ms=1.0, cost_estimate=float("nan"))


def test_invalid_executor_metrics_become_unverifiable_infrastructure_evidence(
    tmp_path,
):
    task, _ = _task(tmp_path)
    run = run_counterfactual(
        task, (Candidate("local", "bad-metrics"),), InvalidMetricsExecutor(),
        output_root=tmp_path / "runs",
    )

    assert run.records[0].execution.status == "error"
    assert run.records[0].verification.task_success is None
    assert run.records[0].execution.failure_cause == "infrastructure"


def test_simulator_plans_are_scenario_and_archetype_specific():
    commands = [{"name": "tests", "argv": ("npm", "test"), "cwd": "."}]
    rename = verification_plan_from_sandbox({
        "archetype": "nextjs_ts", "bad_var": "tmpVal",
        "good_var": "tempValue", "test_commands": commands,
    }, "rename")
    assert rename.command_checks == ()
    assert rename.repository_checks[0].name == "rename_complete"

    rounding = verification_plan_from_sandbox({
        "archetype": "nextjs_ts", "bad_var": "tmpVal",
        "good_var": "tempValue", "test_commands": commands,
    }, "fix_test")
    assert rounding.command_checks == ()
    assert rounding.repository_checks[0].name == "rounding_fix"

    sql = verification_plan_from_sandbox({
        "archetype": "sql_prisma", "bad_var": "st_flag",
        "good_var": "status_flag", "test_commands": [],
    }, "fix_test")
    assert sql.repository_checks[0].name == "indexed_filter_column"
    assert sql.required_new_path_globs


def test_python_fix_plan_turns_real_seeded_failure_green(tmp_path):
    sandbox = make_sandbox(
        tmp_path / "sandboxes", random.Random(1), archetype="python_cli",
        heavy=False, plant_secret=False)
    repository = Path(sandbox["path"])
    baseline = _git(repository, "rev-parse", "HEAD")
    plan = verification_plan_from_sandbox(sandbox, "fix_test")
    baseline_results = run_command_checks(repository, plan.command_checks)
    assert baseline_results[0].status == "fail"

    parser = repository / sandbox["project"] / "parse.py"
    parser.write_text(
        parser.read_text().replace("strptime(raw, fmt)", "strptime(raw.strip(), fmt)"),
        encoding="utf-8",
    )
    result = verify_candidate(
        repository, baseline, plan, baseline_results=baseline_results)
    assert result.task_success is True
    assert result.improvements == ("pytest",)


def test_nextjs_fix_plan_verifies_seeded_rounding_change_without_node_modules(
    tmp_path,
):
    sandbox = make_sandbox(
        tmp_path / "sandboxes", random.Random(2), archetype="nextjs_ts",
        heavy=False, plant_secret=False)
    repository = Path(sandbox["path"])
    baseline = _git(repository, "rev-parse", "HEAD")
    plan = verification_plan_from_sandbox(sandbox, "fix_test")
    baseline_result = run_repository_content_check(
        repository, plan.repository_checks[0])
    assert baseline_result.status == "fail"

    formatter = repository / "lib" / "format.ts"
    formatter.write_text(
        formatter.read_text().replace("Math.trunc", "Math.round"),
        encoding="utf-8",
    )
    result = verify_candidate(
        repository, baseline, plan, baseline_results=(baseline_result,))
    assert result.task_success is True
    assert result.improvements == ("rounding_fix",)


def test_sql_fix_plan_requires_a_new_index_migration(tmp_path):
    sandbox = make_sandbox(
        tmp_path / "sandboxes", random.Random(3), archetype="sql_prisma",
        heavy=False, plant_secret=False)
    repository = Path(sandbox["path"])
    baseline = _git(repository, "rev-parse", "HEAD")
    plan = verification_plan_from_sandbox(sandbox, "fix_test")
    migration = repository / "migrations" / "0003_filter_index" / "migration.sql"
    migration.parent.mkdir(parents=True)
    migration.write_text(
        f'CREATE INDEX "idx_filter" ON "{sandbox["entity"].capitalize()}" '
        f'("{sandbox["bad_var"]}");\n',
        encoding="utf-8",
    )

    result = verify_candidate(repository, baseline, plan)

    assert result.task_success is True
    assert result.changes.new_files == (
        "migrations/0003_filter_index/migration.sql",)


def test_rename_plan_verifies_symbol_replacement_without_unrelated_suite(tmp_path):
    sandbox = make_sandbox(
        tmp_path / "sandboxes", random.Random(4), archetype="nextjs_ts",
        heavy=False, plant_secret=False)
    repository = Path(sandbox["path"])
    baseline = _git(repository, "rev-parse", "HEAD")
    plan = verification_plan_from_sandbox(sandbox, "rename")
    for path in repository.rglob("*"):
        if path.suffix in {".ts", ".tsx"}:
            path.write_text(
                path.read_text().replace(sandbox["bad_var"], sandbox["good_var"]),
                encoding="utf-8",
            )

    result = verify_candidate(repository, baseline, plan)

    assert result.task_success is True
    assert result.repository_results[0].status == "pass"
