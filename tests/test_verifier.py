"""Deterministic verifier: repository state, commands, and three-valued labels."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from router.outcomes import FailureCause
from router.verifier import (
    CheckStatus,
    CommandCheck,
    ContentCheck,
    RepositoryContentCheck,
    VerificationPlan,
    collect_changes,
    run_command_checks,
    verify_candidate,
)


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True, text=True,
        check=True,
    )
    return result.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "source"
    (repository / "tests").mkdir(parents=True)
    (repository / "app.py").write_text("VALUE = 0\n", encoding="utf-8")
    (repository / "tests" / "test_app.py").write_text(
        "def test_placeholder():\n    assert True\n", encoding="utf-8")
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "tests@example.invalid")
    _git(repository, "config", "user.name", "ADRL Tests")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "baseline")
    return repository, _git(repository, "rev-parse", "HEAD")


def _value_check(expected: int) -> CommandCheck:
    script = (
        "from pathlib import Path; "
        f"raise SystemExit(0 if Path('app.py').read_text() == "
        f"'VALUE = {expected}\\n' else 1)"
    )
    return CommandCheck("value", (sys.executable, "-c", script))


def test_candidate_fix_is_verified_against_failing_baseline(tmp_path):
    repository, baseline = _repository(tmp_path)
    plan = VerificationPlan(
        command_checks=(_value_check(1),),
        content_checks=(ContentCheck(
            "content", "app.py", must_contain=("VALUE = 1",)),),
        allowed_path_globs=("app.py",),
    )
    baseline_results = run_command_checks(repository, plan.command_checks)
    assert baseline_results[0].status == CheckStatus.FAIL.value

    (repository / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    result = verify_candidate(
        repository, baseline, plan, baseline_results=baseline_results)

    assert result.task_success is True
    assert result.quality_score == 1.0
    assert result.improvements == ("value",)
    assert result.regressions == ()
    assert result.changes.files == ("app.py",)
    assert result.as_verified_outcome().failure_cause is None


def test_candidate_regression_is_a_definite_failure(tmp_path):
    repository, baseline = _repository(tmp_path)
    plan = VerificationPlan(command_checks=(_value_check(0),))
    baseline_results = run_command_checks(repository, plan.command_checks)
    (repository / "app.py").write_text("VALUE = 2\n", encoding="utf-8")

    result = verify_candidate(
        repository, baseline, plan, baseline_results=baseline_results)

    assert result.task_success is False
    assert result.failure_cause == FailureCause.TASK_CAPABILITY.value
    assert result.regressions == ("value",)


def test_missing_tool_and_invalid_cwd_are_unverifiable(tmp_path):
    repository, baseline = _repository(tmp_path)
    checks = (
        CommandCheck("missing", ("adrl-command-that-does-not-exist",)),
        CommandCheck("unsafe-cwd", (sys.executable, "-V"), cwd="../"),
    )
    result = verify_candidate(
        repository,
        baseline,
        VerificationPlan(command_checks=checks, require_changes=False),
    )

    assert result.task_success is None
    assert result.failure_cause == FailureCause.UNVERIFIABLE.value
    assert [item.status for item in result.command_results] == [
        CheckStatus.UNAVAILABLE.value,
        CheckStatus.ERROR.value,
    ]


def test_change_constraints_detect_scope_and_test_tampering(tmp_path):
    repository, baseline = _repository(tmp_path)
    (repository / "tests" / "test_app.py").write_text(
        "def test_placeholder():\n    assert False\n", encoding="utf-8")
    (repository / "notes.txt").write_text("outside scope\n", encoding="utf-8")
    plan = VerificationPlan(
        protect_tests=True,
        allowed_path_globs=("app.py",),
        max_changed_files=1,
    )

    changes = collect_changes(repository, baseline, plan)

    assert changes.inspection_error is None
    assert changes.files == ("notes.txt", "tests/test_app.py")
    assert any("limit 1" in violation for violation in changes.violations)
    assert any("protected test" in violation for violation in changes.violations)
    assert any("outside task scope" in violation for violation in changes.violations)
    result = verify_candidate(repository, baseline, plan)
    assert result.failure_cause == FailureCause.POLICY_CONSTRAINT.value
    assert result.quality_score == 0.0


def test_functional_failure_is_not_hidden_by_simultaneous_policy_violation(tmp_path):
    repository, baseline = _repository(tmp_path)
    (repository / "tests" / "test_app.py").write_text(
        "def test_placeholder():\n    assert False\n", encoding="utf-8")
    plan = VerificationPlan(
        command_checks=(_value_check(1),),
        protect_tests=True,
    )
    result = verify_candidate(repository, baseline, plan)
    assert result.task_success is False
    assert result.command_results[0].status == CheckStatus.FAIL.value
    assert result.changes.policy_violations
    assert result.failure_cause == FailureCause.TASK_CAPABILITY.value


def test_optional_failed_check_does_not_fail_verified_task(tmp_path):
    repository, baseline = _repository(tmp_path)
    (repository / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    plan = VerificationPlan(command_checks=(CommandCheck(
        "advisory", (sys.executable, "-c", "raise SystemExit(1)"),
        required=False,
    ),))

    result = verify_candidate(repository, baseline, plan)

    assert result.task_success is True
    assert result.command_results[0].status == CheckStatus.FAIL.value


def test_repository_content_checks_cover_tracked_and_new_files(tmp_path):
    repository, baseline = _repository(tmp_path)
    (repository / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / "new_module.py").write_text("RENAMED = True\n", encoding="utf-8")
    check = RepositoryContentCheck(
        "source-invariants",
        path_globs=("*.py", "**/*.py"),
        must_match=(r"\bRENAMED\b",),
        must_not_match=(r"VALUE\s*=\s*0",),
    )

    result = verify_candidate(
        repository, baseline,
        VerificationPlan(repository_checks=(check,)),
    )

    assert result.task_success is True
    assert result.repository_results[0].status == CheckStatus.PASS.value


def test_required_new_path_cannot_be_satisfied_by_editing_old_file(tmp_path):
    repository, baseline = _repository(tmp_path)
    (repository / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    plan = VerificationPlan(
        required_new_path_globs=("migrations/**/migration.sql",))
    first = collect_changes(repository, baseline, plan)
    assert any("required a new file" in item for item in first.violations)

    migration = repository / "migrations" / "0002_index" / "migration.sql"
    migration.parent.mkdir(parents=True)
    migration.write_text("CREATE INDEX idx_value ON app (value);\n", encoding="utf-8")
    second = collect_changes(repository, baseline, plan)
    assert second.new_files == ("migrations/0002_index/migration.sql",)
    assert not any("required a new file" in item for item in second.violations)
