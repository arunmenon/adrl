"""Deterministic task verification for counterfactual and live outcomes.

The verifier never decides whether prose "looks good". It runs structured
commands, checks repository changes and content invariants, and returns a
three-valued result: success, failure, or unverifiable. Missing tooling and
timeouts are unverifiable rather than model-capability failures.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

from .memory_ports import VerifiedOutcome
from .outcomes import FailureCause


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass(frozen=True)
class CommandCheck:
    name: str
    argv: tuple[str, ...]
    cwd: str = "."
    timeout_s: float = 120.0
    required: bool = True
    pass_codes: tuple[int, ...] = (0,)
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ContentCheck:
    name: str
    path: str
    must_contain: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()
    required: bool = True


@dataclass(frozen=True)
class RepositoryContentCheck:
    """Regex invariants evaluated across tracked and newly-created text files."""

    name: str
    path_globs: tuple[str, ...]
    must_match: tuple[str, ...] = ()
    must_not_match: tuple[str, ...] = ()
    required: bool = True
    max_file_bytes: int = 1_000_000


@dataclass(frozen=True)
class VerificationPlan:
    command_checks: tuple[CommandCheck, ...] = ()
    content_checks: tuple[ContentCheck, ...] = ()
    repository_checks: tuple[RepositoryContentCheck, ...] = ()
    require_changes: bool = True
    protect_tests: bool = False
    forbidden_path_globs: tuple[str, ...] = (".git/**",)
    allowed_path_globs: tuple[str, ...] = ()
    required_new_path_globs: tuple[str, ...] = ()
    ignored_untracked_path_globs: tuple[str, ...] = (
        ".pytest_cache/**", "**/.pytest_cache/**",
        "__pycache__/**", "**/__pycache__/**", "*.pyc", "**/*.pyc",
        "node_modules/**", "**/node_modules/**",
        ".next/**", "**/.next/**", ".terraform/**", "**/.terraform/**",
        "dist/**", "**/dist/**", "build/**", "**/build/**",
        "site/**", "**/site/**", "coverage/**", "**/coverage/**",
    )
    max_changed_files: Optional[int] = None


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    required: bool
    duration_ms: float = 0.0
    returncode: Optional[int] = None
    output_sha256: Optional[str] = None
    detail: str = ""


@dataclass(frozen=True)
class ChangeSummary:
    files: tuple[str, ...] = ()
    new_files: tuple[str, ...] = ()
    insertions: int = 0
    deletions: int = 0
    violations: tuple[str, ...] = ()
    policy_violations: tuple[str, ...] = ()
    inspection_error: Optional[str] = None


@dataclass(frozen=True)
class VerificationResult:
    task_success: Optional[bool]
    quality_score: Optional[float]
    confidence: float
    verifier_source: str
    failure_cause: Optional[str]
    command_results: tuple[CheckResult, ...] = ()
    content_results: tuple[CheckResult, ...] = ()
    repository_results: tuple[CheckResult, ...] = ()
    changes: ChangeSummary = field(default_factory=ChangeSummary)
    regressions: tuple[str, ...] = ()
    improvements: tuple[str, ...] = ()
    verified_at: float = 0.0

    def as_verified_outcome(self) -> VerifiedOutcome:
        return VerifiedOutcome(
            task_success=self.task_success,
            quality_score=self.quality_score,
            verifier_source=self.verifier_source,
            confidence=self.confidence,
            verified_at=self.verified_at,
            failure_cause=self.failure_cause,
        )


def _safe_cwd(workspace: Path, relative: str) -> Optional[Path]:
    try:
        root = workspace.resolve()
        candidate = (root / relative).resolve()
        candidate.relative_to(root)
        return candidate if candidate.is_dir() else None
    except (OSError, ValueError):
        return None


def _digest_output(stdout: str, stderr: str) -> str:
    return hashlib.sha256(
        (stdout + "\n---stderr---\n" + stderr).encode("utf-8", "replace")
    ).hexdigest()


def run_command_check(workspace: Path, check: CommandCheck) -> CheckResult:
    started = time.perf_counter()
    cwd = _safe_cwd(workspace, check.cwd)
    if cwd is None or not check.argv:
        return CheckResult(
            check.name, CheckStatus.ERROR.value, check.required,
            detail="invalid command or working directory",
        )
    try:
        proc = subprocess.run(
            list(check.argv), cwd=cwd,
            env={**os.environ, **check.env},
            capture_output=True, text=True, timeout=check.timeout_s,
            check=False,
        )
        status = (
            CheckStatus.PASS if proc.returncode in check.pass_codes
            else CheckStatus.FAIL
        )
        return CheckResult(
            check.name, status.value, check.required,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            returncode=proc.returncode,
            output_sha256=_digest_output(proc.stdout, proc.stderr),
            detail="command passed" if status is CheckStatus.PASS else "command failed",
        )
    except FileNotFoundError:
        return CheckResult(
            check.name, CheckStatus.UNAVAILABLE.value, check.required,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            detail=f"executable unavailable: {check.argv[0]}",
        )
    except subprocess.TimeoutExpired as exc:
        return CheckResult(
            check.name, CheckStatus.TIMEOUT.value, check.required,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            output_sha256=_digest_output(
                str(exc.stdout or ""), str(exc.stderr or "")),
            detail=f"timed out after {check.timeout_s:.1f}s",
        )
    except OSError as exc:
        return CheckResult(
            check.name, CheckStatus.ERROR.value, check.required,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            detail=f"execution error: {type(exc).__name__}",
        )


def run_command_checks(
    workspace: Path, checks: Sequence[CommandCheck]
) -> tuple[CheckResult, ...]:
    return tuple(run_command_check(workspace, check) for check in checks)


def run_content_check(workspace: Path, check: ContentCheck) -> CheckResult:
    started = time.perf_counter()
    root = workspace.resolve()
    try:
        path = (root / check.path).resolve()
        path.relative_to(root)
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError):
        return CheckResult(
            check.name, CheckStatus.FAIL.value, check.required,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            detail="required file is absent, unreadable, or outside workspace",
        )
    missing = [token for token in check.must_contain if token not in text]
    forbidden = [token for token in check.must_not_contain if token in text]
    passed = not missing and not forbidden
    detail = "content passed"
    if missing:
        detail = f"missing {len(missing)} required token(s)"
    elif forbidden:
        detail = f"contains {len(forbidden)} forbidden token(s)"
    return CheckResult(
        check.name,
        CheckStatus.PASS.value if passed else CheckStatus.FAIL.value,
        check.required,
        duration_ms=(time.perf_counter() - started) * 1000.0,
        output_sha256=hashlib.sha256(text.encode()).hexdigest(),
        detail=detail,
    )


def _path_matches(path: str, patterns: Sequence[str]) -> bool:
    candidate = Path(path)
    return any(fnmatch.fnmatch(path, pattern) or candidate.match(pattern)
               for pattern in patterns)


def run_repository_content_check(
    workspace: Path, check: RepositoryContentCheck
) -> CheckResult:
    """Evaluate corpus-wide invariants without exposing repository text."""
    started = time.perf_counter()
    listed = _git(workspace, "ls-files", "--cached", "--others",
                  "--exclude-standard")
    if listed.returncode != 0:
        return CheckResult(
            check.name, CheckStatus.ERROR.value, check.required,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            detail="unable to enumerate repository files",
        )
    try:
        required_patterns = [re.compile(pattern) for pattern in check.must_match]
        forbidden_patterns = [re.compile(pattern) for pattern in check.must_not_match]
    except re.error:
        return CheckResult(
            check.name, CheckStatus.ERROR.value, check.required,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            detail="invalid repository-check pattern",
        )

    root = workspace.resolve()
    texts: list[tuple[str, str]] = []
    digest = hashlib.sha256()
    for relative in sorted(set(listed.stdout.splitlines())):
        if not relative or not _path_matches(relative, check.path_globs):
            continue
        try:
            path = (root / relative).resolve()
            path.relative_to(root)
            if not path.is_file() or path.stat().st_size > check.max_file_bytes:
                continue
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError, ValueError):
            continue
        texts.append((relative, content))
        digest.update(relative.encode("utf-8", "replace"))
        digest.update(b"\0")
        digest.update(content.encode("utf-8", "replace"))

    missing = [
        pattern.pattern for pattern in required_patterns
        if not any(pattern.search(content) for _, content in texts)
    ]
    forbidden = [
        pattern.pattern for pattern in forbidden_patterns
        if any(pattern.search(content) for _, content in texts)
    ]
    passed = bool(texts) and not missing and not forbidden
    detail = f"repository content passed across {len(texts)} file(s)"
    if not texts:
        detail = "no readable files matched repository-check scope"
    elif missing:
        detail = f"missing {len(missing)} required repository pattern(s)"
    elif forbidden:
        detail = f"matched {len(forbidden)} forbidden repository pattern(s)"
    return CheckResult(
        check.name,
        CheckStatus.PASS.value if passed else CheckStatus.FAIL.value,
        check.required,
        duration_ms=(time.perf_counter() - started) * 1000.0,
        output_sha256=digest.hexdigest(),
        detail=detail,
    )


def _git(workspace: Path, *args: str) -> subprocess.CompletedProcess:
    command = ["git", *args]
    try:
        return subprocess.run(
            command, cwd=workspace, capture_output=True, text=True,
            timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return subprocess.CompletedProcess(command, 127, "", "")


def collect_changes(
    workspace: Path, baseline_commit: str, plan: VerificationPlan
) -> ChangeSummary:
    names = _git(workspace, "diff", "--name-only", baseline_commit, "--")
    untracked = _git(workspace, "ls-files", "--others", "--exclude-standard")
    added = _git(
        workspace, "diff", "--name-only", "--diff-filter=A", baseline_commit, "--")
    numstat = _git(workspace, "diff", "--numstat", baseline_commit, "--")
    if any(result.returncode != 0 for result in (names, untracked, added, numstat)):
        return ChangeSummary(inspection_error="unable to inspect git changes")

    tracked_files = {
        line.strip() for line in names.stdout.splitlines() if line.strip()
    }
    untracked_files = {
        line.strip() for line in untracked.stdout.splitlines()
        if line.strip() and not _path_matches(
            line.strip(), plan.ignored_untracked_path_globs)
    }
    files = sorted(tracked_files | untracked_files)
    new_files = sorted({
        line.strip() for line in added.stdout.splitlines() if line.strip()
    } | untracked_files)
    insertions = deletions = 0
    for line in numstat.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) < 2:
            continue
        try:
            insertions += int(parts[0]) if parts[0] != "-" else 0
            deletions += int(parts[1]) if parts[1] != "-" else 0
        except ValueError:
            continue

    violations: list[str] = []
    policy_violations: list[str] = []
    if plan.require_changes and not files:
        violations.append("task required a change but repository is unchanged")
    if plan.max_changed_files is not None and len(files) > plan.max_changed_files:
        violation = f"changed {len(files)} files (limit {plan.max_changed_files})"
        violations.append(violation)
        policy_violations.append(violation)
    if (plan.required_new_path_globs
            and not any(_path_matches(path, plan.required_new_path_globs)
                        for path in new_files)):
        violations.append(
            "task required a new file matching: "
            + ", ".join(plan.required_new_path_globs))
    if plan.protect_tests:
        protected = [path for path in files if _is_test_path(path)]
        if protected:
            violation = f"modified protected test files: {', '.join(protected)}"
            violations.append(violation)
            policy_violations.append(violation)
    forbidden = [
        path for path in files
        if _path_matches(path, plan.forbidden_path_globs)
    ]
    if forbidden:
        violation = f"modified forbidden paths: {', '.join(forbidden)}"
        violations.append(violation)
        policy_violations.append(violation)
    if plan.allowed_path_globs:
        outside = [
            path for path in files
            if not _path_matches(path, plan.allowed_path_globs)
        ]
        if outside:
            violation = f"modified paths outside task scope: {', '.join(outside)}"
            violations.append(violation)
            policy_violations.append(violation)
    return ChangeSummary(
        files=tuple(files),
        new_files=tuple(new_files),
        insertions=insertions,
        deletions=deletions,
        violations=tuple(violations),
        policy_violations=tuple(policy_violations),
    )


def _is_test_path(path: str) -> bool:
    normalized = "/" + path.lower().replace("\\", "/")
    name = Path(path).name.lower()
    return (
        "/tests/" in normalized or "/test/" in normalized
        or name.startswith("test_") or ".test." in name or ".spec." in name
    )


def verify_candidate(
    workspace: Path,
    baseline_commit: str,
    plan: VerificationPlan,
    *,
    baseline_results: Sequence[CheckResult] = (),
) -> VerificationResult:
    command_results = run_command_checks(workspace, plan.command_checks)
    content_results = tuple(
        run_content_check(workspace, check) for check in plan.content_checks)
    repository_results = tuple(
        run_repository_content_check(workspace, check)
        for check in plan.repository_checks
    )
    changes = collect_changes(workspace, baseline_commit, plan)

    all_results = (*command_results, *content_results, *repository_results)
    baseline = {result.name: result for result in baseline_results}
    regressions = tuple(
        result.name for result in all_results
        if result.required and result.status != CheckStatus.PASS.value
        and baseline.get(result.name) is not None
        and baseline[result.name].status == CheckStatus.PASS.value
    )
    improvements = tuple(
        result.name for result in all_results
        if result.status == CheckStatus.PASS.value
        and baseline.get(result.name) is not None
        and baseline[result.name].status == CheckStatus.FAIL.value
    )

    required = [
        result for result in (
            *command_results, *content_results, *repository_results)
        if result.required
    ]
    required_check_failed = any(
        result.status == CheckStatus.FAIL.value for result in required)
    definite_failure = bool(changes.violations) or required_check_failed
    unknown = bool(changes.inspection_error) or any(
        result.status in (
            CheckStatus.UNAVAILABLE.value,
            CheckStatus.TIMEOUT.value,
            CheckStatus.ERROR.value,
        ) for result in required
    )
    if definite_failure:
        task_success: Optional[bool] = False
        # A functional failure answers the routing question even when the same
        # candidate also violated scope. Keep policy evidence in ChangeSummary,
        # but do not let it hide a deterministic capability failure.
        cause = (
            FailureCause.TASK_CAPABILITY.value
            if required_check_failed or not changes.policy_violations
            else FailureCause.POLICY_CONSTRAINT.value
        )
        confidence = 1.0
    elif unknown:
        task_success = None
        cause = FailureCause.UNVERIFIABLE.value
        confidence = 0.0
    else:
        task_success = True
        cause = None
        confidence = 1.0

    scored = required or list(all_results)
    passed = sum(result.status == CheckStatus.PASS.value for result in scored)
    has_change_contract = bool(
        plan.require_changes
        or plan.protect_tests
        or plan.allowed_path_globs
        or plan.required_new_path_globs
        or plan.max_changed_files is not None
    )
    if has_change_contract:
        passed += int(not changes.violations and not changes.inspection_error)
    denominator = len(scored) + int(has_change_contract)
    quality = (
        None if task_success is None
        else (passed / denominator if denominator else 1.0)
    )
    return VerificationResult(
        task_success=task_success,
        quality_score=quality,
        confidence=confidence,
        verifier_source="deterministic:v1",
        failure_cause=cause,
        command_results=command_results,
        content_results=content_results,
        repository_results=repository_results,
        changes=changes,
        regressions=regressions,
        improvements=improvements,
        verified_at=time.time(),
    )
