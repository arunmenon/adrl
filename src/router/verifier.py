"""Deterministic task verification for counterfactual and live outcomes.

The verifier never decides whether prose "looks good". It runs structured
commands, checks repository changes and content invariants, and returns a
three-valued result: success, failure, or unverifiable. Missing tooling and
timeouts are unverifiable rather than model-capability failures.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

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


class VerificationPlanError(ValueError):
    """A serialized plan is malformed, unsafe, or vacuous."""


_PLAN_KEYS = {
    "command_checks", "content_checks", "repository_checks",
    "require_changes", "protect_tests", "forbidden_path_globs",
    "allowed_path_globs", "required_new_path_globs",
    "ignored_untracked_path_globs", "max_changed_files",
}
_COMMAND_KEYS = {
    "name", "argv", "cwd", "timeout_s", "required", "pass_codes", "env",
}
_CONTENT_KEYS = {
    "name", "path", "must_contain", "must_not_contain", "required",
}
_REPOSITORY_KEYS = {
    "name", "path_globs", "must_match", "must_not_match", "required",
    "max_file_bytes",
}


def _plan_object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VerificationPlanError(f"{name} must be an object")
    return value


def _plan_keys(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise VerificationPlanError(
            f"{name} has unknown fields: {', '.join(sorted(unknown))}")


def _plan_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VerificationPlanError(f"{name} must be a non-empty string")
    return value


def _plan_strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise VerificationPlanError(f"{name} must be a list of non-empty strings")
    return tuple(value)


def _relative_plan_path(value: Any, name: str, *, directory: bool = False) -> str:
    text = _plan_text(value, name)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise VerificationPlanError(f"{name} must stay inside the workspace")
    if directory and text == "":
        raise VerificationPlanError(f"{name} cannot be empty")
    return text


def _plan_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise VerificationPlanError(f"{name} must be boolean")
    return value


def verification_plan_from_mapping(value: Mapping[str, Any]) -> VerificationPlan:
    """Parse a strict argv-only plan suitable for trusted local execution.

    A serialized live plan must include at least one required functional or
    content check. Merely changing a file is not verifier-grade task evidence.
    """
    value = _plan_object(value, "verification plan")
    _plan_keys(value, _PLAN_KEYS, "verification plan")

    command_checks: list[CommandCheck] = []
    for index, raw in enumerate(value.get("command_checks", ())):
        item = _plan_object(raw, f"command_checks[{index}]")
        _plan_keys(item, _COMMAND_KEYS, f"command_checks[{index}]")
        argv = _plan_strings(item.get("argv"), f"command_checks[{index}].argv")
        timeout = item.get("timeout_s", 120.0)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise VerificationPlanError("command timeout must be numeric")
        timeout = float(timeout)
        if not math.isfinite(timeout) or not 0.0 < timeout <= 3600.0:
            raise VerificationPlanError("command timeout must be in (0, 3600]")
        pass_codes = item.get("pass_codes", (0,))
        if not isinstance(pass_codes, (list, tuple)) or not pass_codes or any(
            isinstance(code, bool) or not isinstance(code, int) for code in pass_codes
        ):
            raise VerificationPlanError("command pass_codes must be integers")
        env = item.get("env", {})
        if not isinstance(env, Mapping) or any(
            not isinstance(key, str) or not key
            or not isinstance(env_value, str)
            for key, env_value in env.items()
        ):
            raise VerificationPlanError("command env must map strings to strings")
        required = item.get("required", True)
        command_checks.append(CommandCheck(
            name=_plan_text(item.get("name"), f"command_checks[{index}].name"),
            argv=argv,
            cwd=_relative_plan_path(
                item.get("cwd", "."), f"command_checks[{index}].cwd",
                directory=True),
            timeout_s=timeout,
            required=_plan_bool(required, "command required"),
            pass_codes=tuple(pass_codes),
            env=dict(env),
        ))

    content_checks: list[ContentCheck] = []
    for index, raw in enumerate(value.get("content_checks", ())):
        item = _plan_object(raw, f"content_checks[{index}]")
        _plan_keys(item, _CONTENT_KEYS, f"content_checks[{index}]")
        required = item.get("required", True)
        content_checks.append(ContentCheck(
            name=_plan_text(item.get("name"), f"content_checks[{index}].name"),
            path=_relative_plan_path(
                item.get("path"), f"content_checks[{index}].path"),
            must_contain=_plan_strings(
                item.get("must_contain", ()),
                f"content_checks[{index}].must_contain"),
            must_not_contain=_plan_strings(
                item.get("must_not_contain", ()),
                f"content_checks[{index}].must_not_contain"),
            required=_plan_bool(required, "content required"),
        ))

    repository_checks: list[RepositoryContentCheck] = []
    for index, raw in enumerate(value.get("repository_checks", ())):
        item = _plan_object(raw, f"repository_checks[{index}]")
        _plan_keys(item, _REPOSITORY_KEYS, f"repository_checks[{index}]")
        max_bytes = item.get("max_file_bytes", 1_000_000)
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise VerificationPlanError("repository max_file_bytes must be positive")
        required = item.get("required", True)
        repository_checks.append(RepositoryContentCheck(
            name=_plan_text(item.get("name"), f"repository_checks[{index}].name"),
            path_globs=_plan_strings(
                item.get("path_globs"), f"repository_checks[{index}].path_globs"),
            must_match=_plan_strings(
                item.get("must_match", ()),
                f"repository_checks[{index}].must_match"),
            must_not_match=_plan_strings(
                item.get("must_not_match", ()),
                f"repository_checks[{index}].must_not_match"),
            required=_plan_bool(required, "repository check required"),
            max_file_bytes=max_bytes,
        ))

    check_names = [
        check.name for check in (*command_checks, *content_checks, *repository_checks)
    ]
    if len(check_names) != len(set(check_names)):
        raise VerificationPlanError("verification check names must be unique")
    if not any(
        check.required
        for check in (*command_checks, *content_checks, *repository_checks)
    ):
        raise VerificationPlanError(
            "verification plan requires at least one required functional check")

    def string_option(name: str, default: Sequence[str]) -> tuple[str, ...]:
        return _plan_strings(value.get(name, default), name)

    require_changes = value.get("require_changes", True)
    protect_tests = value.get("protect_tests", False)
    max_changed_files = value.get("max_changed_files")
    if max_changed_files is not None and (
        isinstance(max_changed_files, bool)
        or not isinstance(max_changed_files, int)
        or max_changed_files < 0
    ):
        raise VerificationPlanError("max_changed_files must be non-negative")
    forbidden_path_globs = string_option("forbidden_path_globs", (".git/**",))
    if ".git/**" not in forbidden_path_globs:
        raise VerificationPlanError(
            "serialized plans cannot remove the .git/** protected path")
    return VerificationPlan(
        command_checks=tuple(command_checks),
        content_checks=tuple(content_checks),
        repository_checks=tuple(repository_checks),
        require_changes=_plan_bool(require_changes, "require_changes"),
        protect_tests=_plan_bool(protect_tests, "protect_tests"),
        forbidden_path_globs=forbidden_path_globs,
        allowed_path_globs=string_option("allowed_path_globs", ()),
        required_new_path_globs=string_option("required_new_path_globs", ()),
        ignored_untracked_path_globs=string_option(
            "ignored_untracked_path_globs", VerificationPlan().ignored_untracked_path_globs),
        max_changed_files=max_changed_files,
    )


def verification_plan_to_mapping(plan: VerificationPlan) -> dict[str, Any]:
    if not isinstance(plan, VerificationPlan):
        raise VerificationPlanError("plan must be a VerificationPlan")
    return asdict(plan)


def verification_plan_sha256(plan: VerificationPlan) -> str:
    payload = json.dumps(
        verification_plan_to_mapping(plan), allow_nan=False,
        separators=(",", ":"), sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def repository_identity_sha256(workspace: Path, baseline_commit: str) -> str:
    """Stable repository group without persisting paths or remote credentials."""
    remote = _git(workspace, "config", "--get", "remote.origin.url")
    if remote.returncode == 0 and remote.stdout.strip():
        identity = "origin:" + remote.stdout.strip()
    else:
        roots = _git(workspace, "rev-list", "--max-parents=0", baseline_commit)
        identity = "roots:" + (roots.stdout.strip() or baseline_commit)
    return hashlib.sha256(identity.encode("utf-8", "replace")).hexdigest()


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
