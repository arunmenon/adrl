"""Isolated multi-rung counterfactual execution for learned-routing data.

One task is cloned from the same clean git commit into a baseline workspace and
one workspace per candidate. Executors are injected, so tests and offline jobs
share orchestration without coupling the router to a serving framework. Records
store hashes and metrics by default; raw prompts remain tenant-local and opt-in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

from .outcomes import FailureCause
from .verifier import (
    CheckResult,
    CommandCheck,
    RepositoryContentCheck,
    VerificationPlan,
    VerificationResult,
    run_command_checks,
    run_content_check,
    run_repository_content_check,
    verify_candidate,
)


@dataclass(frozen=True)
class Candidate:
    rung: str
    model: str
    harness: str = "claude_code"
    endpoint: str = ""
    route_id: Optional[str] = None


@dataclass(frozen=True)
class CounterfactualTask:
    task_id: str
    prompt: str
    repository: Path
    verification: VerificationPlan
    source: str = "synthetic"
    metadata: dict[str, Any] = field(default_factory=dict)
    require_clean_repository: bool = True
    store_prompt: bool = False


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    duration_ms: float
    returncode: Optional[int] = None
    output_sha256: Optional[str] = None
    cost_estimate: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    session_id: Optional[str] = None
    failure_cause: Optional[str] = None
    detail: str = ""


class CandidateExecutor(Protocol):
    def execute(
        self, task: CounterfactualTask, candidate: Candidate, workspace: Path
    ) -> ExecutionResult: ...


class VerificationSink(Protocol):
    def attach_verification(
        self, route_id: str, verification, *, event_id: Optional[str] = None,
        observed_at: Optional[float] = None,
    ) -> bool: ...


@dataclass(frozen=True)
class CounterfactualRecord:
    run_id: str
    task_id: str
    task_source: str
    prompt_sha256: str
    prompt: Optional[str]
    snapshot_commit: str
    candidate: Candidate
    execution: ExecutionResult
    verification: VerificationResult
    baseline_results: tuple[CheckResult, ...]
    workspace: str
    created_at: str
    verification_attached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CounterfactualRun:
    run_id: str
    task_id: str
    snapshot_commit: str
    records: tuple[CounterfactualRecord, ...]
    dataset_path: str
    total_cost_estimate: float


class ClaudeCliExecutor:
    """Force one model alias through an Anthropic-compatible endpoint."""

    def __init__(self, *, base_url: str, binary: str = "claude",
                 timeout_s: float = 600.0, max_turns: int = 25,
                 allowed_tools: Sequence[str] = ()):
        self.base_url = base_url
        self.binary = binary
        self.timeout_s = timeout_s
        self.max_turns = max_turns
        self.allowed_tools = tuple(allowed_tools)

    def execute(
        self, task: CounterfactualTask, candidate: Candidate, workspace: Path
    ) -> ExecutionResult:
        return self._execute(task, candidate, workspace, max_cost_usd=None)

    def execute_with_budget(
        self, task: CounterfactualTask, candidate: Candidate, workspace: Path,
        *, max_cost_usd: float,
    ) -> ExecutionResult:
        """Delegate the hard per-candidate ceiling to Claude CLI itself."""
        return self._execute(
            task, candidate, workspace, max_cost_usd=max(0.0, float(max_cost_usd)))

    def _execute(
        self, task: CounterfactualTask, candidate: Candidate, workspace: Path,
        *, max_cost_usd: Optional[float],
    ) -> ExecutionResult:
        command = [
            self.binary, "-p", task.prompt, "--output-format", "json",
            "--max-turns", str(self.max_turns), "--model", candidate.model,
        ]
        if max_cost_usd is not None:
            command += ["--max-budget-usd", f"{max_cost_usd:.8f}"]
        if self.allowed_tools:
            command += ["--allowedTools", *self.allowed_tools]
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                command, cwd=workspace,
                env={
                    **os.environ,
                    "ANTHROPIC_BASE_URL": candidate.endpoint or self.base_url,
                },
                capture_output=True, text=True, timeout=self.timeout_s,
                check=False,
            )
        except FileNotFoundError:
            return ExecutionResult(
                "unavailable", (time.perf_counter() - started) * 1000.0,
                failure_cause=FailureCause.INFRASTRUCTURE.value,
                detail=f"executor unavailable: {self.binary}",
            )
        except subprocess.TimeoutExpired as exc:
            output = str(exc.stdout or "") + str(exc.stderr or "")
            return ExecutionResult(
                "timeout", (time.perf_counter() - started) * 1000.0,
                output_sha256=hashlib.sha256(output.encode()).hexdigest(),
                failure_cause=FailureCause.INFRASTRUCTURE.value,
                detail=f"executor timed out after {self.timeout_s:.1f}s",
            )
        except OSError as exc:
            return ExecutionResult(
                "error", (time.perf_counter() - started) * 1000.0,
                failure_cause=FailureCause.INFRASTRUCTURE.value,
                detail=f"executor error: {type(exc).__name__}",
            )

        output = proc.stdout + "\n---stderr---\n" + proc.stderr
        payload: dict[str, Any] = {}
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            pass
        status = "ok" if proc.returncode == 0 else "error"
        return ExecutionResult(
            status=status,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            returncode=proc.returncode,
            output_sha256=hashlib.sha256(output.encode()).hexdigest(),
            cost_estimate=float(
                payload.get("total_cost_usd") or payload.get("cost_usd") or 0.0),
            input_tokens=int(payload.get("input_tokens") or 0),
            output_tokens=int(payload.get("output_tokens") or 0),
            session_id=payload.get("session_id"),
            failure_cause=(None if status == "ok"
                           else FailureCause.INFRASTRUCTURE.value),
            detail="executor completed" if status == "ok" else "executor exited non-zero",
        )


def verification_plan_from_sandbox(
    sandbox: dict, scenario_id: Optional[str] = None, *, protect_tests: bool = True,
    require_changes: bool = True,
) -> VerificationPlan:
    checks_list: list[CommandCheck] = []
    for item in sandbox.get("test_commands") or ():
        argv = tuple(str(part) for part in item["argv"])
        if argv and argv[0] in {"python", "python3"}:
            argv = (sys.executable, *argv[1:])
        checks_list.append(CommandCheck(
            name=str(item["name"]), argv=argv,
            cwd=str(item.get("cwd", "."))))
    checks = tuple(checks_list)
    archetype = str(sandbox.get("archetype") or "")
    repository_checks: tuple[RepositoryContentCheck, ...] = ()
    required_new_path_globs: tuple[str, ...] = ()

    if scenario_id == "rename":
        globs_by_archetype = {
            "python_cli": ("*.py", "**/*.py"),
            "nextjs_ts": ("*.ts", "**/*.ts", "*.tsx", "**/*.tsx"),
            "terraform": ("*.tf", "**/*.tf"),
            "sql_prisma": (
                "*.sql", "**/*.sql", "*.prisma", "**/*.prisma"),
            "monorepo": ("*.ts", "**/*.ts", "*.tsx", "**/*.tsx"),
        }
        bad = re.escape(str(sandbox["bad_var"]))
        good = re.escape(str(sandbox["good_var"]))
        repository_checks = (RepositoryContentCheck(
            name="rename_complete",
            path_globs=globs_by_archetype[archetype],
            must_match=(rf"\b{good}\b",),
            must_not_match=(rf"\b{bad}\b",),
        ),)
        # The seeded suite can contain an unrelated planted failure. The rename
        # invariant is the deterministic label; do not turn that known baseline
        # failure into a false model failure.
        checks = ()
    elif scenario_id == "fix_test" and archetype == "nextjs_ts":
        repository_checks = (RepositoryContentCheck(
            name="rounding_fix",
            path_globs=("lib/format.ts",),
            must_match=(r"\bformatAmount\b",),
            must_not_match=(r"\bMath\.trunc\s*\(",),
        ),)
        checks = ()  # generated sandboxes do not vendor node_modules
    elif scenario_id == "fix_test" and archetype == "sql_prisma":
        column = re.escape(str(sandbox["bad_var"]))
        repository_checks = (RepositoryContentCheck(
            name="indexed_filter_column",
            path_globs=("migrations/*.sql", "migrations/**/*.sql"),
            must_match=(
                rf"(?is)CREATE\s+(?:UNIQUE\s+)?INDEX\b[^;]*"
                rf"\([^;]*\b{column}\b[^;]*\)",
            ),
        ),)
        required_new_path_globs = (
            "migrations/*/migration.sql", "migrations/**/migration.sql")
        checks = ()  # prisma validate does not verify index coverage
    return VerificationPlan(
        command_checks=checks,
        repository_checks=repository_checks,
        require_changes=require_changes,
        protect_tests=protect_tests,
        required_new_path_globs=required_new_path_globs,
    )


def _git(repository: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=repository, capture_output=True, text=True,
        timeout=30, check=False,
    )


def _snapshot_commit(repository: Path, require_clean: bool) -> str:
    if not repository.is_dir():
        raise ValueError(f"repository does not exist: {repository}")
    commit = _git(repository, "rev-parse", "HEAD")
    if commit.returncode != 0 or not commit.stdout.strip():
        raise ValueError("counterfactual source must be a git repository with HEAD")
    if require_clean:
        status = _git(repository, "status", "--porcelain=v1")
        if status.returncode != 0 or status.stdout.strip():
            raise ValueError("counterfactual source repository must be clean")
    return commit.stdout.strip()


def _clone_at(repository: Path, destination: Path, commit: str) -> None:
    clone = subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(repository),
         str(destination)],
        capture_output=True, text=True, timeout=120, check=False,
    )
    if clone.returncode != 0:
        raise RuntimeError("failed to clone counterfactual snapshot")
    checkout = _git(destination, "checkout", "--quiet", "--detach", commit)
    if checkout.returncode != 0:
        raise RuntimeError("failed to checkout counterfactual snapshot")


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return safe[:80] or "candidate"


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    return str(value)


def _append_record(path: Path, record: CounterfactualRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _jsonable(asdict(record))
    encoded = (json.dumps(payload, allow_nan=False, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError("partial counterfactual record write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unverifiable_execution(execution: ExecutionResult) -> VerificationResult:
    return VerificationResult(
        task_success=None,
        quality_score=None,
        confidence=0.0,
        verifier_source="deterministic:v1",
        failure_cause=execution.failure_cause or FailureCause.UNVERIFIABLE.value,
        verified_at=time.time(),
    )


def _execute_safely(
    executor: CandidateExecutor,
    task: CounterfactualTask,
    candidate: Candidate,
    workspace: Path,
    *,
    max_cost_usd: Optional[float] = None,
) -> ExecutionResult:
    """Keep one broken executor/candidate from aborting the experiment."""
    started = time.perf_counter()
    try:
        if max_cost_usd is None:
            result = executor.execute(task, candidate, workspace)
        else:
            budgeted = getattr(executor, "execute_with_budget", None)
            if not callable(budgeted):
                return ExecutionResult(
                    status="skipped", duration_ms=0.0,
                    failure_cause=FailureCause.POLICY_CONSTRAINT.value,
                    detail="executor cannot enforce a hard counterfactual cost limit",
                )
            result = budgeted(
                task, candidate, workspace, max_cost_usd=max_cost_usd)
    except Exception as exc:
        return ExecutionResult(
            status="error",
            duration_ms=(time.perf_counter() - started) * 1000.0,
            failure_cause=FailureCause.INFRASTRUCTURE.value,
            detail=f"executor raised: {type(exc).__name__}",
        )
    if not isinstance(result, ExecutionResult):
        return ExecutionResult(
            status="error",
            duration_ms=(time.perf_counter() - started) * 1000.0,
            failure_cause=FailureCause.INFRASTRUCTURE.value,
            detail="executor returned an invalid result",
        )
    try:
        valid = (
            result.status in {"ok", "error", "unavailable", "timeout", "skipped"}
            and isinstance(result.duration_ms, (int, float))
            and not isinstance(result.duration_ms, bool)
            and math.isfinite(float(result.duration_ms))
            and float(result.duration_ms) >= 0.0
            and isinstance(result.cost_estimate, (int, float))
            and not isinstance(result.cost_estimate, bool)
            and math.isfinite(float(result.cost_estimate))
            and float(result.cost_estimate) >= 0.0
            and isinstance(result.input_tokens, int)
            and not isinstance(result.input_tokens, bool)
            and result.input_tokens >= 0
            and isinstance(result.output_tokens, int)
            and not isinstance(result.output_tokens, bool)
            and result.output_tokens >= 0
        )
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        return ExecutionResult(
            status="error",
            duration_ms=(time.perf_counter() - started) * 1000.0,
            failure_cause=FailureCause.INFRASTRUCTURE.value,
            detail="executor returned invalid metrics",
        )
    return result


def run_counterfactual(
    task: CounterfactualTask,
    candidates: Sequence[Candidate],
    executor: CandidateExecutor,
    *,
    output_root: Path = Path("data/counterfactual"),
    dataset_path: Optional[Path] = None,
    retain_workspaces: bool = True,
    max_total_cost_usd: Optional[float] = None,
    verification_sink: Optional[VerificationSink] = None,
) -> CounterfactualRun:
    if not candidates:
        raise ValueError("at least one counterfactual candidate is required")
    if (not isinstance(task.task_id, str) or not task.task_id.strip()
            or not isinstance(task.prompt, str) or not task.prompt.strip()):
        raise ValueError("counterfactual task id and prompt are required")
    if not isinstance(task.metadata, dict):
        raise ValueError("counterfactual task metadata must be a dictionary")
    if max_total_cost_usd is not None:
        try:
            max_total_cost_usd = float(max_total_cost_usd)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("counterfactual cost limit must be a non-negative number")
        if not math.isfinite(max_total_cost_usd) or max_total_cost_usd < 0.0:
            raise ValueError("counterfactual cost limit must be a non-negative number")
    check_names = [
        check.name for check in (
            *task.verification.command_checks,
            *task.verification.content_checks,
            *task.verification.repository_checks,
        )
    ]
    if any(not name.strip() for name in check_names) or len(set(check_names)) != len(
        check_names
    ):
        raise ValueError("verification check names must be non-empty and unique")
    if any(not isinstance(candidate.rung, str)
           or not isinstance(candidate.model, str)
           or not candidate.rung.strip() or not candidate.model.strip()
           for candidate in candidates):
        raise ValueError("counterfactual candidate rung and model are required")
    identities = set(candidates)
    if len(identities) != len(candidates):
        raise ValueError("counterfactual candidates must be unique")

    repository = Path(task.repository).resolve()
    commit = _snapshot_commit(repository, task.require_clean_repository)
    run_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:10]}"
    run_root = Path(output_root) / _safe_name(task.task_id) / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    dataset = dataset_path or (Path(output_root) / "records.jsonl")

    baseline_workspace = run_root / "baseline"
    _clone_at(repository, baseline_workspace, commit)
    baseline_results = (
        *run_command_checks(
            baseline_workspace, task.verification.command_checks),
        *(run_content_check(baseline_workspace, check)
          for check in task.verification.content_checks),
        *(run_repository_content_check(baseline_workspace, check)
          for check in task.verification.repository_checks),
    )
    records: list[CounterfactualRecord] = []
    total_cost = 0.0
    prompt_hash = hashlib.sha256(task.prompt.encode("utf-8")).hexdigest()
    try:
        for index, candidate in enumerate(candidates):
            workspace = run_root / _safe_name(
                f"{index:02d}-{candidate.rung}-{candidate.model}-{candidate.harness}")
            _clone_at(repository, workspace, commit)
            if max_total_cost_usd is not None and total_cost >= max_total_cost_usd:
                execution = ExecutionResult(
                    status="skipped",
                    duration_ms=0.0,
                    failure_cause=FailureCause.POLICY_CONSTRAINT.value,
                    detail="counterfactual cost limit exhausted",
                )
            else:
                remaining_cost = (
                    None if max_total_cost_usd is None
                    else max(0.0, max_total_cost_usd - total_cost)
                )
                execution = _execute_safely(
                    executor, task, candidate, workspace,
                    max_cost_usd=remaining_cost,
                )
                total_cost += float(execution.cost_estimate)
            verification = (
                verify_candidate(
                    workspace, commit, task.verification,
                    baseline_results=baseline_results)
                if execution.status == "ok"
                else _unverifiable_execution(execution)
            )
            verification_attached = False
            if candidate.route_id and verification_sink is not None:
                try:
                    verification_attached = bool(
                        verification_sink.attach_verification(
                            candidate.route_id,
                            verification.as_verified_outcome(),
                            event_id=(
                                f"counterfactual:{run_id}:{index}:"
                                f"{candidate.route_id}"
                            ),
                            observed_at=verification.verified_at,
                        )
                    )
                except Exception:
                    verification_attached = False
            record = CounterfactualRecord(
                run_id=run_id,
                task_id=task.task_id,
                task_source=task.source,
                prompt_sha256=prompt_hash,
                prompt=task.prompt if task.store_prompt else None,
                snapshot_commit=commit,
                candidate=candidate,
                execution=execution,
                verification=verification,
                baseline_results=baseline_results,
                workspace=str(workspace),
                created_at=datetime.now(timezone.utc).isoformat(),
                verification_attached=verification_attached,
                metadata=dict(task.metadata),
            )
            _append_record(dataset, record)
            records.append(record)
    finally:
        if not retain_workspaces:
            shutil.rmtree(run_root, ignore_errors=True)

    return CounterfactualRun(
        run_id=run_id,
        task_id=task.task_id,
        snapshot_commit=commit,
        records=tuple(records),
        dataset_path=str(dataset),
        total_cost_estimate=total_cost,
    )


def main() -> int:
    from simulator.run_session import ALLOWED_TOOLS
    from simulator.sandbox import make_sandbox
    from simulator.tasks import pick
    import random

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="fix_test", choices=("fix_test", "rename"))
    parser.add_argument("--archetype", default="python_cli")
    parser.add_argument("--models", default="local-code,cheap-cloud,frontier")
    parser.add_argument("--base-url", default="http://localhost:4001")
    parser.add_argument("--output", type=Path, default=Path("data/counterfactual"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--budget-usd", type=float, default=5.0)
    parser.add_argument(
        "--route-id", default="",
        help="existing ledger route_id verified by a single candidate",
    )
    parser.add_argument(
        "--memory-db", type=Path, default=Path("data/router-memory.db"),
        help="ledger receiving --route-id verification evidence",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sandbox = make_sandbox(
        args.output / "sources", rng, archetype=args.archetype,
        heavy=False, plant_secret=False)
    scenario = pick(rng, args.scenario, sandbox)
    prompt = scenario["prompt"](sandbox, rng)
    task = CounterfactualTask(
        task_id=f"{args.scenario}-{args.archetype}-{args.seed}",
        prompt=prompt,
        repository=Path(sandbox["path"]),
        verification=verification_plan_from_sandbox(sandbox, args.scenario),
        metadata={"scenario": args.scenario, "archetype": args.archetype},
    )
    candidates = tuple(
        Candidate(
            rung=model.strip(), model=model.strip(), endpoint=args.base_url,
            route_id=(args.route_id.strip() or None),
        )
        for model in args.models.split(",") if model.strip()
    )
    if args.route_id.strip() and len(candidates) != 1:
        parser.error("--route-id requires exactly one --models candidate")
    verification_sink = None
    if args.route_id.strip():
        from router.memory_sqlite import SqliteProvider
        verification_sink = SqliteProvider(args.memory_db)
    result = run_counterfactual(
        task, candidates,
        ClaudeCliExecutor(
            base_url=args.base_url, allowed_tools=ALLOWED_TOOLS),
        output_root=args.output,
        max_total_cost_usd=args.budget_usd,
        verification_sink=verification_sink,
    )
    print(json.dumps({
        "run_id": result.run_id,
        "task_id": result.task_id,
        "snapshot_commit": result.snapshot_commit,
        "records": len(result.records),
        "cost_estimate": result.total_cost_estimate,
        "verification_attached": sum(
            record.verification_attached for record in result.records),
        "dataset": result.dataset_path,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
