"""C2 — headless scenario runner: generates realistic agent traffic through the
capture proxy, across model families.

Each run: build a fresh sandbox project -> pick a scenario (randomized prompt)
-> spawn a headless session (`claude -p`) inside it, routed through the proxy
-> record cost/turns/session_id to the ledger.

Budget: hard $25 total cap (plan decision D2), enforced from data/sim-ledger.jsonl.
Provenance: simulator sessions are identified by the session_id in the ledger —
join against captures so synthetic traffic never contaminates organic baselines.

Usage:
  PYTHONPATH=src .venv/bin/python -m simulator.run_session --runs 3
  PYTHONPATH=src .venv/bin/python -m simulator.run_session --scenario fix_test --model haiku
  PYTHONPATH=src .venv/bin/python -m simulator.run_session --runs 6 --models default,opus,sonnet,haiku
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .sandbox import make_sandbox
from .tasks import SCENARIOS, applicable, pick

BUDGET_CAP_USD = 25.00  # decision D2 — hard cap, all simulator runs ever
PER_RUN_TIMEOUT_S = 600
MAX_TURNS = 25
# Episodes are now long-tailed (corpus p50 ~24 user-turns, tail to ~221); cap
# real spawned turns per episode so a single fat-tail draw can't blow the budget.
MAX_EPISODE_TURNS = 40
MAX_CLARIFY_TURNS = 6          # clarifying-answer insertions per episode
INTERRUPT_CUTOFF_S = 8         # short cutoff that forces a real mid-stream interrupt
INTERRUPT_MARKER = "[Request interrupted by user]"
# "default" = whatever the CLI is configured to use (Fable-class here); the
# aliases exercise the other families the registry needs evidence for.
DEFAULT_MODELS = ["default", "opus", "sonnet", "haiku"]


def ledger_path(root: Path) -> Path:
    return root / "sim-ledger.jsonl"


def spent(root: Path) -> float:
    p = ledger_path(root)
    if not p.exists():
        return 0.0
    return sum(json.loads(l).get("cost_usd") or 0.0 for l in p.open() if l.strip())


# Scoped permissions, not a sandbox drop: sessions get exactly the tools the
# scenarios need, inside throwaway sandbox dirs. No network tools, no arbitrary
# bash — pytest and git only.
ALLOWED_TOOLS = [
    "Read", "Glob", "Grep", "LS", "Edit", "MultiEdit", "Write", "TodoWrite",
    "Bash(python -m pytest:*)", "Bash(pytest:*)", "Bash(python:*)", "Bash(git:*)",
    "Bash(ls:*)", "Bash(cat:*)",
]


def spawn_turn(prompt: str, model: str, proxy: str, cwd: Path,
               resume_session: str | None = None,
               timeout_s: int = PER_RUN_TIMEOUT_S) -> tuple[dict, str]:
    """Run one headless turn (optionally resuming a session). Returns (result, status).

    `timeout_s` defaults to the per-run ceiling; the episode interrupt path
    passes a short cutoff to force a real mid-stream interruption.
    """
    import os
    cmd = ["claude", "-p", prompt, "--output-format", "json",
           "--max-turns", str(MAX_TURNS), "--allowedTools", *ALLOWED_TOOLS]
    if resume_session:
        cmd += ["--resume", resume_session]
    if model != "default":
        cmd += ["--model", model]
    env = {**os.environ, "ANTHROPIC_BASE_URL": proxy}
    out = ""
    try:
        proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                              text=True, timeout=timeout_s)
        out = proc.stdout.strip()
        result = json.loads(out.splitlines()[-1]) if out else {}
        return result, "ok" if proc.returncode == 0 else f"exit{proc.returncode}"
    except subprocess.TimeoutExpired:
        return {}, "timeout"
    except (json.JSONDecodeError, IndexError):
        return {"raw": out[:500]}, "unparsed"


_PASTE_EXTS = {".py", ".ts", ".tsx", ".sql", ".prisma", ".tf", ".md", ".json",
               ".yaml", ".yml"}


def _sandbox_paths(sb: dict, limit: int = 5) -> list[str]:
    """Real in-sandbox file paths the driver can paste inline (path-paste turns).

    Walks the ACTUAL sandbox so every archetype yields real, existing paths
    (review finding: hardcoded parse.py/stats.py don't exist in a Next.js or
    terraform sandbox — those false paths broke path-paste realism). The
    archetype's dominant-language files are preferred first.
    """
    base = Path(sb["path"])
    dom = sb.get("dominant_ext")
    found = [p for p in base.rglob("*")
             if p.is_file() and ".git" not in p.parts and p.suffix in _PASTE_EXTS]
    found.sort(key=lambda p: (p.suffix != dom, str(p)))   # dominant ext first, deterministic
    return [str(p) for p in found[:limit]]


def _log(data_root: Path, entry: dict) -> None:
    with ledger_path(data_root).open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


def run_one(scenario_id: str, model: str, proxy: str, data_root: Path, rng: random.Random) -> dict:
    sb = make_sandbox(data_root / "sim-sandboxes", rng)
    # Pass sb so a random pick is restricted to the sandbox's archetype (review
    # finding); an explicit --scenario is still honoured as the user's choice.
    sc = pick(rng, scenario_id, sb)
    prompt = sc["prompt"](sb, rng)

    if sc["id"] == "commit_msg":  # needs staged changes to talk about
        readme = sb["path"] / "README.md"
        readme.write_text(readme.read_text() + f"\nHandles padded input gracefully.\n")
        (sb["path"] / f"{sb['project']}/parse.py").write_text(
            (sb["path"] / f"{sb['project']}/parse.py").read_text().replace(
                "strptime(raw, fmt)", "strptime(raw.strip(), fmt)")
        )
        subprocess.run(["git", "add", "-A"], cwd=sb["path"], check=True)

    started = time.time()
    result, status = spawn_turn(prompt, model, proxy, sb["path"])
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": "simulator",
        "scenario": sc["id"],
        "maps_to": sc["maps_to"],
        "difficulty": sc["difficulty"],
        "model_requested": model,
        "prompt": prompt,
        "sandbox": str(sb["path"]),
        "status": status,
        "duration_s": round(time.time() - started, 1),
        "session_id": result.get("session_id"),
        "num_turns": result.get("num_turns"),
        "cost_usd": result.get("total_cost_usd") or result.get("cost_usd"),
        "is_error": result.get("is_error"),
        "result_preview": str(result.get("result", ""))[:200],
    }
    _log(data_root, entry)
    return entry


def run_episode(model: str, proxy: str, data_root: Path, rng: random.Random,
                *, n_threads: int | None = None, length: int | None = None) -> dict:
    """One stochastic, thread-interleaved episode (see episodes.generate_episode).

    Thread seeds are labeled tasks.py scenarios (the gradable skeleton); later
    steps are phrased by the noisy driver (direct API, never through the proxy)
    and resume the same session. Interrupts fire a real short-cutoff turn; the
    clarifying branch answers the agent's trailing question before drifting.

    INVARIANTS preserved: provenance (session_id -> ledger), scoped
    ALLOWED_TOOLS, --seed determinism (all draws from `rng`), mid-episode budget
    guard, and a hard MAX_EPISODE_TURNS cap on real spawns.
    """
    from .driver import next_message
    from .episodes import answer_is_question, clarifying_step, generate_episode

    # Build the sandbox FIRST so the episode's thread seeds can be restricted to
    # scenarios applicable to its archetype (review finding) — otherwise a
    # docs/terraform sandbox gets Python/Next.js tasks and becomes ungradable.
    sb = make_sandbox(data_root / "sim-sandboxes", rng)
    ep = generate_episode(rng, n_threads=n_threads, length=length,
                          allowed_scenarios=set(applicable(sb)))
    paths = _sandbox_paths(sb)

    turns: list[dict] = []
    session_id, last_answer = None, ""
    total_cost = driver_cost_total = 0.0
    clarify_used = 0
    started = time.time()

    def record(step_no, prompt, status, result, expected, thread, kind):
        nonlocal session_id, last_answer, total_cost
        session_id = result.get("session_id") or session_id
        last_answer = str(result.get("result", "")) or last_answer
        cost = result.get("total_cost_usd") or result.get("cost_usd") or 0.0
        total_cost += cost
        turns.append({
            "step": step_no, "kind": kind, "thread": thread, "prompt": prompt[:500],
            "status": status, "session_id": result.get("session_id"),
            "num_turns": result.get("num_turns"), "cost_usd": cost,
            "expected_label": expected,
        })
        return status

    spawned = 0
    for step in ep.steps:
        if spawned >= MAX_EPISODE_TURNS:
            break
        if spent(data_root) + total_cost >= BUDGET_CAP_USD:  # mid-episode budget guard
            break

        # clarifying branch: agent's last answer ended in a question -> answer it
        # in-context (one extra turn) before running the scheduled drift.
        if (step.kind == "driven" and clarify_used < MAX_CLARIFY_TURNS
                and answer_is_question(last_answer)):
            cs = clarifying_step(step.thread, step.topic)
            cprompt, dcost = next_message(cs.intent, last_answer, rng=rng,
                                          archetype=cs.archetype, sandbox_paths=paths)
            driver_cost_total += dcost
            res, st = spawn_turn(cprompt, model, proxy, sb["path"], resume_session=session_id)
            spawned += 1
            record(len(turns) + 1, cprompt, st, res, cs.expected_label, cs.thread, "clarify")
            clarify_used += 1
            if st != "ok":
                break

        if step.kind == "scenario":
            sc = pick(rng, step.scenario)
            prompt = sc["prompt"](sb, rng)
            res, st = spawn_turn(prompt, model, proxy, sb["path"], resume_session=session_id)
            spawned += 1
            record(len(turns) + 1, prompt, st, res, step.expected_label, step.thread, "scenario")
        elif step.kind == "interrupt":
            # real interrupt: fire an on-thread request, cut it off mid-stream.
            prompt, dcost = next_message(
                "Continue this thread with a quick concrete request.", last_answer,
                rng=rng, archetype="coding-ask", sandbox_paths=paths)
            driver_cost_total += dcost
            res, _ = spawn_turn(prompt, model, proxy, sb["path"],
                                resume_session=session_id, timeout_s=INTERRUPT_CUTOFF_S)
            spawned += 1
            # log the user-visible artifact verbatim; status marks the interruption
            record(len(turns) + 1, INTERRUPT_MARKER, "interrupted", res,
                   step.expected_label, step.thread, "interrupt")
            continue  # the generator already queued the organic retry as the next step
        else:  # driven
            prompt, dcost = next_message(
                step.intent, last_answer,
                require_completion_marker=step.require_completion_marker,
                rng=rng, archetype=step.archetype, sandbox_paths=paths)
            driver_cost_total += dcost
            res, st = spawn_turn(prompt, model, proxy, sb["path"], resume_session=session_id)
            spawned += 1
            record(len(turns) + 1, prompt, st, res, step.expected_label, step.thread, "driven")

        if turns and turns[-1]["status"] not in ("ok", "interrupted"):
            break

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": "simulator",
        "episode": "stochastic",
        "n_threads": ep.n_threads,
        "thread_scenarios": ep.thread_scenarios,
        "target_length": ep.target_length,
        "model_requested": model,
        "sandbox": str(sb["path"]),
        "duration_s": round(time.time() - started, 1),
        "session_ids": sorted({t["session_id"] for t in turns if t["session_id"]}),
        "steps_completed": sum(1 for t in turns if t["status"] in ("ok", "interrupted")),
        "steps_total": len(ep.steps),
        "turns_spawned": spawned,
        "cost_usd": total_cost + driver_cost_total,
        "driver_cost_usd": driver_cost_total,
        "turns": turns,
    }
    _log(data_root, entry)
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--scenario", default="random", choices=["random", *SCENARIOS])
    ap.add_argument("--episode", nargs="?", const="stochastic", default=None,
                    help="run stochastic thread-interleaved episodes (noisy driver plays "
                         "the user) instead of single shots")
    ap.add_argument("--threads", type=int, default=None, help="force N concurrent intent threads")
    ap.add_argument("--length", type=int, default=None, help="force episode length (turns)")
    ap.add_argument("--model", default=None, help="single model alias (default/opus/sonnet/haiku)")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="rotation when --model not given")
    ap.add_argument("--proxy", default="http://localhost:4000")
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    # proxy must be up — a simulator run that isn't captured is wasted money
    import urllib.request
    try:
        urllib.request.urlopen(args.proxy + "/", timeout=3)
    except Exception as exc:
        if "404" not in str(exc):  # upstream 404 on "/" means the proxy IS relaying
            print(f"capture proxy not reachable at {args.proxy} — start tools/run_proxy.sh first", file=sys.stderr)
            return 1

    used = spent(args.data_root)
    if used >= BUDGET_CAP_USD:
        print(f"budget cap reached: ${used:.2f} of ${BUDGET_CAP_USD:.2f} spent — refusing (D2)", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    rotation = [args.model] if args.model else args.models.split(",")
    print(f"budget: ${used:.2f} / ${BUDGET_CAP_USD:.2f} used | rotation: {rotation}")

    for i in range(args.runs):
        if spent(args.data_root) >= BUDGET_CAP_USD:
            print("budget cap hit mid-batch — stopping")
            break
        model = rotation[i % len(rotation)]
        if args.episode:
            e = run_episode(model, args.proxy, args.data_root, rng,
                            n_threads=args.threads, length=args.length)
            print(f"[{i+1}/{args.runs}] episode:{'stochastic':<12} model={model:<8} "
                  f"threads={e['n_threads']} spawned={e['turns_spawned']}/{e['target_length']} "
                  f"steps_ok={e['steps_completed']} "
                  f"cost=${e['cost_usd']:.4f} (driver ${e['driver_cost_usd']:.4f}) "
                  f"sessions={[s[:8] for s in e['session_ids']]}")
        else:
            e = run_one(args.scenario, model, args.proxy, args.data_root, rng)
            cost = f"${e['cost_usd']:.4f}" if e["cost_usd"] is not None else "?"
            print(f"[{i+1}/{args.runs}] {e['scenario']:<12} model={model:<8} {e['status']:<8} "
                  f"turns={e['num_turns']} cost={cost} session={str(e['session_id'])[:8]}")

    print(f"total simulator spend: ${spent(args.data_root):.2f} / ${BUDGET_CAP_USD:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
