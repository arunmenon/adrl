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
from .tasks import SCENARIOS, pick

BUDGET_CAP_USD = 25.00  # decision D2 — hard cap, all simulator runs ever
PER_RUN_TIMEOUT_S = 600
MAX_TURNS = 25
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


def run_one(scenario_id: str, model: str, proxy: str, data_root: Path, rng: random.Random) -> dict:
    sb = make_sandbox(data_root / "sim-sandboxes", rng)
    sc = pick(rng, scenario_id)
    prompt = sc["prompt"](sb, rng)

    if sc["id"] == "commit_msg":  # needs staged changes to talk about
        readme = sb["path"] / "README.md"
        readme.write_text(readme.read_text() + f"\nHandles padded input gracefully.\n")
        (sb["path"] / f"{sb['project']}/parse.py").write_text(
            (sb["path"] / f"{sb['project']}/parse.py").read_text().replace(
                "strptime(raw, fmt)", "strptime(raw.strip(), fmt)")
        )
        subprocess.run(["git", "add", "-A"], cwd=sb["path"], check=True)

    # Scoped permissions, not a sandbox drop: the session gets exactly the tools
    # the scenarios need, inside a throwaway sandbox dir. No network tools, no
    # arbitrary bash — pytest and git only.
    allowed = [
        "Read", "Glob", "Grep", "LS", "Edit", "MultiEdit", "Write", "TodoWrite",
        "Bash(python -m pytest:*)", "Bash(pytest:*)", "Bash(python:*)", "Bash(git:*)",
        "Bash(ls:*)", "Bash(cat:*)",
    ]
    cmd = ["claude", "-p", prompt, "--output-format", "json",
           "--max-turns", str(MAX_TURNS), "--allowedTools", *allowed]
    if model != "default":
        cmd += ["--model", model]

    import os
    env = {**os.environ, "ANTHROPIC_BASE_URL": proxy}
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=sb["path"], env=env, capture_output=True,
                              text=True, timeout=PER_RUN_TIMEOUT_S)
        out = proc.stdout.strip()
        result = json.loads(out.splitlines()[-1]) if out else {}
        status = "ok" if proc.returncode == 0 else f"exit{proc.returncode}"
    except subprocess.TimeoutExpired:
        result, status = {}, "timeout"
    except (json.JSONDecodeError, IndexError):
        result, status = {"raw": out[:500]}, "unparsed"

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
    with ledger_path(data_root).open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--scenario", default="random", choices=["random", *SCENARIOS])
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
        e = run_one(args.scenario, model, args.proxy, args.data_root, rng)
        cost = f"${e['cost_usd']:.4f}" if e["cost_usd"] is not None else "?"
        print(f"[{i+1}/{args.runs}] {e['scenario']:<12} model={model:<8} {e['status']:<8} "
              f"turns={e['num_turns']} cost={cost} session={str(e['session_id'])[:8]}")

    print(f"total simulator spend: ${spent(args.data_root):.2f} / ${BUDGET_CAP_USD:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
