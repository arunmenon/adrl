"""The prompt-variant experiment: does a smaller system prompt cost task
success on a small model? (design question from the routing study; the
registry measures, it doesn't guess.)

Grid: prompt variant (full/lean/minimal/minimal-noex) x tool-schema diet
(full/slim) x planted task x repetition. Each cell runs the REAL harness
loop (client + scheduler, YOLO approval) in a fresh sandbox against an
OpenAI-compatible endpoint, then an objective verifier scores the outcome.

Per run we record: success, rounds (model calls), tool calls, tool errors,
invalid-params errors, prompt/output tokens (summed across the chain),
turn-1 prefix size, wall time, and how the run ended (done / round-cap /
loop / error).

Usage:
  PYTHONPATH=src python -m qwen_harness.experiments.prompt_variants \
      --base-url http://localhost:4001/v1 --model local-code \
      --runs 3 --out reports/prompt-variant-eval.md

Defaults are sized for a local rung; --variants/--tasks subset the grid.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..builtins import create_tool_registry
from ..client import GeminiClient
from ..config import ApprovalMode, Config
from ..content_generator import ContentGenerator, GeneratorConfig
from ..prompts import VARIANT_SECTIONS
from ..scheduler import CoreToolScheduler
from ..types import GeminiEventType, Part
from .tasks import TASKS, Task

MAX_ROUNDS = 8          # model-call chain cap per run (keeps 1-N bounded)
RUN_WALL_CAP_S = 300


@dataclass
class RunRecord:
    variant: str
    slim_tools: bool
    task: str
    rep: int
    success: bool
    detail: str
    ended: str                    # done | round_cap | loop | error | timeout
    rounds: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    invalid_params: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    prefix_chars: int = 0
    wall_s: float = 0.0
    final_text: str = ""


async def run_once(task: Task, variant: str, slim_tools: bool, rep: int,
                   base_url: str, model: str, streaming: bool) -> RunRecord:
    sandbox = Path(tempfile.mkdtemp(prefix=f"pv-{task.name}-"))
    task.build(sandbox)

    config = Config(model=model, target_dir=str(sandbox),
                    approval_mode=ApprovalMode.YOLO,
                    prompt_variant=variant, slim_tool_schemas=slim_tools,
                    skip_next_speaker_check=True)  # keep runs deterministic-ish
    config.generator = GeneratorConfig(base_url=base_url, streaming=streaming,
                                       max_retries=2)
    registry = create_tool_registry(config)
    client = GeminiClient(config, generator=ContentGenerator(config.generator),
                          registry=registry)
    chat = client.start_chat()
    scheduler = CoreToolScheduler(config, registry)

    record = RunRecord(variant=variant, slim_tools=slim_tools, task=task.name,
                       rep=rep, success=False, detail="", ended="done",
                       prefix_chars=len(chat.system_instruction)
                       + len(json.dumps(chat.tools)))
    started = time.monotonic()
    prompt_id = client.new_prompt_id()
    current = [Part(text=task.prompt)]
    final_text: list[str] = []

    try:
        for round_no in range(MAX_ROUNDS):
            if time.monotonic() - started > RUN_WALL_CAP_S:
                record.ended = "timeout"
                break
            record.rounds += 1
            requests = []
            round_text: list[str] = []
            async for event in client.send_message_stream(
                    current, prompt_id, is_continuation=round_no > 0):
                if event.type == GeminiEventType.CONTENT:
                    round_text.append(event.value)
                elif event.type == GeminiEventType.TOOL_CALL_REQUEST:
                    requests.append(event.value)
                elif event.type == GeminiEventType.FINISHED and event.value.get("usage"):
                    usage = event.value["usage"]
                    record.prompt_tokens += usage.prompt_tokens
                    record.output_tokens += usage.completion_tokens
                elif event.type == GeminiEventType.LOOP_DETECTED:
                    record.ended = "loop"
                elif event.type == GeminiEventType.ERROR:
                    record.ended = "error"
                    record.detail = str(event.value)[:200]
            if round_text:
                final_text = round_text  # keep the last round's prose
            if record.ended in ("loop", "error"):
                break
            if not requests:
                record.ended = "done"
                break
            record.tool_calls += len(requests)
            responses = await scheduler.schedule(requests)
            for r in responses:
                if r.error is not None:
                    record.tool_errors += 1
                    if "Invalid parameters" in str(r.error):
                        record.invalid_params += 1
            current = [p for r in responses for p in r.response_parts]
        else:
            record.ended = "round_cap"
    except Exception as e:  # transport failures etc. — record, don't crash grid
        record.ended = "error"
        record.detail = f"{type(e).__name__}: {e}"[:200]

    record.wall_s = round(time.monotonic() - started, 1)
    record.final_text = "".join(final_text)[-500:]
    verdict = task.verify(sandbox, record.final_text)
    record.success = verdict.success and record.ended != "error"
    record.detail = record.detail or verdict.detail
    return record


def scorecard(records: list[RunRecord], model: str, base_url: str) -> str:
    lines = [
        "# Prompt-variant experiment — does a smaller system prompt cost task success?",
        "",
        f"Model: `{model}` via `{base_url}` | tasks: "
        f"{', '.join(sorted({r.task for r in records}))} | "
        f"{len(records)} runs | harness: qwen_harness (study port of qwen-code v0.19.8)",
        "",
        "## By variant",
        "",
        "| variant | slim tools | prefix chars | success | runs | avg rounds | "
        "tool calls | tool errs | invalid params | avg prompt tok/run | avg wall s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    keys = sorted({(r.variant, r.slim_tools) for r in records},
                  key=lambda k: (-_variant_rank(k[0]), k[1]))
    for variant, slim in keys:
        group = [r for r in records if r.variant == variant and r.slim_tools == slim]
        wins = sum(r.success for r in group)
        lines.append(
            f"| {variant} | {'yes' if slim else 'no'} "
            f"| {group[0].prefix_chars:,} "
            f"| **{wins}/{len(group)}** "
            f"| {len(group)} "
            f"| {statistics.mean(r.rounds for r in group):.1f} "
            f"| {sum(r.tool_calls for r in group)} "
            f"| {sum(r.tool_errors for r in group)} "
            f"| {sum(r.invalid_params for r in group)} "
            f"| {statistics.mean(r.prompt_tokens for r in group):,.0f} "
            f"| {statistics.mean(r.wall_s for r in group):.0f} |")

    lines += ["", "## By task x variant (successes/runs)", ""]
    tasks = sorted({r.task for r in records})
    header = "| variant (slim) | " + " | ".join(tasks) + " |"
    lines += [header, "|" + "---|" * (len(tasks) + 1)]
    for variant, slim in keys:
        row = [f"| {variant} ({'slim' if slim else 'full'}) "]
        for t in tasks:
            cell = [r for r in records if r.variant == variant
                    and r.slim_tools == slim and r.task == t]
            row.append(f"{sum(r.success for r in cell)}/{len(cell)} ")
        lines.append("|".join(row) + "|")

    fails = [r for r in records if not r.success]
    if fails:
        lines += ["", "## Failure detail", ""]
        for r in fails:
            lines.append(f"- `{r.variant}{'+slim' if r.slim_tools else ''}` "
                         f"{r.task}#{r.rep}: ended={r.ended}, rounds={r.rounds}, "
                         f"{r.detail}")
    return "\n".join(lines) + "\n"


def _variant_rank(name: str) -> int:
    order = list(VARIANT_SECTIONS)
    return len(order) - order.index(name)


async def run_grid(args) -> list[RunRecord]:
    variants = args.variants.split(",")
    tasks = [t for t in TASKS if t.name in set(args.tasks.split(","))]
    slim_options = [False, True] if args.slim_sweep else [args.slim_tools]
    records: list[RunRecord] = []
    total = len(variants) * len(slim_options) * len(tasks) * args.runs
    n = 0
    for variant in variants:
        for slim in slim_options:
            for task in tasks:
                for rep in range(args.runs):
                    n += 1
                    print(f"[{n}/{total}] {variant}"
                          f"{'+slim' if slim else ''} {task.name} rep{rep} ... ",
                          end="", flush=True, file=sys.stderr)
                    record = await run_once(task, variant, slim, rep,
                                            args.base_url, args.model,
                                            streaming=not args.no_stream)
                    records.append(record)
                    print(("OK" if record.success else "FAIL")
                          + f" ({record.ended}, {record.rounds} rounds, "
                            f"{record.wall_s}s)", file=sys.stderr)
                    if args.jsonl:
                        with open(args.jsonl, "a") as f:
                            f.write(json.dumps(asdict(record)) + "\n")
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:4001/v1")
    parser.add_argument("--model", default="local-code")
    parser.add_argument("--runs", type=int, default=3, help="reps per cell")
    parser.add_argument("--variants", default=",".join(VARIANT_SECTIONS))
    parser.add_argument("--tasks", default=",".join(t.name for t in TASKS))
    parser.add_argument("--slim-tools", action="store_true",
                        help="use slim tool schemas for every cell")
    parser.add_argument("--slim-sweep", action="store_true",
                        help="run every variant with BOTH full and slim schemas")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--out", default="reports/prompt-variant-eval.md")
    parser.add_argument("--jsonl", default="data/prompt-variant-runs.jsonl")
    args = parser.parse_args(argv)

    if args.jsonl:
        Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
    records = asyncio.run(run_grid(args))
    report = scorecard(records, args.model, args.base_url)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report)
    print(f"\nwrote {args.out} ({sum(r.success for r in records)}"
          f"/{len(records)} successes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
