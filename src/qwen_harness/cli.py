"""The driver loops. Port of packages/cli/src/nonInteractiveCli.ts
(runNonInteractive) plus a minimal interactive REPL standing in for the
Ink UI.

This file closes the loop the core deliberately leaves open: the client
yields ToolCallRequest events but never executes tools. The canonical
agentic loop is here, exactly as upstream writes it:

    while True:
        stream = client.send_message_stream(current_parts, prompt_id)
        collect tool_call_requests from the stream
        if tool_call_requests:
            responses = scheduler.schedule(tool_call_requests)
            current_parts = [functionResponse parts]   # feed back, loop
        else:
            break                                       # turn chain is done

Approval modes decide what schedule() does with each call; in
non-interactive mode anything that would need a dialog is auto-denied
(the model is told why), matching upstream.

Run it:
    PYTHONPATH=src python -m qwen_harness "explain this repo"      # one shot
    PYTHONPATH=src python -m qwen_harness                          # REPL
    ... --approval-mode yolo|auto-edit|plan --model X --base-url Y
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .builtins import create_tool_registry
from .client import GeminiClient
from .config import ApprovalMode, Config
from .scheduler import CoreToolScheduler
from .tools.base import ToolConfirmationOutcome
from .types import GeminiEventType, Part


async def run_non_interactive(config: Config, client: GeminiClient,
                              scheduler: CoreToolScheduler, prompt: str) -> int:
    prompt_id = client.new_prompt_id()
    current_parts = [Part(text=prompt)]
    turn_count = 0

    while True:
        turn_count += 1
        tool_call_requests = []
        error = False

        async for event in client.send_message_stream(
                current_parts, prompt_id, is_continuation=turn_count > 1):
            if event.type == GeminiEventType.CONTENT:
                print(event.value, end="", flush=True)
            elif event.type == GeminiEventType.THOUGHT and config.debug:
                print(f"\n[thought] {event.value}", file=sys.stderr)
            elif event.type == GeminiEventType.TOOL_CALL_REQUEST:
                tool_call_requests.append(event.value)
            elif event.type == GeminiEventType.CHAT_COMPRESSED:
                print(f"\n[compressed: {event.value.status.name}, "
                      f"{event.value.original_token_count} -> "
                      f"{event.value.new_token_count} tokens]", file=sys.stderr)
            elif event.type == GeminiEventType.LOOP_DETECTED:
                print(f"\n[loop detected: {event.value}]", file=sys.stderr)
            elif event.type == GeminiEventType.MAX_SESSION_TURNS:
                print("\n[max session turns reached]", file=sys.stderr)
                return 53
            elif event.type == GeminiEventType.ERROR:
                print(f"\n{event.value['message']}", file=sys.stderr)
                error = True

        if error:
            return 1
        if not tool_call_requests:
            print()
            return 0  # the model finished without asking for tools: done

        responses = await scheduler.schedule(tool_call_requests)
        for request, response in zip(tool_call_requests, responses):
            marker = "✗" if response.error else "✓"
            print(f"\n[{marker} {request.name}] {response.result_display or ''}"[:300],
                  file=sys.stderr)
        current_parts = [part for r in responses for part in r.response_parts]


async def _interactive_confirm(call, details) -> ToolConfirmationOutcome:
    """Stand-in for the Ink confirmation dialog."""
    print(f"\n--- {details.title} ---")
    if details.type == "exec":
        print(f"  command: {details.command}\n  root: {details.root_command}")
    elif details.type == "edit" and details.file_diff:
        print(details.file_diff[:2000] or "  (no diff)")
    elif details.prompt:
        print(f"  {details.prompt}")
    while True:
        answer = await asyncio.to_thread(
            input, "Allow? [y]es once / [a]lways / [n]o: ")
        answer = answer.strip().lower()
        if answer in ("y", "yes", ""):
            return ToolConfirmationOutcome.PROCEED_ONCE
        if answer in ("a", "always"):
            return ToolConfirmationOutcome.PROCEED_ALWAYS
        if answer in ("n", "no"):
            return ToolConfirmationOutcome.CANCEL


async def run_repl(config: Config, client: GeminiClient,
                   scheduler: CoreToolScheduler) -> int:
    print(f"qwen_harness study REPL — model={config.model} "
          f"approval={config.approval_mode.value} (ctrl-d to exit)")
    while True:
        try:
            prompt = await asyncio.to_thread(input, "\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt.strip():
            continue
        await run_non_interactive(config, client, scheduler, prompt)


def build(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="qwen_harness",
        description="Study re-implementation of the qwen-code agent harness.")
    parser.add_argument("prompt", nargs="?", help="one-shot prompt (omit for REPL)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible endpoint (default: env "
                             "OPENAI_BASE_URL or http://localhost:4001/v1)")
    parser.add_argument("--approval-mode", default="default",
                        choices=[m.value for m in ApprovalMode])
    parser.add_argument("--yolo", "-y", action="store_true",
                        help="shorthand for --approval-mode yolo")
    parser.add_argument("--cwd", default=None, help="workspace directory")
    parser.add_argument("--max-session-turns", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    config = Config()
    if args.model:
        config.model = args.model
    if args.base_url:
        config.generator.base_url = args.base_url
    if args.cwd:
        config.target_dir = args.cwd
    config.approval_mode = (ApprovalMode.YOLO if args.yolo
                            else ApprovalMode(args.approval_mode))
    config.max_session_turns = args.max_session_turns
    config.debug = args.debug
    config.set_interactive(args.prompt is None)

    registry = create_tool_registry(config)
    client = GeminiClient(config, registry=registry)
    client.start_chat()
    scheduler = CoreToolScheduler(
        config, registry,
        confirmation_handler=_interactive_confirm if config.is_interactive() else None)
    return args, config, client, scheduler


def main(argv: list[str] | None = None) -> int:
    args, config, client, scheduler = build(argv)
    if args.prompt:
        return asyncio.run(run_non_interactive(config, client, scheduler, args.prompt))
    return asyncio.run(run_repl(config, client, scheduler))


if __name__ == "__main__":
    sys.exit(main())
