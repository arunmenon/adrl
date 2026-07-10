"""Tool-call scheduler. Port of core/coreToolScheduler.ts (the state machine
and approval flow; upstream adds hooks, an LLM safety classifier, output
persistence to disk, and batch budgets on top).

Every tool call walks this state machine:

    validating ──build() fails──────────────► error
        │
        ├─ permission 'allow' or YOLO ──────► scheduled
        ├─ permission 'deny' ───────────────► error (EXECUTION_DENIED)
        ├─ AUTO_EDIT + edit-type dialog ────► scheduled
        ├─ PLAN + non-info dialog ──────────► error (blocked by plan mode)
        ├─ non-interactive + needs dialog ──► error (cannot prompt)
        └─ else ────────────────────────────► awaiting_approval
                                                 │ user approves ► scheduled
                                                 │ user cancels ─► cancelled
    scheduled ──(whole batch clear)─────────► executing ► success | error

Execution waits until EVERY call in the batch has cleared approval, then
runs in sub-batches: consecutive concurrency-safe calls (read/search/fetch)
run in parallel, each mutator runs alone, in order. Results are converted
to functionResponse parts — success under an `output` key, failure under an
`error` key — because that's the only channel the model ever hears about a
tool through.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .config import ApprovalMode, Config
from .tools.base import (CONCURRENCY_SAFE_KINDS, ConfirmationDetails,
                         DeclarativeTool, ToolConfirmationOutcome, ToolError,
                         ToolErrorType, ToolInvocation, ToolResult)
from .tools.registry import ToolRegistry
from .types import (FunctionResponse, Part, ToolCallRequestInfo,
                    ToolCallResponseInfo)

TRUNCATION_SEPARATOR = "\n\n---\n... [CONTENT TRUNCATED] ...\n---\n\n"
VALIDATION_RETRY_LOOP_THRESHOLD = 3

# type Status = 'validating' | 'scheduled' | 'awaiting_approval'
#             | 'executing' | 'success' | 'error' | 'cancelled'
TERMINAL_STATES = {"success", "error", "cancelled"}

# Called when a tool needs approval: receives the call and dialog details,
# returns an outcome. The interactive CLI wires this to a y/n prompt.
ConfirmationHandler = Callable[["ToolCall", ConfirmationDetails],
                               Awaitable[ToolConfirmationOutcome]]


@dataclass
class ToolCall:
    request: ToolCallRequestInfo
    status: str = "validating"
    tool: DeclarativeTool | None = None
    invocation: ToolInvocation | None = None
    confirmation_details: ConfirmationDetails | None = None
    outcome: ToolConfirmationOutcome | None = None
    response: ToolCallResponseInfo | None = None


def convert_to_function_response(name: str, call_id: str, result: ToolResult) -> list[Part]:
    """convertToFunctionResponse: success -> {output}, failure -> {error}."""
    if result.error is not None:
        payload = {"error": result.error.message}
    else:
        payload = {"output": result.llm_content or "Tool execution succeeded."}
    return [Part(function_response=FunctionResponse(name=name, id=call_id,
                                                    response=payload))]


def truncate_output(text: str, threshold: int, keep: str = "both") -> str:
    """truncateAndSaveToFile shape (without the spill file): 'both' keeps
    1/5 head + 4/5 tail around the truncation marker."""
    if len(text) <= threshold:
        return text
    if keep == "head":
        body = text[:threshold]
    elif keep == "tail":
        body = text[-threshold:]
    else:
        head = threshold // 5
        body = text[:head] + TRUNCATION_SEPARATOR + text[-(threshold - head):]
    return ("Tool output was too large and has been truncated.\n"
            "The truncated output below shows the beginning and end of the "
            "content. The marker '... [CONTENT TRUNCATED] ...' indicates "
            "where content was removed.\n\n" + body)


class CoreToolScheduler:
    def __init__(self, config: Config, registry: ToolRegistry,
                 confirmation_handler: ConfirmationHandler | None = None):
        self.config = config
        self.registry = registry
        self.confirmation_handler = confirmation_handler
        self._validation_failures: dict[str, int] = {}
        self._session_always_allow: set[str] = set()

    async def schedule(self, requests: list[ToolCallRequestInfo]
                       ) -> list[ToolCallResponseInfo]:
        calls = [ToolCall(request=r) for r in requests]

        for call in calls:
            self._validate(call)
        for call in calls:
            if call.status == "validating":
                await self._resolve_approval(call)

        # execution starts only when the whole batch has cleared approvals
        for batch in self._partition([c for c in calls if c.status == "scheduled"]):
            await asyncio.gather(*(self._execute(c) for c in batch))

        return [c.response for c in calls if c.response is not None]

    # -- validating ----------------------------------------------------------

    def _validate(self, call: ToolCall) -> None:
        request = call.request
        tool = self.registry.get_tool(request.name)
        if tool is None:
            hint = self.registry.suggest(request.name)
            suffix = f' Did you mean "{hint}"?' if hint else ""
            self._fail(call, ToolErrorType.TOOL_NOT_REGISTERED,
                       f'Tool "{request.name}" not found in registry.{suffix}')
            return
        call.tool = tool
        try:
            call.invocation = tool.build(dict(request.args))
            call.status = "validating"
        except ValueError as e:
            key = f"{request.name}:{e}"
            self._validation_failures[key] = self._validation_failures.get(key, 0) + 1
            message = f"Error: Invalid parameters provided. Reason: {e}"
            if self._validation_failures[key] >= VALIDATION_RETRY_LOOP_THRESHOLD:
                message += ("\n⚠ RETRY LOOP DETECTED: this exact call has now "
                            "failed validation repeatedly. Do not retry it "
                            "unchanged — fix the parameters or use another tool.")
            self._fail(call, ToolErrorType.INVALID_TOOL_PARAMS, message)

    # -- approval -------------------------------------------------------------

    async def _resolve_approval(self, call: ToolCall) -> None:
        mode = self.config.approval_mode
        permission = call.invocation.get_default_permission()

        if call.request.name in self._session_always_allow:
            permission = "allow"
        if permission == "allow" or mode == ApprovalMode.YOLO:
            call.status = "scheduled"
            return
        if permission == "deny":
            self._fail(call, ToolErrorType.EXECUTION_DENIED,
                       f'Qwen Code requires permission to use "{call.request.name}", '
                       "but that permission was declined.")
            return

        details = call.invocation.get_confirmation_details()
        call.confirmation_details = details

        if mode == ApprovalMode.PLAN and details.type != "info":
            self._fail(call, ToolErrorType.EXECUTION_DENIED,
                       "Plan mode blocked a non-read-only tool call.")
            return
        if mode == ApprovalMode.AUTO_EDIT and details.type in ("edit", "info"):
            call.status = "scheduled"
            return
        if self.confirmation_handler is None:
            # non-interactive auto-deny (nonInteractiveToolExecutor behavior)
            self._fail(call, ToolErrorType.EXECUTION_DENIED,
                       f'Tool "{call.request.name}" requires user confirmation, '
                       "but this session is non-interactive and cannot prompt. "
                       "Rerun with --yolo or --approval-mode auto-edit to allow it.")
            return

        call.status = "awaiting_approval"
        outcome = await self.confirmation_handler(call, details)
        call.outcome = outcome
        if outcome == ToolConfirmationOutcome.CANCEL:
            self._finish(call, "cancelled",
                         error_payload="[Operation Cancelled] Reason: User did "
                                       "not allow tool call",
                         display="User did not allow tool call")
            return
        if outcome == ToolConfirmationOutcome.PROCEED_ALWAYS:
            self._session_always_allow.add(call.request.name)
        call.status = "scheduled"

    # -- executing -------------------------------------------------------------

    def _partition(self, calls: list[ToolCall]) -> list[list[ToolCall]]:
        """Consecutive concurrency-safe calls form one parallel batch; each
        unsafe (mutating) call runs alone, preserving order."""
        batches: list[list[ToolCall]] = []
        for call in calls:
            safe = call.tool is not None and call.tool.kind in CONCURRENCY_SAFE_KINDS
            if safe and batches and all(
                    c.tool.kind in CONCURRENCY_SAFE_KINDS for c in batches[-1]):
                batches[-1].append(call)
            else:
                batches.append([call])
        return batches

    async def _execute(self, call: ToolCall) -> None:
        call.status = "executing"
        try:
            result = await call.invocation.execute()
        except Exception as e:
            self._fail(call, ToolErrorType.UNHANDLED_EXCEPTION,
                       f"Error: Tool call execution failed. Reason: {e}")
            return

        if result.error is not None:
            call.status = "error"
        else:
            call.status = "success"
            threshold = (call.tool.max_output_chars
                         if call.tool.max_output_chars is not None
                         else self.config.truncate_tool_output_threshold)
            if threshold != float("inf") and len(result.llm_content) > threshold:
                result.llm_content = truncate_output(result.llm_content, int(threshold),
                                                     call.tool.truncate_keep)
        call.response = ToolCallResponseInfo(
            call_id=call.request.call_id,
            response_parts=convert_to_function_response(
                call.request.name, call.request.call_id, result),
            result_display=(result.return_display
                            if isinstance(result.return_display, str)
                            else result.llm_content[:500]),
            error=Exception(result.error.message) if result.error else None,
        )

    # -- terminal helpers --------------------------------------------------------

    def _fail(self, call: ToolCall, error_type: ToolErrorType, message: str) -> None:
        self._finish(call, "error", error_payload=message, display=message)

    def _finish(self, call: ToolCall, status: str, error_payload: str,
                display: str) -> None:
        call.status = status
        call.response = ToolCallResponseInfo(
            call_id=call.request.call_id,
            response_parts=[Part(function_response=FunctionResponse(
                name=call.request.name, id=call.request.call_id,
                response={"error": error_payload}))],
            result_display=display,
            error=Exception(error_payload) if status == "error" else None,
        )
