"""A2 — assemble raw transcript records into turns.

A turn = one initiating user prompt + every tool_result continuation and
assistant response until stop_reason=end_turn (or the next promptId).

Grouping key: promptId (present on user records in all corpus eras).
Assistant records carry no promptId; they are attached by walking their
parentUuid chain up to the nearest user record whose turn is known, with a
file-order fallback for orphans (compaction/resume forks break chains).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .parser import (
    SourceFile,
    text_of,
    tool_result_blocks,
    tool_use_blocks,
    version_bucket,
)

INTERRUPT_PREFIX = "[Request interrupted by user"
EDIT_FAIL_MARKER = "String to replace not found in file"
REJECTION_MARKER = "User rejected tool use"

MAX_INSTRUCTION_CHARS = 2000  # stored for intent analysis; data/ never leaves the machine


@dataclass
class Turn:
    session_id: str
    project: str
    source_kind: str  # main | subagent
    agent_id: str | None
    workflow_id: str | None
    prompt_id: str | None
    ts: str | None = None
    version: str = "unknown"
    cwd: str | None = None
    git_branch: str | None = None

    instruction_text: str = ""
    is_meta: bool = False
    is_sidechain: bool = False
    is_command: bool = False  # <command-name> / <local-command-caveat> wrapper
    is_compact_summary: bool = False

    n_continuations: int = 0  # tool_result-bearing user records
    n_assistant_msgs: int = 0  # distinct API messages (message.id), not records
    n_tool_uses: int = 0
    tools_used: dict[str, int] = field(default_factory=dict)
    parallel_tool_use_msgs: int = 0  # API messages with >1 tool_use block
    origin_kind: str | None = None  # e.g. "task-notification" (harness-generated)
    # One API message spans up to 4 transcript records, each repeating the same
    # usage object — accumulate per message.id, never per record.
    _seen_msg_ids: set[str] = field(default_factory=set)
    _tool_uses_by_msg: dict[str, int] = field(default_factory=dict)

    models: set[str] = field(default_factory=set)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    has_thinking: bool = False

    n_error_results: int = 0
    n_edit_failures: int = 0
    interrupted: bool = False
    tool_rejected: bool = False
    final_stop_reason: str | None = None
    duration_ms: int | None = None
    message_count: int | None = None


def _usage_int(usage: Any, key: str) -> int:
    if isinstance(usage, dict):
        v = usage.get(key)
        if isinstance(v, (int, float)):
            return int(v)
    return 0


def assemble_turns(
    source: SourceFile,
    records: list[dict[str, Any]],
) -> list[Turn]:
    """Assemble the (already streamed, deduped) records of one file into turns."""
    by_uuid: dict[str, dict[str, Any]] = {}
    for rec in records:
        u = rec.get("uuid")
        if isinstance(u, str):
            by_uuid[u] = rec

    turns: list[Turn] = []
    turn_by_prompt: dict[str, Turn] = {}
    uuid_turn: dict[str, Turn] = {}  # user/assistant record uuid -> its turn
    current: Turn | None = None

    def turn_for_assistant(rec: dict[str, Any]) -> Turn | None:
        """Walk parentUuid chain to the nearest record with a known turn."""
        seen = 0
        parent = rec.get("parentUuid")
        while isinstance(parent, str) and seen < 200:
            if parent in uuid_turn:
                return uuid_turn[parent]
            parent_rec = by_uuid.get(parent)
            if parent_rec is None:
                break
            parent = parent_rec.get("parentUuid")
            seen += 1
        return current  # file-order fallback (broken chains: compaction, resume)

    for rec in records:
        rtype = rec.get("type")

        if rtype == "user":
            message = rec.get("message")
            results = tool_result_blocks(message)
            prompt_id = rec.get("promptId")
            text = text_of(message)

            is_continuation = bool(results) and (
                prompt_id is None or prompt_id in turn_by_prompt
            )
            if is_continuation:
                turn = turn_by_prompt.get(prompt_id) if prompt_id else current
                if turn is None:
                    turn = current
                if turn is not None:
                    turn.n_continuations += 1
                    for blk in results:
                        blob = str(blk.get("content", ""))
                        if blk.get("is_error"):
                            turn.n_error_results += 1
                        if EDIT_FAIL_MARKER in blob:
                            turn.n_edit_failures += 1
                    if rec.get("toolUseResult") == REJECTION_MARKER:
                        turn.tool_rejected = True
                    u = rec.get("uuid")
                    if isinstance(u, str):
                        uuid_turn[u] = turn
                continue

            # New initiating user record -> new turn
            turn = Turn(
                session_id=rec.get("sessionId") or source.session_id,
                project=source.project,
                source_kind=source.kind,
                agent_id=source.agent_id,
                workflow_id=source.workflow_id,
                prompt_id=prompt_id if isinstance(prompt_id, str) else None,
                ts=rec.get("timestamp"),
                version=version_bucket(rec),
                cwd=rec.get("cwd"),
                git_branch=rec.get("gitBranch"),
                instruction_text=text[:MAX_INSTRUCTION_CHARS],
                is_meta=bool(rec.get("isMeta")),
                is_sidechain=bool(rec.get("isSidechain")),
                is_command="<command-name>" in text or "<local-command-caveat>" in text,
                is_compact_summary=bool(rec.get("isCompactSummary")),
                origin_kind=(rec.get("origin") or {}).get("kind")
                if isinstance(rec.get("origin"), dict)
                else None,
            )
            if text.startswith(INTERRUPT_PREFIX):
                turn.interrupted = True
            turns.append(turn)
            current = turn
            if turn.prompt_id:
                turn_by_prompt[turn.prompt_id] = turn
            u = rec.get("uuid")
            if isinstance(u, str):
                uuid_turn[u] = turn

        elif rtype == "assistant":
            turn = turn_for_assistant(rec)
            if turn is None:
                continue
            u = rec.get("uuid")
            if isinstance(u, str):
                uuid_turn[u] = turn
            message = rec.get("message") if isinstance(rec.get("message"), dict) else {}
            model = message.get("model")
            if isinstance(model, str):
                turn.models.add(model)
            uses = tool_use_blocks(message)
            turn.n_tool_uses += len(uses)
            for blk in uses:
                name = blk.get("name", "<unknown>")
                turn.tools_used[name] = turn.tools_used.get(name, 0) + 1
            if any(b.get("type") == "thinking" for b in (message.get("content") or []) if isinstance(b, dict)):
                turn.has_thinking = True

            # CC splits one API message across up to 4 records, each repeating
            # the same usage — count usage once per message.id and detect
            # parallel tool use across the whole span.
            msg_id = message.get("id")
            key = msg_id if isinstance(msg_id, str) else rec.get("uuid", "")
            if uses:
                prev = turn._tool_uses_by_msg.get(key, 0)
                turn._tool_uses_by_msg[key] = prev + len(uses)
                if prev + len(uses) > 1 >= prev:
                    turn.parallel_tool_use_msgs += 1
            if key not in turn._seen_msg_ids:
                turn._seen_msg_ids.add(key)
                turn.n_assistant_msgs += 1
                usage = message.get("usage")
                turn.input_tokens += _usage_int(usage, "input_tokens")
                turn.output_tokens += _usage_int(usage, "output_tokens")
                turn.cache_read_tokens += _usage_int(usage, "cache_read_input_tokens")
                turn.cache_creation_tokens += _usage_int(usage, "cache_creation_input_tokens")
            stop = message.get("stop_reason")
            if isinstance(stop, str):
                turn.final_stop_reason = stop

        elif rtype == "system":
            if rec.get("subtype") == "turn_duration":
                # Joins to its turn via parentUuid; fall back to latest turn.
                turn = turn_for_assistant(rec)
                if turn is not None:
                    dur = rec.get("durationMs")
                    if isinstance(dur, (int, float)):
                        turn.duration_ms = int(dur)
                    mc = rec.get("messageCount")
                    if isinstance(mc, (int, float)):
                        turn.message_count = int(mc)

    return turns
