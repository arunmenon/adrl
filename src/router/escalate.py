"""C2 — escalation transcript rebuild (design doc §5.5, step 2-3).

When a `cascade=True` local turn trips a wire (§5.5), the failed turn's
transcript is handed to a higher rung. LiteLLM translates the wire *envelope*;
this module does the *semantic* scrubbing that LiteLLM does not:

  1. STRIP thinking blocks (`thinking` / `redacted_thinking`). Their
     provider-minted signatures / encrypted reasoning do NOT verify
     cross-provider — the API rejects them. They carry no tool-call IDs, so
     dropping them cannot orphan a tool_result.
  2. PASS tool_use / tool_result IDs THROUGH UNTOUCHED. B5 proved
     (reports/assumption-tool-ids.md, 2026-07-07) that the Anthropic API only
     requires `tool_use.id == tool_result.tool_use_id` *within the request*,
     not that IDs be provider-minted. The v2 draft's re-minting / `tool_id_map`
     machinery is therefore DROPPED — re-numbering would only risk breaking the
     internal pairing the harness's own transcript already guarantees.
  3. KEEP tool_results verbatim — they are ground truth (what actually happened
     on disk / in the shell) and must reach the frontier unedited.
  4. PREPEND a compact (<=3-line) failure note so the higher rung does not
     repeat the local model's dead end.

The messages array has no `system` role (system is a top-level request param),
so the note is prepended as a `user` message; when the transcript already opens
on a user message the note is merged into it, keeping strict role alternation.
"""

from __future__ import annotations

import copy
from typing import Any

Message = dict[str, Any]

# §5.5 step 1 — signatures / encrypted reasoning that don't cross providers.
THINKING_TYPES = frozenset({"thinking", "redacted_thinking"})

MAX_NOTE_LINES = 3  # §5.5 step 3: "Keep it to 3 lines."


def _is_thinking(block: Any) -> bool:
    return isinstance(block, dict) and block.get("type") in THINKING_TYPES


def _strip_thinking(messages: list[Message]) -> list[Message]:
    """Deep-copy the transcript, dropping thinking blocks. tool_use / tool_result
    blocks (and their IDs) are copied byte-for-byte — never re-mapped (B5)."""
    cleaned: list[Message] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            kept_blocks = [
                copy.deepcopy(block) for block in content if not _is_thinking(block)
            ]
            # A message that was *only* thinking becomes empty; an empty content
            # list is malformed, so drop the message entirely. Safe: such a
            # message held no tool_use, so nothing downstream is orphaned.
            if not kept_blocks:
                continue
            rebuilt = dict(message)  # preserve role (+ any other keys)
            rebuilt["content"] = kept_blocks
            cleaned.append(rebuilt)
        else:
            # string content (or absent) — nothing to strip; copy verbatim.
            cleaned.append(copy.deepcopy(message))
    return cleaned


def _note_text(failure_note_lines: list[str] | None) -> str:
    """Flatten to at most MAX_NOTE_LINES non-blank lines (embedded newlines in a
    single element are counted too, so the <=3-line guarantee is real)."""
    flattened: list[str] = []
    for item in failure_note_lines or []:
        for line in str(item).split("\n"):
            if line.strip():
                flattened.append(line.rstrip())
    return "\n".join(flattened[:MAX_NOTE_LINES])


def _prepend_note(messages: list[Message], note: str) -> list[Message]:
    if not note:
        return messages
    note_block = {"type": "text", "text": note}
    if messages and messages[0].get("role") == "user":
        # Merge into the opening user turn to avoid a user/user pair.
        first = messages[0]
        content = first.get("content")
        if isinstance(content, list):
            merged = [note_block, *content]
        elif isinstance(content, str) and content:
            merged = [note_block, {"type": "text", "text": content}]
        else:
            merged = [note_block]
        merged_first = dict(first)
        merged_first["content"] = merged
        return [merged_first, *messages[1:]]
    return [{"role": "user", "content": [note_block]}, *messages]


def rebuild_for_escalation(
    messages: list[Message],
    failure_note_lines: list[str] | None = None,
) -> list[Message]:
    """Prepare a failed local turn's transcript for a higher rung (§5.5).

    Strips thinking/redacted_thinking blocks, passes tool_use/tool_result IDs
    through unchanged (B5 — no re-minting), keeps tool_results verbatim, and
    prepends a <=3-line failure note. Returns a fresh, well-formed Anthropic
    messages list; the input `messages` is never mutated.
    """
    cleaned = _strip_thinking(messages)
    return _prepend_note(cleaned, _note_text(failure_note_lines))
