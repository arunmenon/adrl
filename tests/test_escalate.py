"""C2 — tests for the escalation transcript rebuild (design §5.5, B5).

The load-bearing invariants:
  - thinking / redacted_thinking blocks are stripped (cross-provider blockers),
  - tool_use / tool_result IDs survive BYTE-IDENTICAL (B5: no re-minting),
  - tool_results are verbatim ground truth,
  - a compact <=3-line failure note is prepended,
  - the result is a well-formed Anthropic messages list (roles alternate, no
    orphaned tool_result, no empty content lists),
  - the caller's transcript is not mutated.
"""

import copy

import pytest

from router.escalate import THINKING_TYPES, rebuild_for_escalation

# Distinctive, deliberately NON-anthropic-format IDs. If any re-minting crept
# back in, these exact strings would change — that is the whole point of B5.
READ_ID = "toolu_LOCAL_readfile_01"
EDIT_ID = "call_edit_99"
EDIT_RESULT = "Error: no exact string match found (0 occurrences)"

NOTE_LINES = [
    "A previous local attempt read utils.py and tried to flip add() to a+b.",
    "The str_replace edit failed to apply twice (no exact match).",
    "Re-read the exact current text before editing.",
]


def _transcript():
    """A failed local turn: read ok, then two failed edits. Every assistant
    message carries a thinking block that must not survive."""
    return [
        {"role": "user", "content": "fix the failing test in utils.py"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "read the file first",
             "signature": "sig_localprovider_AAAA=="},
            {"type": "text", "text": "Let me read the file."},
            {"type": "tool_use", "id": READ_ID, "name": "read_file",
             "input": {"path": "utils.py"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": READ_ID,
             "content": "def add(a, b):\n    return a - b"},
        ]},
        {"role": "assistant", "content": [
            {"type": "redacted_thinking", "data": "encrypted_reasoning_blob_zzz"},
            {"type": "text", "text": "I'll fix the operator."},
            {"type": "tool_use", "id": EDIT_ID, "name": "str_replace",
             "input": {"old": "a - b", "new": "a + b"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": EDIT_ID,
             "content": EDIT_RESULT, "is_error": True},
        ]},
    ]


# ── helpers ─────────────────────────────────────────────────────────────────

def _all_blocks(messages):
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            yield from content


def _assert_well_formed(messages):
    """Roles valid + strictly alternating, no empty content list, and every
    tool_result is preceded by a tool_use with the same id (no orphans)."""
    prev_role = None
    for message in messages:
        assert message["role"] in ("user", "assistant"), message
        content = message["content"]
        if isinstance(content, list):
            assert len(content) > 0, "empty content list is malformed"
        assert message["role"] != prev_role, (
            f"roles must alternate, got {prev_role} then {message['role']}"
        )
        prev_role = message["role"]

    minted = set()
    for block in _all_blocks(messages):
        btype = block.get("type")
        if btype == "tool_use":
            minted.add(block["id"])
        elif btype == "tool_result":
            assert block["tool_use_id"] in minted, (
                f"orphaned tool_result -> {block['tool_use_id']}"
            )


# ── tests ───────────────────────────────────────────────────────────────────

def test_thinking_blocks_stripped():
    out = rebuild_for_escalation(_transcript(), NOTE_LINES)
    types = {b.get("type") for b in _all_blocks(out)}
    assert types.isdisjoint(THINKING_TYPES), f"thinking survived: {types}"
    # nothing anywhere still carries a signature / encrypted-reasoning payload
    for block in _all_blocks(out):
        assert "signature" not in block
        assert block.get("type") != "redacted_thinking"


def test_tool_ids_byte_identical_no_remint():
    out = rebuild_for_escalation(_transcript(), NOTE_LINES)
    use_ids = [b["id"] for b in _all_blocks(out) if b.get("type") == "tool_use"]
    result_ids = [
        b["tool_use_id"] for b in _all_blocks(out) if b.get("type") == "tool_result"
    ]
    # exact strings preserved — the foreign / arbitrary formats prove no re-mint
    assert use_ids == [READ_ID, EDIT_ID]
    assert result_ids == [READ_ID, EDIT_ID]
    # internal pairing intact (the ONLY thing B5 says the API requires)
    assert set(result_ids) <= set(use_ids)


def test_tool_results_verbatim():
    out = rebuild_for_escalation(_transcript(), NOTE_LINES)
    results = [b for b in _all_blocks(out) if b.get("type") == "tool_result"]
    read_result = next(b for b in results if b["tool_use_id"] == READ_ID)
    edit_result = next(b for b in results if b["tool_use_id"] == EDIT_ID)
    assert read_result["content"] == "def add(a, b):\n    return a - b"
    assert edit_result["content"] == EDIT_RESULT
    assert edit_result["is_error"] is True  # extra keys preserved, not dropped


def test_note_present_and_at_most_three_lines():
    out = rebuild_for_escalation(_transcript(), NOTE_LINES)
    # merged into the opening user turn as the leading text block
    first = out[0]
    assert first["role"] == "user"
    assert isinstance(first["content"], list)
    note_block = first["content"][0]
    assert note_block["type"] == "text"
    note = note_block["text"]
    assert note, "note must be present"
    assert note.count("\n") + 1 <= 3, f"note exceeded 3 lines:\n{note}"
    assert "failed to apply twice" in note
    # the original user request is retained right after the note
    assert any(
        b.get("text") == "fix the failing test in utils.py" for b in first["content"]
    )


def test_note_hard_capped_when_caller_overshoots():
    four_lines = [
        "line one of the note",
        "line two of the note\nsmuggled extra line",  # embedded newline counts
        "line four should be dropped",
    ]
    out = rebuild_for_escalation(_transcript(), four_lines)
    note = out[0]["content"][0]["text"]
    # exactly 3 lines kept; the embedded newline consumed the third slot, so the
    # 4th logical line is dropped.
    assert note.count("\n") + 1 == 3
    assert note.splitlines() == [
        "line one of the note",
        "line two of the note",
        "smuggled extra line",
    ]
    assert "line four should be dropped" not in note


def test_result_is_well_formed():
    out = rebuild_for_escalation(_transcript(), NOTE_LINES)
    _assert_well_formed(out)


def test_input_not_mutated():
    original = _transcript()
    snapshot = copy.deepcopy(original)
    rebuild_for_escalation(original, NOTE_LINES)
    assert original == snapshot, "caller's transcript must not be mutated"


def test_pure_thinking_message_dropped_not_emptied():
    # an assistant message that is ONLY a redacted_thinking block must vanish,
    # never survive as an empty-content message.
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [
            {"type": "redacted_thinking", "data": "blob"},
        ]},
    ]
    out = rebuild_for_escalation(messages, ["a note"])
    for message in out:
        if isinstance(message["content"], list):
            assert len(message["content"]) > 0
    # only the (note-merged) user message remains
    assert [m["role"] for m in out] == ["user"]


def test_prepends_user_message_when_transcript_opens_on_assistant():
    messages = [
        {"role": "assistant", "content": [{"type": "text", "text": "resuming"}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "x1", "content": "ok"},
        ]},
    ]
    # note is a standalone leading user message; roles still start user/assistant
    out = rebuild_for_escalation(messages, ["heads up"])
    assert out[0]["role"] == "user"
    assert out[0]["content"][0]["text"] == "heads up"
    assert out[1]["role"] == "assistant"


def test_empty_note_lines_prepends_nothing():
    messages = [{"role": "user", "content": "just do it"}]
    out = rebuild_for_escalation(messages, [])
    assert out == [{"role": "user", "content": "just do it"}]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
