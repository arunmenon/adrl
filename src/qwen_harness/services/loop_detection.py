"""Loop detection. Port of services/loopDetectionService.ts.

Two tiers, checked per streamed event in client.py:

  Tier 1 (`check_always_on_safeties`) cannot be disabled — it guards
  against the failure modes that burn money fast:
    - CONSECUTIVE_IDENTICAL_TOOL_CALLS: same (name, args) hash 5x in a row
    - TURN_TOOL_CALL_CAP: >100 tool calls in one logical turn (the counter
      spans ToolResult continuations and only resets on a new user prompt)

  Tier 2 (`add_and_check_heuristic_loops`) is heuristic and — notably —
  *opt-in* at current HEAD (skipLoopDetection defaults to true upstream;
  false positives evidently cost more than the loops):
    - GLOBAL_TOOL_CALL_DUPLICATE: same call key 6x anywhere in the turn
    - ALTERNATING_TOOL_CALL_PATTERN: strict ABABAB over the last 6 calls
    - READ_FILE_LOOP: >=8 read-ish calls in the last 15 (only after some
      non-read tool has run, so cold-start exploration is exempt)
    - ACTION_STAGNATION: 8 consecutive calls to the same tool name
    - CHANTING_IDENTICAL_SENTENCES: sha256 over a 50-char sliding window of
      streamed text; fires when one chunk hash recurs 10 times with mean
      spacing <= 75 chars. Code blocks/tables/lists reset tracking.
    - REPETITIVE_THOUGHTS: last 3 thought signatures identical
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from enum import Enum

from ..types import GeminiEvent, GeminiEventType

TOOL_CALL_LOOP_THRESHOLD = 5
CONTENT_LOOP_THRESHOLD = 10
CONTENT_CHUNK_SIZE = 50
MAX_HISTORY_LENGTH = 1000
THOUGHT_REPEAT_THRESHOLD = 3
MAX_THOUGHT_HISTORY = 50
FILE_READ_THRESHOLD = 8
FILE_READ_WINDOW = 15
STAGNATION_THRESHOLD = 8
GLOBAL_DUPLICATE_THRESHOLD = 6
ALTERNATING_PATTERN_CYCLES = 3
DEFAULT_MAX_TOOL_CALLS_PER_TURN = 100

_READ_TOOLS = {"read_file", "read_many_files", "list_directory"}


class LoopType(str, Enum):
    CONSECUTIVE_IDENTICAL_TOOL_CALLS = "consecutive_identical_tool_calls"
    CHANTING_IDENTICAL_SENTENCES = "chanting_identical_sentences"
    REPETITIVE_THOUGHTS = "repetitive_thoughts"
    READ_FILE_LOOP = "read_file_loop"
    ACTION_STAGNATION = "action_stagnation"
    GLOBAL_TOOL_CALL_DUPLICATE = "global_tool_call_duplicate"
    ALTERNATING_TOOL_CALL_PATTERN = "alternating_tool_call_pattern"
    TURN_TOOL_CALL_CAP = "turn_tool_call_cap"


def _call_key(name: str, args: dict) -> str:
    return hashlib.sha256(f"{name}:{json.dumps(args, sort_keys=True)}".encode()).hexdigest()


class LoopDetectionService:
    def __init__(self, max_tool_calls_per_turn: int = DEFAULT_MAX_TOOL_CALLS_PER_TURN):
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.last_loop_type: LoopType | None = None
        self.reset("")

    def reset(self, prompt_id: str) -> None:
        self.prompt_id = prompt_id
        # tier 1
        self._last_key: str | None = None
        self._consecutive = 0
        self._turn_total = 0
        self._turn_total_committed = 0
        # tier 2
        self._global_counts: dict[str, int] = defaultdict(int)
        self._alternating: list[str] = []
        self._recent_names: list[str] = []
        self._same_name_streak: tuple[str, int] = ("", 0)
        self._non_read_seen = False
        self._reset_content()
        self._thoughts: list[str] = []

    def _reset_content(self) -> None:
        self._stream = ""
        self._chunk_hits: dict[str, list[int]] = defaultdict(list)
        self._in_code_block = False

    # ------------------------------------------------------------------ tier 1

    def check_always_on_safeties(self, event: GeminiEvent) -> bool:
        if event.type == GeminiEventType.FINISHED:
            self._turn_total_committed = self._turn_total
            return False
        if event.type != GeminiEventType.TOOL_CALL_REQUEST:
            return False
        req = event.value
        key = _call_key(req.name, req.args)

        self._consecutive = self._consecutive + 1 if key == self._last_key else 1
        self._last_key = key
        if self._consecutive >= TOOL_CALL_LOOP_THRESHOLD:
            return self._fire(LoopType.CONSECUTIVE_IDENTICAL_TOOL_CALLS)

        self._turn_total += 1
        if self._turn_total > self.max_tool_calls_per_turn:
            return self._fire(LoopType.TURN_TOOL_CALL_CAP)
        return False

    # ------------------------------------------------------------------ tier 2

    def add_and_check_heuristic_loops(self, event: GeminiEvent) -> bool:
        if event.type == GeminiEventType.TOOL_CALL_REQUEST:
            self._reset_content()
            self._thoughts.clear()
            return self._check_tool_heuristics(event.value)
        if event.type == GeminiEventType.CONTENT:
            return self._check_content(event.value or "")
        if event.type == GeminiEventType.THOUGHT:
            return self._check_thought(event.value)
        return False

    def _check_tool_heuristics(self, req) -> bool:
        key = _call_key(req.name, req.args)

        self._global_counts[key] += 1
        if self._global_counts[key] >= GLOBAL_DUPLICATE_THRESHOLD:
            return self._fire(LoopType.GLOBAL_TOOL_CALL_DUPLICATE)

        window = 2 * ALTERNATING_PATTERN_CYCLES
        self._alternating = (self._alternating + [key])[-window:]
        if len(self._alternating) == window:
            a, b = self._alternating[0], self._alternating[1]
            if a != b and all(k == (a if i % 2 == 0 else b)
                              for i, k in enumerate(self._alternating)):
                return self._fire(LoopType.ALTERNATING_TOOL_CALL_PATTERN)

        is_read = req.name in _READ_TOOLS or req.name.startswith(("read_", "list_"))
        if not is_read:
            self._non_read_seen = True
        self._recent_names = (self._recent_names + [req.name])[-FILE_READ_WINDOW:]
        reads = sum(1 for n in self._recent_names
                    if n in _READ_TOOLS or n.startswith(("read_", "list_")))
        if self._non_read_seen and reads >= FILE_READ_THRESHOLD:
            return self._fire(LoopType.READ_FILE_LOOP)

        name, streak = self._same_name_streak
        self._same_name_streak = (req.name, streak + 1 if name == req.name else 1)
        if self._same_name_streak[1] >= STAGNATION_THRESHOLD:
            return self._fire(LoopType.ACTION_STAGNATION)
        return False

    def _check_content(self, text: str) -> bool:
        # structured output (code, tables, lists, headings) legitimately
        # repeats — reset instead of flagging
        if any(marker in text for marker in ("```", "|", "- ", "# ", "> ")):
            if "```" in text and text.count("```") % 2 == 1:
                self._in_code_block = not self._in_code_block
            self._reset_content()
            return False
        if self._in_code_block:
            return False

        start = len(self._stream)
        self._stream += text
        if len(self._stream) > MAX_HISTORY_LENGTH:
            trim = len(self._stream) - MAX_HISTORY_LENGTH
            self._stream = self._stream[trim:]
            self._chunk_hits = defaultdict(list, {
                h: [i - trim for i in idxs if i - trim >= 0]
                for h, idxs in self._chunk_hits.items()
            })
            start = max(0, start - trim)

        for i in range(max(0, start - CONTENT_CHUNK_SIZE), len(self._stream) - CONTENT_CHUNK_SIZE + 1):
            chunk = self._stream[i:i + CONTENT_CHUNK_SIZE]
            h = hashlib.sha256(chunk.encode()).hexdigest()
            hits = self._chunk_hits[h]
            if not hits or hits[-1] != i:
                hits.append(i)
            if len(hits) >= CONTENT_LOOP_THRESHOLD:
                recent = hits[-CONTENT_LOOP_THRESHOLD:]
                gaps = [b - a for a, b in zip(recent, recent[1:])]
                if sum(gaps) / len(gaps) <= 1.5 * CONTENT_CHUNK_SIZE:
                    return self._fire(LoopType.CHANTING_IDENTICAL_SENTENCES)
        return False

    def _check_thought(self, thought) -> bool:
        subject = (getattr(thought, "subject", None) or str(thought or ""))[:200]
        description = (getattr(thought, "description", "") or "")[:200]
        sig = f"{subject.lower()}|{description.lower()}"
        self._thoughts = (self._thoughts + [sig])[-MAX_THOUGHT_HISTORY:]
        if (len(self._thoughts) >= THOUGHT_REPEAT_THRESHOLD
                and len(set(self._thoughts[-THOUGHT_REPEAT_THRESHOLD:])) == 1):
            return self._fire(LoopType.REPETITIVE_THOUGHTS)
        return False

    def _fire(self, loop_type: LoopType) -> bool:
        self.last_loop_type = loop_type
        return True
