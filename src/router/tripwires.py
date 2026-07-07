"""C1 — the trip-wire evaluator (design doc §5.5).

Stateful, per-turn, mechanical. Watches the sequence of
``(assistant response, tool_results)`` inside ONE ``cascade=True`` turn and fires
the instant the local rung is *mechanically* failing — no LLM-judge, all cheap
and deterministic (a judge smart enough to grade the work costs about as much as
doing the work, §5.5).

Six wires; thresholds are verbatim from the §5.5 table:

    wire            threshold                          type        routes to
    -------------------------------------------------------------------------
    parse_schema    2 malformed tool calls / turn      dialect     registry (§5.4)
    edit_apply      2 "String to replace not found"    dialect     registry
                      in an is_error tool_result
    loop            3 identical canonical calls         difficulty  router
                      (within a 6-action window)
    no_progress     6 actions, none reading a new       difficulty  router
                      file / advancing a diff / emitting
                      a new (non-error) output hash
    turn_budget     caller-supplied token / wall-clock  cost        router
    user_interrupt  1 event                             quality     escalate retry

Why the TYPE is stored with every outcome (B5 / §5.7 label-honesty): a *dialect*
failure — the local model literally can't speak Claude Code's edit/tool grammar
— must train the capability registry, NOT the difficulty model. Otherwise a
harness-dialect miss is mislabeled "this turn was hard" and poisons the router.
``.fired()`` therefore returns the type as a first-class field, and
``routes_to_registry()`` encodes the exact §5.7 split.

Detection points follow §5.4's post-call path: tool-call parse validity is read
off the *response*; edit-failure evidence off the *next* request's tool_results.
The edit wire requires ``is_error`` truthy **on purpose** — the raw marker string
also appears in documents that merely *discuss* edit failures (verified against
the corpus: 3 turns the miner counted were Read results of this very design doc,
not real failures), and ``is_error`` is what separates a failure from a mention.

Canonicalization for the loop wire reuses ``miner.scenarios.canonical_call`` so
the live path and offline replay (§5.7) can never drift on what counts as "the
same call" (trailing-slash paths, key order, whitespace all collapse).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple, Optional

from miner.scenarios import canonical_call

# Marker strings mirrored from miner.turns (kept local so the live hot-path does
# not import the whole offline turn-builder just for two constants).
EDIT_FAIL_MARKER = "String to replace not found in file"
INTERRUPT_PREFIX = "[Request interrupted by user"

# Secondary, best-effort parse signal: schema rejections that surface as an
# is_error tool_result rather than as a malformed response block. Kept tight and
# tool-schema-specific to avoid firing on ordinary command errors. The reliable
# parse signal is the response-side malformed-block check below; these markers
# are harness-version-dependent and only used when the edit marker is absent.
SCHEMA_ERROR_MARKERS = (
    "input validation error",
    "does not match the required schema",
    "invalid tool name",
)

# Tool-name families (compared lowercase) for the no-progress wire's three
# progress signals. Generous on purpose: unknown tools still make progress via a
# new non-error output hash, so the wire only accrues on genuinely stuck actions.
READ_TOOLS = {"read", "read_file", "readfile", "view", "view_file", "cat", "open_file"}
EDIT_TOOLS = {
    "edit", "multiedit", "write", "str_replace", "str_replace_editor",
    "str_replace_based_edit_tool", "apply_patch", "create_file", "notebookedit",
}
_RESOURCE_KEYS = ("file_path", "path", "filename", "notebook_path", "file")


class TripwireType(str, Enum):
    """The routing-relevant class of a trip-wire (stored with each outcome).

    The load-bearing split is DIALECT vs everything else (§5.7/B5): dialect
    failures train the capability registry; the rest train the difficulty model.
    """

    DIALECT = "dialect"        # parse_schema, edit_apply -> capability registry
    DIFFICULTY = "difficulty"  # loop, no_progress        -> difficulty model
    COST = "cost"              # turn_budget               -> router (runaway cost)
    QUALITY = "quality"        # user_interrupt            -> escalate the retry


TRIPWIRE_TYPES: dict[str, TripwireType] = {
    "parse_schema": TripwireType.DIALECT,
    "edit_apply": TripwireType.DIALECT,
    "loop": TripwireType.DIFFICULTY,
    "no_progress": TripwireType.DIFFICULTY,
    "turn_budget": TripwireType.COST,
    "user_interrupt": TripwireType.QUALITY,
}


def routes_to_registry(tripwire_type: TripwireType) -> bool:
    """Dialect failures train the capability registry; all else the router (§5.7)."""
    return tripwire_type is TripwireType.DIALECT


@dataclass(frozen=True)
class TurnBudget:
    """Parameterized runaway-turn guard (§5.5): "2x median tokens for this intent
    class, or 90s wall-clock". Both bounds optional; a ``None`` bound is never
    checked, so the wire stays inert until the caller supplies a class-specific
    budget. Nothing is hardcoded here — the numbers live with the caller."""

    max_tokens: Optional[int] = None
    max_wall_clock_s: Optional[float] = None


class TripwireHit(NamedTuple):
    """A fired trip-wire. ``(name, type, detail)`` — ``type`` is first-class so the
    escalation controller can route dialect vs difficulty without re-deriving it."""

    name: str
    type: TripwireType
    detail: dict


class TripwireState:
    """Per-turn, per-session trip-wire evaluator (§5.5).

    Lifecycle: construct once per ``cascade=True`` turn (or ``reset()`` between
    turns). Feed the turn's action stream in order — ``observe_response`` for each
    assistant message, ``observe_tool_results`` for the tool_results that come
    back — and poll ``fired()``. The first wire to cross threshold *latches*:
    escalation happens at that action boundary and the evaluator is done until
    ``reset()``.
    """

    def __init__(
        self,
        budget: Optional[TurnBudget] = None,
        *,
        parse_strikes: int = 2,
        edit_strikes: int = 2,
        loop_repeats: int = 3,
        loop_window: int = 6,
        no_progress_limit: int = 6,
    ) -> None:
        self.budget = budget
        self.parse_strikes = parse_strikes
        self.edit_strikes = edit_strikes
        self.loop_repeats = loop_repeats
        self.loop_window = loop_window
        self.no_progress_limit = no_progress_limit
        self.reset()

    # ── lifecycle ───────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Clear all per-turn state. Config (budget, thresholds) is preserved."""
        self._hit: Optional[TripwireHit] = None
        self._parse = 0
        self._edit = 0
        self._call_history: list[str] = []        # canonical hashes, issue order
        self._max_loop = 0
        self._pending: dict[str, dict] = {}        # tool_use_id -> {name, input}
        self._seen_read_paths: set[str] = set()
        self._seen_output_hashes: set[str] = set()
        self._no_progress = 0
        self._actions = 0
        self._tokens = 0
        self._elapsed_s = 0.0

    def fired(self) -> Optional[TripwireHit]:
        """The latched hit, if any wire has crossed threshold this turn."""
        return self._hit

    @property
    def strikes(self) -> dict:
        """Snapshot shaped for ``SessionState.strikes`` (§5.6: {parse, edit, loop, noprog})."""
        return {
            "parse": self._parse,
            "edit": self._edit,
            "loop": self._max_loop,
            "noprog": self._no_progress,
        }

    # ── ingestion ───────────────────────────────────────────────────────────
    def observe_response(
        self,
        resp_content_blocks: Any,
        *,
        output_tokens: int = 0,
        elapsed_s: Optional[float] = None,
    ) -> Optional[TripwireHit]:
        """Consume one assistant response. Checks: parse/bad-schema (malformed
        tool_use blocks), identical-call loops, and the token/wall-clock budget."""
        self._tokens += max(0, int(output_tokens or 0))
        if elapsed_s is not None:
            self._elapsed_s = max(self._elapsed_s, float(elapsed_s))

        for blk in _blocks(resp_content_blocks):
            if blk.get("type") != "tool_use":
                continue
            name = blk.get("name")
            args = blk.get("input")

            # (parse_schema) a tool call the harness cannot execute: no name, or
            # input that is not a JSON object. Real Anthropic tool_use always has
            # a dict input, so this fires on local-model malformed output only.
            if not isinstance(name, str) or not name.strip() or not isinstance(args, dict):
                self._parse += 1
                if self._parse >= self.parse_strikes:
                    self._trip("parse_schema", {
                        "strikes": self._parse,
                        "reason": "malformed tool_use block",
                        "name": name if isinstance(name, str) else repr(name),
                    })
                continue

            # (loop) canonicalize exactly like the miner (S4) so cosmetically
            # different calls still count as identical.
            digest = canonical_call(name, args)
            self._call_history.append(digest)
            window = self._call_history[-self.loop_window:]
            reps = window.count(digest)
            if reps > self._max_loop:
                self._max_loop = reps
            if reps >= self.loop_repeats:
                self._trip("loop", {
                    "repeats": reps,
                    "call": name,
                    "canonical": digest,
                    "window": self.loop_window,
                })

            # remember the call so its result can be scored for progress
            tid = blk.get("id")
            if isinstance(tid, str):
                self._pending[tid] = {"name": name, "input": args}

        self._check_budget()
        return self._hit

    def observe_tool_results(
        self,
        user_content_blocks: Any,
        *,
        elapsed_s: Optional[float] = None,
    ) -> Optional[TripwireHit]:
        """Consume the tool_results (and any text) of one user request. Checks:
        edit-apply failures, schema-rejection errors, no-progress accounting, and
        a user interrupt delivered as leading text."""
        if elapsed_s is not None:
            self._elapsed_s = max(self._elapsed_s, float(elapsed_s))

        # An interrupt can arrive as a bare string user message.
        if isinstance(user_content_blocks, str):
            if user_content_blocks.startswith(INTERRUPT_PREFIX):
                self._trip("user_interrupt", {"event": "interrupt",
                                              "text": user_content_blocks[:120]})
            self._check_budget()
            return self._hit

        for blk in _blocks(user_content_blocks):
            btype = blk.get("type")

            if btype == "text":
                text = blk.get("text")
                if isinstance(text, str) and text.startswith(INTERRUPT_PREFIX):
                    self._trip("user_interrupt", {"event": "interrupt", "text": text[:120]})
                continue

            if btype != "tool_result":
                continue

            blob = _content_str(blk.get("content"))
            is_error = bool(blk.get("is_error"))
            tid = blk.get("tool_use_id")
            use = self._pending.pop(tid, None) if isinstance(tid, str) else None

            # (edit_apply) dialect failure — REQUIRES is_error (a real failure,
            # not a document that merely quotes the marker).
            if is_error and EDIT_FAIL_MARKER in blob:
                self._edit += 1
                if self._edit >= self.edit_strikes:
                    self._trip("edit_apply", {
                        "strikes": self._edit,
                        "marker": EDIT_FAIL_MARKER,
                        "evidence": blob[:120],
                    })
            # (parse_schema, secondary) schema rejection surfaced as an error.
            elif is_error and _has_schema_error(blob):
                self._parse += 1
                if self._parse >= self.parse_strikes:
                    self._trip("parse_schema", {
                        "strikes": self._parse,
                        "reason": "schema-rejected tool call",
                        "evidence": blob[:120],
                    })

            # (no_progress) every completed action counts.
            self._score_progress(use, is_error, blob)

        self._check_budget()
        return self._hit

    def note_interrupt(self, detail: str = "") -> Optional[TripwireHit]:
        """Explicit user-interrupt / immediate-rephrase signal — fires on 1 event
        (§5.5: the strongest quality signal you have; escalate the retry)."""
        self._trip("user_interrupt", {"event": "interrupt", "text": str(detail)[:120]})
        return self._hit

    # ── wire helpers ────────────────────────────────────────────────────────
    def _trip(self, name: str, detail: dict) -> None:
        """Latch the first wire to cross threshold (escalation is a one-shot at
        the next action boundary). Later trips do not overwrite it."""
        if self._hit is not None:
            return
        detail = dict(detail)
        detail.setdefault("type", TRIPWIRE_TYPES[name].value)
        self._hit = TripwireHit(name, TRIPWIRE_TYPES[name], detail)

    def _check_budget(self) -> None:
        if self._hit is not None or self.budget is None:
            return
        budget = self.budget
        if budget.max_tokens is not None and self._tokens > budget.max_tokens:
            self._trip("turn_budget", {
                "reason": "tokens", "tokens": self._tokens,
                "limit": budget.max_tokens, "elapsed_s": self._elapsed_s,
            })
        elif budget.max_wall_clock_s is not None and self._elapsed_s > budget.max_wall_clock_s:
            self._trip("turn_budget", {
                "reason": "wall_clock", "elapsed_s": self._elapsed_s,
                "limit": budget.max_wall_clock_s, "tokens": self._tokens,
            })

    def _score_progress(self, use: Optional[dict], is_error: bool, blob: str) -> None:
        """Advance no-progress accounting for one completed action. Progress =
        a new file read OR a successful diff OR a new (non-error) output hash;
        an error result is never progress. Reset the counter on progress; fire
        after ``no_progress_limit`` consecutive stuck actions."""
        self._actions += 1
        progressed = False

        if not is_error:
            name = (use or {}).get("name") or ""
            lname = name.lower() if isinstance(name, str) else ""
            args = (use or {}).get("input") or {}

            if lname in READ_TOOLS:
                key = _resource_key(args)
                if key is not None:
                    if key not in self._seen_read_paths:
                        progressed = True
                    self._seen_read_paths.add(key)

            if lname in EDIT_TOOLS:
                progressed = True  # a successful edit is a diff advance

            digest = hashlib.sha1(blob.encode("utf-8", "replace")).hexdigest()
            if digest not in self._seen_output_hashes:
                progressed = True
            self._seen_output_hashes.add(digest)

        if progressed:
            self._no_progress = 0
        else:
            self._no_progress += 1
            if self._no_progress >= self.no_progress_limit:
                self._trip("no_progress", {
                    "actions": self._no_progress,
                    "total_actions": self._actions,
                })


# ── module-level helpers ────────────────────────────────────────────────────
def _blocks(content: Any) -> list[dict]:
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def _content_str(content: Any) -> str:
    """Normalize a tool_result's content (string, or a list of blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for b in content:
            if isinstance(b, dict):
                parts.append(str(b.get("text", b.get("content", ""))))
            else:
                parts.append(str(b))
        return " ".join(parts)
    if content is None:
        return ""
    return str(content)


def _resource_key(args: dict) -> Optional[str]:
    for key in _RESOURCE_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value:
            return value.rstrip("/")  # match canonical_call's trailing-slash norm
    return None


def _has_schema_error(blob: str) -> bool:
    low = blob.lower()
    return any(marker in low for marker in SCHEMA_ERROR_MARKERS)
