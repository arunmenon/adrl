"""B7 — feature extractor (design doc §5.2).

Turns a user_turn into a small vector of facts in microseconds. No LLM calls,
no embeddings — heuristic on purpose. The intent lexicons are shared with the
miner's scenario matchers so offline replay and the live path agree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Verb classes, ordered easy -> hard. First match wins.
VERB_CLASSES: list[tuple[str, float, re.Pattern]] = [
    # (name, base difficulty score, pattern)
    ("trivial", 0.10, re.compile(
        r"\b(commit message|typo|rename|readme|title|format|lint)\b", re.I)),
    ("explain", 0.20, re.compile(
        r"^(what|why|how|explain|describe|tell me|walk me|give me (a |an )?(overview|summary))", re.I)),
    ("small_edit", 0.35, re.compile(
        r"^(add|update|change|remove|delete|move|tweak)\b", re.I)),
    ("write", 0.45, re.compile(
        r"^(write|create|implement|build|generate|make)\b", re.I)),
    ("fix", 0.55, re.compile(
        r"\b(fix|debug|resolve|failing|broken|crash|error|investigate|why (is|does)nt?)\b", re.I)),
    ("hard", 0.85, re.compile(
        r"\b(refactor|migrat|redesign|architect|overhaul|rewrite)\b", re.I)),
]

SCOPE_BROAD = re.compile(
    r"\b(across (the )?(codebase|repo|services|modules|files)|everywhere|all (files|modules|call ?sites)|entire)\b",
    re.I)
SCOPE_NARROW = re.compile(r"\b(this (function|file|line|variable|test)|just|only)\b", re.I)

# A whole-message approval / continuation ("go ahead", "do it", "try now", "lgtm").
# These are the user greenlighting the agent's proposed action, NOT a fresh task —
# routing them by cold difficulty misfires (a 2-word "go ahead" has no signal), so
# the policy sticks them to the current route instead of classifying them.
TERSE_CONTINUE = re.compile(
    r"^\W*(go\s*ahead|do it|do so|go for it|ok(ay)?|ye(s|p|ah)|sure|lgtm|"
    r"ship it|proceed|continue|carry on|try now|and now|next|fine|please do|"
    r"sounds good|do that|make it so)\W*$", re.I)


@dataclass
class TurnFeatures:
    """Everything the policy engine is allowed to reason about."""

    # intent (from the isolated human sentence only)
    verb_class: str = "unknown"
    verb_score: float = 0.5          # unknown -> middle of the band, on purpose
    instruction_text: str = ""       # the isolated human sentence (for the LLM middle-resolver)
    broad_scope: bool = False
    narrow_scope: bool = False
    # context
    context_tokens: int = 0
    is_terse_continuation: bool = False   # a bare approval/greenlight of prior work
    # trajectory (the signal chat-routers don't have)
    turn_index: int = 0
    recent_errors: int = 0           # is_error tool_results in this session's recent turns
    recent_edit_failures: int = 0
    prev_turn_interrupted: bool = False
    # constraints
    privacy_pinned: bool = False
    escalated_this_episode: bool = False
    extra: dict = field(default_factory=dict)


def classify_intent(text: str) -> tuple[str, float]:
    t = text.strip()
    for name, score, pat in VERB_CLASSES:
        if pat.search(t):
            return name, score
    return "unknown", 0.5


CONTEXT_TOKEN_THRESHOLD = 20_000  # big working sets correlate with harder work


def intent_score(f: TurnFeatures) -> float:
    """Intent-only difficulty in [0,1]: verb base score + scope adjustments +
    the big-context nudge. NO trajectory signals (edit failures, recent errors,
    prior interrupt) — this is what the instruction ALONE says about difficulty.

    Single source shared by ``heuristic_score`` (live path, adds trajectory on
    top) and ``shadow_classifier.intent_only_score`` (offline harness), so the
    two can never drift apart again.
    """
    s = f.verb_score
    if f.broad_scope:
        s += 0.20
    if f.narrow_scope:
        s -= 0.10
    if f.context_tokens > CONTEXT_TOKEN_THRESHOLD:
        s += 0.10
    return max(0.0, min(1.0, s))


def heuristic_score(f: TurnFeatures) -> float:
    """Difficulty in [0,1]. Deliberately dumb: weighted facts, no learning.
    Thresholds T_EASY/T_HARD (policy.py) cut this into three bands."""
    s = intent_score(f)
    if f.recent_edit_failures >= 1:
        s += 0.15                     # session already struggling with edits
    if f.recent_errors >= 3:
        s += 0.10
    if f.prev_turn_interrupted:
        s += 0.30                     # escalate-on-retry: the strongest signal (§5.5)
    return max(0.0, min(1.0, s))


def extract(instruction_text: str, *, context_tokens: int = 0, turn_index: int = 0,
            recent_errors: int = 0, recent_edit_failures: int = 0,
            prev_turn_interrupted: bool = False, privacy_pinned: bool = False,
            escalated_this_episode: bool = False) -> TurnFeatures:
    verb, score = classify_intent(instruction_text or "")
    text = instruction_text or ""
    return TurnFeatures(
        verb_class=verb, verb_score=score, instruction_text=text,
        is_terse_continuation=bool(TERSE_CONTINUE.match(text.strip())),
        broad_scope=bool(SCOPE_BROAD.search(text)),
        narrow_scope=bool(SCOPE_NARROW.search(text)),
        context_tokens=context_tokens, turn_index=turn_index,
        recent_errors=recent_errors, recent_edit_failures=recent_edit_failures,
        prev_turn_interrupted=prev_turn_interrupted,
        privacy_pinned=privacy_pinned,
        escalated_this_episode=escalated_this_episode,
    )
