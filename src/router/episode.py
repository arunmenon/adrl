"""Conservative episode-boundary detection for hysteresis release."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .features import TurnFeatures
from .policy import SessionState


COMPLETION_PHRASE = re.compile(
    r"\b(?:that(?:'s| is) done|done with (?:that|this)|(?:we(?:'re| are) )?finished"
    r"|task (?:is )?complete|great[,;:]?\s+(?:now|next)|new task|next task)\b",
    re.I,
)
FILE_REFERENCE = re.compile(
    r"(?<![\w.-])(?:[\w.-]+/)+[\w.-]+|(?<![\w.-])[\w.-]+\.(?:py|js|ts|tsx|jsx|"
    r"go|rs|java|kt|swift|rb|php|md|yaml|yml|toml|json|sql)(?![\w.-])",
    re.I,
)


@dataclass(frozen=True)
class EpisodeBoundary:
    is_boundary: bool
    score: int
    previous_clean: bool
    completion_phrase: bool
    intent_changed: bool
    low_file_overlap: bool


def file_references(text: str) -> tuple[str, ...]:
    return tuple(sorted({match.group(0).lower() for match in FILE_REFERENCE.finditer(text)}))


def detect_episode_boundary(features: TurnFeatures, session: SessionState) -> EpisodeBoundary:
    """Require three of four local signals; uncertainty keeps hysteresis active."""
    previous_clean = bool(session.last_turn_clean)
    completion = bool(COMPLETION_PHRASE.search(features.instruction_text or ""))
    previous_verb = str(session.episode_verb_class or "")
    current_verb = str(features.verb_class or "")
    intent_changed = (
        bool(previous_verb) and current_verb not in {"", "unknown"}
        and current_verb != previous_verb
    )
    current_files = set(file_references(features.instruction_text or ""))
    previous_files = set(session.episode_files or ())
    low_overlap = bool(current_files and previous_files and current_files.isdisjoint(previous_files))
    score = sum((previous_clean, completion, intent_changed, low_overlap))
    return EpisodeBoundary(
        is_boundary=score >= 3,
        score=score,
        previous_clean=previous_clean,
        completion_phrase=completion,
        intent_changed=intent_changed,
        low_file_overlap=low_overlap,
    )
