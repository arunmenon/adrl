"""A1 — streaming parser for Claude Code transcript JSONL files.

Ground rules (from the Phase 0 corpus profile):
- Files reach 30MB: always stream line-by-line, never slurp.
- Every field is optional: schema drifted across CC 2.1.150 -> 2.1.199, and
  non-conversational record types (mode, ai-title, last-prompt, ...) carry no
  envelope at all. Use .get() everywhere.
- Unknown record types must be counted, not crashed on.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# Record types observed in the corpus (CC 2.1.150 - 2.1.199).
CONVERSATIONAL_TYPES = {"user", "assistant", "system", "attachment"}
UTILITY_EVENT_TYPES = {
    "file-history-snapshot",
    "mode",
    "permission-mode",
    "last-prompt",
    "ai-title",
    "queue-operation",
    "pr-link",
    "summary",
}
KNOWN_TYPES = CONVERSATIONAL_TYPES | UTILITY_EVENT_TYPES

_AGENT_FILE_RE = re.compile(r"agent-([0-9a-f]+)\.jsonl$")
_WORKFLOW_DIR_RE = re.compile(r"(wf_[a-z0-9-]+)")


@dataclass
class SourceFile:
    """One transcript file plus everything derivable from its path."""

    path: Path
    project: str
    session_id: str
    kind: str  # "main" | "subagent"
    agent_id: str | None = None
    workflow_id: str | None = None


@dataclass
class ParseStats:
    files: int = 0
    lines: int = 0
    bad_lines: int = 0
    records_by_type: Counter = field(default_factory=Counter)
    unknown_types: Counter = field(default_factory=Counter)
    files_by_kind: Counter = field(default_factory=Counter)
    duplicate_uuids: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "lines": self.lines,
            "bad_lines": self.bad_lines,
            "records_by_type": dict(self.records_by_type),
            "unknown_types": dict(self.unknown_types),
            "files_by_kind": dict(self.files_by_kind),
            "duplicate_uuids": self.duplicate_uuids,
        }


def iter_source_files(corpus_root: Path) -> Iterator[SourceFile]:
    """Enumerate transcript files under a snapshot of ~/.claude/projects.

    Layout: <project>/<sessionId>.jsonl                        (main session)
            <project>/<sessionId>/subagents/**/agent-*.jsonl   (subagent)
            <project>/<sessionId>/subagents/workflows/wf_*/journal.jsonl (skipped)

    Sorted by mtime ascending so that, when a resumed session replays records,
    the original file is seen first and the replay dedups against it.
    """
    found: list[tuple[float, SourceFile]] = []
    for project_dir in sorted(corpus_root.iterdir()):
        if not project_dir.is_dir():
            continue
        project = project_dir.name
        for main in project_dir.glob("*.jsonl"):
            found.append(
                (
                    main.stat().st_mtime,
                    SourceFile(main, project, main.stem, "main"),
                )
            )
        for sub in project_dir.glob("*/subagents/**/agent-*.jsonl"):
            session_id = sub.relative_to(project_dir).parts[0]
            m = _AGENT_FILE_RE.search(sub.name)
            wf = _WORKFLOW_DIR_RE.search(str(sub))
            found.append(
                (
                    sub.stat().st_mtime,
                    SourceFile(
                        sub,
                        project,
                        session_id,
                        "subagent",
                        agent_id=m.group(1) if m else None,
                        workflow_id=wf.group(1) if wf else None,
                    ),
                )
            )
    for _, sf in sorted(found, key=lambda t: t[0]):
        yield sf


def iter_records(source: SourceFile, stats: ParseStats) -> Iterator[dict[str, Any]]:
    """Stream records from one file, counting types and bad lines."""
    stats.files += 1
    stats.files_by_kind[source.kind] += 1
    with source.path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            stats.lines += 1
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                stats.bad_lines += 1
                continue
            if not isinstance(rec, dict):
                stats.bad_lines += 1
                continue
            rtype = rec.get("type", "<missing>")
            stats.records_by_type[rtype] += 1
            if rtype not in KNOWN_TYPES:
                stats.unknown_types[rtype] += 1
            yield rec


# ---------------------------------------------------------------------------
# Content helpers — message.content is either a plain string or a block list.
# ---------------------------------------------------------------------------


def content_blocks(message: Any) -> list[dict[str, Any]]:
    """Normalize message.content to a list of block dicts."""
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def text_of(message: Any) -> str:
    """Concatenated text blocks of a message."""
    return "\n".join(
        b.get("text", "") for b in content_blocks(message) if b.get("type") == "text"
    )


def tool_result_blocks(message: Any) -> list[dict[str, Any]]:
    return [b for b in content_blocks(message) if b.get("type") == "tool_result"]


def tool_use_blocks(message: Any) -> list[dict[str, Any]]:
    return [b for b in content_blocks(message) if b.get("type") == "tool_use"]


def version_bucket(rec: dict[str, Any]) -> str:
    """Coarse CC-version bucket for schema-drift-aware reporting."""
    v = rec.get("version")
    if not isinstance(v, str):
        return "unknown"
    parts = v.split(".")
    return ".".join(parts[:2]) + ".x" if len(parts) >= 3 else v
