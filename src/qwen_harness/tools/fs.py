"""Filesystem read tools: list_directory, read_file, glob, grep_search.
Ports of tools/ls.ts, read-file.ts, glob.ts, grep.ts.

Common conventions to notice:
  - every path parameter must be ABSOLUTE (the model is told so in the
    system prompt; validation enforces it),
  - reads inside the workspace are permission-'allow' (no dialog), paths
    outside it are 'ask',
  - every tool caps its own output (entries, lines, chars) and *says so in
    the output text* so the model knows it saw a truncated view.
"""

from __future__ import annotations

import fnmatch
import os
import re
import time
from pathlib import Path

from ..config import Config
from .base import (DeclarativeTool, Kind, ToolError, ToolErrorType,
                   ToolInvocation, ToolResult)
from .names import ToolNames

MAX_ENTRY_COUNT = 100        # ls.ts
MAX_FILE_COUNT = 100         # glob.ts
GREP_MAX_OUTPUT_CHARS = 20_000
DEFAULT_READ_LINE_LIMIT = 1_000
DEFAULT_READ_CHAR_LIMIT = 25_000
ONE_DAY_S = 24 * 60 * 60

_IGNORED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "dist"}


def _require_absolute(path: str) -> str | None:
    if not os.path.isabs(path):
        return f"File path must be absolute, but was relative: {path}. You must provide an absolute path."
    return None


def _in_workspace(path: str, config: Config) -> bool:
    try:
        Path(path).resolve().relative_to(Path(config.target_dir).resolve())
        return True
    except ValueError:
        return False


class _WorkspacePermissionMixin:
    def get_default_permission(self) -> str:
        target = self.params.get("path") or self.params.get("file_path") or ""
        if not target:
            return "allow"
        return "allow" if _in_workspace(target, self._config) else "ask"


# ---------------------------------------------------------------------- ls


class LSTool(DeclarativeTool):
    name = ToolNames.LS
    display_name = "ListFiles"
    kind = Kind.SEARCH
    description = ("Lists the names of files and subdirectories directly within "
                   "a specified directory path. Use an absolute path.")
    parameter_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "The absolute path to the directory to list."},
            "ignore": {"type": "array", "items": {"type": "string"},
                       "description": "List of glob patterns to ignore."},
        },
        "required": ["path"],
    }

    def __init__(self, config: Config):
        self.config = config

    def validate_tool_param_values(self, params):
        return _require_absolute(params["path"])

    def create_invocation(self, params):
        return _LSInvocation(params, self.config)


class _LSInvocation(_WorkspacePermissionMixin, ToolInvocation):
    def __init__(self, params, config):
        super().__init__(params)
        self._config = config

    def get_description(self):
        return self.params["path"]

    async def execute(self, update_output=None) -> ToolResult:
        path = Path(self.params["path"])
        if not path.exists():
            return _error(ToolErrorType.FILE_NOT_FOUND, f"Directory not found: {path}")
        if not path.is_dir():
            return _error(ToolErrorType.PATH_IS_NOT_A_DIRECTORY,
                          f"Path is not a directory: {path}")
        ignore = self.params.get("ignore") or []
        entries = []
        for entry in path.iterdir():
            if any(fnmatch.fnmatch(entry.name, pattern) for pattern in ignore):
                continue
            entries.append(entry)
        # dirs first, then alphabetical
        entries.sort(key=lambda e: (not e.is_dir(), e.name.lower()))
        truncated = max(0, len(entries) - MAX_ENTRY_COUNT)
        shown = entries[:MAX_ENTRY_COUNT]
        lines = [f"[DIR] {e.name}" if e.is_dir() else e.name for e in shown]
        body = f"Listed {len(shown)} item(s) in {path}:\n---\n" + "\n".join(lines)
        if truncated:
            body += f"\n---\n[{truncated} items truncated] ..."
        return ToolResult(llm_content=body, return_display=f"Listed {len(shown)} item(s).")


# ---------------------------------------------------------------- read_file


class ReadFileTool(DeclarativeTool):
    name = ToolNames.READ_FILE
    display_name = "ReadFile"
    kind = Kind.READ
    max_output_chars = float("inf")  # self-managed truncation
    description = ("Reads and returns the content of a specified file. Handles "
                   "text files with optional line offset and limit for large files. "
                   "Use an absolute path.")
    parameter_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string",
                          "description": "The absolute path to the file to read."},
            "offset": {"type": "integer",
                       "description": "0-based line number to start reading from."},
            "limit": {"type": "integer",
                      "description": "Maximum number of lines to read."},
        },
        "required": ["file_path"],
    }

    def __init__(self, config: Config, read_tracker: "FileReadTracker | None" = None):
        self.config = config
        self.read_tracker = read_tracker

    def validate_tool_param_values(self, params):
        if err := _require_absolute(params["file_path"]):
            return err
        if (offset := params.get("offset")) is not None and offset < 0:
            return "Offset must be a non-negative number"
        if (limit := params.get("limit")) is not None and limit <= 0:
            return "Limit must be a positive number"
        return None

    def create_invocation(self, params):
        return _ReadFileInvocation(params, self.config, self.read_tracker)


class _ReadFileInvocation(_WorkspacePermissionMixin, ToolInvocation):
    def __init__(self, params, config, read_tracker):
        super().__init__(params)
        self._config = config
        self._read_tracker = read_tracker

    def get_description(self):
        return os.path.relpath(self.params["file_path"], self._config.target_dir)

    async def execute(self, update_output=None) -> ToolResult:
        path = Path(self.params["file_path"])
        if not path.is_file():
            return _error(ToolErrorType.FILE_NOT_FOUND,
                          f"File not found: {path}")
        try:
            text = path.read_text(errors="replace")
        except OSError as e:
            return _error(ToolErrorType.FILE_NOT_FOUND, f"Failed to read file: {e}")

        lines = text.splitlines()
        total = len(lines)
        offset = self.params.get("offset") or 0
        limit = self.params.get("limit") or self._config.truncate_tool_output_lines
        window = lines[offset:offset + limit]

        # per-read char budget, applied line-by-line with a cut marker
        char_budget = self._config.truncate_tool_output_threshold
        out_lines, used = [], 0
        for line in window:
            if used + len(line) > char_budget:
                out_lines.append(line[:max(0, char_budget - used)] + "... [truncated]")
                break
            out_lines.append(line)
            used += len(line) + 1
        content = "\n".join(out_lines)

        truncated = offset > 0 or len(out_lines) < total
        if self._read_tracker is not None:
            self._read_tracker.record_read(path, full=not truncated)
        if truncated:
            start, end = offset + 1, offset + len(out_lines)
            content = (f"Showing lines {start}-{end} of {total} total lines.\n\n---\n\n"
                       + content)
            display = f"Read lines {start}-{end} of {total} from {self.get_description()}"
        else:
            display = f"Read {total} lines from {self.get_description()}"
        return ToolResult(llm_content=content, return_display=display)


class FileReadTracker:
    """Session record of reads, feeding edit's prior-read enforcement."""

    def __init__(self):
        self._reads: dict[str, float] = {}

    def record_read(self, path: Path, full: bool = True) -> None:
        self._reads[str(path.resolve())] = time.time()

    def was_read(self, path: Path) -> bool:
        return str(path.resolve()) in self._reads

    def changed_since_read(self, path: Path) -> bool:
        key = str(path.resolve())
        if key not in self._reads:
            return False
        try:
            return path.stat().st_mtime > self._reads[key]
        except OSError:
            return False


# --------------------------------------------------------------------- glob


class GlobTool(DeclarativeTool):
    name = ToolNames.GLOB
    display_name = "Glob"
    kind = Kind.SEARCH
    description = ("Efficiently finds files matching specific glob patterns "
                   "(e.g., 'src/**/*.ts'), returning absolute paths sorted by "
                   "modification time (newest first).")
    parameter_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string",
                        "description": "The glob pattern to match against."},
            "path": {"type": "string",
                     "description": "Absolute path of the directory to search "
                                    "(defaults to the workspace root)."},
        },
        "required": ["pattern"],
    }

    def __init__(self, config: Config):
        self.config = config

    def create_invocation(self, params):
        return _GlobInvocation(params, self.config)


class _GlobInvocation(_WorkspacePermissionMixin, ToolInvocation):
    def __init__(self, params, config):
        super().__init__(params)
        self._config = config

    def get_description(self):
        return f"'{self.params['pattern']}'"

    async def execute(self, update_output=None) -> ToolResult:
        root = Path(self.params.get("path") or self._config.target_dir)
        pattern = self.params["pattern"]
        try:
            matches = [p for p in root.glob(pattern)
                       if p.is_file() and not _ignored(p, root)]
        except (OSError, ValueError) as e:
            return _error(ToolErrorType.GLOB_EXECUTION_ERROR, f"Glob failed: {e}")
        if not matches:
            return ToolResult(
                llm_content=f'No files found matching pattern "{pattern}" within {root}.',
                return_display="No files found.")

        # recency-aware sort: files touched in the last 24h newest-first,
        # older files after them, alphabetical (sortFileEntries)
        now = time.time()
        def sort_key(p: Path):
            mtime = p.stat().st_mtime
            recent = (now - mtime) < ONE_DAY_S
            return (0, -mtime, "") if recent else (1, 0, str(p))
        matches.sort(key=sort_key)

        truncated = max(0, len(matches) - MAX_FILE_COUNT)
        shown = matches[:MAX_FILE_COUNT]
        body = (f'Found {len(shown)} file(s) matching "{pattern}" within {root}, '
                "sorted by modification time (newest first):\n---\n"
                + "\n".join(str(p.resolve()) for p in shown))
        if truncated:
            body += f"\n---\n[{truncated} files truncated] ..."
        return ToolResult(llm_content=body, return_display=f"Found {len(shown)} file(s).")


def _ignored(path: Path, root: Path) -> bool:
    return any(part in _IGNORED_DIRS for part in path.relative_to(root).parts)


# -------------------------------------------------------------------- grep


class GrepTool(DeclarativeTool):
    name = ToolNames.GREP
    display_name = "Grep"
    kind = Kind.SEARCH
    max_output_chars = GREP_MAX_OUTPUT_CHARS
    description = ("Searches for a regular expression pattern within the content "
                   "of files, returning matching lines as file:line:content. "
                   "Case-insensitive by default.")
    parameter_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string",
                        "description": "The regular expression to search for."},
            "path": {"type": "string",
                     "description": "Absolute path of a file or directory to search."},
            "glob": {"type": "string",
                     "description": "Glob pattern to filter which files are searched."},
            "limit": {"type": "integer",
                      "description": "Maximum number of matching lines to return."},
        },
        "required": ["pattern"],
    }

    def __init__(self, config: Config):
        self.config = config

    def validate_tool_param_values(self, params):
        try:
            re.compile(params["pattern"])
        except re.error as e:
            return f"Invalid regular expression pattern: {e}"
        if (limit := params.get("limit")) is not None and limit < 1:
            return "Limit must be >= 1"
        return None

    def create_invocation(self, params):
        return _GrepInvocation(params, self.config)


class _GrepInvocation(_WorkspacePermissionMixin, ToolInvocation):
    """Upstream tries ripgrep, then git grep, then system grep, then this:
    a pure in-process line scan. The study port keeps only the last rung —
    the observable contract (output format, caps) is identical."""

    def __init__(self, params, config):
        super().__init__(params)
        self._config = config

    def get_description(self):
        return f"'{self.params['pattern']}'"

    async def execute(self, update_output=None) -> ToolResult:
        pattern = re.compile(self.params["pattern"], re.IGNORECASE)
        root = Path(self.params.get("path") or self._config.target_dir)
        file_glob = self.params.get("glob")
        limit = min(self.params.get("limit") or self._config.truncate_tool_output_lines,
                    self._config.truncate_tool_output_lines)

        files = [root] if root.is_file() else [
            p for p in root.rglob(file_glob or "*")
            if p.is_file() and not _ignored(p, root)]
        rows: list[str] = []
        for f in sorted(files):
            try:
                for lineno, line in enumerate(
                        f.read_text(errors="replace").splitlines(), 1):
                    if pattern.search(line):
                        rows.append(f"{f}:{lineno}:{line.strip()[:500]}")
                        if len(rows) >= limit:
                            break
            except OSError:
                continue
            if len(rows) >= limit:
                break

        where = f"in {root}" + (f' (filter: "{file_glob}")' if file_glob else "")
        if not rows:
            return ToolResult(
                llm_content=f'No matches found for pattern "{self.params["pattern"]}" {where}.',
                return_display="No matches found.")
        body = (f'Found {len(rows)} match(es) for pattern '
                f'"{self.params["pattern"]}" {where}:\n---\n' + "\n".join(rows))
        return ToolResult(llm_content=body, return_display=f"Found {len(rows)} match(es).")


def _error(error_type: ToolErrorType, message: str) -> ToolResult:
    return ToolResult(llm_content=f"Error: {message}", return_display=message,
                      error=ToolError(message=message, type=error_type))
