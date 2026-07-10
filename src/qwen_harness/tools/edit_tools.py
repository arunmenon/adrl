"""Mutating file tools: edit + write_file. Ports of tools/edit.ts and
tools/write-file.ts.

Study notes:
  - Both produce an 'edit'-type confirmation carrying a unified diff, which
    is exactly what AUTO_EDIT mode auto-approves.
  - edit's contract is exact-literal old_string matching with counted
    occurrences: 0 occurrences is an error that *teaches* the model to
    re-read the file; >1 without replace_all is an error naming the count.
    A normalization fallback (trailing-whitespace-tolerant line matching)
    rescues near-misses before giving up.
  - old_string='' means "create this file" — and creating an existing file
    is its own error type.
  - Prior-read enforcement (fork addition): you may not edit a file this
    session hasn't read, or one that changed on disk since the read.
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path

from ..config import Config
from .base import (ConfirmationDetails, DeclarativeTool, FileDiff, Kind,
                   ToolError, ToolErrorType, ToolInvocation, ToolResult)
from .fs import FileReadTracker, _require_absolute
from .names import ToolNames

SNIPPET_CONTEXT_LINES = 4


def _unified_diff(old: str, new: str, name: str) -> str:
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"Current/{name}", tofile=f"Proposed/{name}"))


def _error(error_type: ToolErrorType, message: str) -> ToolResult:
    return ToolResult(llm_content=f"Error: {message}", return_display=message,
                      error=ToolError(message=message, type=error_type))


def _check_prior_read(path: Path, tracker: FileReadTracker | None) -> ToolResult | None:
    if tracker is None or not path.exists():
        return None
    if not tracker.was_read(path):
        return _error(ToolErrorType.EXECUTION_FAILED,
                      f"You must read the file {path} with the read_file tool "
                      "before modifying it in this session.")
    if tracker.changed_since_read(path):
        return _error(ToolErrorType.EXECUTION_FAILED,
                      f"File {path} has changed since it was last read. "
                      "Read it again before modifying it.")
    return None


# --------------------------------------------------------------------- edit


class EditTool(DeclarativeTool):
    name = ToolNames.EDIT
    display_name = "Edit"
    kind = Kind.EDIT
    description = (
        "Replaces text within a file. Replaces a single occurrence of old_string "
        "by default, or every occurrence when replace_all is true. This tool "
        "requires providing significant context around the change to ensure "
        "precise targeting. Always use the read_file tool to examine the file's "
        "current content before attempting a text replacement. Setting "
        "old_string to an empty string creates a new file.")
    parameter_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string",
                          "description": "The absolute path to the file to modify."},
            "old_string": {"type": "string",
                           "description": "The exact literal text to replace. "
                                          "Include enough context to match uniquely."},
            "new_string": {"type": "string",
                           "description": "The exact literal text to replace "
                                          "old_string with."},
            "replace_all": {"type": "boolean",
                            "description": "Replace every occurrence (default false)."},
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    def __init__(self, config: Config, read_tracker: FileReadTracker | None = None):
        self.config = config
        self.read_tracker = read_tracker

    def validate_tool_param_values(self, params):
        return _require_absolute(params["file_path"])

    def create_invocation(self, params):
        return _EditInvocation(params, self.config, self.read_tracker)


class _EditInvocation(ToolInvocation):
    def __init__(self, params, config, tracker):
        super().__init__(params)
        self._config = config
        self._tracker = tracker

    def get_description(self):
        rel = os.path.relpath(self.params["file_path"], self._config.target_dir)
        return f"{rel}: replace text"

    def get_default_permission(self):
        return "ask"

    def get_confirmation_details(self):
        path = Path(self.params["file_path"])
        old_content = path.read_text(errors="replace") if path.is_file() else ""
        proposed, _err = self._compute(old_content, path.exists())
        rel = os.path.relpath(path, self._config.target_dir)
        return ConfirmationDetails(
            type="edit", title=f"Confirm Edit: {rel}", file_name=rel,
            file_diff=_unified_diff(old_content, proposed if proposed is not None
                                    else old_content, rel))

    def _compute(self, content: str, exists: bool) -> tuple[str | None, ToolResult | None]:
        old, new = self.params["old_string"], self.params["new_string"]
        replace_all = bool(self.params.get("replace_all"))

        if old == "":
            if exists:
                return None, _error(
                    ToolErrorType.ATTEMPT_TO_CREATE_EXISTING_FILE,
                    "File already exists, cannot create. Use a non-empty "
                    "old_string to edit its content.")
            return new, None

        count = content.count(old)
        if count == 0:
            # flexible fallback: trailing-whitespace-tolerant line matching
            replaced = _flexible_replace(content, old, new, replace_all)
            if replaced is not None:
                return replaced, None
            return None, _error(
                ToolErrorType.EDIT_NO_OCCURRENCE_FOUND,
                "Failed to edit, could not find the string to replace. Ensure "
                "you're not escaping content incorrectly and check whitespace, "
                "indentation, and context. Use read_file tool to verify.")
        if count > 1 and not replace_all:
            return None, _error(
                ToolErrorType.EDIT_EXPECTED_OCCURRENCE_MISMATCH,
                f"Found {count} occurrences of old_string but replace_all was "
                "not enabled. Provide more context to match uniquely or set "
                "replace_all to true.")
        result = content.replace(old, new) if replace_all else content.replace(old, new, 1)
        if result == content:
            return None, _error(ToolErrorType.EDIT_NO_CHANGE,
                                "No changes to apply: old_string and new_string "
                                "produce identical content.")
        return result, None

    async def execute(self, update_output=None) -> ToolResult:
        path = Path(self.params["file_path"])
        if blocked := _check_prior_read(path, self._tracker):
            return blocked
        exists = path.is_file()
        content = path.read_text(errors="replace") if exists else ""
        new_content, failure = self._compute(content, path.exists())
        if failure:
            return failure

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_content)
        if self._tracker:
            self._tracker.record_read(path)  # our own write shouldn't trip TOCTOU
        rel = os.path.relpath(path, self._config.target_dir)

        if not exists:
            message = f"Created new file: {path} with provided content."
        else:
            message = f"The file: {path} has been updated."
            message += "\n" + _snippet(new_content, self.params["new_string"])
        return ToolResult(
            llm_content=message,
            return_display=FileDiff(file_diff=_unified_diff(content, new_content, rel),
                                    file_name=rel, original_content=content,
                                    new_content=new_content))


def _flexible_replace(content: str, old: str, new: str, replace_all: bool) -> str | None:
    """Line-based match tolerant of trailing whitespace (editHelper's
    middle rung; upstream also tries unicode punctuation normalization)."""
    content_lines = content.split("\n")
    old_lines = old.split("\n")
    if not old_lines:
        return None
    stripped_old = [line.rstrip() for line in old_lines]
    matches = []
    for i in range(len(content_lines) - len(old_lines) + 1):
        window = [line.rstrip() for line in content_lines[i:i + len(old_lines)]]
        if window == stripped_old:
            matches.append(i)
    if not matches or (len(matches) > 1 and not replace_all):
        return None
    new_lines = new.split("\n")
    for i in reversed(matches if replace_all else matches[:1]):
        content_lines[i:i + len(old_lines)] = new_lines
    return "\n".join(content_lines)


def _snippet(content: str, needle: str) -> str:
    lines = content.split("\n")
    hit = 0
    first = needle.split("\n")[0]
    for i, line in enumerate(lines):
        if first and first in line:
            hit = i
            break
    start = max(0, hit - SNIPPET_CONTEXT_LINES)
    end = min(len(lines), hit + SNIPPET_CONTEXT_LINES + 1)
    body = "\n".join(lines[start:end])
    return (f"Showing lines {start + 1}-{end} of {len(lines)} from the edited "
            f"file:\n\n---\n\n{body}")


# --------------------------------------------------------------- write_file


class WriteFileTool(DeclarativeTool):
    name = ToolNames.WRITE_FILE
    display_name = "WriteFile"
    kind = Kind.EDIT
    description = ("Writes content to a specified file. If the file exists, it "
                   "will be overwritten. If it doesn't exist, it (and any "
                   "necessary parent directories) will be created.")
    parameter_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string",
                          "description": "The absolute path to the file to write to."},
            "content": {"type": "string",
                        "description": "The content to write into the file."},
        },
        "required": ["file_path", "content"],
    }

    def __init__(self, config: Config, read_tracker: FileReadTracker | None = None):
        self.config = config
        self.read_tracker = read_tracker

    def validate_tool_param_values(self, params):
        if err := _require_absolute(params["file_path"]):
            return err
        if Path(params["file_path"]).is_dir():
            return f"Path is a directory, not a file: {params['file_path']}"
        return None

    def create_invocation(self, params):
        return _WriteFileInvocation(params, self.config, self.read_tracker)


class _WriteFileInvocation(ToolInvocation):
    def __init__(self, params, config, tracker):
        super().__init__(params)
        self._config = config
        self._tracker = tracker

    def get_description(self):
        rel = os.path.relpath(self.params["file_path"], self._config.target_dir)
        return f"Writing to {rel}"

    def get_default_permission(self):
        return "ask"

    def get_confirmation_details(self):
        path = Path(self.params["file_path"])
        old = path.read_text(errors="replace") if path.is_file() else ""
        rel = os.path.relpath(path, self._config.target_dir)
        return ConfirmationDetails(
            type="edit", title=f"Confirm Write: {rel}", file_name=rel,
            file_diff=_unified_diff(old, self.params["content"], rel))

    async def execute(self, update_output=None) -> ToolResult:
        path = Path(self.params["file_path"])
        if blocked := _check_prior_read(path, self._tracker):
            return blocked
        existed = path.is_file()
        old = path.read_text(errors="replace") if existed else ""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.params["content"])
        except PermissionError as e:
            return _error(ToolErrorType.PERMISSION_DENIED, str(e))
        except IsADirectoryError as e:
            return _error(ToolErrorType.TARGET_IS_DIRECTORY, str(e))
        except OSError as e:
            return _error(ToolErrorType.FILE_WRITE_FAILURE, str(e))
        if self._tracker:
            self._tracker.record_read(path)
        rel = os.path.relpath(path, self._config.target_dir)
        message = (f"Successfully overwrote file: {path}." if existed
                   else f"Successfully created and wrote to new file: {path}.")
        return ToolResult(
            llm_content=message,
            return_display=FileDiff(
                file_diff=_unified_diff(old, self.params["content"], rel),
                file_name=rel, original_content=old if existed else None,
                new_content=self.params["content"]))
