"""Tool base abstractions. Port of tools/tools.ts + tools/tool-error.ts.

The core pattern is the DeclarativeTool / ToolInvocation split:

  DeclarativeTool  the *catalog entry* — name, description, JSON schema,
                   Kind. `build(params)` validates and returns…
  ToolInvocation   a *validated, ready-to-execute call* — it can describe
                   itself, say what it will touch, decide its intrinsic
                   permission ('allow'/'ask'), produce confirmation-dialog
                   details, and execute.

Validation therefore happens once, up front, and everything downstream
(approval UI, scheduler, execution) works with an object that is already
known to be well-formed. ToolResult splits `llm_content` (goes back to the
model) from `return_display` (rendered to the human) — the two audiences
routinely get different views of the same outcome.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Kind(str, Enum):
    READ = "read"
    EDIT = "edit"
    DELETE = "delete"
    MOVE = "move"
    SEARCH = "search"
    EXECUTE = "execute"
    THINK = "think"
    FETCH = "fetch"
    OTHER = "other"


# Kinds safe to run in parallel with each other (CONCURRENCY_SAFE_KINDS)
CONCURRENCY_SAFE_KINDS = {Kind.READ, Kind.SEARCH, Kind.FETCH}
MUTATOR_KINDS = {Kind.EDIT, Kind.DELETE, Kind.MOVE, Kind.EXECUTE}


class ToolErrorType(str, Enum):
    INVALID_TOOL_PARAMS = "invalid_tool_params"
    UNKNOWN = "unknown"
    UNHANDLED_EXCEPTION = "unhandled_exception"
    TOOL_NOT_REGISTERED = "tool_not_registered"
    EXECUTION_FAILED = "execution_failed"
    EXECUTION_DENIED = "execution_denied"
    FILE_NOT_FOUND = "file_not_found"
    FILE_WRITE_FAILURE = "file_write_failure"
    ATTEMPT_TO_CREATE_EXISTING_FILE = "attempt_to_create_existing_file"
    PERMISSION_DENIED = "permission_denied"
    TARGET_IS_DIRECTORY = "target_is_directory"
    PATH_NOT_IN_WORKSPACE = "path_not_in_workspace"
    EDIT_NO_OCCURRENCE_FOUND = "edit_no_occurrence_found"
    EDIT_EXPECTED_OCCURRENCE_MISMATCH = "edit_expected_occurrence_mismatch"
    EDIT_NO_CHANGE = "edit_no_change"
    GLOB_EXECUTION_ERROR = "glob_execution_error"
    GREP_EXECUTION_ERROR = "grep_execution_error"
    LS_EXECUTION_ERROR = "ls_execution_error"
    PATH_IS_NOT_A_DIRECTORY = "path_is_not_a_directory"
    SHELL_EXECUTE_ERROR = "shell_execute_error"
    MEMORY_TOOL_EXECUTION_ERROR = "memory_tool_execution_error"


@dataclass
class ToolError:
    message: str
    type: ToolErrorType = ToolErrorType.UNKNOWN


@dataclass
class ToolResult:
    llm_content: str                      # what the model sees
    return_display: Any = None            # what the human sees (str | FileDiff)
    error: ToolError | None = None        # presence == failure

    @property
    def display(self) -> str:
        return self.return_display if isinstance(self.return_display, str) else self.llm_content


@dataclass
class FileDiff:
    file_diff: str
    file_name: str
    original_content: str | None
    new_content: str


class ToolConfirmationOutcome(str, Enum):
    PROCEED_ONCE = "proceed_once"
    PROCEED_ALWAYS = "proceed_always"
    CANCEL = "cancel"


@dataclass
class ConfirmationDetails:
    """Discriminated union of confirmation dialogs (tools.ts).

    type: 'edit' carries a diff; 'exec' carries the command + root command;
    'info' is a generic prompt (also used for fetch URLs).
    """

    type: str                              # 'edit' | 'exec' | 'info'
    title: str
    # edit
    file_name: str | None = None
    file_diff: str | None = None
    # exec
    command: str | None = None
    root_command: str | None = None
    # info
    prompt: str | None = None
    urls: list[str] = field(default_factory=list)


@dataclass
class ToolLocation:
    path: str
    line: int | None = None


class ToolInvocation(ABC):
    def __init__(self, params: dict[str, Any]):
        self.params = params

    @abstractmethod
    def get_description(self) -> str: ...

    def tool_locations(self) -> list[ToolLocation]:
        return []

    def get_default_permission(self) -> str:
        """Intrinsic permission: 'allow' | 'ask'. Read-only tools inside the
        workspace return 'allow'; mutating tools return 'ask'."""
        return "allow"

    def get_confirmation_details(self) -> ConfirmationDetails:
        return ConfirmationDetails(type="info",
                                   title=f"Confirm {type(self).__name__}",
                                   prompt=self.get_description())

    @abstractmethod
    async def execute(self, update_output=None) -> ToolResult: ...


class DeclarativeTool(ABC):
    name: str = ""
    display_name: str = ""
    description: str = ""
    kind: Kind = Kind.OTHER
    parameter_schema: dict[str, Any] = {}
    is_output_markdown: bool = True
    can_update_output: bool = False
    # None = use the global config threshold; float('inf') = self-managed
    max_output_chars: int | None = None
    truncate_keep: str = "both"  # 'head' | 'tail' | 'both'

    @property
    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "parameters": self.parameter_schema}

    def validate_tool_params(self, params: dict[str, Any]) -> str | None:
        """Schema check (upstream uses a full JSON-schema validator; a
        required/type check keeps the study port dependency-free)."""
        schema = self.parameter_schema
        for req in schema.get("required", []):
            if req not in params:
                return f"params must have required property '{req}'"
        types = {"string": str, "integer": int, "number": (int, float),
                 "boolean": bool, "array": list, "object": dict}
        for key, value in params.items():
            spec = schema.get("properties", {}).get(key)
            if spec is None:
                continue
            expected = types.get(spec.get("type"))
            if expected and not isinstance(value, expected):
                return f"params/{key} must be {spec['type']}"
            if isinstance(value, bool) and spec.get("type") in ("integer", "number"):
                return f"params/{key} must be {spec['type']}"
        return self.validate_tool_param_values(params)

    def validate_tool_param_values(self, params: dict[str, Any]) -> str | None:
        return None

    @abstractmethod
    def create_invocation(self, params: dict[str, Any]) -> ToolInvocation: ...

    def build(self, params: dict[str, Any]) -> ToolInvocation:
        error = self.validate_tool_params(params)
        if error:
            raise ValueError(error)
        return self.create_invocation(json.loads(json.dumps(params)))
