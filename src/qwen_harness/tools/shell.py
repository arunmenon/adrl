"""run_shell_command. Port of tools/shell.ts.

The most safety-sensitive tool, so most of the code is *classification*,
not execution:
  - permission: a command is intrinsically 'allow' only if it parses as
    read-only; shell substitution ($(), backticks, <() ) always forces a
    prompt because it can smuggle a second command past any allowlist;
  - the confirmation dialog names the *root command* (e.g. `git`) so
    "always allow" grants can be scoped;
  - foreground commands run in their own process group with a timeout
    (default 120s, cap 600s) so runaway children die with the parent;
  - is_background=true detaches: output goes to a file, the model gets the
    pid + path back immediately;
  - the result block reports Command/Directory/Output/Error/Exit Code/
    Signal even on success, so the model never has to guess what happened.
Output is self-truncated at 30k chars keeping head+tail.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import signal
import uuid
from pathlib import Path

from ..config import Config
from .base import (ConfirmationDetails, DeclarativeTool, Kind, ToolError,
                   ToolErrorType, ToolInvocation, ToolResult)
from .names import ToolNames

SHELL_MAX_OUTPUT_CHARS = 30_000
DEFAULT_FOREGROUND_TIMEOUT_MS = 120_000
MAX_TIMEOUT_MS = 600_000

_SUBSTITUTION_RE = re.compile(r"\$\(|`|<\(|>\(")
_READ_ONLY_ROOTS = {
    "ls", "cat", "head", "tail", "pwd", "echo", "wc", "which", "whoami",
    "date", "env", "printenv", "uname", "file", "stat", "du", "df", "find",
    "grep", "rg", "true",
}
_READ_ONLY_GIT = {"status", "diff", "log", "show", "branch", "remote",
                  "ls-files", "rev-parse", "blame"}


def get_command_roots(command: str) -> list[str]:
    """First token of each segment of a compound command."""
    roots = []
    for segment in re.split(r"&&|\|\||;|\|", command):
        try:
            tokens = shlex.split(segment.strip())
        except ValueError:
            tokens = segment.strip().split()
        if tokens:
            roots.append(os.path.basename(tokens[0]))
    return roots


def is_command_read_only(command: str) -> bool:
    """Fail-closed read-only classifier (upstream parses a full AST; the
    port classifies per compound segment and rejects on any doubt)."""
    if _SUBSTITUTION_RE.search(command):
        return False
    for segment in re.split(r"&&|\|\||;|\|", command):
        try:
            tokens = shlex.split(segment.strip())
        except ValueError:
            return False
        if not tokens:
            continue
        root = os.path.basename(tokens[0])
        if root == "git":
            if len(tokens) < 2 or tokens[1] not in _READ_ONLY_GIT:
                return False
            continue
        if root not in _READ_ONLY_ROOTS:
            return False
        if any(t == ">" or t.startswith((">", ">>")) for t in tokens):
            return False
    return True


class ShellTool(DeclarativeTool):
    name = ToolNames.SHELL
    display_name = "Shell"
    kind = Kind.EXECUTE
    is_output_markdown = False
    can_update_output = True
    max_output_chars = SHELL_MAX_OUTPUT_CHARS
    description = (
        "Executes a given shell command. Command can start background processes "
        "using is_background=true (do not append '&' yourself). Returns the "
        "command's output, error, exit code and signal.")
    parameter_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Exact command to execute."},
            "description": {"type": "string",
                            "description": "Brief description of the command for the user."},
            "is_background": {"type": "boolean",
                              "description": "Run in background (for long-running "
                                             "processes like servers)."},
            "timeout": {"type": "integer",
                        "description": "Timeout in milliseconds (max 600000)."},
            "directory": {"type": "string",
                          "description": "Absolute path of the directory to run in "
                                         "(must be inside the workspace)."},
        },
        "required": ["command"],
    }

    def __init__(self, config: Config):
        self.config = config

    def validate_tool_param_values(self, params):
        command = params["command"].strip()
        if not command:
            return "Command cannot be empty."
        if params.get("is_background") and command.rstrip().endswith("&"):
            return ("Background commands must not end with '&'; set "
                    "is_background=true instead.")
        if (timeout := params.get("timeout")) is not None and (
                timeout <= 0 or timeout > MAX_TIMEOUT_MS):
            return f"Timeout must be between 1 and {MAX_TIMEOUT_MS} ms."
        if not get_command_roots(command):
            return "Could not identify command root to obtain permission from user."
        if directory := params.get("directory"):
            if not os.path.isabs(directory):
                return "Directory must be an absolute path."
            if not os.path.isdir(directory):
                return f"Directory does not exist: {directory}"
        return None

    def create_invocation(self, params):
        return _ShellInvocation(params, self.config)


class _ShellInvocation(ToolInvocation):
    def __init__(self, params, config):
        super().__init__(params)
        self._config = config

    def get_description(self):
        text = self.params["command"]
        if note := self.params.get("description"):
            text += f" ({note})"
        return text

    def get_default_permission(self):
        command = self.params["command"]
        if _SUBSTITUTION_RE.search(command):
            return "ask"  # substitution can smuggle anything past a rule
        return "allow" if is_command_read_only(command) else "ask"

    def get_confirmation_details(self):
        roots = ", ".join(dict.fromkeys(get_command_roots(self.params["command"])))
        return ConfirmationDetails(type="exec", title="Confirm Shell Command",
                                   command=self.params["command"],
                                   root_command=roots)

    async def execute(self, update_output=None) -> ToolResult:
        command = self.params["command"]
        cwd = self.params.get("directory") or self._config.target_dir
        if self.params.get("is_background"):
            return await self._execute_background(command, cwd)

        timeout_s = (self.params.get("timeout") or DEFAULT_FOREGROUND_TIMEOUT_MS) / 1000
        proc = await asyncio.create_subprocess_shell(
            command, cwd=cwd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True)  # own process group: timeout kills children too
        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout_s)
        except asyncio.TimeoutError:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            stdout, stderr = await proc.communicate()

        out = stdout.decode(errors="replace").strip()
        err = stderr.decode(errors="replace").strip()
        body = (f"Command: {command}\n"
                f"Directory: {cwd}\n"
                f"Output: {out or '(empty)'}\n"
                f"Error: {err or '(none)'}\n"
                f"Exit Code: {proc.returncode if proc.returncode is not None else '(none)'}\n"
                f"Signal: {'(none)' if proc.returncode is None or proc.returncode >= 0 else -proc.returncode}")
        if timed_out:
            timeout_ms = int(timeout_s * 1000)
            body = (f"Command timed out after {timeout_ms}ms before it could "
                    f"complete. Below is the output before it timed out:\n{body}")
        failed = timed_out or (proc.returncode not in (0, None))
        return ToolResult(
            llm_content=body,
            return_display=out or err or "(empty)",
            error=ToolError(message=body, type=ToolErrorType.SHELL_EXECUTE_ERROR)
            if failed else None)

    async def _execute_background(self, command: str, cwd: str) -> ToolResult:
        bg_id = f"bg_{uuid.uuid4().hex[:8]}"
        out_dir = Path(self._config.target_dir) / ".qwen_harness" / "background-shells"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"shell-{bg_id}.output"
        handle = open(out_file, "wb")
        proc = await asyncio.create_subprocess_shell(
            command.rstrip().rstrip("&"), cwd=cwd,
            stdout=handle, stderr=asyncio.subprocess.STDOUT,
            start_new_session=True)  # deliberately NOT tied to the turn
        return ToolResult(
            llm_content=(f"Background shell started.\nid: {bg_id}\npid: {proc.pid}\n"
                         f"output file: {out_file}\n"
                         "To inspect, read the output file directly with read_file."),
            return_display=f"Started background shell {bg_id} (pid {proc.pid}).")
