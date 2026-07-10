"""todo_write + save_memory. Ports of tools/todoWrite.ts and the classic
qwen-code memoryTool.ts.

todo_write is a Kind.THINK tool: it mutates no project state the model
can't see — its entire effect is the <system-reminder> echoed back, which
keeps the plan inside the context window. Note the whole-list-replacement
semantics: the model always sends the full list, never a delta.

save_memory is the upstream qwen-code tool (the current fork replaced it
with an auto-memory subsystem): append a fact as '- <fact>' under the
'## Qwen Added Memories' header in ~/.qwen/QWEN.md, where the next
session's memory discovery (memory.py) will pick it up.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import Config
from ..memory import MEMORY_SECTION_HEADER, QWEN_DIR
from .base import (DeclarativeTool, Kind, ToolError, ToolErrorType,
                   ToolInvocation, ToolResult)
from .names import ToolNames

VALID_TODO_STATUS = {"pending", "in_progress", "completed"}


class TodoWriteTool(DeclarativeTool):
    name = ToolNames.TODO_WRITE
    display_name = "TodoList"
    kind = Kind.THINK
    description = (
        "Use this tool to create and manage a structured task list for your "
        "current coding session. This helps you track progress, organize "
        "complex tasks, and demonstrate thoroughness to the user. The tool "
        "replaces the entire todo list with the provided array.")
    parameter_schema = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "content": {"type": "string", "minLength": 1},
                        "status": {"type": "string",
                                   "enum": sorted(VALID_TODO_STATUS)},
                    },
                    "required": ["id", "content", "status"],
                },
            },
        },
        "required": ["todos"],
    }

    def __init__(self, config: Config):
        self.config = config
        self.todos: list[dict] = []  # session-scoped store

    def validate_tool_param_values(self, params):
        seen = set()
        for todo in params["todos"]:
            if not isinstance(todo, dict):
                return "Each todo must be an object."
            for key in ("id", "content", "status"):
                if not todo.get(key):
                    return f"Todo missing required non-empty field '{key}'."
            if todo["status"] not in VALID_TODO_STATUS:
                return (f"Invalid status '{todo['status']}'; must be one of "
                        f"{sorted(VALID_TODO_STATUS)}.")
            if todo["id"] in seen:
                return f"Duplicate todo id '{todo['id']}'."
            seen.add(todo["id"])
        return None

    def create_invocation(self, params):
        return _TodoWriteInvocation(params, self)


class _TodoWriteInvocation(ToolInvocation):
    def __init__(self, params, tool: TodoWriteTool):
        super().__init__(params)
        self._tool = tool

    def get_description(self):
        n = len(self.params["todos"])
        return f"Update todo list ({n} item{'s' if n != 1 else ''})"

    async def execute(self, update_output=None) -> ToolResult:
        self._tool.todos = self.params["todos"]
        if not self._tool.todos:
            return ToolResult(
                llm_content="Todo list has been cleared.\n<system-reminder>\n"
                            "Your todo list is now empty. DO NOT mention this "
                            "explicitly to the user.\n</system-reminder>",
                return_display="Cleared todo list.")
        listing = json.dumps(self._tool.todos, indent=2)
        return ToolResult(
            llm_content=("Todos have been modified successfully.\n"
                         "<system-reminder>\nYour todo list has changed. DO NOT "
                         "mention this explicitly to the user. Here are the "
                         f"latest contents of your todo list:\n\n{listing}. "
                         "Continue on with the tasks at hand if applicable.\n"
                         "</system-reminder>"),
            return_display="\n".join(
                f"[{'x' if t['status'] == 'completed' else '~' if t['status'] == 'in_progress' else ' '}] {t['content']}"
                for t in self._tool.todos))


class MemoryTool(DeclarativeTool):
    name = ToolNames.MEMORY
    display_name = "SaveMemory"
    kind = Kind.THINK
    description = (
        "Saves a specific piece of information or fact to your long-term memory. "
        "Use this when the user explicitly asks you to remember something, or "
        "when they state a clear, concise fact that seems important to retain "
        "for future interactions.")
    parameter_schema = {
        "type": "object",
        "properties": {
            "fact": {"type": "string",
                     "description": "The specific fact or piece of information "
                                    "to remember. Should be a clear, "
                                    "self-contained statement."},
        },
        "required": ["fact"],
    }

    def __init__(self, config: Config, home: Path | None = None):
        self.config = config
        self.home = home or Path.home()

    def create_invocation(self, params):
        return _MemoryInvocation(params, self.home)


class _MemoryInvocation(ToolInvocation):
    def __init__(self, params, home: Path):
        super().__init__(params)
        self._home = home

    def get_description(self):
        return f"Save '{self.params['fact']}' to memory"

    async def execute(self, update_output=None) -> ToolResult:
        fact = self.params["fact"].strip()
        target = self._home / QWEN_DIR / "QWEN.md"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            text = target.read_text() if target.is_file() else ""
            entry = f"- {fact}"
            if MEMORY_SECTION_HEADER not in text:
                text = (text.rstrip() + ("\n\n" if text.strip() else "")
                        + f"{MEMORY_SECTION_HEADER}\n{entry}\n")
            else:
                head, _, tail = text.partition(MEMORY_SECTION_HEADER)
                lines = tail.split("\n")
                # insert before the next top-level '## ' heading in the section
                insert_at = len(lines)
                for i, line in enumerate(lines[1:], 1):
                    if line.startswith("## "):
                        insert_at = i
                        break
                lines.insert(insert_at, entry)
                text = head + MEMORY_SECTION_HEADER + "\n".join(lines)
            target.write_text(text)
        except OSError as e:
            message = f"Failed to save memory: {e}"
            return ToolResult(llm_content=f"Error: {message}", return_display=message,
                              error=ToolError(message=message,
                                              type=ToolErrorType.MEMORY_TOOL_EXECUTION_ERROR))
        return ToolResult(
            llm_content=f'Okay, I\'ve remembered that: "{fact}"',
            return_display=f"Saved to {target}.")
