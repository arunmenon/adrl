"""Wire up the built-in tool set (Config.createToolRegistry equivalent)."""

from __future__ import annotations

from .config import Config
from .tools.edit_tools import EditTool, WriteFileTool
from .tools.fs import FileReadTracker, GlobTool, GrepTool, LSTool, ReadFileTool
from .tools.misc import MemoryTool, TodoWriteTool
from .tools.registry import ToolRegistry
from .tools.shell import ShellTool


def create_tool_registry(config: Config) -> ToolRegistry:
    registry = ToolRegistry()
    tracker = FileReadTracker()  # shared: read_file records, edit/write enforce
    for tool in (
        LSTool(config),
        ReadFileTool(config, tracker),
        GlobTool(config),
        GrepTool(config),
        EditTool(config, tracker),
        WriteFileTool(config, tracker),
        ShellTool(config),
        TodoWriteTool(config),
        MemoryTool(config),
    ):
        registry.register_tool(tool)
    return registry
