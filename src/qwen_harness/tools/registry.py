"""Tool registry. Port of tools/tool-registry.ts (core slice: registration,
lookup, function declarations sorted by name for prompt-cache stability).
Upstream additionally supports lazy factories, deferred tools revealed via
tool_search, command-discovered tools, and MCP tools.
"""

from __future__ import annotations

from .base import DeclarativeTool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, DeclarativeTool] = {}

    def register_tool(self, tool: DeclarativeTool) -> None:
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> DeclarativeTool | None:
        return self._tools.get(name)

    def get_all_tool_names(self) -> list[str]:
        return sorted(self._tools)

    def get_function_declarations(self) -> list[dict]:
        # sorted by name: keeps the tools block byte-stable across turns,
        # which protects provider prompt caches
        return [self._tools[n].schema for n in sorted(self._tools)]

    def suggest(self, name: str) -> str | None:
        """Cheap 'did you mean' for TOOL_NOT_REGISTERED errors."""
        candidates = sorted(self._tools, key=lambda t: _distance(name, t))
        return candidates[0] if candidates else None


def _distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]
