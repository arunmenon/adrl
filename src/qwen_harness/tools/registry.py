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

    def get_function_declarations(self, slim: bool = False) -> list[dict]:
        # sorted by name: keeps the tools block byte-stable across turns,
        # which protects provider prompt caches
        declarations = [self._tools[n].schema for n in sorted(self._tools)]
        return [_slim_declaration(d) for d in declarations] if slim else declarations

    def suggest(self, name: str) -> str | None:
        """Cheap 'did you mean' for TOOL_NOT_REGISTERED errors."""
        candidates = sorted(self._tools, key=lambda t: _distance(name, t))
        return candidates[0] if candidates else None


def _first_sentence(text: str) -> str:
    for stop in (". ", ".\n"):
        if stop in text:
            return text[:text.index(stop) + 1]
    return text


def _slim_declaration(declaration: dict) -> dict:
    """Schema diet (experiment knob): keep names, types, and required flags
    intact; cut every description to its first sentence."""
    slim = dict(declaration)
    slim["description"] = _first_sentence(declaration.get("description", ""))
    params = dict(declaration.get("parameters", {}))
    if props := params.get("properties"):
        params["properties"] = {
            key: {**spec, **({"description": _first_sentence(spec["description"])}
                             if "description" in spec else {})}
            for key, spec in props.items()}
    slim["parameters"] = params
    return slim


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
