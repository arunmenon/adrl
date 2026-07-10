"""Hierarchical instruction memory (QWEN.md). Port of utils/memoryDiscovery.ts.

Discovery order (most general first, most specific last, so later files
can override earlier ones):

  1. global   ~/.qwen/QWEN.md            (user-wide standing instructions)
  2. upward   <project root>/QWEN.md ... <cwd>/QWEN.md   (root-most first)
  3. local    <project root>/.qwen/QWEN.local.md          (per-developer)

Filenames searched: QWEN.md and AGENTS.md. The concatenated result becomes
`user_memory`, which prompts.py appends to the system prompt after a
`\n\n---\n\n` separator. `@path` imports inside a file are inlined
recursively (depth-capped, confined to the project root).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

CONTEXT_FILENAMES = ["QWEN.md", "AGENTS.md"]
QWEN_DIR = ".qwen"
LOCAL_MEMORY_NAME = "QWEN.local.md"
MAX_IMPORT_DEPTH = 5
MEMORY_SECTION_HEADER = "## Qwen Added Memories"  # used by the save_memory tool


def find_project_root(start: Path) -> Path | None:
    """Nearest ancestor containing .git (dir or file)."""
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def discover_memory_files(cwd: str | Path, home: Path | None = None) -> list[Path]:
    cwd = Path(cwd).resolve()
    home = (home or Path.home()).resolve()
    project_root = find_project_root(cwd)
    found: list[Path] = []

    def add(p: Path) -> None:
        if p.is_file() and p not in found:
            found.append(p)

    # 1. global
    for name in CONTEXT_FILENAMES:
        add(home / QWEN_DIR / name)

    # 2. upward scan cwd -> project root, results ordered root-most first
    stop = project_root.parent if project_root else home.parent
    upward: list[Path] = []
    current = cwd
    while current != stop and current != current.parent:
        for name in CONTEXT_FILENAMES:
            candidate = current / name
            if candidate.is_file() and candidate not in found and candidate not in upward:
                upward.insert(0, candidate)  # unshift: root-most ends up first
        current = current.parent
    found.extend(upward)

    # 3. per-developer local slot, loaded last so it can override
    if project_root:
        add(project_root / QWEN_DIR / LOCAL_MEMORY_NAME)

    return found


def process_imports(text: str, base_dir: Path, project_root: Path | None,
                    depth: int = 0, seen: set[Path] | None = None) -> str:
    """Inline `@path` imports (memoryImportProcessor.ts, 'tree' format)."""
    if depth >= MAX_IMPORT_DEPTH:
        return text
    seen = seen if seen is not None else set()
    out: list[str] = []
    in_code_block = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code_block = not in_code_block
        if in_code_block:
            out.append(line)
            continue
        out.append(re.sub(
            r"(?<![\w`])@([./\w][^\s,;!?()\[\]{}]*)",
            lambda m: _resolve_import(m, base_dir, project_root, depth, seen),
            line,
        ))
    return "\n".join(out)


def _resolve_import(match: re.Match, base_dir: Path, project_root: Path | None,
                    depth: int, seen: set[Path]) -> str:
    raw = match.group(1)
    if raw.startswith(("http://", "https://", "file://")):
        return match.group(0)
    target = (base_dir / raw).resolve()
    if project_root and project_root not in [target, *target.parents]:
        return f"<!-- Import failed: {raw} - Path traversal attempt -->"
    if target in seen:
        return f"<!-- File already processed: {raw} -->"
    if not target.is_file():
        return match.group(0)  # ENOENT keeps the literal @path text
    seen.add(target)
    content = process_imports(target.read_text(errors="replace"), target.parent,
                              project_root, depth + 1, seen)
    return f"<!-- Imported from: {raw} -->\n{content}\n<!-- End of import from: {raw} -->"


def load_hierarchical_memory(cwd: str | Path, home: Path | None = None) -> tuple[str, int]:
    """Returns (concatenated memory text, file count) — loadServerHierarchicalMemory."""
    cwd = Path(cwd).resolve()
    project_root = find_project_root(cwd)
    blocks: list[str] = []
    files = discover_memory_files(cwd, home)
    for f in files:
        content = f.read_text(errors="replace").strip()
        if not content:
            continue
        content = process_imports(content, f.parent, project_root)
        rel = os.path.relpath(f, cwd)
        blocks.append(
            f"--- Context from: {rel} ---\n{content}\n--- End of Context from: {rel} ---"
        )
    return "\n\n".join(blocks), len(files)
