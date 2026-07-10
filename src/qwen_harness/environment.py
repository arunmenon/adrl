"""Startup environment context. Port of utils/environmentContext.ts +
utils/getFolderStructure.ts.

The harness never assumes the model knows where it is: the first user
message carries the date, OS, working directory, and a truncated folder
tree. Everything the model 'knows' about the session enters through
history — a useful property when studying the wire traffic.
"""

from __future__ import annotations

import datetime as _dt
import fnmatch
import platform
from pathlib import Path

# Upstream MAX_ITEMS was 200 in gemini-cli; the current fork trims to 20 to
# save prompt tokens. We keep it configurable with the fork default.
MAX_ITEMS = 20
_DEFAULT_IGNORES = {"node_modules", ".git", "dist", ".venv", "__pycache__"}


def format_date_for_context(now: _dt.date | None = None) -> str:
    d = now or _dt.date.today()
    return d.strftime("%A, %B %-d, %Y")  # 'Friday, July 10, 2026'


def get_folder_structure(root: str | Path, max_items: int = MAX_ITEMS) -> str:
    """BFS listing, alphabetical, files before subfolders, ignored dirs
    shown as 'name/...'. Faithful to getFolderStructure.ts in shape."""
    root = Path(root).resolve()
    lines: list[str] = []
    count = 0
    truncated = False

    def walk(directory: Path, prefix: str) -> None:
        nonlocal count, truncated
        if truncated:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (p.is_dir(), p.name.lower()))
        except OSError:
            return
        visible = [e for e in entries if not _is_hidden_junk(e)]
        for i, entry in enumerate(visible):
            if count >= max_items:
                lines.append(f"{prefix}...")
                truncated = True
                return
            connector = "└───" if i == len(visible) - 1 else "├───"
            if entry.is_dir():
                if entry.name in _DEFAULT_IGNORES:
                    lines.append(f"{prefix}{connector}{entry.name}/...")
                    count += 1
                    continue
                lines.append(f"{prefix}{connector}{entry.name}/")
                count += 1
                extension = "    " if i == len(visible) - 1 else "│   "
                walk(entry, prefix + extension)
            else:
                lines.append(f"{prefix}{connector}{entry.name}")
                count += 1

    walk(root, "")
    header = f"Showing up to {max_items} items:\n\n{root}/"
    return header + ("\n" + "\n".join(lines) if lines else "")


def _is_hidden_junk(p: Path) -> bool:
    return fnmatch.fnmatch(p.name, ".DS_Store")


def get_directory_context_string(dirs: list[str]) -> str:
    if len(dirs) == 1:
        working = f"I'm currently working in the directory: {dirs[0]}"
    else:
        listing = "\n".join(f"  - {d}" for d in dirs)
        working = f"I'm currently working in the following directories:\n{listing}"
    trees = "\n".join(get_folder_structure(d) for d in dirs)
    return (
        f"{working}\n"
        "Here is the folder structure of the current working directories:\n\n"
        f"{trees}"
    )


def get_environment_context(cwd: str, extra_dirs: list[str] | None = None) -> str:
    """The exact startup context text (environmentContext.ts)."""
    dirs = [cwd, *(extra_dirs or [])]
    return (
        "This is the Qwen Code. We are setting up the context for our chat.\n"
        f"Today's date is {format_date_for_context()}.\n"
        f"My operating system is: {platform.system().lower()}\n"
        f"{get_directory_context_string(dirs)}"
    )


def wrap_system_reminder(text: str) -> str:
    """Wrap injected context so the model can distinguish harness-injected
    material from real user input (fork behavior; stock qwen-code sends the
    env context bare, followed by a canned model ack)."""
    return f"<system-reminder>\n{text}\n</system-reminder>"
