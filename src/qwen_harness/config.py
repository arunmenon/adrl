"""Session configuration. Port of the relevant slice of config/config.ts
and config/approval-mode.ts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

from .content_generator import GeneratorConfig


class ApprovalMode(str, Enum):
    """Who has to say yes before a tool runs.

    DEFAULT    every tool whose permission resolves to 'ask' prompts
    AUTO_EDIT  edit-type confirmations (edit/write_file diffs) auto-approve;
               shell still prompts
    YOLO       everything auto-approves
    PLAN       read-only: any tool needing a non-info confirmation is blocked
    """

    PLAN = "plan"
    DEFAULT = "default"
    AUTO_EDIT = "auto-edit"
    YOLO = "yolo"


@dataclass
class Config:
    model: str = field(default_factory=lambda: os.environ.get(
        "QWEN_HARNESS_MODEL", "qwen2.5:7b-instruct-q4_K_M"))
    target_dir: str = field(default_factory=os.getcwd)
    approval_mode: ApprovalMode = ApprovalMode.DEFAULT
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)

    max_session_turns: int = 0        # 0/negative = unlimited (getMaxSessionTurns)
    max_turns: int = 100              # MAX_TURNS recursion bound in client.ts
    max_tool_calls_per_turn: int = 100
    skip_next_speaker_check: bool = False
    skip_loop_detection: bool = True  # fork default: heuristics opt-in
    debug: bool = False

    # tool output budgets (config.ts defaults)
    truncate_tool_output_threshold: int = 25_000
    truncate_tool_output_lines: int = 1_000

    def is_interactive(self) -> bool:
        return getattr(self, "_interactive", False)

    def set_interactive(self, value: bool) -> None:
        self._interactive = value
