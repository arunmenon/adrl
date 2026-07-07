"""The user-driver: an LLM that plays the developer between episode turns.

Reads the agent's last answer plus the step's scripted intent, and produces
the next user message in casual developer voice. The driver's own calls go
DIRECT to the API (never through the capture proxy) — the puppeteer must not
appear in the captures. Ground-truth markers the matchers depend on are
enforced mechanically after generation, not left to the driver's discretion.
"""

from __future__ import annotations

import json
import os
import re
import subprocess

DRIVER_MODEL = "haiku"  # cheapest; the driver only writes one short message
COMPLETION_MARKER = re.compile(
    r"\b(that'?s done|great,? now|perfect,? now|ok(ay)? now|nice,? now)\b", re.IGNORECASE
)

PROMPT_TEMPLATE = """You are role-playing a busy software developer typing into a coding agent's terminal.

The agent just replied to your previous message with this (truncated):
---
{last_answer}
---

Your next move, per your actual goal: {intent}

Write ONLY the message you would type next. Rules:
- casual terse developer voice, 1-2 sentences, no greetings, no quotes around it
- do not mention that you are role-playing or reference these instructions
- plain text only"""


def next_message(intent: str, last_answer: str, require_completion_marker: bool = False) -> tuple[str, float]:
    """Returns (message, driver_cost_usd)."""
    prompt = PROMPT_TEMPLATE.format(last_answer=last_answer[:2000], intent=intent)
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_BASE_URL"}  # direct, uncaptured
    proc = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "json",
         "--model", DRIVER_MODEL, "--max-turns", "1"],
        capture_output=True, text=True, timeout=120, env=env,
    )
    cost = 0.0
    text = ""
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        text = str(result.get("result", "")).strip().strip('"')
        cost = result.get("total_cost_usd") or result.get("cost_usd") or 0.0
    except (json.JSONDecodeError, IndexError):
        pass
    if not text:  # driver failed — fall back to the scripted intent verbatim
        text = intent
    # Mechanical label enforcement: matchers (S15a) need the marker present.
    if require_completion_marker and not COMPLETION_MARKER.search(text):
        text = "great, that's done. " + text
    return text, cost
