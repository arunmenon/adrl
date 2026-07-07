"""Episode script grammar — multi-turn scenarios with mechanical ground truth.

The division of labor that keeps simulator data gradable (see plan C2/C5):
the SKELETON is ours — which scenario starts the episode, what each follow-up
intends, where the episode boundary or retry signal sits. The SURFACE is the
driver LLM's — how a busy developer would actually phrase each message.

Each episode is a list of steps. Step 1 is a labeled scenario from tasks.py
(planted ground truth). Later steps carry an `intent` the driver phrases,
plus flags the runner enforces mechanically (e.g. a completion marker so the
S15a episode-boundary matcher has something to find even if the driver gets
creative).
"""

from __future__ import annotations

EPISODES: dict[str, dict] = {
    # S15a — the scenario with ZERO organic matches: an episode boundary.
    # Easy task completes, user acknowledges completion and pivots to a new easy task.
    "episode_boundary": {
        "maps_to": "S15a",
        "steps": [
            {"kind": "scenario", "scenario": "rename"},
            {
                "kind": "driven",
                "intent": "The previous task is finished. Acknowledge it briefly, then ask for a "
                          "NEW unrelated easy task: write a short usage section into the README "
                          "explaining how to run the cli and the tests.",
                "require_completion_marker": True,
                "expected_label": "new_episode",
            },
        ],
    },
    # S6-adjacent — retry signal: user is unsatisfied and immediately re-asks
    # the same thing more specifically (headless can't interrupt mid-stream,
    # but the rephrase-right-after signal is the label that matters).
    "rephrase_retry": {
        "maps_to": "S6",
        "steps": [
            {"kind": "scenario", "scenario": "investigate"},
            {
                "kind": "driven",
                "intent": "You are NOT satisfied with that answer — it was too vague or missed the "
                          "point. Re-ask the same question, more specific and slightly annoyed: you "
                          "want the exact config flag and file/line that disables rate limiting, "
                          "nothing else.",
                "expected_label": "retry_signal",
            },
        ],
    },
    # Working-summary reference — 'do the same for the other one' resolution.
    "same_for_other": {
        "maps_to": "working_summary (§5.6)",
        "steps": [
            {"kind": "scenario", "scenario": "feature"},
            {
                "kind": "driven",
                "intent": "Now ask, tersely, for the same treatment somewhere else: apply the same "
                          "kind of change to the stats module (whatever the agent just did, mirrored "
                          "there). Deliberately use a back-reference like 'same thing for' instead of "
                          "restating the requirement.",
                "expected_label": "continuation_of_episode",
            },
        ],
    },
    # Hysteresis shape — easy turn, then a hard follow-up in the same episode.
    "easy_then_hard": {
        "maps_to": "S10/hysteresis",
        "steps": [
            {"kind": "scenario", "scenario": "explain"},
            {
                "kind": "driven",
                "intent": "Now escalate: ask for the big refactor — introduce a Settings class and "
                          "inject it instead of importing config directly across stats, limiter and "
                          "cli, keeping tests green.",
                "expected_label": "hard_turn_same_episode",
            },
        ],
    },
    # Assessment -> action boundary: read-only investigation, then 'now fix it'.
    "investigate_then_fix": {
        "maps_to": "S4->S3",
        "steps": [
            {"kind": "scenario", "scenario": "investigate"},
            {
                "kind": "driven",
                "intent": "The diagnosis sounds right. Tell the agent to go ahead and fix it "
                          "properly, and to run the tests after.",
                "expected_label": "action_after_assessment",
            },
        ],
    },
    # Long-episode shape: three turns of accumulating work then a commit ask.
    "work_then_commit": {
        "maps_to": "S8-lite",
        "steps": [
            {"kind": "scenario", "scenario": "fix_test"},
            {
                "kind": "driven",
                "intent": "Ask for one more small improvement: also make the parser tolerate a "
                          "missing amount column by defaulting it to 0.0, with a test.",
                "expected_label": "continuation_of_episode",
            },
            {
                "kind": "driven",
                "intent": "Wrap up: ask the agent to stage everything and commit with a sensible "
                          "message summarizing what was done this session.",
                "require_completion_marker": True,
                "expected_label": "episode_wrap_up",
            },
        ],
    },
}
