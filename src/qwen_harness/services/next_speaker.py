"""Next-speaker checker. Port of utils/nextSpeakerChecker.ts.

After a turn ends with no tool calls, the harness asks: should the model
keep going? Three cheap rules short-circuit the LLM call; otherwise a JSON
side-query judges the model's own last message. If the verdict is 'model',
client.py re-invokes the loop with the literal message 'Please continue.'
— this is the mechanism that lets the agent narrate a plan and then follow
through without user input.
"""

from __future__ import annotations

import json
from typing import Any

from ..types import Content, Part

CHECK_PROMPT = """Analyze *only* the content and structure of your immediately preceding response (your last turn in the conversation history). Based *strictly* on that response, determine who should logically speak next: the 'user' or the 'model' (you).
**Decision Rules (apply in order):**
1.  **Model Continues:** If your last response explicitly states an immediate next action *you* intend to take (e.g., "Next, I will...", "Now I'll process...", "Moving on to analyze...", indicates an intended tool call that didn't execute), OR if the response seems clearly incomplete (cut off mid-thought without a natural conclusion), then the **'model'** should speak next.
2.  **Question to User:** If your last response ends with a direct question specifically addressed *to the user*, then the **'user'** should speak next.
3.  **Waiting for User:** If your last response completed a thought, statement, or task *and* does not meet the criteria for Rule 1 (Model Continues) or Rule 2 (Question to User), it implies a pause expecting user input or reaction. In this case, the **'user'** should speak next."""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": ("Brief explanation justifying the 'next_speaker' choice based "
                            "*strictly* on the applicable rule and the content/structure of "
                            "the previous turn."),
        },
        "next_speaker": {"type": "string", "enum": ["user", "model"]},
    },
    "required": ["reasoning", "next_speaker"],
}

CONTINUE_MESSAGE = "Please continue."


def _is_function_response(content: Content) -> bool:
    return (content.role == "user" and bool(content.parts)
            and all(p.function_response is not None for p in content.parts))


async def check_next_speaker(chat, generator, model: str) -> str | None:
    """Returns 'user' | 'model' | None (None == treat as user).

    `chat` is a GeminiChat; `generator` a ContentGenerator (side queries
    bypass the main history — they're stateless one-shots).
    """
    comprehensive = chat.get_history(curated=False)
    if not comprehensive:
        return None
    last = comprehensive[-1]

    # Rule shortcuts (no LLM call needed):
    if _is_function_response(last):
        return "model"  # tool results always hand the floor back to the model
    if last.role == "model" and not last.parts:
        return "model"  # filler model message with no content
    curated = chat.get_history(curated=True)
    if not curated or curated[-1].role != "model":
        return None

    last_model = curated[-1]
    contents = [last_model, Content(role="user", parts=[Part(text=CHECK_PROMPT)])]
    try:
        raw = await generator.generate_json(model=model, contents=contents,
                                            schema=RESPONSE_SCHEMA)
        verdict = json.loads(raw) if isinstance(raw, str) else raw
        speaker = verdict.get("next_speaker")
        return speaker if speaker in ("user", "model") else None
    except Exception:
        return None  # any failure -> assume the user speaks next
