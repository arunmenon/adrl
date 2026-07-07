"""P1-D live demo: local model attempts a turn, trip-wire fires, escalate to cloud.

Proves the SEMANTIC escalation ladder end-to-end with LIVE inference:
  local-code (ollama) attempts an edit -> real response -> EscalationController
  observes -> on a trip-wire, the escalated request is reissued to the cloud rung
  (real) which completes it.

Unlike LiteLLM's infra fallback (S7, fires on a dead endpoint), this fires on a
bad-but-valid local response — the thing a transport proxy can't see.

Routes through LiteLLM (:4001): local-code = ollama, frontier = Anthropic.
Usage: PYTHONPATH=src .venv/bin/python -m router.demo_escalation
"""

from __future__ import annotations

import json
import sys
import urllib.request

from router.escalation_controller import EscalationController
from router.state import DictSessionStore

LITELLM = "http://localhost:4001/v1/messages"

# A file with mixed whitespace + a tricky exact-string edit — the classic case a
# small model botches (regenerates the snippet from memory, tabs vs spaces).
FILE = 'def parse(s):\n\tfmt = "%Y-%m-%d"          # note: tab indent + trailing spaces\n\treturn datetime.strptime(s, fmt)\n'
EDIT_INSTR = (
    "Here is a file:\n\n" + FILE + "\n"
    "Make exactly one Edit tool call that changes the return line to strip the input: "
    "`return datetime.strptime(s.strip(), fmt)`. The Edit tool requires old_string to be a "
    "BYTE-EXACT substring of the file (including the tab indentation and trailing spaces). "
    "Emit a single tool_use for a tool named Edit with old_string and new_string."
)


def _call(model: str, messages: list, max_tokens: int = 400) -> dict:
    body = json.dumps({
        "model": model, "max_tokens": max_tokens,
        "tools": [{"name": "Edit", "description": "Exact-string file edit.",
                   "input_schema": {"type": "object", "properties": {
                       "old_string": {"type": "string"}, "new_string": {"type": "string"}},
                       "required": ["old_string", "new_string"]}}],
        "messages": messages,
    }).encode()
    req = urllib.request.Request(LITELLM, data=body, method="POST", headers={
        "content-type": "application/json", "x-api-key": "sk-x",
        "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def _edit_applies(blocks: list, file_text: str) -> tuple[bool, str]:
    for b in blocks:
        if b.get("type") == "tool_use" and b.get("name") == "Edit":
            old = b.get("input", {}).get("old_string", "")
            return (old in file_text), old[:60]
    return False, "(no Edit call emitted)"


def main() -> int:
    store = DictSessionStore()
    store.set_route("demo", "local-code")
    ec = EscalationController(store)
    ec.new_turn("demo")
    sid = "demo"
    messages = [{"role": "user", "content": EDIT_INSTR}]

    print(f"session route starts at: {ec.current_route(sid)}")
    print("=" * 64)

    escalated = False
    for attempt in range(1, 4):   # the harness would retry a failed edit; cap at 3
        rung = ec.current_route(sid)
        if rung != "local-code":
            escalated = True
            break
        print(f"\n[attempt {attempt}] asking LOCAL rung ({rung}) to make the edit...")
        try:
            resp = _call("local-code", messages)
        except Exception as exc:
            print(f"  local call failed (infra): {exc}"); return 1
        blocks = resp.get("content", [])
        applies, old = _edit_applies(blocks, FILE)
        print(f"  local proposed old_string={old!r}")
        print(f"  edit applies to the real file? {applies}")

        if applies:
            print("\nLocal succeeded — no escalation needed (a valid outcome).")
            print(f"final route: {ec.current_route(sid)}")
            return 0

        # simulate the harness returning the edit-failure tool_result
        tool_id = next((b["id"] for b in blocks if b.get("type") == "tool_use"), "t?")
        fail_result = [{"type": "tool_result", "tool_use_id": tool_id, "is_error": True,
                        "content": "<tool_use_error>String to replace not found in file."}]
        messages.append({"role": "assistant", "content": blocks})
        messages.append({"role": "user", "content": fail_result})
        decision = ec.observe_tool_results(sid, fail_result)
        print(f"  edit failed -> trip-wire check: {decision.reason or 'strike recorded, no fire yet'}")
        if decision.escalate:
            print(f"\n⚡ ESCALATION: {decision.from_rung} -> {decision.to_rung} "
                  f"({decision.tripwire}, {decision.tripwire_type})")
            escalated = True
            break

    if not escalated:
        print("\nNo escalation fired within the attempt budget.")
        return 0

    # reissue to the escalated (cloud) rung — real cloud inference
    target = ec.current_route(sid)
    print(f"\n[escalated] reissuing the turn to {target} (real cloud)...")
    try:
        resp = _call(target, messages)
    except Exception as exc:
        print(f"  cloud call failed: {exc}"); return 1
    blocks = resp.get("content", [])
    applies, old = _edit_applies(blocks, FILE)
    txt = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")[:80]
    print(f"  cloud proposed old_string={old!r}")
    print(f"  edit applies? {applies}  served_by={resp.get('model')}")
    print("\n" + "=" * 64)
    print("RESULT: local rung tripped the edit-apply wire -> semantic escalation to "
          f"cloud -> {'cloud completed the edit ✓' if applies else 'cloud responded'}. "
          "The ladder works on a bad-but-valid local response (not an infra failure).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
