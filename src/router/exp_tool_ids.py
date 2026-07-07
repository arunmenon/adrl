"""B5 — tool-call ID cross-provider experiment.

The design's escalation rebuild (§5.5) assumes tool_use/tool_result IDs must be
re-minted when a transcript moves between providers, and carries persistent
ID-map machinery for it. The internal review flagged this as probably
overstated: the API may only require IDs to be *internally consistent within a
request*, not provider-minted. This settles it empirically.

Sends a two-turn conversation (assistant tool_use -> user tool_result) under
several ID regimes and records which the Anthropic API accepts (HTTP 200) vs
rejects (400). Reads the key from data/anthropic-key; never prints it.

Usage: PYTHONPATH=src .venv/bin/python -m router.exp_tool_ids
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

KEY_FILE = Path("data/anthropic-key")
URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5"  # cheapest; the question is protocol acceptance, not quality


def _key() -> str:
    return KEY_FILE.read_text().strip()


def _tool_def() -> list:
    return [{
        "name": "get_weather",
        "description": "Get weather for a city.",
        "input_schema": {"type": "object",
                         "properties": {"city": {"type": "string"}},
                         "required": ["city"]},
    }]


def _conversation(use_id: str, result_id: str) -> list:
    """A completed tool round-trip with configurable IDs on each side."""
    return [
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": use_id, "name": "get_weather", "input": {"city": "Paris"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": result_id, "content": "18C, clear"},
        ]},
    ]


def _post(messages: list) -> tuple[int, str]:
    body = json.dumps({
        "model": MODEL, "max_tokens": 64,
        "tools": _tool_def(), "messages": messages,
    }).encode()
    req = urllib.request.Request(URL, data=body, method="POST", headers={
        "content-type": "application/json",
        "x-api-key": _key(),
        "anthropic-version": "2023-06-01",
    })
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, "ok"
    except urllib.error.HTTPError as e:
        detail = json.loads(e.read().decode()).get("error", {}).get("message", "")[:120]
        return e.code, detail


CASES = [
    # (name, tool_use id, tool_result id, what it tests)
    ("anthropic_native_consistent", "toolu_01ABC", "toolu_01ABC",
     "baseline: normal IDs, matched"),
    ("foreign_format_consistent", "call_9",       "call_9",
     "OpenAI-style IDs, internally matched — does format alone matter?"),
    ("arbitrary_format_consistent", "L1-local-xyz", "L1-local-xyz",
     "made-up local IDs, internally matched — provenance vs consistency"),
    ("mismatched_ids", "toolu_01ABC", "toolu_01ZZZ",
     "well-formed but tool_result points at a different id — internal consistency broken"),
]


def main() -> int:
    if not KEY_FILE.exists() or not KEY_FILE.read_text().strip():
        print(f"no key at {KEY_FILE}", file=sys.stderr)
        return 1

    rows = []
    for name, uid, rid, desc in CASES:
        status, detail = _post(_conversation(uid, rid))
        verdict = "ACCEPT" if status == 200 else f"REJECT({status})"
        rows.append((name, verdict, desc, detail))
        print(f"{name:<30} {verdict:<12} {desc}")
        if status != 200:
            print(f"    -> {detail}")

    accept = {n for n, v, *_ in rows if v == "ACCEPT"}
    foreign_ok = "foreign_format_consistent" in accept and "arbitrary_format_consistent" in accept
    mismatch_rejected = "mismatched_ids" not in accept

    L = ["# B5 — tool-call ID cross-provider experiment", "",
         f"Model: {MODEL}. Each case is a completed tool round-trip with IDs varied.", "",
         "| Case | Result | Tests |", "|---|---|---|"]
    for name, verdict, desc, _ in rows:
        L.append(f"| `{name}` | {verdict} | {desc} |")
    L += ["", "## Verdict", ""]
    if foreign_ok and mismatch_rejected:
        L += [
            "**IDs need only be internally consistent, not provider-minted.** Foreign-format "
            "and arbitrary IDs are accepted as long as tool_use.id == tool_result.tool_use_id; "
            "a mismatch is rejected.",
            "",
            "**Design impact (§5.5):** the escalation rebuild does NOT need to re-mint IDs into "
            "a provider's namespace. It only needs to preserve internal pairing — which the "
            "harness's own transcript already does. The persistent ID-map machinery the design "
            "carried can be dropped; the internal review's suspicion is confirmed. Thinking-block "
            "signatures and encrypted reasoning remain the real cross-provider blockers "
            "(unchanged).",
        ]
    else:
        L += [
            f"Foreign/arbitrary IDs accepted: {foreign_ok}. Mismatch rejected: {mismatch_rejected}.",
            "Result is more nuanced than a clean 'internal-consistency-only' — see the table; "
            "the design's re-minting assumption may hold in part. Inspect rejection messages.",
        ]
        for name, verdict, desc, detail in rows:
            if verdict != "ACCEPT":
                L.append(f"- `{name}`: {detail}")

    out = Path("reports/assumption-tool-ids.md")
    out.write_text("\n".join(L) + "\n")
    print("\n" + "\n".join(L[6:]))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
