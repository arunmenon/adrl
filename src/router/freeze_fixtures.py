"""B10 — freeze scrubbed wire-shape fixtures for the discriminator canary test.

Takes real captures and reduces each to the MINIMAL SHAPE the discriminator
reads — path, max_tokens, tools-present flag, last-message kind, system marker,
a fake session id. All payload content (prompts, file bodies, secrets) is
dropped, so fixtures are committable. The canary test replays them and fails
if any label changes — catching harness updates that shift wire shapes, and
regressions in our own rules.

Usage: PYTHONPATH=src .venv/bin/python -m router.freeze_fixtures
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

from .discriminator import classify

PER_LABEL = 4
OUT = Path("tests/fixtures/handshakes")


def scrub(path: str, body: dict | None) -> dict | None:
    if not isinstance(body, dict):
        return None
    slim: dict = {}
    if "max_tokens" in body:
        slim["max_tokens"] = body["max_tokens"]
    if body.get("tools"):
        slim["tools"] = [{"name": "redacted"}] * min(len(body["tools"]), 2)
    sys_text = body.get("system", "")
    if isinstance(sys_text, list):
        sys_text = " ".join(b.get("text", "") for b in sys_text if isinstance(b, dict))
    markers = []
    for m in ("cc_entrypoint=cli", "cc_entrypoint=sdk", "summarizing conversations",
              "summarize this conversation"):
        if m in sys_text.lower() or m in sys_text:
            markers.append(m)
    if markers:
        slim["system"] = " ".join(markers)
    msgs = body.get("messages") or []
    if msgs and isinstance(msgs[-1], dict):
        last = msgs[-1]
        content = last.get("content")
        if isinstance(content, list):
            kinds = [{"type": b.get("type", "text")} for b in content if isinstance(b, dict)][:3]
            slim["messages"] = [{"role": last.get("role", "user"), "content": kinds}]
        else:
            slim["messages"] = [{"role": last.get("role", "user"), "content": "x"}]
        # preserve message-count signal for the sidecar rule without any payload
        slim["_n_messages"] = len(msgs)
        slim["messages"] = [{"role": "user", "content": "x"}] * (min(len(msgs), 3) - 1) + slim["messages"]
    if body.get("metadata"):
        slim["metadata"] = {"user_id": json.dumps({"session_id": "fixture-session"})}
    return slim


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    kept: dict[str, int] = {}
    n = 0
    for f in sorted(glob.glob("data/captures/*/*.json")):
        r = json.load(open(f))
        try:
            body = json.loads(r.get("request_body") or "null")
        except json.JSONDecodeError:
            body = None
        label = classify(r["method"], r["path"], body)
        if kept.get(label, 0) >= PER_LABEL:
            continue
        slim = scrub(r["path"], body)
        # the fixture must classify identically to the original or it's useless
        if classify(r["method"], r["path"], slim) != label:
            continue
        kept[label] = kept.get(label, 0) + 1
        n += 1
        fixture = {"method": r["method"], "path": r["path"], "body": slim,
                   "expected_label": label}
        (OUT / f"{label.replace(':', '_')}-{kept[label]}.json").write_text(
            json.dumps(fixture, indent=2))
    print(f"froze {n} fixtures: {kept}")


if __name__ == "__main__":
    main()
