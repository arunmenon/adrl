"""S7 — induce an infra fallback and capture it.

Starts a session on the local rung, then kills the ollama endpoint mid-flight
so the next request gets a connection error. LiteLLM's router_settings.fallbacks
(local-code -> cheap-cloud) should transparently reroute to the cloud — the
infra path (design §8.3), distinct from the router's semantic escalation.

This proves the design's claim that two ladders coexist: exceptions ->
LiteLLM fallbacks, bad-but-valid responses -> our trip-wires.

Usage: PYTHONPATH=src .venv/bin/python -m simulator.induce_fallback
Requires: ollama (:11434), litellm (:4001), capture proxy (:4002), a cloud key.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROXY = "http://localhost:4002"
LEDGER = Path("data/sim-ledger.jsonl")


def _messages_request(model: str) -> tuple[int, dict]:
    body = json.dumps({
        "model": model, "max_tokens": 64,
        "messages": [{"role": "user", "content": "In one sentence, what is a rate limiter?"}],
    }).encode()
    req = urllib.request.Request(PROXY + "/v1/messages", data=body, method="POST", headers={
        "content-type": "application/json", "x-api-key": "sk-1234",
        "anthropic-version": "2023-06-01",
    })
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def _served_by(resp: dict) -> str:
    """The model field on the response tells us which rung actually answered."""
    return resp.get("model", "?")


def main() -> int:
    print("1. baseline request on local-code (ollama up)...")
    status, resp = _messages_request("local-code")
    print(f"   status={status} served_by={_served_by(resp)}")

    print("2. killing ollama to force an infra failure...")
    subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True)
    # also stop any ollama-runner subprocess holding the port
    subprocess.run(["pkill", "-f", "ollama runner"], capture_output=True)
    time.sleep(3)
    up = True
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
    except Exception:
        up = False
    print(f"   ollama reachable: {up}")

    print("3. request on local-code with ollama DOWN -> expect LiteLLM fallback to cheap-cloud...")
    status, resp = _messages_request("local-code")
    served = _served_by(resp)
    text = "".join(b.get("text", "") for b in resp.get("content", []) if isinstance(b, dict))[:80]
    print(f"   status={status} served_by={served}")
    print(f"   text: {text!r}")

    fell_back = status == 200 and "local" not in served.lower()
    print()
    if fell_back:
        print(f"S7 CONFIRMED: local endpoint down -> served by {served} (infra fallback fired, "
              "no router involvement). This is the exception ladder, not semantic escalation.")
    else:
        print(f"S7 inconclusive: status={status}, served_by={served}. Check data/litellm.log "
              "for the fallback decision; fallbacks require the cloud key to be loaded.")

    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as fh:
        fh.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "source": "simulator", "scenario": "S7_induced_fallback",
            "maps_to": "S7", "ollama_killed": True,
            "served_by_after_kill": served, "status": status,
            "fell_back": fell_back,
        }) + "\n")

    print("\n(restart ollama with tools/run_ollama.sh when done)")
    return 0 if fell_back else 2


if __name__ == "__main__":
    sys.exit(main())
