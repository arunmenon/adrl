"""P1 telemetry — aggregate routing captures into the §11 metrics view.

Reads data/captures/ (organic, live) and reports what the router is actually
doing now that P1-A routing is live: local-serve rate, fail-open rate, latency
(local vs the cloud baseline), estimated spend avoided, and error rate. This is
the "don't run live routing blind" readout.

Usage:
  PYTHONPATH=src .venv/bin/python -m router.telemetry            # full history
  PYTHONPATH=src .venv/bin/python -m router.telemetry --today    # today only
  PYTHONPATH=src .venv/bin/python -m router.telemetry --last 500 # most recent N
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Cloud price for the models utility calls were originally headed to ($/MTok in/out).
PRICES = {
    "claude-opus-4-8": (5.0, 25.0), "claude-opus-4-7": (5.0, 25.0),
    "claude-fable-5": (10.0, 50.0), "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5": (3.0, 15.0),
}


def _price(model: str) -> tuple[float, float]:
    for k, v in PRICES.items():
        if model.startswith(k):
            return v
    return PRICES["claude-opus-4-8"]


def _pct(a: int, b: int) -> str:
    return f"{100*a/b:.1f}%" if b else "n/a"


def _median(xs: list) -> float:
    return statistics.median(xs) if xs else 0.0


def load(pattern: str, today: bool, last: int | None) -> list[dict]:
    files = sorted(glob.glob(pattern))
    if today:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        files = [f for f in files if f"/{day}/" in f]
    if last:
        files = files[-last:]
    out = []
    for f in files:
        try:
            out.append(json.load(open(f)))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def build(caps: list[dict]) -> str:
    n = len(caps)
    routed = [c for c in caps if c.get("routed_to_local")]
    served_local = [c for c in routed if not c.get("local_fallback")]
    fell_back = [c for c in routed if c.get("local_fallback")]
    errors = [c for c in routed if isinstance(c.get("status"), int) and c["status"] >= 400
              and not c.get("local_fallback")]

    # latency: locally-served utility vs. a cloud-served baseline (non-routed 200s)
    local_lat = sorted(c.get("latency_ms", 0) for c in served_local)
    cloud_lat = sorted(c.get("latency_ms", 0) for c in caps
                       if not c.get("routed_to_local") and c.get("status") == 200
                       and c.get("latency_ms"))

    # spend avoided: each locally-served utility call would have cost cloud tokens
    avoided = 0.0
    orig_models = Counter()
    for c in served_local:
        try:
            body = json.loads(c.get("request_body") or "{}")
            resp = json.loads(c.get("response_body") or "{}")
        except json.JSONDecodeError:
            continue
        model = body.get("model", "?")
        orig_models[model] += 1
        pin, pout = _price(model)
        u = resp.get("usage", {}) if isinstance(resp, dict) else {}
        avoided += (u.get("input_tokens", 0) / 1e6 * pin
                    + u.get("output_tokens", 0) / 1e6 * pout)

    L = ["# P1 routing telemetry", "",
         f"Captures analyzed: {n} | utility calls routed to local rung: **{len(routed)}** "
         f"({_pct(len(routed), n)} of traffic)", ""]

    L += ["## Local rung health", "",
          f"- Served locally: **{len(served_local)}** ({_pct(len(served_local), len(routed))} of routed)",
          f"- Fell open to cloud (fail-open fired): **{len(fell_back)}** "
          f"({_pct(len(fell_back), len(routed))}) — the safety net; higher = local stack unhealthy",
          f"- Errors surfaced to user on routed calls: **{len(errors)}** "
          f"(target: 0 — fail-open should catch local failures)", ""]

    L += ["## Latency (ms)", "",
          f"- Utility served **local**: p50 {_median(local_lat):.0f}, "
          f"p90 {local_lat[int(len(local_lat)*0.9)] if local_lat else 0}, "
          f"max {local_lat[-1] if local_lat else 0}",
          f"- Cloud baseline (non-routed 200s): p50 {_median(cloud_lat):.0f}",
          f"- Read: if local p50 >> cloud p50, the local rung is slow for housekeeping "
          "(cold loads / memory pressure) — consider a smaller model or keep-warm.", ""]

    L += ["## Spend avoided (est.)", "",
          f"- Locally-served utility calls would have cost **${avoided:.4f}** at cloud rates.",
          f"- Original models (what we stopped paying for): "
          f"{', '.join(f'{m}×{c}' for m, c in orig_models.most_common()) or '—'}",
          "- Note: this is the direct saving on housekeeping only — the modest slice. "
          "The subagent lever (P1-D) is where real money moves.", ""]

    # alarms
    fb_rate = len(fell_back) / len(routed) if routed else 0
    L += ["## Alarms", ""]
    L.append(f"- Fail-open rate {_pct(len(fell_back), len(routed))}: "
             + ("OK" if fb_rate < 0.2 else "**HIGH — local stack may be unhealthy, investigate**"))
    L.append(f"- User-facing errors on routed calls: "
             + ("OK (0)" if not errors else f"**{len(errors)} — fail-open gap, investigate**"))
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captures", default="data/captures/*/*.json")
    ap.add_argument("--today", action="store_true")
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--report", type=Path, default=None,
                    help="also write the readout to this path")
    args = ap.parse_args()

    caps = load(args.captures, args.today, args.last)
    if not caps:
        print("no captures found", file=sys.stderr)
        return 1
    md = build(caps)
    print(md)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
