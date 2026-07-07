"""P1-A2 — offline shadow of the utility-pinning hook.

Replays hook.route_model over every capture and reports what it WOULD rewrite,
without touching live traffic. This is the "shadow-before-live" gate: hand-review
the predicted rewrites, confirm zero real turns get mis-pinned, before flipping
the hook live in the LiteLLM path.

Usage: PYTHONPATH=src .venv/bin/python -m router.shadow_hook
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

from .discriminator import classify
from .hook import route_model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captures", default="data/captures/*/*.json")
    ap.add_argument("--report", type=Path, default=Path("reports/p1-utility-shadow.md"))
    args = ap.parse_args()

    files = sorted(glob.glob(args.captures))
    rewrites = Counter()
    passthrough = Counter()
    # safety check: no user_turn / continuation / passthrough should ever be rewritten
    unsafe = []
    orig_models = Counter()

    for f in files:
        r = json.load(open(f))
        try:
            body = json.loads(r.get("request_body") or "null")
        except json.JSONDecodeError:
            body = None
        label = classify(r["method"], r["path"], body)
        target = route_model(r["method"], r["path"], body)
        if target:
            rewrites[label] += 1
            if isinstance(body, dict):
                orig_models[body.get("model", "?")] += 1
            if label.startswith(("user_turn", "continuation", "passthrough")):
                unsafe.append((f, label, target))
        else:
            passthrough[label] += 1

    total = len(files)
    n_rewrite = sum(rewrites.values())
    L = ["# P1-A shadow — utility-pinning hook (offline, no live routing)", "",
         f"Replayed over {total} captures. **{n_rewrite} would be rewritten to local-small "
         f"({100*n_rewrite/total:.1f}%)**; the rest pass through unchanged.", "",
         "## Would rewrite (label -> local-small)", "", "| Label | n |", "|---|---|"]
    for label, n in rewrites.most_common():
        L.append(f"| {label} | {n} |")
    L += ["", "## Original model of rewritten requests (what we'd stop paying for)", ""]
    for m, n in orig_models.most_common():
        L.append(f"- {m}: {n}")
    L += ["", "## Left unchanged", "", "| Label | n |", "|---|---|"]
    for label, n in passthrough.most_common():
        L.append(f"| {label} | {n} |")
    L += ["", "## Safety check", ""]
    if unsafe:
        L.append(f"- **FAIL — {len(unsafe)} non-utility requests would be rewritten** (must be zero):")
        for f, label, target in unsafe[:10]:
            L.append(f"  - {label} -> {target}  ({Path(f).name})")
    else:
        L.append("- PASS — only utility housekeeping is rewritten; zero user_turn / "
                 "continuation / passthrough requests touched.")
    L += ["", f"**Verdict: {'SAFE TO GO LIVE' if not unsafe else 'DO NOT GO LIVE'}** "
          "(pending user approval of live routing, decision D-1)."]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0 if not unsafe else 2


if __name__ == "__main__":
    sys.exit(main())
