"""Shared outcome labels (WS1).

``outcome_proxy_hard`` is the router's only ground-truth-ish signal for
"this turn really was hard". It started life inside the classifier shadow
harness (shadow_classifier.py) but is consumed by the memory ledger (WS1),
rule health (WS3), and the retrieval router (WS4) — so it lives here as the
single source of truth and everyone imports it (precedent: canon.py).

Import-only module — no CLI.
"""

from __future__ import annotations


def outcome_proxy_hard(row: dict) -> bool:
    """Weak execution-friction proxy for 'this turn really was hard'.

    Same definition the bake-off used: any edit-apply failure, any errored tool
    result, a user interrupt, or a runaway (>=10 continuations). Non-validating
    (see classifier-bakeoff.md caveats) but the only ground-truth-ish signal we
    have for whether frontier help was warranted.
    """
    return (
        (row.get("n_edit_failures") or 0) >= 1
        or (row.get("n_error_results") or 0) >= 1
        or bool(row.get("interrupted"))
        or (row.get("n_continuations") or 0) >= 10
    )
