#!/usr/bin/env python3
"""Validate the frozen readiness method, baseline, current score, and history."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router.learning_readiness import (  # noqa: E402
    ReadinessError,
    validate_persisted_score_artifacts,
)


def main() -> int:
    try:
        result = validate_persisted_score_artifacts()
    except ReadinessError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        "Readiness scoring OK: "
        f"{result['current_score']:.1f}/100 "
        f"({result['current_delta']:+.1f} from "
        f"{result['baseline_id']}), contract {result['contract_version']}, "
        f"{result['history_entries']} history entries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
