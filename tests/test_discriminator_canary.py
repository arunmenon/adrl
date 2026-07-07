"""B10 — discriminator drift canary.

Replays frozen, scrubbed wire-shape fixtures (real captures reduced to the
fields the discriminator reads — no payloads, no secrets) and fails if any
label changes. A failure means either our rules regressed or a harness update
shifted the wire shapes — both are stop-and-look events, caught at the desk
instead of as silent misrouting in production.

Run: PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
Refresh fixtures (after intentional rule changes): python -m router.freeze_fixtures
"""

import json
from pathlib import Path

import pytest

from router.discriminator import classify

FIXTURES = sorted(Path(__file__).parent.glob("fixtures/handshakes/*.json"))


def test_fixtures_exist():
    assert len(FIXTURES) >= 10, "canary needs a representative fixture set"


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_label_frozen(path):
    fx = json.loads(path.read_text())
    got = classify(fx["method"], fx["path"], fx["body"])
    assert got == fx["expected_label"], (
        f"{path.name}: label drifted {fx['expected_label']!r} -> {got!r} — "
        "harness wire shape changed or a rule regressed"
    )
