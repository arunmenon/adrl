"""Dependency-light canonicalization shared by the loop trip-wire and the miner.

Kept free of any heavy imports (no pyarrow / miner.parser) so the live post-call
path can import it on the proxy hot path. Both router.tripwires and
miner.scenarios import canonical_call from here.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def canonical_call(name: str, args: Any) -> str:
    """Stable hash of a tool call with args canonicalized so cosmetically-different
    repeats (trailing slash, key order, whitespace) collapse to one signature (S4)."""
    try:
        normalized = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        normalized = str(args)
    normalized = re.sub(r"/+(?=[\"'])", "", normalized)  # trailing path slashes
    normalized = re.sub(r"\s+", " ", normalized)
    return hashlib.sha1(f"{name}:{normalized}".encode()).hexdigest()
