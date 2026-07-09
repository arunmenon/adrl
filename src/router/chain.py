"""FallbackChain — the one chain-of-responsibility implementation (WS0).

Both the backend adapters (WS0) and the memory providers (WS1 facade) need the
same semantics: try each implementation in configured order; the first
non-None answer wins; a member that fails (returns None or raises) simply
passes the call down the chain. This module is the single implementation so
the pattern is never copy-pasted.

The chain duck-types: calling any method on it forwards that method to each
member that has it, in order. Members remain individually fail-safe; the chain
adds one more guarantee — iterating members never raises either.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence


class FallbackChain:
    """Ordered fallback over implementations sharing a duck-typed contract.

    >>> chain = FallbackChain([primary, secondary, terminal])
    >>> chain.chat(messages)          # first member whose .chat returns non-None
    """

    def __init__(self, members: Sequence[Any]):
        if not members:
            raise ValueError("FallbackChain needs at least one member")
        self.members = list(members)

    def __getattr__(self, name: str):
        # Only called for attributes not found on the chain itself; forwards
        # the method call down the members in order.
        def dispatch(*args, **kwargs) -> Optional[Any]:
            for member in self.members:
                method = getattr(member, name, None)
                if method is None:
                    continue
                try:
                    result = method(*args, **kwargs)
                except Exception:
                    continue  # member misbehaved — fall through, never raise
                if result is not None:
                    return result
            return None

        return dispatch

    def __repr__(self) -> str:
        names = " -> ".join(type(m).__name__ for m in self.members)
        return f"FallbackChain({names})"
