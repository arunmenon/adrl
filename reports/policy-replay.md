# Policy replay — historical corpus (B9-lite)

user_turns replayed: 754 (main sessions, trajectory state walked in order)

## Predicted routing

| Rung | n | share |
|---|---|---|
| local | 615 | 81.6% |
| frontier | 113 | 15.0% |
| cheap_cloud | 26 | 3.4% |

## Decided by

| Layer | n | share |
|---|---|---|
| gate:pin_context_conflict | 431 | 57.2% |
| gate:feasibility | 139 | 18.4% |
| gate:privacy | 119 | 15.8% |
| middle_default | 63 | 8.4% |
| heuristic | 2 | 0.3% |

Phase-3 learned router would own only the `middle_default` band — 63 turns (8.4%). Pin-context conflicts surfaced (§5.8): 431.

## Ground-truth checks

- PASS — secret-flagged sessions always pinned (550 turns)
- PASS — oversized-context turns never local (570 turns)
- PASS — hard-verb turns -> frontier (1 turns)
- PASS — post-interrupt turns escalate (15 turns)
- PASS — decision latency p50 < 20ms

Latency: p50 0.5us, p99 3.0us (budget: <20ms; <30ms decision path §11)

**Verdict: ALL CHECKS PASS**

## The finding this replay surfaced

The gate distribution is dominated by privacy pinning: the 7 secret-flagged sessions
(A8 scan) contain 550 of 754 user turns — they are the *longest* sessions, so a pin
early in a big session compounds across everything after it, and 431 of those turns
also outgrow local context (the §5.8 pin-context collision — not an edge case in this
corpus; the dominant case).

Two readings, both actionable:
1. **If the A8 patterns are right**, the pin-vs-growth collision handling (§5.8) is
   front-line behavior, not a corner case — and local-context size directly bounds how
   much of a pinned session stays usable.
2. **If the A8 patterns over-trigger** (the `env_assignment` regex is deliberately
   loose), then secret-scanner precision is the single highest-leverage tuning knob in
   the whole policy — design §13.4's "tune false-positive rate" open question is now
   priority one, with data to tune against.

Excluding pinned sessions: the remaining 204 turns split ~68% local-cascade
(heuristic+middle), ~13% feasibility-gated to cheap-cloud, and the retry/hard rules
fired correctly on all their cases. The Phase-3 learned router would own just 8.4%
of decisions — consistent with the design's "dumb rules decide most things" intent.
