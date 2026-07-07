# Policy replay — historical corpus (B9-lite)

user_turns replayed: 754 (main sessions, trajectory state walked in order)

## Predicted routing

| Rung | n | share |
|---|---|---|
| local | 480 | 63.7% |
| frontier | 236 | 31.3% |
| cheap_cloud | 38 | 5.0% |

## Decided by

| Layer | n | share |
|---|---|---|
| gate:pin_context_conflict | 297 | 39.4% |
| gate:feasibility | 273 | 36.2% |
| gate:privacy | 97 | 12.9% |
| middle_default | 82 | 10.9% |
| heuristic | 4 | 0.5% |
| heuristic:retry_signal | 1 | 0.1% |

Phase-3 learned router would own only the `middle_default` band — 82 turns (10.9%). Pin-context conflicts surfaced (§5.8): 297.

## Ground-truth checks

- PASS — secret-flagged sessions always pinned (394 turns)
- PASS — oversized-context turns never local (570 turns)
- PASS — hard-verb turns -> frontier (1 turns)
- PASS — post-interrupt turns escalate (15 turns)
- PASS — decision latency p50 < 20ms

Latency: p50 0.8us, p99 4.5us (budget: <20ms; <30ms decision path §11)

**Verdict: ALL CHECKS PASS**
