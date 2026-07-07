# Discriminator evaluation — live wire captures

Captures: 1020 | fingerprints from cc 2.1.201-202 wire evidence

## Label distribution

| Label | n | share |
|---|---|---|
| passthrough:count_tokens | 740 | 72.5% |
| continuation | 135 | 13.2% |
| user_turn | 72 | 7.1% |
| utility:sidecar | 37 | 3.6% |
| passthrough:non_api | 30 | 2.9% |
| utility:prewarm | 6 | 0.6% |

## By entrypoint (cli = interactive, sdk = headless/simulator)

- **unknown**: passthrough:count_tokens=740, passthrough:non_api=30, utility:prewarm=6
- **cli**: user_turn=45, continuation=44, utility:sidecar=31
- **sdk**: continuation=91, user_turn=27, utility:sidecar=6

## Ground-truth checks

- PASS — count_tokens -> passthrough
- PASS — tool_result tail -> continuation (135 cases)
- PASS — simulator episode openers -> user_turn (9 sessions)

## Latency (per classification)

- p50: 0.3 µs | p99: 24.3 µs | budget: microseconds on the passthrough flood, <20ms total decision path — discriminator share is negligible

**Verdict: ALL CHECKS PASS**
