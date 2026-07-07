# Discriminator evaluation — live wire captures

Captures: 1630 | fingerprints from cc 2.1.201-202 wire evidence

## Label distribution

| Label | n | share |
|---|---|---|
| passthrough:count_tokens | 1136 | 69.7% |
| continuation | 228 | 14.0% |
| user_turn | 132 | 8.1% |
| utility:sidecar | 98 | 6.0% |
| passthrough:non_api | 30 | 1.8% |
| utility:prewarm | 6 | 0.4% |

## By entrypoint (cli = interactive, sdk = headless/simulator)

- **unknown**: passthrough:count_tokens=1136, passthrough:non_api=30, utility:sidecar=7, utility:prewarm=6, user_turn=1
- **cli**: continuation=99, user_turn=86, utility:sidecar=83
- **sdk**: continuation=129, user_turn=45, utility:sidecar=8

## Ground-truth checks

- PASS — count_tokens -> passthrough
- PASS — tool_result tail -> continuation (228 cases)
- PASS — simulator episode openers -> user_turn (12 sessions)

## Latency (per classification)

- p50: 0.3 µs | p99: 51.0 µs | budget: microseconds on the passthrough flood, <20ms total decision path — discriminator share is negligible

**Verdict: ALL CHECKS PASS**
