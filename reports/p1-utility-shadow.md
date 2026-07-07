# P1-A shadow — utility-pinning hook (offline, no live routing)

Replayed over 1822 captures. **107 would be rewritten to local-small (5.9%)**; the rest pass through unchanged.

## Would rewrite (label -> local-small)

| Label | n |
|---|---|
| utility:sidecar | 107 |

## Original model of rewritten requests (what we'd stop paying for)

- claude-opus-4-8: 92
- claude-haiku-4-5: 9
- claude-sonnet-5: 6

## Left unchanged

| Label | n |
|---|---|
| passthrough:count_tokens | 1290 |
| continuation | 244 |
| user_turn | 145 |
| passthrough:non_api | 30 |
| utility:prewarm | 6 |

## Safety check

- PASS — only utility housekeeping is rewritten; zero user_turn / continuation / passthrough requests touched.

**Verdict: SAFE TO GO LIVE** (pending user approval of live routing, decision D-1).
