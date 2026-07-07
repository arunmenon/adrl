# P1-A shadow — utility-pinning hook (offline, no live routing)

Replayed over 19039 captures. **133 would be rewritten to local-small (0.7%)**; the rest pass through unchanged.

## Would rewrite (label -> local-small)

| Label | n |
|---|---|
| utility:sidecar | 133 |

## Original model of rewritten requests (what we'd stop paying for)

- local-small: 122
- claude-haiku-4-5: 11

## Left unchanged

| Label | n |
|---|---|
| utility:sidecar | 16275 |
| passthrough:count_tokens | 1704 |
| continuation | 453 |
| utility:prewarm | 230 |
| user_turn | 210 |
| passthrough:non_api | 34 |

## Safety check

- PASS — only utility housekeeping is rewritten; zero user_turn / continuation / passthrough requests touched.

**Verdict: SAFE TO GO LIVE** (pending user approval of live routing, decision D-1).
