# B4 — Session-keying assumption test: ANSWERED

**Date:** 2026-07-06 · **Evidence:** first live wire captures (resumed session + parallel session traffic through the B2 proxy)

## Verdict

`metadata.user_id` is **not** an opaque per-user hash. It is a JSON object with three components:

```json
{
  "device_id":   "cbdb361b… (64-hex, stable per machine)",
  "account_uuid": "fda637e4-… (stable per account)",
  "session_id":  "3d43a1e7-… (UNIQUE PER SESSION)"
}
```

Two concurrent sessions produced two distinct `session_id` values with identical
`device_id`/`account_uuid` — confirmed per-session granularity.

## Consequences for the design

1. **Session keying is solved for Claude Code**: parse `metadata.user_id` as JSON and key on
   `session_id`. The internal-review worry (keying on user_id might merge concurrent sessions
   and corrupt strike counters) is resolved — no merging occurs if we key correctly.
2. The design's fallback (`hash(system_prompt + first_user_message)`) is **not needed** for
   Claude Code traffic; keep it only for harnesses that send no session identity.
3. Design doc §5.6 "Session identity" and scenario S1's keying note can be updated from
   "verify this" to "verified — parse the JSON, key on session_id".

## Bonus wire observations from the same captures (feed B6 fingerprints)

- **`/v1/messages/count_tokens` is a request kind the design never enumerated.** The harness
  fires many of them (context accounting; 20 of the first 31 captures). They are not
  completions and must be a fourth passthrough kind in the discriminator — label
  `utility:count_tokens`, always pass through, never route.
- Main-thread completion requests hit `/v1/messages?beta=true` and their first system block
  begins `x-anthropic-billing-header:` — both are fingerprint constants for B6.
- Two 429s (rate limits) on the primary model were retried by the harness and served 200 —
  the infra-retry path (S7-adjacent) visible in organic traffic.
- Request bodies up to 3.4MB (full history replay of a long resumed session) — the proxy's
  256MB body cap and streaming relay handled them without issue.
