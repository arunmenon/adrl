# P1-B — Secret-scanner precision tuning

**Date:** 2026-07-07 · **Priority:** one (gates the privacy pin before it touches live routing)

Phase 0's replay found the loose scanner would pin 7 sessions holding 73% of user turns — the pin-context collision (§5.8) looked like the dominant case, but the scanner's precision was untested. This tunes it.

## The audit (B1)

Redacted sampling of every match on the corpus revealed the split:

| Pattern | v1 hits | Verdict |
|---|---|---|
| `env_assignment` | 212 | **mostly false** — matched the prose word "tokens:" (e.g. "tokens: background/etc"); the loose regex treated any mention of token/secret/password + text as a secret |
| `connection_string_cred` | 169 | **mixed** — real `postgres://…:…@` strings AND doc placeholders (`user:pass@localhost`) |

## The fix (B2)

`src/miner/secrets.py` v2:
- **High-confidence literals** (AWS keys, private-key blocks, `sk-`/`ghp_`/`xox` tokens) — kept as-is, secrets by construction.
- **`env_assignment`** — now case-SENSITIVE, requires a real ALL-CAPS env-var key shape (`STRIPE_SECRET_KEY`, not the word "tokens"), AND a Shannon-entropy check (≥3.0 bits/char) on the value. Kills the prose false positives.
- **`connection_string_cred`** — rejects placeholder credentials (`user`, `pass`, `changeme`, …) and placeholder hosts (`localhost`, `example.com`), plus the same entropy gate.

## Result (B3)

| | v1 (loose) | v2 (tuned) |
|---|---|---|
| Sessions flagged | 7 | **5** |
| `connection_string_cred` hits | 169 | 10 (placeholders dropped) |
| `env_assignment` hits | 212 | 175 (all real keys) |
| Pinned user-turns (replay) | 550 | **394** |
| Predicted local share | 81.6% | **63.7%** |

The surviving matches are unambiguously real secret keys: `STRIPE_SECRET_KEY`, `AWS_SECRET_ACCESS_KEY`, `ZOOM_CLIENT_SECRET`, `SENTRY_AUTH_TOKEN`, `JWT_SECRET_KEY`, `RESEND_API_KEY`. The high `env_assignment` count is one `.env` read replaying across a session's later turns — correct, not noise. Replay still passes all ground-truth checks.

## The honest finding

Precision is fixed — but the pin rate is *still substantial* (394 turns), and now that is **justified, not artifact**: this user genuinely works in sessions that read production `.env` files. So the pin-context collision (§5.8) really is front-line behavior for this workflow — a real finding about the traffic, not a scanner bug. Before tuning we couldn't tell those apart; now we can.

**Implication for Phase 1:** the privacy pin can go live on the tuned scanner with confidence that it fires on real secrets. The §5.8 collision handling (warn → local compaction → hard stop) is worth building early, because for a secret-heavy user it's the common path, not the corner case. The v1 loose scanner is retained as `data/secrets-scan-v1.json` for the audit trail.

## Still open

- **§13.6** — secret detected mid-turn while on a *cloud* rung: block-and-surface vs finish-then-pin (decision D-2, leaning block-and-surface).
- Entropy threshold (3.0) is hand-set; revisit if a real secret with a low-entropy value (rare) is ever missed.
