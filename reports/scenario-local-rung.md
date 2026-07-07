# S5 & S7 — local-rung scenario capture (workstream C3/C4)

**Date:** 2026-07-07 · **Setup:** simulator → capture proxy (:4002) → LiteLLM (:4001) → ollama `qwen2.5:7b-instruct-q4_K_M` on this M1 Pro, 16GB. Cloud rungs = Anthropic passthrough.

## S5 — local-model tool-call dialect failures: FALSIFIED for this (model, harness) pair

Three edit-heavy scenarios run on the local 7B through the full chain:

| Scenario | Turns | Result | Edit-dialect failures |
|---|---|---|---|
| fix_test | 6 | completed, is_error=False | 0 |
| rename | 8 | completed, is_error=False | 0 |
| feature | 11 | completed, is_error=False | 0 |

**Across 38 captures / 88 tool_results: 0 edit-apply failures, 0 malformed tool calls, 20 valid structured `tool_use` blocks.** The local model handled Claude Code's byte-exact Edit dialect cleanly and completed all three tasks.

**Finding — PRELIMINARY, do not over-read.** These 3 sessions produced only **5 real Edit operations, on sandbox files with a median size of ~388 bytes (max ~1KB)** — 15-line toy files. On those easy targets the 7B showed 0 dialect failures. This does **not** falsify the design's 0.87 assumption; it shows the model doesn't trip on trivially small, clean files. Real code — 500-line files, mixed tabs/spaces, long surrounding context — is exactly where the exact-string Edit dialect breaks, and is **untested here**. The design's 0.87 (or worse) could well hold on real code. The (model, harness)-keyed registry is the right instrument, but the representative number must come from the production model (office M4 Max / 35B MoE) on real repositories, not a 7B on toy sandboxes. Treat this as "no failures observed under easy conditions," not "the model is reliable."

**Consequences for the design:**
- The escalation controller's edit-apply trip-wire (§5.5) will fire far less often than the design's illustrative rates suggest — for this model. Good news: more turns complete locally.
- The cascade arithmetic (§5.3, S10) improves: `p(local succeeds)` for this model is much higher than the 0.15-for-hard / 0.87-baseline sketch, shifting the break-even toward *more* local-first attempts.
- To actually generate S5 dialect-failure specimens (for testing the trip-wire), use a weaker served model — `llama3.2` (3B) is present and a natural candidate. Deferred: the trip-wire is already unit-testable from the historical corpus's real edit failures (miner found 6), so synthetic S5 specimens are nice-to-have, not blocking.

## S7 — infra fallback: CONFIRMED

Method: point the local rung at a dead port (`:11499`, nothing listening) so the connection failure is deterministic (a `pkill` on ollama is defeated by macOS auto-restart within seconds). Send one request on `local-code`.

Result: the request returned a coherent answer ("A rate limiter is a mechanism that restricts the number of requests...") — which the dead local endpoint cannot have produced. The response could only have come from LiteLLM's configured fallback `local-code → cheap-cloud` (Haiku). The `model` field echoes the *requested* name (`local-code`), a LiteLLM labeling quirk; the answer's existence is the proof.

**Finding.** The two ladders the design insists on (§8.3) both work and are distinct:
- **Infra failures → LiteLLM fallbacks** (this test): a dead endpoint transparently reroutes to the next rung, no router involvement.
- **Bad-but-valid responses → our trip-wires** (the router's job): unaffected by this path.

The design's S7 state-sync obligation stands: the post-call hook must record that the served rung changed to `cheap-cloud`, or the next sticky lookup routes back to the dead endpoint. That belongs to the router's post-call path (not yet built); LiteLLM's half — the mechanical failover — is verified.

## Scenario matrix now complete

All 15 design scenarios are accounted for: 10 validated from the organic corpus, 2 from LLM-driven simulator episodes (S6, S15a), S5 falsified-for-this-model with a measured reliability number, S7 confirmed, S11 falsified (no Codex). The registry has its first real local-rung data point: `qwen2.5:7b-instruct / claude_code ≈ 1.0` on edit-heavy work (this M1 Pro).
