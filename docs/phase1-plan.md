# Phase 1 Implementation Plan — Adaptive Routing Layer

**Status:** Draft · **Date:** 2026-07-07 · **Predecessor:** Phase 0 complete (`reports/phase0-exit.md`)

Phase 0 proved the machinery and the assumptions offline. **Phase 1 is the first phase where the router changes what model serves real traffic** — so it starts with the slices where being wrong is nearly free, keeps a one-keystroke kill switch, and gates every escalation of scope behind a measured exit criterion.

Two things Phase 0 explicitly deferred are prerequisites here and appear as workstreams: **secret-scanner precision** (the dominant-case knob) and a **constrained/enterprise baseline** (so the economics generalize beyond the single-user Max-plan window).

**Effort key:** S < half a day · M ~1 day · L 1–3 days.

---

## Guardrails (apply to every workstream)

- **Kill switch:** unset `ANTHROPIC_BASE_URL`; all traffic goes direct. Nothing in Phase 1 removes this.
- **Fail-open:** any router error → pass the request through unrouted, log it. A routing bug must never fail a user's request.
- **Shadow-before-live:** every routing rule runs in log-only mode first and is diffed against what it *would* have done, before it acts.
- **Canary:** `tests/test_discriminator_canary.py` must stay green; extend it as rules are added.
- **Provenance:** organic vs simulator vs pilot traffic stays separable in every dataset.

---

## Workstream P1-A — Utility-call pinning (near-zero risk, ship first)

The design's original Phase-1 win, now with a corrected fingerprint (Phase 0: sidecars are Opus/Sonnet-shaped, not Haiku-class). Route housekeeping to `local-small`.

| Step | Deliverable | Acceptance | Effort |
|---|---|---|---|
| A1 | Pre-call hook (`src/router/hook.py`) in the LiteLLM path: on `discriminator.classify() == utility:*`, rewrite the model to `local-small`; everything else untouched | Hook loads; `passthrough`/`continuation`/`user_turn` traffic is byte-for-byte unchanged | M |
| A2 | Shadow diff: run the hook in log-only mode over live traffic for 1 day | ≥50 utility calls seen; predicted rewrites reviewed by hand, zero false rewrites of real turns | S |
| A3 | Go live on `utility:light` only (titles, topic detection); `utility:compaction` stays cloud (quality-critical, §5.1) | 1 week, no harness breakage; session titles still generated | S |
| A4 | Measure: utility calls served local, tokens saved | Report `reports/p1-utility.md` with before/after | S |

**Exit:** utility:light pinned local for 1 week, zero breakage, kill switch verified.

---

## Workstream P1-B — Secret-scanner precision (PRIORITY ONE, gates the pin)

Phase 0's replay found the 7 secret-flagged sessions hold 73% of user turns — the pin-context collision is the *dominant* case, so a loose scanner mis-pins most traffic. **This must be tuned before any pin gates live routing.**

| Step | Deliverable | Acceptance | Effort |
|---|---|---|---|
| B1 | Precision audit: hand-label the current matches in `data/secrets-scan.json` (esp. the loose `env_assignment` regex) as true/false positive | Labeled set; measured precision of each pattern | M |
| B2 | Tighten patterns (entropy threshold, known-prefix allowlists, context requirements) → `src/miner/secrets.py` v2 | False-positive rate < target (set from B1); true secrets (AWS keys, connection strings) still caught | M |
| B3 | Re-run replay with tuned scanner | `reports/policy-replay.md` v2: pinned-session share drops to a defensible number; document the new pin rate | S |
| B4 | Decide the mid-turn-on-cloud-rung secret case (design §13.6, still open): block-and-surface vs finish-then-pin | Decision recorded in design doc; leaning block-and-surface | S |

**Exit:** scanner precision measured and defensible; pin rate reflects real secrets, not regex noise.

---

## Workstream P1-C — The post-call path (the main engineering lift)

Everything so far is pre-call (pick a model). Escalation needs the **post-call path**: inspect responses, run trip-wires, rebuild-and-reissue upward, sync fallback state. This is the gate to Phase 2 (routing real user turns) and to the subagent pilot.

| Step | Deliverable | Acceptance | Effort |
|---|---|---|---|
| C1 | Trip-wire evaluator (`src/router/tripwires.py`): edit-apply ×2, malformed-call ×2, identical-call ×3 (canonicalized), no-progress ×6, turn-budget — reads the post-call response + next request | Unit-tested against the miner's real edit-failure turns and the simulator loop traces | L |
| C2 | Escalation rebuild (`src/router/escalate.py`): strip thinking blocks, keep tool IDs as-is (B5: no re-minting), prepend 3-line failure note, reissue to next rung | Replays a captured failing local turn → frontier accepts it (no 400); note present | M |
| C3 | Session state store (`src/router/state.py`): the §5.6 fields (route, strikes, privacy_pinned, escalated_this_episode) — dict-backed behind an interface, Redis-swappable | `get/set/incr/pin` covered by tests; strikes reset per turn | M |
| C4 | Fallback state-sync (S7): post-call hook records the *actually served* rung so the next sticky lookup doesn't route to a dead endpoint | Induced-fallback test shows `session.route` updated to the served rung | S |
| C5 | Shadow the whole post-call path over live traffic (log-only): "would have escalated here" | Predicted escalation rate < 30% on real turns (design §10 Phase-2 gate, measured early) | M |

**Exit:** post-call path passes on replayed + shadow traffic; escalation rebuild verified end-to-end; nothing live yet.

---

## Workstream P1-D — Subagent-local pilot (the dollar lever, needs C)

The re-prioritized value bet. Route bounded, read-only subagent turns to `local-code`, trip-wires armed (needs P1-C). Start behind shadow, then live on the safest subset.

| Step | Deliverable | Acceptance | Effort |
|---|---|---|---|
| D1 | Subagent-scope classifier: read-only tool set, fresh small context, bounded task (design S13) → eligible-for-local flag | Labels the 3,478 workflow-subagent turns in the corpus; precision hand-checked | M |
| D2 | Shadow: predict subagent-local routing + escalation over live/replayed subagent traffic | Predicted subagent escalation rate ≤ frontier baseline; $ saved estimated | M |
| D3 | Live pilot on read-only `Explore`/`Grep`-class subagents only, kill switch armed | 1 week; subagent failure rate ≤ frontier baseline; user notices no quality drop | M |
| D4 | Measure real savings on real subagent traffic | `reports/p1-subagent.md` — the first *live* dollar number | S |

**Exit:** read-only subagent slice served local for 1 week, quality parity, measured savings.

---

## Workstream P1-E — Representativeness (make the economics generalize)

Phase 0's economics are single-user, Max-plan, workflow-heavy. To claim anything for a metered enterprise population, get a constrained baseline and real-codebase reliability.

| Step | Deliverable | Acceptance | Effort |
|---|---|---|---|
| E1 | **Constrained baseline:** capture a developer working under normal token limits *without* the local rung (a colleague, or self with workflows rationed) | ≥1 week of "before" traffic; traffic-mix contrast vs this window quantified | M (needs a participant) |
| E2 | **Real-codebase edit reliability:** run the registry-measurement harness on the office M4 Max / 35B MoE against real repos, not toy sandboxes | Measured `local-code / claude_code` reliability on 500-line-file edits — the number that replaces the 0.87 sketch | M (needs office hardware) |
| E3 | Reframe the value model with both stories sized: cost-reduction ($ on flowing traffic) AND capability-unlock (suppressed workflow demand released) | `reports/value-model.md` with the enterprise pitch, honestly bounded | S |

**Exit:** an economics story defensible for the target population, not just this corpus.

---

## Sequencing

1. **Ship now, in parallel:** P1-A (utility pinning) and P1-B (scanner precision) — both effectively risk-free, both immediately useful, and B de-risks the pin before it ever gates traffic.
2. **Then the lift:** P1-C (post-call path) — the main engineering work; unlocks D and Phase 2.
3. **Then the payoff:** P1-D (subagent pilot) — first live dollar number, needs C.
4. **In parallel, opportunistic:** P1-E — needs a second participant and office hardware; start whenever they're available.

## Phase 1 exit → Phase 2 gate

Phase 1 is done when: utility calls pinned live (A), scanner precision defensible (B), post-call path proven in shadow (C), a read-only subagent slice served live at quality parity with a measured saving (D). At that point the design's Phase-2 criteria (escalation rate <30%, user-retry ≤ frontier baseline, zero pin violations) are within reach on real user turns — which is where Phase 2 begins.

## Decisions needed from the user

- **D-1:** Approve crossing into live routing at all (Phase 1 acts on real traffic, even if only the safe slices).
- **D-2:** The mid-turn-secret-on-cloud-rung policy (§13.6) — block-and-surface (recommended) or finish-then-pin.
- **D-3:** A participant for the constrained baseline (E1) and access to the office M4 Max for E2 — or accept the economics stay single-user-bounded for now.
