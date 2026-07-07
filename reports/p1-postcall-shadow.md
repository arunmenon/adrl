# P1 post-call shadow — escalation controller (offline, no live routing)

Wires C1 `TripwireState` + C2 `rebuild_for_escalation` + C3 `SessionStore` over the historical corpus (§5.5 post-call flow). Cascade eligibility is the real `router.policy` decision; the escalation rate is measured **of cascaded turns** (§11). No live traffic is touched.

## Headline

- Turns replayed: **6111** (3403 cascaded / trip-wires armed) — assembled from the raw corpus JSONL (the action-level `(response, tool_results)` stream the wires need, which `turns.parquet` aggregates away; count differs from the parquet's parentUuid segmentation, but the rate is a ratio and robust to that).
- Would-have-escalated (cascaded): **55**
- **Predicted escalation rate: 1.62%** of cascaded turns (Phase-2 gate §10: < 30%)
- Pin-blocked escalations (target = USER, never cloud): **4**
- Cloud-bound escalations (target = higher rung): **51**

### Gate: PASS (1.62% < 30%)

## Trip-wire TYPE distribution (fired, cascaded)

Type drives the flywheel split (§5.7/B5): DIALECT trains the capability registry, everything else the difficulty/router model.

| Type | Trains | n |
|---|---|---|
| quality | router | 55 |

Dialect escalations (train registry): **0** · difficulty/cost/quality (train router): **55**.

## By wire

| Wire | n |
|---|---|
| user_interrupt | 55 |

## Cascade band mix (denominator)

| Policy layer | n |
|---|---|
| middle_default | 2033 |
| gate:privacy | 1099 |
| heuristic | 271 |

Turns the router would NOT cascade (trip-wires never armed — excluded from the rate):

| Policy layer | n |
|---|---|
| gate:pin_context_conflict | 2082 |
| gate:feasibility | 613 |
| heuristic | 11 |
| heuristic:retry_signal | 2 |

`gate:pin_context_conflict` (§5.8): a pinned session outgrew local context — no legal rung, surfaced to the user rather than cascaded.

## Escalation transcript rebuild (C2)

- Rebuilt + validated on every fired turn: **55 OK**, 0 malformed.
- Validation: thinking stripped, tool_use/tool_result IDs paired (B5 internal-pairing), no empty messages.

## Privacy pin

- Pinned sessions expected (A8 scan): **5**; seen in replay: **5** — PASS.
- Every escalation inside a pinned session targets the USER, never a cloud rung (4 such events). A pin is one-way (§5.6).

## Decision latency

- Per-turn route + full trip-wire drive: p50 **3.5us**, p99 **475.3us** (§11 budget: <5ms sticky, <30ms decision path).
- SessionState.strikes shape matches TripwireState.strikes: PASS.

## Cost-wire sensitivity (labeled, EXCLUDED from headline)

The `turn_budget` wire is a *cost* guard sized to 2x median tokens **of the local model**. Replayed frontier token counts measure frontier verbosity, not local runaway, so it is inert in the headline (budget = None).
- If a 2x-median-token budget (>246 output tok/turn) were applied to these frontier turns, **1012** additional cascaded turns would trip the cost wire — a frontier-verbosity artifact, not a local failure signal.

## Caveats (why this is a lower bound)

- Corpus is 100% Claude Code **frontier** traffic. We ask what the wires would catch had these trajectories come from the **local** rung; the frontier model rarely fails mechanically, so mechanical wires fire rarely. A local model would fail *more* — this rate is a **lower bound**.
- `parse_schema` reads 0 by construction: S5 (malformed local tool calls) is FALSIFIED for this corpus (`miner.scenarios`) — no local traffic exists to trip it. It needs workstream C (Ollama traffic via the capture proxy).
- `edit_apply` requires `is_error` truthy (not just the marker string), so doc-mentions of the marker do not inflate the count (C1 design note).

**Verdict: PASS** — escalation gate PASS, rebuild PASS, pin coverage PASS.

## Adversarial review + fixes (2026-07-07)

Two adversarial reviewers ran against the build. Verdict FIX_NEEDED; contract checks
otherwise passed (no ID re-minting confirmed byte-for-byte; pin never escalates to cloud;
thresholds match §5.5; 96/96 tests green). Fixes applied and re-verified:

- **[major] hung-local fail-open gap** (capture_proxy.py): the shared client had no read
  timeout, so a *hung-but-listening* LiteLLM (accepts TCP, never answers) would hang forever
  instead of falling back. Fixed: the local attempt now uses a bounded 45s timeout so a stall
  raises → Anthropic fallback. **Live-verified**: dead upstream → `local_fallback=True`, served
  by Anthropic (no 502); healthy upstream → served by local-small.
- **[major] concurrent-session bleed** (shadow_postcall.py): the parser stamps subagent
  transcripts with the parent session id, so keying trip-wire state on session_id alone merged
  independent trajectories. Fixed: group by `(session_id, source_path)`. Escalation rate moved
  0.63% → **1.62%** (more accurate — merging was under-counting), still far under the 30% gate.
- **[minor] 4xx-on-local fallback**: a local 4xx now falls open too (Anthropic gets the original body).
- **[minor] load_pins shape guard** added.

Deferred (non-blocking, not yet live): `tripwires`/`state` transitively import `miner.parser`
(pyarrow) — fine for offline shadow, but `canonical_call` + markers should move to a
dependency-light module before C is wired into the live proxy hot path (Phase 2). Interrupt
during-tool detection and `validate_rebuilt` string-content strictness are lower-bound-safe.
