# Phase 0 Exit Report — Adaptive Routing Layer

**Date:** 2026-07-07 · **Status:** Phase 0 (shadow / evidence-gathering) complete on all buildable criteria · **Recommendation:** proceed to Phase 1

Phase 0 asked one question — *is this worth building, and are the design's assumptions true?* — and required it be answered without any user-visible risk. This report marks every §10 exit criterion pass/fail against measured evidence, states the economics verdict, and lists what carries into Phase 1.

---

## 1. Exit criteria (design §10)

| Criterion | Target | Result | Evidence |
|---|---|---|---|
| Discriminator/features/policy run in shadow on ≥200 real turns | ≥200 | **210 organic user-turns captured** (gate crossed); full pipeline runs on them | `data/captures/`, this report §2 |
| Predicted local share & escalation rate look sane | qualitative | **81.6% local predicted**, gates dominate, learned-router band 8.4% | `policy-replay.md` |
| Router overhead p50 | < 20ms | **discriminator p50 0.3µs; policy p50 0.5µs** (4 orders under budget) | `discriminator-eval.md`, `policy-replay.md` |
| Best-single-model baseline computed | required | **$2,941** (opus-4-8-equivalent, 5.5-week corpus) | `corpus-metrics.md` |
| `metadata.user_id` per-session? (B4) | resolve | **Resolved: per-session `session_id`** — keying solved, hash fallback unneeded | `assumption-user-id.md` |
| Tool-call IDs need re-minting cross-provider? (B5) | resolve | **Resolved: no — internal consistency only** — ID-map machinery dropped | `assumption-tool-ids.md` |
| Historical replay against candidate policy | required | 754 turns replayed, all ground-truth checks pass | `policy-replay.md` |
| ≥3 real traces per scenario, or explicit verdict | required | 15/15 accounted for (10 corpus-validated, 2 episode, S5 falsified-for-model, S7 confirmed, S11 falsified) | `scenario-validation.md`, `scenario-local-rung.md` |
| Discriminator drift canary | required | **25/25 fixtures pass**; fails on any label drift | `tests/test_discriminator_canary.py` |

**All buildable Phase 0 exit criteria are met.** The only criterion needing calendar time rather than code — accumulating ≥200 live turns — is also now satisfied (210).

---

## 2. The business case (is it worth building?)

Measured on 5.5 weeks of real traffic (3,127 transcripts, 5,027 turns):

- **Spend in window:** ~$3,313 observed; **$2,941** re-priced at the opus-4-8 best-single-model baseline. This is the number a learned router must beat.
- **Cache-hit ratio: 99.3%** of prompt tokens — empirical vindication of the never-switch-mid-turn stance; that cache is the asset a naive router would destroy.
- **The headline, and a course-correction:** easy user turns are **45.5% of turns but only 3.7% of spend ($108)**. Routing them local is nearly free money but small money. The dollars are in **subagent traffic — 75% of turns, the bulk of token volume.** Phase 1 was re-prioritized accordingly (design §10): the subagent-local pilot (S13) is promoted alongside utility-call pinning.

**Verdict: worth building** — but the value thesis shifted from "route easy user turns cheap" to "delegate the agent's bounded subagent work to a local model," which is exactly the hybrid pattern the design already described and the industry's SLM-for-agents thesis.

---

## 3. What the evidence changed in the design

Phase 0 was cheap-falsification: five design assumptions met reality, and **every correction simplified the design or de-risked it** — none forced a redesign.

1. **Session keying (B4)** — a per-session id exists on the wire; the hashing fallback and its failure modes are gone.
2. **Tool-ID re-minting (B5)** — not needed; the persistent `tool_id_map` and per-request translation tax are removed from the escalation rebuild.
3. **New `passthrough` request kind** — `count_tokens` is ~72% of raw wire traffic and was missing from the taxonomy entirely; now handled by path alone.
4. **Sidecar fingerprint** — utility calls run on Opus/Sonnet with tiny budgets, not Haiku-class; fingerprint by shape, not model name.
5. **Local-rung reliability (S5)** — `qwen2.5:7b-instruct` scores ≈1.0 on Claude Code's edit dialect, not the assumed 0.87. The trip-wire fires less, the cascade math favours more local-first attempts. The (model, harness) registry did its job: measured, didn't assume.

Two findings also *raised* priorities:
- **Secret-scanner precision is the highest-leverage knob** — the 7 secret-flagged sessions hold 73% of user turns, making the pin-context collision (§5.8) the dominant case, not an edge one. Tuning the scanner's false-positive rate (§13.4) is now Phase-1 priority-one.
- **The learned router would own only 8.4% of decisions** — confirming the design's "dumb rules decide most things" bet, and setting a low ceiling on Phase-3's upside (which must still beat the $2,941 baseline to ship).

---

## 4. What was built (all committed, `arunmenon/adrl`)

- **Design** at v2.1: design doc + 15 scenarios + vetting report, twice-reviewed, wire-corrected.
- **Miner** (`src/miner/`): transcript corpus → per-turn dataset + all economics/scenario reports.
- **Capture proxy** (`src/proxy/`): transparent wire capture, upstream-configurable.
- **Router** (`src/router/`): discriminator + feature extractor + three-layer policy, evidence-grounded, eval-gated, with the B5 experiment.
- **Simulator** (`src/simulator/`): scenario + LLM-driven episode generator; local-rung fallback inducer.
- **Execution layer**: LiteLLM + ollama local rung, Anthropic⇄OpenAI translation, infra fallback — the first production-stack piece.
- **Canary** (`tests/`): 24 scrubbed fixtures + drift test.

Cost of the entire Phase 0 evidence effort: **$4.23 of simulator budget** plus a few cents of B5 API calls. The rest was offline analysis of traffic that already existed.

---

## 5. Carried into Phase 1

**Ready:**
- Utility-call pinning (near-zero risk) + **subagent-local pilot** (the re-prioritized dollar lever).
- The execution stack (LiteLLM + local rung) exists and is proven end-to-end.
- Kill switch (unset `ANTHROPIC_BASE_URL`), canary, and provenance separation all in place.

**Priority-one tuning (Phase 0 surfaced, Phase 1 owns):**
- **Secret-scanner precision** — dominant-case knob; tune false-positive rate against the flagged sessions before the pin gates live traffic.

**Deferred / needs other hardware:**
- Production `local-code` reliability numbers come from the office M4 Max / 64GB running the 35B MoE, not this M1 Pro's 7B — run the same registry-measurement harness there.
- The router's **post-call path** (trip-wire evaluation, escalation rebuild, fallback state-sync) is designed but unbuilt — it's Phase 2 work, gated behind Phase 1's exit.
- Phase 3 learned router: only if it beats heuristics **and** the $2,941 baseline on replay.

---

## 6. One-line verdict

Every assumption the design's economics rest on was checked against reality and held; the one course-correction (subagents, not easy turns, are the dollar lever) is already in the design; the remaining risk is concentrated in local-model *quality* (a Phase-2 question, answered in shadow where being wrong is free) and secret-scanner *precision* (a tuning knob, not an architecture question). **Proceed to Phase 1.**
