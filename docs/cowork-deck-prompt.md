# Prompt for Claude Cowork — Presentation deck on the Adaptive Routing Layer project

Copy everything below the line into Claude Cowork.

---

Build presentation slide content for a project called the **Adaptive Routing Layer (adrl)**. I am the author of the project. Produce a deck of roughly 12–16 slides: a title slide, an executive summary, then sections for the problem, the design, how it was validated, early results with real data, and the roadmap. For each slide give me a headline, 3–5 tight bullet points, and a speaker note of 2–3 sentences. Suggest one simple visual per slide (diagram, table, or big number) where it helps. The audience is technical: engineering leadership and senior engineers who know what LLMs and coding agents are but have not seen this project. Tone: confident, evidence-first, no hype. Everything you need is below — do not invent numbers beyond what is given.

## 1. The problem

Coding agents (Claude Code, Codex CLI) send every request to whatever model a static config names — usually a frontier model. But agent traffic is extremely heterogeneous: one user instruction ("fix the failing test") explodes into a burst of HTTP requests — the model reads files, runs tests, applies edits, each round-trip a separate request. Measured on my own traffic, 73% of requests are mid-turn tool continuations, ~11% are harness housekeeping (session titles, compaction), and only ~16% are genuine new user instructions. Sending all of it to a frontier model wastes money on work a local model could do; naively routing per-request breaks the agent entirely.

## 2. The core idea

**Route turns, not HTTP requests.** The routing layer sits between the coding harness and the model gateway (LiteLLM) and decides once per user turn which rung of a model ladder handles it: local model (llama.cpp/MLX, ~$0) → cheap cloud (Haiku-class) → frontier (Opus-class). Everything mid-turn inherits the turn's route via a sub-millisecond sticky lookup. Two and only two legal moments to change models: a new user turn (full policy decision) and a trip-wire firing at an action boundary (controlled escalation). Rationale: mid-turn switching destroys prompt caches on both sides (measured cache-hit ratio on my traffic: 99.3% of prompt tokens — that asset dies on a switch), confuses the new model with another model's half-finished reasoning, and breaks provider protocol (thinking-block signatures don't transfer).

## 3. The architecture (six components)

1. **Call-type discriminator** — labels every request user_turn / continuation / utility / subagent with mechanical rules only, microseconds, no ML. The expensive machinery runs on only ~15% of requests.
2. **Three-layer policy engine, cheapest first** — Layer 0: hard gates that nothing overrides (privacy pin, context feasibility, endpoint health, escalation hysteresis). Layer 1: hand-tuned heuristics on intent/scope/error-history — decides ~70% of turns. Layer 2: a learned router for only the ambiguous middle, trained on captured outcomes (later phase).
3. **Escalation controller** — deterministic trip-wires, no LLM judge: edit fails to apply twice, malformed tool call twice, identical call three times, no progress over six actions, user interrupt (the strongest signal). Escalation rebuilds the transcript for the higher rung (strip foreign reasoning, re-map tool IDs, prepend a 3-line failure note) and hysteresis keeps the episode on the higher rung.
4. **Capability registry** — measured (not benchmark) profiles keyed by (model, harness) pairs, because tool-call reliability is dialect-specific: the same local model scores 0.93 under Codex's patch-based edits but 0.87 under Claude Code's exact-string edits. Rungs are model classes; endpoints (llama.cpp vs MLX serving the same model) are handled by LiteLLM failover.
5. **Privacy pin** — if a secret (API key, connection string) appears in any tool result, the session is pinned to the local model permanently, enforced structurally at gate layer 0. No routing layer on the market documents an equivalent.
6. **Flywheel** — every routed turn logs features, decision, outcome; feeds dashboards, threshold tuning, and eventually the learned router.

## 4. Validation — two review passes before any code

**Internal consistency review** found five design flaws that were fixed on paper (cheap) instead of in production (expensive): a deadlock between the privacy pin and context limits; trip-wires accidentally disarmed on pinned sessions; the escalation transcript-scrub incorrectly modeled as one-shot when it must run on every subsequent request; an unresolved two-vs-three-rung launch decision; and two load-bearing assumptions flagged for empirical testing.

**Deep-research vetting** (multi-agent research harness: 108 agents, 26 sources, 129 claims extracted, 25 adversarially verified by 3-vote panels — 23 confirmed, 2 refuted) checked the design against mid-2026 state of the art:

- **Validated:** GitHub Copilot's production Auto router independently converged on the same core choice — never switch models mid-conversation, route only at cache boundaries, because cache-break costs exceed routing savings. GitHub's own benchmarking validates keying capability on (model, harness) pairs — the same model scores up to 16 points differently across harnesses.
- **Validated:** heuristics-first is the right default. Under unified evaluation (LLMRouterBench), several learned routers — including a commercial one — score *below* the trivial "always use the best model" baseline (the commercial router: −24.7%). No published learned router has demonstrated gains on multi-turn agentic traffic.
- **Gaps found and fixed in v2:** no real-time health/availability gate (Copilot treats health as co-equal with task routing); no leading confidence signal on local output (the cascade-routing literature, ICML 2025, identifies estimator quality as THE decisive factor); model selection conflated with endpoint selection (OpenRouter decomposes them); reimplementation risk vs what LiteLLM already ships.

## 5. Phase 0 — proving it's worth building, with real data

Phase 0 is shadow mode: run the decision logic on real traffic, log what it *would* do, change nothing. Discovery that accelerated everything: the harness already writes full session transcripts to local disk — 3,127 files, 927MB, ~5.5 weeks of real coding-agent history (and it garbage-collects after ~30 days, so day one was an emergency snapshot).

Built and shipped: a streaming transcript miner (parser → turn assembler → discriminator-twin labeler → parquet dataset), scenario fingerprint matchers, a metrics report, a secrets scanner, and a transparent wire-capture proxy (one env var routes live sessions through it; kill switch is unsetting the var).

**Results on 5,027 assembled turns (zero parse errors, ~5s full-corpus run):**

- **The traffic model held.** Predicted vs measured request shares: continuations 70–90% predicted / 73.3% measured; housekeeping 5–15% / 10.7%; user turns 5–15% / 16.1%.
- **10 of 15 design scenarios validated with real traces** from the corpus (edit-failure trip-wires, interrupt-then-rephrase, huge-context easy asks, compaction, subagent spawns, parallel tool calls...). One scenario falsified (Codex dialect — no Codex usage here), three correctly deferred to wire capture / simulator.
- **The privacy scenario is real:** 7 sessions contained credentials in tool results (212 env-var secrets, 169 connection strings) and would have been pinned.
- **One "invariant violation" investigated and cleared:** a single turn showed two models — it was the provider-side Fable 5 → Opus 4.8 fallback, the one *legal* kind of mid-turn switch (the provider owns protocol consistency).
- **Cache-hit ratio: 99.3%** of prompt tokens served from cache — empirical backing for the no-mid-turn-switch stance.
- **Economics baseline:** observed spend ≈ $3,313 in the window; re-priced entirely at Opus 4.8 (the best-single-model baseline any learned router must beat): **$2,941**.

## 6. The strategic finding (best slide of the deck)

**Turn-count share ≠ dollar share.** A naive filter marks 45.5% of user turns as local-routable — but they're only **$108 of $2,941 (3.7%)** of baseline spend. The volume — and the money — is in *subagent* traffic: 3,773 subagent turns with 27,116 tool continuations, versus 1,254 main-session turns. Implication: the biggest lever isn't routing easy user turns to a local model; it's the hybrid pattern where the frontier model does the thinking and delegates bounded grunt-work subtasks (small context, clear goal, read-only) to a free local model. This inverts the project's Phase 1 priorities and matches the industry's "small-language-models-for-agents" thesis.

## 7. Roadmap

- **Now (Phase 0, ~2 weeks):** live wire capture running on daily traffic; next: test two assumptions empirically (is the harness session ID per-session or per-user; do tool-call IDs need re-minting across providers), extract wire fingerprints, run the shadow discriminator on ≥200 live turns at <20ms overhead.
- **Phase 1:** pin harness housekeeping calls to a local model — near-zero risk, instant savings; possibly prioritize subagent-local routing given the economics finding.
- **Phase 2:** live routing of user turns with trip-wires armed and a one-line kill switch. Exit gates: escalation rate <30%, user-retry rate no worse than frontier baseline, privacy-pin violations exactly zero.
- **Phase 3:** learned router for the ambiguous middle — only ships if it beats both the hand-tuned heuristics AND the $2,941-style best-single-model baseline on replayed traffic, a bar most published learned routers fail.

## 8. Facts and figures cheat sheet (use exactly these)

- Corpus: 3,127 transcript files, 927MB, ~5.5 weeks, single user, 3,060 parsed, 178,681 records, 0 bad lines, 5,027 turns.
- Traffic: 73.3% continuations, 10.7% utility, 16.1% user turns (main sessions, transcript-visible).
- Subagents: 3,773 turns / 27,116 continuations vs 1,254 main-session turns.
- Cache-hit ratio: 99.3%. Spend: ~$3,313 observed, $2,941 Opus-4.8-equivalent baseline.
- Easy-turn filter: 45.5% of user turns, 3.7% of baseline spend ($108).
- Privacy: 7 sessions would have pinned; 212 + 169 secret pattern hits.
- Scenarios: 10/15 validated, 1 falsified, 3 deferred, 1 invariant cleared after investigation.
- Vetting: 108 research agents, 26 sources, 129 claims, 25 verified (23 confirmed / 2 refuted).
- Design docs: ~1,200 lines across a design doc (16 sections) and 15 wire-level scenario walkthroughs, both at v2 after two review passes.
- Prices used (per MTok): Opus 4.8 $5 in / $25 out; Haiku 4.5 $1/$5; Fable 5 $10/$50; cache reads 0.1×, cache writes 1.25×.
- Repo: private GitHub repo `arunmenon/adrl` — design docs, vetting report, Phase 0 plan, miner (src/miner), capture proxy (src/proxy), reports.

## 9. Framing guidance

- The through-line: **design → adversarial validation → cheap empirical falsification → build**. Every design claim was either verified against production practice (Copilot, OpenRouter), the research literature, or my own 927MB of traffic before implementation money was spent.
- Be honest about limitations on a dedicated slide: single-user corpus skewed by workflow-heavy usage; transcript counts are lower bounds (sidecar calls invisible until wire capture); learned-router evidence from the literature is single-turn, not agentic; savings projections are ceilings, not promises.
- The memorable one-liners: "Route turns, not requests." "A nurse triages, a junior doctor treats under supervision, and there's a protocol for paging the specialist — including handing over the chart." "45% of the turns, 4% of the money." "The best routing decision is often declining to use the cheap model."
