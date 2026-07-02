# Deep-Research Vetting Report — Adaptive Routing Layer

**Date:** 2026-07-02 · **Method:** multi-agent research harness (108 agents): 5 search angles → 26 sources fetched → 129 claims extracted → top 25 adversarially verified by 3-vote panels (23 confirmed, 2 refuted, 0 unverified)
**Subject:** `adaptive-routing-layer-design.md` Draft v1 · **Outcome:** amendments folded into Draft v2 (see design doc §16 changelog)

---

## Verdict

The design is directionally aligned with mid-2026 best practice. Its most distinctive choice — sticky routing that never switches models mid-conversation — was independently converged on by GitHub Copilot's production Auto router, and GitHub's own harness benchmarking validates keying a capability registry on (model, harness) pairs. The heuristics-first architecture with a learned router only for the ambiguous middle is defensible against the research literature. The highest-impact gaps: no real-time health/availability signal layer, no leading quality/uncertainty estimator on local output, conflation of model selection with provider selection, and reimplementation risk against machinery LiteLLM already ships. All four are addressed in v2.

---

## Verified findings

### 1. Sticky, cache-boundary routing is validated by production practice — confidence: high, votes 3-0

GitHub Copilot's Auto model selection deliberately avoids switching models mid-conversation because breaking the prompt cache can cost more than the routing change saves; it routes only at natural cache boundaries (first turn, post-compaction). Directly corroborates the design's §2 core idea. Copilot's stickiness is per cache segment (until compaction) — *coarser* than the design's per-turn stickiness — and suggests escalation's cache loss should be an explicit cost term in the escalate-or-persist decision (adopted in v2 §5.5 step 5).

> "Switching models mid-conversation breaks that cache, which can cost more than the routing change saves... Auto avoids that by routing at natural cache boundaries." — GitHub blog, 2026-06-17

Sources: [Copilot context handling & model routing](https://github.blog/ai-and-ml/github-copilot/getting-more-from-each-token-how-copilot-improves-context-handling-and-model-routing/) · [auto model selection docs](https://docs.github.com/en/copilot/concepts/models/auto-model-selection)

### 2. GAP — no health/availability signal layer — confidence: high, votes 3-0 (×4 claims)

Copilot's production Auto combines two co-equal signal systems: a learned task-aware routing model (HyDRA — reasoning depth, code complexity, debugging difficulty, tool orchestration) AND a dynamic engine tracking real-time model availability, utilization, speed, error rates, and cost across 20+ models. "A model may be capable of handling a task, but that does not mean it is the best choice at that moment." Health-gated routing is also default in OpenRouter and available in LiteLLM. The v1 design had task-feature routing only. **Adopted in v2: health gate in policy layer 0 (§5.3), fed by LiteLLM health checks.**

Sources: Copilot blog/docs above · [HyDRA paper](https://arxiv.org/pdf/2605.17106) · [Copilot harness evaluation](https://github.blog/ai-and-ml/github-copilot/evaluating-performance-and-efficiency-of-the-github-copilot-agentic-harness-across-models-and-tasks/)

### 3. The (model, harness) registry key is validated — confidence: high, votes 3-0

With the same model and benchmark tasks, the Copilot harness matches vendor harnesses (Claude Code, Codex CLI) on task resolution (statistically insignificant 64–67% band) while consuming ~30–40% fewer input / ~20–25% fewer output tokens in most configurations, with notable per-pair exceptions. Independent corroboration: Harness-Bench shows the same model scoring up to 16 points differently across harnesses. Tool-dialect and efficiency must be measured per (model, harness) pair — exactly the design's §5.4 position.

Sources: Copilot harness evaluation above · [Harness-Bench](https://arxiv.org/abs/2605.27922)

### 4. GAP — model selection and provider selection should be decomposed — confidence: high, votes 3-0 (×3 claims)

OpenRouter treats routing as two independent decisions: which model answers, and which provider serves that model. Default provider selection is health-gated then price-weighted (providers with significant outages in the last 30s drop to the back; inverse-square price weighting), with automatic provider-level failover on 5xx/rate-limits within a model, separate from user-configured model-level fallbacks. The v1 ladder handled model-level decisions only. **Adopted in v2: rung/endpoint split in the registry (§5.4); endpoint failover delegated to LiteLLM.**

Sources: [OpenRouter model routing](https://openrouter.ai/blog/insights/model-routing/) · [provider selection](https://openrouter.ai/docs/guides/routing/provider-selection) · [model fallbacks](https://openrouter.ai/docs/guides/routing/model-fallbacks)

### 5. GAP — reimplementation risk vs LiteLLM — confidence: high, votes 3-0, 2-1, 3-0

LiteLLM already ships pluggable routing strategies (simple-shuffle default, latency-based, usage-based-v2, least-busy, cost-based, custom via `CustomRoutingStrategyBase`), ordered priority failover, within-group retry before cross-group fallback (flag-gated: `enable_weighted_failover`, simple-shuffle only — the 2-1 vote nuance), and pre-call feasibility checks (`enable_pre_call_checks=True`) filtering deployments whose context window is smaller than the request. Plus proactive background health checks (default 300s) and hierarchical budget enforcement (proxy/team/user/key/model levels). **Adopted in v2: ownership table (§8.5), `enable_pre_call_checks` in the config sketch.**

Sources: [LiteLLM routing](https://docs.litellm.ai/docs/routing) · [reliability](https://docs.litellm.ai/docs/proxy/reliability) · [health checks](https://docs.litellm.ai/docs/proxy/health_check_routing) · [budgets](https://docs.litellm.ai/docs/proxy/users)

### 6. GAP — no leading quality/uncertainty estimator — confidence: high, votes 3-0 (×2 claims)

The ETH cascade-routing paper (ICML 2025) unifies routing and cascading into a theoretically optimal strategy and identifies good quality estimators as THE critical factor for whether any model-selection paradigm improves cost-performance. The design's trip-wires are lagging failure detectors; best practice adds a leading confidence estimate on the cheap model's output (calibrated uncertainty, logprobs, or a verifier) so escalation fires before wasted actions accumulate. Caveats: optimality holds under the paper's formalization; evaluation is single-query, not agentic multi-turn; no published calibration recipe exists for tool-call outputs. **Adopted in v2: leading-confidence soft trip-wire, Phase 3, flywheel-calibrated (§5.5); new open question §13.5.**

Sources: [Cascade routing (Dekoninck et al.)](https://arxiv.org/abs/2410.10347) · [code](https://github.com/eth-sri/cascade-routing)

### 7. Heuristics-first is defensible against learned routers — confidence: high, votes 3-0 (×3 claims)

Under LLMRouterBench's unified evaluation (13 flagship LLMs, 10 datasets, ~392k instances), several learned routing methods — including OpenRouter's commercial NotDiamond-powered auto-router — fail to outperform the trivial always-use-the-best-model baseline; OpenRouter scored **−24.7%** relative to Best Single. Corroborated by the Routing Plateau paper (21 methods converge to near-best-single policies). Qualifications: platform-defined pool differences weaken the OpenRouter comparison; evaluation is primarily single-turn. **Adopted in v2: best-single-model baseline computed in Phase 0 as a hard Phase-3 gate (§10); risk row in §12.**

Sources: [LLMRouterBench](https://arxiv.org/html/2601.07206v1) · [OpenRouter auto-router](https://openrouter.ai/docs/guides/routing/routers/auto-router) · [Routing Plateau](https://arxiv.org/abs/2606.07587) · [RouterArena](https://arxiv.org/abs/2510.00202)

### 8. The research frontier frames routing as sequential RL — with no agentic-coding validation yet — confidence: high, votes 3-0 (×5 claims)

Router-R1 (NeurIPS'25) trains an LLM router with RL to interleave think/route actions; xRouter (Salesforce) frames routing as tool-calling trained end-to-end with a cost-aware reward. But Router-R1 is validated only on seven general/multi-hop QA benchmarks and xRouter is single-decision orchestration — neither establishes learned-routing gains on multi-turn agentic tool-use traffic. The design's flywheel plan is ahead of published evidence; xRouter's cost-aware reward (quality − λ·normalized cost) is a better training target than imitating heuristics. **Adopted in v2: training-target note (§5.7).**

Sources: [Router-R1](https://arxiv.org/abs/2506.09033) · [xRouter](https://arxiv.org/abs/2510.08439)

---

## Refuted claims (excluded by the adversarial panel, 0-3 each)

1. "Existing LLM routers typically perform single-round one-to-one mapping, which limits the design under review" — refuted; the framing doesn't limit a per-turn router.
2. "Cascade routing consistently outperforms pure routing and pure cascading by a large margin" — refuted; the optimality is theoretical, empirical margins are conditional on estimator accuracy.

## Caveats

- Copilot and OpenRouter findings rest on first-party vendor descriptions (blogs/docs) — appropriate for descriptive claims about internals, but self-favorable. GitHub's harness benchmark is first-party, partially corroborated independently.
- Essentially all learned-routing research evidence is from single-turn/QA workloads; there is **no published benchmark for learned routing on multi-turn agentic tool-use traffic**, so pro- and anti-learned-router evidence applies to this design's setting only by analogy.
- Cited LiteLLM behaviors are partly flag-gated non-defaults (`enable_weighted_failover`, `enable_pre_call_checks`).
- **No verified claims survived** on several sub-questions — the report's silence there is absence of evidence, not evidence of absence: privacy/secret handling in routing layers, SLM-first agentic designs, speculative/parallel rung racing, prompt compression/context carving, per-token/segment routing, local-model logprob confidence recipes, Portkey/Kong/Cloudflare hardening specifics.
- Key sources were 1 day to 3 weeks old at research time; this space moves fast.

## Open questions carried forward

1. Do learned routers beat well-tuned heuristics on multi-turn agentic coding traffic? No public benchmark exists — building one from captured flywheel traces (with a best-single-model baseline per LLMRouterBench methodology) is both an open research gap and a prerequisite for the design's Phase 3.
2. Quantified tradeoff of scrubbed-transcript replay on escalation vs Copilot's compaction-boundary-only re-routing — when does cache loss + rebuild exceed the value of the stronger model finishing the turn?
3. Can local-model confidence signals be calibrated for tool-call outputs at interactive latency? (Design §13.5.)
4. How do production gateways implement privacy-driven routing constraints? No verified evidence surfaced — the design's one-way pin is unvalidated against (and possibly ahead of) industry practice.
