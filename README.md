# adrl — Adaptive Routing Layer

A routing layer for LLM coding agents that sits between coding harnesses (Claude Code / Codex CLI) and LiteLLM, deciding per user turn whether work goes to a local model (llama.cpp / MLX), cheap cloud, or a frontier model — with sticky per-turn routing to protect prompt caches, deterministic escalation trip-wires, a one-way privacy pin, and a flywheel of logged outcomes for tuning.

## Documents

- [Design doc](docs/adaptive-routing-layer-design.md) — architecture, components, rollout plan (Draft v2)
- [Scenario walkthroughs](docs/adaptive-routing-scenarios.md) — 15 wire-level traces through the layer (Draft v2)
- [Deep-research vetting report](docs/deep-research-vetting.md) — verified findings, sources, and open questions behind the v2 amendments
- [Phase 0 plan](docs/phase0-plan.md) — three workstreams with acceptance criteria; reports land in `reports/`

## Decks

- [Handover deck](decks/adaptive-routing-layer-handover.pptx) (47 slides) — teaches the design itself: every component, all 15 scenarios as wire-level traces. For the team implementing/operating it; self-study density.
- [Overview deck](decks/adrl-project-overview.pptx) (16 slides) — tells the project story with measured Phase 0 evidence. For leadership and engineers new to the project; meeting-ready.

Draft v2 incorporates an internal consistency review and deep-research vetting against production routers (GitHub Copilot Auto, OpenRouter, LiteLLM) and the routing literature; see the design doc's changelog (§16).
