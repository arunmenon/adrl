# adrl — Adaptive Routing Layer

A routing layer for LLM coding agents that sits between coding harnesses (Claude Code / Codex CLI) and LiteLLM, deciding per user turn whether work goes to a local model (llama.cpp / MLX), cheap cloud, or a frontier model — with sticky per-turn routing to protect prompt caches, deterministic escalation trip-wires, a one-way privacy pin, and a flywheel of logged outcomes for tuning.

## Documents

- [Design doc](docs/adaptive-routing-layer-design.md) — architecture, components, rollout plan (Draft v2)
- [Scenario walkthroughs](docs/adaptive-routing-scenarios.md) — 15 wire-level traces through the layer (Draft v2)
- [Deep-research vetting report](docs/deep-research-vetting.md) — verified findings, sources, and open questions behind the v2 amendments
- [Phase 0 plan](docs/phase0-plan.md) — three workstreams with acceptance criteria; reports land in `reports/`

## Setup

Requires [uv](https://docs.astral.sh/uv/) (no system Python assumptions; pure wheels only — no compiler needed):

```bash
uv venv -p 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

All data (transcripts, wire captures, derived datasets) lives under `data/`, which is
**gitignored — it contains real secrets and never leaves the machine** (plan decision D5).

## The transcript miner (workstream A)

Mines the local Claude Code transcript archive (`~/.claude/projects/`) into a per-turn
dataset and the Phase 0 reports. Run in order:

```bash
./tools/snapshot_corpus.sh                     # A0 — snapshot transcripts into data/corpus/
                                               #      (re-run weekly: the harness GCs after ~30 days)
PYTHONPATH=src .venv/bin/python -m miner.extract    # A1-A4 -> data/turns.parquet + parse stats
PYTHONPATH=src .venv/bin/python -m miner.secrets    # A8 -> data/secrets-scan.json (privacy-pin evidence)
PYTHONPATH=src .venv/bin/python -m miner.scenarios  # A5 -> reports/scenario-validation.md + raw traces in data/
PYTHONPATH=src .venv/bin/python -m miner.report     # A6/A7 -> reports/corpus-metrics.md (incl. cost baseline)
```

Committed outputs (aggregates only, no payloads) land in `reports/`.

## The wire-capture proxy (workstream B)

A transparent proxy between the coding harness and `api.anthropic.com` that logs every
request/response verbatim (auth headers redacted, bodies raw) to `data/captures/`:

```bash
./tools/run_proxy.sh          # starts detached on :4000; pid in data/proxy.pid, log in data/proxy.log
```

Then route any session through it — this is the only per-session step:

```bash
ANTHROPIC_BASE_URL=http://localhost:4000 claude          # new session
ANTHROPIC_BASE_URL=http://localhost:4000 claude --resume # resumed session (works the same)
```

Notes for operators:

- **Kill switch:** launch without the env var — sessions go direct, nothing breaks.
  Forgetting the prefix just means that session isn't captured.
- **Stop the proxy:** `kill $(cat data/proxy.pid)`. It does not auto-start on reboot;
  re-run `./tools/run_proxy.sh`.
- The proxy changes nothing about the traffic: SSE streams are relayed live, capture I/O
  happens after response bytes are delivered, and upstream failures surface as 502s with
  the error logged.
- `PORT=4001 ./tools/run_proxy.sh` to run on a different port.

## Decks

- [Handover deck](decks/adaptive-routing-layer-handover.pptx) (47 slides) — teaches the design itself: every component, all 15 scenarios as wire-level traces. For the team implementing/operating it; self-study density.
- [Overview deck](decks/adrl-project-overview.pptx) (16 slides) — tells the project story with measured Phase 0 evidence. For leadership and engineers new to the project; meeting-ready.

Draft v2 incorporates an internal consistency review and deep-research vetting against production routers (GitHub Copilot Auto, OpenRouter, LiteLLM) and the routing literature; see the design doc's changelog (§16).

## The scenario simulator (workstream C)

Generates realistic agent traffic through the capture proxy: builds a throwaway
project with genuinely planted bugs (failing test, disabled rate limiter, badly
named variable), picks a scenario with a **randomized prompt phrasing**, and runs
a headless session inside it — across model families:

```bash
PYTHONPATH=src .venv/bin/python -m simulator.run_session --runs 4          # random scenarios, model rotation default/opus/sonnet/haiku
PYTHONPATH=src .venv/bin/python -m simulator.run_session --scenario fix_test --model haiku
```

Scenarios: `explain` `rename` `fix_test` `investigate` `commit_msg` `feature` `refactor`
(mapped to design scenarios S1-S10). Sessions run with **scoped tool permissions**
(read/edit/pytest/git only — no sandbox drop). Hard budget cap: $25 total (decision D2),
enforced from `data/sim-ledger.jsonl`; every run logs its `session_id` there, which is
the provenance key separating synthetic from organic captures.
