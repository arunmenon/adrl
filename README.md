# adrl — Adaptive Routing Layer

A routing layer for LLM coding agents that sits between coding harnesses (Claude Code / Codex CLI) and LiteLLM, deciding per user turn whether work goes to a local model (llama.cpp / MLX), cheap cloud, or a frontier model — with sticky per-turn routing to protect prompt caches, deterministic escalation trip-wires, a one-way privacy pin, and a flywheel of logged outcomes for tuning.

## Status (2026-07-12)

**Phase 0 is complete; later phases have not passed their exit gates.** The repository
now contains the live routing/cascade path, privacy and health gates, transaction memory,
retrieval shadowing, deterministic verification, and an isolated counterfactual runner.
The read-only subagent pilot, representative production-model evaluation, sufficient
verified labels, and a trained Layer-2 router remain open. See
`reports/phase0-exit.md`, `docs/phase1-plan.md`, and `reports/retrieval-shadow.md` for
the measured gates rather than treating component presence as rollout completion.

## Repo map

```
docs/        design doc, 15 scenarios, vetting report, Phase 0 plan, deck prompt
src/
  miner/     workstream A — transcript corpus -> turns.parquet + reports
  proxy/     workstream B — wire-capture proxy (ANTHROPIC_BASE_URL target)
  simulator/ workstream C — scenario & episode traffic generator
tools/       snapshot_corpus.sh, run_proxy.sh
reports/     committed evidence (aggregates only, no payloads)
decks/       presentation decks
data/        gitignored — corpus snapshot, wire captures, datasets, ledgers (real secrets live here)
```

## Documents

- **[Frozen ADR taxonomy and decision index](docs/adr-index.md)** - canonical architecture buckets, permanent decision IDs, maturity, and change-control rules (Taxonomy v1.0)
- [Design doc](docs/adaptive-routing-layer-design.md) — architecture, components, rollout plan (Draft v2)
- [Scenario walkthroughs](docs/adaptive-routing-scenarios.md) — 15 wire-level traces through the layer (Draft v2)
- [Deep-research vetting report](docs/deep-research-vetting.md) — verified findings, sources, and open questions behind the v2 amendments
- [Phase 0 plan](docs/phase0-plan.md) — three workstreams with acceptance criteria; reports land in `reports/`
- [Phase 1 plan](docs/phase1-plan.md) — utility pinning, secret-scanner tuning, the post-call path, subagent pilot, representativeness

Draft v2 incorporates an internal consistency review and deep-research vetting against production routers (GitHub Copilot Auto, OpenRouter, LiteLLM) and the routing literature; see the design doc's changelog (§16).

## Reports (evidence produced so far)

- [Corpus metrics](reports/corpus-metrics.md) — traffic shares vs design §4, trip-wire frequencies, per-intent medians, token economics, and the opus-4-8 best-single-model baseline
- [Scenario validation](reports/scenario-validation.md) — S1-S15 match counts and verdicts against the ≥3-traces bar, incl. the investigated S15b finding
- [Session-keying assumption (B4)](reports/assumption-user-id.md) — answered: `metadata.user_id` carries a per-session `session_id`
- [Tool-ID assumption (B5)](reports/assumption-tool-ids.md) — answered: IDs need internal consistency, not re-minting
- [Local-rung scenarios (S5/S7)](reports/scenario-local-rung.md) — S5 falsified for this model (~1.0 reliability), S7 fallback confirmed
- **[Phase 0 exit report](reports/phase0-exit.md)** — every §10 criterion pass/fail, the business case, and the Phase-1 handoff
- **[Learning-readiness scorecard](reports/learning-readiness.md)** - transparent ADR evidence index, organic-label counts, statistical label audit, and hard graduation blockers
- [Readiness scoring contract](docs/readiness-scoring.md) - frozen formula, `40.5` v1 baseline, append-only milestone history, and change-control rules

## Setup

Requires [uv](https://docs.astral.sh/uv/) (no system Python assumptions; pure wheels only — no compiler needed):

```bash
uv venv -p 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

All data (transcripts, wire captures, derived datasets) lives under `data/`, which is
**gitignored — it contains real secrets and never leaves the machine** (plan decision D5).

## Deterministic verification for organic routes

Organic verifier labels use an explicit two-step job. Obtain the exact `route_id` from
the routing capture or harness callback; never infer it from the nearest session or
timestamp. Before the first repository edit, provide an opaque task ID, the model that
actually served the turn, and a repository-specific JSON plan:

```json
{
  "plan_version": "targeted-tests-v1",
  "command_checks": [
    {
      "name": "targeted-tests",
      "argv": [".venv/bin/python", "-m", "pytest", "tests/test_target.py", "-q"],
      "timeout_s": 300
    }
  ],
  "require_changes": true,
  "forbidden_path_globs": [".git/**", "data/**"],
  "max_changed_files": 20
}
```

The plan is argv-only and must contain a required functional/content check; a file diff
alone cannot become a success label. Start from a clean Git workspace, then finish after
the task's edits and tool calls complete:

```bash
PYTHONPATH=src .venv/bin/python -m router.live_verification begin \
  --route-id ROUTE_ID --task-id opaque-task-001 --workspace /path/to/repo \
  --plan /path/to/verification-plan.json --served-rung local-code \
  --model local/model-id --harness codex

PYTHONPATH=src .venv/bin/python -m router.live_verification finish \
  --job-id JOB_ID
```

`finish` enriches the immutable route ledger and appends scrubbed provenance to
`data/verified-evidence.jsonl`. Raw prompts and repository paths are not copied into
that evidence stream. The local job file retains the workspace and plan at mode `0600`
so the exact baseline can be verified and retried idempotently.

## Same-snapshot counterfactual pairs

`router.counterfactual.run_counterfactual` clones one clean Git commit into an isolated
workspace per candidate and runs the same deterministic plan for every rung. Its JSONL
records now implement `counterfactual-evidence-v1`: pair ID, scrubbed predecision
features, repository/snapshot/plan hashes, exact model and harness, cost/latency, and
verifier result. Set `CounterfactualTask.source="organic"` only when the task came from
organic traffic; simulator-generated tasks remain `synthetic` and cannot satisfy the
representative-pair gate.

Persist the canonical evidence snapshot and graduation gates after a milestone:

```bash
PYTHONPATH=src .venv/bin/python -m router.learning_readiness --persist
.venv/bin/python tools/check_readiness_score.py
```

The generator writes matching Markdown and JSON reports and appends one history
entry per distinct evidence state. The architecture index is comparable only to
the frozen `40.5` baseline under the exact v1 contract hash; production readiness
remains a separate hard-gate verdict.

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

### Episode mode (LLM-driven multi-turn scenarios)

Single shots can't exercise episode-level design behavior (S15a boundaries, S6 retry
signals, hysteresis). Episode mode fixes that: step 1 is a labeled scenario, later steps
are phrased by a **driver LLM playing the user** (direct API, never through the proxy)
and resume the same session. The episode *skeleton* (scenario, intents, expected labels,
required markers) stays mechanical — ground truth is never delegated to the driver.

```bash
PYTHONPATH=src .venv/bin/python -m simulator.run_session --episode random --runs 3
PYTHONPATH=src .venv/bin/python -m simulator.run_session --episode episode_boundary --model haiku
```

Episodes: `episode_boundary` (S15a) `rephrase_retry` (S6) `same_for_other` (working-summary)
`easy_then_hard` (hysteresis) `investigate_then_fix` (S4->S3) `work_then_commit` (3 turns).

## The shadow router (workstream B7 — in progress)

`src/router/` — the routing layer's decision components, built test-first against
captured traffic. So far: the **call-type discriminator** (design §5.1), with
fingerprints copied from live wire evidence rather than guessed. Evaluate it
against all captures at any time:

```bash
PYTHONPATH=src .venv/bin/python -m router.eval_captures   # -> reports/discriminator-eval.md
```

## The execution layer — LiteLLM + local rung (workstream C3/C4)

The local model rung, served by ollama and exposed to the harness in Anthropic
`/v1/messages` format via LiteLLM (design §8.3). This is the first piece of the
production execution stack, not just a Phase-0 experiment.

```bash
./tools/run_ollama.sh     # ollama on :11434 (serves the local model)
./tools/run_litellm.sh    # LiteLLM on :4001 (reads data/anthropic-key for cloud rungs)
```

`config/litellm-local.yaml` maps rungs: `local-code`/`local-small` -> ollama,
`cheap-cloud`/`frontier` -> Anthropic passthrough, with infra fallback
local -> cheap-cloud -> frontier (the S7 path). Verified end-to-end: an
Anthropic-format request routes to `qwen2.5:7b-instruct-q4_K_M` on this M1 Pro
(~29 tok/s decode) and returns a valid Anthropic response, tool calls included.

### S5/S7 scenario capture on the local rung

```bash
./tools/run_ollama.sh && ./tools/run_litellm.sh
PYTHONPATH=src nohup .venv/bin/python -m proxy.capture_proxy --port 4002 \
  --upstream http://localhost:4001 --captures data/captures-local > data/proxy-local.log 2>&1 &
# S5 — dialect failures: edit scenarios on the local model
PYTHONPATH=src .venv/bin/python -m simulator.run_session --scenario fix_test --model local-code --proxy http://localhost:4002
# S7 — infra fallback: dead local endpoint -> cloud (config/litellm-fallback-test.yaml)
PYTHONPATH=src .venv/bin/python -m simulator.induce_fallback
```

Findings in `reports/scenario-local-rung.md`: this 7B does NOT reproduce S5 (measured
reliability ≈1.0 vs the design's 0.87 assumption — the registry measuring, not guessing);
S7 fallback confirmed.
