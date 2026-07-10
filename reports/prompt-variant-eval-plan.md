# Prompt-variant experiment — setup, footprints, and pipeline validation

**Question** (from the harness study, docs/qwen-code-harness-study.md): on the
local rung, does shrinking the fixed prompt prefix cost task success? Cloud
rungs are insulated by prompt caching (99.3% measured cache-hit,
reports/corpus-metrics.md), so the static prefix is a latency/attention
question for small local models, not a dollar question.

**Instrument**: `qwen_harness` (the study port of qwen-code v0.19.8) gained two
experiment knobs — `--prompt-variant` (section ablation) and `--slim-tools`
(first-sentence tool/param descriptions) — and a grid runner,
`qwen_harness.experiments.prompt_variants`, that drives the real agentic loop
(client → scheduler → tools, YOLO approval) through planted-task sandboxes
with **objective verifiers** (unittest passes, rename complete, factual answer
correct, created module behaves). Ground truth is never delegated to a model.

## Measured: turn-1 fixed prefix per cell (system prompt + tool schemas + env context)

| variant | tools | system chars | tools chars | total ~tokens | vs full |
|---|---|---|---|---|---|
| full | full | 11,481 | 5,412 | **4,413** | 100% |
| full | slim | 11,481 | 4,218 | **4,115** | 93% |
| lean | full | 8,481 | 5,412 | **3,663** | 83% |
| lean | slim | 8,481 | 4,218 | **3,365** | 76% |
| minimal | full | 3,609 | 5,412 | **2,445** | 55% |
| minimal | slim | 3,609 | 4,218 | **2,147** | 49% |
| minimal-noex | full | 1,481 | 5,412 | **1,913** | 43% |
| minimal-noex | slim | 1,481 | 4,218 | **1,615** | 37% |

(chars/4 estimate; env context = 762 chars, constant. Note the crossover: at
`minimal` and below, **tool schemas outweigh the system prompt** — the schema
diet matters more the leaner the prose gets.)

What each variant drops (`src/qwen_harness/prompts.py::VARIANT_SECTIONS`):

- **full** — the faithful port: mandates, task management, workflows,
  operational guidelines, sandbox, actions-with-care, git, examples.
- **lean** — drops the ceremony: task-management nagging, tone/communication
  rules, sandbox section, actions-with-care. Keeps mandates, workflows,
  safety, tool rules, git, examples.
- **minimal** — condensed mandates (5 bullets) + core tool rules + examples.
- **minimal-noex** — minimal without the few-shot examples block. This cell
  isolates what the examples buy — the study's key hypothesis is that
  examples are the *last* thing to cut on small models (upstream ships them
  in three notations matched to model families).

## Pipeline validation (this container has no model access — network policy
blocks HF/ollama-registry/ModelScope/GitHub-releases)

A scripted **oracle model** (a competent-agent policy over the wire protocol,
`tests/test_qwen_harness_experiment.py`) drove the full grid through the real
CLI entry point: 4 variants × 2 schema settings × 4 tasks = **32/32 verified
successes**, all ending `done`, with per-run metrics (rounds, tool calls,
errors, tokens, wall time) recorded. The runner, sandboxes, verifiers, and
scorecard are known-good; any failure on a real model is the model's.

## Run the real measurement (M1 Pro, local rung)

```bash
./tools/run_ollama.sh && ./tools/run_litellm.sh     # serves local-code on :4001
PYTHONPATH=src .venv/bin/python -m qwen_harness.experiments.prompt_variants \
    --base-url http://localhost:4001/v1 --model local-code \
    --slim-sweep --runs 3 --no-stream \
    --out reports/prompt-variant-eval.md --jsonl data/prompt-variant-runs.jsonl
```

- 8 cells × 4 tasks × 3 reps = 96 runs; at ~30-90s/run on the 7B expect
  1-2 hours. `--variants full,minimal-noex --tasks fix_test,read_qa --runs 1`
  for a 10-minute smoke first.
- `--no-stream` recommended through LiteLLM→ollama: tool calls arrive as one
  chunk, sidestepping streaming tool-call quirks; token accounting still works.
- Runs are sandboxed to fresh temp dirs, YOLO-approved (the planted projects
  are throwaway), capped at 8 model-call rounds / 300s wall each.
- Interpretation guide: compare **success rate** first, then
  **invalid-params/tool errors** (schema diet cost) and **rounds** (flailing).
  If `minimal` matches `full` on success, the ~2,000-token ceremony is free to
  cut on this model; if `minimal-noex` collapses, the examples earn their
  ~530 tokens.
