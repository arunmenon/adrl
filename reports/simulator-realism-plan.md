# Simulator Realism Plan — closing the gap to the real corpus

**Status:** design only. No code under `src/simulator/` is modified by this document.
**Ground truth:** `data/turns.parquet` (5027 turn-rows, 14 main sessions) + `data/corpus/**` raw sessions.
**Committable:** distributions and scrubbed snippets only. All planted-secret work below uses FAKE, masked-shaped values.

---

## 0. What the simulator is today (anchored to source)

| File / symbol | Current behavior | Consequence |
| --- | --- | --- |
| `sandbox.py::make_sandbox` + `PROJECT_NAMES` | ONE 7-file Python CSV toy, renamed per run | 100% Python; 0 infra; tiny clean context |
| `tasks.py::SCENARIOS` (7 scenarios) | 7 clean phrasings, all CSV-toy coding | no non-coding asks, no secrets, no diversity |
| `driver.py::next_message` + `PROMPT_TEMPLATE` | haiku writes "casual terse 1-2 sentences", proper case/punctuation | opposite of the real surface distribution |
| `episodes.py::EPISODES` | 5 fixed 2-3 step scripts, coherent on-topic thread | fixed length ~2-3; no drift, interrupts, terse polls |
| `run_session.py::spawn_turn` + `DEFAULT_MODELS` | routes to `default/opus/sonnet/haiku` cloud only; `MAX_TURNS=25` | frontier failure FLOOR; escalation controller starved |
| `run_session.py::ALLOWED_TOOLS` | Read/Edit/Write/pytest/git only; single-agent | zero subagents, zero workflow_ids, narrow tool surface |
| `induce_fallback.py` | one-shot infra kill (S7); LiteLLM fallback only | tests the exception ladder, never the semantic trip-wire |
| `config/litellm-local.yaml` | `local-code`=qwen2.5:7b exists but sim never routes to it | the failure-rich local rung is dark |

The four measured profiles say the sim reproduces **none** of: real surface noise, real behavioral noise, the multi-project/big-context/subagent world, or the local-model failure classes the escalation machinery depends on.

---

## 1. Prioritized upgrades

Each upgrade names (a) the measured corpus distribution it reproduces, (b) the concrete change and sim file, (c) an acceptance criterion comparing sim output vs corpus, (d) priority + effort. Metrics are computed on sim-labelled captures joined by `session_id` from `data/sim-ledger.jsonl` — see the scorecard (§2).

### P0-1 — Noisy user-driver: surface distribution
**Reproduces:** instruction-surface profile. word-count p10/p50/p90/p99 = 2/10/72/297 (mean ~31); 24.4% <5 words; 5.7% >150 words; 64.6% no terminal punctuation; 26.1% all-lowercase; 33.3% verbless fragments; 21.1% typos/txt-speak; 0.3% fenced code; 0.0% tracebacks; ~0% human emoji; 13.3% embed an inline file path.
**Approach (`driver.py`):** replace the single tidy `PROMPT_TEMPLATE` path in `next_message`. (1) Sample a **target length bucket** from the empirical word-count CDF before calling the driver, and pass it as a hard constraint. (2) Sample independent **style flags** (lowercase / no-terminal-punct / fragment / question / carries-path / txt-speak) at corpus marginal rates. (3) Apply a mechanical **roughening post-pass** after generation: lowercase with prob 0.26, strip terminal punctuation with prob 0.65, substitute `you->u, your->ur, please->plz, cannot->cant, dont/doesnt`, inject occasional double-spaces and a corpus-derived misspelling, and with prob 0.13 splice in a real sandbox file path. (4) Keep a **terse phrase bank** (`status`, `done yet`, `pushed?`, `go ahead`, `try now`, `plz do so`) sampled directly for the <5-word bucket rather than round-tripping the LLM. Suppress fenced code / tracebacks entirely (they are <0.5% real).
**Acceptance:** over N>=300 generated user turns: word-count p50 in [8,12] and p90 in [58,86] (±20% of 10/72); no-terminal-punct in [55%,75%]; all-lowercase in [20%,32%]; typo/txt-speak in [16%,26%]; fenced-code <1%; emoji ~0%.
**Priority P0 · Effort med**

### P0-2 — Noisy episodes: behavioral distribution
**Reproduces:** behavioral profile. terse follow-ups (<=3 words) 18.9%; questions 34.6%; voice disfluency (uh/um) 13.5%; interrupts 2.5% + organic retry 2.3%; explicit mind-change 1.5%; topic drift median consecutive-turn Jaccard 0.02 (45% zero content-word overlap, 66% <=0.05); session length p50=24, p90=136, max=221 user-turns (0% single-turn, 71% >10); episode depth 1-7 continuations/turn (mean 2.7, tail to ~60).
**Approach (`episodes.py` + `run_session.py::run_episode`):** replace the 5 hand-written fixed `EPISODES` scripts with a **stochastic episode generator**. Draw session length from a long-tailed distribution centered on p50~24 (fat tail to ~220). For each step draw a **turn-type** from a categorical with corpus weights: terse-status-poll, approval-nudge, question/deliberation, mind-change (`actually / no i meant / instead`), new-subproblem, non-coding aside, code follow-up. Drive **topic drift** by, most of the time, sampling the next `intent` from a *different* scenario/subproblem pool rather than continuing the current thread (target near-zero lexical carryover). Add a **disfluency flag** (13.5%) that instructs the driver to produce a run-on with `uh/um` and a mid-sentence restart. Implement **interrupts** by killing the `spawn_turn` subprocess after a random cutoff (2.5% of turns), writing `[Request interrupted by user]` verbatim, and following it with an organic retry/rephrase turn.
**Acceptance:** over generated episodes: terse (<=3-word) in [14%,24%]; questions in [28%,42%]; disfluency in [10%,17%]; interrupt rate in [1.5%,4%] each followed by a retry; median consecutive-turn Jaccard <0.10; session-length p50 in [18,30] with a tail crossing 100 turns; per-turn continuations mean in [2,4].
**Priority P0 · Effort high**

### P0-3 — Local-rung escalation: real failure traffic
**Reproduces:** failure/escalation profile. >=1 error on ~8-10% of tool-using turns; >=2 errors 2.5%; category mix file-not-read 18% / fs-not-found 7% / file-modified 5% / exec-exception 2.5% / timeout 1% / policy-block 2.4% / string-not-found (genuine apply miss) rare; 2-strike edit trip-wire fires on 0.02% at frontier. The corpus is ~100% Opus and is an explicit LOWER BOUND — a small quantized local worker inflates malformed tool JSON, wrong-file/non-unique Edit targets, read-before-write violations, hallucinated-API exceptions, and repeated-identical-call loops.
**Approach (`run_session.py::spawn_turn` routing + `config/litellm-local.yaml` + episodes):** add `local-code` (qwen2.5:7b already in the LiteLLM config) to the model rotation and route a **configurable fraction of edit-heavy turns** (rename/refactor/fix-test) to it via the proxy chain. The smaller model organically emits edit-apply misses, read-before-write errors, malformed tool calls, and identical-call loops — exactly the classes absent from both corpus and sim. Document this rung as a lower-bound-exceeding generator. Frontier turns should first be calibrated to *match* the corpus floor (see acceptance).
**Acceptance:** frontier slice reproduces the floor — >=1 error on [7%,11%] of tool-using turns, >=2 on [1.5%,3.5%]. Local-rung slice EXCEEDS it — >=1 error on >=15% of turns, 2-strike edit trip-wire (`edit_tripwire`) fires on >=2% of local edit turns (vs 0.02% frontier), and >=1 malformed/dialect tool call per ~20 local turns. Net: the escalation controller receives >=1 trip-wire event per sim batch (today: ~0).
**Priority P0 · Effort med**

### P0-4 — Realism scorecard (continuous sim-vs-corpus comparison)
**Reproduces:** meta — the whole metric battery. This is the acceptance-gating instrument for every other upgrade.
**Approach (new tool OUTSIDE `src/simulator/`, e.g. `tools/realism_scorecard.py` + `reports/realism-scorecard.md`):** recompute the identical metric battery on (i) the corpus parquet and (ii) sim-labelled captures (join sim `session_id`s from `data/sim-ledger.jsonl`). Emit a per-metric table: `metric | corpus | sim | ratio | tolerance | PASS/FAIL`, plus one aggregate realism score (share of metrics in tolerance). Group by the four dimensions (surface, behavioral, structural, failure). Re-run after every sim batch; a metric drifting out of band flags automatically.
**Acceptance:** the scorecard runs and reports all P0 metrics; the sim is not declared "realistic" until all P0-1/P0-2/P0-3 rows read PASS. Scorecard itself is deterministic and re-runnable against a fixed corpus snapshot.
**Priority P0 · Effort med**

### P1-5 — Sandbox population + language diversity
**Reproduces:** structural profile. 10 distinct projects; file-touch language mix .ts 6903 / .tsx 2613 / .md 2298 / .sql 1534 / .py 800 / +.sh/.tf/.toml/.yml/.swift — TS/TSX ~58%, Markdown ~14%, SQL ~9%, Python only 4.9%; framework/IaC signal files package.json 191, main.tf 152, next.config 141, tsconfig 40, Dockerfile 8.
**Approach (`sandbox.py::make_sandbox`):** turn `make_sandbox` into a **project-template registry** with >=6 archetypes — Next.js+TypeScript webapp, Terraform/IaC module, SQL/Prisma schema+migrations, Python CLI (the current toy), Swift mobile, and a mixed monorepo — each with matching scaffolding (`package.json`, `next.config.ts`, `main.tf`, `Dockerfile`, `tsconfig.json`) and one gradable planted issue in the dominant language. Weight archetype selection so the aggregate file-touch mix matches the corpus.
**Acceptance:** over a batch, the extension distribution of Edited/Written files is within ±10pp of corpus per major bucket (.ts+.tsx ~58%, .md ~14%, .sql ~9%, .py <=10%); >=6 distinct archetypes emitted; at least one non-Python planted issue graded successfully.
**Priority P1 · Effort high**

### P1-6 — Two-regime big context
**Reproduces:** structural profile. per-turn (cache_read+input)/n_assistant_msgs p50 18k, p90 432.7k, p99 911.4k; 30.5% of turns >32k local-fit ceiling, 20.9% >100k; two regimes — main p50 325.3k / p90 795.3k vs subagent p50 17.0k / p90 43.8k. Also episode depth: continuations p50 4, p90 14, p95 20, tail 200+.
**Approach (`sandbox.py` seeding + `run_session.py`):** seed sandboxes with realistic bulk — large existing codebases, long READMEs and pasted plan/review docs — so main-orchestrator turns carry genuine token load rather than a tiny clean prompt. Distinguish a heavy `main` regime (large repo, long accumulated history) from light isolated subagent tasks. Raise the `MAX_TURNS` ceiling and let episodes accumulate context across resume so continuation depth grows a real tail.
**Acceptance:** measured on sim captures, main-turn (cache_read+input)/n_assistant_msgs crosses the 32k ceiling in >=25% of turns and 100k in >=15%; subagent turns stay light (p50 <30k); continuations p90 >=12.
**Priority P1 · Effort high**

### P1-7 — Subagent-spawning scenarios
**Reproduces:** structural profile. 75.1% of turns are `source_kind=subagent` (3773/5027); 67 distinct workflow_ids; 12/14 sessions spawn subagents; per-session label mix subagent:user_turn:utility ~= 134:24:13.
**Approach (`run_session.py::ALLOWED_TOOLS` + scenarios):** widen `ALLOWED_TOOLS` to include Task/Agent/Workflow + SendMessage/TaskCreate/TaskUpdate, and add orchestration scenarios (the `/team-plan -> /build` pattern) that make the top agent decompose work and spawn subagents. Verify captures carry `source_kind=subagent` and a shared `workflow_id`.
**Acceptance:** >=50% of sim turn-rows are `source_kind=subagent`; every multi-file orchestration session carries >=1 `workflow_id`; per-session subagent:user_turn ratio >=3:1 (corpus ~5.7:1) and utility turns present.
**Priority P1 · Effort med**

### P1-8 — Planted secrets / privacy pressure (FAKE values only)
**Reproduces:** structural profile. 5/3060 session files (0.16%) carry secret-shaped content, all in subagent turns; patterns env_assignment (175 hits) + connection_string_cred (10). Credential-handoff turns ~2.6%.
**Approach (`sandbox.py` + `tasks.py`):** in a small fraction of sandboxes plant a **clearly-fake** `.env` / connection string (masked-shaped placeholder values, never a real secret) and add scenarios that surface them (`paste a DB URL`, `add these creds`, MFA/token handoff), concentrated in subagent turns, so the utility-pinning + secret-scanner path is exercised. Oversample to ~2-5% during scanner testing, then dial to ~0.15% for realism runs.
**Acceptance:** the secret-scanner fires on >= the planted fraction of sessions with zero false negatives on planted items; both env_assignment and connection_string_cred represented; >=1 credential-handoff non-coding turn present. No real secret ever written to disk or logs.
**Priority P1 · Effort med**

### P1-9 — Non-coding / interstitial turns
**Reproduces:** ~17.5% non-typed artifacts (bash-mode 6.6%, `[Image #n]` refs 6.8%, `<teammate-message>` 3.2%, `[Request interrupted by user]` 2.5%, slash-cmd 0.3%) and ~12-18% non-coding asks (image-paste 7.3%, inline-bash 7.2%, money/ops-infra 8.6%, media review 3.4%, credential handoff 2.6%).
**Approach (`episodes.py` interstitial injector + `driver.py`):** add an injector to the episode generator that, at corpus rates, emits non-prose turns — bash-mode command/output blocks, `[Image #n]` attachment refs, `<teammate-message>` inter-agent blocks, and mid-run interrupt markers — interleaved with the coding thread. Some non-coding asks (RunPod/gcloud ops, add-credits, review-a-video) are scenario-level; others are pure interstitials.
**Acceptance:** non-prose interstitial share of generated turns in [12%,22%]; each of bash-mode / image-ref / interrupt-marker present within ±3pp of its corpus rate.
**Priority P2 · Effort med**

---

## 2. Realism scorecard — the standing instrument (detail for P0-4)

A single re-runnable table, one row per measured metric, four sections. Illustrative shape (targets are the corpus column; tolerances shown are the acceptance bands above):

| dim | metric | corpus | sim (now) | tolerance | gate |
| --- | --- | --- | --- | --- | --- |
| surface | word-count p50 / p90 | 10 / 72 | tidy ~15-25 (LLM prose) | ±20% | P0-1 |
| surface | % no terminal punct | 64.6% | ~0% | 55-75% | P0-1 |
| surface | % all-lowercase | 26.1% | ~0% | 20-32% | P0-1 |
| surface | % typos/txt-speak | 21.1% | ~0% | 16-26% | P0-1 |
| behavioral | % terse <=3 words | 18.9% | ~0% | 14-24% | P0-2 |
| behavioral | % questions | 34.6% | low | 28-42% | P0-2 |
| behavioral | interrupt rate | 2.5% | 0% | 1.5-4% | P0-2 |
| behavioral | median consec-turn Jaccard | 0.02 | high (on-topic) | <0.10 | P0-2 |
| behavioral | session length p50 | 24 | ~2-3 | 18-30 | P0-2 |
| failure | >=1 error / tool-turn | 8-10% | ~0 | 7-11% (frontier) | P0-3 |
| failure | 2-strike edit trip-wire | 0.02% | 0% | >=2% (local slice) | P0-3 |
| structural | .py share of file touches | 4.9% | 100% | <=10% | P1-5 |
| structural | ctx/msg p90 > 32k | 30.5% | ~0% | >=25% | P1-6 |
| structural | % turns source_kind=subagent | 75.1% | 0% | >=50% | P1-7 |
| structural | % sessions w/ secret-shaped | 0.16% | 0% | >= planted | P1-8 |

Aggregate realism score = fraction of rows in tolerance. Gate: sim is "realistic" only when every P0 row PASSes. Implemented as `tools/realism_scorecard.py`, joining sim `session_id`s from `data/sim-ledger.jsonl` against the same capture pipeline the corpus metrics use, so sim and corpus are measured by identical code.

---

## 3. Sequencing

1. **P0-4 scorecard first** — you cannot calibrate P0-1/2/3 without the measuring instrument.
2. **P0-1 + P0-2** (driver surface + behavioral) — biggest realism delta, no infra dependency.
3. **P0-3 local-rung** — needs ollama + LiteLLM up (already configured); unblocks the escalation controller, the whole reason the sim exists.
4. **P1-5/6/7** (sandbox population, big context, subagents) — structural realism; higher effort, requires new project templates.
5. **P1-8, P2-9** (secrets, interstitials) — thin slices layered onto the above.

Every step is validated by re-running the scorecard; nothing is declared done on narration.

---

## 4. Scrubbed corpus evidence (illustrative, all masked/truncated <=140 chars)

- terse + txt-speak follow-up: `can u move it workspace ?` (lowercase, `u`, fragment, no period)
- impatient status poll: `stsus now ?` / `statys` / `is it done`
- voice disfluency run-on: `So are you saying that you will, uh, do the fixes... or will you first do the QR code analysis?`
- inline path drop (the real "paste" shape): `/Users/<user>/projects/<repo>/...-datasets.md  Plz review ..makig adjustments`
- planted-secret shape (FAKE, masked): `DB_URL=postgres://***:***@***/***` — env_assignment + connection_string_cred class
- local-model failure class the sim must generate: `<tool_use_error>String to replace not found in file...</tool_use_error>`
- policy-block class absent from sim: `Permission ... denied by the auto mode classifier. Reason: [Credential Materialization] gcloud auth pri...`
