# Phase 0 Implementation Plan — Adaptive Routing Layer (adrl)

Scope: everything in design doc §10 Phase 0 (shadow) plus the scenario-validation bar from the scenarios doc. No Phase 1–3 implementation. Repo root: `/Users/arunmenon/projects/adrl`.

**Effort key**: S = under half a day, M = half to one day, L = one to three days.

**Environment ground rules** (from the audit — non-negotiable):
- No Xcode CLT → nothing may compile from sdist. All Python work runs in `uv venv -p 3.12` (uv fetches a standalone interpreter; 3.12 has mature binary-wheel coverage; system 3.14 risks missing cp314 wheels for pyarrow/orjson/cryptography).
- `ANTHROPIC_BASE_URL` is unset globally → set per-invocation only; unsetting it is the instant kill switch.
- Ollama server is not running → `ollama serve` must be started before any local-model step; `qwen2.5:7b-instruct-q4_K_M` is the only comfortable model on 16 GiB RAM (mistral-small:24b is too tight).
- Transcripts and wire captures contain secrets → `data/` is gitignored from day one; nothing mined leaves the machine unscrubbed.

---

## Workstream A — Transcript miner (first; free data; answers "is this worth building")

### A0. Corpus snapshot — do this immediately
- **Deliverable**: `tools/snapshot_corpus.sh` (rsync `~/.claude/projects` → `data/corpus/`, dated manifest with file counts/sizes/mtimes) plus `.gitignore` covering `data/`.
- **Why first**: severe survivorship bias is live — main-session transcripts older than ~30 days are already gone (17 main files vs 32 session dirs). Every day of delay loses labeled turns.
- **Acceptance**: snapshot completes; manifest shows ≥3,000 .jsonl files / ~931MB; `git status` shows no `data/` files tracked.
- **Effort**: S

### A1. Streaming record parser
- **Deliverable**: `src/miner/parser.py` — line-streaming JSONL reader (never slurps; files reach 30MB), permissive `.get` access everywhere, explicit record-type allowlist with unknown-type and bad-line counters, envelope normalization across the 2.1.150→2.1.199 schema drift (toolEndsTurn, origin/promptSource, attribution* fields, envelope-less utility records), uuid dedup, `version` bucketing.
- **Acceptance**: parses the full snapshot with 0 crashes; emits a parse-stats summary (records by type, unknown types, bad lines) matching the scout's corpus profile within tolerance; handles the two 10MB+ files under ~200MB RSS.
- **Effort**: M

### A2. Turn assembler
- **Deliverable**: `src/miner/turns.py` — groups user records by `promptId` (first non-meta str/text record = instruction; subsequent tool_result records = continuations), attaches assistant records via `parentUuid` chain (per-file uuid→record index), closes turns at `stop_reason=end_turn` or next promptId, joins `system/turn_duration` for wall-clock/messageCount, handles compact_boundary/resume forks without double-counting.
- **Acceptance**: on 10 hand-picked transcripts (including one workflow subagent and one compacted main session), assembled turns match manual inspection; per-turn round-trip counts reproduce observed range (1 to 250+).
- **Effort**: M

### A3. Label classifier (offline twin of the discriminator)
- **Deliverable**: `src/miner/labels.py` — labels every turn `user_turn` / `continuation` / `utility` (split `utility:light` vs `utility:compaction` where derivable) / `subagent`, per the discriminator rules: isSidechain/agentId/subagents-path → subagent; isMeta or `<command-name>` → utility; tool_result-bearing → continuation; tie-break → user_turn. Flags: interrupt (`[Request interrupted by user` prefix), tool rejection (`toolUseResult == 'User rejected tool use'`), is_error tool_results, synthetic messages.
- **Acceptance**: labels 100% of turns; a stratified sample of 50 turns (all four labels represented) hand-verified with ≥48/50 agreement; ambiguous cases resolve to user_turn.
- **Effort**: M

### A4. Flat turn dataset
- **Deliverable**: `src/miner/extract.py` → `data/turns.parquet` (pyarrow in the 3.12 venv; CSV fallback if any wheel issue). One row per turn: session, project, ts, label, n_tool_roundtrips, tool-name histogram, MCP/builtin split, model id, tokens (in/out/cache_read/cache_creation), duration_ms, stop_reason, thinking presence/size, error/interrupt/rejection flags, gitBranch/cwd, version bucket.
- **Acceptance**: row count reconciles with A2 turn count after uuid dedup; spot-check 10 rows against raw JSONL; loads in pandas without error.
- **Effort**: M

### A5. Scenario matchers (corpus-answerable fingerprints)
- **Deliverable**: `src/miner/scenarios.py` + extracted trace files under `reports/scenario-matches/S<nn>/` (≥3 real traces each, or an explicit FALSIFIED verdict). Covers the 11 scenarios the corpus can validate (mapping table below): S2, S3, S4, S6, S8, S9, S10, S12, S13, S14 (transcript side), S15 (both parts — part B checks the zero-mid-turn-switch invariant via per-turn model-id constancy). S11 (Codex) is run as an absence check — zero matches expected → doc fix per the validation note.
- **Acceptance**: per-scenario match counts printed; every scenario has either ≥3 extracted traces on disk or a falsification note naming what to change in `docs/adaptive-routing-scenarios.md`; S15 part B reports mid-turn model switches = 0 (any nonzero is a stop-and-investigate finding).
- **Effort**: L

### A6. Metrics report (the miner's required output)
- **Deliverable**: `src/miner/report.py` → `reports/corpus-metrics.md`, containing all of:
  1. Traffic shares by label vs the §4 illustrative table (continuation 70–90% / utility 5–15% / user_turn 5–15% / subagent occasional), reported per-category and overall, with the workflow-subagent skew called out.
  2. Trip-wire frequencies per §5.5: edit-apply failure rate (2-strike), parse/schema failure rate, canonicalized identical-tool-call loop rate (3 identical), no-progress runs (6 actions, no new file read/diff/output hash), turn-budget exceedance (2× median tokens or 90s per intent class), interrupt-then-rephrase rate.
  3. Per-intent-class medians for tokens and wall-clock (sets §5.5 turn-budget thresholds).
  4. Token economics: $ per session (joined to price table), tokens/turn, cache-hit ratios, stop_reason distribution.
  5. User-retry/interrupt rate baseline by model (the Phase 2 "≤ frontier baseline" gate number).
  6. Per-(model, harness) tool-call reliability (registry input, §5.4), dialect failures tagged `label_quality=dialect_failure`.
  7. Regret proxy: turns with failure signatures that heuristics should have sent direct to frontier, priced in wall-clock.
  8. Privacy exposure count: sessions where a tool_result carried a secret and later requests left the machine (offline analogue of the §11 zero-violations alarm).
- **Acceptance**: report generates from `data/turns.parquet` in one command; every table names its denominator; §4 comparison explicitly states whether the continuation-dominance assumption held.
- **Effort**: L

### A7. Best-single-model baseline
- **Deliverable**: `src/miner/baseline.py` → `reports/baseline.md`. Prices all replayed historical turns as if served by the chosen best single model (user decision D1), per LLMRouterBench methodology: cost side from usage × price table (including cache-read/creation pricing); quality side from the corpus's observed frontier-served outcomes (retry/interrupt/error rates) — note the corpus is ~all Opus/Fable traffic, so it is nearly a native best-single-model sample, which makes this the cleanest number Phase 0 produces. This is the number the Phase 3 learned router must beat.
- **Acceptance**: single headline table — $/session, $/turn, tokens/turn, retry rate — for the baseline model, with methodology and price-table version pinned in the report.
- **Effort**: M

### A8. Secret-detection calibration (Phase 2 prep, §13.4)
- **Deliverable**: `src/miner/secrets.py` — both cheap variants (regex/entropy scan on request-side text; scanner on file-read tool_results) run over the corpus; FP/TP counts on a hand-labeled sample of 100 flagged hits → section in `reports/corpus-metrics.md`.
- **Acceptance**: measured false-positive rate for each variant; ≥3 true secret-exposure sessions identified (also satisfies S9).
- **Effort**: M

---

## Workstream B — LiteLLM wire-capture proxy (second; answers what transcripts cannot)

The corpus never shows the wire: no system prompts as sent, no metadata.user_id, no sidecar utility requests, no cache_control layout, no sampling params. B exists for exactly those gaps plus the live shadow run.

### B1. Environment + proxy skeleton
- **Deliverable**: `tools/setup_env.sh` (`uv venv -p 3.12 .venv && uv pip install litellm[proxy] pyarrow pandas`), `config/litellm-capture.yaml` (pure passthrough to Anthropic, model alias preserved), `tools/run_proxy.sh` (starts proxy on localhost, prints the `ANTHROPIC_BASE_URL` export line).
- **Acceptance**: `claude -p "say hi" ` with `ANTHROPIC_BASE_URL=http://localhost:<port>` returns a normal response; zero sdist compilation during install; unsetting the env var restores direct traffic (kill switch verified).
- **Effort**: M

### B2. Capture callback
- **Deliverable**: `src/proxy/capture_callback.py` (LiteLLM `success_callback`/`failure_callback`) writing one JSONL record per request to `data/captures/YYYY-MM-DD.jsonl`: full request body (model, system, messages, tools, metadata, max_tokens, temperature, cache_control placement), response (content, stop_reason, usage), timing (arrival, TTFB, total), and a request-side secret scrub pass per decision D5.
- **Acceptance**: a 5-turn interactive Claude Code session produces captures whose request count ≥ transcript record count for the same session (sidecar calls now visible); every previously "missing_feature" field (system prompt on the wire, metadata, max_tokens, cache_control) is present in captures.
- **Effort**: M

### B3. Live capture period
- **Deliverable**: routing daily Claude Code work through the proxy for 5–10 working days (decision D6); `data/captures/` accumulating ≥200 genuine user turns plus the surrounding continuation/utility traffic.
- **Acceptance**: ≥200 user_turn-labeled requests captured; ≥10 distinct sessions; ≥3 session-open bursts (feeds S1).
- **Effort**: S to set up, elapsed time dominates

### B4. Assumption test 1 — metadata.user_id semantics (v2 exit criterion)
- **Deliverable**: `src/experiments/user_id_test.py` → `reports/assumption-user-id.md`. Compares `metadata.user_id` across ≥5 captured sessions (and across `claude` restarts): constant → per-user, unusable alone for session keying, adopt/validate the §5.6 fallback `hash(system_prompt + first_user_message)`; varying per session → usable natively. Report includes the session-keying method distribution metric (§12) and validates the chosen key against captured sessions (no two concurrent sessions collide, no session splits mid-way).
- **Acceptance**: a definitive per-session vs per-user verdict with evidence, plus a stated session-keying decision for Phase 1.
- **Effort**: S

### B5. Assumption test 2 — tool-call ID re-minting (v2 exit criterion)
- **Deliverable**: `src/experiments/tool_id_test.py` → `reports/assumption-tool-ids.md`. Takes a captured Anthropic transcript containing `toolu_*` tool_use/tool_result pairs and replays it (a) back to Anthropic with synthetically re-minted but internally consistent IDs, (b) to Ollama qwen2.5:7b via LiteLLM translation, (c) cross-format (Anthropic-shaped IDs in an OpenAI-dialect request via LiteLLM). Records accept/reject per case; explicitly separates provider rejections from LiteLLM translation bugs (run one case direct-to-API without LiteLLM as control).
- **Acceptance**: a yes/no answer to "must IDs be provider-minted, or only internally consistent?" with the raw request/response evidence attached; consequence for the escalation-rebuild scrubber stated in one paragraph.
- **Effort**: M

### B6. Wire-only fingerprint grounding (S1, S14, utility tells)
- **Deliverable**: `src/proxy/wire_fingerprints.py` + 5 real captures per wire-validated scenario filed under `data/captures/scenarios/S<nn>/`, and `reports/wire-fingerprints.md` documenting the real utility tells (actual Haiku-class model ID requested, actual max_tokens, actual topic-detection/title/compaction system prompts verbatim) to replace the docs' representative payloads.
- **Acceptance**: S1 (session-open sidecar burst) and S14 (compaction request wire shape: summarization system prompt, full transcript, ~8000 max_tokens, cache-cold successor) each have ≥3 matching real captures (5 preferred); the utility fingerprint constants used by the discriminator are copied from captures, not from the doc.
- **Effort**: M

### B7. Shadow pipeline — discriminator + features + policy, log-only
- **Deliverable**: `src/router/discriminator.py` (mechanical rules from §5.1, constants sourced from B6), `src/router/features.py` (intent isolation: last non-tool user message with known-prefix boilerplate stripping; prompt-size, tool-mix, session-depth features), `src/router/policy.py` (heuristic-only Phase-0 policy: pins, T_HARD-style scoring, trip-wire stubs), wired as an inline LiteLLM callback that logs `{label, features, would_route_to, two_rung_counterfactual, latency_us}` per request to `data/shadow/decisions.jsonl` — never alters routing.
- **Acceptance**: runs on live traffic for the remainder of B3; zero requests altered or failed by the callback (verified by comparing capture success rates with/without shadow enabled).
- **Effort**: L

### B8. Shadow run report + overhead measurement
- **Deliverable**: `reports/shadow-run.md` — over ≥200 real live turns: predicted local share, predicted escalation rate, label shares vs §4, router overhead p50/p99 measured separately for the sticky path (continuation → state lookup) and the decision path (user_turn → full scoring), and the two-rung counterfactual rung per turn (§13.1).
- **Acceptance**: the §10 Phase 0 gate line, filled in: ≥200 turns, overhead p50 < 20ms (report against the stricter §11 targets of <5ms sticky / <30ms decision too), and a stated judgment on whether local share and escalation rate "look sane" with reasons.
- **Effort**: M

### B9. Offline replay harness
- **Deliverable**: `src/replay/replay.py` — replays historical captures (B) and mined corpus turns (A, via an adapter that reconstructs request-shaped inputs from transcript records, with a documented fidelity caveat: transcripts lack the wire system prompt) through the shadow pipeline offline; emits the same decisions.jsonl schema → `reports/replay.md` with counterfactual local-share/escalation/cost estimates per §5.7 use 3.
- **Acceptance**: full corpus replays without error; replay decisions on the ≥200 live turns match the live shadow decisions bit-for-bit (determinism check); counterfactual cost table sits alongside the A7 baseline in one comparison.
- **Effort**: L

### B10. Discriminator drift canary
- **Deliverable**: `tests/test_discriminator_canary.py` — 20 stored real handshakes (from `data/captures/scenarios/`, scrubbed) checked into `tests/fixtures/handshakes/`; test replays them through the discriminator and fails on any label change. Runnable via `pytest`; wire into CI when the repo gets CI.
- **Acceptance**: test passes on current code; deliberately perturbing one fingerprint constant makes it fail (verified once).
- **Effort**: S

---

## Workstream C — Headless-Claude simulator (last; gap-filling only)

Gate: C starts only after `reports/gap-analysis.md` (C1) shows which scenarios still lack ≥3 traces. Expected gaps: **S5** (needs local-model traffic that neither corpus nor Anthropic-only captures contain) and **S7** (needs an induced infra fallback). S1 only if B3 produced <3 session-open bursts.

### C1. Gap assessment
- **Deliverable**: `reports/gap-analysis.md` — per-scenario tally from A5 + B6: validated (≥3 traces), falsified (0 matches, doc fix filed), or gap (needs C).
- **Acceptance**: every one of S1–S15 has exactly one status; each falsified scenario has a corresponding edit proposed for `docs/adaptive-routing-scenarios.md` (expected: S11 falsified — no Codex usage — recommend drop/defer).
- **Effort**: S

### C2. Simulator harness
- **Deliverable**: `src/simulator/run_session.py` — spawns `claude -p` (headless, `--output-format stream-json`, `--dangerously-skip-permissions`) in throwaway sandbox repos under the scratchpad, with `ANTHROPIC_BASE_URL` pointed at the capture proxy and `~/.local/bin` on PATH; task scripts in `tools/sim_tasks/*.md` (small real coding tasks — no mock work: clone a small OSS repo, fix a real failing test, etc.); per-run token accounting against budget D2.
- **Acceptance**: one end-to-end simulated session completes, appears in `data/captures/`, and is correctly labeled by the shadow discriminator; token spend logged per run.
- **Effort**: M

### C3. S5 — malformed tool calls via local model
- **Deliverable**: LiteLLM route `local-qwen` → Ollama (`tools/run_ollama.sh` starts `ollama serve` and health-checks :11434); simulator sessions driven against qwen2.5:7b-instruct-q4_K_M until ≥3 captures show tool calls emitted as text (`<tool_call>{json}</tool_call>` in a text block) with the harness nudge + repeat (2 parse strikes). Traces filed under `reports/scenario-matches/S05/`.
- **Acceptance**: ≥3 real parse-failure traces captured; the parse-strike fingerprint in the scenarios doc updated to match observed reality; failures tagged `label_quality=dialect_failure` in the dataset.
- **Effort**: M (model behavior is stochastic; may need several sessions)

### C4. S7 — induced infra fallback
- **Deliverable**: LiteLLM config `config/litellm-fallback-test.yaml` with a primary route pointed at a dead port and a healthy fallback; simulator run producing mid-turn APIConnectionError → retried on the fallback route, captured end to end. Traces to `reports/scenario-matches/S07/`.
- **Acceptance**: ≥3 captures showing connection error + same-turn retry served by a different route with no semantic-failure strikes nearby; observed wire shape documented back into the scenarios doc.
- **Effort**: S

### C5. Top-up runs (conditional)
- **Deliverable**: additional simulator sessions for any other scenario still under the ≥3 bar (e.g., S1 session-open bursts), clearly tagged `source=simulator` in all datasets so synthetic traces never contaminate the organic baseline numbers.
- **Acceptance**: every non-falsified scenario reaches ≥3 traces; organic vs simulated provenance is queryable in every report.
- **Effort**: S–M depending on gaps

---

## Final deliverable

### F1. Phase 0 exit report
- **Deliverable**: `reports/phase0-exit.md` — one page per §10 exit criterion: shadow sanity (local share, escalation rate) on ≥200 turns; overhead p50; best-single-model baseline number; user_id verdict; tool-ID verdict; replay completed; S1–S15 validation table; and the business-case answer ("is this worth it?") in dollars from A6/A7/B9.
- **Acceptance**: every criterion marked pass/fail/blocked with a pointer to evidence; doc edits from falsified scenarios merged into `docs/adaptive-routing-scenarios.md`.
- **Effort**: S

---

## Scenario → workstream mapping

| Scenario | Validating workstream | Notes |
|---|---|---|
| S1 cold open (sidecar burst) | **B** (C5 top-up if <3 bursts) | Sidecar utility requests are invisible in transcripts — wire-only |
| S2 one-turn multi-request | **A** (wire prefix-sharing confirmed in B) | promptId grouping shows it directly; 99% prefix overlap needs captures |
| S3 edit trip-wire | **A** | 8 corpus files already contain the exact `is_error` string |
| S4 grep loop / no-progress | **A** | Canonicalized tool_use args from transcripts |
| S5 malformed tool call | **C** (via B proxy + Ollama) | Requires local-model traffic; none exists in corpus or Anthropic captures |
| S6 interrupt-and-rephrase | **A** | 77 + 11 interrupt markers already counted in corpus |
| S7 infra fallback | **C** (induced, captured by B) | Not observable in transcripts; unlikely to occur organically in a short window |
| S8 easy intent, huge context | **A** | usage tokens + isolated prompt text |
| S9 secret exposure | **A** | tool_result scanning (A8) |
| S10 hard direct-to-frontier | **A** | Heuristic markers over prompt text |
| S11 Codex dialect | **A (absence check)** | Expect zero matches → falsify, drop/defer, fix doc (decision D4) |
| S12 parallel tool calls | **A** | Multi-tool_use assistant messages visible in transcripts |
| S13 subagent spawn | **A** (wire linkage confirmed in B) | Task tool_use + 3,024 subagent files; fresh-conversation wire shape from captures |
| S14 auto-compaction | **A + B** | compact_boundary/isCompactSummary in corpus; request shape (system prompt, max_tokens ~8000) wire-only |
| S15 episode boundary / no mid-turn switch | **A** (part B re-checked on B captures) | Part B invariant: zero mid-turn model switches, same-provider tool ids/thinking blocks |

---

## File layout

```
adrl/
├── config/
│   ├── litellm-capture.yaml
│   └── litellm-fallback-test.yaml
├── data/                      # gitignored — secrets live here
│   ├── corpus/                # A0 snapshot of ~/.claude/projects
│   ├── turns.parquet          # A4
│   ├── captures/              # B2 wire logs (+ scenarios/S<nn>/)
│   └── shadow/decisions.jsonl # B7
├── src/
│   ├── miner/       parser.py turns.py labels.py extract.py scenarios.py report.py baseline.py secrets.py
│   ├── proxy/       capture_callback.py wire_fingerprints.py
│   ├── router/      discriminator.py features.py policy.py
│   ├── replay/      replay.py
│   ├── experiments/ user_id_test.py tool_id_test.py
│   └── simulator/   run_session.py
├── tools/           snapshot_corpus.sh setup_env.sh run_proxy.sh run_ollama.sh sim_tasks/*.md
├── tests/           test_discriminator_canary.py fixtures/handshakes/ (scrubbed)
└── reports/         corpus-metrics.md baseline.md assumption-user-id.md assumption-tool-ids.md
                     wire-fingerprints.md shadow-run.md replay.md gap-analysis.md
                     scenario-matches/S<nn>/ phase0-exit.md
```

## Sequencing and rough calendar

1. **Day 1**: A0 snapshot (immediately), B1 env setup in parallel, decisions D1–D6 from user.
2. **Days 2–5**: A1–A4 (parser → dataset), then A5–A8 (scenarios, metrics, baseline, secrets). Start B2–B3 live capture in the background from day 2 so the ≥200-turn clock runs while mining.
3. **Days 5–9**: B4–B6 (assumption tests, wire fingerprints), B7–B8 (shadow pipeline + run), B9 replay, B10 canary.
4. **Days 9–12**: C1 gap assessment → C2–C5 only for actual gaps.
5. **Day 12–14**: F1 exit report; doc fixes for falsified scenarios.

## Risks

1. **Corpus decay (highest urgency)**: main transcripts are garbage-collected on a ~30-day window; A0 must run before anything else. Mitigation: snapshot day 1; re-snapshot weekly during Phase 0.
2. **Single-user, workflow-skewed corpus**: 2,882 of ~3,100 files are workflow subagents; raw shares will not match typical traffic. Mitigation: report every metric per-category, never from raw file shares; treat the §4 comparison as per-category checks.
3. **No CLT / Python 3.14 wheels**: any sdist build fails. Mitigation: `uv venv -p 3.12`, binary wheels only; pyarrow fallback to CSV+sqlite (stdlib) if needed.
4. **Proxy in the loop of real work**: B3 puts LiteLLM between the user and Anthropic for daily work; a proxy bug stalls real sessions. Mitigation: kill switch = unset ANTHROPIC_BASE_URL; shadow callback is fail-open (exceptions logged, request proceeds); verify with B7 acceptance check.
5. **Secrets on disk**: both corpus snapshot and wire captures contain credentials. Mitigation: `data/` gitignored from commit 1; scrub-at-write option (D5); canary fixtures scrubbed before entering `tests/`.
6. **≥200 live turns may take longer than planned**: organic user_turn volume is 5–15% of traffic. Mitigation: start capture on day 2; top up with simulator turns only if D2 allows, always tagged `source=simulator` and reported separately.
7. **Tool-ID test confounding**: LiteLLM translation bugs can masquerade as provider ID requirements. Mitigation: B5 includes a direct-to-API control case without LiteLLM.
8. **Local-model resource limits**: 16 GiB RAM; Ollama server not running by default. Mitigation: qwen2.5:7b only; `tools/run_ollama.sh` health-checks before any C3 run; never run simulator + mistral-small concurrently.
9. **Fingerprint drift between corpus era (2.1.150–2.1.199) and live CLI (2.1.201+)**: constants mined from old transcripts may not match live wire traffic. Mitigation: B6 sources discriminator constants from live captures, not the corpus; B10 canary locks them.
10. **Transcripts under-count requests**: sidecar calls and `usage.iterations` mean corpus-derived request volume is a lower bound. Mitigation: all volume claims in reports labeled "transcript-visible"; wire-true volume comes from B only.

## Decisions the user must make (before day 2)

- **D1 — Baseline model + price table**: which model is the "best single model" for the A7 baseline (claude-opus-4-8 vs claude-fable-5?), and which price-table version/source to pin in the report.
- **D2 — Simulator token budget**: hard cap on tokens/$ the headless-Claude simulator may burn (C2–C5 run against real Anthropic billing except the Ollama-served S5 runs); suggested framing: a per-scenario cap and a total cap.
- **D3 — Live-capture commitment**: how many working days of real Claude Code usage route through the proxy (B3), and whether all projects or only selected ones.
- **D4 — S11 disposition**: confirm dropping/deferring the Codex-dialect fingerprint if the absence check comes back zero (no Codex usage on this machine).
- **D5 — Capture scrubbing policy**: store raw wire bodies (max fidelity, secrets on local disk) vs scrub-at-write (safer, risks destroying fingerprint evidence). Recommendation: raw on encrypted local disk, gitignored, scrub only what leaves `data/`.
- **D6 — Local model for dialect tests**: confirm qwen2.5:7b-instruct-q4_K_M as the S5/tool-ID local target (only comfortable fit in 16 GiB), or pull a different small model within the ~99 GiB disk headroom.