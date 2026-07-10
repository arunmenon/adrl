# The qwen-code harness, re-implemented for study

`src/qwen_harness/` is a working Python port of the core architecture of
[qwen-code](https://github.com/QwenLM/qwen-code) (Apache-2.0, itself a fork of
google-gemini/gemini-cli). The purpose is to make the harness *legible*: every
module is a readable ~100-300 line port of one upstream TypeScript file, with
the exact constants, state machines, prompt texts, and wire formats preserved,
so you can study how a production coding agent actually works — and run it
against this repo's local LiteLLM/ollama rung to watch it on the wire.

Pinned upstream: **qwen-code v0.19.8, commit `7ad2a5f` (2026-07-10)**, studied
via four parallel deep-reads of `packages/core` and `packages/cli`.

## Run it

```bash
uv pip install --python .venv/bin/python aiohttp   # already in requirements
./tools/run_ollama.sh && ./tools/run_litellm.sh    # the local rung (:4001)

PYTHONPATH=src .venv/bin/python -m qwen_harness "explain src/router"   # one-shot
PYTHONPATH=src .venv/bin/python -m qwen_harness                        # REPL
# knobs: --model, --base-url, --approval-mode {plan,default,auto-edit,yolo}, -y
```

It speaks OpenAI-compatible chat completions (what DashScope, LiteLLM, ollama,
and vLLM all serve), so `--base-url http://localhost:4002/v1` routes it through
the capture proxy and every request/response of the agentic loop lands in
`data/captures/` for inspection.

Tests: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_qwen_harness_*.py`
(35 tests, including a full loop over a real SSE wire against a scripted server).

## The life of one prompt

```
user prompt
  │
  ▼
GeminiClient.send_message_stream          client.py  (client.ts)
  │  reset loop detector (top-level only); session turn cap
  │  compression gate: est. tokens ≥ auto threshold? → summarize history
  ▼
Turn.run                                  turn.py    (turn.ts)
  │  GeminiChat.send_message_stream       chat.py    (geminiChat.ts)
  │    │  push user turn into history (before the call!)
  │    │  repair orphaned tool calls
  │    │  curate history → convert to OpenAI messages → POST /chat/completions
  │    │                                  content_generator.py (openaiContentGenerator/)
  │    │  stream chunks; reassemble fragmented tool calls
  │    │  validate stream (finish reason + content, else retry ×2)
  │    └  record ONE consolidated model turn after stream end
  │  adapt chunks → typed events: Content / Thought / ToolCallRequest / Finished
  ▼
per event: loop-detection safeties        services/loop_detection.py
  │
  ├─ pending tool calls? → CoreToolScheduler.schedule   scheduler.py (coreToolScheduler.ts)
  │     validating → (approval) → scheduled → executing → success/error/cancelled
  │     results → functionResponse parts → send_message_stream(again)  ← THE LOOP
  │
  └─ no tool calls → next-speaker check   services/next_speaker.py
        verdict 'model' → send_message_stream("Please continue.", turns-1)
        verdict 'user'  → turn chain ends, control returns to the human
```

The single most important architectural fact: **the model loop and tool
execution are decoupled**. `Turn` only *announces* tool calls; the driver
(`cli.py`, port of `nonInteractiveCli.ts`) executes them and starts a
continuation. That seam is where approval UIs, permission systems, schedulers,
and (in this repo's future) routing layers plug in.

## Module map and what to notice in each

| Python | Upstream TS | Study notes |
|---|---|---|
| `types.py` | `turn.ts`, @google/genai | qwen-code is *Gemini-native inside* despite serving Qwen over OpenAI wire; conversion happens only at the edge. |
| `prompts.py` | `core/prompts.ts` | One big static system prompt with tool names interpolated; QWEN.md memory appended after `\n\n---\n\n`; sandbox/git sections computed at build time; examples come in three tool-call *notations* selected by model-name regex (qwen-coder XML vs qwen-vl JSON vs generic) — matching the token syntax each model family was trained on. |
| `memory.py` | `utils/memoryDiscovery.ts` | Hierarchy = global `~/.qwen/QWEN.md` → upward scan (root-most first) → `.qwen/QWEN.local.md` last so it overrides; each file framed with `--- Context from: … ---`; `@path` imports inline recursively (depth 5, confined to project root). |
| `environment.py` | `utils/environmentContext.ts` | Everything the model "knows" about the session (date, OS, cwd, folder tree) enters as history[0]. Folder tree capped at 20 items — trimmed from 200 to save prompt tokens. |
| `token_limits.py` | `core/tokenLimits.ts` | Ordered regex table over an aggressively normalized model id; first match wins; default 200k. |
| `content_generator.py` | `openaiContentGenerator/` | Gemini↔OpenAI conversion; fragmented `tool_calls` deltas reassembled per index and emitted only at finish_reason; truncated tool-call JSON downgrades finish to MAX_TOKENS; retry 429/5xx exponential (1.5s→30s, jitter 0.3, Retry-After honored); stream-idle watchdog 240s. |
| `chat.py` | `core/geminiChat.ts` | Two histories: *comprehensive* (everything) vs *curated* (what the model sees — invalid model runs dropped whole, consecutive user turns merged). Model turn recorded once, consolidated, after the stream completes. Empty/finish-less streams raise and retry (2× budget). User turn pushed *before* the call. |
| `turn.py` | `core/turn.ts` | Stream chunks → typed events; never executes tools; errors become ERROR events, not exceptions. |
| `client.py` | `core/client.ts` | MAX_TURNS=100 bounded recursion; recursion exhaustion is *silent* (upstream behavior); compression gate pre-send; loop-detector state spans tool-result continuations; next-speaker 'model' → literal `"Please continue."` |
| `scheduler.py` | `core/coreToolScheduler.ts` | The state machine: `validating → scheduled/awaiting_approval → executing → success/error/cancelled`. Execution waits for the *whole batch* to clear approvals; consecutive read/search/fetch calls run in parallel, mutators run alone in order. Results reach the model only as `functionResponse` parts: `{output: …}` on success, `{error: …}` on failure. Repeated identical validation failures (≥3) append a `⚠ RETRY LOOP DETECTED` directive. |
| `tools/base.py` | `tools/tools.ts` | The DeclarativeTool/ToolInvocation split: validate once, then everything downstream holds a known-good object. `llm_content` vs `return_display` — model and human get different views. |
| `tools/fs.py` | `ls.ts` `read-file.ts` `glob.ts` `grep.ts` | Absolute paths mandatory; workspace reads auto-'allow'; every cap is *announced in the output* ("Showing lines 1-1000 of 5000…") so the model knows its view is partial; glob sorts last-24h files newest-first. |
| `tools/edit_tools.py` | `edit.ts` `write-file.ts` | Exact-literal `old_string` with counted occurrences; the error messages are *pedagogical* (they tell the model to re-read the file); whitespace-tolerant fallback matching; `old_string:""` = create; prior-read enforcement (can't edit what you haven't read; can't edit what changed since you read it). |
| `tools/shell.py` | `shell.ts` | Mostly classification, not execution: substitution (`$()`, backticks) always forces a prompt; read-only classification is fail-closed; process groups + timeout (120s default / 600s cap); background mode detaches with an output file; the result block always reports Command/Directory/Output/Error/Exit Code/Signal. |
| `tools/misc.py` | `todoWrite.ts`, classic `memoryTool.ts` | todo_write's entire effect is the `<system-reminder>` echoed into context (whole-list replacement, never a delta); save_memory appends `- fact` under `## Qwen Added Memories` where next session's discovery finds it. |
| `services/loop_detection.py` | `services/loopDetectionService.ts` | Two tiers: always-on safeties (5 consecutive identical calls; 100-call turn cap) vs opt-in heuristics (duplicate@6, ABABAB, read-loop 8-of-15, 50-char sha256 sliding-window chanting@10). |
| `services/next_speaker.py` | `utils/nextSpeakerChecker.ts` | Verbatim CHECK_PROMPT + JSON schema; three rule shortcuts avoid the LLM call; any failure = "user speaks next". |
| `services/compression.py` | `services/chatCompressionService.ts` | Threshold ladder (warn/auto/hard around 0.7·window); full-history summarization into a 9-section `<state_snapshot>` XML; reseeds history as `[user(summary+trailer), model("Got it. Thanks for the additional context!")]`; circuit breaker after 3 failures. |
| `cli.py` | `nonInteractiveCli.ts` | The canonical `while True: stream → collect tool calls → schedule → feed responses back` loop; non-interactive mode auto-denies anything needing a dialog (and tells the model why). |

## What we learned about how the harness *evolved* (v0.19.8 vs the gemini-cli lineage)

The current HEAD has drifted a long way from the fork point, mostly by
converging on claude-code idioms. Load-bearing changes found during the study:

1. **Compression was rewritten.** The classic gemini-cli design summarized the
   *first 70%* of history and kept the last 30% verbatim
   (`COMPRESSION_TOKEN_THRESHOLD`/`findIndexAfterFraction`). Current qwen-code
   summarizes the **entire** history into one `<state_snapshot>` XML document
   (with a private `<analysis>` scratchpad stripped before it enters history),
   re-reads the 5 most recently touched files from disk afterwards, and seeds
   the new history with a canned model ack. The port implements the current
   design.
2. **Heuristic loop detection is now opt-in** (`skipLoopDetection` defaults to
   true); only a small set of always-on safeties (consecutive-identical @5,
   per-turn cap @100, git-status stagnation) runs unconditionally. The
   LLM-based loop check that gemini-cli had was removed entirely. Evidently
   false positives cost more than loops.
3. **The environment prelude lost its ack.** Classic: `[user(env context),
   model("Got it. Thanks for the context!")]`. Current: a single
   `<system-reminder>`-wrapped user message, no synthetic model turn.
4. **`shouldConfirmExecute()` split into `getDefaultPermission()` (intrinsic
   allow/ask/deny) + `getConfirmationDetails()` (dialog payload)**, feeding a
   layered permission pipeline (tool intrinsic → persisted rules → mode).
5. **edit switched from `expected_replacements: N` to `replace_all: bool`**,
   and gained prior-read enforcement (must read a file this session before
   editing it; TOCTOU-checked).
6. **`save_memory`, `read_many_files`, and web-search are gone** from the
   registered toolset (memory became an automatic subsystem; the port keeps
   the classic `save_memory` since it closes the loop with QWEN.md discovery).
7. Newer machinery on top (not ported, noted for completeness): plan mode,
   skills, subagent runtime (`agent` tool with fork/background modes), hooks
   (UserPromptSubmit/PreToolUse/Stop), deferred tools + `tool_search`, model
   fallback chains, microcompaction, output persistence gates with
   `<persisted-output>` stubs, and a per-batch 200k-char output budget.

## Deliberate simplifications in the port

Kept: every mechanism above with its real constants, formats, and prompts.
Simplified: JSON-schema validation is a required/type check; grep is the
pure-Python rung of upstream's rg→git-grep→grep→JS chain; the shell read-only
classifier is an allowlist rather than a full AST; truncation doesn't spill to
disk; telemetry, IDE context, MCP, OAuth, and the Ink UI are absent; thoughts
ride the OpenAI `reasoning_content` field only. Each omission is flagged in
the module docstrings.

## Why this lives in adrl

The routing layer treats the harness as a black box on the wire. This port
makes the box transparent: it shows exactly *why* the traffic looks the way it
does — why turn 1 of a session is huge (system prompt + env context + memory),
why continuations are cheap (curated history is append-only until compression
fires), what a compression event does to the cache prefix, and where
tool-result turns come from. Point it at the capture proxy and the wire
evidence and the source that generated it sit side by side.
