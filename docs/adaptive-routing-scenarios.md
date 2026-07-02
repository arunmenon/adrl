# Adaptive Routing Layer — 15 Scenario Walkthroughs

**Supporting document to:** `adaptive-routing-layer-design.md` (§ references point there)
**Status:** Draft v2 (synced with design doc v2 — see its §16 changelog) · **Date:** 2026-07-02

Each scenario traces real harness ↔ proxy handshakes through the routing layer, request by request. Wire payloads are **abbreviated and representative** (built from the public Anthropic Messages API and OpenAI Responses API specs plus harness docs as of July 2026). Before trusting any fingerprint rule in code, replace these with 5 real captures per scenario from your own capture proxy — that's an afternoon of work and it's Phase 0 anyway.

---

## How to read a trace

```
WIRE      what the harness actually sent (abbreviated JSON)
ROUTER    numbered steps inside the layer (discriminator → gates → heuristics → ...)
OUTCOME   what happened, what got logged
NUANCE    the one thing this scenario exists to teach
```

**Assumed rung config** (from design doc §5.4):

| Alias | Model class | Serving endpoints | Practical ctx | Notes |
|---|---|---|---|---|
| `local-code` | qwen3-coder-30b (Qwen3.6-class) | llama.cpp + MLX (two endpoints, one rung — design §5.4) | 32k | main local rung |
| `local-small` | 4B-class | llama.cpp | 16k | utility pinning target |
| `cheap-cloud` | Haiku-class | API | 200k | middle rung |
| `frontier` | Opus / GPT-5-class | API | 200k+ | top rung |

Model IDs follow the design-doc registry (§5.4), which is the single source of truth. A rung is a model class; LiteLLM picks and fails over among the endpoints serving it.

**Scenario index:**

| # | Group | Scenario | Teaches |
|---|---|---|---|
| S1 | Anatomy | Cold open: launching Claude Code | one keystroke ≠ one request; utility pinning |
| S2 | Anatomy | Rename variable: one turn, nine requests | the sticky hot path; KV-cache economics |
| S3 | Escalation | Fix failing test → edit trip-wire | escalation replay is a rebuild, not a resend |
| S4 | Escalation | The grep loop | no-progress detection; arg canonicalization |
| S5 | Escalation | Malformed tool call | dialect failure ≠ task difficulty |
| S6 | Escalation | User hits Esc and rephrases | the strongest quality signal |
| S7 | Escalation | llama.cpp falls over | infra fallback vs semantic escalation |
| S8 | Gates | 112k context, "write a commit message" | context cliffs invert easy=local |
| S9 | Gates | The .env leak that almost happened | privacy pin is one-way and structural |
| S10 | Gates | Cross-service refactor | direct-to-frontier; why not cascade everything |
| S11 | Dialects | Same task under Codex | Responses-API-only world; (model, harness) registry |
| S12 | Dialects | Parallel tool calls | parallelism is a harness–model negotiation |
| S13 | Dialects | Frontier delegates to a local subagent | the hybrid sweet spot; state inheritance |
| S14 | Dialects | Auto-compaction | quality-critical "housekeeping"; cache reset |
| S15 | Lifecycle | Coming back down + the forbidden mid-turn switch | hysteresis; why sticky is protocol correctness |

---

## Where the adaptive layer lives in these traces — crosswalk to the design doc

Every `ROUTER:` block in a trace **is** the adaptive layer acting; the wire snippets are what flows through it. Each behavior maps to a design-doc section:

| Behavior you see in traces | Scenarios | Design doc home |
|---|---|---|
| Utility fingerprinting + pinning | S1, S14 | §5.1 discriminator · §10 Phase 1 |
| Sticky continuation lookup (<1ms) | S2, every mid-turn request | §2 core idea · §5.6 session state |
| Full decision: gates → heuristics → learned | S1, S3, S10 | §5.3 policy engine |
| Hard gates (context, privacy, hysteresis) | S8, S9, S15 | §5.3 layer 0 · §7 state machine |
| Trip-wires + strike counters | S3, S4, S5, S6, S12 | §5.5 escalation controller |
| Transcript rebuild on escalation | S3, S11 | §5.5 steps 1–4 · §8.4 format gotchas |
| Capability registry lookups | S5, S11, S12 | §5.4 registry ((model, harness) pairs) |
| Infra fallback + route-state sync | S7 | §8.3 LiteLLM config · §12 risks |
| Episode boundaries, de-escalation | S15 | §5.3 hysteresis · §7 |
| Privacy pin inheritance (subagents) | S9, S13 | §5.6 session state · §7 |
| Flywheel outcome logging | all | §5.7 |
| Compaction routed by episode stakes | S14 | §5.1 (utility subtypes) |

Validation note (Phase 0 acceptance): for each scenario, find ≥3 real traces in the capture logs matching the pattern. A scenario with zero matches is wrong for this setup — fix the doc before writing the fingerprint code.

---

# Group A — Anatomy basics

## S1 · Cold open: launching Claude Code in a repo

**Setup:** User opens a terminal in `~/repo`, runs `claude`, types *"what does this repo do?"*
**Session state:** none (fresh).

### WIRE — what actually arrives (3 requests, not 1)

**Request 1 & 2 — background calls the user never sees.** Claude Code fires Haiku-class sidecar requests for housekeeping (terminal title, topic detection). It asks for whatever model ID `ANTHROPIC_DEFAULT_HAIKU_MODEL` resolves to (the old `ANTHROPIC_SMALL_FAST_MODEL` name is deprecated). If your proxy doesn't claim that ID, these 400 and Claude Code gets flaky:

```jsonc
POST /v1/messages
{
  "model": "claude-haiku-4-5",          // or your env-var override
  "max_tokens": 512,
  "system": [{"type":"text","text":"Analyze if this message indicates a new conversation topic..."}],
  "messages": [{"role":"user","content":"what does this repo do?"}]
}
```

**Request 3 — the real turn.** Big stable prefix, tiny human payload at the end:

```jsonc
POST /v1/messages
{
  "model": "adaptive-code",              // our alias — CC has no idea rungs exist
  "max_tokens": 32000, "stream": true,
  "system": [
    {"type":"text","text":"You are Claude Code, ...", "cache_control":{"type":"ephemeral"}},
    {"type":"text","text":"<env>cwd:~/repo git:main …</env>\n<CLAUDE.md contents>…",
     "cache_control":{"type":"ephemeral"}}          // Raschka's "stable prompt prefix"
  ],
  "tools": [ Bash, Read, Edit, Glob, Grep, WebFetch, Task, TodoWrite, … ],   // ~15 defs, ~10k tok
  "metadata": {"user_id": "a1f3…"},      // harness session identity — use it for session keying
  "messages": [{"role":"user","content":"what does this repo do?"}]
}
```

### ROUTER

1. **Req 1–2 → discriminator:** matches utility fingerprint (Haiku-class model ID + small `max_tokens` + known system-prompt prefix). → `model = "local-small"`. Never reaches policy. ~0.2ms.
2. **Req 3 → discriminator:** newest content is genuine human text, no active turn → `user_turn`.
3. **Gates:** no privacy pin, ctx ≈ 14k < 25.6k local budget, no escalation history → pass.
4. **Heuristics:** verb = *explain*, scope = repo-level summary, read-only intent, zero error history → score 0.22 < `T_EASY` → **local-code, cascade armed**.
5. Session created: `route=local-code, active_turn=t_001`, keyed on `metadata.user_id` + session hash.

### OUTCOME

Local model reads `README.md` + two source files (3 sticky continuations), answers. Fully local, $0.00, ~28s. Flywheel logs `{decision: heuristic/0.22, escalated: false}`.

> **NUANCE:** One keystroke produced 3+ requests, and two of them weren't "the task" at all. Utility pinning requires you to (a) map the Haiku-class model ID in LiteLLM so background calls don't 400, and (b) fingerprint them in the discriminator. This is the zero-risk Phase-1 win — and if you skip it, your router will earnestly run difficulty heuristics on *terminal title generation*.

---

## S2 · "Rename `usr_cnt` to `user_count`" — one turn, nine requests

**Setup:** Same session, turn 2. A textbook easy task.
**Session state:** `route=local-code`, healthy.

### WIRE — the shape of a turn

```
req 1   messages[..., {"role":"user","content":"rename usr_cnt to user_count everywhere"}]
        ← assistant: tool_use Grep {pattern:"usr_cnt"}                    (id: toolu_L1)
req 2   messages[..., {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_L1",
        "content":"src/stats.py:14: usr_cnt = 0\nsrc/stats.py:29: return usr_cnt\n…"}]}]
        ← assistant: tool_use Read {file_path:"src/stats.py"}             (toolu_L2)
req 3   … tool_result(toolu_L2) …  ← tool_use Edit {old_string:"usr_cnt = 0", new_string:"user_count = 0"}
req 4-8 … Edit ✓, Edit ✓, Bash("pytest -q") ✓ …
req 9   … tool_result(pytest passed) … ← assistant: "Renamed in 3 places, tests pass."
```

Requests 2–9 all end in a `tool_result` block. **That's the tell.**

### ROUTER

- **Req 1:** full decision path (like S1) → local, cascade armed. ~12ms.
- **Req 2–9:** discriminator sees `tool_result` tail + `active_turn=t_002` → `continuation` → **one state lookup, `model = session.route`, done.** <1ms each. No features, no policy, no ML — eight times in a row.

### The cache math that stickiness protects

llama.cpp keeps the session's token prefix in a slot cache. Req 2 shares ~99% of its prefix with req 1, so prefill only pays the delta (the new tool_result). If the router re-decided per request and flapped local↔cloud even once mid-turn:

| | prefix warm (sticky) | prefix cold (after a flap) |
|---|---|---|
| prefill work at req 5 (~18k tok ctx) | ~400 new tokens | full 18k re-prefill |
| wall-clock on M-series (~900 tok/s prefill) | <1s | **~20s before the first output token** |

Same story on the cloud side: Anthropic's prompt cache (`cache_control` breakpoints on system+tools) makes repeat input ~90% cheaper — but only if consecutive requests keep hitting the same provider.

> **NUANCE:** This is design-doc §2 made concrete: ~85% of traffic must travel a code path that is one dict/Redis lookup long. If your router's median added latency isn't measured in *microseconds* for continuations, you built it wrong.

---

# Group B — The escalation family

## S3 · "Fix the failing test" — the edit trip-wire (flagship scenario)

**Setup:** Turn 6. `pytest` is red. Medium-difficulty on paper.
**Session state:** `route=local-code`, `strikes={}`.

### WIRE + ROUTER interleaved

```
req 1   user: "fix the failing test"
        ROUTER: user_turn → gates pass (ctx 18k) → heuristic 0.48 (medium band)
                → local-code, cascade ARMED, route_id r_1041
req 2   ← tool_use Bash {command:"pytest -x -q"}   → tool_result: "FAILED test_parse_dates …"
req 3   ← tool_use Read {file_path:"src/parse.py"} → tool_result: <file body>
req 4   ← tool_use Edit {file_path:"src/parse.py",
                 old_string:"fmt = '%Y-%m-%d'\n    return datetime.strptime(s, fmt)",
                 new_string:"fmt = '%Y-%m-%d'\n    return datetime.strptime(s.strip(), fmt)"}
        → tool_result (is_error): "String to replace not found in file."
        ROUTER post-call: strike edit_apply = 1
req 5   ← tool_use Edit {…old_string differs by whitespace again…}
        → tool_result (is_error): "String to replace not found in file."
        ROUTER post-call: strike edit_apply = 2  →  ⚡ TRIP-WIRE (threshold 2, §5.5)
```

Why local models hit exactly this wall: Claude Code's `Edit` demands a **byte-exact `old_string` match**. The local model regenerated the snippet from memory — tabs vs spaces — instead of copying from the `Read` result. Frontier models are heavily post-trained on this harness dialect; 30B locals are not.

### ESCALATION — the rebuild (never a resend)

1. **Freeze at the action boundary** — req 5's tool_result is in; nothing in flight.
2. **Rebuild the transcript for the frontier:**
   - strip the local model's `thinking` blocks — Anthropic validates thinking-block **signatures**; locally generated ones won't verify, and foreign reasoning styles mislead anyway;
   - re-key every `tool_use`/`tool_result` pair (`toolu_L*` → fresh consistent IDs) — each provider validates its own ID space;
   - keep tool_results verbatim (they're ground truth), re-apply `cache_control` breakpoints for the new provider.
3. **Append the 3-line failure note** (§5.5 design decision):
   ```
   [routing note] A previous attempt: identified test_parse_dates failure, read src/parse.py,
   then failed twice to apply an Edit (old_string mismatch — whitespace).
   The diagnosis (strip input before strptime) may be correct; the edit was not.
   ```
4. Reissue to `frontier`. Set `escalated_this_episode=true` (hysteresis).
5. **Persist the scrub.** The harness's own client-side transcript still holds the local-era artifacts (old tool IDs, translated reasoning) and resends them with *every* subsequent request — so the ID map is saved to session state (`tool_id_map`) and step 2's scrubbing re-runs on every post-escalation request until the episode ends (design §5.5). A one-shot rebuild would 400 on the very next continuation.

### OUTCOME

Frontier re-reads the file (its choice), applies the edit on attempt 1, reruns pytest ✓. Flywheel row: `{tripwire:"edit_apply", local_tokens_wasted:3800, wasted_wall_clock:41s, frontier_completed:true}`. Cost of the failed local attempt: $0 and 41 seconds — that 41s is the real price, which is why `T_HARD` exists (see S10).

> **NUANCE:** Escalation = freeze → scrub → re-key → annotate → reissue. Get any of the middle steps wrong and the frontier call 400s (bad tool IDs), rejects (unsigned thinking), or wastes tokens re-walking the same dead end (missing failure note).

---

## S4 · The grep loop — valid actions, zero progress

**Setup:** *"why is the rate limiter never triggering?"* — local model starts investigating.

### WIRE + ROUTER

```
req 2  ← tool_use Grep {pattern:"RateLimiter", path:"src/"}        → 14 matches
req 3  ← tool_use Grep {pattern:"RateLimiter", path:"src"}         → same 14 matches
req 4  ← tool_use Grep {pattern:"RateLimiter",  path:"src/"}       → same again
        ROUTER post-call: identical-call strike = 3  →  ⚡ TRIP-WIRE
```

The three calls are **not byte-identical** — trailing slash, whitespace. The detector must canonicalize before hashing: `hash(tool_name + json.dumps(args, sort_keys=True, normalized_paths=True))`. Without canonicalization, the loop detector misses every "cosmetically different" loop, which is most of them.

The backstop is the **no-progress counter** (design §5.5): across the last 6 actions — no file read that wasn't read before, no diff advanced, no new command output hash. That catches the subtler spiral where the model alternates between two greps and a read, forever.

### OUTCOME

Escalate with failure note *"attempted repeated identical searches; no hypothesis formed."* Frontier reads the config file the greps kept pointing at, finds the limiter is behind a disabled feature flag, answers in 4 calls.

> **NUANCE:** Every action was schema-valid — a per-request validator would have said "all good." Progress is a *trajectory* property. This trip-wire is the single best argument for session state: you cannot detect a loop if you can't remember the last request.

---

## S5 · The malformed tool call — dialect failure, not difficulty

**Setup:** Easy-band task ("add a `--verbose` flag"). Local rung recently switched to a new GGUF build.

### WIRE

The model's answer comes back with the tool call **as text**, not as a structured block:

```jsonc
// assistant message content:
[{"type":"text","text":"I'll read the CLI entrypoint first.\n<tool_call>\n{\"name\": \"Read\", \"arguments\": {\"file_path\": \"src/cli.py\"}}\n</tool_call>"}]
```

llama.cpp's chat template (`--jinja`) didn't map the model's native tool-call tokens to the OpenAI `tool_calls` field, so LiteLLM translated a *text* answer back to Claude Code — which sees a model that talked about calling a tool and never did. Harness nudges; model does it again. Two parse strikes → trip-wire.

### ROUTER — the labeling decision that matters

```
tripwire = "schema_parse"  →  outcome routed to the REGISTRY, not the difficulty model:
   registry["local-code"].tool_call_reliability["claude_code"]  0.87 → 0.79 (rolling)
   flywheel row tagged label_quality="dialect_failure"  →  EXCLUDED from difficulty training
```

Escalate the turn (user comes first), but the *fix* is operational: repin the chat template / correct `--jinja` mapping / disable the offending quant. If reliability recovers in canary evals, the rung resumes. If this failure had been labeled "task too hard," the learned router would slowly conclude that *adding CLI flags* requires a frontier model.

> **NUANCE:** Design-doc pitfall #5 in the wild. Trip-wire *type* must travel with every outcome record precisely so dialect failures train the registry and difficulty failures train the router. One `tripwire` enum field is the difference between a flywheel and a garbage compactor.

---

## S6 · Esc — the user rephrases

**Setup:** Local model is 40s into a meandering attempt at *"make the retry logic more robust"*. User hits **Esc**, types *"just add exponential backoff with jitter to fetch_with_retry"*.

### WIRE

```
req N   (previous turn's messages now end with:)
        {"role":"assistant","content":[…partial…]},
        {"role":"user","content":"[Request interrupted by user]"}       ← harness marker
req N+1 {"role":"user","content":"just add exponential backoff with jitter to fetch_with_retry"}
```

### ROUTER

1. Discriminator: `user_turn`, and the previous turn carries the **interrupted** marker.
2. Feature extractor: high lexical overlap with previous turn's intent + interruption ⇒ `retry_signal=true`.
3. Policy: **escalate-on-retry rule** — regardless of heuristic score (this one scores *easier* than the original!), bump one rung: → `frontier` (or cheap-cloud if you run three rungs). Also: log a hard negative outcome for route `r_previous`.

### OUTCOME

Frontier does it in 3 calls. Two flywheel rows: the interrupted turn (negative label, high confidence) and the retry (positive).

> **NUANCE:** An interrupt-then-rephrase is the only label in the whole system that comes straight from the human, in-band, within seconds. It outranks every heuristic — including "this looks easy." The user just told you the local model was annoying them; believe them, and *bank the label*, because these are rare (design §5.7's noisy-label warning: this is the one clean signal you get).

---

## S7 · llama.cpp falls over — infra fallback ≠ semantic escalation

**Setup:** Mid-turn (req 4 of 9), the MLX/llama.cpp server OOMs under a parallel workload and the socket resets.

### What fires — and what doesn't

```
req 4 → local-code:  APIConnectionError (connection refused)
        LiteLLM router_settings.fallbacks: local-code → cheap-cloud   ← INFRA path (§8.3)
        retried request served by cheap-cloud ✓
```

No strikes. No trip-wire. The escalation controller never wakes up — nothing was *semantically* wrong. But two router responsibilities trigger:

1. **State sync (the classic bug):** the post-call hook sees `fallback_used=cheap-cloud` and must update `session.route=cheap-cloud`. If it doesn't, req 5's sticky lookup routes back to the dead endpoint, fails, falls back again — every subsequent request eats a timeout before succeeding.
2. **Health masking:** LiteLLM health checks mark `local-code` unhealthy; the policy engine treats unhealthy rungs as nonexistent for *new* turn decisions — this is now a layer-0 gate (design §5.3), not an afterthought.
3. **Endpoints before rungs:** `local-code` is a model class that may be served by several endpoints (llama.cpp *and* MLX on the same box — design §5.4). The correct first fallback is the same model on another endpoint; crossing to cheap-cloud is the second resort. A dead server is not a reason to change brains.

Also note what the fallback *cannot* do: the cheap-cloud provider now cold-starts the whole prefix (no shared cache with llama.cpp), so this request is slow and full-price. Expected; log it as infra cost, not model quality.

> **NUANCE:** Two ladders exist side by side. **Exceptions → LiteLLM fallbacks. Bad-but-valid responses → our trip-wires.** They share one session store, and the store must always reflect *where the last response actually came from*, not where the router intended it to go.

---

# Group C — Gates and constraints

## S8 · 112k-token session asks for a commit message

**Setup:** Turn 23 of a long frontier session. User: *"write a commit message for what we just did"*. Trivially easy intent — and the context is 112k tokens.

### ROUTER

```
1. discriminator: user_turn
2. GATE (feasibility): 112k > 0.8 × 32k local ctx → local rung STRUCTURALLY OUT
   (never mind that heuristic score would be 0.08, the easiest thing all day)
3. policy fallthrough: cheapest feasible rung = cheap-cloud (200k ctx) → direct, no cascade
```

Why not "just truncate to fit local"? Because *which* 26k of the 112k tokens? The diff is spread across the transcript; naive tail-truncation loses the early design decisions the message should mention. Compression *changes the task* (design pitfall #9).

**The Phase-3 upgrade (flagged, not shipped in v1):** context-carving — recognize intents whose true context is computable and small: a commit message needs `git diff --staged` + the last user goal, ~2k tokens. The router synthesizes a *fresh* minimal request and sends *that* to local. Powerful (this class of turn is common in long sessions), but it means the router now edits prompts, with its own failure modes — keep it out of v1, let the flywheel first prove how much money it would save.

> **NUANCE:** Easy-task-inside-huge-context is the case that breaks "difficulty routing" mental models. Feasibility gates outrank difficulty — and the honest v1 answer is "cheap cloud," not clever truncation.

---

## S9 · The .env leak that almost happened

**Setup:** Turn 3: *"why does the app crash on startup?"* Local model investigates.

### WIRE + ROUTER

```
req 3  ← tool_use Read {file_path:".env"}
       → tool_result: "DATABASE_URL=postgres://admin:S3cr3tPw@prod-db:5432/app
                        AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY…"
       ROUTER post-call scanner (regex + entropy on tool_results, ~1ms):
           match: AWS key pattern + connection-string credential
           → session.privacy_pinned = TRUE   (one-way, §7 state machine)
```

Turn 7, same session: *"refactor the config loading to support multiple environments"* — genuinely hard, heuristic 0.74 > `T_HARD`, wants frontier.

```
GATE order (§5.3, layer 0 first — by construction):
   privacy_pinned → Route(local, cascade=False, pinned=True)
   difficulty logic NEVER RUNS. There is no code path where "it looked hard" outranks the pin.
```

Mid-turn, the local model struggles — edit strike 2 — trip-wire fires. **Escalation controller checks the pin and refuses.** Instead, surface to the user (harness-visible message):

> *"This session touched credentials at turn 3, so it's pinned to the local model. Options: (a) continue locally — I'll break the task into smaller steps; (b) start a fresh session without the secret in context; (c) explicitly override the pin."*

(Mechanics: pinned routes are `cascade=True` with the escalation *target* set to the user rather than a higher rung — design §5.3/§5.5. And when a pinned session's context outgrows the local rung entirely, design §5.8 defines the ladder: early warning at 60% of local context → forced local compaction at 75% → hard stop with these same three options.)

Why not just redact and escalate? Because the secret **propagates**: it's quoted in the tool_result, echoed in the model's turn-3 answer ("your DATABASE_URL points at prod…"), and may appear in later command output. Reliable retroactive redaction of a 40k-token transcript is a research problem; a one-way session pin is an `if` statement.

> **NUANCE:** The pin must be *structural* (gate layer 0, escalation-controller check) — not a score penalty, not a strong preference. A leaked secret is an incident; the design must make it unrepresentable, and the metric for it (design §11) is an alarm with target **zero**.

---

## S10 · "Refactor auth to the new session service across api/, worker/ and cli/"

**Setup:** Fresh episode. Multi-service refactor + migration. The heuristic layer sees: verb=*refactor*, scope=*across* + 3 top-level dirs, migration keyword, 6 files already in context → score 0.86 > `T_HARD`.

### ROUTER

```
→ Route(frontier, cascade=False)      # direct. No local attempt. Done in 9ms.
```

### Why not cascade "just in case"? The arithmetic:

Registry says p(local completes a 0.86-scored turn without trip-wire) ≈ 0.15. Local attempt before trip-wire ≈ 60–90s wall-clock (illustrative medians from your own flywheel).

| Strategy | Expected cost |
|---|---|
| cascade (local first) | 0.85 × (75s wasted + frontier full cost) + 0.15 × (local, free) ≈ **64s user latency tax + 85% of frontier cost anyway** |
| direct to frontier | frontier cost, zero wasted latency |

Cascading is only rational where p(local success) is decent — that's *the entire job* of `T_HARD`. And these hard turns are exactly where users are least tolerant of a 60-second wrong-model detour (pitfall #6: cascades double-pay, and they pay in the currency users care about).

**Bonus economics — hysteresis is self-reinforcing (S15 preview):** once this episode is on frontier, Anthropic's prompt cache holds the 24k-token system+tools+transcript prefix; turns 2–8 of the episode pay ~10% input price. Yo-yoing rungs between turns would re-cold-start both caches every time.

> **NUANCE:** "Adaptive" does not mean "always try cheap first." A good router's most valuable decisions are the ones where it *declines* to use the local model.

---

# Group D — Harness dialects (where July 2026 specifics bite)

## S11 · The same "fix the failing test" — under Codex CLI

**Setup:** S3's task, but the user runs Codex. Everything about the handshake changes.

### WIRE — Codex speaks Responses API, and *only* Responses API

OpenAI deprecated and **removed the chat-completions wire from Codex in February 2026** — `wire_api = "responses"` is now the only value. (Every 2025-era guide showing `wire_api = "chat"` for local providers is broken; this is reportedly the #1 source of dead Codex+local configs.)

```jsonc
POST /v1/responses
{
  "model": "adaptive-code",
  "instructions": "You are Codex, …",          // system prompt lives here, not in input
  "input": [
    {"type":"message","role":"user","content":[{"type":"input_text","text":"fix the failing test"}]},
    {"type":"reasoning","encrypted_content":"gAAAAABo…"},      // OPAQUE — provider-encrypted
    {"type":"function_call","name":"shell",
     "arguments":"{\"command\":[\"bash\",\"-lc\",\"pytest -x -q\"]}","call_id":"call_9"},
    {"type":"function_call_output","call_id":"call_9","output":"FAILED test_parse_dates…"}
  ],
  "tools": [{"type":"function","name":"shell"}, {"type":"function","name":"apply_patch"}, …],
  "store": false, "include": ["reasoning.encrypted_content"], "stream": true
}
```

Three consequences for the stack:

1. **Translation is now mandatory, not optional.** llama.cpp/MLX serve chat-completions; Codex sends `/v1/responses`. LiteLLM sits in between translating Responses ⇄ chat for the local rung. Budget real testing here — this seam is young and function-call fidelity through double translation is exactly where S5-style parse failures breed.
2. **Encrypted `reasoning` items cannot cross rungs.** They're provider-opaque (that's the point of `encrypted_content`). On any escalation or fallback that changes provider, the rebuild (S3 step 2) must drop them entirely. Keeping them = instant 4xx.
3. **The edit dialect flips.** Codex edits via `apply_patch` (unified-diff-ish patches), not exact-string `Edit`. Your registry (design §5.4) already prices this: the same Qwen local model scores 0.93 on `apply_patch` vs 0.87 on Claude Code's string-match — patches tolerate the whitespace drift that killed S3. Raschka's evals found the same direction: Qwen3.6 performed *better* inside Codex than in its own native harness.

### ROUTER differences

- Session keying: Codex session id from originator headers/config, not `metadata.user_id`.
- Discriminator tells: continuation = trailing `function_call_output` item; utility = Codex's own compaction calls.
- Policy: same turn scores **medium under Claude Code but easy-ish under Codex** because `p(edit success | apply_patch)` is higher → `T_EASY`/`T_HARD` bands are per-harness values, not global constants.

> **NUANCE:** "One router" is really a family of dialect adapters around one policy. Anything that touches message structure — discriminator tells, transcript rebuilds, registry reliability numbers, even thresholds — is keyed by harness. Store `harness` in the session state at first contact and never re-detect it mid-session.

---

## S12 · Parallel tool calls — a harness–model negotiation

**Setup:** Investigation turn: the model wants to read three files.

### What frontier does (Claude Code)

```jsonc
// one assistant message, THREE tool_use blocks:
[{"type":"tool_use","id":"toolu_F1","name":"Read","input":{"file_path":"src/a.py"}},
 {"type":"tool_use","id":"toolu_F2","name":"Read","input":{"file_path":"src/b.py"}},
 {"type":"tool_use","id":"toolu_F3","name":"Grep","input":{"pattern":"configure\\("}}]
// harness executes all three, replies with ONE user message carrying THREE tool_results
```

One round-trip, three results. The harness happily batches because the model emitted well-formed parallel blocks.

### What the local rung does

Registry: `parallel_tool_calls: false` for `local-code` (evals showed ID mismatches when the 30B model emits multiple blocks — orphaned `tool_result`s, which the API rejects). The llama.cpp template is configured single-call. Same investigation:

```
req a: tool_use Read a.py  → result     req b: tool_use Read b.py → result     req c: Grep → result
```

Three round-trips instead of one. Each is cheap, but **turn latency ≈ 3× the round-trip overhead** — and the router's wall-clock predictor (S8, design §12) must price this in: a "medium fan-out" investigation turn on local costs more *time* than its token count suggests.

The failure mode if you don't enforce this: the local model emits two `tool_use` blocks, one with a malformed ID → the harness's paired `tool_result` references an ID the API never saw → hard 400 → looks like an infra failure (S7) but is actually a dialect failure (S5). Confusing trifecta; the registry flag prevents it from ever starting.

> **NUANCE:** Parallelism isn't a model capability *or* a harness feature — it's a negotiation between them, and the router referees it via the registry. Disable it per-rung until your eval pack proves it, and charge the sequential latency tax in routing decisions.

---

## S13 · Frontier delegates to a local subagent

**Setup:** Escalated episode (post-S3), frontier is mid-refactor and needs a side answer.

### WIRE

```jsonc
// frontier's main thread emits:
{"type":"tool_use","id":"toolu_F9","name":"Task",
 "input":{"subagent_type":"Explore","prompt":"Find every place RateLimiter is configured or constructed, list file:line"}}
```

Claude Code now opens a **brand-new API conversation**: fresh (smaller) agent-specific system prompt, empty history, read-only tool set, its own request stream. From the proxy's viewpoint this is a new session that happens to start mid-someone-else's-turn.

### ROUTER

1. **Discriminator:** fresh system prompt + subagent markers → `subagent`, new session key `s_17.sub_3`, **linked to parent**.
2. **Inheritance rule:** child inherits the parent's `privacy_pinned` flag (it can read the same files!) — but *not* the parent's escalation state. Clean slate on difficulty.
3. **Policy:** read-only scope, fresh 3k context, bounded task → textbook local: `local-code`, cascade armed.
4. Subagent runs 5 tool calls locally, returns a file:line list; harness folds it into the parent turn as `tool_result(toolu_F9)`; parent (frontier, sticky) continues.

**Why this is the hybrid sweet spot:** the frontier model does the thinking; grunt-work sub-lookups run free and local. Raschka's "delegation with bounded subagents" component becomes, in routing terms, *the frontier model generating perfectly-shaped local workloads* — small context, clear goal, read-only. Sessions like this can push a majority of raw token volume to the local rung even while every "hard thought" stays frontier.

> **NUANCE:** Two session-keying traps: (1) if the subagent's conversation hashes into the *parent's* key, its 5 requests corrupt the parent's strike counters and turn state; (2) if it gets a fully independent key, it silently *loses the privacy pin*. You need linked-but-separate: own state, inherited constraints.

---

## S14 · Auto-compaction — housekeeping that isn't trivial

**Setup:** The escalated session crosses ~92% of the context window. Claude Code auto-compacts.

### WIRE

```jsonc
POST /v1/messages
{ "model": "adaptive-code",
  "system":[{"type":"text","text":"You are a helpful AI assistant tasked with summarizing conversations…"}],
  "messages":[ …the ENTIRE 150k-token transcript…,
    {"role":"user","content":"Summarize this conversation, preserving: current task state, files modified, key decisions, pending work…"}],
  "max_tokens": 8000 }
```

It *smells* like a utility call (harness-generated prompt, no human text). But it is **not** S1's title generation:

1. **The output becomes the episode's memory.** Every subsequent turn of this *frontier-grade* episode reasons from this summary. A weak local summary that drops "we decided to keep the legacy API behind a flag" quietly poisons every later decision — you'd never trace the regression back to the compactor.
2. **Routing rule:** compaction quality must match the *episode's* rung, not "utility": frontier episode → `cheap-cloud` minimum (a Haiku-class model summarizes reliably; local-small does not, at 150k input it can't even fit). Local episode → local is fine. So the discriminator labels it `utility:compaction` and policy branches on `session.current_rung`.
3. **Cache reset follows.** Post-compaction, the prompt prefix is brand new: provider prompt cache cold, llama.cpp slot cold. The next 1–2 requests are slow and full-price *by design*. Two router adjustments: exempt the first post-compaction request from wall-clock trip-wire budgets, and don't let the flywheel record the latency spike as rung slowness.

> **NUANCE:** "Housekeeping" is a spectrum: titles are throwaway (pin to local-small forever); compaction is memory-critical (route by episode stakes). One `utility` bucket is one bucket too few — this is the sole place the v1 discriminator needs a second label.

---

# Group E — Lifecycle

## S15 · Coming back down — and why you never switch mid-turn

### Part A — hysteresis releases at an episode boundary

Post-S3, the session is `escalated_this_episode=true`; turns 7–9 (all refactor follow-ups) stay frontier via the layer-0 gate. Then:

```
turn 10: "great, that's done. now write a README for the new config module"
```

**Episode-boundary detector** (all cheap, all local signals):

| Signal | Value here |
|---|---|
| Previous turn ended clean (no strikes, success summary) | ✓ |
| Explicit completion phrase in user text ("that's done", "great, now…") | ✓ |
| New intent verb class (write-new vs refactor) | ✓ |
| File-overlap with episode's touched-set | low (README vs src/) |

Three of four → **new episode**. Hysteresis gate releases, heuristic scores 0.18 → back to `local-code`. The state machine (design §7) walks `FrontierSticky → LocalSticky` through the only door that exists: an episode boundary plus an easy turn. When the detector is unsure, it stays up — a wasted frontier turn costs cents; a premature de-escalation costs an S3 replay.

### Part B — the counterfactual: re-deciding on a continuation

Suppose the router "optimized" req 5 of an in-flight frontier turn down to local. Concretely, in order of failure:

1. **Protocol break.** The pending `tool_result` references `toolu_F2` — an ID minted by the other provider. The local endpoint (or strict translation layer) has never seen it:
   `400: unexpected tool_use_id in tool_result block`.
   The Anthropic side breaks symmetrically on foreign IDs, and additionally rejects replayed `thinking` blocks whose **signatures** don't verify. Codex's encrypted `reasoning` items are unreadable by anyone else, period (S11).
2. **Cache demolition.** Even if you re-keyed everything mid-turn: llama.cpp must prefill the full 60k-token prefix cold (≈ 60–70s before token one, S2 math), and when you hop back, the provider-side prompt cache has to rebuild too. You paid twice to go nowhere.
3. **Cognitive whiplash.** The local model inherits half-executed frontier reasoning — a plan it didn't make, referencing conclusions it can't verify — and reliably does the thing trip-wires exist to catch: flails convincingly.

This is why "sticky" isn't a tuning choice that Phase 3 ML might learn to override. It's **protocol correctness first, economics second, model quality third** — three independent reasons, any one of which suffices.

> **NUANCE:** The router has exactly two legal movement moments: a new turn (policy) and a trip-wire at an action boundary (controlled rebuild). Everything else in the request stream is untouchable. If you remember one sentence from all fifteen scenarios, make it that one.

---

## Coverage matrix — what each scenario exercises

| | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 | S13 | S14 | S15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Discriminator (call types) | ● | ● | | | | ● | | | | | ● | | ● | ● | |
| Sticky hot path | | ● | ● | | | | ● | | | | | ● | | | ● |
| Hard gates (L0) | | | | | | | | ● | ● | | | | | | ● |
| Heuristics (L1) | ● | ● | ● | | | | | | | ● | ● | | ● | | ● |
| Trip-wires | | | ● | ● | ● | ● | | | ● | | | ● | | | |
| Escalation rebuild | | | ● | ● | ● | | | | | | ● | | | | ● |
| Registry / dialects | | | ● | | ● | | | | | | ● | ● | | | |
| Session state / keying | ● | ● | | ● | | | ● | | ● | | ● | | ● | | ● |
| Privacy | | | | | | | | | ● | | | | ● | | |
| Infra fallback (LiteLLM) | | | | | | | ● | | | | | ● | | | |
| Flywheel / labels | ● | | ● | | ● | ● | ● | | | ● | | | | ● | |
| Cache economics | | ● | | | | | ● | | | ● | | | | ● | ● |
| Wall-clock / latency | | ● | ● | | | | | ● | | ● | | ● | | ● | |

Every design-doc component appears in ≥2 scenarios; every scenario earns its place with ≥1 nuance no other scenario covers.

---

## Edge cases deliberately out of scope for v1 (log them, don't solve them)

- **Two terminals, one repo:** two harness sessions editing the same workspace — session keying handles it, but flywheel labels get muddied (whose edit made the tests pass?). Punt.
- **MCP servers in the harness:** user-configured MCP tools change the tool list per user → tool-set hash becomes part of the registry key. Punt until it appears in your captures.
- **Streaming aborts mid-SSE:** client disconnects mid-stream — distinguish from S6 interrupts via harness marker presence. Log only.
- **Harness version drift:** a Claude Code/Codex update changes a fingerprint (new system-prompt wording, new background call) and the discriminator silently misclassifies. Mitigation now, not later: a canary test in CI that replays 20 stored handshakes against the discriminator and alarms on any label change.
- **Git worktrees / monorepo subdirs:** workspace identity vs repo identity for repo-level features. Punt.

---

## Sources

- Raschka — [Components of a Coding Agent](https://magazine.sebastianraschka.com/p/components-of-a-coding-agent) · [Using Local Coding Agents](https://magazine.sebastianraschka.com/p/using-local-coding-agents) (harness token profiles, Qwen-in-Codex results, local speed data)
- Claude Code — [model configuration & background-model env vars](https://code.claude.com/docs/en/model-config) · proxy mapping guides: [claude-code-proxy](https://github.com/1rgs/claude-code-proxy), [enterprise model mapping](https://medium.com/@trevor00/claude-code-in-the-enterprise-model-mapping-for-llm-proxies-b0d8069c6aa3)
- Codex — [advanced config / wire_api](https://developers.openai.com/codex/config-advanced) · [chat-completions deprecation discussion](https://github.com/openai/codex/discussions/7782) · [provider config guides](https://www.morphllm.com/codex-provider-configuration) · [Ollama integration](https://docs.ollama.com/integrations/codex)
- LiteLLM — [fallbacks & routing](https://docs.litellm.ai/docs/routing) · [call hooks](https://docs.litellm.ai/docs/proxy/call_hooks)


