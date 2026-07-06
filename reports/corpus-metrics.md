# Corpus Metrics — Phase 0 (A6/A7)

Turns: 5027 total (1254 main-session, 3773 subagent). All request counts are transcript-visible (lower bound — sidecar calls invisible; wire truth comes from workstream B).

## 1. Traffic shares (main sessions) vs design doc §4

| Request kind | Design §4 | Measured |
|---|---|---|
| Tool continuations | 70-90% | 73.3% |
| Utility/housekeeping (initiations) | 5-15% | 10.7% |
| New user turns (initiations) | 5-15% | 16.1% |
| Subagent turns (separate files) | occasional | 3773 turns, 27116 continuations |

## 2. Trip-wire signal frequencies (design §5.5), per user_turn

- Edit-apply failure (>=1): 1 (0.1%); 2-strike trip-wire: 0 (0.0%)
- Any is_error tool_result: 70 (9.3%)
- User interrupts (main sessions): 19
- Tool-use rejections: 0
- Turns with parallel tool calls (S12, per-rung registry flag): 2225 (44.3% of all turns)

## 3. Per-intent medians, user_turns (turn-budget thresholds, §5.5)

| Intent | n | median out-tokens | median duration (s) | median continuations |
|---|---|---|---|---|
| other | 685 | 2229 | 66 | 1 |
| explain | 44 | 2351 | 42 | 0 |
| write | 14 | 8368 | 146 | 3 |
| run | 7 | 6444 | 124 | 1 |
| edit | 2 | 18397 | 378 | 30 |
| fix | 2 | 39288 | 678 | 32 |

Design §5.5 turn-budget rule '2x median tokens or 90s per intent class' can now be instantiated from this table.

## 4. Token economics (observed)

| Model | turns | input | output | cache read | cache write | est. cost |
|---|---|---|---|---|---|---|
| claude-opus-4-8 | 3394 | 17,556,666 | 5,572,779 | 1,920,678,917 | 124,709,975 | $1966.88 |
| claude-fable-5 | 839 | 4,084,285 | 1,194,292 | 462,521,963 | 18,934,496 | $799.76 |
| claude-opus-4-7 | 268 | 46,222 | 1,138,336 | 613,459,166 | 31,882,585 | $534.69 |
| claude-haiku-4-5-20251001 | 68 | 18,389 | 48,353 | 38,203,928 | 2,217,755 | $6.85 |
| claude-sonnet-4-6 | 9 | 134 | 38,415 | 5,602,589 | 221,550 | $5.15 |
| unknown | 449 | 0 | 0 | 0 | 0 | $0.00 |

**Total estimated spend in corpus window: $3313.33.** Cache-hit ratio: 99.3% of prompt tokens served from cache — this is the asset mid-turn model switching would destroy (design §2).

## 5. Best-single-model baseline (A7, decision D1)

- Same token volumes re-priced entirely at **claude-opus-4-8** ($5.0/MTok in, $25.0/MTok out, cache read 0.1x, write 1.25x): **$2940.86**
- Naive local-routable candidates (user_turns with <=3 continuations, no errors, <2k out-tokens): 343/754 turns (45.5%), worth $108.41 at baseline prices (3.7% of baseline spend)

The Phase-3 learned router must beat this baseline on replayed traffic (design §10). The 'easy candidates' row is the ceiling on savings from the heuristic layer alone — a deliberately naive filter; the real policy engine should do better.

*Caveats: single-user corpus, workflow-skewed; transcripts under-count requests; unknown models priced as baseline; prices pinned 2026-07-06.*
