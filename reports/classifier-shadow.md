# Classifier shadow — resolving the regex middle (UNIT B)

Offline harness. Does NOT touch live routing (no policy/hook import).
Classifier model: `qwen2.5:3b-instruct`

## Corpus & regex abstention

| Metric | Value |
|---|---|
| main-session user_turns (with text) | 754 |
| regex-uncertain (unknown verb OR score in [0.35, 0.7]) | 684 (90.7%) |
| adjudicated this run (ALL) | 684 |

Regex verb-class mix over the full corpus:

| verb_class | n | share |
|---|---|---|
| unknown | 655 | 86.9% |
| explain | 47 | 6.2% |
| trivial | 23 | 3.1% |
| fix | 17 | 2.3% |
| write | 8 | 1.1% |
| small_edit | 3 | 0.4% |
| hard | 1 | 0.1% |

## Reclassification of regex-uncertain turns

Fallback / unavailable (classifier returned None): 0/684 (0.0%)
Confidently resolved: 684/684 (100.0%)

| classifier tier | n | share of resolved | routes to |
|---|---|---|---|
| trivial | 370 | 54.1% | local |
| standard | 234 | 34.2% | local |
| hard | 80 | 11.7% | frontier |

Pulled to a confident LOCAL call: 605 (88.5% of resolved). Escalated to FRONTIER: 79 (11.5% of resolved).

## Outcome-proxy correlation (is it better than the middle coin-flip?)

Real hard-rate = share of turns with n_edit_failures>=1 OR n_error_results>=1 OR interrupted OR n_continuations>=10.

| Bucket | n | real hard-rate |
|---|---|---|
| classifier needs_frontier=True | 79 | 24.1% |
| classifier needs_frontier=False | 605 | 15.9% |
| all sampled uncertain (today's middle) | 684 | 16.8% |

Separation (frontier hard-rate - local hard-rate): +8.2 pts. A positive lift means the classifier's frontier calls concentrate the genuinely-hard turns better than the undifferentiated middle does.

## Verdict

- Resolves >=50% of regex-uncertain: PASS (100.0% resolved)
- Frontier calls track higher real hard-rate: PASS (24.1% vs 15.9%)

**Verdict: PASS** — the classifier confidently resolves the regex middle and its frontier calls are better than a coin-flip.

_Caveats: outcome_proxy is execution-friction, not intrinsic difficulty (non-validating; see classifier-bakeoff.md). Single-user, all-frontier source corpus. Sample is stratified by verb_class under a fixed seed; re-run with --all for the full population._
