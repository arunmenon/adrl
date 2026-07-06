# Scenario Validation — corpus pass (A5)

Bar: >=3 real traces, or an explicit verdict. Raw traces (secrets included)
live under `data/scenario-matches/` — gitignored, local only (plan D5).

| Scenario | Matches | Status | Notes |
|---|---|---|---|
| S1 | 0 | wire-only — needs workstream B captures | sidecar utility burst |
| S2 | 104 | VALIDATED (5 traces extracted) | one turn, many requests (sticky hot path) |
| S3 | 6 | VALIDATED (5 traces extracted) | edit-apply trip-wire |
| S4 | 6 | VALIDATED (0 traces extracted) | identical-call loops |
| S5 | 0 | needs workstream C (Ollama traffic via B proxy) | malformed tool call (local model) |
| S6 | 15 | VALIDATED (5 traces extracted) | interrupt-then-rephrase |
| S7 | 0 | needs workstream C (induced endpoint failure) | infra fallback |
| S8 | 27 | VALIDATED (5 traces extracted) | easy intent inside huge context |
| S9 | 7 | VALIDATED (0 traces extracted) | secret exposure (privacy pin) |
| S10 | 1 | WEAK (1 < 3) — top up via B/C | hard direct-to-frontier intents |
| S11 | 0 | FALSIFIED for this setup — corpus is 100% Claude Code; defer per plan D4 | Codex dialect (absence check) |
| S12 | 2225 | VALIDATED (5 traces extracted) | parallel tool calls |
| S13 | 68 | VALIDATED (5 traces extracted) | subagent spawn (Task tool in main session) |
| S14 | 5 | VALIDATED (5 traces extracted) | auto-compaction |
| S15a | 0 | NO MATCHES — falsify or top up via B/C | episode boundary phrases |
| S15b | 1 | VIOLATED x1 — investigate | mid-turn model switch (invariant: must be ZERO) |

## S15b violation — investigated

The single "violation" is a workflow subagent turn showing `claude-fable-5, claude-opus-4-8`:
the provider-side Fable 5 -> Opus 4.8 fallback (Claude Code ships built-in fallbacks for
refusals/unavailability). This is a server-side switch with provider-guaranteed protocol
consistency — not a router-style mid-turn re-route. Invariant holds for routing purposes.
Design insight: provider-side fallback is the one legal mid-turn model change, precisely
because the provider owns bookkeeping consistency (cf. design doc §2, S15).

S15a note: the completion-phrase regex found 0 matches — this user's episode boundaries
don't use textbook phrases. The episode-boundary detector (design §5.3/S15) should not
weight explicit phrases heavily; intent-verb change + file-overlap signals matter more here.
