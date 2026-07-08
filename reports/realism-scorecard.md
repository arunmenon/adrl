# Realism scorecard — simulator vs corpus

Recomputes one metric battery on the real corpus and on simulator-labelled captures. Univariate bands come from the acceptance criteria in `reports/simulator-realism-plan.md`; **joint** bands are derived from the corpus (ground truth for joint structure) so a sim that is marginally right but jointly wrong still FAILS.

- corpus turns: **5027** (user-turn stream: 754)
- sim turns: **54** (user-turn stream: 54) from `data/sim-ledger.jsonl` joined to captures
- **aggregate realism score: 5/27 rows in band** (19%)
- **P0 gate (5/23 P0 rows PASS): NOT YET REALISTIC**
- seed: 0  ·  deterministic, re-runnable

## surface

| metric | corpus | sim | band | gate | verdict |
| --- | --- | --- | --- | --- | --- |
| word-count p50 | 10 | 12 | [8, 12] | P0-1 | PASS |
| word-count p90 | 72.4 | 23.7 | [58, 86] | P0-1 | **FAIL** |
| % <5 words | 24.4 | 11.1 | [18, 32] | P0-1 | **FAIL** |
| % no terminal punct | 64.7 | 81.5 | [55, 75] | P0-1 | **FAIL** |
| % all-lowercase | 26.1 | 57.4 | [20, 32] | P0-1 | **FAIL** |
| % typos/txt-speak | 22.0 | 9.3 | [16, 26] | P0-1 | **FAIL** |
| % inline file path | 11.8 | 3.7 | [8, 20] | P0-1 | **FAIL** |
| % fenced code | 0.265 | 0 | [0, 1] | P0-1 | PASS |
| % human emoji | 0.928 | 0 | [0, 1] | P0-1 | PASS |

## behavioral

| metric | corpus | sim | band | gate | verdict |
| --- | --- | --- | --- | --- | --- |
| % terse <=3 words | 17.6 | 11.1 | [14, 24] | P0-2 | **FAIL** |
| % questions | 38.3 | 25.9 | [28, 42] | P0-2 | **FAIL** |
| % voice disfluency (uh/um) | 12.5 | 5.6 | [10, 17] | P0-2 | **FAIL** |
| interrupt rate | 2.5 | 0 | [1.5, 4] | P0-2 | **FAIL** |
| median consec-turn Jaccard | 0 | 0 | [0, 0.100] | P0-2 | PASS |
| session length p50 | 23.5 | 2 | [18, 30] | P0-2 | **FAIL** |
| session length p90 | 136.4 | 6.3 | [100, 250] | P0-2 | **FAIL** |
| continuations/turn mean | 2.7 | 6.2 | [2, 4] | P0-2 | **FAIL** |

## structural

| metric | corpus | sim | band | gate | verdict |
| --- | --- | --- | --- | --- | --- |
| % source_kind=subagent | 75.1 | 0 | [50, 100] | P1-7 | **FAIL** |
| % ctx/msg >32k | 33.1 | 0 | [25, 100] | P1-6 | **FAIL** |
| % ctx/msg >100k | 22.7 | 0 | [15, 100] | P1-6 | **FAIL** |
| % sessions secret-shaped <br>_stress generator; >= planted fraction_ | 0 | 0 | [0.050, 100] | P1-8 | **FAIL** |

## failure

| metric | corpus | sim | band | gate | verdict |
| --- | --- | --- | --- | --- | --- |
| >=1 error / tool-turn (frontier) | 9.9 | 76.5 | [7, 11] | P0-3 | **FAIL** |
| >=2 errors / tool-turn (frontier) | 3.1 | 72.5 | [1.5, 3.5] | P0-3 | **FAIL** |
| 2-strike edit tripwire (local slice) <br>_stress generator; frontier floor 0.02%, local slice >=2%_ | 0.024 | 0 | [2, 100] | P0-3 | **FAIL** |

## joint

| metric | corpus | sim | band | gate | verdict |
| --- | --- | --- | --- | --- | --- |
| P(all-lowercase \| >=50 words) <br>_chimera guard_ | 0 | — | [0, 8] | P0-1/2 | no data |
| P(>=50w & lowercase & verbless) <br>_chimera guard_ | 0 | 0 | [0, 5] | P0-1/2 | PASS |
| archetype-mix TV distance <br>_joint distribution_ | 0 | 0.272 | [0, 0.150] | P0-1/2 | **FAIL** |
| max \|wc median dev\| by archetype (words) <br>_joint distribution_ | 0 | 18 | [0, 12] | P0-1/2 | **FAIL** |

