# ADRL learning-readiness scorecard

Generated: `2026-07-13T18:07:08.517975+00:00`
Snapshot: `readiness:5fb2f4f45864a9ce69be337378e5811838509e0841f18bb532471f5da11fb5c2`

## Verdict

- Production readiness: **BLOCKED**
- Architecture evidence index: **42.8/100**
- Frozen baseline: **40.5/100** (`taxonomy-v1-freeze-a46ca8c` at `a46ca8ca`)
- Change from baseline: **+2.3 points**
- Scoring contract: `architecture-evidence-index-v1` (`sha256:d26f5175f43a66aac99d80887c88de81ac6683d39127c53553a3d859df677bb4`)
- The index summarizes checked-in ADR D-levels; it is not a production-readiness score.
- Scores without this exact contract version and hash are not part of this comparable series.
- Blocking gates: organic_verifier_labels, label_precision, learned_router_authority

## Method

- Maturity points: D0=0, D1=20, D2=40, D3=60, D4=80, D5=100.
- Points, bucket weights, decision weighting, and rounding come from the versioned scoring contract.
- Under v1, each decision has equal weight inside its bucket and each frozen bucket has weight 1/9.
- Confidence is exposed as each bucket's D-level evidence profile below.
- Production authority is a hard-gate verdict and is never inferred from the arithmetic mean.
- Synthetic pairs may pass integrity checks but cannot pass the organic representative-data gate.

## Bucket evidence

| Bucket | Index | Weight | Evidence profile |
|---|---:|---:|---|
| `FND` | 60.0 | 0.111 | D2=1, D3=3, D4=1 |
| `SEM` | 43.3 | 0.111 | D0=1, D2=2, D3=3 |
| `SAF` | 42.9 | 0.111 | D2=6, D3=1 |
| `RTG` | 45.0 | 0.111 | D0=1, D2=3, D3=4 |
| `CAS` | 45.7 | 0.111 | D2=5, D3=2 |
| `MEM` | 42.2 | 0.111 | D2=8, D3=1 |
| `LRN` | 22.9 | 0.111 | D0=3, D2=4 |
| `EVL` | 35.6 | 0.111 | D0=3, D2=2, D3=4 |
| `OPS` | 47.5 | 0.111 | D1=1, D2=5, D4=2 |

## Learning evidence

| Measure | Count |
|---|---:|
| Organic decisions | 763 |
| Finalized outcomes | 790 |
| Eligible ambiguous-middle decisions | 84 |
| Ledger rows with any verification | 1 |
| Valid organic verification contexts | 1 |
| Cause-clean organic verifier labels | 1 |
| Usable finalized middle-band labels | 0 |
| Valid same-snapshot local/frontier pairs | 1 |
| Production-representative pairs | 1 |
| Valid versioned counterfactual records | 2 |
| Invalid counterfactual records | 0 |
| Invalid verification contexts | 0 |

## Gates

| Gate | Status | Evidence |
|---|---|---|
| `feature_contract` | **PASS** | 0 ledger rows violate the versioned schema |
| `organic_verifier_labels` | **BLOCKED** | 0 cause-clean, finalized, eligible organic labels |
| `classifier_provenance` | **NOT_EVALUATED** | 0/0 usable labels have complete provenance |
| `label_precision` | **BLOCKED** | 1 valid independent reviews under label-precision-v1 |
| `paired_counterfactual_integrity` | **PASS** | 1 valid local/frontier same-snapshot pairs |
| `representative_pairs` | **PASS** | 1 pairs originate from organic tasks |
| `learned_router_authority` | **BLOCKED** | no calibrated artifact has beaten required baselines on clean holdouts |

## Label precision

Policy: `label-precision-v1`; 95% Wilson lower bound must be at least 90% in every required stratum.

| Stratum | Reviewed | Agreements | Rate | Lower bound | Pass |
|---|---:|---:|---:|---:|---|
| `success` | 1 | 1 | 100.0% | 20.7% | NO |
| `task_capability_failure` | 0 | 0 | 0.0% | 0.0% | NO |

## Next measured action

Run the exact-route `router.live_verification` begin/finish flow on eligible organic tasks, independently review both successful and task-capability-failure labels, and run `router.counterfactual` candidates from the same organic snapshot. Training remains blocked until precision, pair-integrity, and representative-data gates pass.
