# retrieval router - shadow evaluation (WS4)

- db: `data/router-memory.db`
- middle-band turns with embeddings: 100
- finalized outcomes in memory: 784 (cold-start gate: 8)
- graduation needs: >= 300 EVALUATED (non-abstained) middle-band decisions
- evaluated (non-abstained): 42  (coverage 42%, abstained 58)

## preview confusion matrix (leave-one-out)

| | actual hard | actual easy |
|--|--:|--:|
| **predicted frontier** | 0 (TP) | 0 (FP) |
| **predicted local** | 0 (FN) | 42 (TN) |

- accuracy: 100%
- frontier recall: 0% (of turns that really went hard, how many it would send to frontier)
- frontier send-rate: 0%  (baseline always-local = 0%, always-frontier = 100%)
- actual hard-rate in this band: 0%

**Baselines on the same evaluated set** (cost-aware graduation target):
- always-local: recall 0%, sends 0% to frontier (cheapest, misses every hard turn)
- always-frontier: recall 100%, sends 100% to frontier (safest, most expensive)
- the retrieval router must beat always-local on accuracy WITHOUT dropping frontier recall below the LLM classifier's (classifier verdicts not yet stamped in the ledger; that comparison lands once classifier_tier is populated live).

## verdict: INSUFFICIENT DATA - 258 more EVALUATED middle-band decisions needed
The matrix above is a preview, not a graduation decision. Keep the flywheel running; re-run this report as the evaluated set grows.
