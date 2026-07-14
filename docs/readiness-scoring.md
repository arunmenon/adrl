# ADRL readiness scoring contract

This document defines the only comparable ADRL architecture-score series. It is
governed by `ADRL-EVL-009`. Production readiness is deliberately a separate
hard-gate verdict and is never inferred from the numeric index.

## Canonical persisted artifacts

| Artifact | Purpose | Change rule |
|---|---|---|
| `config/readiness-score-v1.json` | Frozen formula: maturity points, buckets, weights, decision weighting, rounding, and verdict strategy. | Immutable after the v1 baseline is recorded. A methodology change creates v2. |
| `reports/readiness-baseline-v1.json` | Reproducible starting point for the v1 series. | Immutable; a new scoring contract gets a new baseline. |
| `reports/learning-readiness.json` | Latest complete machine-readable measurement. | Replaced only by the generator. |
| `reports/learning-readiness.md` | Human-readable rendering of the same JSON snapshot. | Replaced only by the generator. |
| `reports/readiness-history.jsonl` | One hash-chained aggregate record per distinct evidence state. | Append only; duplicate states are not added. |

Private ledger data remains under gitignored `data/`. Committed score artifacts
contain aggregate counts, ADR maturity profiles, gates, and hashes only.

## Frozen v1 baseline

- Baseline ID: `taxonomy-v1-freeze-a46ca8c`
- Source commit: `a46ca8ca5d0c293fb701fedfe2575a78d8d491d8`
- Architecture Evidence Index: `40.5/100`
- Contract: `architecture-evidence-index-v1`
- Contract SHA-256:
  `d26f5175f43a66aac99d80887c88de81ac6683d39127c53553a3d859df677bb4`

The baseline is the frozen Taxonomy v1.0 index before the learning-contract
milestone. The current `42.8` measurement is therefore `+2.3` points under the
same formula. Scores stated before this contract existed are historical
estimates, not entries in this comparable series.

## Formula

1. Score each accepted ADR using D0=0, D1=20, D2=40, D3=60, D4=80, D5=100.
2. Average decisions with equal weight inside their bucket.
3. Average the nine frozen bucket scores with equal weight.
4. Round the displayed result to one decimal place.
5. Show each bucket's maturity profile and all hard blockers beside the number.

The implementation reads this method from the versioned JSON contract. It does
not keep a second editable copy of the weights or point scale in source code.

## Milestone workflow

After a milestone has durable evidence and its ADR maturity rows are updated:

```bash
PYTHONPATH=src .venv/bin/python -m router.learning_readiness --persist
.venv/bin/python tools/check_readiness_score.py
python3 tools/check_adr_taxonomy.py
```

`--persist` writes matching Markdown and JSON outputs and appends history only
when the evidence state changed. Each history record hashes its contents and
references its predecessor. Re-running without new evidence leaves the existing
snapshot timestamp and history unchanged.

CI runs `tools/check_readiness_score.py` without private data. It recomputes the
architecture index from `docs/adr-index.md`, validates the baseline against the
contract, verifies the current JSON and Markdown agree, and requires the current
snapshot to occur exactly once in history.

## Change control

Scores are comparable only when both `contract_version` and `contract_sha256`
match. To change points, weights, bucket treatment, decision treatment, or
rounding:

1. Add a new versioned contract instead of editing v1.
2. Record why the methodology changed under `ADRL-EVL-009`.
3. Establish a new baseline and label the old and new series non-comparable.
4. Keep the old contract, baseline, and history available for audit.

An ADR maturity increase still requires durable evidence linked from
`docs/adr-index.md`. Regenerating the scorecard by itself cannot increase a D-level.
