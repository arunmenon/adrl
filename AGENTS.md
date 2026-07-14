# ADRL Repository Instructions

These instructions apply to every substantive task in this repository. The
canonical architecture source is `docs/adr-index.md` (Taxonomy v1.0).

## Architecture grounding

Before planning, reviewing, or editing:

1. Read the relevant bucket and decision rows in `docs/adr-index.md`.
2. Assign exactly one primary ADRL ID to the task. Add secondary IDs only for
   real cross-cutting contracts.
3. State the classification in the first substantive progress update using:

   ```text
   Primary ADRL ID: ADRL-___-___
   Secondary ADRL IDs: none
   Decision effect: implements | adds evidence | changes | supersedes | none
   Expected maturity effect: D_ -> D_ | none
   ```

4. If no existing decision fits, do not invent a bucket or silently expand a
   decision. Propose a new row inside one of the nine frozen buckets. If no
   bucket fits, stop implementation at that boundary and propose the Taxonomy
   v2.0 amendment required by the index.

A task implements an indexed decision; it does not need a new decision ID merely
because it creates a new file, class, endpoint, or test. Do not add ADR comments
throughout source code unless they explain a genuinely non-obvious constraint.

## Evidence discipline

- Treat an index row's decision state and maturity as separate facts.
- Do not describe code as production-ready merely because it exists or has unit
  tests. Use the D0-D5 definitions in the index exactly.
- Ground architectural and readiness claims in repository evidence: code, tests,
  committed reports, or measured runtime output. Distinguish observed fact from
  inference and proposal.
- A maturity increase requires durable evidence linked from the index. Update
  `docs/adr-index.md` in the same change.
- A change to an accepted decision must update or supersede its index row. Code
  alone cannot change the architecture.
- Safety (`SAF`) decisions and evaluation graduation gates (`EVL`) cannot be
  overridden by routing, learning, or implementation convenience.
- Readiness claims must come from `reports/learning-readiness.json` under the
  exact version and hash in `config/readiness-score-v1.json`. Do not blend older
  estimates into that comparable series. A formula change requires a new
  versioned contract and baseline; never rewrite the v1 baseline or history.

## Implementation workflow

- Keep one primary architectural owner even when many files are touched.
- Preserve the primary ID through plans, progress updates, review findings, and
  the final summary.
- Before editing, inspect the full decision row and its linked evidence, not only
  the bucket title.
- For code changes, add focused tests proportional to risk and run the relevant
  subset before the full suite when appropriate.
- Before finishing any substantive task, run:

  ```bash
  python3 tools/check_adr_taxonomy.py
  ```

- If ADR maturity or readiness evidence changes, persist and validate the score:

  ```bash
  PYTHONPATH=src .venv/bin/python -m router.learning_readiness --persist
  .venv/bin/python tools/check_readiness_score.py
  ```

- For the full Python suite, use:

  ```bash
  PYTHONPATH=src .venv/bin/python -m pytest tests -q
  ```

## Review requirements

Code review must verify:

1. The stated primary ADRL ID matches the behavior changed.
2. The implementation preserves every relevant accepted decision and hard gate.
3. Live and shadow paths have not drifted in ordering, inputs, labels, or
   defaults.
4. Decision or maturity claims are backed by the linked evidence.
5. New behavior is not hidden inside an `adds evidence`, `none`, or refactor
   classification.

Missing or incorrect architecture classification is a review finding. A direct
violation of an accepted `SAF`, `FND`, or graduation-gate decision is a blocking
finding.

## Final response

For substantive repository work, include a compact architecture closeout:

```text
Primary ADRL ID: ADRL-___-___
Decision effect: ...
Maturity effect: ...
Evidence: tests/reports/runtime checks
```

Do not claim a maturity change when the work only adds implementation plumbing.
