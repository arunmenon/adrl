# Code review

You are performing a **high-recall code review**. The goal is to catch every real
defect in the change under review. A missed bug that ships is far worse than a
false positive you raise and then discard, so err toward surfacing, but every
finding you keep must name a concrete failure (inputs/state -> wrong output or
crash). Run at your highest reasoning effort.

Review target (optional): `$ARGUMENTS`
- If a git range is given (e.g. `HEAD~3`, `main...HEAD`), review that range.
- If a path or PR number is given, review that.
- If nothing is given, review the working changes (see Phase 0).

Do not modify code in this review unless the user explicitly asks you to apply
fixes. Read, run, and reason; then report.

---

## Phase 0 - Scope the diff

Establish exactly what you are reviewing before you judge anything.

1. Resolve the scope from `$ARGUMENTS`:
   - **Empty**: review the WORKING changes - `git status --short`, then
     `git diff HEAD`. If the tree is clean, fall back to the last commit
     (`git diff HEAD~1 HEAD`).
   - **A git range** (contains `..`, or a ref like `HEAD~3`): `git diff <range>`.
   - **A path** (a file/directory that exists): review just that path -
     `git diff HEAD -- <path>` plus read the current file(s).
   - **A PR number/URL**: `gh pr diff <n>` (or check it out) and review that.
   - If both an argument and uncommitted changes are in play, review both - the
     review often runs before the commit.
2. Read every changed hunk. Then, for each changed function, read the **whole
   enclosing function** - bugs in unchanged lines of a touched function are in
   scope, because the change re-exposes or fails to fix them.
3. Note the languages/frameworks touched so you apply the right pitfalls in
   Phase 2.

## Phase 1 - Load the project's rules

Read the convention files that govern the changed code and treat their rules as
review criteria:
- Repo root `AGENTS.md` and/or `CLAUDE.md`, plus any `AGENTS.md`/`CLAUDE.md` in a
  directory that is an ancestor of a changed file (a directory's file only
  applies at or below it).
- Any linter/formatter/test config the repo pins.

Only flag a convention violation when you can quote the exact rule and the exact
line that breaks it. No style preferences, no "spirit of the doc."

## Phase 2 - Find candidates (work each angle explicitly)

Go through these angles one at a time. Do not let one angle's conclusion suppress
another's - if two angles flag the same line for different reasons, keep both.
For each candidate, write down the file:line and the concrete trigger.

Correctness angles:
1. **Line-by-line.** For every changed line: what input, state, timing, or
   platform makes this wrong? Inverted/wrong conditions, off-by-one, null/None
   deref, missing await, falsy-zero checks, wrong-variable copy-paste, swallowed
   errors, unescaped regex metacharacters, integer/float edge cases.
2. **Removed behavior.** For every deleted or replaced line, name the invariant
   or guard it enforced, then find where the new code re-establishes it. If you
   can't, that's a candidate: a dropped validation, error path, or test case.
3. **Cross-file / contract.** For each changed function, find its callers and
   callees (grep the symbol). Does the change break a call site - a new
   precondition, changed return shape, new exception, new ordering/timing
   dependency? Does a sibling change in the same diff make a call unsafe?
4. **Language / framework pitfalls.** The classic footguns of the touched
   language: Python mutable-default args and late-binding closures; JS falsy-zero,
   `==` coercion, loop-var capture; Go nil-map write and range-var capture; SQL
   injection; timezone/DST; float equality; concurrency (lock scope, shared
   mutable state).
5. **Wrapper / proxy / shared-path integrity.** When the change adds or edits a
   type that wraps another (cache, proxy, adapter, decorator) or extracts a
   shared helper: check every method routes to the right delegate (not back
   through a registry/global, causing re-entry or recursion), that it forwards
   all methods callers use, and - critically - that any code path meant to mirror
   another (an evaluator vs. the thing it evaluates; a shadow vs. live path) does
   not silently diverge in ordering, gating, or defaults.

Cleanup / design angles (lower priority than correctness):
6. **Reuse.** New code re-implementing something the repo already has. Grep
   shared/util modules and files adjacent to the change; name the existing helper
   to call instead. Duplicated constants/logic that can now drift are findings.
7. **Simplification.** Redundant or derivable state, dead parameters, unused
   imports, copy-paste with slight variation, needless nesting.
8. **Efficiency.** Redundant computation or repeated I/O the diff introduces,
   independent work run sequentially, quadratic scans, or long-lived objects that
   capture large scopes. Name the cheaper form.
9. **Altitude.** Is the change a fragile special-case bandaid where generalizing
   an existing mechanism would be sound? Special cases piled on shared
   infrastructure are the smell.
10. **Config / claims integrity.** Does a config block, flag, or setting the diff
    adds actually get read on the live path, or is it inert (dead config)? Does a
    commit message or docstring claim behavior the wiring doesn't deliver? These
    read as done but aren't.

## Phase 3 - Verify each candidate adversarially

For every candidate, try to **refute it**. Default to skeptical.
- Can you name the exact inputs/state that trigger it and the wrong output or
  crash? Then it is **CONFIRMED** - quote the line.
- Is the mechanism real but the trigger uncertain (depends on timing, env,
  config)? Then it is **PLAUSIBLE** - state what would confirm it.
- Is it guarded elsewhere, or factually wrong about what the code does? Then it
  is **REFUTED** - quote the line that disproves it, and drop it.

Where you can, verify at runtime rather than by argument: run the test suite,
write a throwaway script, seed a fixture DB, or reproduce the input. A finding
you reproduced is worth ten you argued. Note any regression the change may have
introduced, not just pre-existing issues.

Keep CONFIRMED and PLAUSIBLE findings. Drop REFUTED ones.

## Phase 4 - Sweep for gaps

Re-read the diff once more as a fresh reviewer who already has the verified list,
looking only for defects **not** already found. Focus on what a first pass
misses: moved/extracted code that dropped a guard, boundary/off-by-one at a
newly-introduced threshold, setup/teardown asymmetry in tests, a test that now
passes vacuously (asserts a constant, or would pass even if the code were
broken), a fail-safe/error contract quietly narrowed, a default flipped. Add any
new candidates and verify them through Phase 3.

---

## Output

Report the surviving findings as a numbered list, **most severe first**.
Correctness outranks cleanup, altitude, and conventions when you must cut. For
each finding:

- **file:line** and a one-sentence statement of the defect.
- **Failure scenario:** concrete inputs/state -> the wrong output or crash (for
  cleanup/convention items, state the concrete cost instead: what duplicates,
  wastes, breaks later, or which quoted rule it violates).
- **Verdict:** CONFIRMED or PLAUSIBLE, and how you verified (tests run, repro,
  or reasoning).

If a machine-readable form is wanted, also emit a JSON array of
`{file, line, summary, failure_scenario, verdict}`.

If nothing survives verification, say so plainly and note what you checked. Do
not pad the list - a short, true report beats a long, hedged one. End by stating
which parts, if any, you could not fully verify and why.

---

## Guardrails

- Read-only by default. Propose fixes in prose; apply them only if asked.
- Honor the repo's own rules from Phase 1 over any generic preference here.
- Never exfiltrate repo contents; keep local data local.
- Match the surrounding code's conventions when you suggest changes.
