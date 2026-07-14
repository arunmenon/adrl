# ADRL Architectural Decision Index

> **Taxonomy v1.0 - FROZEN on 2026-07-13**

This is the canonical architectural index for ADRL. It freezes the vocabulary
used to classify architectural decisions and future work. It does not create a
folder of separate ADR documents. Existing design documents, code, tests, and
reports remain the detailed rationale and evidence behind each indexed decision.

## 1. What is frozen

The following rules are frozen for Taxonomy v1.x:

1. ADRL has the nine buckets in section 3, with the listed three-letter codes.
2. Every architectural decision has an ID in the form `ADRL-{BUCKET}-{NNN}`.
3. An ID is permanent and is never reused, even if its decision is rejected or
   superseded.
4. Every new work item names exactly one **primary** ADRL ID. It may name
   secondary IDs when it affects cross-cutting contracts.
5. New decisions are added inside an existing bucket. If none fits, work stops
   at the design boundary and proposes a taxonomy amendment instead of creating
   an informal tenth bucket.
6. Changing an accepted decision requires an index update that either changes
   its state or supersedes it with a new ID. Code alone does not change the
   architecture.

Adding a decision or advancing its evidence maturity does not change the
taxonomy version. Renaming, splitting, merging, adding, or removing a bucket
requires an explicit architecture review and a Taxonomy v2.0 amendment.

## 2. Two independent states

Architectural agreement and implementation evidence are deliberately separate.
An accepted idea is not automatically built, and built code is not automatically
ready for production.

### Decision state

| State | Meaning |
|---|---|
| `Proposed` | Under discussion; implementation must remain experimental. |
| `Accepted` | The current architectural direction. New work must preserve it. |
| `Deferred` | Intentionally postponed; not an active implementation commitment. |
| `Rejected` | Considered and deliberately not adopted. |
| `Superseded` | Replaced; the row must point to its replacement ID. |

### Evidence maturity

| Level | Meaning |
|---|---|
| `D0 Design` | Rationale exists; no implementation claim. |
| `D1 Code` | A code path exists; focused verification is incomplete. |
| `D2 Tested` | Focused automated tests exercise the contract. |
| `D3 Shadow` | Replay, offline, or shadow evidence exists on representative traffic. |
| `D4 Pilot` | Constrained live use has been observed with rollback available. |
| `D5 Graduated` | Exit metrics passed and the behavior is approved for normal production use. |

Maturity is monotonic only when the linked evidence still applies. A regression,
model change, data-distribution shift, or invalidated assumption can lower it.

## 3. Frozen bucket taxonomy

| Code | Bucket | Owns | Does not own |
|---|---|---|---|
| `FND` | System Boundary and Principles | Product boundary, control-plane ownership, invariants, architectural shape | Individual routing thresholds or model training |
| `SEM` | Interaction Semantics | Request, turn, continuation, session, episode, utility call, and subagent meaning | Which model wins for a classified turn |
| `SAF` | Safety, Privacy, and Hard Constraints | Non-negotiable gates, secret handling, feasibility, blocking, protected resources | Quality optimization inside the allowed set |
| `RTG` | Routing Intelligence and Economics | Runtime choice of rung, utility, cost, uncertainty, and policy precedence | How a chosen rung executes or how models are trained |
| `CAS` | Execution, Cascade, and Recovery | Dispatch, trip-wires, escalation, context transfer, stickiness, and fallback | Initial semantic classification or long-term learning |
| `MEM` | Memory, Evidence, and Label Integrity | Decision/outcome ledger, lifecycle, retrieval evidence, provenance, and label correctness | Choosing a route directly or training a model |
| `LRN` | Learning and Adaptation | Training data, targets, features, calibration, artifacts, retraining, and abstention | Live rollout approval or hard safety gates |
| `EVL` | Evaluation, Graduation, and Rollout | Baselines, holdouts, scorecards, exit gates, shadowing, canaries, and readiness claims | Runtime business logic |
| `OPS` | Platform, Runtime, and Operations | State backends, health, observability, CI, budgets, deployment, rollback, and operator controls | Semantic routing policy |

### Ownership test

Choose the bucket by asking, in order:

1. Is it an inviolable privacy, safety, or feasibility rule? Use `SAF`.
2. Does it define what an incoming interaction *means*? Use `SEM`.
3. Does it choose the best permitted model/rung? Use `RTG`.
4. Does it execute, escalate, retry, or recover after that choice? Use `CAS`.
5. Does it establish trustworthy history or labels? Use `MEM`.
6. Does it train or update intelligence from that evidence? Use `LRN`.
7. Does it prove readiness or control exposure? Use `EVL`.
8. Does it operate the runtime and supporting platform? Use `OPS`.
9. Otherwise, if it changes the whole system boundary or governing principle,
   use `FND`.

The primary ID follows the behavior being changed, not the directory containing
the code. For example, a database migration that preserves outcome-label meaning
is primarily `MEM`, while deployment of that migration is secondarily `OPS`.

## 4. Architectural flow

```text
wire request
    |
    v
SEM: understand request / turn / session / episode
    |
    v
SAF: reduce choices to the permitted and feasible set
    |
    v
RTG: select the best permitted rung for expected quality, cost, and risk
    |
    v
CAS: dispatch, observe deterministic failure, escalate, and recover
    |
    v
MEM: record decision, actual service, outcome, verification, and provenance
    |
    +------> LRN: learn a better policy from cause-clean evidence
    |
    +------> EVL: compare against baselines and decide whether to graduate

OPS supports every stage. FND defines the boundary and invariants around all of it.
```

## 5. Indexed decisions

These rows are conceptual ADRs: stable decision identities with links to their
current rationale and evidence. The maturity column describes the repository as
of the freeze date; it is not a promise about deployment outside this repository.

### FND - System Boundary and Principles

| ID | Decision | State | Maturity | Rationale / evidence |
|---|---|---|---|---|
| `ADRL-FND-001` | ADRL is a transparent control layer between the coding harness and model gateway. | Accepted | D3 Shadow | [Design sections 1-3](adaptive-routing-layer-design.md), [Phase 0 exit](../reports/phase0-exit.md) |
| `ADRL-FND-002` | ADRL owns semantic policy; LiteLLM and providers own mechanical model execution. | Accepted | D2 Tested | [Integration ownership](adaptive-routing-layer-design.md), [live router](../src/router/live_router.py) |
| `ADRL-FND-003` | The normal routing boundary is a user turn, not every HTTP request. | Accepted | D3 Shadow | [Core idea](adaptive-routing-layer-design.md), [discriminator evidence](../reports/discriminator-eval.md) |
| `ADRL-FND-004` | Failure defaults to unchanged upstream behavior, with a one-step operator bypass. | Accepted | D4 Pilot | [Risk controls](adaptive-routing-layer-design.md), [capture proxy](../src/proxy/capture_proxy.py) |
| `ADRL-FND-005` | Scope expands only through measured phase gates; component presence is not readiness. | Accepted | D3 Shadow | [Rollout plan](adaptive-routing-layer-design.md), [Phase 0 exit](../reports/phase0-exit.md) |

### SEM - Interaction Semantics

| ID | Decision | State | Maturity | Rationale / evidence |
|---|---|---|---|---|
| `ADRL-SEM-001` | Requests are mechanically classified as user turns, continuations, utility calls, subagents, or passthrough traffic. | Accepted | D3 Shadow | [Discriminator design](adaptive-routing-layer-design.md), [evaluation](../reports/discriminator-eval.md) |
| `ADRL-SEM-002` | `metadata.user_id` is the preferred session key; a stable anonymous fallback prevents unrelated traffic from sharing state. | Accepted | D3 Shadow | [Session-key evidence](../reports/assumption-user-id.md), [live router](../src/router/live_router.py) |
| `ADRL-SEM-003` | Continuations inherit the sticky route and do not trigger a fresh difficulty decision. | Accepted | D2 Tested | [Session lifecycle](adaptive-routing-layer-design.md), [policy](../src/router/policy.py) |
| `ADRL-SEM-004` | Utility calls may use a small local model; protocol and unknown calls pass through unchanged. | Accepted | D3 Shadow | [Traffic taxonomy](adaptive-routing-layer-design.md), [Phase 1 plan](phase1-plan.md) |
| `ADRL-SEM-005` | Episode boundaries are conservative semantic events that may release escalation hysteresis. | Accepted | D2 Tested | [episode detector](../src/router/episode.py), [state tests](../tests/test_episode.py) |
| `ADRL-SEM-006` | Subagents are linked to the parent for constraints but have separate routing identity and evidence. | Accepted | D0 Design | Implementation is deferred; see the [Phase 1 subagent pilot](phase1-plan.md) |

### SAF - Safety, Privacy, and Hard Constraints

| ID | Decision | State | Maturity | Rationale / evidence |
|---|---|---|---|---|
| `ADRL-SAF-001` | Hard gates execute before and cannot be overridden by heuristics or learned policy. | Accepted | D2 Tested | [policy](../src/router/policy.py) |
| `ADRL-SAF-002` | Privacy pinning is one-way for the session: once local-only, it cannot silently unpin. | Accepted | D2 Tested | [Privacy design](adaptive-routing-layer-design.md), [privacy module](../src/router/privacy.py) |
| `ADRL-SAF-003` | Secret detection occurs before routing and suppresses prompt-derived embeddings and identifiers. | Accepted | D3 Shadow | [Secret scanner report](../reports/p1-secret-scanner.md), [memory facade](../src/router/memory_facade.py) |
| `ADRL-SAF-004` | A pinned session cannot fall back or escalate to cloud; unresolved failure is surfaced to the user. | Accepted | D2 Tested | [Pin collision behavior](adaptive-routing-layer-design.md), [escalation controller](../src/router/escalation_controller.py) |
| `ADRL-SAF-005` | Privacy-context conflicts block instead of leaking data or silently truncating context. | Accepted | D2 Tested | [Pin feasibility](adaptive-routing-layer-design.md), [policy](../src/router/policy.py) |
| `ADRL-SAF-006` | Unhealthy or context-infeasible rungs are removed before optimization. | Accepted | D2 Tested | [Capability registry](adaptive-routing-layer-design.md), [health monitor](../src/router/health.py) |
| `ADRL-SAF-007` | Verification commands are constrained by protected paths and explicit execution policy. | Accepted | D2 Tested | [verifier](../src/router/verifier.py), [verifier tests](../tests/test_verifier.py) |

### RTG - Routing Intelligence and Economics

| ID | Decision | State | Maturity | Rationale / evidence |
|---|---|---|---|---|
| `ADRL-RTG-001` | The policy reasons over three capability/cost rungs: local, cheap cloud, and frontier. | Accepted | D3 Shadow | [Policy design](adaptive-routing-layer-design.md), [scenario evidence](adaptive-routing-scenarios.md) |
| `ADRL-RTG-002` | Within hard constraints, select the cheapest healthy rung likely to complete the task. | Accepted | D2 Tested | [policy](../src/router/policy.py) |
| `ADRL-RTG-003` | Deterministic rules own clear cases; learned intelligence is reserved for the ambiguous middle. | Accepted | D3 Shadow | [Three-layer policy](adaptive-routing-layer-design.md), [classifier shadow](../reports/classifier-shadow.md) |
| `ADRL-RTG-004` | Local-first is conditional and is used only when a controlled cascade remains feasible. | Accepted | D2 Tested | [policy](../src/router/policy.py), [live router](../src/router/live_router.py) |
| `ADRL-RTG-005` | Runtime optimization uses expected verified quality, retry risk, latency, and cost rather than difficulty alone. | Accepted | D3 Shadow | [Utility shadow](../reports/p1-utility-shadow.md) |
| `ADRL-RTG-006` | The current LLM classifier is a fail-safe middle-band advisor, not a safety authority or final learned router. | Accepted | D3 Shadow | [classifier](../src/router/llm_classifier.py), [bake-off](../reports/classifier-bakeoff.md) |
| `ADRL-RTG-007` | The target learned decision is the marginal utility of frontier versus local, with calibrated uncertainty and abstention. | Accepted | D0 Design | Cross-reference `ADRL-LRN-003`, `ADRL-LRN-005`, and `ADRL-EVL-007` |
| `ADRL-RTG-008` | Capability-rung selection is separate from endpoint/provider selection inside a rung. | Accepted | D2 Tested | [backend roles](../src/router/backends.py), [live router](../src/router/live_router.py) |

### CAS - Execution, Cascade, and Recovery

| ID | Decision | State | Maturity | Rationale / evidence |
|---|---|---|---|---|
| `ADRL-CAS-001` | Escalation is triggered by deterministic post-call trip-wires, not free-form model self-judgment. | Accepted | D3 Shadow | [Escalation design](adaptive-routing-layer-design.md), [post-call shadow](../reports/p1-postcall-shadow.md) |
| `ADRL-CAS-002` | Failures are typed as task difficulty, model dialect/capability, infrastructure, or policy so each signal reaches the correct owner. | Accepted | D2 Tested | [outcomes](../src/router/outcomes.py), [trip-wires](../src/router/tripwires.py) |
| `ADRL-CAS-003` | Escalation occurs only at a controlled action boundary so partial tool execution is not replayed blindly. | Accepted | D2 Tested | [escalation controller](../src/router/escalation_controller.py) |
| `ADRL-CAS-004` | Cross-model continuation preserves tool IDs/results, removes private reasoning, and adds a short handoff note. | Accepted | D3 Shadow | [Post-call shadow](../reports/p1-postcall-shadow.md), [tool-ID evidence](../reports/assumption-tool-ids.md) |
| `ADRL-CAS-005` | Escalation is sticky within an episode; only a conservative episode boundary may lower the rung. | Accepted | D2 Tested | [episode detector](../src/router/episode.py), [session state](../src/router/state.py) |
| `ADRL-CAS-006` | Sticky state records the rung that actually served the response, including transport fallback. | Accepted | D2 Tested | [capture proxy](../src/proxy/capture_proxy.py), [router proxy tests](../tests/test_router_proxy.py) |
| `ADRL-CAS-007` | Failure at the top rung, or on a privacy-pinned route, is surfaced rather than hidden by another automatic retry. | Accepted | D2 Tested | [escalation controller](../src/router/escalation_controller.py) |

### MEM - Memory, Evidence, and Label Integrity

| ID | Decision | State | Maturity | Rationale / evidence |
|---|---|---|---|---|
| `ADRL-MEM-001` | Transaction memory is an append-oriented decision/outcome/event ledger keyed by immutable `route_id`. | Accepted | D2 Tested | [memory contract](../src/router/memory_ports.py), [SQLite provider](../src/router/memory_sqlite.py) |
| `ADRL-MEM-002` | Outcome lifecycle is explicit: `pending`, `closed_turn`, then `closed_final` after late retry/interruption evidence. | Accepted | D2 Tested | [memory contract](../src/router/memory_ports.py), [memory tests](../tests/test_memory_sqlite.py) |
| `ADRL-MEM-003` | Deterministic verification enriches an outcome without overwriting observed telemetry. | Accepted | D2 Tested | [verifier](../src/router/verifier.py), [organic verification bridge](../src/router/live_verification.py), [tests](../tests/test_live_verification.py); representative organic labels remain open |
| `ADRL-MEM-004` | Training labels keep task difficulty separate from dialect/capability, infrastructure, privacy, and policy failures. | Accepted | D2 Tested | [outcomes](../src/router/outcomes.py), [rule health](../src/router/rule_health.py) |
| `ADRL-MEM-005` | Raw prompts are not stored by default; embeddings and instruction hashes are suppressed for private/secret turns. | Accepted | D2 Tested | [memory facade](../src/router/memory_facade.py), [privacy tests](../tests/test_privacy.py) |
| `ADRL-MEM-006` | The router uses a fail-safe memory facade and provider port; SQLite is the current local provider, not a semantic dependency. | Accepted | D2 Tested | [memory facade](../src/router/memory_facade.py), [provider contract](../src/router/memory_ports.py) |
| `ADRL-MEM-007` | Derived retrieval indexes are rebuildable projections and detect cross-process database changes. | Accepted | D2 Tested | [SQLite provider](../src/router/memory_sqlite.py) |
| `ADRL-MEM-008` | Retrieval remains advisory/shadow until evaluated-label quantity and quality gates pass. | Accepted | D3 Shadow | [retrieval report](../reports/retrieval-shadow.md), [shadow router](../src/router/shadow_retrieval.py) |
| `ADRL-MEM-009` | Counterfactual evidence attaches only to an explicit `route_id`; time or session proximity is never used to guess ownership. | Accepted | D2 Tested | [counterfactual runner](../src/router/counterfactual.py), [organic verification bridge](../src/router/live_verification.py), [tests](../tests/test_live_verification.py) |

### LRN - Learning and Adaptation

| ID | Decision | State | Maturity | Rationale / evidence |
|---|---|---|---|---|
| `ADRL-LRN-001` | Verified, cause-clean outcomes outrank heuristic proxy labels for training. | Accepted | D2 Tested | Plumbing: [verifier](../src/router/verifier.py), [organic verification bridge](../src/router/live_verification.py), [outcomes](../src/router/outcomes.py); representative label volume remains open |
| `ADRL-LRN-002` | Counterfactual training data uses paired local/frontier attempts from the same sanitized snapshot. | Accepted | D2 Tested | [Counterfactual runner](../src/router/counterfactual.py), [versioned evidence contract](../src/router/learning_contract.py), [tests](../tests/test_counterfactual.py), and [readiness evidence](../reports/learning-readiness.md); one valid organic same-snapshot pair is recorded, while representative volume and generalization remain open |
| `ADRL-LRN-003` | Train a calibrated utility estimator for marginal frontier gain, not a classifier that imitates current heuristic routes. | Accepted | D0 Design | Cross-reference `ADRL-RTG-007` |
| `ADRL-LRN-004` | Learned models may use only information available before the routing decision; outcome and future-session leakage are prohibited. | Accepted | D2 Tested | [learning contract](../config/learning-contract-v1.json), [contract enforcement](../src/router/learning_contract.py), [tests](../tests/test_learning_contract.py) |
| `ADRL-LRN-005` | Every learned artifact versions its feature schema, data snapshot, objective, calibration, thresholds, and policy compatibility. | Accepted | D2 Tested | [artifact contract](../config/learning-contract-v1.json), [manifest validation](../src/router/learning_contract.py), [tests](../tests/test_learning_contract.py) |
| `ADRL-LRN-006` | Uncertain or out-of-distribution predictions abstain to the deterministic safe policy. | Accepted | D0 Design | Cross-reference `ADRL-RTG-007` |
| `ADRL-LRN-007` | Learning may propose policy updates, but deployment requires offline evaluation and explicit graduation; no autonomous online promotion. | Accepted | D0 Design | Cross-reference `ADRL-EVL-006` |

### EVL - Evaluation, Graduation, and Rollout

| ID | Decision | State | Maturity | Rationale / evidence |
|---|---|---|---|---|
| `ADRL-EVL-001` | Every routing policy is compared with best-single, always-local, always-frontier, current heuristic, and current classifier baselines. | Accepted | D3 Shadow | [Phase 0 exit](../reports/phase0-exit.md), [policy replay](../reports/policy-replay.md) |
| `ADRL-EVL-002` | Evaluation uses temporal and repository/task holdouts to detect memorization and distribution leakage. | Accepted | D2 Tested | [split contract](../config/learning-contract-v1.json), [split enforcement](../src/router/learning_contract.py), [tests](../tests/test_learning_contract.py) |
| `ADRL-EVL-003` | Success means lower cost/latency at non-inferior verified quality, retry, privacy, and reliability rates. | Accepted | D3 Shadow | [Metrics](adaptive-routing-layer-design.md), [utility shadow](../reports/p1-utility-shadow.md) |
| `ADRL-EVL-004` | Retrieval cannot graduate without at least 300 evaluated middle-band decisions and adequate hard-case support. | Accepted | D3 Shadow | [retrieval report](../reports/retrieval-shadow.md) |
| `ADRL-EVL-005` | Simulator evidence cannot substitute for organic evidence until the realism gate passes. | Accepted | D3 Shadow | [realism scorecard](../reports/realism-scorecard.md), [realism plan](../reports/simulator-realism-plan.md) |
| `ADRL-EVL-006` | Policy exposure advances through offline/replay, shadow, constrained canary, and graduated operation with rollback at every live stage. | Accepted | D0 Design | [Rollout plan](adaptive-routing-layer-design.md) |
| `ADRL-EVL-007` | A learned router must beat both the heuristic policy and best-single baseline before receiving live authority. | Accepted | D0 Design | Required by `ADRL-RTG-007` |
| `ADRL-EVL-008` | Production-readiness claims require representative constrained-model and real production-model evaluation, not simulator-only results. | Accepted | D0 Design | Execution is deferred; see [Phase 1 representativeness](phase1-plan.md) and [Phase 0 carryover](../reports/phase0-exit.md) |
| `ADRL-EVL-009` | Readiness scores are evidence-derived per bucket; an aggregate score must expose weights, gates, and confidence rather than average away blockers. | Accepted | D2 Tested | [frozen scoring contract](../config/readiness-score-v1.json), [baseline and change control](readiness-scoring.md), [readiness implementation](../src/router/learning_readiness.py), [scorecard](../reports/learning-readiness.md), [tests](../tests/test_learning_readiness.py) |

### OPS - Platform, Runtime, and Operations

| ID | Decision | State | Maturity | Rationale / evidence |
|---|---|---|---|---|
| `ADRL-OPS-001` | Use an in-process session store for one worker; introduce Redis only when multi-worker consistency or durability requires it. | Accepted | D2 Tested | [State backend rationale](adaptive-routing-layer-design.md), [state store](../src/router/state.py) |
| `ADRL-OPS-002` | Organic, simulator, shadow, and counterfactual traffic carry explicit provenance and do not silently contaminate each other. | Accepted | D2 Tested | [live router](../src/router/live_router.py), [memory contract](../src/router/memory_ports.py) |
| `ADRL-OPS-003` | Runtime health is a circuit input with an injectable external probe, not a static configuration claim. | Accepted | D2 Tested | [health monitor](../src/router/health.py), [health tests](../tests/test_health.py) |
| `ADRL-OPS-004` | Experiments run under explicit time, cost, command, path, and concurrency budgets enforced by the execution boundary. | Accepted | D2 Tested | [counterfactual runner](../src/router/counterfactual.py), [verifier](../src/router/verifier.py) |
| `ADRL-OPS-005` | CI runs the supported Python test matrix and is required before maturity claims advance. | Accepted | D1 Code | [CI workflow](../.github/workflows/test.yml); first remote run remains open |
| `ADRL-OPS-006` | Observability records intended route, actually served route, policy provenance, latency, cost, and typed failure cause. | Accepted | D2 Tested | [memory contract](../src/router/memory_ports.py), [telemetry](../src/router/telemetry.py) |
| `ADRL-OPS-007` | Sensitive datasets remain local and gitignored; committed reports contain aggregate evidence only. | Accepted | D4 Pilot | [repository data policy](../README.md), [miner](../src/miner/secrets.py) |
| `ADRL-OPS-008` | Every live routing authority has a bypass/kill switch, version identity, and rollback path. | Accepted | D4 Pilot | [capture proxy](../src/proxy/capture_proxy.py), [Phase 0 plan](phase0-plan.md) |

## 6. Portfolio view

This is the architectural position at freeze time. It summarizes the decision
rows; it is not an arithmetic readiness score.

| Bucket | Current center of gravity | Principal evidence gap |
|---|---|---|
| `FND` | Boundary and fail-safe shape are at shadow/pilot maturity. | Graduate the user-visible control path, not only transparent capture. |
| `SEM` | Main-session request, turn, continuation, and episode semantics are tested or shadowed. | Implement and validate linked-but-separate subagent semantics. |
| `SAF` | Privacy, secret, feasibility, health, and verifier gates are tested; some have shadow data. | Run adversarial live-pilot evidence without weakening hard precedence. |
| `RTG` | Heuristic and LLM-advisor policy is shadowed. | Replace noisy difficulty inference with calibrated marginal utility. |
| `CAS` | Trip-wires, recovery, and state contracts are tested and replayed. | Constrained live escalation pilot with rollback and outcome audit. |
| `MEM` | Ledger, lifecycle, privacy, provenance, and retrieval contracts are tested; retrieval is shadow-only. | Accumulate representative verified outcomes and counterfactual pairs. |
| `LRN` | Label and counterfactual plumbing is tested; the target learned estimator remains design-only. | Train, calibrate, version, and validate the estimator on clean holdouts. |
| `EVL` | Existing baselines and shadow reports exist. | Temporal/repository holdouts, representative models, and canary graduation. |
| `OPS` | Local operation, health, budgets, observability, and rollback paths exist at mixed maturity. | First remote CI proof; multi-worker state only when deployment requires it. |

## 7. Current roadmap through the taxonomy

The next learned-routing milestone is not one large "classifier" feature. It is
a chain of decisions with separate owners:

| Sequence | Primary ID | Deliverable | Gate before moving on |
|---|---|---|---|
| 1 | `ADRL-MEM-003` | Produce deterministic verifier outcomes on eligible organic tasks. | Label precision audit passes. |
| 2 | `ADRL-LRN-002` | Collect budgeted same-snapshot local/frontier pairs. | Provenance, isolation, and pairing audits pass. |
| 3 | `ADRL-MEM-004` | Exclude dialect, infrastructure, privacy, and policy failures from task-hard labels. | Manual sample agrees with automatic causes. |
| 4 | `ADRL-LRN-003` | Train and calibrate the marginal-utility estimator. | Temporal and repo holdouts are clean. |
| 5 | `ADRL-EVL-001` | Compare learned, heuristic, classifier, and best-single baselines. | Learned policy wins on the declared utility objective. |
| 6 | `ADRL-RTG-007` | Integrate the estimator behind abstention in shadow mode. | No hard-gate overrides; latency and calibration pass. |
| 7 | `ADRL-EVL-006` | Run constrained canary and graduate or roll back. | Quality non-inferiority and operational gates pass. |

This sequence explains why memory and verification precede a stronger learned
router: without trustworthy paired outcomes, a model can only learn the existing
heuristics or noisy failure artifacts.

## 8. Registration block for new work

Every design note, plan, issue, or pull request that changes behavior should
include this block:

```text
Primary ADRL ID: ADRL-___-___
Secondary ADRL IDs: none
Decision effect: implements | adds evidence | changes | supersedes
Maturity transition: D_ -> D_ (or none)
Evidence to produce: tests/report/metric/rollback proof
Architectural non-goals: what this work deliberately does not change
```

Rules for use:

- A change may touch many files but still has one primary architectural owner.
- Refactoring with no behavioral decision uses the ID whose contract it preserves
  and sets `Decision effect: adds evidence` or `none` in the pull request.
- `changes` or `supersedes` requires updating this index in the same change.
- A claimed maturity increase requires a durable evidence link.
- Work with no valid ID is architecturally unclassified and is not ready to merge.

### Enforcement surfaces

| Surface | Enforcement |
|---|---|
| `AGENTS.md` | Codex must classify substantive work before planning or editing and carry the primary ID through implementation and closeout. |
| `.github/pull_request_template.md` | Every pull request records classification, decision effect, maturity effect, evidence, and non-goals. |
| `tools/check_adr_taxonomy.py` | Validates frozen buckets, IDs, states, maturity vocabulary, evidence links, and completed PR classification. |
| `tools/check_readiness_score.py` | Reconstructs the frozen baseline, recomputes the current architecture index, and validates matching JSON, Markdown, and append-only history. |
| `.github/workflows/test.yml` | Fails the status check when taxonomy, readiness artifacts, PR classification, or tests fail. Branch protection must require this check to block merges. |
| `.codex/prompts/code-review.md` | Reviews taxonomy ownership, accepted-decision compliance, and evidence-backed maturity claims. |

These layers serve different purposes. `AGENTS.md` guides agent reasoning;
validation and CI provide deterministic enforcement. An ID alone is not
grounding: Codex must read the row and use its linked evidence.

## 9. Taxonomy amendment procedure

A taxonomy amendment must include:

1. The concrete decisions that cannot be classified under v1.x.
2. Why primary-plus-secondary IDs are insufficient.
3. The proposed bucket change and ownership boundaries.
4. Migration mapping for every affected existing ID.
5. Explicit approval and a major taxonomy version update.

Until that amendment is accepted, Taxonomy v1.0 remains the source of truth.
