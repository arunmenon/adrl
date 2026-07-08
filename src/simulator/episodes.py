"""Stochastic episode generator — thread-structured multi-turn sessions.

The old fixed 5-script grammar produced short, perfectly-coherent episodes:
the opposite of the corpus, where sessions are long-tailed (p50 24 user-turns,
p90 136) and consecutive turns drift hard (median content-word Jaccard 0.02)
while still reading as a real developer's day.

Design (see reports/simulator-realism-plan.md P0-2 + the critique):

1. THREAD-STRUCTURED DRIFT, not per-turn random. A session runs 2-4 concurrent
   intent THREADS (each seeded by a labeled tasks.py scenario, each with its own
   disjoint topic vocabulary). Steps interleave the threads: at each step we
   mostly SWITCH thread (bursty) but sometimes STAY (a sticky run). Because most
   consecutive pairs are cross-thread the GLOBAL median Jaccard collapses toward
   0, yet the sticky runs preserve LOCAL coherence -- so we never emit the
   incoherent non-sequiturs that forced per-turn drift produces.

2. LONG-TAILED LENGTH. Session length is drawn from a lognormal centered on the
   corpus p50 (~24) with a tail crossing 100.

3. INTERRUPTS + ORGANIC RETRY. ~2.5% of turns are interrupted; each interrupt is
   followed by a rephrase/retry turn (the label that matters downstream).

4. CLARIFYING BRANCH. When the agent's previous answer ENDS in a question, the
   driver answers it in-context before drifting (detected at runtime by the
   runner via `answer_is_question` + `clarifying_step`).

Every step still carries a labeled `intent` + `expected_label` so the episode
stays GRADABLE no matter how noisy the surface gets. The seed step of each
thread maps to a planted tasks.py scenario (the ground-truth skeleton).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# ── thread library ────────────────────────────────────────────────────────────
# Each entry: a tasks.py seed scenario + a pool of on-thread follow-ups (so a
# sticky run stays coherent) + a disjoint topic-token set (so cross-thread pairs
# have ~zero lexical overlap -> low global Jaccard). Follow-ups carry the
# expected_label the matchers grade against; some hint a driver archetype.
@dataclass(frozen=True)
class FollowUp:
    intent: str
    expected_label: str
    archetype: Optional[str] = None
    require_completion_marker: bool = False


THREAD_LIBRARY: dict[str, dict] = {
    "explain": {
        "seed": "explain",
        "topic": frozenset({"overview", "architecture", "flow", "structure", "codebase"}),
        "follow_ups": [
            FollowUp("Ask a pointed follow-up about how one specific module fits the flow.",
                     "continuation_of_episode", "question"),
            FollowUp("Ask, tersely, which file is the entrypoint.", "continuation_of_episode", "terse-poll"),
        ],
    },
    "rename": {
        "seed": "rename",
        "topic": frozenset({"rename", "variable", "naming", "identifier", "everywhere"}),
        "follow_ups": [
            FollowUp("Ask the agent to also rename the related helper the same way, terse back-reference "
                     "('same for the helper').", "continuation_of_episode", "coding-ask"),
            FollowUp("Confirm the rename is done and tests pass, then close it out.",
                     "episode_wrap_up", "approval-nudge", True),
        ],
    },
    "fix_test": {
        "seed": "fix_test",
        "topic": frozenset({"test", "pytest", "failing", "bug", "parse", "padded"}),
        "follow_ups": [
            FollowUp("Ask for one more small hardening: tolerate a missing amount column, defaulting to "
                     "0.0, with a test.", "continuation_of_episode", "coding-ask"),
            FollowUp("You are NOT satisfied -- re-ask more specifically for the exact failing assertion "
                     "and the root cause, slightly annoyed.", "retry_signal", "question"),
        ],
    },
    "investigate": {
        "seed": "investigate",
        "topic": frozenset({"ratelimit", "limiter", "throttle", "load", "trigger", "window"}),
        "follow_ups": [
            FollowUp("The diagnosis sounds right -- tell the agent to go ahead and fix it and run the "
                     "tests after.", "action_after_assessment", "approval-nudge"),
            FollowUp("Ask which exact config flag and file/line disables the limiting.",
                     "continuation_of_episode", "path-paste"),
        ],
    },
    "feature": {
        "seed": "feature",
        "topic": frozenset({"flag", "cli", "option", "feature", "verbose", "json"}),
        "follow_ups": [
            FollowUp("Escalate: ask for the same option wired through JSON output too, with a test.",
                     "hard_turn_same_episode", "coding-ask"),
            FollowUp("Ask, tersely, if it's done yet.", "continuation_of_episode", "terse-poll"),
        ],
    },
    "refactor": {
        "seed": "refactor",
        "topic": frozenset({"refactor", "settings", "inject", "layering", "callsites", "interfaces"}),
        "follow_ups": [
            FollowUp("Change your mind mid-stream: 'actually, no -- keep config module but just add a "
                     "typed accessor instead of a full Settings class'.", "mind_change", "disfluent-runon"),
            FollowUp("Wrap up: ask the agent to stage everything and commit with a summary of the "
                     "session.", "episode_wrap_up", "coding-ask", True),
        ],
    },
    "commit_msg": {
        "seed": "commit_msg",
        "topic": frozenset({"commit", "git", "stage", "message", "diff"}),
        "follow_ups": [
            FollowUp("Ask to amend the commit message to mention the padded-input fix.",
                     "continuation_of_episode", "coding-ask"),
        ],
    },
}

# aside / non-coding intents that can be dropped in as their own throwaway thread
NON_CODING_ASIDES = [
    FollowUp("Ask an unrelated ops aside: how much GPU credit is left on the runpod box.",
             "non_coding_aside", "question"),
    FollowUp("Ask the agent to summarize what's been done so far this session, terse.",
             "working_summary", "terse-poll"),
]


# ── step model ────────────────────────────────────────────────────────────────
@dataclass
class Step:
    kind: str                       # "scenario" | "driven" | "interrupt"
    thread: int                     # which intent thread this turn belongs to
    topic: frozenset               # disjoint topic tokens (drift measurement)
    expected_label: str            # gradable ground-truth label
    intent: str = ""               # what the driver should phrase (driven steps)
    scenario: Optional[str] = None  # tasks.py scenario id (seed steps)
    archetype: Optional[str] = None
    require_completion_marker: bool = False


@dataclass
class Episode:
    steps: list[Step]
    n_threads: int
    thread_scenarios: list[str]
    target_length: int
    seed_meta: dict = field(default_factory=dict)


# ── length sampling ───────────────────────────────────────────────────────────
def sample_length(rng, *, cap: int = 221) -> int:
    """Long-tailed session length centered on the corpus p50 (~24 user turns).

    Lognormal(mu=ln 24, sigma=1.25) reproduces p50~24 with a tail crossing 100;
    clamped to [4, 221] to match the corpus support (0% single-turn, max ~221).
    """
    mu = math.log(24)
    val = int(round(rng.lognormvariate(mu, 1.25)))
    return max(4, min(cap, val))


# ── interleaving ──────────────────────────────────────────────────────────────
def _thread_queue(scenario_id: str, rng) -> list[FollowUp]:
    """A shuffled, repeatable stream of on-thread follow-ups for a thread."""
    fus = list(THREAD_LIBRARY[scenario_id]["follow_ups"])
    rng.shuffle(fus)
    return fus


def generate_episode(
    rng,
    *,
    n_threads: Optional[int] = None,
    length: Optional[int] = None,
    switch_p: float = 0.65,
    interrupt_rate: float = 0.025,
) -> Episode:
    """Build one thread-interleaved episode plan.

    `switch_p` is the per-step probability of jumping to a DIFFERENT active
    thread (bursty switching); staying produces a sticky run (local coherence).
    High switch_p is what drives the global median consec-turn Jaccard toward 0
    while sticky runs keep neighbouring same-thread turns coherent.
    """
    scenarios = list(THREAD_LIBRARY)
    n_threads = n_threads or rng.randint(2, 4)
    n_threads = min(n_threads, len(scenarios))
    chosen = rng.sample(scenarios, n_threads)
    target_length = length or sample_length(rng)

    # per-thread state: queue of follow-ups + whether the seed has been emitted
    queues = {t: _thread_queue(sc, rng) for t, sc in enumerate(chosen)}
    seeded = {t: False for t in range(n_threads)}
    cursor = {t: 0 for t in range(n_threads)}

    steps: list[Step] = []
    current = rng.randrange(n_threads)

    def topic_of(t):
        return THREAD_LIBRARY[chosen[t]]["topic"]

    def emit_from_thread(t):
        sc = chosen[t]
        if not seeded[t]:
            seeded[t] = True
            steps.append(Step(kind="scenario", thread=t, topic=topic_of(t),
                              scenario=sc, expected_label="thread_seed",
                              intent=f"seed scenario {sc}"))
            return
        pool = queues[t]
        if cursor[t] < len(pool):
            fu = pool[cursor[t]]
            cursor[t] += 1
        else:  # thread exhausted its scripted follow-ups -> generic on-thread turn.
            # archetype=None lets the driver sample its CALIBRATED prior, so the
            # bulk of a long session reproduces the corpus surface marginals
            # (the scripted follow-ups above are the labeled minority).
            fu = FollowUp(f"Ask a brief on-thread follow-up about the {sc} work.",
                          "continuation_of_episode", None)
        steps.append(Step(kind="driven", thread=t, topic=topic_of(t),
                          intent=fu.intent, expected_label=fu.expected_label,
                          archetype=fu.archetype,
                          require_completion_marker=fu.require_completion_marker))

    while len(steps) < target_length:
        # bursty switch vs sticky stay
        if n_threads > 1 and rng.random() < switch_p:
            others = [t for t in range(n_threads) if t != current]
            current = rng.choice(others)

        # interrupt injection: mark THIS turn interrupted, then an organic retry
        if steps and rng.random() < interrupt_rate:
            steps.append(Step(kind="interrupt", thread=current, topic=topic_of(current),
                              expected_label="user_interrupt",
                              intent="[Request interrupted by user]"))
            steps.append(Step(kind="driven", thread=current, topic=topic_of(current),
                              intent="Rephrase the previous request more specifically after "
                                     "interrupting -- you cut it off, now say what you actually want.",
                              expected_label="organic_retry",
                              archetype=rng.choice(["coding-ask", "question"])))
            continue

        emit_from_thread(current)

    return Episode(steps=steps[:max(target_length, 1)], n_threads=n_threads,
                   thread_scenarios=chosen, target_length=target_length,
                   seed_meta={"switch_p": switch_p, "interrupt_rate": interrupt_rate})


# ── clarifying-question branch (runtime helpers used by run_session) ───────────
def answer_is_question(text: str) -> bool:
    """True when the agent's last answer ends in a question -> the user should
    answer it in-context before drifting to the next thread."""
    return text.rstrip().endswith("?")


def clarifying_step(thread: int, topic: frozenset) -> Step:
    """A driven step that answers the agent's trailing question in context."""
    return Step(
        kind="driven", thread=thread, topic=topic,
        intent="The agent just asked you a clarifying question. Answer it directly and briefly, "
               "in context, then let it proceed.",
        expected_label="clarifying_answer", archetype="approval-nudge",
    )
