"""Calibration tests for the noisy user-driver + stochastic episode generator.

These assert that the SIMULATED user surface reproduces the measured corpus
bands (reports/simulator-realism-plan.md P0-1/P0-2) and -- critically -- that
style is sampled JOINTLY per turn-archetype, so we never manufacture chimeras
(a 250-word all-lowercase verbless fragment) by combining independent flags.

The LLM is STUBBED here: prose archetypes get filler text of the requested
length, so there is no API cost and no ollama/network dependency. The stub
returns exactly `target_words` words, which is precisely the driver-controlled
length constraint under test; templated archetypes never touch the LLM at all.

Run:
  PYTHONPATH=src .venv/bin/python -m pytest tests/test_sim_driver.py -q
"""

from __future__ import annotations

import random
import statistics
from collections import Counter

from simulator.driver import ARCHETYPES, generate_turn
from simulator.episodes import (
    answer_is_question,
    clarifying_step,
    generate_episode,
    sample_length,
)

# large N: the stub is free, so we drive variance down well below the >=300 floor
N = 4000
SEED = 20240707

# lowercase filler so the driver's casing flag is what decides all-lowercase
_STUB_WORDS = (
    "rename the limiter flag so it triggers under load keep tests green check "
    "the parse date strip bug in stats because it fails on padded input add a "
    "verbose option with a test update readme usage for the cli entrypoint and "
    "inject settings across modules migrate callsites keep everything passing"
).split()

_SANDBOX_PATHS = [
    "/Users/dev/projects/shiplog/shiplog/parse.py",
    "/Users/dev/projects/orderflow/src/limiter.py",
    "src/metricd/stats.py",
]


def stub_llm(prompt, target_words, rng):
    """Deterministic filler of exactly target_words lowercase words. No cost."""
    return " ".join(rng.choice(_STUB_WORDS) for _ in range(max(1, target_words))), 0.0


# ── metric helpers (no numpy/pandas in this env) ──────────────────────────────
def _pct(sorted_vals, q):
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def _frac(items, pred):
    return sum(1 for x in items if pred(x)) / len(items)


def _gen_turns(n=N, seed=SEED):
    rng = random.Random(seed)
    return [
        generate_turn("do the thing on this file", "ok, here is what i found so far",
                      rng=rng, llm=stub_llm, sandbox_paths=_SANDBOX_PATHS)
        for _ in range(n)
    ]


# ── surface marginals ─────────────────────────────────────────────────────────
def test_surface_word_count_bands():
    turns = _gen_turns()
    wc = sorted(t.word_count for t in turns)
    p50, p90 = _pct(wc, 0.50), _pct(wc, 0.90)
    assert 8 <= p50 <= 12, f"word p50 {p50} out of [8,12]"
    assert 58 <= p90 <= 86, f"word p90 {p90} out of [58,86]"
    # a real long tail must exist (corpus p99 ~297) but not blow past corpus max
    assert _pct(wc, 0.99) >= 120, f"word p99 {_pct(wc, 0.99)} — tail too thin"


def test_surface_punctuation_and_casing_bands():
    turns = _gen_turns()
    no_punct = _frac(turns, lambda t: not t.has_terminal_punct)
    lower = _frac(turns, lambda t: t.is_lowercase)
    assert 0.55 <= no_punct <= 0.75, f"no-terminal-punct {no_punct:.3f} out of [.55,.75]"
    assert 0.20 <= lower <= 0.32, f"all-lowercase {lower:.3f} out of [.20,.32]"


def test_surface_typo_and_disfluency_bands():
    turns = _gen_turns()
    typo = _frac(turns, lambda t: t.has_typos)
    disfluency = _frac(turns, lambda t: t.is_disfluent)
    question = _frac(turns, lambda t: t.is_question)
    assert 0.16 <= typo <= 0.26, f"typo/txt-speak {typo:.3f} out of [.16,.26]"
    assert 0.10 <= disfluency <= 0.17, f"disfluency {disfluency:.3f} out of [.10,.17]"
    assert 0.28 <= question <= 0.42, f"question rate {question:.3f} out of [.28,.42]"


def test_surface_near_zero_code_and_emoji():
    turns = _gen_turns()
    assert _frac(turns, lambda t: "```" in t.text) < 0.01          # fenced code ~0
    assert _frac(turns, lambda t: "Traceback (most" in t.text) == 0  # tracebacks ~0
    assert _frac(turns, lambda t: any(ord(c) > 0x2600 for c in t.text)) == 0.0  # emoji ~0


def test_inline_path_present_but_minority():
    turns = _gen_turns()
    has_path = _frac(turns, lambda t: t.has_path)
    assert 0.06 <= has_path <= 0.18, f"inline-path share {has_path:.3f} (corpus ~0.13)"


# ── JOINT anti-chimera checks (the whole point of archetype-based sampling) ────
def test_joint_no_chimeras():
    turns = _gen_turns()
    by_arch = {}
    for t in turns:
        by_arch.setdefault(t.archetype, []).append(t)

    # terse archetypes are SHORT and UNPUNCTUATED and marked fragments — never
    # a long punctuated essay.
    for arch in ("terse-poll", "approval-nudge"):
        for t in by_arch[arch]:
            assert t.word_count <= 5, f"{arch} produced {t.word_count} words: {t.text!r}"
            assert not t.has_terminal_punct, f"{arch} kept terminal punct: {t.text!r}"
            assert t.is_fragment

    # the long-plan archetype is a real pasted plan: long, NOT an all-lowercase
    # verbless fragment (the exact chimera independent flags would manufacture).
    for t in by_arch["long-plan-paste"]:
        assert t.word_count >= 30, f"long-plan too short: {t.word_count}"
        assert not t.is_lowercase, f"long-plan went all-lowercase: {t.text[:60]!r}"
        assert not t.is_fragment

    # disfluent run-ons actually carry dictation disfluency and are not terse.
    for t in by_arch["disfluent-runon"]:
        assert t.is_disfluent
        assert t.word_count >= 10


def test_archetype_mix_is_broad():
    turns = _gen_turns()
    seen = Counter(t.archetype for t in turns)
    # every declared archetype must actually fire (no dead branch)
    assert set(seen) == set(ARCHETYPES), f"missing archetypes: {set(ARCHETYPES) - set(seen)}"
    # no single archetype dominates the surface
    assert max(seen.values()) / len(turns) < 0.30


# ── determinism + gradability ─────────────────────────────────────────────────
def test_determinism_given_seed():
    a = [t.text for t in _gen_turns(500, seed=123)]
    b = [t.text for t in _gen_turns(500, seed=123)]
    c = [t.text for t in _gen_turns(500, seed=124)]
    assert a == b, "same seed must reproduce identical turns"
    assert a != c, "different seed must diverge"


def test_completion_marker_enforced_for_gradability():
    rng = random.Random(7)
    for _ in range(50):
        t = generate_turn("wrap it up and confirm done", "sure", rng=rng, llm=stub_llm,
                          require_completion_marker=True)
        from simulator.driver import COMPLETION_MARKER
        assert COMPLETION_MARKER.search(t.text), f"marker missing: {t.text!r}"


# ── episode structure (thread-structured drift, interrupts, length) ───────────
def test_session_length_long_tailed():
    rng = random.Random(3)
    lens = sorted(sample_length(rng) for _ in range(5000))
    p50, p90 = _pct(lens, 0.50), _pct(lens, 0.90)
    assert 18 <= p50 <= 30, f"session-length p50 {p50} out of [18,30]"
    assert p90 >= 100, f"session-length p90 {p90} — tail must cross 100"
    assert max(lens) <= 221, "length must stay within corpus support"
    assert min(lens) >= 1 and _pct(lens, 0.0) >= 1  # 0% single-turn handled by clamp floor


def test_episode_threads_labels_and_interrupts():
    rng = random.Random(11)
    kinds = Counter()
    thread_counts, total = [], 0
    for _ in range(300):
        ep = generate_episode(rng)
        assert 2 <= ep.n_threads <= 4
        thread_counts.append(ep.n_threads)
        total += len(ep.steps)
        for s in ep.steps:
            kinds[s.kind] += 1
            assert s.expected_label, "every step must carry a gradable expected_label"
            if s.kind == "scenario":
                assert s.scenario, "seed step must reference a tasks.py scenario"
    interrupt_rate = kinds["interrupt"] / total
    assert 0.015 <= interrupt_rate <= 0.04, f"interrupt rate {interrupt_rate:.4f} out of band"
    assert kinds["driven"] > kinds["scenario"], "driven follow-ups must dominate seeds"
    assert set(thread_counts) == {2, 3, 4}


def test_topic_drift_is_thread_structured():
    """Global median consec-turn Jaccard collapses toward 0 (bursty switches),
    yet sticky runs keep some neighbouring pairs coherent (not forced drift)."""
    rng = random.Random(5)
    session_medians, cross_fracs = [], []
    for _ in range(300):
        ep = generate_episode(rng)
        js, cross = [], 0
        for a, b in zip(ep.steps, ep.steps[1:]):
            union = a.topic | b.topic
            js.append(len(a.topic & b.topic) / len(union) if union else 0.0)
            cross += a.thread != b.thread
        if js:
            session_medians.append(statistics.median(js))
            cross_fracs.append(cross / len(js))
    assert statistics.median(session_medians) < 0.10, "topic drift too coherent"
    mean_cross = statistics.mean(cross_fracs)
    assert 0.4 < mean_cross < 0.95, (
        f"cross-thread frac {mean_cross:.3f}: sticky runs must exist (local coherence) "
        "but switches must dominate (global drift)")


def test_clarifying_branch_helpers():
    assert answer_is_question("so should i use a Settings class?")
    assert not answer_is_question("done, tests pass.")
    cs = clarifying_step(2, frozenset({"limiter"}))
    assert cs.kind == "driven" and cs.expected_label == "clarifying_answer"


# ── integration: the surface reproduces THROUGH the episode pipeline ──────────
def test_episode_pipeline_reproduces_surface():
    """Render the driven turns of real generated episodes and confirm the core
    surface bands still hold end-to-end (not just for isolated driver draws)."""
    rng = random.Random(99)
    turns = []
    while len(turns) < 1500:
        ep = generate_episode(rng)
        last = "ok here is what i found"
        for s in ep.steps:
            if s.kind != "driven":
                continue
            t = generate_turn(s.intent, last, rng=rng, llm=stub_llm,
                              archetype=s.archetype, sandbox_paths=_SANDBOX_PATHS,
                              require_completion_marker=s.require_completion_marker)
            turns.append(t)
            last = "and here is the next step, does that work?"
    wc = sorted(t.word_count for t in turns)
    p50 = _pct(wc, 0.50)
    no_punct = _frac(turns, lambda t: not t.has_terminal_punct)
    lower = _frac(turns, lambda t: t.is_lowercase)
    # slightly wider bands: episode hints (labeled follow-ups) tilt the mix a bit
    assert 7 <= p50 <= 13, f"pipeline word p50 {p50}"
    assert 0.50 <= no_punct <= 0.78, f"pipeline no-terminal-punct {no_punct:.3f}"
    assert 0.18 <= lower <= 0.34, f"pipeline all-lowercase {lower:.3f}"
