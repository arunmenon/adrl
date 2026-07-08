"""The LLM classifier resolves the ambiguous middle — injected, fail-safe."""

from dataclasses import dataclass

from router.features import extract
from router.policy import route_turn, SessionState


@dataclass
class StubVerdict:
    tier: str
    needs_frontier: bool


def _middle_feature():
    # an unknown verb -> verb_score 0.5 -> lands in the ambiguous middle band
    f = extract("do the thing with the widget", context_tokens=3000)
    assert f.verb_class == "unknown"        # guard: genuinely the middle
    assert f.instruction_text               # text carried through for the classifier
    return f


def test_classifier_hard_routes_frontier():
    f = _middle_feature()
    r = route_turn(f, SessionState("s"), classifier=lambda t: StubVerdict("hard", True))
    assert r.rung == "frontier" and r.layer == "classifier" and not r.cascade


def test_classifier_local_routes_local_with_cascade():
    f = _middle_feature()
    r = route_turn(f, SessionState("s"), classifier=lambda t: StubVerdict("standard", False))
    assert r.rung == "local" and r.layer == "classifier" and r.cascade


def test_classifier_none_falls_back_to_middle_default():
    f = _middle_feature()
    r = route_turn(f, SessionState("s"), classifier=lambda t: None)   # abstained
    assert r.rung == "local" and r.layer == "middle_default"


def test_no_classifier_is_backwards_compatible():
    f = _middle_feature()
    r = route_turn(f, SessionState("s"))          # no classifier wired
    assert r.layer == "middle_default"


def test_classifier_never_consulted_when_a_gate_fires():
    # privacy pin must win before the middle is ever reached
    f = _middle_feature()
    called = []
    def spy(t):
        called.append(t); return StubVerdict("trivial", False)
    r = route_turn(f, SessionState("s", privacy_pinned=True), classifier=spy)
    assert r.layer == "gate:privacy" and not called   # classifier not called


def test_classifier_not_consulted_on_clear_heuristic_end():
    # an obvious-easy turn is decided by Layer 1; the classifier never runs
    f = extract("write a commit message", context_tokens=1000)
    called = []
    r = route_turn(f, SessionState("s"), classifier=lambda t: called.append(t))
    assert r.layer == "heuristic" and r.rung == "local" and not called


def test_terse_approval_sticks_not_frontier():
    # "go ahead" / "do it" must NOT be cold-classified to frontier
    for msg in ("go ahead", "Do it", "try now", "lgtm", "and now", "proceed"):
        f = extract(msg, context_tokens=3000)
        assert f.is_terse_continuation, msg
        # even with a classifier that would say frontier, the continuation rule wins first
        r = route_turn(f, SessionState("s", route="local"),
                       classifier=lambda t: StubVerdict("hard", True))
        assert r.layer == "continuation" and r.rung == "local", f"{msg} -> {r.rung}/{r.layer}"

def test_real_task_not_mistaken_for_approval():
    f = extract("fix the failing test", context_tokens=3000)
    assert not f.is_terse_continuation
