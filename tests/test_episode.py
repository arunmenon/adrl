"""Conservative episode-boundary scoring."""

from router.episode import detect_episode_boundary, file_references
from router.features import extract
from router.policy import SessionState


def test_three_independent_signals_release_episode():
    session = SessionState(
        "s", escalated_this_episode=True, last_turn_clean=True,
        episode_verb_class="hard", episode_files=("src/router.py",),
    )
    boundary = detect_episode_boundary(
        extract("great, that's done. now write README.md"), session)
    assert boundary.is_boundary
    assert boundary.previous_clean
    assert boundary.completion_phrase
    assert boundary.intent_changed


def test_completion_phrase_alone_does_not_release_episode():
    session = SessionState("s", escalated_this_episode=True)
    boundary = detect_episode_boundary(
        extract("that's done, now continue"), session)
    assert not boundary.is_boundary
    assert boundary.score == 1


def test_disjoint_file_references_supply_fourth_signal():
    session = SessionState(
        "s", last_turn_clean=True, episode_verb_class="fix",
        episode_files=("src/auth.py",),
    )
    boundary = detect_episode_boundary(
        extract("great, that's done. now write docs/router.md"), session)
    assert boundary.low_file_overlap
    assert file_references("update src/auth.py and docs/router.md") == (
        "docs/router.md", "src/auth.py")
