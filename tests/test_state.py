"""C3 — tests for the session state store (design §5.6, §6).

Covers the four behaviours the store exists to guarantee:
  * unknown session auto-creates
  * route set/get (the sticky lookup)
  * strikes increment atomically and reset per turn
  * the privacy pin is ONE-WAY — clearable only by a fresh session
plus the escalation flag, turn counter, and the no-op idle-eviction hook.
"""

import time
from abc import ABC

import pytest

from router.policy import SessionState
from router.state import IDLE_TTL_S, DictSessionStore, SessionStore


@pytest.fixture
def store() -> DictSessionStore:
    return DictSessionStore()


# ── interface shape ────────────────────────────────────────────────────────
def test_abc_cannot_be_instantiated():
    assert issubclass(SessionStore, ABC)
    with pytest.raises(TypeError):
        SessionStore()  # abstract — must not be constructible directly


def test_dict_store_is_a_session_store(store):
    assert isinstance(store, SessionStore)


# ── auto-create ────────────────────────────────────────────────────────────
def test_unknown_session_auto_creates(store):
    state = store.get_session("s_new")
    assert isinstance(state, SessionState)
    assert state.session_id == "s_new"
    # fresh defaults per the dataclass
    assert state.route == "local"
    assert state.privacy_pinned is False
    assert state.escalated_this_episode is False
    assert state.turn_count == 0
    assert state.strikes == {}
    assert state.continuation_count == 0


def test_get_session_is_stable_same_object(store):
    first = store.get_session("s_1")
    first.turn_count = 5
    second = store.get_session("s_1")
    assert second is first  # canonical live object, not a fresh copy
    assert second.turn_count == 5


def test_mutators_auto_create_unknown_session(store):
    # a mutator on an unseen id must not KeyError — it creates then acts
    assert store.incr_strike("never_seen", "edit") == 1
    assert store.set_route("also_new", "frontier").route == "frontier"


# ── route set/get (sticky lookup) ──────────────────────────────────────────
def test_set_route_then_get(store):
    returned = store.set_route("s_2", "frontier")
    assert returned.route == "frontier"
    assert store.get_session("s_2").route == "frontier"


def test_set_route_overwrites(store):
    store.set_route("s_2", "cheap_cloud")
    store.set_route("s_2", "frontier")
    assert store.get_session("s_2").route == "frontier"


# ── strikes: increment + per-turn reset ────────────────────────────────────
def test_incr_strike_returns_running_count(store):
    assert store.incr_strike("s_3", "edit") == 1
    assert store.incr_strike("s_3", "edit") == 2
    assert store.incr_strike("s_3", "edit") == 3
    assert store.get_session("s_3").strikes["edit"] == 3


def test_incr_strike_independent_per_kind(store):
    store.incr_strike("s_3", "edit")
    store.incr_strike("s_3", "edit")
    store.incr_strike("s_3", "parse")
    strikes = store.get_session("s_3").strikes
    assert strikes == {"edit": 2, "parse": 1}


def test_reset_strikes_clears_all_kinds(store):
    store.incr_strike("s_4", "edit")
    store.incr_strike("s_4", "loop")
    store.reset_strikes("s_4")
    assert store.get_session("s_4").strikes == {}
    # and counting resumes from zero after a reset (per-turn semantics)
    assert store.incr_strike("s_4", "edit") == 1


def test_reset_strikes_does_not_touch_route_or_pin(store):
    store.set_route("s_4", "frontier")
    store.pin_privacy("s_4")
    store.incr_strike("s_4", "edit")
    store.reset_strikes("s_4")
    state = store.get_session("s_4")
    assert state.route == "frontier"
    assert state.privacy_pinned is True


def test_set_strikes_replaces_with_canonical_snapshot(store):
    store.incr_strike("s_sync", "edit")
    store.set_strikes("s_sync", {"parse": 1, "edit": 2, "loop": 0})
    assert store.get_session("s_sync").strikes == {
        "parse": 1, "edit": 2, "loop": 0,
    }


def test_continuation_count_is_per_turn(store):
    assert store.incr_continuation("s_cont") == 1
    assert store.incr_continuation("s_cont") == 2
    store.reset_continuations("s_cont")
    assert store.get_session("s_cont").continuation_count == 0


def test_incr_strike_atomic_under_threads(store):
    # §6.3: concurrent continuations must not lose strike increments.
    import threading

    def bump():
        for _ in range(200):
            store.incr_strike("s_race", "edit")

    threads = [threading.Thread(target=bump) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.get_session("s_race").strikes["edit"] == 8 * 200


# ── privacy pin: ONE-WAY ───────────────────────────────────────────────────
def test_pin_privacy_sets_flag(store):
    assert store.get_session("s_5").privacy_pinned is False
    store.pin_privacy("s_5")
    assert store.get_session("s_5").privacy_pinned is True


def test_pin_is_idempotent(store):
    store.pin_privacy("s_5")
    store.pin_privacy("s_5")
    assert store.get_session("s_5").privacy_pinned is True


def test_pin_survives_direct_field_clear_attempt(store):
    # An attempt to clear the pin by flipping the dataclass field must be
    # ignored: the store re-asserts the one-way pin on the next read (§5.6/§9).
    store.pin_privacy("s_6")
    store.get_session("s_6").privacy_pinned = False  # the illegitimate "un-pin"
    assert store.get_session("s_6").privacy_pinned is True


def test_pin_survives_route_and_strike_churn(store):
    store.pin_privacy("s_6")
    store.set_route("s_6", "frontier")
    store.incr_strike("s_6", "edit")
    store.reset_strikes("s_6")
    store.mark_escalated("s_6")
    assert store.get_session("s_6").privacy_pinned is True


def test_pin_is_per_session_not_global(store):
    store.pin_privacy("s_pinned")
    assert store.get_session("s_other").privacy_pinned is False


def test_fresh_session_is_the_only_way_to_clear_pin(store):
    store.pin_privacy("s_7")
    assert store.get_session("s_7").privacy_pinned is True
    fresh = store.start_fresh_session("s_7")
    assert fresh.privacy_pinned is False
    # and the cleared state is durable — a later read stays un-pinned
    assert store.get_session("s_7").privacy_pinned is False


def test_no_public_unpin_method(store):
    # There is deliberately no un-pin verb on the interface.
    assert not hasattr(store, "unpin_privacy")
    assert not hasattr(store, "clear_privacy")


# ── escalation + turn counter ──────────────────────────────────────────────
def test_mark_escalated(store):
    assert store.get_session("s_8").escalated_this_episode is False
    store.mark_escalated("s_8")
    assert store.get_session("s_8").escalated_this_episode is True


def test_incr_turn(store):
    assert store.incr_turn("s_9") == 1
    assert store.incr_turn("s_9") == 2
    assert store.get_session("s_9").turn_count == 2


def test_start_fresh_session_resets_everything(store):
    store.set_route("s_10", "frontier")
    store.incr_strike("s_10", "edit")
    store.mark_escalated("s_10")
    store.incr_turn("s_10")
    fresh = store.start_fresh_session("s_10")
    assert fresh.route == "local"
    assert fresh.strikes == {}
    assert fresh.escalated_this_episode is False
    assert fresh.turn_count == 0


def test_new_episode_releases_hysteresis_but_preserves_session_and_privacy(store):
    store.pin_privacy("episode")
    store.set_route("episode", "frontier")
    store.mark_escalated("episode")
    store.incr_turn("episode")
    store.record_episode_intent("episode", "hard", ("src/auth.py",))
    store.mark_turn_clean("episode", True)
    state = store.start_new_episode("episode")
    assert state.route == "local"
    assert state.escalated_this_episode is False
    assert state.privacy_pinned is True
    assert state.turn_count == 1
    assert state.episode_index == 1
    assert state.episode_verb_class == ""
    assert state.episode_files == ()
    assert state.last_turn_clean is False


# ── touch / expire: documented no-op-now hook ──────────────────────────────
def test_touch_auto_creates_and_does_not_evict(store):
    store.touch("s_11")
    assert "s_11" in store._sessions


def test_expire_idle_is_a_noop_now(store):
    # The dict impl never actively evicts (§6): even a session idle well past
    # the TTL survives expire_idle, which reports zero evictions.
    store.get_session("s_12")
    future = time.time() + 10 * IDLE_TTL_S  # far beyond 4h idle
    evicted = store.expire_idle(now=future)
    assert evicted == 0
    assert store.get_session("s_12").session_id == "s_12"  # still present


def test_idle_sessions_reports_without_evicting(store):
    store.get_session("s_13")
    future = time.time() + 10 * IDLE_TTL_S
    assert "s_13" in store.idle_sessions(now=future)
    # read-only: still there afterwards
    assert "s_13" in store._sessions
