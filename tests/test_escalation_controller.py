"""Live escalation controller: trip-wire fires -> sticky route flips to cloud."""

from router.escalation_controller import EscalationController
from router.state import DictSessionStore

EDIT_FAIL = {"type": "tool_result", "is_error": True,
             "content": "<tool_use_error>String to replace not found in file.\nString: x"}


def _edit_fail_response():
    return [{"type": "tool_use", "id": "t1", "name": "Edit",
             "input": {"old_string": "a", "new_string": "b"}}]


def test_two_edit_fails_escalate_local_to_cloud():
    store = DictSessionStore()
    store.set_route("s1", "local-code")
    ec = EscalationController(store)
    ec.new_turn("s1")
    # strike 1: no escalation yet
    d = ec.observe_tool_results("s1", [EDIT_FAIL])
    assert not d.escalate and ec.current_route("s1") == "local-code"
    # Partial evidence is mirrored before the threshold fires.
    assert store.get_session("s1").strikes["edit"] == 1
    # strike 2: edit-apply trip-wire fires -> escalate to cheap-cloud, sticky
    d = ec.observe_tool_results("s1", [EDIT_FAIL])
    assert d.escalate and d.to_rung == "cheap-cloud" and d.tripwire == "edit_apply"
    assert d.tripwire_type == "dialect"
    assert ec.current_route("s1") == "cheap-cloud"          # route flipped
    assert store.get_session("s1").escalated_this_episode   # hysteresis


def test_escalation_is_sticky():
    store = DictSessionStore()
    store.set_route("s2", "local-code")
    ec = EscalationController(store)
    ec.new_turn("s2")
    ec.observe_tool_results("s2", [EDIT_FAIL])
    ec.observe_tool_results("s2", [EDIT_FAIL])
    assert ec.current_route("s2") == "cheap-cloud"
    # Observing another response while the hit remains latched does not climb
    # cheap-cloud -> frontier in the same turn.
    d = ec.observe_response("s2", [{"type": "text", "text": "continuing"}])
    assert not d.escalate
    assert ec.current_route("s2") == "cheap-cloud"
    # a subsequent clean turn stays escalated (episode hysteresis)
    ec.new_turn("s2")
    assert ec.current_route("s2") == "cheap-cloud"


def test_privacy_pin_blocks_cloud_escalation():
    store = DictSessionStore()
    store.set_route("s3", "local-code")
    store.pin_privacy("s3")
    ec = EscalationController(store)
    ec.new_turn("s3")
    ec.observe_tool_results("s3", [EDIT_FAIL])
    d = ec.observe_tool_results("s3", [EDIT_FAIL])
    assert d.escalate and d.to_user and d.to_rung is None    # to the user, never cloud
    assert ec.current_route("s3") == "local-code"            # NOT escalated to cloud


def test_clean_turn_no_escalation():
    store = DictSessionStore()
    store.set_route("s4", "local-code")
    ec = EscalationController(store)
    ec.new_turn("s4")
    d = ec.observe_response("s4", [{"type": "text", "text": "done, tests pass"}])
    assert not d.escalate and ec.current_route("s4") == "local-code"


def test_isolation_between_sessions():
    store = DictSessionStore()
    for s in ("a", "b"):
        store.set_route(s, "local-code")
    ec = EscalationController(store)
    ec.new_turn("a"); ec.new_turn("b")
    ec.observe_tool_results("a", [EDIT_FAIL])
    ec.observe_tool_results("a", [EDIT_FAIL])
    assert ec.current_route("a") == "cheap-cloud"
    assert ec.current_route("b") == "local-code"   # b unaffected


def test_snapshot_carries_cumulative_errors_and_continuations():
    store = DictSessionStore()
    ec = EscalationController(store)
    ec.new_turn("evidence")
    ec.note_continuation("evidence")
    ec.observe_tool_results("evidence", [
        {"type": "tool_result", "is_error": True, "content": "pytest failed"},
        EDIT_FAIL,
    ])
    snap = ec.snapshot("evidence")
    assert snap.error_results == 2
    assert snap.edit_failures == 1
    assert snap.continuation_count == 1
    assert snap.strikes["edit"] == 1

    ec.new_turn("evidence")
    reset = ec.snapshot("evidence")
    assert reset.error_results == 0
    assert reset.continuation_count == 0
