"""C1 — tests for the trip-wire evaluator (design §5.5).

The edit-apply wire is tested against REAL corpus records: the two-strike fire
case, the one-strike no-fire case, and the is_error guard that rejects turns the
miner mislabeled from a document that merely quotes the marker. The loop /
no-progress / budget / interrupt wires are exercised with synthetic streams
(local-model malformed traffic is not in the corpus — S5 needs workstream C).
"""

import hashlib
import json
from pathlib import Path

import pytest

from router.tripwires import (
    EDIT_FAIL_MARKER,
    TRIPWIRE_TYPES,
    TripwireState,
    TripwireType,
    TurnBudget,
    routes_to_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "data" / "corpus"
TURNS = REPO_ROOT / "data" / "turns.parquet"


# ── real-data helpers ────────────────────────────────────────────────────────
def _iter_real_edit_error_blocks(source_path: str):
    """Yield the *real* is_error edit-fail tool_result blocks in one corpus file
    (marker present AND is_error truthy) — i.e. what the wire must fire on."""
    path = CORPUS / source_path
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if EDIT_FAIL_MARKER not in line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for blk in content:
                if (
                    isinstance(blk, dict)
                    and blk.get("type") == "tool_result"
                    and blk.get("is_error")
                    and EDIT_FAIL_MARKER in str(blk.get("content", ""))
                ):
                    yield blk


def _edit_fail_files():
    """Map corpus source_path -> count of real is_error edit-fail blocks, for the
    turns.parquet rows the miner flagged (n_edit_failures >= 1)."""
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    if not TURNS.exists():
        pytest.skip(f"missing {TURNS}")
    rows = pq.read_table(TURNS).to_pylist()
    paths = {r["source_path"] for r in rows if (r.get("n_edit_failures") or 0) >= 1}
    return {p: len(list(_iter_real_edit_error_blocks(p))) for p in paths}


def _err(blob, *, tool_use_id="toolu_x", is_error=True):
    return {"type": "tool_result", "tool_use_id": tool_use_id,
            "content": blob, "is_error": is_error}


def _ok(blob, *, tool_use_id="toolu_x"):
    return {"type": "tool_result", "tool_use_id": tool_use_id,
            "content": blob, "is_error": False}


def _use(name, args, *, tool_use_id="toolu_x"):
    return {"type": "tool_use", "id": tool_use_id, "name": name, "input": args}


# ── edit-apply wire on REAL data ─────────────────────────────────────────────
def test_edit_apply_fires_at_two_real_error_results():
    files = _edit_fail_files()
    two_strike = next((p for p, n in files.items() if n >= 2), None)
    assert two_strike, f"corpus has no turn with >=2 real edit failures: {files}"

    blocks = list(_iter_real_edit_error_blocks(two_strike))[:2]
    assert len(blocks) == 2

    tw = TripwireState()
    # first real failure: one strike, must NOT fire (threshold is 2)
    assert tw.observe_tool_results([blocks[0]]) is None
    assert tw.fired() is None
    assert tw.strikes["edit"] == 1

    # second real failure: 2 strikes -> edit_apply fires, typed as a dialect miss
    hit = tw.observe_tool_results([blocks[1]])
    assert hit is not None
    assert hit.name == "edit_apply"
    assert hit.type is TripwireType.DIALECT
    assert hit.detail["strikes"] == 2
    assert hit.detail["marker"] == EDIT_FAIL_MARKER
    # dialect failures train the registry, not the difficulty model (§5.7/B5)
    assert routes_to_registry(hit.type) is True


def test_edit_apply_one_real_strike_does_not_fire():
    files = _edit_fail_files()
    one_strike = next((p for p, n in files.items() if n == 1), None)
    assert one_strike, f"corpus has no turn with exactly 1 real edit failure: {files}"

    block = next(_iter_real_edit_error_blocks(one_strike))
    tw = TripwireState()
    assert tw.observe_tool_results([block]) is None
    assert tw.fired() is None
    assert tw.strikes["edit"] == 1


def test_is_error_required_rejects_documentation_mention():
    """The marker string appears verbatim in this design doc; the miner counted
    3 such Read results as edit failures. The wire's is_error guard must reject
    them — a non-error result carrying the marker is a mention, not a failure."""
    mention = _ok("1\t" + EDIT_FAIL_MARKER + " (a line quoted from the design doc)")
    tw = TripwireState()
    # feed it twice — even two mentions must not reach the 2-strike threshold
    tw.observe_tool_results([mention])
    tw.observe_tool_results([mention])
    assert tw.fired() is None
    assert tw.strikes["edit"] == 0


# ── loop wire (canonicalization reuse) ───────────────────────────────────────
def test_loop_fires_on_three_canonically_identical_calls():
    tw = TripwireState()
    # cosmetically different (trailing slashes) but canonically the same call
    r1 = tw.observe_response([_use("Read", {"file_path": "/a/b"}, tool_use_id="t1")])
    r2 = tw.observe_response([_use("Read", {"file_path": "/a/b/"}, tool_use_id="t2")])
    assert r1 is None and r2 is None  # 2 identical -> not yet a loop
    hit = tw.observe_response([_use("Read", {"file_path": "/a/b//"}, tool_use_id="t3")])
    assert hit is not None
    assert hit.name == "loop"
    assert hit.type is TripwireType.DIFFICULTY
    assert hit.detail["repeats"] >= 3
    assert routes_to_registry(hit.type) is False  # difficulty -> router, not registry


def test_loop_does_not_fire_on_distinct_calls():
    tw = TripwireState()
    for i in range(6):
        tw.observe_response([_use("Read", {"file_path": f"/f/{i}"}, tool_use_id=f"t{i}")])
    assert tw.fired() is None


# ── no-progress wire ─────────────────────────────────────────────────────────
def test_no_progress_fires_after_six_stuck_actions():
    tw = TripwireState()
    # six actions that each produce nothing new: identical failing command output
    for i in range(5):
        assert tw.observe_tool_results([_err("command failed: exit 1", tool_use_id=f"e{i}")]) is None
    hit = tw.observe_tool_results([_err("command failed: exit 1", tool_use_id="e5")])
    assert hit is not None
    assert hit.name == "no_progress"
    assert hit.type is TripwireType.DIFFICULTY
    assert hit.detail["actions"] == 6


def test_no_progress_counter_resets_on_real_progress():
    tw = TripwireState()
    for i in range(5):
        tw.observe_tool_results([_err("stuck", tool_use_id=f"a{i}")])
    assert tw.strikes["noprog"] == 5
    # a novel, non-error output = progress -> counter resets
    tw.observe_tool_results([_ok("brand new output that advances things", tool_use_id="p")])
    assert tw.strikes["noprog"] == 0
    assert tw.fired() is None
    # now it takes a fresh run of six stuck actions to fire
    for i in range(5):
        assert tw.observe_tool_results([_err("stuck again", tool_use_id=f"b{i}")]) is None
    assert tw.observe_tool_results([_err("stuck again", tool_use_id="b5")]).name == "no_progress"


def test_repeated_identical_success_output_is_not_progress():
    tw = TripwireState()
    same = _ok("identical output", tool_use_id="s")
    # first occurrence is progress; the next six identical ones are not
    tw.observe_tool_results([same])
    for i in range(5):
        assert tw.observe_tool_results([_ok("identical output", tool_use_id=f"s{i}")]) is None
    assert tw.observe_tool_results([_ok("identical output", tool_use_id="s6")]).name == "no_progress"


# ── turn-budget wire (parameterized, nothing hardcoded) ──────────────────────
def test_turn_budget_fires_on_tokens():
    tw = TripwireState(budget=TurnBudget(max_tokens=100))
    assert tw.observe_response([], output_tokens=60) is None
    hit = tw.observe_response([], output_tokens=60)  # cumulative 120 > 100
    assert hit is not None
    assert hit.name == "turn_budget"
    assert hit.type is TripwireType.COST
    assert hit.detail["reason"] == "tokens"


def test_turn_budget_fires_on_wall_clock():
    tw = TripwireState(budget=TurnBudget(max_wall_clock_s=90))
    assert tw.observe_response([], elapsed_s=45) is None
    hit = tw.observe_response([], elapsed_s=95)
    assert hit is not None and hit.name == "turn_budget"
    assert hit.detail["reason"] == "wall_clock"


def test_turn_budget_inert_without_budget():
    tw = TripwireState()  # no budget supplied
    for _ in range(50):
        tw.observe_response([], output_tokens=100_000, elapsed_s=10_000)
    assert tw.fired() is None


# ── user-interrupt wire ──────────────────────────────────────────────────────
def test_user_interrupt_note_fires_once():
    tw = TripwireState()
    hit = tw.note_interrupt("actually, do it differently")
    assert hit is not None
    assert hit.name == "user_interrupt"
    assert hit.type is TripwireType.QUALITY


def test_user_interrupt_from_leading_text_block():
    tw = TripwireState()
    hit = tw.observe_tool_results(
        [{"type": "text", "text": "[Request interrupted by user for tool use]"}]
    )
    assert hit is not None and hit.name == "user_interrupt"


def test_user_interrupt_from_bare_string():
    tw = TripwireState()
    assert tw.observe_tool_results("[Request interrupted by user]").name == "user_interrupt"


# ── parse/bad-schema wire (synthetic: local-model malformed output) ──────────
def test_parse_schema_fires_on_two_malformed_tool_calls():
    tw = TripwireState()
    # input is not a JSON object (a raw string the local model emitted)
    assert tw.observe_response([_use("Edit", "not-a-dict", tool_use_id="m1")]) is None
    hit = tw.observe_response([_use("", {}, tool_use_id="m2")])  # empty name too
    assert hit is not None
    assert hit.name == "parse_schema"
    assert hit.type is TripwireType.DIALECT
    assert routes_to_registry(hit.type) is True


def test_parse_schema_secondary_marker_in_error_result():
    tw = TripwireState()
    tw.observe_tool_results([_err("<tool_use_error>Input validation error: foo", tool_use_id="v1")])
    hit = tw.observe_tool_results([_err("Input validation error: bar", tool_use_id="v2")])
    assert hit is not None and hit.name == "parse_schema"


def test_valid_tool_calls_never_trip_parse():
    tw = TripwireState()
    for i in range(4):
        tw.observe_response([_use("Read", {"file_path": f"/x/{i}"}, tool_use_id=f"ok{i}")])
    assert tw.fired() is None
    assert tw.strikes["parse"] == 0


# ── latching, reset, and type map ────────────────────────────────────────────
def test_first_wire_latches_and_is_not_overwritten():
    tw = TripwireState()
    # trip edit_apply first
    tw.observe_tool_results([_err("<tool_use_error>" + EDIT_FAIL_MARKER, tool_use_id="e1")])
    tw.observe_tool_results([_err("<tool_use_error>" + EDIT_FAIL_MARKER, tool_use_id="e2")])
    assert tw.fired().name == "edit_apply"
    # now drive an obvious loop — must NOT overwrite the latched edit_apply
    for i in range(4):
        tw.observe_response([_use("Read", {"file_path": "/same"}, tool_use_id=f"l{i}")])
    assert tw.fired().name == "edit_apply"


def test_reset_clears_all_state():
    tw = TripwireState()
    tw.observe_tool_results([_err("<tool_use_error>" + EDIT_FAIL_MARKER, tool_use_id="e1")])
    tw.observe_tool_results([_err("<tool_use_error>" + EDIT_FAIL_MARKER, tool_use_id="e2")])
    assert tw.fired() is not None
    tw.reset()
    assert tw.fired() is None
    assert tw.strikes == {"parse": 0, "edit": 0, "loop": 0, "noprog": 0}
    # usable again after reset
    tw.observe_tool_results([_err("<tool_use_error>" + EDIT_FAIL_MARKER, tool_use_id="e3")])
    assert tw.fired() is None  # only one strike post-reset


def test_type_map_matches_registry_routing():
    assert TRIPWIRE_TYPES["edit_apply"] is TripwireType.DIALECT
    assert TRIPWIRE_TYPES["parse_schema"] is TripwireType.DIALECT
    assert TRIPWIRE_TYPES["loop"] is TripwireType.DIFFICULTY
    assert TRIPWIRE_TYPES["no_progress"] is TripwireType.DIFFICULTY
    assert TRIPWIRE_TYPES["turn_budget"] is TripwireType.COST
    assert TRIPWIRE_TYPES["user_interrupt"] is TripwireType.QUALITY
    # exactly the two dialect wires route to the capability registry
    dialect = {n for n, t in TRIPWIRE_TYPES.items() if routes_to_registry(t)}
    assert dialect == {"edit_apply", "parse_schema"}
