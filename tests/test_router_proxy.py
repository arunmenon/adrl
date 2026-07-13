"""WS2 UNIT B — unit tests for the LIVE routing mode in proxy.capture_proxy.

These pin the routing-mode seams the workflow added on top of the capture-only
proxy, with NO live services (no ollama, no LiteLLM, no Anthropic, no real
sockets to upstreams). Three kinds of test:

  1. PURE decision/dispatch seams, exercised directly:
       * ``liveplan_attempts`` — a ``router.live_router.RoutePlan`` -> the ordered
         (upstream, body) attempt list, with the model rewritten per rung and the
         Anthropic fail-open preserved for local rungs.
       * ``acquire_upstream`` (reused from P1-A) over those attempts, with an
         *injected* ``send`` — proves local failure falls open to Anthropic with
         the ORIGINAL body, exactly like the utility path.
       * ``session_id_for`` / ``reconstruct_blocks`` — the session key + the SSE/
         JSON block reconstruction the escalation controller observes.

  2. ``make_app`` WIRING — flag OFF constructs NO router live-decision object
     (byte-identical capture-only); flag ON wires the injected singletons.

  3. ``handle`` END-TO-END over an aiohttp TestServer with a FAKE upstream
     client — proves a routed user_turn (a) starts a trip-wire turn, (b) feeds
     ``escalation.observe_response`` the reconstructed blocks, (c) attaches the
     outcome to the recorder, (d) stamps routed_rung/route_layer/route_id into
     the capture — and that flag-OFF stays capture-only (no routing fields).

Run: PYTHONPATH=src .venv/bin/python -m pytest tests/test_router_proxy.py -q
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from proxy.capture_proxy import (
    acquire_upstream,
    liveplan_attempts,
    make_app,
    prepare_escalation_body,
    reconstruct_blocks,
    session_id_for,
)
from router.escalation_controller import EscalationController
from router.live_router import LiveRouter, RoutePlan as LiveRoutePlan
from router.memory_facade import RouterMemory, RoutingRecorder
from router.memory_sqlite import SqliteProvider
from router.state import DictSessionStore

ANTHROPIC = "https://api.anthropic.com"
LITELLM = "http://localhost:4001"
PATH = "/v1/messages"


def _run(coro):
    return asyncio.run(coro)


def _user_turn_body(sid: str | None = "sess-abc") -> dict:
    body: dict = {
        "model": "claude-opus-4-20250514",
        "max_tokens": 4096,
        "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": "Refactor the auth module."}],
    }
    if sid is not None:
        body["metadata"] = {"user_id": json.dumps({"session_id": sid})}
    return body


def _router() -> LiveRouter:
    # A pure LiveRouter, only used here for build_forward_body (no I/O).
    return LiveRouter(litellm_url=LITELLM, anthropic_url=ANTHROPIC)


def _local_plan() -> LiveRoutePlan:
    return LiveRoutePlan(
        primary_model="local-code", primary_upstream=LITELLM,
        fallback_model=None, fallback_upstream=ANTHROPIC,
        rung="local", label="user_turn", route_id="route-xyz", layer="L1-features")


def _cheap_cloud_plan() -> LiveRoutePlan:
    return LiveRoutePlan(
        primary_model="claude-haiku-4-5", primary_upstream=ANTHROPIC,
        fallback_model=None, fallback_upstream=None,
        rung="cheap_cloud", label="user_turn", route_id="route-c", layer="L1-features")


def _frontier_plan() -> LiveRoutePlan:
    return LiveRoutePlan(
        primary_model=None, primary_upstream=ANTHROPIC,
        fallback_model=None, fallback_upstream=None,
        rung="frontier", label="user_turn", route_id="route-f", layer="L2-classifier")


# ── send() double for acquire_upstream (all network I/O injected) ────────────
class _FakeResp:
    def __init__(self, status: int):
        self.status = status
        self.released = False

    async def release(self):
        self.released = True


def _make_send(behaviour):
    """behaviour: dict upstream_base -> ('raise', exc) | ('status', int)."""
    calls: list[tuple[str, bytes | None]] = []

    async def send(upstream: str, body):
        calls.append((upstream, body))
        kind, payload = behaviour[upstream]
        if kind == "raise":
            raise payload
        return _FakeResp(payload)

    return send, calls


# ════════════════════════════════════════════════════════════════════════════
# liveplan_attempts — RoutePlan -> ordered (upstream, body) attempts
# ════════════════════════════════════════════════════════════════════════════
def test_local_plan_attempts_are_litellm_then_anthropic_with_original_fallback():
    router = _router()
    body = _user_turn_body()
    attempts = liveplan_attempts(router, _local_plan(), body)
    # primary = LiteLLM with the model rewritten to the local alias
    assert [u for u, _ in attempts] == [LITELLM, ANTHROPIC]
    assert json.loads(attempts[0][1])["model"] == "local-code"
    # fallback = Anthropic carrying the ORIGINAL body+model (fallback_model is None)
    assert json.loads(attempts[1][1])["model"] == "claude-opus-4-20250514"


def test_cheap_cloud_plan_is_single_anthropic_attempt_model_rewritten():
    router = _router()
    body = _user_turn_body()
    attempts = liveplan_attempts(router, _cheap_cloud_plan(), body)
    assert [u for u, _ in attempts] == [ANTHROPIC]     # no fail-open — already on Anthropic
    assert json.loads(attempts[0][1])["model"] == "claude-haiku-4-5"


def test_frontier_plan_is_single_anthropic_attempt_body_unchanged():
    router = _router()
    body = _user_turn_body()
    attempts = liveplan_attempts(router, _frontier_plan(), body)
    assert [u for u, _ in attempts] == [ANTHROPIC]
    # frontier does NOT rewrite the model — body forwarded unchanged
    assert json.loads(attempts[0][1])["model"] == "claude-opus-4-20250514"


# ── acquire_upstream over a local plan (reuses the P1-A fail-open loop) ───────
def test_local_success_serves_local_no_fallback():
    router = _router()
    attempts = liveplan_attempts(router, _local_plan(), _user_turn_body())
    send, calls = _make_send({LITELLM: ("status", 200), ANTHROPIC: ("status", 200)})
    result = _run(acquire_upstream(attempts, send))
    assert result.response.status == 200
    assert result.used_fallback is False
    assert json.loads(result.sent_body)["model"] == "local-code"
    assert calls == [(LITELLM, attempts[0][1])]        # Anthropic never touched


def test_local_failure_falls_open_to_anthropic_with_original_body():
    router = _router()
    attempts = liveplan_attempts(router, _local_plan(), _user_turn_body())
    send, calls = _make_send({
        LITELLM: ("raise", ConnectionRefusedError("litellm down")),
        ANTHROPIC: ("status", 200),
    })
    result = _run(acquire_upstream(attempts, send))
    assert result.response.status == 200
    assert result.used_fallback is True
    # served by Anthropic with the ORIGINAL model, not the rewritten local alias
    assert json.loads(result.sent_body)["model"] == "claude-opus-4-20250514"
    assert [u for u, _ in calls] == [LITELLM, ANTHROPIC]


def test_local_5xx_falls_open_to_anthropic():
    router = _router()
    attempts = liveplan_attempts(router, _local_plan(), _user_turn_body())
    send, calls = _make_send({LITELLM: ("status", 503), ANTHROPIC: ("status", 200)})
    result = _run(acquire_upstream(attempts, send))
    assert result.response.status == 200
    assert result.used_fallback is True
    assert [u for u, _ in calls] == [LITELLM, ANTHROPIC]


def test_cheap_cloud_single_attempt_no_phantom_fallback():
    router = _router()
    attempts = liveplan_attempts(router, _cheap_cloud_plan(), _user_turn_body())
    send, calls = _make_send({ANTHROPIC: ("status", 200)})
    result = _run(acquire_upstream(attempts, send))
    assert result.response.status == 200
    assert result.used_fallback is False
    assert len(calls) == 1                              # no fail-open on a cloud rung


# ════════════════════════════════════════════════════════════════════════════
# session_id_for — the key the escalation controller + recorder share
# ════════════════════════════════════════════════════════════════════════════
def test_session_id_for_uses_metadata_user_id():
    assert session_id_for(_user_turn_body("sess-42")) == "sess-42"


def test_session_id_for_falls_back_to_anon_hash_when_no_metadata():
    sid = session_id_for(_user_turn_body(sid=None))
    assert sid.startswith("anon:")                     # stable hashed key, never crashes


def test_session_id_for_never_raises_on_garbage():
    assert isinstance(session_id_for({}), str)
    assert isinstance(session_id_for({"metadata": {"user_id": "not-json"}}), str)


# ════════════════════════════════════════════════════════════════════════════
# reconstruct_blocks — what escalation.observe_response sees
# ════════════════════════════════════════════════════════════════════════════
def _sse(events: list[dict]) -> bytes:
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()


def test_reconstruct_sse_assembles_text_and_tooluse_and_tokens():
    raw = _sse([
        {"type": "message_start", "message": {"usage": {"output_tokens": 0}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "Reading file"}},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": "{\"file\": \"a.py\"}"}},
        {"type": "message_delta", "usage": {"output_tokens": 42}},
    ])
    blocks, out = reconstruct_blocks(raw)
    assert out == 42
    assert blocks[0] == {"type": "text", "text": "Reading file"}
    tool = blocks[1]
    assert tool["type"] == "tool_use" and tool["name"] == "Read"
    assert tool["input"] == {"file": "a.py"}            # assembled from the json delta


def test_reconstruct_sse_malformed_toolinput_kept_raw_for_parse_tripwire():
    raw = _sse([
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "tool_use", "id": "t", "name": "Edit", "input": {}}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": "{not valid"}},
    ])
    blocks, _ = reconstruct_blocks(raw)
    # a local model emitting bad tool JSON -> raw string kept so parse trip-wire fires
    assert blocks[0]["input"] == "{not valid"


def test_reconstruct_plain_json_response():
    raw = json.dumps({
        "content": [{"type": "text", "text": "hi"}],
        "usage": {"output_tokens": 7},
    }).encode()
    blocks, out = reconstruct_blocks(raw)
    assert out == 7
    assert blocks == [{"type": "text", "text": "hi"}]


def test_reconstruct_empty_is_failsafe():
    assert reconstruct_blocks(b"") == ([], 0)
    assert reconstruct_blocks(b"garbage not sse not json") == ([], 0)


def test_prepare_escalation_body_strips_thinking_and_adds_failure_note():
    body = _user_turn_body()
    body["messages"] = [
        {"role": "user", "content": "Fix it."},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "private reasoning"},
            {"type": "tool_use", "id": "t1", "name": "Edit", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "failed"},
        ]},
    ]
    decision = SimpleNamespace(tripwire="edit_apply", tripwire_type="dialect")
    rebuilt = prepare_escalation_body(body, decision)
    assert "Failure detector: edit_apply (dialect)." in str(rebuilt["messages"][0])
    assert "private reasoning" not in json.dumps(rebuilt)
    assert "t1" in json.dumps(rebuilt)
    assert "private reasoning" in json.dumps(body)  # input is immutable


# ════════════════════════════════════════════════════════════════════════════
# make_app — the routing flag gates ALL live-decision construction
# ════════════════════════════════════════════════════════════════════════════
def test_make_app_flag_off_constructs_no_router(tmp_path: Path):
    async def _go():
        app = await make_app(tmp_path, route_user_turns=False)
        try:
            assert app["route_user_turns"] is False
            assert app["live_router"] is None
            assert app["escalation"] is None
            assert app["recorder"] is None
            assert app["outcome_cls"] is None
        finally:
            await app["client"].close()
    _run(_go())


def test_make_app_flag_on_wires_injected_singletons(tmp_path: Path):
    async def _go():
        sentinel_router = object()
        sentinel_esc = object()
        sentinel_rec = object()
        app = await make_app(
            tmp_path, route_user_turns=True,
            live_router=sentinel_router, escalation=sentinel_esc, recorder=sentinel_rec)
        try:
            assert app["live_router"] is sentinel_router
            assert app["escalation"] is sentinel_esc
            assert app["recorder"] is sentinel_rec
            assert app["outcome_cls"] is not None       # OutcomeEvent for attach()
        finally:
            await app["client"].close()
    _run(_go())


# ════════════════════════════════════════════════════════════════════════════
# handle — END-TO-END over a TestServer with a FAKE upstream client
# ════════════════════════════════════════════════════════════════════════════
class _FakeContent:
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk


class _FakeUpstreamResp:
    def __init__(self, status: int, headers: dict, chunks: list[bytes]):
        self.status = status
        self.headers = headers
        self.content = _FakeContent(chunks)
        self.released = False

    async def release(self):
        self.released = True


class _FakeClient:
    """Stands in for the aiohttp ClientSession make_app creates. Records every
    upstream call and returns a canned SSE response — no real socket."""

    def __init__(self, chunks: list[bytes], status: int = 200):
        self.calls: list[str] = []
        self.bodies: list[bytes | None] = []
        self._chunks = chunks
        self._status = status

    async def request(self, method, url, headers=None, data=None,
                      allow_redirects=False, **kwargs):
        self.calls.append(url)
        self.bodies.append(data)
        return _FakeUpstreamResp(
            self._status, {"content-type": "text/event-stream"}, self._chunks)

    async def close(self):
        pass


class _FallbackClient(_FakeClient):
    """Local transport fails once; Anthropic then serves the request."""

    async def request(self, method, url, headers=None, data=None,
                      allow_redirects=False, **kwargs):
        self.calls.append(url)
        self.bodies.append(data)
        if url.startswith(LITELLM):
            raise ConnectionRefusedError("local unavailable")
        return _FakeUpstreamResp(
            200, {"content-type": "text/event-stream"}, self._chunks)


class _FakeRouter:
    """Returns a fixed plan so the test controls routing without invoking the
    real policy. Provides the litellm_url + build_forward_body handle() needs."""

    def __init__(self, plan: LiveRoutePlan):
        self._plan = plan
        self.litellm_url = LITELLM
        self.anthropic_url = ANTHROPIC

    def plan(self, method, path, body):
        return self._plan

    def build_forward_body(self, body, model):
        if model is None:
            return json.dumps(body).encode()
        rewritten = dict(body)
        rewritten["model"] = model
        return json.dumps(rewritten).encode()


class _SpyEscalation:
    def __init__(self):
        self.new_turns: list[str] = []
        self.observed: list[tuple] = []
        self.tool_results: list[tuple] = []
        self.continuations: list[str] = []
        self.infrastructure_failures: list[str] = []

    def new_turn(self, session_id):
        self.new_turns.append(session_id)

    def observe_tool_results(self, session_id, content):
        self.tool_results.append((session_id, content))
        return None

    def observe_response(self, session_id, blocks, *, output_tokens=0):
        self.observed.append((session_id, blocks, output_tokens))
        return SimpleNamespace(escalate=False, tripwire=None, tripwire_type=None)

    def note_continuation(self, session_id):
        self.continuations.append(session_id)
        return len(self.continuations)

    def note_infrastructure_failure(self, session_id):
        self.infrastructure_failures.append(session_id)
        return len(self.infrastructure_failures)


class _SpyRecorder:
    def __init__(self):
        self.attached: list[tuple] = []

    def attach(self, session_id, outcome):
        self.attached.append((session_id, outcome))
        return True


_SSE_TOOLUSE = "".join(f"data: {json.dumps(e)}\n\n" for e in [
    {"type": "message_start", "message": {"usage": {"output_tokens": 0}}},
    {"type": "content_block_start", "index": 0,
     "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}}},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "input_json_delta", "partial_json": "{\"file\": \"a.py\"}"}},
    {"type": "message_delta", "usage": {"output_tokens": 42}},
]).encode()


async def _drive(app, raw_body: bytes):
    """POST raw_body through the app's handle() over a TestServer, returning the
    response status and text. The app's upstream client must already be a fake."""
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        resp = await client.post(PATH, data=raw_body)
        return resp.status, await resp.text()
    finally:
        await client.close()


def _only_capture(capture_root: Path) -> dict:
    files = list(capture_root.rglob("*.json"))
    assert len(files) == 1, f"expected exactly one capture, got {files}"
    return json.loads(files[0].read_text())


def test_routed_user_turn_observes_reconstructed_blocks_and_records(tmp_path: Path):
    spy_esc = _SpyEscalation()
    spy_rec = _SpyRecorder()
    fake_router = _FakeRouter(_local_plan())

    async def _go():
        app = await make_app(
            tmp_path, route_user_turns=True,
            live_router=fake_router, escalation=spy_esc, recorder=spy_rec)
        await app["client"].close()
        app["client"] = _FakeClient([_SSE_TOOLUSE])     # local hit, one attempt
        status, _ = await _drive(app, json.dumps(_user_turn_body("sess-abc")).encode())
        return status

    status = _run(_go())
    assert status == 200

    # (a) a user_turn opened a fresh trip-wire window
    assert spy_esc.new_turns == ["sess-abc"]
    # (b) observe_response got the RECONSTRUCTED blocks + output_tokens
    assert len(spy_esc.observed) == 1
    sid, blocks, out_tokens = spy_esc.observed[0]
    assert sid == "sess-abc"
    assert out_tokens == 42
    assert blocks and blocks[0]["type"] == "tool_use"
    assert blocks[0]["input"] == {"file": "a.py"}
    # (c) the outcome was attached to the recorder as a closed_turn
    assert len(spy_rec.attached) == 1
    rec_sid, outcome = spy_rec.attached[0]
    assert rec_sid == "sess-abc"
    assert outcome.status == "closed_turn"
    assert outcome.output_tokens == 42
    assert outcome.escalated is False
    # (d) the capture carries the WS2 routing fields
    record = _only_capture(tmp_path)
    assert record["routed_rung"] == "local"
    assert record["route_layer"] == "L1-features"
    assert record["route_id"] == "route-xyz"


def test_capture_only_mode_does_no_routing(tmp_path: Path):
    async def _go():
        app = await make_app(tmp_path, route_user_turns=False)
        assert app["live_router"] is None               # nothing constructed
        await app["client"].close()
        app["client"] = _FakeClient([b'data: {"type":"message_stop"}\n\n'])
        status, _ = await _drive(app, json.dumps(_user_turn_body()).encode())
        return status

    status = _run(_go())
    assert status == 200
    # capture-only path stamps NO routing fields — byte-identical to the old proxy
    record = _only_capture(tmp_path)
    assert record["routed_rung"] is None
    assert record["route_layer"] is None
    assert record["route_id"] is None


def test_routed_continuation_does_not_open_a_new_turn(tmp_path: Path):
    # A continuation observes/records but must NOT call new_turn (only user_turns do).
    spy_esc = _SpyEscalation()
    spy_rec = _SpyRecorder()
    cont_plan = LiveRoutePlan(
        primary_model="local-code", primary_upstream=LITELLM,
        fallback_model=None, fallback_upstream=ANTHROPIC,
        rung="local", label="continuation", route_id="route-cont", layer="continuation")
    fake_router = _FakeRouter(cont_plan)

    async def _go():
        app = await make_app(
            tmp_path, route_user_turns=True,
            live_router=fake_router, escalation=spy_esc, recorder=spy_rec)
        await app["client"].close()
        app["client"] = _FakeClient([_SSE_TOOLUSE])
        return await _drive(app, json.dumps(_user_turn_body("sess-cont")).encode())

    status, _ = _run(_go())
    assert status == 200
    assert spy_esc.new_turns == []                      # continuation: no new turn
    assert spy_esc.continuations == ["sess-cont"]
    assert len(spy_esc.observed) == 1                   # but still observed + recorded
    assert len(spy_rec.attached) == 1


def test_routed_outcome_populates_friction_proxy_hard(tmp_path: Path):
    # A turn that accrued edit/parse strikes but did NOT escalate must still be
    # recorded hard via outcome_proxy_hard (review finding: friction was dropped,
    # so went_hard mislabeled such turns easy).
    class _Store:
        def get_session(self, sid):
            return SimpleNamespace(escalated_this_episode=False,
                                   strikes={"edit": 2, "parse": 1})

    spy_esc = _SpyEscalation()
    spy_esc.store = _Store()
    spy_rec = _SpyRecorder()
    plan = LiveRoutePlan(
        primary_model="local-code", primary_upstream=LITELLM,
        fallback_model=None, fallback_upstream=ANTHROPIC,
        rung="local", label="user_turn", route_id="route-fr", layer="heuristic")

    async def _go():
        app = await make_app(
            tmp_path, route_user_turns=True,
            live_router=_FakeRouter(plan), escalation=spy_esc, recorder=spy_rec)
        await app["client"].close()
        app["client"] = _FakeClient([_SSE_TOOLUSE])
        return await _drive(app, json.dumps(_user_turn_body("sess-fr")).encode())

    status, _ = _run(_go())
    assert status == 200
    assert len(spy_rec.attached) == 1
    _, outcome = spy_rec.attached[0]
    assert outcome.edit_failures == 2
    assert outcome.error_results == 1
    assert outcome.escalated is False
    assert outcome.outcome_proxy_hard is True   # friction -> hard even without escalation
    assert outcome.task_signal_hard is False    # no task-capability evidence


def test_real_controller_evidence_reaches_final_proxy_outcome(tmp_path: Path):
    """Regression for the former store/controller key mismatch.

    Drive a real controller across request boundaries. The first edit failure
    must be visible before escalation; the second must preserve cumulative
    continuation/error evidence while classifying the cause as dialect, not
    task difficulty.
    """
    sid = "sess-integrated"
    store = DictSessionStore()
    store.set_route(sid, "local-code")
    escalation = EscalationController(store)
    recorder = _SpyRecorder()

    class _TurnAwareRouter(_FakeRouter):
        def plan(self, method, path, body):
            content = (body.get("messages") or [{}])[-1].get("content")
            continuation = isinstance(content, list) and any(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in content
            )
            return LiveRoutePlan(
                primary_model="local-code", primary_upstream=LITELLM,
                fallback_model=None, fallback_upstream=ANTHROPIC,
                rung="local", label="continuation" if continuation else "user_turn",
                route_id="route-integrated", layer="continuation" if continuation else "test",
            )

    edit_failure = {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "is_error": True,
        "content": "String to replace not found in file",
    }

    def continuation_body():
        body = _user_turn_body(sid)
        body["messages"] = [
            {"role": "user", "content": "Fix the file."},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "Edit", "input": {}},
            ]},
            {"role": "user", "content": [edit_failure]},
        ]
        return body

    async def _go():
        app = await make_app(
            tmp_path, route_user_turns=True,
            live_router=_TurnAwareRouter(_local_plan()),
            escalation=escalation, recorder=recorder,
        )
        await app["client"].close()
        app["client"] = _FakeClient([_SSE_TOOLUSE])
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            for body in (_user_turn_body(sid), continuation_body(), continuation_body()):
                response = await client.post(PATH, data=json.dumps(body).encode())
                assert response.status == 200
                await response.read()
        finally:
            await client.close()

    _run(_go())
    assert len(recorder.attached) == 3
    partial = recorder.attached[1][1]
    assert partial.edit_failures == 1
    assert partial.continuation_count == 1
    assert partial.escalated is False

    final = recorder.attached[2][1]
    assert final.edit_failures == 2
    assert final.error_results == 2
    assert final.continuation_count == 2
    assert final.output_tokens == 126
    assert final.escalated is True
    assert final.tripwire_name == "edit_apply"
    assert final.tripwire_type == "dialect"
    assert final.failure_cause == "harness_dialect"
    assert final.outcome_proxy_hard is True
    assert final.task_signal_hard is False
    assert store.get_session(sid).route == "cheap-cloud"


def test_streamed_dialect_failure_persists_cause_clean_memory_row(tmp_path: Path):
    """Full live boundary: proxy stream -> controller -> recorder -> SQLite."""
    sid = "sess-persisted"
    store = DictSessionStore()
    escalation = EscalationController(store)
    provider = SqliteProvider(tmp_path / "memory.db")

    class _NoEmbedding:
        def embed(self, texts):
            return None

    recorder = RoutingRecorder(
        RouterMemory([provider], embedder=_NoEmbedding(), policy_version="test-v2"))
    router = LiveRouter(store=store, memory=recorder)
    edit_failure = {
        "type": "tool_result", "tool_use_id": "toolu_1", "is_error": True,
        "content": "String to replace not found in file",
    }

    first = _user_turn_body(sid)
    first["messages"] = [{"role": "user", "content": "Write a commit message."}]
    continuation = _user_turn_body(sid)
    continuation["messages"] = [
        {"role": "user", "content": "Write a commit message."},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "Edit", "input": {}},
        ]},
        {"role": "user", "content": [edit_failure]},
    ]

    async def _go():
        app = await make_app(
            tmp_path / "captures", route_user_turns=True,
            live_router=router, escalation=escalation, recorder=recorder,
        )
        await app["client"].close()
        app["client"] = _FakeClient([_SSE_TOOLUSE])
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            for body in (first, continuation, continuation):
                response = await client.post(PATH, data=json.dumps(body).encode())
                assert response.status == 200
                await response.read()
        finally:
            await client.close()

    _run(_go())
    route_id = recorder.active_route_id(sid)
    assert route_id
    conn = sqlite3.connect(provider.db_path)
    row = conn.execute(
        "SELECT status, escalated, tripwire_name, tripwire_type, edit_failures, "
        "error_results, output_tokens, continuation_count, failure_cause, "
        "outcome_proxy_hard, task_signal_hard FROM outcomes WHERE route_id=?",
        (route_id,),
    ).fetchone()
    assert row == (
        "closed_turn", 1, "edit_apply", "dialect", 2, 2, 126, 2,
        "harness_dialect", 1, 0,
    )
    assert conn.execute(
        "SELECT COUNT(*) FROM outcome_events WHERE route_id=?", (route_id,)
    ).fetchone()[0] == 4
    conn.close()


def test_live_fallback_syncs_actual_served_rung_for_next_continuation(tmp_path: Path):
    sid = "sess-fallback-sync"
    store = DictSessionStore()
    router = LiveRouter(store=store, anthropic_url=ANTHROPIC, litellm_url=LITELLM)
    escalation = EscalationController(store)

    body = _user_turn_body(sid)
    body["messages"] = [{"role": "user", "content": "Fix the README typo."}]

    async def _go():
        app = await make_app(
            tmp_path, route_user_turns=True, live_router=router,
            escalation=escalation, recorder=_SpyRecorder(),
        )
        await app["client"].close()
        fake = _FallbackClient([_SSE_TOOLUSE])
        app["client"] = fake
        status, _ = await _drive(app, json.dumps(body).encode())
        return status, fake

    status, fake = _run(_go())
    assert status == 200
    assert len(fake.calls) == 2
    assert store.get_session(sid).route == "frontier"
    assert router.health.snapshot()["local"]["healthy"] is False
    capture = _only_capture(tmp_path)
    assert capture["local_fallback"] is True
    assert capture["served_rung"] == "frontier"


def test_live_privacy_pin_disables_cloud_fallback(tmp_path: Path):
    sid = "sess-private"
    store = DictSessionStore()
    router = LiveRouter(store=store, anthropic_url=ANTHROPIC, litellm_url=LITELLM)
    escalation = EscalationController(store)
    body = _user_turn_body(sid)
    body["messages"] = [{
        "role": "user",
        "content": "Fix the README using key AKIAABCDEFGHIJKLMNOP.",
    }]

    async def _go():
        app = await make_app(
            tmp_path, route_user_turns=True, live_router=router,
            escalation=escalation, recorder=_SpyRecorder(),
        )
        await app["client"].close()
        fake = _FakeClient([_SSE_TOOLUSE], status=500)
        app["client"] = fake
        status, _ = await _drive(app, json.dumps(body).encode())
        return status, fake

    status, fake = _run(_go())
    assert status == 500
    assert store.get_session(sid).privacy_pinned is True
    assert len(fake.calls) == 1
    assert fake.calls[0].startswith(LITELLM)
    capture = _only_capture(tmp_path)
    assert "aws_access_key" in capture["privacy_hits"]
    assert capture["local_fallback"] is False


def test_live_privacy_context_conflict_is_blocked_without_upstream(tmp_path: Path):
    sid = "sess-private-large"
    store = DictSessionStore()
    router = LiveRouter(store=store, anthropic_url=ANTHROPIC, litellm_url=LITELLM)
    escalation = EscalationController(store)
    body = _user_turn_body(sid)
    body["messages"] = [{
        "role": "user",
        "content": "AKIAABCDEFGHIJKLMNOP " + ("x" * 110_000),
    }]

    async def _go():
        app = await make_app(
            tmp_path, route_user_turns=True, live_router=router,
            escalation=escalation, recorder=_SpyRecorder(),
        )
        await app["client"].close()
        fake = _FakeClient([_SSE_TOOLUSE])
        app["client"] = fake
        status, text = await _drive(app, json.dumps(body).encode())
        return status, text, fake

    status, text, fake = _run(_go())
    assert status == 409
    assert "privacy_policy_conflict" in text
    assert fake.calls == []
    assert _only_capture(tmp_path)["served_rung"] == "blocked"


def test_second_edit_failure_rebuilds_and_hands_off_same_continuation(tmp_path: Path):
    sid = "sess-handoff"
    store = DictSessionStore()
    router = LiveRouter(store=store, anthropic_url=ANTHROPIC, litellm_url=LITELLM)
    escalation = EscalationController(store)
    edit_failure = {
        "type": "tool_result", "tool_use_id": "toolu_1", "is_error": True,
        "content": "String to replace not found in file",
    }

    first = _user_turn_body(sid)
    first["messages"] = [{"role": "user", "content": "Fix the README typo."}]

    def continuation():
        body = _user_turn_body(sid)
        body["messages"] = [
            {"role": "user", "content": "Fix the README typo."},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "provider-private"},
                {"type": "tool_use", "id": "toolu_1", "name": "Edit", "input": {}},
            ]},
            {"role": "user", "content": [edit_failure]},
        ]
        return body

    async def _go():
        app = await make_app(
            tmp_path, route_user_turns=True, live_router=router,
            escalation=escalation, recorder=_SpyRecorder(),
        )
        await app["client"].close()
        fake = _FakeClient([_SSE_TOOLUSE])
        app["client"] = fake
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            for body in (first, continuation(), continuation()):
                response = await client.post(PATH, data=json.dumps(body).encode())
                assert response.status == 200
                await response.read()
        finally:
            await client.close()
        return fake

    fake = _run(_go())
    assert len(fake.calls) == 3
    assert fake.calls[:2] == [LITELLM + PATH, LITELLM + PATH]
    assert fake.calls[2] == ANTHROPIC + PATH
    handed_off = json.loads(fake.bodies[2])
    assert handed_off["model"] == "claude-haiku-4-5"
    assert "Failure detector: edit_apply (dialect)." in json.dumps(handed_off)
    assert "provider-private" not in json.dumps(handed_off)
    assert store.get_session(sid).route == "cheap-cloud"
