"""Core-loop tests for the qwen_harness study port: history curation, the
turn event stream, the driver loop end-to-end against a scripted fake
model, loop detection, and compression thresholds."""

import asyncio

import pytest

from qwen_harness.chat import (GeminiChat, InvalidStreamError,
                               extract_curated_history, is_valid_content)
from qwen_harness.client import GeminiClient
from qwen_harness.config import ApprovalMode, Config
from qwen_harness.scheduler import CoreToolScheduler
from qwen_harness.services.compression import compute_thresholds
from qwen_harness.services.loop_detection import (LoopDetectionService,
                                                  LoopType)
from qwen_harness.token_limits import token_limit
from qwen_harness.types import (Content, FunctionCall, GeminiEvent,
                                GeminiEventType, ModelResponseChunk, Part,
                                ToolCallRequestInfo, UsageMetadata,
                                model_text, user_text)


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- fake model


class FakeGenerator:
    """Scripted ContentGenerator: each call pops the next canned response.
    A response is a list of ModelResponseChunk."""

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.requests = []

    async def generate_content_stream(self, model, contents, system_instruction=None,
                                      tools=None, max_output_tokens=None,
                                      temperature=None):
        self.requests.append({"contents": contents, "system": system_instruction,
                              "tools": tools})
        if not self.scripts:
            raise AssertionError("FakeGenerator script exhausted")
        for chunk in self.scripts.pop(0):
            yield chunk

    async def generate_text(self, model, contents, system_instruction=None,
                            max_output_tokens=None):
        parts = []
        async for c in self.generate_content_stream(model, contents,
                                                    system_instruction):
            if c.text:
                parts.append(c.text)
        return "".join(parts)

    async def generate_json(self, model, contents, schema):
        raise RuntimeError("no side-query scripted")


def text_stream(text, usage=None):
    return [ModelResponseChunk(text=text),
            ModelResponseChunk(finish_reason="STOP", usage=usage)]


def tool_stream(name, args, call_id="call-1"):
    return [ModelResponseChunk(function_calls=[FunctionCall(name=name, args=args,
                                                            id=call_id)],
                               finish_reason="STOP")]


# ------------------------------------------------------------- history views


def test_curated_history_drops_invalid_model_runs():
    history = [
        user_text("q1"),
        Content(role="model", parts=[Part(text="")]),   # invalid: empty text
        user_text("q2"),
        model_text("fine"),
    ]
    curated = extract_curated_history(history)
    # invalid model run dropped; q1 merges into q2's user turn
    assert [c.role for c in curated] == ["user", "model"]
    assert [p.text for p in curated[0].parts] == ["q1", "q2"]
    assert curated[1].parts[0].text == "fine"


def test_valid_content_rules():
    assert not is_valid_content(Content(role="model", parts=[]))
    assert not is_valid_content(Content(role="model", parts=[Part(text="")]))
    assert is_valid_content(Content(role="model", parts=[Part(text="", thought=True)]))
    assert is_valid_content(Content(role="model", parts=[
        Part(function_call=FunctionCall(name="x"))]))


def test_chat_consolidates_model_turn_and_validates_stream():
    gen = FakeGenerator([
        [ModelResponseChunk(text="hel"), ModelResponseChunk(text="lo"),
         ModelResponseChunk(thought="thinking"),
         ModelResponseChunk(finish_reason="STOP",
                            usage=UsageMetadata(prompt_tokens=10))],
    ])
    chat = GeminiChat(gen, "m", "sys")

    async def go():
        async for _ in chat.send_message_stream([Part(text="hi")]):
            pass
    run(go())
    assert len(chat.history) == 2
    model_turn = chat.history[1]
    assert model_turn.parts[0].thought and model_turn.parts[0].text == "thinking"
    assert model_turn.parts[1].text == "hello"
    assert chat.last_prompt_token_count == 10


def test_chat_retries_empty_stream_then_raises():
    empty = [ModelResponseChunk(finish_reason="STOP")]
    gen = FakeGenerator([list(empty), list(empty), list(empty)])
    chat = GeminiChat(gen, "m", "sys")

    async def go():
        async for _ in chat.send_message_stream([Part(text="hi")]):
            pass
    with pytest.raises(InvalidStreamError):
        run(go())
    assert len(gen.requests) == 3  # 1 try + 2 retries


def test_orphaned_tool_call_repair():
    chat = GeminiChat(FakeGenerator([]), "m", "sys")
    chat.history = [
        user_text("do it"),
        Content(role="model", parts=[
            Part(function_call=FunctionCall(name="edit", args={}, id="c1"))]),
        # no functionResponse follows: session died mid-tool
    ]
    chat.repair_orphaned_tool_calls()
    assert chat.history[-1].role == "user"
    fr = chat.history[-1].parts[0].function_response
    assert fr.id == "c1" and "error" in fr.response


# ------------------------------------------------------------ the full loop


def make_client(scripts, tmp_path, approval=ApprovalMode.YOLO):
    from qwen_harness.builtins import create_tool_registry
    config = Config(model="qwen3-coder-plus", target_dir=str(tmp_path),
                    approval_mode=approval, skip_next_speaker_check=True)
    registry = create_tool_registry(config)
    gen = FakeGenerator(scripts)
    client = GeminiClient(config, generator=gen, registry=registry)
    client.start_chat()
    scheduler = CoreToolScheduler(config, registry, confirmation_handler=None)
    return config, client, scheduler, gen


def test_agentic_loop_executes_tool_and_feeds_response_back(tmp_path):
    (tmp_path / "hello.txt").write_text("line one\nline two\n")
    scripts = [
        tool_stream("read_file", {"file_path": str(tmp_path / "hello.txt")}),
        text_stream("the file says: line one"),
    ]
    config, client, scheduler, gen = make_client(scripts, tmp_path)

    from qwen_harness.cli import run_non_interactive
    code = run(run_non_interactive(config, client, scheduler, "read hello.txt"))
    assert code == 0
    # second request must contain the functionResponse fed back as user turn
    second = gen.requests[1]["contents"]
    fr_parts = [p for c in second for p in c.parts if p.function_response]
    assert fr_parts and "line one" in fr_parts[0].function_response.response["output"]
    # and the tools block advertises the built-ins, sorted by name
    names = [t["name"] for t in gen.requests[0]["tools"]]
    assert names == sorted(names) and "run_shell_command" in names


def test_env_context_is_history_zero(tmp_path):
    config, client, scheduler, gen = make_client([text_stream("hi")], tmp_path)
    first = client.get_chat().history[0]
    assert first.role == "user"
    assert "This is the Qwen Code" in first.parts[0].text
    assert "<system-reminder>" in first.parts[0].text


def test_max_session_turns(tmp_path):
    config, client, scheduler, gen = make_client(
        [text_stream("a"), text_stream("b")], tmp_path)
    config.max_session_turns = 1

    async def collect(prompt):
        events = []
        async for e in client.send_message_stream([Part(text=prompt)],
                                                  client.new_prompt_id()):
            events.append(e)
        return events

    first = run(collect("one"))
    assert any(e.type == GeminiEventType.FINISHED for e in first)
    second = run(collect("two"))
    assert [e.type for e in second] == [GeminiEventType.MAX_SESSION_TURNS]


# ------------------------------------------------------------ loop detection


def req_event(name, args):
    return GeminiEvent(GeminiEventType.TOOL_CALL_REQUEST,
                       ToolCallRequestInfo(call_id="c", name=name, args=args))


def test_consecutive_identical_tool_calls_fire_at_5():
    detector = LoopDetectionService()
    detector.reset("p")
    for i in range(4):
        assert not detector.check_always_on_safeties(req_event("read_file", {"file_path": "/a"}))
    assert detector.check_always_on_safeties(req_event("read_file", {"file_path": "/a"}))
    assert detector.last_loop_type == LoopType.CONSECUTIVE_IDENTICAL_TOOL_CALLS


def test_different_args_do_not_trip_consecutive_detector():
    detector = LoopDetectionService()
    detector.reset("p")
    for i in range(20):
        assert not detector.check_always_on_safeties(
            req_event("read_file", {"file_path": f"/a{i}"}))


def test_turn_tool_call_cap():
    detector = LoopDetectionService(max_tool_calls_per_turn=10)
    detector.reset("p")
    tripped = False
    for i in range(12):
        if detector.check_always_on_safeties(req_event("t", {"i": i})):
            tripped = True
            break
    assert tripped and detector.last_loop_type == LoopType.TURN_TOOL_CALL_CAP


def test_chanting_detection():
    detector = LoopDetectionService()
    detector.reset("p")
    tripped = False
    for _ in range(60):
        event = GeminiEvent(GeminiEventType.CONTENT,
                            "I will now proceed to fix the bug. ")
        if detector.add_and_check_heuristic_loops(event):
            tripped = True
            break
    assert tripped
    assert detector.last_loop_type == LoopType.CHANTING_IDENTICAL_SENTENCES


def test_alternating_pattern_detection():
    detector = LoopDetectionService()
    detector.reset("p")
    tripped = False
    for i in range(8):
        name = "read_file" if i % 2 == 0 else "list_directory"
        args = {"path": "/x"}
        if detector.add_and_check_heuristic_loops(req_event(name, args)):
            tripped = True
            break
    assert tripped
    assert detector.last_loop_type in (LoopType.ALTERNATING_TOOL_CALL_PATTERN,
                                       LoopType.GLOBAL_TOOL_CALL_DUPLICATE)


# ---------------------------------------------------------------- compression


def test_compute_thresholds_ladder():
    t = compute_thresholds(200_000)
    assert t.warn < t.auto < t.hard <= 200_000
    assert t.auto == max(0.7 * 200_000, (200_000 - 20_000) - 13_000)


def test_token_limits():
    assert token_limit("qwen3-coder-plus") == 1_000_000
    assert token_limit("Qwen/Qwen3-Coder-480B") == 262_144
    # ollama-style tags lose the family prefix in normalize() (the last
    # ':'-segment is kept, faithful to upstream) and fall to the default
    assert token_limit("qwen2.5:7b-instruct-q4_K_M") == 200_000
    assert token_limit("claude-opus-4-8") == 200_000
    assert token_limit("unknown-model-x") == 200_000
