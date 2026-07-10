"""End-to-end wire test: the full harness loop against a scripted
OpenAI-compatible SSE server (what LiteLLM/ollama/DashScope all speak).
Exercises content_generator's HTTP + SSE parsing + streaming tool-call
reassembly, which the FakeGenerator tests bypass."""

import asyncio
import json

from aiohttp import web

from qwen_harness.builtins import create_tool_registry
from qwen_harness.client import GeminiClient
from qwen_harness.config import ApprovalMode, Config
from qwen_harness.content_generator import ContentGenerator, GeneratorConfig
from qwen_harness.scheduler import CoreToolScheduler
from qwen_harness.types import GeminiEventType, Part


def sse(payload):
    return f"data: {json.dumps(payload)}\n\n".encode()


def chunk(delta, finish=None, usage=None):
    body = {"choices": [{"delta": delta, "finish_reason": finish}]}
    if usage:
        body["usage"] = usage
    return body


class ScriptedServer:
    """Each request pops the next scripted list of SSE chunks; records the
    request bodies for wire-format assertions."""

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.requests = []

    async def handle(self, request):
        self.requests.append(await request.json())
        response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for c in self.scripts.pop(0):
            await response.write(sse(c))
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response


async def run_session(tmp_path, scripts, prompt):
    server = ScriptedServer(scripts)
    app = web.Application()
    app.router.add_post("/v1/chat/completions", server.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]

    config = Config(model="test-model", target_dir=str(tmp_path),
                    approval_mode=ApprovalMode.YOLO,
                    skip_next_speaker_check=True)
    config.generator = GeneratorConfig(base_url=f"http://127.0.0.1:{port}/v1",
                                       max_retries=2)
    registry = create_tool_registry(config)
    client = GeminiClient(config, generator=ContentGenerator(config.generator),
                          registry=registry)
    client.start_chat()
    scheduler = CoreToolScheduler(config, registry)

    events = []
    prompt_id = client.new_prompt_id()
    current = [Part(text=prompt)]
    for round_no in range(5):
        calls = []
        async for event in client.send_message_stream(
                current, prompt_id, is_continuation=round_no > 0):
            events.append(event)
            if event.type == GeminiEventType.TOOL_CALL_REQUEST:
                calls.append(event.value)
        if not calls:
            break
        responses = await scheduler.schedule(calls)
        current = [p for r in responses for p in r.response_parts]

    await runner.cleanup()
    return events, server


def test_full_loop_over_the_wire(tmp_path):
    (tmp_path / "notes.txt").write_text("the answer is 42\n")

    # turn 1: model streams a fragmented tool call for read_file
    args = json.dumps({"file_path": str(tmp_path / "notes.txt")})
    turn1 = [
        chunk({"role": "assistant", "content": "Let me look."}),
        chunk({"tool_calls": [{"index": 0, "id": "call_a",
                               "function": {"name": "read_file",
                                            "arguments": args[:10]}}]}),
        chunk({"tool_calls": [{"index": 0,
                               "function": {"arguments": args[10:]}}]}),
        chunk({}, finish="tool_calls",
              usage={"prompt_tokens": 100, "completion_tokens": 20,
                     "total_tokens": 120}),
    ]
    # turn 2: model answers from the tool result
    turn2 = [
        chunk({"content": "The notes say the answer is 42."}),
        chunk({}, finish="stop",
              usage={"prompt_tokens": 150, "completion_tokens": 10,
                     "total_tokens": 160}),
    ]

    events, server = asyncio.run(run_session(tmp_path, [turn1, turn2], "what do the notes say?"))

    kinds = [e.type for e in events]
    assert GeminiEventType.TOOL_CALL_REQUEST in kinds
    texts = "".join(e.value for e in events if e.type == GeminiEventType.CONTENT)
    assert "answer is 42" in texts

    # wire-format assertions on what the harness actually sent
    first, second = server.requests
    assert first["messages"][0]["role"] == "system"
    assert "You are Qwen Code" in first["messages"][0]["content"]
    assert first["stream"] is True
    assert first["stream_options"] == {"include_usage": True}
    tool_names = [t["function"]["name"] for t in first["tools"]]
    assert "read_file" in tool_names and tool_names == sorted(tool_names)

    # second request: assistant tool_calls turn + role:'tool' response
    roles = [m["role"] for m in second["messages"]]
    assert "tool" in roles
    tool_msg = next(m for m in second["messages"] if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_a"
    assert "the answer is 42" in tool_msg["content"]
    assistant_turns = [m for m in second["messages"]
                       if m["role"] == "assistant" and m.get("tool_calls")]
    assert assistant_turns and assistant_turns[0]["tool_calls"][0]["id"] == "call_a"
