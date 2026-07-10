"""Tool-layer tests: scheduler state machine + approval modes, edit
semantics, shell classification, converter wire format, prompts/memory."""

import asyncio
import json

from qwen_harness.config import ApprovalMode, Config
from qwen_harness.builtins import create_tool_registry
from qwen_harness.content_generator import (StreamingToolCallParser,
                                            contents_to_openai_messages)
from qwen_harness.memory import load_hierarchical_memory
from qwen_harness.prompts import get_core_system_prompt
from qwen_harness.scheduler import CoreToolScheduler, truncate_output
from qwen_harness.tools.base import ToolConfirmationOutcome
from qwen_harness.tools.shell import get_command_roots, is_command_read_only
from qwen_harness.types import (Content, FunctionCall, FunctionResponse, Part,
                                ToolCallRequestInfo)


def run(coro):
    return asyncio.run(coro)


def make_scheduler(tmp_path, approval=ApprovalMode.DEFAULT, handler=None):
    config = Config(target_dir=str(tmp_path), approval_mode=approval)
    registry = create_tool_registry(config)
    return config, registry, CoreToolScheduler(config, registry, handler)


def request(name, args, call_id="c1"):
    return ToolCallRequestInfo(call_id=call_id, name=name, args=args)


# ------------------------------------------------------------- state machine


def test_unknown_tool_reports_not_registered(tmp_path):
    _, _, scheduler = make_scheduler(tmp_path)
    [response] = run(scheduler.schedule([request("no_such_tool", {})]))
    payload = response.response_parts[0].function_response.response
    assert "not found in registry" in payload["error"]


def test_invalid_params_error_and_retry_loop_hint(tmp_path):
    _, _, scheduler = make_scheduler(tmp_path)
    for i in range(3):
        [response] = run(scheduler.schedule([request("read_file", {})]))
    payload = response.response_parts[0].function_response.response
    assert "Invalid parameters" in payload["error"]
    assert "RETRY LOOP DETECTED" in payload["error"]


def test_read_is_auto_allowed_in_workspace(tmp_path):
    (tmp_path / "f.txt").write_text("data")
    _, _, scheduler = make_scheduler(tmp_path)  # DEFAULT mode, no handler
    [response] = run(scheduler.schedule(
        [request("read_file", {"file_path": str(tmp_path / "f.txt")})]))
    assert response.error is None
    assert "data" in response.response_parts[0].function_response.response["output"]


def test_write_denied_when_non_interactive_default_mode(tmp_path):
    _, _, scheduler = make_scheduler(tmp_path)
    [response] = run(scheduler.schedule(
        [request("write_file", {"file_path": str(tmp_path / "x.txt"),
                                "content": "hi"})]))
    payload = response.response_parts[0].function_response.response
    assert "non-interactive" in payload["error"]
    assert not (tmp_path / "x.txt").exists()


def test_auto_edit_approves_edits_but_not_shell(tmp_path):
    _, _, scheduler = make_scheduler(tmp_path, ApprovalMode.AUTO_EDIT)
    [w] = run(scheduler.schedule(
        [request("write_file", {"file_path": str(tmp_path / "x.txt"),
                                "content": "hi"})]))
    assert w.error is None and (tmp_path / "x.txt").read_text() == "hi"
    [s] = run(scheduler.schedule([request("run_shell_command",
                                          {"command": "touch y.txt"})]))
    assert s.error is not None  # shell still needs a prompt -> auto-denied


def test_yolo_runs_everything(tmp_path):
    _, _, scheduler = make_scheduler(tmp_path, ApprovalMode.YOLO)
    [s] = run(scheduler.schedule([request(
        "run_shell_command", {"command": "echo hello"})]))
    out = s.response_parts[0].function_response.response["output"]
    assert "hello" in out and "Exit Code: 0" in out


def test_plan_mode_blocks_mutations_allows_reads(tmp_path):
    (tmp_path / "f.txt").write_text("data")
    _, _, scheduler = make_scheduler(tmp_path, ApprovalMode.PLAN)
    [r] = run(scheduler.schedule(
        [request("read_file", {"file_path": str(tmp_path / "f.txt")})]))
    assert r.error is None
    [w] = run(scheduler.schedule(
        [request("write_file", {"file_path": str(tmp_path / "x.txt"),
                                "content": "hi"})]))
    assert "Plan mode blocked" in w.response_parts[0].function_response.response["error"]


def test_confirmation_handler_cancel_and_always(tmp_path):
    outcomes = [ToolConfirmationOutcome.CANCEL, ToolConfirmationOutcome.PROCEED_ALWAYS]
    seen = []

    async def handler(call, details):
        seen.append(details.type)
        return outcomes.pop(0)

    _, _, scheduler = make_scheduler(tmp_path, ApprovalMode.DEFAULT, handler)
    args = {"file_path": str(tmp_path / "x.txt"), "content": "hi"}
    [denied] = run(scheduler.schedule([request("write_file", args)]))
    assert "Cancelled" in denied.response_parts[0].function_response.response["error"]
    assert denied.error is None  # cancellation is not a failure

    [approved] = run(scheduler.schedule([request("write_file", args)]))
    assert approved.error is None
    # PROCEED_ALWAYS: third call skips the handler entirely
    [again] = run(scheduler.schedule([request("write_file", args)]))
    assert again.error is None and len(seen) == 2
    assert seen == ["edit", "edit"]


# ----------------------------------------------------------------- edit tool


def edit_args(path, old, new, **kw):
    return {"file_path": str(path), "old_string": old, "new_string": new, **kw}


def test_edit_occurrence_semantics(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\ny = 1\n")
    _, _, scheduler = make_scheduler(tmp_path, ApprovalMode.YOLO)

    # must read before editing (prior-read enforcement)
    [blocked] = run(scheduler.schedule([request("edit", edit_args(f, "= 1", "= 2"))]))
    assert "read the file" in blocked.response_parts[0].function_response.response["error"]

    run(scheduler.schedule([request("read_file", {"file_path": str(f)})]))
    [ambiguous] = run(scheduler.schedule([request("edit", edit_args(f, "= 1", "= 2"))]))
    assert "2 occurrences" in ambiguous.response_parts[0].function_response.response["error"]

    [ok] = run(scheduler.schedule([request("edit",
                                           edit_args(f, "= 1", "= 2", replace_all=True))]))
    assert ok.error is None and f.read_text() == "x = 2\ny = 2\n"

    [missing] = run(scheduler.schedule([request("edit", edit_args(f, "z = 9", "z = 0"))]))
    assert "could not find the string" in \
        missing.response_parts[0].function_response.response["error"]


def test_edit_creates_file_with_empty_old_string(tmp_path):
    f = tmp_path / "new.txt"
    _, _, scheduler = make_scheduler(tmp_path, ApprovalMode.YOLO)
    [ok] = run(scheduler.schedule([request("edit", edit_args(f, "", "content"))]))
    assert ok.error is None and f.read_text() == "content"
    [dup] = run(scheduler.schedule([request("edit", edit_args(f, "", "other"))]))
    assert "already exists" in dup.response_parts[0].function_response.response["error"]


def test_edit_flexible_whitespace_fallback(tmp_path):
    f = tmp_path / "w.py"
    f.write_text("def f():   \n    return 1\n")  # trailing spaces on line 1
    _, _, scheduler = make_scheduler(tmp_path, ApprovalMode.YOLO)
    run(scheduler.schedule([request("read_file", {"file_path": str(f)})]))
    [ok] = run(scheduler.schedule([request("edit", edit_args(
        f, "def f():\n    return 1", "def f():\n    return 2"))]))
    assert ok.error is None and "return 2" in f.read_text()


# --------------------------------------------------------------------- shell


def test_shell_read_only_classification():
    assert is_command_read_only("git status && git diff")
    assert is_command_read_only("ls -la | grep foo")
    assert not is_command_read_only("git push")
    assert not is_command_read_only("echo $(rm -rf /)")   # substitution
    assert not is_command_read_only("rm -rf /tmp/x")
    assert get_command_roots("git add . && npm test") == ["git", "npm"]


def test_shell_output_block_format(tmp_path):
    _, _, scheduler = make_scheduler(tmp_path, ApprovalMode.YOLO)
    [r] = run(scheduler.schedule([request("run_shell_command",
                                          {"command": "false"})]))
    payload = r.response_parts[0].function_response.response
    assert "Exit Code: 1" in payload["error"]
    assert r.error is not None


# ---------------------------------------------------------- truncation shape


def test_truncate_output_keeps_head_and_tail():
    text = "A" * 5000 + "MIDDLE" + "Z" * 5000
    out = truncate_output(text, 1000, keep="both")
    assert "CONTENT TRUNCATED" in out
    assert out.count("A") >= 100 and out.count("Z") >= 100 and "MIDDLE" not in out


# --------------------------------------------------------- wire conversion


def test_gemini_to_openai_message_mapping():
    contents = [
        Content(role="user", parts=[Part(text="hi")]),
        Content(role="model", parts=[
            Part(text="checking"),
            Part(function_call=FunctionCall(name="read_file",
                                            args={"file_path": "/a"}, id="c1"))]),
        Content(role="user", parts=[
            Part(function_response=FunctionResponse(
                name="read_file", id="c1", response={"output": "data"}))]),
    ]
    messages = contents_to_openai_messages(contents, system_instruction="SYS")
    assert messages[0] == {"role": "system", "content": "SYS"}
    assert messages[1] == {"role": "user", "content": "hi"}
    assistant = messages[2]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "c1"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == \
        {"file_path": "/a"}
    tool = messages[3]
    assert tool == {"role": "tool", "tool_call_id": "c1", "content": "data"}


def test_streaming_tool_call_parser_reassembles_fragments():
    parser = StreamingToolCallParser()
    parser.add_delta([{"index": 0, "id": "c9",
                       "function": {"name": "edit", "arguments": '{"file'}}])
    parser.add_delta([{"index": 0, "function": {"arguments": '_path": "/a"}'}}])
    assert not parser.has_incomplete_calls()
    [call] = parser.emit()
    assert call.id == "c9" and call.name == "edit"
    assert call.args == {"file_path": "/a"}


def test_incomplete_json_detected():
    parser = StreamingToolCallParser()
    parser.add_delta([{"index": 0, "function": {"name": "edit",
                                                "arguments": '{"file_path": "/a'}}])
    assert parser.has_incomplete_calls()


# ------------------------------------------------------------ prompts/memory


def test_memory_discovery_order_and_framing(tmp_path):
    home = tmp_path / "home"
    (home / ".qwen").mkdir(parents=True)
    (home / ".qwen" / "QWEN.md").write_text("global rule")
    project = tmp_path / "proj"
    (project / ".git").mkdir(parents=True)
    (project / "QWEN.md").write_text("project rule")
    sub = project / "sub"
    sub.mkdir()
    (sub / "QWEN.md").write_text("sub rule")
    (project / ".qwen").mkdir()
    (project / ".qwen" / "QWEN.local.md").write_text("local override")

    memory, count = load_hierarchical_memory(sub, home=home)
    assert count == 4
    order = [memory.index(s) for s in
             ("global rule", "project rule", "sub rule", "local override")]
    assert order == sorted(order)  # general -> specific, local last
    assert "--- Context from:" in memory and "--- End of Context from:" in memory


def test_system_prompt_memory_suffix_separator(tmp_path):
    p = get_core_system_prompt(user_memory="remember me", cwd=str(tmp_path))
    assert p.endswith("\n\n---\n\nremember me")
    base = get_core_system_prompt(cwd=str(tmp_path))
    assert "---\n\nremember me" not in base
    assert base.startswith("You are Qwen Code")
    assert "# Final Reminder" in base
