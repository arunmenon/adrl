"""Prompt-variant experiment pipeline validation.

An 'oracle model' — an aiohttp server that plays a competent agent with
hardcoded per-task policies (read the file, apply the right edit, run the
tests, answer) — drives the REAL experiment grid end-to-end: harness loop,
scheduler, sandboxes, verifiers, and scorecard. If the oracle can't score
100%, the pipeline (not the model) is broken. Also unit-tests the variant
prompt assembly and the slim-schema diet."""

import asyncio
import json
import re

from aiohttp import web

from qwen_harness.builtins import create_tool_registry
from qwen_harness.config import Config
from qwen_harness.experiments.prompt_variants import (RunRecord, run_once,
                                                      scorecard)
from qwen_harness.experiments.tasks import TASKS
from qwen_harness.prompts import VARIANT_SECTIONS, get_core_system_prompt


# ------------------------------------------------------------ oracle model


class OracleModel:
    """Stateless policy over the visible conversation: decide the next
    assistant message the way a competent agent would."""

    def respond(self, body: dict) -> dict:
        messages = body["messages"]
        text = json.dumps(messages)
        cwd_match = re.search(r"working in the directory: (\S+?)[\\\"']", text)
        cwd = cwd_match.group(1).rstrip("\\") if cwd_match else "/tmp"
        # note: the harness merges the env-context reminder and the task
        # prompt into one user message (curated-history merging), so match
        # task markers against the whole conversation
        user_prompt = text
        did = {json.loads(tc["function"]["arguments"]).get("file_path", "")
               + ":" + tc["function"]["name"]
               for m in messages if m.get("tool_calls")
               for tc in m["tool_calls"]}
        ran_shell = any(tc["function"]["name"] == "run_shell_command"
                        for m in messages if m.get("tool_calls")
                        for tc in m["tool_calls"])

        if "RETRY_LIMIT" in user_prompt:
            if not did:
                return self._tool("read_file", {"file_path": f"{cwd}/settings.py"})
            return self._text("7")

        if "slugify" in user_prompt:
            if not did:
                content = ("import re\n\n\ndef slugify(text):\n"
                           "    return re.sub(r'\\s+', '-', text.strip().lower())"
                           ".strip('-')\n")
                return self._tool("write_file",
                                  {"file_path": f"{cwd}/strutil.py",
                                   "content": content})
            return self._text("Created strutil.py with slugify.")

        if "test suite" in user_prompt and "failing" in user_prompt:
            if f"{cwd}/calculator.py:read_file" not in did:
                return self._tool("read_file", {"file_path": f"{cwd}/calculator.py"})
            if f"{cwd}/calculator.py:edit" not in did:
                return self._tool("edit", {
                    "file_path": f"{cwd}/calculator.py",
                    "old_string": "    return a + b  # planted bug",
                    "new_string": "    return a - b"})
            if not ran_shell:
                return self._tool("run_shell_command",
                                  {"command": "python3 -m unittest discover -v"})
            return self._text("Fixed subtract; all tests pass.")

        if "parse_config" in user_prompt:
            reads = [f"{cwd}/config_loader.py", f"{cwd}/app.py"]
            for path in reads:
                if f"{path}:read_file" not in did:
                    return self._tool("read_file", {"file_path": path})
            if f"{cwd}/config_loader.py:edit" not in did:
                return self._tool("edit", {
                    "file_path": f"{cwd}/config_loader.py",
                    "old_string": "def do_stuff(path):",
                    "new_string": "def parse_config(path):"})
            if f"{cwd}/app.py:edit" not in did:
                return self._tool("edit", {
                    "file_path": f"{cwd}/app.py",
                    "old_string": "from config_loader import do_stuff\n\n"
                                  "def startup(config_path):\n"
                                  "    settings = do_stuff(config_path)",
                    "new_string": "from config_loader import parse_config\n\n"
                                  "def startup(config_path):\n"
                                  "    settings = parse_config(config_path)"})
            if not ran_shell:
                return self._tool("run_shell_command",
                                  {"command": "python3 -m unittest discover -v"})
            return self._text("Renamed do_stuff to parse_config everywhere.")

        return self._text("Nothing to do.")

    @staticmethod
    def _tool(name, args):
        return {"tool_calls": [{"id": f"call_{name}", "type": "function",
                                "function": {"name": name,
                                             "arguments": json.dumps(args)}}],
                "finish": "tool_calls"}

    @staticmethod
    def _text(text):
        return {"content": text, "finish": "stop"}


async def _serve_oracle():
    oracle = OracleModel()

    async def handle(request):
        body = await request.json()
        decision = oracle.respond(body)
        message = {"role": "assistant",
                   "content": decision.get("content"),
                   "tool_calls": decision.get("tool_calls")}
        return web.json_response({
            "choices": [{"message": message,
                         "finish_reason": decision["finish"]}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10,
                      "total_tokens": 110}})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, runner.addresses[0][1]


def test_experiment_grid_end_to_end_with_oracle():
    async def go():
        runner, port = await _serve_oracle()
        records = []
        try:
            for task in TASKS:
                for variant in ("full", "minimal-noex"):
                    record = await run_once(
                        task, variant, slim_tools=(variant != "full"), rep=0,
                        base_url=f"http://127.0.0.1:{port}/v1",
                        model="oracle", streaming=False)
                    records.append(record)
        finally:
            await runner.cleanup()
        return records

    records = asyncio.run(go())
    failures = [(r.task, r.variant, r.ended, r.detail)
                for r in records if not r.success]
    assert not failures, f"oracle should score 100%: {failures}"
    assert all(r.ended == "done" for r in records)
    assert all(r.prompt_tokens > 0 for r in records)
    # variant plumbing reached the wire: minimal prefix is far smaller
    full = next(r for r in records if r.variant == "full")
    mini = next(r for r in records if r.variant == "minimal-noex")
    assert mini.prefix_chars < full.prefix_chars * 0.5

    report = scorecard(records, "oracle", "local")
    assert "## By variant" in report and "8/8" not in report  # grouped rows
    assert "| full | no |" in report and "| minimal-noex | yes |" in report


# ---------------------------------------------------------- variant details


def test_variant_sizes_strictly_decrease(tmp_path):
    sizes = [len(get_core_system_prompt(cwd=str(tmp_path), variant=v))
             for v in ("full", "lean", "minimal", "minimal-noex")]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[-1] < sizes[0] / 5


def test_variant_section_presence(tmp_path):
    cwd = str(tmp_path)
    full = get_core_system_prompt(cwd=cwd, variant="full")
    for heading in ("# Core Mandates", "# Task Management", "# Primary Workflows",
                    "# Examples", "# Final Reminder"):
        assert heading in full
    lean = get_core_system_prompt(cwd=cwd, variant="lean")
    assert "# Task Management" not in lean and "# Examples" in lean
    assert "Executing actions with care" not in lean
    mini = get_core_system_prompt(cwd=cwd, variant="minimal")
    assert "# Primary Workflows" not in mini
    assert "## Using Your Tools" in mini and "# Examples" in mini
    noex = get_core_system_prompt(cwd=cwd, variant="minimal-noex")
    assert "# Examples" not in noex
    # every variant keeps the identity line and the memory contract
    for v in VARIANT_SECTIONS:
        p = get_core_system_prompt(user_memory="M", cwd=cwd, variant=v)
        assert p.startswith("You are Qwen Code")
        assert p.endswith("\n\n---\n\nM")


def test_slim_tool_schemas(tmp_path):
    config = Config(target_dir=str(tmp_path))
    registry = create_tool_registry(config)
    full = registry.get_function_declarations()
    slim = registry.get_function_declarations(slim=True)
    # ~22% off even though the port's descriptions are already condensed;
    # against upstream's verbose schemas the same diet cuts far more
    assert len(json.dumps(slim)) < len(json.dumps(full)) * 0.85
    by_name = {d["name"]: d for d in slim}
    edit = by_name["edit"]
    assert edit["description"].endswith(".") and "\n" not in edit["description"]
    # structure untouched: same params, same required
    full_edit = next(d for d in full if d["name"] == "edit")
    assert edit["parameters"]["required"] == full_edit["parameters"]["required"]
    assert set(edit["parameters"]["properties"]) == \
        set(full_edit["parameters"]["properties"])
