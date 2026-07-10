"""Planted-task sandboxes for the prompt-variant experiment.

Each task builds a throwaway project with a genuinely planted objective and
an *objective verifier* — ground truth is never delegated to a model (same
principle as the simulator's episode skeletons). Verifiers use only the
Python stdlib (unittest, not pytest) so sandboxes run anywhere.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class TaskResult:
    success: bool
    detail: str


@dataclass
class Task:
    name: str
    prompt: str                                   # phrased per-run at call site
    build: Callable[[Path], None]                 # plant the sandbox
    verify: Callable[[Path, str], TaskResult]     # (sandbox, final_answer_text)


def _run_unittest(sandbox: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(sandbox), "-v"],
        capture_output=True, text=True, timeout=60, cwd=sandbox)


# ------------------------------------------------------------------ fix_test


def _build_fix_test(sandbox: Path) -> None:
    (sandbox / "calculator.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "def subtract(a, b):\n"
        "    return a + b  # planted bug\n"
        "\n"
        "def multiply(a, b):\n"
        "    return a * b\n")
    (sandbox / "test_calculator.py").write_text(
        "import unittest\n"
        "import calculator\n"
        "\n"
        "class TestCalculator(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(calculator.add(2, 3), 5)\n"
        "    def test_subtract(self):\n"
        "        self.assertEqual(calculator.subtract(5, 3), 2)\n"
        "    def test_multiply(self):\n"
        "        self.assertEqual(calculator.multiply(4, 3), 12)\n")


def _verify_fix_test(sandbox: Path, answer: str) -> TaskResult:
    result = _run_unittest(sandbox)
    ok = result.returncode == 0
    return TaskResult(ok, "unittest passed" if ok else
                      f"unittest failed: {result.stderr.strip()[-200:]}")


# -------------------------------------------------------------------- rename


def _build_rename(sandbox: Path) -> None:
    (sandbox / "config_loader.py").write_text(
        "def do_stuff(path):\n"
        '    """Parse a key=value config file into a dict."""\n'
        "    result = {}\n"
        "    for line in open(path):\n"
        "        line = line.strip()\n"
        "        if line and not line.startswith('#'):\n"
        "            key, _, value = line.partition('=')\n"
        "            result[key.strip()] = value.strip()\n"
        "    return result\n")
    (sandbox / "app.py").write_text(
        "from config_loader import do_stuff\n"
        "\n"
        "def startup(config_path):\n"
        "    settings = do_stuff(config_path)\n"
        "    return settings.get('mode', 'default')\n")
    (sandbox / "test_app.py").write_text(
        "import unittest, tempfile, os\n"
        "from app import startup\n"
        "\n"
        "class TestApp(unittest.TestCase):\n"
        "    def test_startup(self):\n"
        "        fd, p = tempfile.mkstemp(); os.close(fd)\n"
        "        with open(p, 'w') as f: f.write('mode = fast\\n')\n"
        "        self.assertEqual(startup(p), 'fast')\n")


def _verify_rename(sandbox: Path, answer: str) -> TaskResult:
    source = "".join(p.read_text() for p in sandbox.glob("*.py"))
    if "do_stuff" in source:
        return TaskResult(False, "old name 'do_stuff' still present")
    if "parse_config" not in source:
        return TaskResult(False, "new name 'parse_config' not found")
    result = _run_unittest(sandbox)
    ok = result.returncode == 0
    return TaskResult(ok, "renamed + tests pass" if ok else
                      f"renamed but tests fail: {result.stderr.strip()[-200:]}")


# ------------------------------------------------------------------- read_qa


def _build_read_qa(sandbox: Path) -> None:
    filler = "\n".join(f"OPTION_{i} = {i * 3}" for i in range(40))
    (sandbox / "settings.py").write_text(
        f"# generated settings\n{filler}\nRETRY_LIMIT = 7\n"
        + "\n".join(f"FLAG_{i} = False" for i in range(40)) + "\n")


def _verify_read_qa(sandbox: Path, answer: str) -> TaskResult:
    ok = "7" in answer and "17" not in answer and "27" not in answer
    return TaskResult(ok, f"answer text: {answer.strip()[-120:]!r}")


# --------------------------------------------------------------- create_file


def _build_create_file(sandbox: Path) -> None:
    (sandbox / "README.md").write_text(
        "# strutil\nSmall string utilities live in strutil.py.\n")


def _verify_create_file(sandbox: Path, answer: str) -> TaskResult:
    target = sandbox / "strutil.py"
    if not target.is_file():
        return TaskResult(False, "strutil.py was not created")
    result = subprocess.run(
        [sys.executable, "-c",
         "from strutil import slugify; "
         "assert slugify('Hello World') == 'hello-world', slugify('Hello World'); "
         "assert slugify('  A  B  ') == 'a-b', slugify('  A  B  '); "
         "print('ok')"],
        capture_output=True, text=True, timeout=30, cwd=sandbox)
    ok = result.returncode == 0
    return TaskResult(ok, "slugify behaves" if ok else
                      f"slugify wrong: {(result.stderr or result.stdout).strip()[-200:]}")


TASKS: list[Task] = [
    Task(
        name="fix_test",
        prompt=("The test suite in this project is failing. Find the bug, fix "
                "it, and verify by running: python3 -m unittest discover -v"),
        build=_build_fix_test,
        verify=_verify_fix_test,
    ),
    Task(
        name="rename",
        prompt=("Rename the badly named function 'do_stuff' in config_loader.py "
                "to 'parse_config', updating every usage in the project. Verify "
                "with: python3 -m unittest discover -v"),
        build=_build_rename,
        verify=_verify_rename,
    ),
    Task(
        name="read_qa",
        prompt=("What is the value of RETRY_LIMIT in settings.py? "
                "Answer with just the number."),
        build=_build_read_qa,
        verify=_verify_read_qa,
    ),
    Task(
        name="create_file",
        prompt=("Create strutil.py containing a function slugify(text) that "
                "lowercases the text, collapses all whitespace runs to a single "
                "hyphen, and strips leading/trailing hyphens. No other functions."),
        build=_build_create_file,
        verify=_verify_create_file,
    ),
]
