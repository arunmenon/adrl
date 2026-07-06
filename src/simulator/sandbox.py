"""C2 — sandbox project generator for simulator sessions.

Generates a small but REAL python project (no mock work — the agent reads real
code, runs real pytest, applies real edits) with deliberately planted issues,
one per scenario type:

- a badly named variable used across two modules      (rename scenario, S2)
- a genuinely failing test: whitespace bug in parsing (fix scenario, S3)
- a rate limiter disabled by a config flag            (investigate scenario, S4)
- a CLI missing a --verbose flag                      (feature scenario)
- config access scattered across modules              (refactor scenario, S10)

Names are randomized per sandbox so repeated runs don't produce byte-identical
traffic (the point is realistic variety, not fixtures).
"""

from __future__ import annotations

import random
import subprocess
import time
from pathlib import Path

PROJECT_NAMES = ["shiplog", "orderflow", "metricd", "queuepilot", "tagstore", "fleetsync"]
BAD_VARS = ["usr_cnt", "tmp_val", "dat_lst", "res_obj", "cfg_dct"]
GOOD_VARS = {"usr_cnt": "user_count", "tmp_val": "temp_value", "dat_lst": "data_list",
             "res_obj": "result", "cfg_dct": "config_dict"}
ENTITIES = ["order", "ticket", "event", "record", "invoice"]


def make_sandbox(root: Path, rng: random.Random) -> dict:
    name = rng.choice(PROJECT_NAMES)
    bad_var = rng.choice(BAD_VARS)
    entity = rng.choice(ENTITIES)
    proj = root / f"{int(time.time())}-{name}"
    pkg = proj / name
    tests = proj / "tests"
    pkg.mkdir(parents=True)
    tests.mkdir()

    (pkg / "__init__.py").write_text("")

    (pkg / "config.py").write_text(f'''"""Runtime configuration for {name}."""

DEBUG = False
ENABLE_RATE_LIMIT = False  # planted: investigate scenario asks why limiting never happens
RATE_LIMIT_PER_MINUTE = 60
DEFAULT_BATCH_SIZE = 25
''')

    (pkg / "stats.py").write_text(f'''"""Aggregate statistics over processed {entity}s."""

from {name} import config


def summarize({bad_var}, values):
    total = sum(values)
    mean = total / len(values) if values else 0.0
    return {{
        "count": {bad_var},
        "total": total,
        "mean": mean,
        "batch_size": config.DEFAULT_BATCH_SIZE,
    }}


def merge_counts(a, b):
    {bad_var} = a.get("count", 0) + b.get("count", 0)
    return {{"count": {bad_var}}}
''')

    (pkg / "parse.py").write_text(f'''"""Parsing helpers for inbound {entity} payloads."""

from datetime import datetime


def parse_date(raw):
    # planted: fails on padded input — real users paste " 2024-01-02 "
    fmt = "%Y-%m-%d"
    return datetime.strptime(raw, fmt)


def parse_{entity}(line):
    ident, date_str, amount = line.split(",")
    return {{
        "id": ident.strip(),
        "date": parse_date(date_str),
        "amount": float(amount),
    }}
''')

    (pkg / "limiter.py").write_text(f'''"""Naive fixed-window rate limiter for the {entity} API."""

import time

from {name} import config


class RateLimiter:
    def __init__(self):
        self.window_start = time.time()
        self.count = 0

    def allow(self):
        if not config.ENABLE_RATE_LIMIT:
            return True
        now = time.time()
        if now - self.window_start > 60:
            self.window_start = now
            self.count = 0
        self.count += 1
        return self.count <= config.RATE_LIMIT_PER_MINUTE
''')

    (pkg / "cli.py").write_text(f'''"""Command-line entrypoint: {name} <file> — summarize a {entity} export."""

import argparse
import sys

from {name}.parse import parse_{entity}
from {name}.stats import summarize


def main(argv=None):
    parser = argparse.ArgumentParser(prog="{name}")
    parser.add_argument("path", help="csv export of {entity}s")
    args = parser.parse_args(argv)

    rows = []
    for line in open(args.path):
        line = line.strip()
        if line:
            rows.append(parse_{entity}(line))
    result = summarize(len(rows), [r["amount"] for r in rows])
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
''')

    (tests / "test_parse.py").write_text(f'''from {name}.parse import parse_date, parse_{entity}


def test_parse_date_plain():
    assert parse_date("2024-01-02").day == 2


def test_parse_date_padded():
    # fails until parse_date strips input — the planted fix-scenario bug
    assert parse_date(" 2024-01-02 ").day == 2


def test_parse_{entity}():
    row = parse_{entity}("a1,2024-01-02,9.5")
    assert row["amount"] == 9.5
''')

    (tests / "test_stats.py").write_text(f'''from {name}.stats import merge_counts, summarize


def test_summarize():
    out = summarize(2, [1.0, 3.0])
    assert out["mean"] == 2.0


def test_merge_counts():
    assert merge_counts({{"count": 2}}, {{"count": 3}})["count"] == 5
''')

    (proj / "README.md").write_text(
        f"# {name}\n\nSmall {entity}-processing tool: parse a csv export, summarize amounts, "
        f"rate-limit API calls.\nRun tests with `python -m pytest -q`.\n"
    )
    (proj / "pytest.ini").write_text(f"[pytest]\npythonpath = .\n")

    subprocess.run(["git", "init", "-q"], cwd=proj, check=True)
    subprocess.run(["git", "add", "-A"], cwd=proj, check=True)
    subprocess.run(
        ["git", "-c", "user.name=sim", "-c", "user.email=sim@local", "commit", "-qm", "initial"],
        cwd=proj, check=True,
    )

    return {
        "path": proj,
        "project": name,
        "bad_var": bad_var,
        "good_var": GOOD_VARS[bad_var],
        "entity": entity,
    }
