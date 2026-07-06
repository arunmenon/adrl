"""C2 — scenario library: randomized, realistic user asks.

Each scenario maps to a design-doc scenario (S#) and produces a prompt from
phrasing pools, so repeated runs generate varied-but-realistic traffic instead
of byte-identical fixtures. Phrasings intentionally include terse, sloppy, and
polite variants — real users are not uniform.
"""

from __future__ import annotations

import random

SCENARIOS: dict[str, dict] = {}


def scenario(sid: str, maps_to: str, difficulty: str):
    def wrap(fn):
        SCENARIOS[sid] = {"id": sid, "maps_to": maps_to, "difficulty": difficulty, "prompt": fn}
        return fn
    return wrap


@scenario("explain", maps_to="S1/S2", difficulty="easy")
def explain(sb: dict, rng: random.Random) -> str:
    return rng.choice([
        "what does this repo do?",
        "give me a quick overview of this codebase",
        "explain what this project is and how the pieces fit together",
        f"what's the flow when someone runs the {sb['project']} cli?",
        "walk me through this repo, keep it short",
    ])


@scenario("rename", maps_to="S2", difficulty="easy")
def rename(sb: dict, rng: random.Random) -> str:
    return rng.choice([
        f"rename {sb['bad_var']} to {sb['good_var']} everywhere",
        f"{sb['bad_var']} is a terrible name, change it to {sb['good_var']} across the codebase",
        f"please rename the variable {sb['bad_var']} -> {sb['good_var']} in all files and make sure tests still pass",
    ])


@scenario("fix_test", maps_to="S3", difficulty="medium")
def fix_test(sb: dict, rng: random.Random) -> str:
    return rng.choice([
        "the tests are failing, fix them",
        "pytest is red. figure out why and fix it",
        "one of the tests fails — find the bug and fix it, don't just change the test",
        "CI says test_parse_date_padded fails. fix the underlying issue",
    ])


@scenario("investigate", maps_to="S4", difficulty="medium")
def investigate(sb: dict, rng: random.Random) -> str:
    return rng.choice([
        "why is the rate limiter never triggering?",
        "the rate limiter doesn't seem to do anything in prod. investigate",
        "requests are never getting rate limited even under load — why? just explain, don't change anything",
    ])


@scenario("commit_msg", maps_to="S8", difficulty="easy")
def commit_msg(sb: dict, rng: random.Random) -> str:
    return rng.choice([
        "look at the git diff and write a commit message for the staged changes, then commit",
        "commit what's staged with a sensible message",
    ])


@scenario("feature", maps_to="write-intent", difficulty="medium")
def feature(sb: dict, rng: random.Random) -> str:
    return rng.choice([
        "add a --verbose flag to the cli that prints each parsed row",
        f"add a --limit N option to the cli so it only processes the first N {sb['entity']}s. add a test",
        "add json output to the cli behind a --json flag",
    ])


@scenario("refactor", maps_to="S10", difficulty="hard")
def refactor(sb: dict, rng: random.Random) -> str:
    return rng.choice([
        "refactor config access across the codebase: introduce a Settings class, inject it instead of "
        "importing config directly in stats, limiter and cli. keep tests green and migrate all call sites",
        f"restructure this project: split parsing, stats and cli into a proper layered design with an "
        f"interfaces module, update everything that touches them across the codebase, tests must pass",
    ])


def pick(rng: random.Random, only: str | None = None) -> dict:
    if only and only != "random":
        return SCENARIOS[only]
    return SCENARIOS[rng.choice(list(SCENARIOS))]
