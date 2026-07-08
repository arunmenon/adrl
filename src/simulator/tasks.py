"""C2 — scenario library: randomized, ARCHETYPE-AWARE user asks.

Each scenario maps to a design-doc scenario (S#) and produces a prompt from
phrasing pools, so repeated runs generate varied-but-realistic traffic instead
of byte-identical fixtures. Phrasings intentionally include terse, sloppy, and
polite variants — real users are not uniform.

Archetype-awareness: a rename/fix/investigate/refactor ask must make sense for
the sandbox's stack. Each scenario declares which archetypes it applies to
(``archetypes``) and its prompt fn branches on ``sb['archetype']`` so a
"fix the failing test" ask names vitest for a Next.js repo and pytest for the
Python CLI. The gradable skeleton (bad_var/good_var, planted_issue) is carried
by the sandbox, so the ground truth survives even as surface phrasing varies.

``pick(rng, only=None, sb=None)`` keeps its original ``(rng, only)`` call shape
(Unit A's runner is unchanged); passing ``sb`` restricts random selection to
scenarios that fit the sandbox's archetype.
"""

from __future__ import annotations

import random

SCENARIOS: dict[str, dict] = {}

ALL_ARCHETYPES = {"python_cli", "nextjs_ts", "terraform", "sql_prisma", "docs", "monorepo"}


def scenario(sid: str, maps_to: str, difficulty: str, archetypes: set[str] | None = None):
    def wrap(fn):
        SCENARIOS[sid] = {
            "id": sid, "maps_to": maps_to, "difficulty": difficulty,
            "prompt": fn, "archetypes": archetypes or set(ALL_ARCHETYPES),
        }
        return fn
    return wrap


def _arch(sb: dict) -> str:
    return sb.get("archetype", "python_cli")


@scenario("explain", maps_to="S1/S2", difficulty="easy")
def explain(sb: dict, rng: random.Random) -> str:
    return rng.choice([
        "what does this repo do?",
        "give me a quick overview of this codebase",
        "explain what this project is and how the pieces fit together",
        f"what's the flow through {sb['project']}?",
        "walk me through this repo, keep it short",
    ])


@scenario("rename", maps_to="S2", difficulty="easy",
          archetypes={"python_cli", "nextjs_ts", "terraform", "sql_prisma", "monorepo"})
def rename(sb: dict, rng: random.Random) -> str:
    bad, good = sb["bad_var"], sb["good_var"]
    arch = _arch(sb)
    if arch == "terraform":
        pool = [
            f"rename the variable {bad} to {good} everywhere and update the outputs",
            f"var.{bad} is a bad name, change it to {good} across main.tf and outputs.tf",
        ]
    elif arch == "sql_prisma":
        pool = [
            f"rename the {bad} column to {good} in the schema and both migrations",
            f"{bad} is a terrible column name, change it to {good} across schema.prisma and the sql",
        ]
    else:
        pool = [
            f"rename {bad} to {good} everywhere",
            f"{bad} is a terrible name, change it to {good} across the codebase",
            f"please rename {bad} -> {good} in all files and make sure things still build",
        ]
    return rng.choice(pool)


@scenario("fix_test", maps_to="S3", difficulty="medium",
          archetypes={"python_cli", "nextjs_ts", "sql_prisma", "monorepo"})
def fix_test(sb: dict, rng: random.Random) -> str:
    arch = _arch(sb)
    if arch == "nextjs_ts":
        return rng.choice([
            "the vitest run is red, fix it",
            "npm test fails on formatAmount — find the bug and fix it, don't change the test",
            "amounts show truncated instead of rounded. fix the formatting bug",
        ])
    if arch == "sql_prisma":
        return rng.choice([
            "queries filtering on that column are slow — add the missing index in a new migration",
            "the migration is missing an index the app relies on. add it",
        ])
    if arch == "monorepo":
        return rng.choice([
            "the api pytest fails on padded dates. fix the parse bug, don't touch the test",
            "services/api tests are red — figure out why and fix the underlying issue",
        ])
    return rng.choice([
        "the tests are failing, fix them",
        "pytest is red. figure out why and fix it",
        "one of the tests fails — find the bug and fix it, don't just change the test",
        "CI says test_parse_date_padded fails. fix the underlying issue",
    ])


@scenario("investigate", maps_to="S4", difficulty="medium",
          archetypes={"python_cli", "terraform", "sql_prisma", "docs", "monorepo"})
def investigate(sb: dict, rng: random.Random) -> str:
    arch = _arch(sb)
    if arch == "terraform":
        return rng.choice([
            "why would `terraform validate` pass but the bucket still be non-compliant? just explain",
            "the s3 bucket is flagged for missing tags — investigate and tell me where, don't apply anything",
        ])
    if arch == "sql_prisma":
        return rng.choice([
            "why are the label queries doing a full scan? explain, don't change anything",
            "investigate why filtering is slow on this table",
        ])
    if arch == "docs":
        return rng.choice([
            "which docs are stale or incomplete? just tell me, don't edit",
            "audit the runbook — is anything missing?",
        ])
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


@scenario("feature", maps_to="write-intent", difficulty="medium",
          archetypes={"python_cli", "nextjs_ts", "terraform", "sql_prisma", "monorepo"})
def feature(sb: dict, rng: random.Random) -> str:
    arch = _arch(sb)
    ent = sb.get("entity", "record")
    if arch == "nextjs_ts":
        return rng.choice([
            f"add a search input to the {ent} list that filters by label",
            "add a total row at the bottom of the list showing the summed amount",
            "add a loading state to the page while data resolves",
        ])
    if arch == "terraform":
        return rng.choice([
            "add cost-allocation tags (Environment, Project) to every resource",
            "add a variable for instance_type and wire it into the worker",
        ])
    if arch == "sql_prisma":
        return rng.choice([
            f"add a created_at timestamp column to the {ent.capitalize()} model and a migration for it",
            "add a status enum column with a default and a migration",
        ])
    return rng.choice([
        "add a --verbose flag to the cli that prints each parsed row",
        f"add a --limit N option so it only processes the first N {ent}s. add a test",
        "add json output behind a --json flag",
    ])


@scenario("refactor", maps_to="S10", difficulty="hard",
          archetypes={"python_cli", "nextjs_ts", "monorepo"})
def refactor(sb: dict, rng: random.Random) -> str:
    arch = _arch(sb)
    if arch == "nextjs_ts":
        return rng.choice([
            "extract all the formatting helpers into a shared lib/format module and update every import",
            "the page component is doing too much — split data loading into a hook and keep the view dumb",
        ])
    if arch == "monorepo":
        return rng.choice([
            "the web app and api both format amounts differently — unify on one shared helper",
            "move the shared types into a packages/shared workspace and update both apps to import from it",
        ])
    return rng.choice([
        "refactor config access across the codebase: introduce a Settings class, inject it instead of "
        "importing config directly in stats, limiter and cli. keep tests green and migrate all call sites",
        "restructure this project: split parsing, stats and cli into a proper layered design with an "
        "interfaces module, update everything that touches them, tests must pass",
    ])


def applicable(sb: dict | None) -> list[str]:
    """Scenario ids that fit the sandbox's archetype (all if sb is None)."""
    if sb is None:
        return list(SCENARIOS)
    arch = _arch(sb)
    return [sid for sid, sc in SCENARIOS.items() if arch in sc["archetypes"]]


def pick(rng: random.Random, only: str | None = None, sb: dict | None = None) -> dict:
    """Pick a scenario. ``only`` forces a specific id (original behavior). When
    ``only`` is None/"random" and ``sb`` is given, selection is restricted to
    scenarios that fit the sandbox's archetype."""
    if only and only != "random":
        return SCENARIOS[only]
    choices = applicable(sb)
    return SCENARIOS[rng.choice(choices)]
