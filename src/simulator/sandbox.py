"""C2 — project-archetype registry for simulator sandboxes.

Every sandbox is a REAL, buildable minimal project (no mock work — the agent
reads real code, runs the stack's real checks, applies real edits) with
deliberately planted, GRADABLE issues appropriate to its stack. The registry
exists so synthetic traffic reproduces the corpus file-touch mix (TS/TSX ~58%,
Markdown ~14%, SQL ~9%, Python <=10%, plus Terraform/Docker/config) instead of
the old 100%-Python toy.

Archetypes (see ARCHETYPES):
    nextjs_ts   Next.js + TypeScript webapp        (.ts/.tsx dominant)
    monorepo    mixed apps/services/infra/db/docs   (ts + py + tf + sql + md)
    docs        docs-heavy repo                     (.md dominant)
    sql_prisma  Prisma schema + SQL migrations      (.sql/.prisma)
    terraform   Terraform / IaC module              (.tf)
    python_cli  the original CSV tool               (.py)

Each archetype plants at least one gradable issue in its dominant language:
    - a badly named identifier used across >=2 files   (rename scenario, S2)
    - a stack-appropriate defect                        (fix/investigate, S3/S4)
so scenarios have genuine work regardless of stack.

BIG-CONTEXT seeding (heavy=True) adds large REAL source/text bulk so a main
turn's accumulated context crosses the 32k local-fit ceiling (two-regime
structural profile). context_tokens_estimate reports the seeded bulk.

STRESS GENERATOR (plant_secret=True): a small fraction of sandboxes get a
CLEARLY-FAKE .env / connection string with masked-shaped PLACEHOLDER values
(never a real secret). This is a stress generator for the secret-scanner /
utility-pinning path, NOT a realism-calibration target — 5 secret events from
one user have no statistical power.

make_sandbox() return shape (keys Unit A's runner depends on are preserved):
    path                    Path    sandbox root (a real git repo)
    project                 str     primary package/app name
    bad_var                 str     planted bad identifier (rename target)
    good_var                str     its intended name
    entity                  str     domain noun used across the project
    archetype               str     registry key (new)
    language                str     dominant language label (new)
    context_tokens_estimate int     ~chars/4 over text source files (new)
    dominant_ext            str     e.g. ".tsx" (new)
    test_cmd                str     the stack's real check command (new)
    planted_issue           str     short label of the gradable defect (new)
    planted_secret          bool    whether a FAKE secret was planted (new)
"""

from __future__ import annotations

import random
import subprocess
import time
from pathlib import Path

# --- rename targets, per language family (identifier used across >=2 files) ---
PY_BAD = {"usr_cnt": "user_count", "tmp_val": "temp_value", "dat_lst": "data_list",
          "res_obj": "result", "cfg_dct": "config_dict"}
TS_BAD = {"usrCnt": "userCount", "tmpVal": "tempValue", "datLst": "dataList",
          "resObj": "result", "cfgDct": "configDict"}
SQL_BAD = {"usr_cnt": "user_count", "tot_amt": "total_amount", "crt_at": "created_at",
           "st_flag": "status_flag"}
TF_BAD = {"inst_cnt": "instance_count", "bkt_nm": "bucket_name", "env_tg": "environment_tag"}

ENTITIES = ["order", "ticket", "event", "record", "invoice", "shipment", "account"]
PROJECT_NAMES = ["shiplog", "orderflow", "metricd", "queuepilot", "tagstore",
                 "fleetsync", "webly", "dashkit", "gridpay", "notevault"]

# Selection weights tuned so a batch's file-extension mix approximates the
# corpus (TS/TSX-dominant). TS-heavy archetypes carry the most weight.
ARCHETYPE_WEIGHTS = {
    "nextjs_ts": 0.45,
    "monorepo": 0.20,
    "docs": 0.09,
    "sql_prisma": 0.10,
    "terraform": 0.10,
    "python_cli": 0.06,
}

# Compatibility with the previous module surface (kept so nothing importing
# these names breaks). The current code paths use the per-language maps above.
BAD_VARS = list(PY_BAD)
GOOD_VARS = dict(PY_BAD)

TEXT_EXTS = {".ts", ".tsx", ".js", ".jsx", ".py", ".md", ".sql", ".prisma",
             ".tf", ".json", ".yml", ".yaml", ".toml", ".sh", ".css", ".env",
             ".txt", ".cfg", ".ini", "Dockerfile"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _iter_source_files(proj: Path):
    for p in proj.rglob("*"):
        if not p.is_file():
            continue
        if ".git" in p.parts:
            continue
        if p.suffix in TEXT_EXTS or p.name in TEXT_EXTS:
            yield p


def _estimate_tokens(proj: Path) -> int:
    """~chars/4 over text source files (excludes .git). Cheap, stack-agnostic."""
    chars = 0
    for p in _iter_source_files(proj):
        try:
            chars += len(p.read_text())
        except (UnicodeDecodeError, OSError):
            continue
    return chars // 4


def _git_init(proj: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=proj, check=True)
    subprocess.run(["git", "add", "-A"], cwd=proj, check=True)
    subprocess.run(
        ["git", "-c", "user.name=sim", "-c", "user.email=sim@local",
         "commit", "-qm", "initial"],
        cwd=proj, check=True,
    )


def _plant_fake_secret(proj: Path, rng: random.Random) -> None:
    """STRESS GENERATOR ONLY. Writes a CLEARLY-FAKE .env with masked-shaped
    PLACEHOLDER values. Never a real credential. Exercises the secret-scanner /
    utility-pinning path; not a realism-calibration target."""
    token = f"FAKE{rng.randrange(10**8):08d}NOTREAL"
    _write(proj / ".env.example", (
        "# FAKE placeholder secrets for simulator stress-testing — NOT real.\n"
        "# Values are masked-shaped placeholders; do not use.\n"
        "DATABASE_URL=postgres://PLACEHOLDER_USER:PLACEHOLDER_PASS@localhost:5432/appdb\n"
        f"API_KEY=REPLACE_ME_{token}\n"
        "STRIPE_SECRET_KEY=sk_test_0000000000000000000000000000\n"
        "JWT_SIGNING_SECRET=CHANGE_ME_placeholder_do_not_use\n"
    ))


# --------------------------------------------------------------------------- #
# heavy (big-context) seeding — real, valid bulk in the dominant language
# --------------------------------------------------------------------------- #
def _gen_tsx_component(name: str, entity: str, i: int) -> str:
    return f'''import {{ useState, useMemo }} from "react";

export interface {name}Props {{
  {entity}s: Array<{{ id: string; label: string; amount: number }}>;
  onSelect?: (id: string) => void;
}}

/** Generated real component #{i} for the {entity} dashboard. */
export function {name}({{ {entity}s, onSelect }}: {name}Props) {{
  const [query, setQuery] = useState("");
  const filtered = useMemo(
    () => {entity}s.filter((r) => r.label.toLowerCase().includes(query.toLowerCase())),
    [{entity}s, query],
  );
  const total = filtered.reduce((acc, r) => acc + r.amount, 0);
  return (
    <section className="{name.lower()}">
      <input value={{query}} onChange={{(e) => setQuery(e.target.value)}} placeholder="filter" />
      <ul>
        {{filtered.map((r) => (
          <li key={{r.id}} onClick={{() => onSelect?.(r.id)}}>
            {{r.label}} — {{r.amount.toFixed(2)}}
          </li>
        ))}}
      </ul>
      <footer>total: {{total.toFixed(2)}}</footer>
    </section>
  );
}}
'''


def _gen_ts_module(name: str, i: int) -> str:
    return f'''// Generated real module #{i}: pure helpers, no side effects.
export function {name}Reduce(values: number[]): number {{
  return values.reduce((acc, v) => acc + v, 0);
}}

export function {name}Normalize(rows: Record<string, number>): Record<string, number> {{
  const total = Object.values(rows).reduce((a, b) => a + b, 0) || 1;
  const out: Record<string, number> = {{}};
  for (const [k, v] of Object.entries(rows)) out[k] = v / total;
  return out;
}}

export interface {name}Options {{ limit: number; ascending: boolean; }}
export const default{name}Options: {name}Options = {{ limit: 50, ascending: true }};
'''


def _gen_sql_block(entity: str, i: int) -> str:
    return f'''-- Generated real migration block #{i}
CREATE TABLE IF NOT EXISTS {entity}_archive_{i} (
    id           TEXT PRIMARY KEY,
    {entity}_ref TEXT NOT NULL,
    total_amount NUMERIC(12, 2) NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_{entity}_archive_{i}_ref ON {entity}_archive_{i} ({entity}_ref);
'''


def _gen_md_doc(title: str, entity: str, i: int) -> str:
    body = "\n\n".join(
        f"### {title} section {j}\n\nThe {entity} pipeline processes records in "
        f"batches. This section documents invariant {j}: every {entity} carries a "
        f"stable id, a monotonic created_at, and a validated amount. Downstream "
        f"consumers must not assume ordering across batches. See ADR-{i}-{j} for the "
        f"rationale and the migration path from the legacy {entity} store."
        for j in range(1, 9)
    )
    return f"# {title} ({i})\n\n{body}\n"


def _heavy_seed(proj: Path, rng: random.Random, archetype: str, entity: str) -> None:
    """Add large REAL bulk so context_tokens_estimate crosses the 32k ceiling.
    Content is generated but valid (it parses) — not mock data."""
    target_chars = rng.randint(160_000, 520_000)  # ~40k..130k tokens
    written = 0
    i = 0
    # A long, real design/plan doc always accompanies heavy sandboxes (mirrors
    # the "pasted plan doc" shape in the corpus).
    plan = _gen_md_doc("Architecture & migration plan", entity, 0)
    plan = plan + "\n\n".join(_gen_md_doc("Appendix", entity, k) for k in range(1, 6))
    _write(proj / "docs" / "DESIGN.md", plan)
    written += len(plan)

    while written < target_chars:
        i += 1
        if archetype in ("nextjs_ts", "monorepo"):
            comp = _gen_tsx_component(f"Panel{i}", entity, i)
            mod = _gen_ts_module(f"agg{i}", i)
            _write(proj / "vendor" / "components" / f"Panel{i}.tsx", comp)
            _write(proj / "vendor" / "lib" / f"agg{i}.ts", mod)
            written += len(comp) + len(mod)
        elif archetype == "sql_prisma":
            blk = _gen_sql_block(entity, i)
            _write(proj / "migrations" / f"9{i:03d}_archive" / "migration.sql", blk)
            written += len(blk)
        else:  # docs / terraform / python — bulk as real long-form docs
            doc = _gen_md_doc(f"Runbook {i}", entity, i)
            _write(proj / "docs" / f"runbook_{i}.md", doc)
            written += len(doc)


# --------------------------------------------------------------------------- #
# archetype builders — each writes a REAL minimal project, returns extra sb keys
# --------------------------------------------------------------------------- #
def _build_python_cli(proj: Path, rng: random.Random, entity: str) -> dict:
    name = rng.choice(PROJECT_NAMES)
    bad_var = rng.choice(list(PY_BAD))
    good_var = PY_BAD[bad_var]
    pkg = proj / name

    _write(pkg / "__init__.py", "")
    _write(pkg / "config.py", f'''"""Runtime configuration for {name}."""

DEBUG = False
ENABLE_RATE_LIMIT = False  # planted: investigate scenario asks why limiting never happens
RATE_LIMIT_PER_MINUTE = 60
DEFAULT_BATCH_SIZE = 25
''')
    _write(pkg / "stats.py", f'''"""Aggregate statistics over processed {entity}s."""

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
    _write(pkg / "parse.py", f'''"""Parsing helpers for inbound {entity} payloads."""

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
    _write(pkg / "limiter.py", f'''"""Naive fixed-window rate limiter for the {entity} API."""

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
    _write(pkg / "cli.py", f'''"""Command-line entrypoint: {name} <file> — summarize a {entity} export."""

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
    {bad_var} = len(rows)  # shares the badly-named identifier across modules
    result = summarize({bad_var}, [r["amount"] for r in rows])
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
''')
    _write(proj / "tests" / "test_parse.py", f'''from {name}.parse import parse_date, parse_{entity}


def test_parse_date_plain():
    assert parse_date("2024-01-02").day == 2


def test_parse_date_padded():
    # fails until parse_date strips input — the planted fix-scenario bug
    assert parse_date(" 2024-01-02 ").day == 2


def test_parse_{entity}():
    row = parse_{entity}("a1,2024-01-02,9.5")
    assert row["amount"] == 9.5
''')
    _write(proj / "tests" / "test_stats.py", f'''from {name}.stats import merge_counts, summarize


def test_summarize():
    out = summarize(2, [1.0, 3.0])
    assert out["mean"] == 2.0


def test_merge_counts():
    assert merge_counts({{"count": 2}}, {{"count": 3}})["count"] == 5
''')
    _write(proj / "README.md", (
        f"# {name}\n\nSmall {entity}-processing tool: parse a csv export, summarize "
        f"amounts, rate-limit API calls.\nRun tests with `python -m pytest -q`.\n"
    ))
    _write(proj / "pytest.ini", "[pytest]\npythonpath = .\n")
    return {
        "project": name, "bad_var": bad_var, "good_var": good_var,
        "language": "python", "dominant_ext": ".py",
        "test_cmd": "python -m pytest -q",
        "planted_issue": "parse_date fails on padded input (fix); rate limit disabled (investigate)",
    }


def _build_nextjs_ts(proj: Path, rng: random.Random, entity: str) -> dict:
    name = rng.choice(PROJECT_NAMES)
    bad_var = rng.choice(list(TS_BAD))
    good_var = TS_BAD[bad_var]
    Ent = entity.capitalize()

    _write(proj / "package.json", (
        '{\n'
        f'  "name": "{name}",\n'
        '  "version": "0.1.0",\n'
        '  "private": true,\n'
        '  "scripts": {\n'
        '    "dev": "next dev",\n'
        '    "build": "next build",\n'
        '    "test": "vitest run",\n'
        '    "typecheck": "tsc --noEmit"\n'
        '  },\n'
        '  "dependencies": { "next": "14.2.3", "react": "18.3.1", "react-dom": "18.3.1" },\n'
        '  "devDependencies": { "typescript": "5.4.5", "vitest": "1.6.0", "@types/react": "18.3.3" }\n'
        '}\n'
    ))
    _write(proj / "tsconfig.json", (
        '{\n'
        '  "compilerOptions": {\n'
        '    "target": "ES2020",\n'
        '    "lib": ["dom", "dom.iterable", "esnext"],\n'
        '    "strict": true,\n'
        '    "jsx": "preserve",\n'
        '    "moduleResolution": "bundler",\n'
        '    "esModuleInterop": true,\n'
        '    "skipLibCheck": true\n'
        '  },\n'
        '  "include": ["**/*.ts", "**/*.tsx"]\n'
        '}\n'
    ))
    _write(proj / "next.config.ts",
           'import type { NextConfig } from "next";\n\n'
           'const config: NextConfig = { reactStrictMode: true };\n\n'
           'export default config;\n')
    _write(proj / "app" / "layout.tsx",
           'export default function RootLayout({ children }: { children: React.ReactNode }) {\n'
           '  return (\n    <html lang="en">\n      <body>{children}</body>\n    </html>\n  );\n}\n')
    _write(proj / "app" / "page.tsx", f'''import {{ {Ent}List }} from "../components/{Ent}List";
import {{ load{Ent}s }} from "../lib/{entity}";

export default async function Page() {{
  const {entity}s = await load{Ent}s();
  return (
    <main>
      <h1>{Ent}s</h1>
      <{Ent}List {entity}s={{{entity}s}} />
    </main>
  );
}}
''')
    _write(proj / "components" / f"{Ent}List.tsx", f'''import {{ formatAmount }} from "../lib/format";

export interface {Ent} {{ id: string; label: string; amount: number; }}

export function {Ent}List({{ {entity}s }}: {{ {entity}s: {Ent}[] }}) {{
  return (
    <ul>
      {{{entity}s.map(({entity}) => (
        <li key={{{entity}.id}}>{{{entity}.label}}: {{formatAmount({entity}.amount)}}</li>
      ))}}
    </ul>
  );
}}
''')
    _write(proj / "components" / "Button.tsx",
           'export function Button({ label, onClick }: { label: string; onClick?: () => void }) {\n'
           '  return <button onClick={onClick}>{label}</button>;\n}\n')
    _write(proj / "components" / "Header.tsx",
           'export function Header({ title }: { title: string }) {\n'
           '  return (\n    <header>\n      <h1>{title}</h1>\n    </header>\n  );\n}\n')
    _write(proj / "components" / f"{Ent}Card.tsx", f'''import {{ formatAmount }} from "../lib/format";

export function {Ent}Card({{ label, amount }}: {{ label: string; amount: number }}) {{
  return (
    <article className="card">
      <span>{{label}}</span>
      <strong>{{formatAmount(amount)}}</strong>
    </article>
  );
}}
''')
    # bad_var planted across two lib modules (rename target)
    _write(proj / "lib" / "format.ts", f'''// planted: formatAmount truncates instead of rounding — fix scenario.
export function formatAmount(value: number): string {{
  const {bad_var} = Math.trunc(value * 100) / 100; // BUG: should round to 2 dp
  return {bad_var}.toFixed(2);
}}
''')
    _write(proj / "lib" / f"{entity}.ts", f'''import type {{ {Ent} }} from "../components/{Ent}List";

export async function load{Ent}s(): Promise<{Ent}[]> {{
  const {bad_var} = 3; // sample size; shares the badly-named identifier
  return Array.from({{ length: {bad_var} }}, (_, i) => ({{
    id: String(i),
    label: "{entity}-" + i,
    amount: (i + 1) * 10.005,
  }}));
}}
''')
    _write(proj / "lib" / "format.test.ts", '''import { describe, it, expect } from "vitest";
import { formatAmount } from "./format";

describe("formatAmount", () => {
  it("rounds to two decimals", () => {
    // fails until the truncation bug is fixed
    expect(formatAmount(10.005)).toBe("10.01");
  });
});
''')
    _write(proj / "README.md", (
        f"# {name}\n\nNext.js + TypeScript dashboard for {entity}s. "
        f"`npm run dev` to start, `npm test` (vitest) to check.\n\n"
        f"Known issue: amounts display truncated, not rounded.\n"
    ))
    _write(proj / "Dockerfile",
           "FROM node:20-alpine\nWORKDIR /app\nCOPY . .\nRUN npm ci && npm run build\n"
           "CMD [\"npm\", \"start\"]\n")
    return {
        "project": name, "bad_var": bad_var, "good_var": good_var,
        "language": "typescript", "dominant_ext": ".tsx",
        "test_cmd": "npm test",
        "planted_issue": f"formatAmount truncates instead of rounding (fix); {bad_var} poorly named (rename)",
    }


def _build_terraform(proj: Path, rng: random.Random, entity: str) -> dict:
    name = rng.choice(PROJECT_NAMES)
    bad_var = rng.choice(list(TF_BAD))
    good_var = TF_BAD[bad_var]

    _write(proj / "versions.tf",
           'terraform {\n  required_version = ">= 1.5.0"\n'
           '  required_providers {\n    aws = {\n'
           '      source  = "hashicorp/aws"\n      version = "~> 5.0"\n'
           '    }\n  }\n}\n')
    _write(proj / "variables.tf", f'''variable "{bad_var}" {{
  description = "number of {entity} worker instances"
  type        = number
  default     = 2
}}

variable "region" {{
  type    = string
  default = "us-east-1"
}}

variable "environment" {{
  type    = string
  default = "dev"
}}
''')
    _write(proj / "main.tf", f'''provider "aws" {{
  region = var.region
}}

# planted: missing tags on the bucket — investigate/fix scenario.
resource "aws_s3_bucket" "{entity}_store" {{
  bucket = "{name}-{entity}-store"
}}

resource "aws_instance" "{entity}_worker" {{
  count         = var.{bad_var}
  ami           = "ami-0abcdef1234567890"
  instance_type = "t3.micro"
}}
''')
    _write(proj / "outputs.tf", f'''output "worker_count" {{
  value = var.{bad_var}
}}

output "bucket_name" {{
  value = aws_s3_bucket.{entity}_store.bucket
}}
''')
    _write(proj / "README.md", (
        f"# {name} infra\n\nTerraform module provisioning the {entity} workers and "
        f"store bucket.\nRun `terraform init && terraform validate` to check.\n\n"
        f"Known issue: the S3 bucket is missing required cost-allocation tags.\n"
    ))
    return {
        "project": name, "bad_var": bad_var, "good_var": good_var,
        "language": "terraform", "dominant_ext": ".tf",
        "test_cmd": "terraform validate",
        "planted_issue": f"S3 bucket missing tags (fix); var.{bad_var} poorly named (rename)",
    }


def _build_sql_prisma(proj: Path, rng: random.Random, entity: str) -> dict:
    name = rng.choice(PROJECT_NAMES)
    bad_var = rng.choice(list(SQL_BAD))
    good_var = SQL_BAD[bad_var]
    Ent = entity.capitalize()

    _write(proj / "prisma" / "schema.prisma", f'''generator client {{
  provider = "prisma-client-js"
}}

datasource db {{
  provider = "postgresql"
  url      = env("DATABASE_URL")
}}

model {Ent} {{
  id       String  @id @default(cuid())
  {bad_var} Int     @default(0)
  label    String
  amount   Decimal @db.Decimal(12, 2)
}}
''')
    _write(proj / "migrations" / "0001_init" / "migration.sql", f'''-- initial schema for {entity}s
CREATE TABLE "{Ent}" (
    "id"      TEXT PRIMARY KEY,
    "{bad_var}" INTEGER NOT NULL DEFAULT 0,
    "label"   TEXT NOT NULL,
    "amount"  NUMERIC(12, 2) NOT NULL
);
''')
    _write(proj / "migrations" / "0002_add_index" / "migration.sql", f'''-- planted: query filters on {bad_var} but there is no index — investigate scenario
CREATE INDEX "idx_{Ent}_label" ON "{Ent}" ("label");
''')
    _write(proj / "seed.sql", f'''INSERT INTO "{Ent}" ("id", "{bad_var}", "label", "amount") VALUES
    ('a1', 1, '{entity}-1', 10.00),
    ('a2', 2, '{entity}-2', 20.00);
''')
    _write(proj / "README.md", (
        f"# {name} schema\n\nPrisma schema + raw SQL migrations for the {entity} "
        f"store.\nApply with `prisma migrate deploy`.\n\n"
        f"Known issue: queries filter on `{bad_var}` but no index covers it.\n"
    ))
    return {
        "project": name, "bad_var": bad_var, "good_var": good_var,
        "language": "sql", "dominant_ext": ".sql",
        "test_cmd": "prisma validate",
        "planted_issue": f"missing index on {bad_var} (investigate); {bad_var} poorly named (rename)",
    }


def _build_docs(proj: Path, rng: random.Random, entity: str) -> dict:
    name = rng.choice(PROJECT_NAMES)
    bad_var = rng.choice(list(SQL_BAD))
    good_var = SQL_BAD[bad_var]

    _write(proj / "README.md", (
        f"# {name} handbook\n\nOperations and architecture docs for the {entity} "
        f"platform.\n\n- [Architecture](docs/architecture.md)\n"
        f"- [Runbook](docs/runbook.md)\n- [API](docs/api.md)\n"
        f"- [ADR 0001](docs/adr/0001-storage.md)\n"
    ))
    _write(proj / "docs" / "architecture.md",
           _gen_md_doc("Architecture", entity, 1))
    _write(proj / "docs" / "runbook.md",
           "# Runbook\n\n## Deploy\n\nRun the pipeline, watch the dashboard.\n\n"
           "## Rollback\n\n<!-- planted: this section is empty — fill it in -->\n")
    _write(proj / "docs" / "api.md",
           _gen_md_doc("API reference", entity, 2))
    _write(proj / "docs" / "adr" / "0001-storage.md",
           f"# ADR 0001: {entity} storage\n\nStatus: accepted\n\n"
           f"We store {entity}s in Postgres. Trade-offs documented here.\n")
    _write(proj / "mkdocs.yml",
           f"site_name: {name} handbook\nnav:\n  - Home: README.md\n"
           "  - Architecture: docs/architecture.md\n  - Runbook: docs/runbook.md\n")
    return {
        "project": name, "bad_var": bad_var, "good_var": good_var,
        "language": "markdown", "dominant_ext": ".md",
        "test_cmd": "mkdocs build --strict",
        "planted_issue": "runbook rollback section empty (fill-in); docs drift",
    }


def _build_monorepo(proj: Path, rng: random.Random, entity: str) -> dict:
    name = rng.choice(PROJECT_NAMES)
    bad_var = rng.choice(list(TS_BAD))
    good_var = TS_BAD[bad_var]
    Ent = entity.capitalize()

    _write(proj / "package.json", (
        '{\n'
        f'  "name": "{name}-monorepo",\n'
        '  "private": true,\n'
        '  "workspaces": ["apps/*", "services/*"]\n'
        '}\n'
    ))
    _write(proj / "turbo.json",
           '{\n  "$schema": "https://turbo.build/schema.json",\n'
           '  "pipeline": { "build": { "dependsOn": ["^build"] }, "test": {} }\n}\n')
    # apps/web — TS/TSX
    _write(proj / "apps" / "web" / "package.json",
           f'{{\n  "name": "@{name}/web",\n  "dependencies": {{ "next": "14.2.3", "react": "18.3.1" }}\n}}\n')
    _write(proj / "apps" / "web" / "app" / "page.tsx", f'''import {{ formatAmount }} from "../lib/format";

export default function Page() {{
  const {bad_var} = 42; // badly-named identifier shared with lib
  return <main>total {{formatAmount({bad_var})}}</main>;
}}
''')
    _write(proj / "apps" / "web" / "lib" / "format.ts", f'''export function formatAmount(value: number): string {{
  const {bad_var} = Math.round(value * 100) / 100;
  return {bad_var}.toFixed(2);
}}
''')
    _write(proj / "apps" / "web" / "components" / f"{Ent}Row.tsx",
           f'export function {Ent}Row({{ label }}: {{ label: string }}) {{\n'
           f'  return <div className="row">{{label}}</div>;\n}}\n')
    # services/api — Python with the planted whitespace bug
    _write(proj / "services" / "api" / "__init__.py", "")
    _write(proj / "services" / "api" / "parse.py",
           '"""Parse inbound payloads."""\n\nfrom datetime import datetime\n\n\n'
           'def parse_date(raw):\n'
           '    # planted: fails on padded input\n'
           '    return datetime.strptime(raw, "%Y-%m-%d")\n')
    _write(proj / "services" / "api" / "test_parse.py",
           'from parse import parse_date\n\n\n'
           'def test_padded():\n    assert parse_date(" 2024-01-02 ").day == 2\n')
    # infra — Terraform
    _write(proj / "infra" / "main.tf",
           f'resource "aws_s3_bucket" "{entity}_store" {{\n'
           f'  bucket = "{name}-{entity}"\n}}\n')
    # db — SQL
    _write(proj / "db" / "schema.sql",
           f'CREATE TABLE "{Ent}" (\n    "id" TEXT PRIMARY KEY,\n'
           f'    "label" TEXT NOT NULL\n);\n')
    # docs
    _write(proj / "README.md",
           f"# {name} monorepo\n\nMixed workspace: `apps/web` (Next.js/TS), "
           f"`services/api` (Python), `infra` (Terraform), `db` (SQL).\n")
    _write(proj / "docs" / "overview.md", _gen_md_doc("Overview", entity, 1))
    return {
        "project": name, "bad_var": bad_var, "good_var": good_var,
        "language": "typescript", "dominant_ext": ".tsx",
        "test_cmd": "npm test && (cd services/api && python -m pytest -q)",
        "planted_issue": f"api parse_date padded-input bug (fix); {bad_var} poorly named (rename)",
    }


ARCHETYPES = {
    "python_cli": _build_python_cli,
    "nextjs_ts": _build_nextjs_ts,
    "terraform": _build_terraform,
    "sql_prisma": _build_sql_prisma,
    "docs": _build_docs,
    "monorepo": _build_monorepo,
}


def _pick_archetype(rng: random.Random) -> str:
    keys = list(ARCHETYPE_WEIGHTS)
    weights = [ARCHETYPE_WEIGHTS[k] for k in keys]
    return rng.choices(keys, weights=weights, k=1)[0]


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def make_sandbox(
    root: Path,
    rng: random.Random,
    archetype: str | None = None,
    heavy: bool | None = None,
    plant_secret: bool | None = None,
) -> dict:
    """Build a real, buildable sandbox of the chosen (or weighted-random)
    archetype and return its descriptor.

    archetype     force a registry key; None -> weighted-random from the corpus mix.
    heavy         add big-context bulk (>32k tokens). None -> ~30% chance.
    plant_secret  plant a CLEARLY-FAKE .env (stress generator). None -> ~4% chance.

    Deterministic given rng. Return keys documented at module top; the
    path/project/bad_var/good_var/entity keys are preserved for Unit A's runner.
    """
    arch = archetype or _pick_archetype(rng)
    if arch not in ARCHETYPES:
        raise KeyError(f"unknown archetype {arch!r}; known: {sorted(ARCHETYPES)}")
    if heavy is None:
        heavy = rng.random() < 0.30
    if plant_secret is None:
        plant_secret = rng.random() < 0.04

    entity = rng.choice(ENTITIES)
    # unique dir name (time prefix for realism + rng suffix to avoid same-second
    # collisions when a batch is built in a tight loop)
    suffix = f"{rng.randrange(1 << 30):08x}"
    proj = root / f"{int(time.time())}-{arch}-{suffix}"
    proj.mkdir(parents=True, exist_ok=True)

    extra = ARCHETYPES[arch](proj, rng, entity)

    if heavy:
        _heavy_seed(proj, rng, arch, entity)
    if plant_secret:
        _plant_fake_secret(proj, rng)

    _git_init(proj)

    sb = {
        "path": proj,
        "entity": entity,
        "archetype": arch,
        "heavy": bool(heavy),
        "planted_secret": bool(plant_secret),
        "context_tokens_estimate": _estimate_tokens(proj),
    }
    sb.update(extra)  # project, bad_var, good_var, language, dominant_ext, test_cmd, planted_issue
    return sb
