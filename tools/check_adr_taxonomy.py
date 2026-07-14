#!/usr/bin/env python3
"""Validate the frozen ADR taxonomy and pull-request classification."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "docs" / "adr-index.md"
FROZEN_BUCKETS = ("FND", "SEM", "SAF", "RTG", "CAS", "MEM", "LRN", "EVL", "OPS")
DECISION_STATES = {"Proposed", "Accepted", "Deferred", "Rejected", "Superseded"}
MATURITY_LEVELS = {
    "D0 Design",
    "D1 Code",
    "D2 Tested",
    "D3 Shadow",
    "D4 Pilot",
    "D5 Graduated",
}
DECISION_ID_RE = re.compile(r"^ADRL-([A-Z]{3})-(\d{3})$")
MARKDOWN_LINK_RE = re.compile(r"\]\(([^)]+)\)")
PR_FIELD_RE = re.compile(
    r"^[ \t]*-[ \t]*([^:\r\n]+):[ \t]*([^\r\n]*?)[ \t]*$", re.MULTILINE
)
DECISION_EFFECTS = {"implements", "adds evidence", "changes", "supersedes", "none"}


def _section(text: str, heading: str, next_heading: str) -> str:
    try:
        return text.split(heading, 1)[1].split(next_heading, 1)[0]
    except IndexError:
        return ""


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_decisions(text: str) -> tuple[list[dict[str, str]], list[str]]:
    decisions: list[dict[str, str]] = []
    errors: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.startswith("| `ADRL-"):
            continue
        cells = _table_cells(line)
        if len(cells) != 5:
            errors.append(
                f"line {line_number}: decision row must have 5 columns, found {len(cells)}"
            )
            continue
        decision_id = cells[0].strip("`")
        decisions.append(
            {
                "id": decision_id,
                "decision": cells[1],
                "state": cells[2],
                "maturity": cells[3],
                "evidence": cells[4],
                "line": str(line_number),
            }
        )
    return decisions, errors


def validate_index(
    index_path: Path = DEFAULT_INDEX, *, check_links: bool = True
) -> tuple[list[dict[str, str]], list[str]]:
    errors: list[str] = []
    if not index_path.is_file():
        return [], [f"index not found: {index_path}"]

    text = index_path.read_text(encoding="utf-8")
    decisions, parse_errors = parse_decisions(text)
    errors.extend(parse_errors)

    taxonomy = _section(text, "## 3. Frozen bucket taxonomy", "## 4.")
    bucket_rows: list[str] = []
    for line in taxonomy.splitlines():
        cells = _table_cells(line) if line.startswith("| `") else []
        if cells and re.fullmatch(r"`[A-Z]{3}`", cells[0]):
            bucket_rows.append(cells[0].strip("`"))
    if tuple(bucket_rows) != FROZEN_BUCKETS:
        errors.append(
            "frozen buckets changed: expected "
            + ", ".join(FROZEN_BUCKETS)
            + "; found "
            + ", ".join(bucket_rows)
        )

    seen: set[str] = set()
    bucket_numbers: dict[str, list[int]] = {bucket: [] for bucket in FROZEN_BUCKETS}
    for row in decisions:
        decision_id = row["id"]
        match = DECISION_ID_RE.fullmatch(decision_id)
        if not match:
            errors.append(f"line {row['line']}: malformed decision ID {decision_id!r}")
            continue
        if decision_id in seen:
            errors.append(f"line {row['line']}: duplicate decision ID {decision_id}")
        seen.add(decision_id)

        bucket, number_text = match.groups()
        if bucket not in bucket_numbers:
            errors.append(f"line {row['line']}: unknown bucket {bucket}")
        else:
            bucket_numbers[bucket].append(int(number_text))
        if row["state"] not in DECISION_STATES:
            errors.append(f"line {row['line']}: unknown decision state {row['state']!r}")
        if row["maturity"] not in MATURITY_LEVELS:
            errors.append(f"line {row['line']}: unknown maturity {row['maturity']!r}")
        if not row["decision"] or not row["evidence"]:
            errors.append(f"line {row['line']}: decision and evidence must be non-empty")

    for bucket, numbers in bucket_numbers.items():
        expected = list(range(1, len(numbers) + 1))
        if numbers != expected:
            errors.append(
                f"{bucket} IDs must be ordered, contiguous, and start at 001; found {numbers}"
            )

    if check_links:
        for target in sorted(set(MARKDOWN_LINK_RE.findall(text))):
            path_text = target.split("#", 1)[0]
            if not path_text or "://" in path_text or path_text.startswith("mailto:"):
                continue
            resolved = (index_path.parent / path_text).resolve()
            if not resolved.exists():
                errors.append(f"broken local evidence link: {target}")

    return decisions, errors


def _pr_fields(body: str) -> dict[str, str]:
    return {
        key.strip().lower(): value.strip().strip("`")
        for key, value in PR_FIELD_RE.findall(body)
    }


def validate_pr_body(body: str, valid_ids: set[str]) -> list[str]:
    errors: list[str] = []
    fields = _pr_fields(body)

    primary = fields.get("primary adrl id", "")
    if primary not in valid_ids:
        errors.append(f"PR primary ADRL ID is missing or unknown: {primary!r}")

    secondary = fields.get("secondary adrl ids", "")
    if not secondary:
        errors.append("PR secondary ADRL IDs field is missing; use 'none' when empty")
    elif secondary.lower() != "none":
        for decision_id in re.split(r"\s*,\s*", secondary):
            cleaned = decision_id.strip().strip("`")
            if cleaned not in valid_ids:
                errors.append(f"PR secondary ADRL ID is unknown: {cleaned!r}")

    effect = fields.get("decision effect", "").lower()
    if effect not in DECISION_EFFECTS:
        errors.append(
            "PR decision effect must select exactly one of: "
            + ", ".join(sorted(DECISION_EFFECTS))
        )

    transition = fields.get("maturity transition", "")
    if transition.lower() != "none" and not re.fullmatch(
        r"D[0-5](?:\s+[A-Za-z]+)?\s*->\s*D[0-5](?:\s+[A-Za-z]+)?", transition
    ):
        errors.append("PR maturity transition must be 'none' or 'D0 -> D1' form")

    if not fields.get("evidence produced", ""):
        errors.append("PR evidence produced field must be completed")
    readiness = fields.get("readiness contract/delta", "")
    if readiness.lower() != "none" and not re.fullmatch(
        r"[A-Za-z0-9._-]+\s+[0-9a-f]{64}:\s*"
        r"\d+(?:\.\d+)?\s*->\s*\d+(?:\.\d+)?"
        r"(?:\s*\([+-]\d+(?:\.\d+)?\))?",
        readiness,
    ):
        errors.append(
            "PR readiness contract/delta must be 'none' or "
            "'VERSION SHA256: OLD -> NEW'"
        )
    if not fields.get("architectural non-goals", ""):
        errors.append("PR architectural non-goals field must be completed")

    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument(
        "--pr-body-env",
        metavar="NAME",
        help="also validate a pull-request body read from environment variable NAME",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    decisions, errors = validate_index(args.index)
    if args.pr_body_env:
        if args.pr_body_env not in os.environ:
            errors.append(f"environment variable {args.pr_body_env!r} is not set")
        else:
            errors.extend(
                validate_pr_body(
                    os.environ[args.pr_body_env], {row["id"] for row in decisions}
                )
            )

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(
        f"ADR taxonomy OK: {len(decisions)} decisions across "
        f"{len(FROZEN_BUCKETS)} frozen buckets"
    )
    if args.pr_body_env:
        print("PR architecture classification OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
