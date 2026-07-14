from pathlib import Path

from tools.check_adr_taxonomy import (
    DEFAULT_INDEX,
    validate_index,
    validate_pr_body,
)


def test_canonical_index_is_valid():
    decisions, errors = validate_index()

    assert errors == []
    assert len(decisions) >= 66


def test_index_rejects_duplicate_id(tmp_path: Path):
    text = DEFAULT_INDEX.read_text(encoding="utf-8")
    duplicate = next(line for line in text.splitlines() if line.startswith("| `ADRL-FND-001`"))
    candidate = tmp_path / "adr-index.md"
    candidate.write_text(text + "\n" + duplicate + "\n", encoding="utf-8")

    _, errors = validate_index(candidate, check_links=False)

    assert any("duplicate decision ID ADRL-FND-001" in error for error in errors)


def test_valid_pr_classification_passes():
    decisions, errors = validate_index()
    assert errors == []
    body = """
- Primary ADRL ID: `ADRL-OPS-005`
- Secondary ADRL IDs: `ADRL-FND-005`, `ADRL-EVL-009`
- Decision effect: `adds evidence`
- Maturity transition: `none`
- Evidence produced: validator tests and CI execution
- Readiness contract/delta: `none`
- Architectural non-goals: no runtime routing behavior changes
"""

    assert validate_pr_body(body, {row["id"] for row in decisions}) == []


def test_pr_placeholders_and_unknown_ids_fail():
    decisions, errors = validate_index()
    assert errors == []
    body = """
- Primary ADRL ID: `ADRL-___-___`
- Secondary ADRL IDs: `ADRL-RTG-999`
- Decision effect: `implements | changes`
- Maturity transition: `D_ -> D_ | none`
- Evidence produced:
- Readiness contract/delta: `none | VERSION HASH: OLD -> NEW`
- Architectural non-goals:
"""

    pr_errors = validate_pr_body(body, {row["id"] for row in decisions})

    assert len(pr_errors) == 7
    assert any("primary ADRL ID" in error for error in pr_errors)
    assert any("secondary ADRL ID" in error for error in pr_errors)
