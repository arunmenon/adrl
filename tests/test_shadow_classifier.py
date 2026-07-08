"""Unit tests for router.shadow_classifier (UNIT B).

These run WITHOUT the sibling llm_classifier and WITHOUT ollama: we monkeypatch
shadow_classifier.classify_intent_llm with a deterministic stub. They verify the
regex-uncertainty logic, the outcome proxy, stratified sampling, and that the
end-to-end run writes a scrubbed report with the expected PASS/FAIL math.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from router import shadow_classifier as sc
from router.policy import T_EASY, T_HARD


@dataclass
class FakeVerdict:
    tier: str
    needs_frontier: bool
    score: float


def _row(text, *, n_edit_failures=0, n_error_results=0, interrupted=False,
         n_continuations=0, cache_read_tokens=0, input_tokens=0, n_assistant_msgs=1):
    return {
        "source_kind": "main", "label": "user_turn", "instruction_text": text,
        "n_edit_failures": n_edit_failures, "n_error_results": n_error_results,
        "interrupted": interrupted, "n_continuations": n_continuations,
        "cache_read_tokens": cache_read_tokens, "input_tokens": input_tokens,
        "n_assistant_msgs": n_assistant_msgs,
    }


# ── intent-only score + uncertainty band ──

def test_intent_only_score_scope_adjustments():
    # "write" base 0.45; broad scope pushes up, narrow pulls down.
    base = sc.intent_only_score("write a parser", 0)
    broad = sc.intent_only_score("write a parser across the entire codebase", 0)
    narrow = sc.intent_only_score("write a parser in just this file", 0)
    assert broad > base > narrow
    assert base == pytest.approx(0.45)


def test_intent_only_score_big_context_nudge():
    assert sc.intent_only_score("write a parser", 0) == pytest.approx(0.45)
    assert sc.intent_only_score("write a parser", 50_000) == pytest.approx(0.55)


def test_regex_uncertain_definition():
    # unknown verb -> uncertain regardless of score
    assert sc.is_regex_uncertain("unknown", 0.5) is True
    # score inside the band -> uncertain
    assert sc.is_regex_uncertain("write", (T_EASY + T_HARD) / 2) is True
    # clear-easy known verb below T_EASY -> certain
    assert sc.is_regex_uncertain("trivial", 0.10) is False
    # clear-hard known verb above T_HARD -> certain
    assert sc.is_regex_uncertain("hard", 0.85) is False


# ── outcome proxy ──

@pytest.mark.parametrize("kwargs,expected", [
    ({}, False),
    ({"n_edit_failures": 1}, True),
    ({"n_error_results": 2}, True),
    ({"interrupted": True}, True),
    ({"n_continuations": 10}, True),
    ({"n_continuations": 9}, False),
])
def test_outcome_proxy_hard(kwargs, expected):
    assert sc.outcome_proxy_hard(_row("x", **kwargs)) is expected


# ── stratified sampling ──

def test_stratified_sample_size_and_determinism():
    pop = []
    for i in range(40):
        pop.append({"_verb_class": "write" if i % 2 else "unknown",
                    "_row": _row("x"), "_score": 0.5})
    a = sc.stratified_sample(pop, 10, seed=1729)
    b = sc.stratified_sample(pop, 10, seed=1729)
    assert len(a) == 10
    assert [id(x) for x in a] == [id(x) for x in b]  # deterministic under seed


def test_stratified_sample_returns_all_when_limit_exceeds_pop():
    pop = [{"_verb_class": "unknown", "_row": _row("x"), "_score": 0.5} for _ in range(5)]
    assert len(sc.stratified_sample(pop, 100, seed=1)) == 5


# ── end-to-end run against a stubbed classifier ──

def test_run_end_to_end_writes_scrubbed_report(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Build a tiny corpus: uncertain turns (unknown verb) with a clean
    # correlation — hard-outcome turns get classified needs_frontier=True.
    rows = []
    secret_text = "PLEASE_DO_NOT_LEAK_THIS_INSTRUCTION"
    for i in range(12):
        hard = i < 6
        rows.append(_row(
            f"{secret_text} number {i}",
            n_error_results=1 if hard else 0,
        ))
    # a couple of clearly-easy turns that the regex resolves (should be excluded)
    rows.append(_row("fix the typo in the readme title"))

    table = pa.Table.from_pylist([
        {**r, "source_kind": "main", "label": "user_turn"} for r in rows
    ])
    turns_path = tmp_path / "turns.parquet"
    pq.write_table(table, turns_path)

    def fake_classify(text, **_):
        # Deterministic: turns whose row carried an error proxy read as hard.
        # We can't see the row here, only text — encode via the index suffix.
        idx = int(text.rsplit(" ", 1)[-1]) if text.rsplit(" ", 1)[-1].isdigit() else 99
        if idx < 6:
            return FakeVerdict(tier="hard", needs_frontier=True, score=0.9)
        return FakeVerdict(tier="standard", needs_frontier=False, score=0.4)

    monkeypatch.setattr(sc, "classify_intent_llm", fake_classify)

    report_path = tmp_path / "classifier-shadow.md"
    rc = sc.run(turns_path, report_path, limit=100, use_all=True, model="stub")
    assert rc == 0
    text = report_path.read_text()

    # report is scrubbed: no raw instruction text leaks
    assert secret_text not in text
    # structural expectations
    assert "Classifier shadow" in text
    assert "needs_frontier=True" in text
    # correlation is clean in this fixture -> PASS on both bars
    assert "**Verdict: PASS**" in text


def test_run_handles_all_none_fallback(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [{**_row(f"ambiguous request {i}"), "source_kind": "main",
             "label": "user_turn"} for i in range(5)]
    turns_path = tmp_path / "turns.parquet"
    pq.write_table(pa.Table.from_pylist(rows), turns_path)

    monkeypatch.setattr(sc, "classify_intent_llm", lambda text, **_: None)
    report_path = tmp_path / "out.md"
    rc = sc.run(turns_path, report_path, limit=100, use_all=True, model="down")
    assert rc == 0
    text = report_path.read_text()
    # 100% fallback -> resolve bar fails, verdict FAIL, no crash on empty buckets
    assert "**Verdict: FAIL**" in text
    assert "Insufficient split" in text
