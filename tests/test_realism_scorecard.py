"""Tests for tools/realism_scorecard.py.

The load-bearing assertions:

  * a sim whose turn distribution MATCHES the corpus scores all-PASS (excluding
    the two explicit STRESS-GENERATOR rows — the local-slice edit tripwire and
    planted secrets — which the frontier corpus is a lower bound for, per the
    realism critique point 4);
  * a sim that is MARGINALLY right (lowercase share in band) but JOINTLY wrong
    — it lowercases turns independently of length, so long plan-pastes get
    lowercased the way the corpus never does — still FAILs, and fails on the
    joint conditional row while its marginal row passes;
  * the tool runs and reports "no sim data" cleanly when no sim turns exist;
  * the surface primitives, archetype classifier, and band logic behave.

The scoring tests use the real corpus (data/turns.parquet) as the fixture — it
is the committed ground truth, so it is guaranteed to sit inside the acceptance
bands and is the most faithful stand-in for a matching sim. They skip cleanly if
the parquet is absent.

Run:
    PYTHONPATH=src .venv/bin/python -m pytest tests/test_realism_scorecard.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

# Load the tool module directly (it lives in tools/, not on the package path).
_TOOL = Path(__file__).resolve().parent.parent / "tools" / "realism_scorecard.py"
_spec = importlib.util.spec_from_file_location("realism_scorecard", _TOOL)
rs = importlib.util.module_from_spec(_spec)
sys.modules["realism_scorecard"] = rs
_spec.loader.exec_module(rs)

_PARQUET = Path(__file__).resolve().parent.parent / "data" / "turns.parquet"

# 60-word properly-cased plan paste WITH verbs (matches the corpus fact that
# long turns are dictated/pasted prose, never lowercase fragments).
_LONG = (
    "Please refactor the authentication module so that the token refresh logic "
    "lives in its own service, then update every caller to use the new interface "
    "and add unit tests that cover the expired-token path as well as the happy "
    "path, and finally run the full pytest suite to confirm nothing regressed "
    "before you open the pull request for my review later this afternoon thanks"
)
assert len(_LONG.split()) >= 50
_DISFLUENT = (
    "so are you saying that you will uh do the fixes first or will you um start "
    "with the analysis of the failing test because i am not totally sure which"
)


def _corpus() -> list["rs.TurnRec"]:
    if not _PARQUET.is_file():
        pytest.skip("real corpus parquet not present")
    return rs.load_corpus_turns(_PARQUET)


def _chimera(turns: list["rs.TurnRec"]) -> list["rs.TurnRec"]:
    """Marginally-right, jointly-wrong transform: first strip all existing
    lowercase (capitalise every user turn), then RE-lowercase ~25% of user turns
    chosen INDEPENDENTLY of length. The marginal lowercase share stays ~26% (in
    band) but long turns now get lowercased exactly like short ones — the chimera
    the critique warns about."""
    out: list[rs.TurnRec] = []
    for i, t in enumerate(turns):
        text = t.text
        if t.is_user and text:
            text = text[:1].upper() + text[1:]
            if i % 4 == 0:
                text = text.lower()
        out.append(replace(t, text=text))
    return out


# --------------------------------------------------------------------------- #
# Surface / archetype primitive tests (hermetic).
# --------------------------------------------------------------------------- #


def test_surface_primitives():
    assert rs.all_lowercase("fix the bug") is True
    assert rs.all_lowercase("Fix the bug") is False
    assert rs.all_lowercase("12345") is False  # no alpha
    assert rs.no_terminal_punct("status") is True
    assert rs.no_terminal_punct("done.") is False
    assert rs.is_question("why is this failing") is True
    assert rs.is_question("pushed?") is True
    assert rs.is_question("add a test") is False
    assert rs.has_inline_path("see /Users/x/a.py") is True
    assert rs.has_inline_path("no path here") is False
    assert rs.has_disfluency("so uh can you") is True
    assert rs.has_disfluency("under the hut") is False  # 'hut' is not a filler
    assert rs.is_txt_speak("can u plz") is True
    assert rs.is_txt_speak("dont do that") is True
    assert rs.is_txt_speak("please fix the parser") is False
    assert rs.is_verbless("the padded date thing") is True
    assert rs.is_verbless("fix the padded date") is False
    assert rs.has_fenced_code("```py\nx=1\n```") is True
    assert rs.has_emoji("ship it 🚀") is True
    assert rs.has_emoji("ship it") is False


def test_archetype_classification():
    assert rs.archetype_of("status") == "terse-poll"
    assert rs.archetype_of("go ahead") == "approval-nudge"
    assert rs.archetype_of("why is the test failing here") == "question"
    assert rs.archetype_of("check /Users/x/y/server.ts please now") == "path-paste"
    assert rs.archetype_of(_LONG) == "long-plan-paste"
    assert rs.archetype_of(_DISFLUENT) == "disfluent-runon"
    assert rs.archetype_of("add a strip call to parse date") == "coding-ask"


def test_percentile_helper():
    assert rs.percentile([], 50) is None
    assert rs.percentile([5], 50) == 5
    assert rs.percentile([1, 2, 3, 4], 50) == 2.5
    assert rs.percentile([1, 2, 3, 4], 90) == pytest.approx(3.7)


def test_jaccard_and_content_words():
    assert rs.jaccard(set(), set()) == 0.0
    assert rs.jaccard({"a"}, {"a"}) == 1.0
    assert rs.jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    # stopwords dropped, distinct content -> zero overlap
    assert rs.content_words("fix the parser") & rs.content_words("run the tests") == set()


# --------------------------------------------------------------------------- #
# Core scoring tests (real-corpus fixture).
# --------------------------------------------------------------------------- #

# Stress-generator rows: the frontier corpus is an explicit LOWER BOUND for
# these, so they are not realism-calibration targets and are excluded from the
# "matching sim scores all-PASS" assertion.
_STRESS = {
    "2-strike edit tripwire (local slice)",
    "% sessions secret-shaped",
}


def test_corpus_reproduces_plan_targets():
    """Sanity: the battery on the real corpus reproduces the plan's headline
    corpus numbers (this is what makes the corpus a valid in-band fixture)."""
    m = rs.compute_metrics(_corpus())
    assert m["surface::wc_p50"] == pytest.approx(10, abs=1)
    assert m["surface::wc_p90"] == pytest.approx(72, abs=6)
    assert m["surface::pct_no_terminal_punct"] == pytest.approx(64.6, abs=4)
    assert m["surface::pct_all_lowercase"] == pytest.approx(26.1, abs=3)
    assert m["behavioral::pct_questions"] == pytest.approx(34.6, abs=5)
    assert m["structural::pct_subagent"] == pytest.approx(75.1, abs=3)
    # the joint corpus structure is clean: long turns are essentially never
    # all-lowercase — this is the fact the chimera violates.
    assert m["joint::pct_lowercase_given_long"] < 5.0


def test_matching_sim_scores_all_pass():
    corpus = _corpus()
    cm = rs.compute_metrics(corpus)
    rows = rs.score(cm, rs.compute_metrics(list(corpus)))
    graded = [r for r in rows if r.verdict in ("PASS", "FAIL")]
    assert graded, "expected graded rows"
    fails = [r for r in graded if r.verdict == "FAIL" and r.metric not in _STRESS]
    assert not fails, f"identical sim should not FAIL (non-stress): {[(r.metric, r.sim) for r in fails]}"
    # joint rows in particular pass
    joint = [r for r in graded if r.dim == "joint"]
    assert joint and all(r.verdict == "PASS" for r in joint)
    # both excluded rows are indeed the marked stress generators
    stress_rows = [r for r in graded if r.metric in _STRESS]
    assert all("stress generator" in (r.note or "") for r in stress_rows)


def test_chimera_sim_fails_jointly_but_passes_marginals():
    corpus = _corpus()
    cm = rs.compute_metrics(corpus)
    sm = rs.compute_metrics(_chimera(corpus))
    rows = rs.score(cm, sm)
    by_metric = {r.metric: r for r in rows}

    # Marginal lowercase still in band -> the marginal row PASSES.
    marg = by_metric["% all-lowercase"]
    assert marg.verdict == "PASS", (marg.sim,)

    # But the JOINT conditional is blown out: long turns are now lowercased at
    # the marginal rate, which the corpus never does.
    cond = by_metric["P(all-lowercase | >=50 words)"]
    assert cond.sim is not None and cond.sim > cond.high
    assert cond.verdict == "FAIL"

    # Net: the scorecard is not all-PASS despite the marginal looking fine.
    assert any(r.verdict == "FAIL" for r in rows)


def test_scrambled_archetype_mix_fails_joint():
    """A sim that gets the marginals right but destroys the archetype MIX (e.g.
    every turn becomes a long paste) must fail the joint archetype-mix row."""
    corpus = _corpus()
    cm = rs.compute_metrics(corpus)
    # replace every user turn's text with the same long paste -> archetype
    # collapses to one bucket; TV distance from corpus mix is large.
    collapsed = [replace(t, text=_LONG) if t.is_user else t for t in corpus]
    rows = rs.score(cm, rs.compute_metrics(collapsed))
    tvd = next(r for r in rows if r.metric == "archetype-mix TV distance")
    assert tvd.verdict == "FAIL" and tvd.sim > tvd.high


# --------------------------------------------------------------------------- #
# No-data + determinism + rendering.
# --------------------------------------------------------------------------- #


def test_no_sim_data_reports_cleanly():
    corpus = _corpus()
    rows = rs.score(rs.compute_metrics(corpus), None)
    assert rows, "should still emit corpus rows"
    assert all(r.verdict == "NO DATA" for r in rows)
    surf = next(r for r in rows if r.metric == "% all-lowercase")
    assert surf.corpus is not None and surf.low is not None  # corpus + band shown
    md = rs.render_markdown(rows, {"have_sim": False, "corpus_turns": len(corpus),
                                   "corpus_users": 754, "seed": 0})
    assert "no sim data" in md and "# Realism scorecard" in md


def test_empty_turns_no_crash():
    """compute_metrics on [] must not crash; score treats it as no-data."""
    empty = rs.compute_metrics([])
    assert empty["_aux::n_turns"] == 0
    rows = rs.score(rs.compute_metrics(_corpus()),
                    empty if empty.get("_aux::n_turns", 0) > 0 else None)
    assert all(r.verdict == "NO DATA" for r in rows)


def test_load_sim_turns_missing_ledger():
    assert rs.load_sim_turns(Path("/nonexistent/ledger.jsonl"), None) == []


def test_load_sim_turns_from_ledger(tmp_path):
    """Ledger-driven sim reconstruction: single-shot and episode shapes both
    yield user turns keyed by session_id."""
    ledger = tmp_path / "sim-ledger.jsonl"
    ledger.write_text(
        '{"session_id": "aaa", "prompt": "status", "num_turns": 3}\n'
        '{"session_ids": ["bbb"], "turns": ['
        '{"session_id": "bbb", "prompt": "fix the padded date test", "num_turns": 4},'
        '{"session_id": "bbb", "prompt": "great thanks, now add a readme", "num_turns": 2}]}\n'
        "\n"  # blank line tolerated
        "{bad json}\n"  # malformed line skipped
    )
    turns = rs.load_sim_turns(ledger, None)
    assert len(turns) == 3
    assert {t.session_id for t in turns} == {"aaa", "bbb"}
    assert all(t.is_user for t in turns)
    # session ordering preserved within bbb
    bbb = sorted([t for t in turns if t.session_id == "bbb"], key=lambda t: t.idx)
    assert bbb[0].text.startswith("fix") and bbb[1].idx == 1
    # metrics run on the reconstructed sim without crashing
    m = rs.compute_metrics(turns)
    assert m["_aux::n_users"] == 3


def test_deterministic():
    corpus = _corpus()
    a = rs.compute_metrics(corpus)
    b = rs.compute_metrics(rs.load_corpus_turns(_PARQUET))
    for k, v in a.items():
        if k.startswith("_aux"):
            continue
        assert v == b[k], k


def test_render_markdown_shape():
    corpus = _corpus()
    rows = rs.score(rs.compute_metrics(corpus), rs.compute_metrics(list(corpus)))
    md = rs.render_markdown(rows, {"have_sim": True, "corpus_turns": len(corpus),
                                   "corpus_users": 754, "sim_turns": len(corpus),
                                   "sim_users": 754, "seed": 0, "ledger": "x"})
    for dim in ("surface", "behavioral", "structural", "failure", "joint"):
        assert f"## {dim}" in md
    assert "| metric | corpus | sim | band | gate | verdict |" in md
    assert "P0 gate" in md


def test_build_report_end_to_end():
    """build_report against the real parquet (no sim ledger) must not crash."""
    if not _PARQUET.is_file():
        pytest.skip("real corpus parquet not present")
    rows, meta = rs.build_report(_PARQUET, None, None, seed=0)
    assert meta["corpus_turns"] > 0 and meta["have_sim"] is False
    surf = [r for r in rows if r.dim == "surface"]
    assert surf and all(r.corpus is not None for r in surf)
