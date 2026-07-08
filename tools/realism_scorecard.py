"""Realism scorecard — recompute one metric battery on the real corpus and on
simulator-labelled captures, then emit a PASS/FAIL table across the four
dimensions (surface, behavioral, structural, failure) plus JOINT checks.

The whole point (per the realism critique): a simulator that is *marginally*
right (correct univariate percentiles) but *jointly* wrong (e.g. applies
lowercase independently of turn length, manufacturing 250-word all-lowercase
verbless fragments) must still FAIL. So the battery includes joint /
conditional metrics, not only marginal percentiles.

Design
------
* Corpus turns come straight from ``data/turns.parquet`` (already featurised by
  the miner — the exact same fields the live discriminator sees).
* Sim turns are reconstructed from ``data/sim-ledger.jsonl`` (the labelled
  skeleton: session_id + driver prompt + episode structure — the provenance
  invariant guarantees session_id is logged) joined to ``data/captures/**`` by
  ``X-Claude-Code-Session-Id`` for the behavioural/failure/structural signals
  that only exist in the realised API traffic.
* ``compute_metrics`` runs on an arbitrary list of ``TurnRec`` so corpus and sim
  are measured by *identical* code.
* If no sim captures/ledger exist yet, every sim cell reads ``no data`` and the
  tool still emits a clean report (so it works before the first sim batch).

CLI
---
    PYTHONPATH=src .venv/bin/python tools/realism_scorecard.py \
        --corpus data/turns.parquet \
        --ledger data/sim-ledger.jsonl \
        --captures data/captures \
        --out reports/realism-scorecard.md [--seed 0] [--json]

Deterministic: no sampling. ``--seed`` is accepted and recorded for
reproducibility bookkeeping but the battery itself is a pure function of inputs.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# --------------------------------------------------------------------------- #
# Canonical per-turn record — the single shape every metric is computed over.
# --------------------------------------------------------------------------- #


@dataclass
class TurnRec:
    session_id: str
    idx: int  # 0-based order of this turn within its session
    text: str  # instruction / driver prompt text
    is_user: bool  # genuine human/driver instruction turn (label == user_turn)
    source_kind: str = "main"  # main | subagent (structural)
    n_continuations: int = 0
    n_assistant_msgs: int = 0
    n_tool_uses: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    had_errors: bool = False
    n_error_results: int = 0
    edit_tripwire: bool = False
    interrupted: bool = False
    secret_shaped: bool = False  # session-level; set on any turn of the session

    @property
    def ctx_per_msg(self) -> float:
        n = max(1, self.n_assistant_msgs)
        return (self.cache_read_tokens + self.input_tokens) / n


# --------------------------------------------------------------------------- #
# Surface-feature primitives (deterministic, de-idiolected).
# --------------------------------------------------------------------------- #

_TERMINAL_PUNCT = (".", "!", "?", "…")

# Verb pool for the verbless-fragment test (broad, lowercased, apostrophe-free).
_VERBS = set(
    """is are was were be been being am has have had do does did add adds added
    fix fixes fixed change changed make made makes run runs ran running create
    created write wrote update updated remove removed delete deleted move moved
    rename renamed refactor refactored check checked review reviewed use used
    build built set sets get gets got put show shows showed tell give gave
    explain explained implement implemented test tested push pushed pushes commit
    committed merge merged revert reverted can could should would will lets let
    need needs want wants try tried trying see seeing look looking go going does
    open close start stop keep help handle wrap ensure verify confirm skip drop
    call calls install pull rebase deploy generate parse print return send read
    load save split join replace apply revert refactor rework wire hook mock""".split()
)

# Voice-dictation disfluency markers (standalone tokens only).
_DISFLUENCY = {"uh", "um", "umm", "uhh", "erm", "hmm", "hm", "eh", "err", "uhm"}

# Broad public txt-speak / netspeak pool (NOT one user's idiolect).
_TXTSPEAK = {
    "u", "ur", "r", "y", "k", "kk", "thx", "plz", "pls", "pln", "btw", "imo",
    "imho", "idk", "tbh", "asap", "rn", "ppl", "msg", "cuz", "cus", "coz",
    "gonna", "wanna", "dunno", "lemme", "gimme", "kinda", "sorta", "outta",
    "gotta", "hafta", "yea", "yeah", "yep", "nope", "nvm", " smth", "smth",
    "b4", "l8r", "w/", "w/o", "abt", "def", "prob", "prolly", "obv", "tho",
    "afaik", "fyi", "wtf", "omg", "lol", "ok", "okay",
}

# Contractions written without the apostrophe — a common typo/txt class.
_APOS_MISSING = {
    "dont", "cant", "wont", "isnt", "arent", "wasnt", "werent", "didnt",
    "doesnt", "couldnt", "shouldnt", "wouldnt", "hasnt", "havent", "hadnt",
    "im", "ive", "id", "ill", "youre", "youve", "youd", "theyre", "theyve",
    "thats", "whats", "lets", "its", "wont", "aint", "gimme",
}

_PATH_RE = re.compile(r"(/[\w.\-]+){2,}")  # >=2 slash-separated path segments
_ABS_PATH_RE = re.compile(r"(/Users/|/home/|~/|[A-Za-z]:\\)")
_FENCE_RE = re.compile(r"```")
_EMOJI_RE = re.compile(
    "[" "\U0001f300-\U0001faff" "\U00002600-\U000027bf" "\U0001f000-\U0001f0ff" "]"
)
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _norm_tokens(text: str) -> list[str]:
    return [w.strip(".,!?:;()[]\"'").lower() for w in text.split()]


def word_count(text: str) -> int:
    return len(text.split())


def all_lowercase(text: str) -> bool:
    return any(c.isalpha() for c in text) and text == text.lower()


def no_terminal_punct(text: str) -> bool:
    t = text.rstrip()
    return bool(t) and not t.endswith(_TERMINAL_PUNCT)


def is_question(text: str) -> bool:
    if "?" in text:
        return True
    first = (_norm_tokens(text)[:1] or [""])[0]
    return first in {"what", "why", "how", "when", "where", "who", "which",
                     "can", "could", "should", "would", "is", "are", "do",
                     "does", "did", "will", "any", "anything"}


def has_inline_path(text: str) -> bool:
    return bool(_ABS_PATH_RE.search(text)) or bool(_PATH_RE.search(text))


def has_fenced_code(text: str) -> bool:
    return bool(_FENCE_RE.search(text))


def has_emoji(text: str) -> bool:
    return bool(_EMOJI_RE.search(text))


def has_disfluency(text: str) -> bool:
    return any(tok in _DISFLUENCY for tok in _norm_tokens(text))


def is_txt_speak(text: str) -> bool:
    toks = _norm_tokens(text)
    tokset = set(toks)
    if tokset & _TXTSPEAK:
        return True
    if tokset & _APOS_MISSING:
        return True
    # doubled-punct / repeated-char netspeak (e.g. "soooo", "!!!")
    if re.search(r"([a-z])\1{2,}", text.lower()):
        return True
    if "!!" in text or "??" in text:
        return True
    return False


def is_verbless(text: str) -> bool:
    toks = _norm_tokens(text)
    return not any(t in _VERBS for t in toks)


# --------------------------------------------------------------------------- #
# Turn archetype — the primary axis of the JOINT distribution.
# Sampled JOINTLY in the simulator; here we classify each turn into exactly one.
# Precedence is deliberate: strong structural signals win over generic buckets.
# --------------------------------------------------------------------------- #

ARCHETYPES = (
    "disfluent-runon",
    "long-plan-paste",
    "path-paste",
    "terse-poll",
    "approval-nudge",
    "question",
    "coding-ask",
)

_POLL_VOCAB = {"status", "done", "pushed", "ready", "yet", "now", "finished",
               "update", "progress", "eta", "stuck", "working"}
_APPROVAL_VOCAB = {"go", "ahead", "do", "it", "yes", "yep", "sure", "ok", "okay",
                   "proceed", "try", "again", "continue", "ship", "send", "approved"}


def archetype_of(text: str) -> str:
    wc = word_count(text)
    if has_disfluency(text) and wc >= 8:
        return "disfluent-runon"
    if wc >= 50:
        return "long-plan-paste"
    if has_inline_path(text):
        return "path-paste"
    toks = set(_norm_tokens(text))
    if wc <= 3:
        if toks & _POLL_VOCAB or is_question(text):
            return "terse-poll"
        if toks & _APPROVAL_VOCAB:
            return "approval-nudge"
        return "terse-poll"
    if wc <= 6 and (toks & _APPROVAL_VOCAB) and not is_question(text):
        return "approval-nudge"
    if is_question(text):
        return "question"
    return "coding-ask"


# --------------------------------------------------------------------------- #
# Small stats helpers (no numpy dependency).
# --------------------------------------------------------------------------- #


def percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(s[int(k)])
    return float(s[lo] * (hi - k) + s[hi] * (k - lo))


def median(values: list[float]) -> Optional[float]:
    return percentile(values, 50)


def pct(num: int, den: int) -> Optional[float]:
    if den == 0:
        return None
    return 100.0 * num / den


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


_CONTENT_STOP = set(
    "the a an and or of to in on for is are be it this that with as at by from "
    "i you we they he she can could should would will do does did please".split()
)


def content_words(text: str) -> set[str]:
    return {t for t in _norm_tokens(text) if t and t not in _CONTENT_STOP and len(t) > 1}


# --------------------------------------------------------------------------- #
# THE metric battery. Runs identically on corpus and sim turn lists.
# Returns a flat dict metric-name -> value (float|None), plus '_aux::' keys
# carrying the archetype structures the reference-requiring joint metrics need.
# --------------------------------------------------------------------------- #


def compute_metrics(turns: list[TurnRec]) -> dict[str, Any]:
    m: dict[str, Any] = {}
    users = [t for t in turns if t.is_user and t.text and t.text.strip()]
    nu = len(users)

    # ---- SURFACE (user stream) ------------------------------------------- #
    wcs = [word_count(t.text) for t in users]
    m["surface::wc_p10"] = percentile([float(x) for x in wcs], 10)
    m["surface::wc_p50"] = percentile([float(x) for x in wcs], 50)
    m["surface::wc_p90"] = percentile([float(x) for x in wcs], 90)
    m["surface::wc_p99"] = percentile([float(x) for x in wcs], 99)
    m["surface::pct_lt5_words"] = pct(sum(1 for w in wcs if w < 5), nu)
    m["surface::pct_no_terminal_punct"] = pct(sum(no_terminal_punct(t.text) for t in users), nu)
    m["surface::pct_all_lowercase"] = pct(sum(all_lowercase(t.text) for t in users), nu)
    m["surface::pct_txt_speak"] = pct(sum(is_txt_speak(t.text) for t in users), nu)
    m["surface::pct_inline_path"] = pct(sum(has_inline_path(t.text) for t in users), nu)
    m["surface::pct_fenced_code"] = pct(sum(has_fenced_code(t.text) for t in users), nu)
    m["surface::pct_emoji"] = pct(sum(has_emoji(t.text) for t in users), nu)

    # ---- BEHAVIORAL (user stream + session structure) -------------------- #
    m["behavioral::pct_terse_le3"] = pct(sum(1 for w in wcs if w <= 3), nu)
    m["behavioral::pct_questions"] = pct(sum(is_question(t.text) for t in users), nu)
    m["behavioral::pct_disfluency"] = pct(sum(has_disfluency(t.text) for t in users), nu)
    # interrupts surface as their own user-stream turns ("[Request interrupted...")
    m["behavioral::interrupt_rate"] = pct(sum(1 for t in users if t.interrupted), nu)

    # session structure
    by_session: dict[str, list[TurnRec]] = {}
    for t in turns:
        by_session.setdefault(t.session_id, []).append(t)
    user_by_session: dict[str, list[TurnRec]] = {}
    for t in users:
        user_by_session.setdefault(t.session_id, []).append(t)

    session_lengths = [len(v) for v in user_by_session.values()]
    m["behavioral::session_len_p50"] = percentile([float(x) for x in session_lengths], 50)
    m["behavioral::session_len_p90"] = percentile([float(x) for x in session_lengths], 90)

    # continuations/turn over user-initiated turns (episode depth)
    conts = [t.n_continuations for t in users]
    m["behavioral::continuations_mean"] = (sum(conts) / len(conts)) if conts else None

    # consecutive-turn topic drift (median Jaccard of content words), within session
    jac: list[float] = []
    for sid, seq in user_by_session.items():
        seq = sorted(seq, key=lambda t: t.idx)
        for a, b in zip(seq, seq[1:]):
            jac.append(jaccard(content_words(a.text), content_words(b.text)))
    m["behavioral::median_consec_jaccard"] = median(jac)

    # ---- STRUCTURAL (all turns) ------------------------------------------ #
    nt = len(turns)
    m["structural::pct_subagent"] = pct(sum(1 for t in turns if t.source_kind == "subagent"), nt)
    ctx_turns = [t for t in turns if t.n_assistant_msgs > 0 or t.cache_read_tokens or t.input_tokens]
    nct = len(ctx_turns)
    m["structural::pct_ctx_over_32k"] = pct(sum(1 for t in ctx_turns if t.ctx_per_msg > 32_000), nct)
    m["structural::pct_ctx_over_100k"] = pct(sum(1 for t in ctx_turns if t.ctx_per_msg > 100_000), nct)
    secret_sessions = {t.session_id for t in turns if t.secret_shaped}
    m["structural::pct_sessions_secret_shaped"] = pct(len(secret_sessions), len(by_session))

    # ---- FAILURE (tool-using turns) -------------------------------------- #
    tool_turns = [t for t in turns if t.n_tool_uses > 0]
    ntt = len(tool_turns)
    m["failure::pct_ge1_error"] = pct(sum(1 for t in tool_turns if t.n_error_results >= 1), ntt)
    m["failure::pct_ge2_error"] = pct(sum(1 for t in tool_turns if t.n_error_results >= 2), ntt)
    edit_turns = [t for t in tool_turns if t.n_error_results or t.edit_tripwire or "edit" in t.text.lower()]
    m["failure::edit_tripwire_rate"] = pct(sum(1 for t in tool_turns if t.edit_tripwire), ntt)

    # ---- JOINT / CONDITIONAL (user stream) ------------------------------- #
    # Chimera guards: independent-flag sims fail these even with right marginals.
    long_turns = [t for t in users if word_count(t.text) >= 50]
    m["joint::pct_lowercase_given_long"] = pct(
        sum(all_lowercase(t.text) for t in long_turns), len(long_turns)
    )
    m["joint::pct_long_lowercase_verbless"] = pct(
        sum(
            1
            for t in users
            if word_count(t.text) >= 50 and all_lowercase(t.text) and is_verbless(t.text)
        ),
        nu,
    )
    # P(terse | archetype!=terse) should be ~0: terse turns must be terse-poll/
    # approval, never long-plan. Guards length-conditioned-on-archetype coherence.
    arche = [archetype_of(t.text) for t in users]
    arche_share: dict[str, float] = {}
    arche_wc_median: dict[str, Optional[float]] = {}
    for a in ARCHETYPES:
        members = [t for t, k in zip(users, arche) if k == a]
        arche_share[a] = (len(members) / nu) if nu else 0.0
        arche_wc_median[a] = median([float(word_count(t.text)) for t in members])
    m["_aux::archetype_share"] = arche_share
    m["_aux::archetype_wc_median"] = arche_wc_median
    m["_aux::n_users"] = nu
    m["_aux::n_turns"] = nt
    return m


# --------------------------------------------------------------------------- #
# Bands (from reports/simulator-realism-plan.md acceptance criteria).
# Univariate bands are fixed; joint bands are derived from the corpus at score
# time (corpus is ground truth for joint structure).
# --------------------------------------------------------------------------- #


@dataclass
class Band:
    dim: str
    label: str
    low: Optional[float]
    high: Optional[float]
    gate: str  # which upgrade this row gates (P0-1 etc.)
    note: str = ""
    ref: str = "fixed"  # "fixed" | "corpus" (band derived from corpus value)


# metric-name -> Band (fixed univariate bands per the plan)
FIXED_BANDS: dict[str, Band] = {
    "surface::wc_p50": Band("surface", "word-count p50", 8, 12, "P0-1"),
    "surface::wc_p90": Band("surface", "word-count p90", 58, 86, "P0-1"),
    "surface::pct_lt5_words": Band("surface", "% <5 words", 18, 32, "P0-1"),
    "surface::pct_no_terminal_punct": Band("surface", "% no terminal punct", 55, 75, "P0-1"),
    "surface::pct_all_lowercase": Band("surface", "% all-lowercase", 20, 32, "P0-1"),
    "surface::pct_txt_speak": Band("surface", "% typos/txt-speak", 16, 26, "P0-1"),
    "surface::pct_inline_path": Band("surface", "% inline file path", 8, 20, "P0-1"),
    "surface::pct_fenced_code": Band("surface", "% fenced code", 0, 1, "P0-1"),
    "surface::pct_emoji": Band("surface", "% human emoji", 0, 1, "P0-1"),
    "behavioral::pct_terse_le3": Band("behavioral", "% terse <=3 words", 14, 24, "P0-2"),
    "behavioral::pct_questions": Band("behavioral", "% questions", 28, 42, "P0-2"),
    "behavioral::pct_disfluency": Band("behavioral", "% voice disfluency (uh/um)", 10, 17, "P0-2"),
    "behavioral::interrupt_rate": Band("behavioral", "interrupt rate", 1.5, 4.0, "P0-2"),
    "behavioral::median_consec_jaccard": Band("behavioral", "median consec-turn Jaccard", 0.0, 0.10, "P0-2"),
    "behavioral::session_len_p50": Band("behavioral", "session length p50", 18, 30, "P0-2"),
    "behavioral::session_len_p90": Band("behavioral", "session length p90", 100, 250, "P0-2"),
    "behavioral::continuations_mean": Band("behavioral", "continuations/turn mean", 2, 4, "P0-2"),
    "structural::pct_subagent": Band("structural", "% source_kind=subagent", 50, 100, "P1-7"),
    "structural::pct_ctx_over_32k": Band("structural", "% ctx/msg >32k", 25, 100, "P1-6"),
    "structural::pct_ctx_over_100k": Band("structural", "% ctx/msg >100k", 15, 100, "P1-6"),
    "structural::pct_sessions_secret_shaped": Band(
        "structural", "% sessions secret-shaped", 0.05, 100, "P1-8",
        note="stress generator; >= planted fraction"),
    "failure::pct_ge1_error": Band("failure", ">=1 error / tool-turn (frontier)", 7, 11, "P0-3"),
    "failure::pct_ge2_error": Band("failure", ">=2 errors / tool-turn (frontier)", 1.5, 3.5, "P0-3"),
    "failure::edit_tripwire_rate": Band(
        "failure", "2-strike edit tripwire (local slice)", 2, 100, "P0-3",
        note="stress generator; frontier floor 0.02%, local slice >=2%"),
}

# Joint metrics whose band is derived from the corpus value + a margin.
# (dim, label, gate, margin_pp, floor_high) — band = [0, max(floor_high, corpus+margin)]
JOINT_SPECS = {
    "joint::pct_lowercase_given_long": ("joint", "P(all-lowercase | >=50 words)", "P0-1/2", 8.0, 8.0),
    "joint::pct_long_lowercase_verbless": ("joint", "P(>=50w & lowercase & verbless)", "P0-1/2", 5.0, 5.0),
}

# Joint metrics computed in score() against the corpus reference structures.
JOINT_REF = {
    "joint::archetype_mix_tvd": ("joint", "archetype-mix TV distance", "P0-1/2", 0.0, 0.15),
    "joint::wc_median_dev_by_archetype": ("joint", "max |wc median dev| by archetype (words)", "P0-1/2", 0.0, 12.0),
}


# --------------------------------------------------------------------------- #
# Scoring: compare corpus vs sim metric dicts, apply bands, PASS/FAIL.
# --------------------------------------------------------------------------- #


@dataclass
class Row:
    dim: str
    metric: str
    corpus: Optional[float]
    sim: Optional[float]
    low: Optional[float]
    high: Optional[float]
    gate: str
    verdict: str  # PASS | FAIL | NO DATA
    note: str = ""


def _in_band(v: Optional[float], low: Optional[float], high: Optional[float]) -> bool:
    if v is None:
        return False
    if low is not None and v < low:
        return False
    if high is not None and v > high:
        return False
    return True


def _tvd(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def score(corpus: dict[str, Any], sim: Optional[dict[str, Any]]) -> list[Row]:
    rows: list[Row] = []
    have_sim = sim is not None and sim.get("_aux::n_turns", 0) > 0

    # ---- fixed univariate bands ------------------------------------------ #
    for metric, band in FIXED_BANDS.items():
        c = corpus.get(metric)
        s = sim.get(metric) if have_sim else None
        if not have_sim or s is None:
            verdict = "NO DATA"
        else:
            verdict = "PASS" if _in_band(s, band.low, band.high) else "FAIL"
        rows.append(Row(band.dim, band.label, c, s, band.low, band.high,
                        band.gate, verdict, band.note))

    # ---- joint metrics with corpus-derived [0, corpus+margin] bands ------ #
    for metric, (dim, label, gate, margin, floor_high) in JOINT_SPECS.items():
        c = corpus.get(metric)
        high = max(floor_high, (c or 0.0) + margin)
        s = sim.get(metric) if have_sim else None
        if not have_sim or s is None:
            verdict = "NO DATA"
        else:
            verdict = "PASS" if _in_band(s, 0.0, high) else "FAIL"
        rows.append(Row(dim, label, c, s, 0.0, round(high, 2), gate, verdict,
                        "chimera guard"))

    # ---- joint metrics requiring both corpus and sim reference structures  #
    c_share = corpus.get("_aux::archetype_share", {})
    c_wcmed = corpus.get("_aux::archetype_wc_median", {})
    if have_sim:
        s_share = sim.get("_aux::archetype_share", {})
        s_wcmed = sim.get("_aux::archetype_wc_median", {})
        tvd = _tvd(c_share, s_share)
        devs = [
            abs((s_wcmed.get(a) or 0.0) - (c_wcmed.get(a) or 0.0))
            for a in ARCHETYPES
            if c_wcmed.get(a) is not None and s_wcmed.get(a) is not None
        ]
        wcdev = max(devs) if devs else None
    else:
        tvd = None
        wcdev = None
    for metric, (dim, label, gate, low, high) in JOINT_REF.items():
        if metric.endswith("tvd"):
            s = tvd
            corpus_val = 0.0
        else:
            s = wcdev
            corpus_val = 0.0
        if not have_sim or s is None:
            verdict = "NO DATA"
        else:
            verdict = "PASS" if _in_band(s, low, high) else "FAIL"
        rows.append(Row(dim, label, corpus_val, s, low, high, gate, verdict,
                        "joint distribution"))

    return rows


# --------------------------------------------------------------------------- #
# Corpus loader (parquet -> TurnRec).
# --------------------------------------------------------------------------- #


def load_corpus_turns(parquet_path: Path) -> list[TurnRec]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    n = table.num_rows

    # order turns within session by ts (fallback to file order)
    order = list(range(n))
    ts = cols.get("ts") or [None] * n
    order.sort(key=lambda i: (cols["session_id"][i], ts[i] or "", i))
    session_counter: dict[str, int] = {}

    def col(name: str, default: Any = None) -> list[Any]:
        return cols.get(name, [default] * n)

    label = col("label", "")
    source_kind = col("source_kind", "main")
    text = col("instruction_text", "")
    n_cont = col("n_continuations", 0)
    n_amsg = col("n_assistant_msgs", 0)
    n_tool = col("n_tool_uses", 0)
    inp = col("input_tokens", 0)
    cache = col("cache_read_tokens", 0)
    had_err = col("had_errors", False)
    n_err = col("n_error_results", 0)
    tripwire = col("edit_tripwire", False)
    interrupted = col("interrupted", False)

    recs: list[TurnRec] = []
    for i in order:
        sid = cols["session_id"][i] or "unknown"
        idx = session_counter.get(sid, 0)
        session_counter[sid] = idx + 1
        recs.append(
            TurnRec(
                session_id=sid,
                idx=idx,
                text=text[i] or "",
                is_user=(label[i] == "user_turn"),
                source_kind=source_kind[i] or "main",
                n_continuations=int(n_cont[i] or 0),
                n_assistant_msgs=int(n_amsg[i] or 0),
                n_tool_uses=int(n_tool[i] or 0),
                input_tokens=int(inp[i] or 0),
                cache_read_tokens=int(cache[i] or 0),
                had_errors=bool(had_err[i]),
                n_error_results=int(n_err[i] or 0),
                edit_tripwire=bool(tripwire[i]),
                interrupted=bool(interrupted[i]),
            )
        )
    return recs


# --------------------------------------------------------------------------- #
# Sim loader (ledger + captures -> TurnRec).
#
# The ledger is the labelled skeleton and the stable contract (session_id +
# driver prompt are provenance invariants). Captures enrich the turns that join
# by X-Claude-Code-Session-Id. Everything is defensive: unknown/absent fields
# degrade to None-yielding metrics rather than crashing.
# --------------------------------------------------------------------------- #

_PROMPT_KEYS = ("prompt", "instruction_text", "instruction", "text", "message")


def _prompt_text(obj: dict[str, Any]) -> Optional[str]:
    for k in _PROMPT_KEYS:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _iter_ledger_turns(entry: dict[str, Any]) -> Iterable[tuple[str, str, dict]]:
    """Yield (session_id, prompt_text, turn_obj) for each user turn in a ledger
    entry, handling both single-shot and episode (turns[]) shapes."""
    turns = entry.get("turns")
    if isinstance(turns, list) and turns:
        for t in turns:
            if not isinstance(t, dict):
                continue
            sid = t.get("session_id") or entry.get("session_id")
            txt = _prompt_text(t) or _prompt_text(entry)
            if sid and txt:
                yield str(sid), txt, t
        return
    sid = entry.get("session_id")
    if not sid:
        sids = entry.get("session_ids")
        if isinstance(sids, list) and sids:
            sid = sids[0]
    txt = _prompt_text(entry)
    if sid and txt:
        yield str(sid), txt, entry


def _index_captures(captures_dir: Path) -> dict[str, list[Path]]:
    idx: dict[str, list[Path]] = {}
    if not captures_dir or not captures_dir.is_dir():
        return idx
    for f in glob.glob(str(captures_dir / "**" / "*.json"), recursive=True):
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        sid = (d.get("request_headers") or {}).get("X-Claude-Code-Session-Id")
        if sid:
            idx.setdefault(sid, []).append(Path(f))
    return idx


def _enrich_from_captures(sid: str, files: list[Path]) -> dict[str, Any]:
    """Best-effort per-session signals from raw proxy captures. Returns counts;
    empty dict if nothing parseable. Never raises."""
    tool_uses = 0
    error_results = 0
    subagent_hits = 0
    max_ctx = 0
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        rb = d.get("response_body")
        if isinstance(rb, str):
            tool_uses += rb.count('"type":"tool_use"')
        reqb = d.get("request_body")
        if isinstance(reqb, str):
            error_results += reqb.count('"is_error":true')
            # Task/subagent spawn signature in the tool surface
            if '"name":"Task"' in reqb or '"isSidechain":true' in reqb:
                subagent_hits += 1
    out: dict[str, Any] = {}
    if tool_uses or error_results or subagent_hits:
        out = {
            "tool_uses": tool_uses,
            "error_results": error_results,
            "subagent_hits": subagent_hits,
            "calls": len(files),
        }
    return out


def load_sim_turns(ledger_path: Path, captures_dir: Optional[Path]) -> list[TurnRec]:
    if not ledger_path or not ledger_path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    with open(ledger_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    cap_idx = _index_captures(captures_dir) if captures_dir else {}
    session_counter: dict[str, int] = {}
    recs: list[TurnRec] = []
    # secret-shaped provenance if the ledger flags it
    secret_sessions: set[str] = set()

    for entry in entries:
        for sid, txt, tobj in _iter_ledger_turns(entry):
            idx = session_counter.get(sid, 0)
            session_counter[sid] = idx + 1
            enrich = _enrich_from_captures(sid, cap_idx[sid]) if sid in cap_idx else {}
            n_tool = int(enrich.get("tool_uses", 0))
            n_err = int(enrich.get("error_results", 0))
            src = "subagent" if enrich.get("subagent_hits", 0) else "main"
            interrupted = bool(tobj.get("interrupted")) or txt.startswith("[Request interrupted by user")
            secret = bool(tobj.get("secret_shaped") or entry.get("secret_shaped"))
            if secret:
                secret_sessions.add(sid)
            recs.append(
                TurnRec(
                    session_id=sid,
                    idx=idx,
                    text=txt,
                    is_user=True,
                    source_kind=src,
                    n_continuations=int(tobj.get("num_turns") or entry.get("num_turns") or 0),
                    n_assistant_msgs=int(enrich.get("calls", 0)),
                    n_tool_uses=n_tool,
                    n_error_results=n_err,
                    had_errors=n_err > 0,
                    edit_tripwire=bool(tobj.get("edit_tripwire")),
                    interrupted=interrupted,
                    secret_shaped=secret,
                )
            )
    for r in recs:
        if r.session_id in secret_sessions:
            r.secret_shaped = True
    return recs


# --------------------------------------------------------------------------- #
# Markdown rendering.
# --------------------------------------------------------------------------- #


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "—"
    if abs(v) < 1 and v != 0:
        return f"{v:.3f}"
    if v == int(v):
        return str(int(v))
    return f"{v:.1f}"


def _band_str(low: Optional[float], high: Optional[float]) -> str:
    lo = "" if low is None else _fmt(low)
    hi = "" if high is None else _fmt(high)
    if low is not None and high is not None:
        return f"[{lo}, {hi}]"
    if high is not None:
        return f"<= {hi}"
    if low is not None:
        return f">= {lo}"
    return "—"


def render_markdown(rows: list[Row], meta: dict[str, Any]) -> str:
    have_sim = meta.get("have_sim", False)
    graded = [r for r in rows if r.verdict in ("PASS", "FAIL")]
    n_pass = sum(1 for r in graded if r.verdict == "PASS")
    n_graded = len(graded)
    lines: list[str] = []
    lines.append("# Realism scorecard — simulator vs corpus")
    lines.append("")
    lines.append("Recomputes one metric battery on the real corpus and on "
                 "simulator-labelled captures. Univariate bands come from the "
                 "acceptance criteria in `reports/simulator-realism-plan.md`; "
                 "**joint** bands are derived from the corpus (ground truth for "
                 "joint structure) so a sim that is marginally right but jointly "
                 "wrong still FAILS.")
    lines.append("")
    lines.append(f"- corpus turns: **{meta.get('corpus_turns', 0)}** "
                 f"(user-turn stream: {meta.get('corpus_users', 0)})")
    if have_sim:
        lines.append(f"- sim turns: **{meta.get('sim_turns', 0)}** "
                     f"(user-turn stream: {meta.get('sim_users', 0)}) "
                     f"from `{meta.get('ledger')}` joined to captures")
        lines.append(f"- **aggregate realism score: {n_pass}/{n_graded} rows in band** "
                     f"({pct(n_pass, n_graded):.0f}%)" if n_graded else
                     "- aggregate realism score: n/a")
        p0 = [r for r in graded if r.gate.startswith("P0")]
        p0_pass = all(r.verdict == "PASS" for r in p0)
        lines.append(f"- **P0 gate ({sum(1 for r in p0 if r.verdict=='PASS')}/{len(p0)} P0 rows PASS): "
                     f"{'REALISTIC' if p0_pass and p0 else 'NOT YET REALISTIC'}**")
    else:
        lines.append("- sim turns: **no sim data** "
                     "(ledger empty/missing or no session join) — corpus column "
                     "and bands shown; sim + verdict columns will populate after "
                     "the first sim batch")
    lines.append(f"- seed: {meta.get('seed')}  ·  deterministic, re-runnable")
    lines.append("")

    for dim in ("surface", "behavioral", "structural", "failure", "joint"):
        drows = [r for r in rows if r.dim == dim]
        if not drows:
            continue
        lines.append(f"## {dim}")
        lines.append("")
        lines.append("| metric | corpus | sim | band | gate | verdict |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for r in drows:
            badge = {"PASS": "PASS", "FAIL": "**FAIL**", "NO DATA": "no data"}[r.verdict]
            metric = r.metric.replace("|", "\\|")  # keep table columns intact
            note = f" <br>_{r.note.replace('|', chr(92) + '|')}_" if r.note else ""
            lines.append(
                f"| {metric}{note} | {_fmt(r.corpus)} | {_fmt(r.sim)} | "
                f"{_band_str(r.low, r.high)} | {r.gate} | {badge} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #


def build_report(
    corpus_path: Path,
    ledger_path: Optional[Path],
    captures_dir: Optional[Path],
    seed: int = 0,
) -> tuple[list[Row], dict[str, Any]]:
    corpus_turns = load_corpus_turns(corpus_path)
    corpus_metrics = compute_metrics(corpus_turns)
    sim_turns = load_sim_turns(ledger_path, captures_dir) if ledger_path else []
    sim_metrics = compute_metrics(sim_turns) if sim_turns else None
    rows = score(corpus_metrics, sim_metrics)
    meta = {
        "seed": seed,
        "have_sim": bool(sim_turns),
        "corpus_turns": len(corpus_turns),
        "corpus_users": corpus_metrics.get("_aux::n_users", 0),
        "sim_turns": len(sim_turns),
        "sim_users": (sim_metrics or {}).get("_aux::n_users", 0),
        "ledger": str(ledger_path) if ledger_path else None,
    }
    return rows, meta


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("data/turns.parquet"))
    ap.add_argument("--ledger", type=Path, default=Path("data/sim-ledger.jsonl"))
    ap.add_argument("--captures", type=Path, default=Path("data/captures"))
    ap.add_argument("--out", type=Path, default=Path("reports/realism-scorecard.md"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="also print rows as JSON to stdout")
    args = ap.parse_args(argv)

    if not args.corpus.is_file():
        print(f"corpus parquet not found: {args.corpus}")
        return 1

    ledger = args.ledger if args.ledger and args.ledger.is_file() else None
    captures = args.captures if args.captures and args.captures.is_dir() else None
    rows, meta = build_report(args.corpus, ledger, captures, seed=args.seed)

    md = render_markdown(rows, meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(f"wrote {args.out}  ({meta['corpus_turns']} corpus turns, "
          f"{'no sim data' if not meta['have_sim'] else str(meta['sim_turns']) + ' sim turns'})")

    if args.json:
        print(json.dumps([r.__dict__ for r in rows], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
