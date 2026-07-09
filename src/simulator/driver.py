"""The user-driver: a noisy developer voice between episode turns.

Reads the agent's last answer plus the step's scripted intent and produces the
next user message. The driver's own LLM calls go DIRECT to the API (never
through the capture proxy) -- the puppeteer must not appear in the captures.

Realism model (see reports/simulator-realism-plan.md P0-1):
surface style is sampled JOINTLY per turn-archetype, never as independent
flags. Independent flags manufacture chimeras (a 250-word all-lowercase
verbless fragment); archetypes carry a *correlated* style profile so that the
aggregate marginals (word-count p50/p90, no-terminal-punct %, all-lowercase %,
typo %, disfluency %, question %) emerge from a coherent per-turn draw.

Archetypes:
  terse-poll     'status', 'is it done'          templated, 1-3 words
  approval-nudge 'go ahead', 'do it'             templated, 1-4 words
  question       'why is X never ...?'           LLM-phrased, ends '?'
  path-paste     '<path>  fix the strip bug'     LLM-phrased + inline path
  coding-ask     'add a --limit flag with a test' LLM-phrased, the normal ask
  disfluent-runon voice-dictation with uh/um     LLM-phrased + disfluency
  long-plan-paste a pasted multi-para plan        LLM-phrased, long, NOT a
                                                   lowercase fragment

Ground-truth markers the matchers depend on (COMPLETION_MARKER) are enforced
mechanically after generation, not left to the driver's discretion.

Typos are DE-IDIOLECTED: parameterized by class and rate from a broad public
pool (common misspellings, generic txt-speak, char-level slips) -- never one
user's personal habits.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

DRIVER_MODEL = "sonnet"  # the fake user: a stronger base voice before roughening
                         # (one short message/turn, so the cost stays small)
COMPLETION_MARKER = re.compile(
    r"\b(that'?s done|great,? now|perfect,? now|ok(ay)? now|nice,? now)\b", re.IGNORECASE
)

# An LLM callable: (prompt, target_words, rng) -> (text, cost_usd).
# The real one shells out to `claude`; tests inject a stub so no cost / no net.
LLMFn = Callable[[str, int, "object"], "tuple[str, float]"]


# ── de-idiolected typo pools (broad/public, not one user's habits) ────────────
_MISSPELLINGS = {
    "the": "teh", "because": "becuase", "separate": "seperate",
    "definitely": "definately", "receive": "recieve", "which": "whcih",
    "function": "funciton", "should": "shoud", "really": "realy",
    "probably": "probaly", "tomorrow": "tommorow", "occurred": "occured",
    "until": "untill", "doesnt": "doesnt", "there": "ther", "their": "thier",
    "then": "hten", "with": "wiht", "just": "jsut", "make": "amke",
    "does": "deos", "want": "wnat", "need": "ned", "when": "wehn",
}
_TXT_SPEAK = {
    "you": "u", "your": "ur", "youre": "ur", "are": "r", "to": "2",
    "for": "4", "please": "plz", "thanks": "thx", "because": "bc",
    "though": "tho", "with": "w", "about": "abt", "and": "n",
    "before": "b4", "people": "ppl", "okay": "k", "yes": "ya",
}
_DISFLUENCY = ["uh", "um", "er", "like", "i mean", "you know"]


@dataclass
class TurnSurface:
    """A generated user turn plus the surface labels used for calibration.

    The runner only needs `text` + `cost_usd`; the label fields exist so the
    test battery can measure the marginals and run the joint anti-chimera
    checks against the archetype that produced each turn.
    """

    text: str
    archetype: str
    target_words: int
    is_lowercase: bool
    has_terminal_punct: bool
    is_question: bool
    has_path: bool
    is_disfluent: bool
    has_typos: bool
    is_fragment: bool
    cost_usd: float

    @property
    def word_count(self) -> int:
        return len(self.text.split())


# ── archetype specification ───────────────────────────────────────────────────
# Each archetype samples its length + style *jointly*. Weights are chosen so
# the aggregate surface marginals match the corpus (see the calibration test).
def _tri(rng, lo, hi, mode):
    return max(1, int(round(rng.triangular(lo, hi, mode))))


def _long_plan_words(rng):
    # heavy right tail: mostly 45-120, occasional paste up to ~300
    if rng.random() < 0.15:
        return _tri(rng, 150, 300, 200)
    return _tri(rng, 45, 130, 70)


ARCHETYPES: dict[str, dict] = {
    "terse-poll": {
        "weight": 0.14, "templated": True, "fragment": True,
        "words": lambda rng: 2,
        "p_question": 0.50, "p_terminal_punct": 0.0, "p_cap": 0.62, "p_typo": 0.25,
    },
    "approval-nudge": {
        "weight": 0.10, "templated": True, "fragment": True,
        "words": lambda rng: 2,
        "p_question": 0.0, "p_terminal_punct": 0.0, "p_cap": 0.62, "p_typo": 0.12,
    },
    "question": {
        "weight": 0.18, "templated": False, "fragment": False,
        "words": lambda rng: _tri(rng, 3, 14, 7),
        "p_question": 1.0, "p_terminal_punct": 0.45, "p_cap": 0.70, "p_typo": 0.26,
    },
    "coding-ask": {
        "weight": 0.21, "templated": False, "fragment": False,
        "words": lambda rng: _tri(rng, 6, 24, 12),
        "p_question": 0.15, "p_terminal_punct": 0.30, "p_cap": 0.82, "p_typo": 0.28,
    },
    "path-paste": {
        "weight": 0.12, "templated": False, "fragment": True,
        "words": lambda rng: _tri(rng, 3, 16, 7),
        "p_question": 0.15, "p_terminal_punct": 0.15, "p_cap": 0.68, "p_typo": 0.20,
    },
    "disfluent-runon": {
        "weight": 0.12, "templated": False, "fragment": False,
        "words": lambda rng: _tri(rng, 14, 45, 26),
        "p_question": 0.35, "p_terminal_punct": 0.30, "p_cap": 0.65, "p_typo": 0.18,
    },
    "long-plan-paste": {
        "weight": 0.13, "templated": False, "fragment": False,
        "words": _long_plan_words,
        "p_question": 0.10, "p_terminal_punct": 0.80, "p_cap": 1.0, "p_typo": 0.10,
    },
}
PROSE_ARCHETYPES = [k for k, v in ARCHETYPES.items() if not v["templated"]]

# Templated phrase banks (stored WITHOUT terminal punctuation; the sampler adds
# it, or not, per the archetype's p_terminal_punct).
_TERSE_BANK = [
    "status", "status now", "is it done", "done yet", "any update",
    "pushed", "hows it going", "whats the status", "did it work", "still going",
    "you there", "wdyt", "and now", "results", "how far",
]
_APPROVAL_BANK = [
    "go ahead", "do it", "yes do that", "ok proceed", "try now", "lgtm ship it",
    "sounds good", "yep go", "please do so", "yeah go for it", "ok go", "do that",
    "makes sense proceed", "fine do it",
]
# synthetic fallback paths when the sandbox provides none
_FALLBACK_PATHS = [
    "/Users/dev/projects/shiplog/shiplog/parse.py",
    "/Users/dev/projects/orderflow/src/limiter.py",
    "src/metricd/stats.py",
    "/Users/dev/work/queuepilot/cli.py",
]


def _pick_archetype(rng) -> str:
    keys = list(ARCHETYPES)
    weights = [ARCHETYPES[k]["weight"] for k in keys]
    return rng.choices(keys, weights=weights, k=1)[0]


# ── mechanical roughening ─────────────────────────────────────────────────────
def _apply_typos(text: str, rng) -> tuple[str, bool]:
    """Apply de-idiolected typos by CLASS at a modest per-turn rate.

    Returns (text, changed). A turn marked typo-active gets 1-3 substitutions
    drawn from the misspelling / txt-speak / char-slip classes.
    """
    words = text.split()
    if not words:
        return text, False
    n_sub = rng.randint(1, 3)
    idxs = list(range(len(words)))
    rng.shuffle(idxs)
    changed = False
    for i in idxs:
        if n_sub <= 0:
            break
        raw = words[i]
        low = re.sub(r"[^a-z]", "", raw.lower())
        cls = rng.random()
        new = None
        if cls < 0.40 and low in _MISSPELLINGS:
            new = _MISSPELLINGS[low]
        elif cls < 0.80 and low in _TXT_SPEAK:
            new = _TXT_SPEAK[low]
        elif len(low) >= 4:  # char-level slip: transpose / drop / double
            chars = list(low)
            op = rng.randint(0, 2)
            j = rng.randint(0, len(chars) - 2)
            if op == 0:
                chars[j], chars[j + 1] = chars[j + 1], chars[j]
            elif op == 1:
                del chars[j]
            else:
                chars.insert(j, chars[j])
            new = "".join(chars)
        if new and new != low:
            words[i] = raw.replace(low, new) if low and low in raw else new
            n_sub -= 1
            changed = True
    return " ".join(words), changed


def _inject_disfluency(text: str, rng) -> str:
    """Voice-dictation roughening: filler tokens + a mid-sentence restart."""
    words = text.split()
    if len(words) < 3:
        return " ".join([rng.choice(_DISFLUENCY), *words])
    # sprinkle 1-2 fillers
    for _ in range(rng.randint(1, 2)):
        pos = rng.randint(1, len(words) - 1)
        words.insert(pos, rng.choice(_DISFLUENCY) + ",")
    # a restart somewhere in the first half
    if len(words) > 6 and rng.random() < 0.6:
        pos = rng.randint(2, len(words) // 2)
        words.insert(pos, "-- wait no,")
    return " ".join(words)


def _inject_path(text: str, rng, sandbox_paths: Optional[list[str]]) -> str:
    path = rng.choice(sandbox_paths) if sandbox_paths else rng.choice(_FALLBACK_PATHS)
    if rng.random() < 0.5:
        return f"{path}  {text}"
    return f"{text} {path}"


def _strip_terminal(text: str) -> str:
    return re.sub(r"[.!?]+\s*$", "", text).rstrip()


def _ensure_terminal(text: str, is_question: bool) -> str:
    text = _strip_terminal(text)
    return text + ("?" if is_question else ".")


# ── prose prompt for the real LLM ─────────────────────────────────────────────
_STYLE_HINT = {
    "question": "phrase it as a direct question",
    "coding-ask": "a concrete coding request",
    "path-paste": "reference a specific file, terse",
    "disfluent-runon": "rambling, as if dictated out loud, one run-on",
    "long-plan-paste": "a detailed multi-paragraph plan or spec you are pasting in",
}


def _phrase_prompt(intent, last_answer, target_words, archetype) -> str:
    return (
        "You are role-playing a busy software developer typing into a coding "
        "agent's terminal. The agent just replied (truncated):\n---\n"
        f"{last_answer[:1500]}\n---\n"
        f"Your next move: {intent}\n\n"
        f"Write ONLY the message you would type, about {target_words} words, "
        f"{_STYLE_HINT.get(archetype, 'casual developer voice')}. No greetings, "
        "no quotes, plain text, do not mention these instructions."
    )


def _real_llm(prompt: str, target_words: int, rng) -> tuple[str, float]:
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_BASE_URL"}  # uncaptured
    try:
        proc = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "json",
             "--model", DRIVER_MODEL, "--max-turns", "1"],
            capture_output=True, text=True, timeout=120, env=env,
        )
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        text = str(result.get("result", "")).strip().strip('"')
        cost = result.get("total_cost_usd") or result.get("cost_usd") or 0.0
        return text, float(cost or 0.0)
    except (json.JSONDecodeError, IndexError, subprocess.SubprocessError, OSError):
        return "", 0.0


# ── the generator ─────────────────────────────────────────────────────────────
def generate_turn(
    intent: str,
    last_answer: str,
    *,
    rng,
    llm: Optional[LLMFn] = None,
    archetype: Optional[str] = None,
    require_completion_marker: bool = False,
    sandbox_paths: Optional[list[str]] = None,
) -> TurnSurface:
    """Sample an archetype, draw its correlated style, render, roughen.

    `rng` is a random.Random (determinism). `llm` renders prose archetypes;
    inject a stub in tests so there is no cost and no network. Terse/approval
    archetypes are templated (no LLM round-trip).
    """
    llm = llm or _real_llm
    arch = archetype or _pick_archetype(rng)
    spec = ARCHETYPES[arch]

    target_words = spec["words"](rng)
    is_question = rng.random() < spec["p_question"]
    cost = 0.0

    if spec["templated"]:
        bank = _TERSE_BANK if arch == "terse-poll" else _APPROVAL_BANK
        text = rng.choice(bank)
    else:
        prompt = _phrase_prompt(intent, last_answer, target_words, arch)
        text, cost = llm(prompt, target_words, rng)
        text = (text or "").strip()
        if not text:  # LLM failed -> fall back to the scripted intent verbatim
            text = intent

    has_path = False
    if arch == "path-paste":
        text = _inject_path(text, rng, sandbox_paths)
        has_path = True

    is_disfluent = False
    if arch == "disfluent-runon":
        text = _inject_disfluency(text, rng)
        is_disfluent = True

    # typos (per-turn Bernoulli by archetype rate, then 1-3 substitutions)
    has_typos = False
    if rng.random() < spec["p_typo"]:
        text, has_typos = _apply_typos(text, rng)

    # casing: capitalize (proper-ish) or force all-lowercase
    apply_cap = rng.random() < spec["p_cap"]
    if apply_cap:
        text = text[:1].upper() + text[1:] if text else text
    else:
        text = text.lower()

    # terminal punctuation
    keep_punct = rng.random() < spec["p_terminal_punct"]
    if keep_punct or is_question:
        # questions keep '?' only when they'd keep punctuation at all
        if keep_punct:
            text = _ensure_terminal(text, is_question)
        else:
            text = _strip_terminal(text)
    else:
        text = _strip_terminal(text)

    # ground-truth marker enforcement (gradability) -- last, so it survives
    if require_completion_marker and not COMPLETION_MARKER.search(text):
        text = ("great, that's done. " + text).strip()

    final_has_punct = bool(re.search(r"[.!?]\s*$", text))
    final_is_lower = text == text.lower() and any(c.isalpha() for c in text)

    return TurnSurface(
        text=text,
        archetype=arch,
        target_words=target_words,
        is_lowercase=final_is_lower,
        has_terminal_punct=final_has_punct,
        is_question=is_question,
        has_path=has_path,
        is_disfluent=is_disfluent,
        has_typos=has_typos,
        is_fragment=bool(spec["fragment"]),
        cost_usd=cost,
    )


def next_message(
    intent: str,
    last_answer: str,
    require_completion_marker: bool = False,
    *,
    rng=None,
    llm: Optional[LLMFn] = None,
    archetype: Optional[str] = None,
    sandbox_paths: Optional[list[str]] = None,
) -> tuple[str, float]:
    """Back-compat entry point for the runner: returns (message, driver_cost_usd).

    Delegates to generate_turn. When `rng` is None a fresh Random() is used
    (non-deterministic); the runner passes a seeded rng for reproducibility.
    """
    import random as _random

    turn = generate_turn(
        intent, last_answer,
        rng=rng or _random.Random(),
        llm=llm, archetype=archetype,
        require_completion_marker=require_completion_marker,
        sandbox_paths=sandbox_paths,
    )
    return turn.text, turn.cost_usd
