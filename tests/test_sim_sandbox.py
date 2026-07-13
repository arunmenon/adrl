"""Tests for the project-archetype sandbox registry (Unit B).

Asserts: every archetype builds a real git repo whose dominant-language files
import/parse; the file-extension mix over a weighted batch approximates the
corpus buckets (TS/TSX-dominant); heavy-seeded sandboxes cross the 32k token
ceiling while light ones stay under it; the FAKE-secret stress generator is
clearly labeled; and scenarios stay archetype-appropriate and backward
compatible with Unit A's runner call shape.
"""

from __future__ import annotations

import ast
import json
import random
import subprocess
from collections import Counter

import pytest

from simulator.sandbox import ARCHETYPES, make_sandbox
from simulator.tasks import applicable, pick

CONTRACT_KEYS = {
    "path", "project", "bad_var", "good_var", "entity",   # preserved for Unit A
    "archetype", "language", "context_tokens_estimate",   # new
    "dominant_ext", "test_cmd", "planted_issue", "planted_secret",
    "test_commands",
}


def _is_git_repo(path) -> bool:
    r = subprocess.run(["git", "rev-parse", "--verify", "HEAD"],
                       cwd=path, capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def _balanced(text: str, open_c: str = "{", close_c: str = "}") -> bool:
    depth = 0
    for ch in text:
        if ch == open_c:
            depth += 1
        elif ch == close_c:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _source_files(path):
    return [p for p in path.rglob("*") if p.is_file() and ".git" not in p.parts]


def _validate_parses(path) -> None:
    """Stack-appropriate parse check over the sandbox's source files."""
    for f in _source_files(path):
        text = f.read_text(errors="ignore")
        if f.suffix == ".py":
            ast.parse(text)  # raises SyntaxError on bad python
        elif f.suffix == ".json":
            json.loads(text)  # raises on malformed json
        elif f.suffix in (".ts", ".tsx", ".tf", ".prisma"):
            assert text.strip(), f"empty source file {f}"
            assert _balanced(text, "{", "}"), f"unbalanced braces in {f}"
            assert _balanced(text, "(", ")"), f"unbalanced parens in {f}"


# --------------------------------------------------------------------------- #
def test_registry_has_at_least_six_archetypes():
    assert len(ARCHETYPES) >= 6


@pytest.mark.parametrize("arch", sorted(ARCHETYPES))
def test_each_archetype_builds_real_git_repo(tmp_path, arch):
    rng = random.Random(hash(arch) & 0xFFFF)
    sb = make_sandbox(tmp_path, rng, archetype=arch, heavy=False, plant_secret=False)

    # contract keys all present
    assert CONTRACT_KEYS <= set(sb), f"missing keys: {CONTRACT_KEYS - set(sb)}"
    assert sb["archetype"] == arch
    assert isinstance(sb["project"], str) and sb["project"]
    assert isinstance(sb["bad_var"], str) and sb["bad_var"]
    assert isinstance(sb["good_var"], str) and sb["good_var"]
    assert isinstance(sb["entity"], str) and sb["entity"]
    assert isinstance(sb["context_tokens_estimate"], int)
    assert sb["test_commands"]
    assert all(isinstance(command["argv"], tuple) for command in sb["test_commands"])

    # a real git repo with a real initial commit
    assert (sb["path"] / ".git").is_dir()
    assert _is_git_repo(sb["path"])

    # dominant-language files import / parse
    _validate_parses(sb["path"])

    # a file with the dominant extension actually exists
    exts = {p.suffix for p in _source_files(sb["path"])}
    assert sb["dominant_ext"] in exts, f"{arch}: no {sb['dominant_ext']} file"


@pytest.mark.parametrize(
    "arch",
    sorted(a for a in ARCHETYPES if "rename" in applicable({"archetype": a})),
)
def test_rename_target_is_actually_planted(tmp_path, arch):
    """For archetypes where rename applies, bad_var must appear in the source so
    the rename scenario is gradable."""
    rng = random.Random(1234)
    sb = make_sandbox(tmp_path, rng, archetype=arch, heavy=False, plant_secret=False)
    hits = sum(sb["bad_var"] in p.read_text(errors="ignore") for p in _source_files(sb["path"]))
    assert hits >= 2, f"{arch}: bad_var {sb['bad_var']!r} appears in {hits} files (<2)"


def test_extension_mix_approximates_corpus(tmp_path):
    """Over a weighted batch, TS/TSX dominates and other buckets are present —
    the corpus shape, not the old 100%-Python toy."""
    rng = random.Random(7)
    ext = Counter()
    seen_archetypes = set()
    for i in range(48):
        sb = make_sandbox(tmp_path / f"b{i}", rng, heavy=False, plant_secret=False)
        seen_archetypes.add(sb["archetype"])
        for p in _source_files(sb["path"]):
            if p.suffix:
                ext[p.suffix] += 1

    total = sum(ext.values())
    assert total > 0
    ts_tsx = (ext[".ts"] + ext[".tsx"]) / total
    md = ext[".md"] / total
    sql = ext[".sql"] / total
    py = ext[".py"] / total

    # TS/TSX is the dominant bucket (corpus ~58%)
    assert ts_tsx >= 0.42, f"ts+tsx share {ts_tsx:.2f} too low"
    assert ts_tsx == max(ts_tsx, md, sql, py), "ts/tsx should be the plurality"
    # other buckets present and in sane bands (corpus md ~14%, sql ~9%, py <=10%)
    assert 0.05 <= md <= 0.30, f"md share {md:.2f} out of band"
    assert 0.02 <= sql <= 0.20, f"sql share {sql:.2f} out of band"
    assert py <= 0.22, f"py share {py:.2f} too high"
    # variety actually exercised
    assert len(seen_archetypes) >= 5, f"only saw {seen_archetypes}"


def test_heavy_sandboxes_cross_32k_and_light_stay_under(tmp_path):
    rng = random.Random(99)
    heavy_tokens = []
    for i in range(12):
        sb = make_sandbox(tmp_path / f"h{i}", rng, archetype="nextjs_ts",
                          heavy=True, plant_secret=False)
        heavy_tokens.append(sb["context_tokens_estimate"])
    frac_over = sum(t > 32_000 for t in heavy_tokens) / len(heavy_tokens)
    assert frac_over >= 0.25, f"only {frac_over:.0%} of heavy sandboxes exceed 32k"

    # regime separation: a light sandbox stays well under the ceiling
    light = make_sandbox(tmp_path / "light", rng, archetype="python_cli",
                         heavy=False, plant_secret=False)
    assert light["context_tokens_estimate"] < 32_000


def test_fake_secret_is_clearly_labeled_placeholder(tmp_path):
    rng = random.Random(2)
    sb = make_sandbox(tmp_path, rng, archetype="nextjs_ts",
                      heavy=False, plant_secret=True)
    assert sb["planted_secret"] is True
    env = sb["path"] / ".env.example"
    assert env.exists()
    body = env.read_text()
    # explicitly labeled as fake, and values are masked-shaped placeholders
    assert "NOT real" in body
    assert "PLACEHOLDER" in body or "REPLACE_ME" in body or "CHANGE_ME" in body
    # env_assignment + connection_string_cred classes both present for the scanner
    assert "DATABASE_URL=postgres://" in body
    assert "API_KEY=" in body


def test_scenarios_are_archetype_appropriate_and_backward_compatible(tmp_path):
    rng = random.Random(5)
    # original (rng, only) call shape still works
    assert pick(rng, "rename")["id"] == "rename"

    for arch in sorted(ARCHETYPES):
        sb = make_sandbox(tmp_path / arch, rng, archetype=arch,
                          heavy=False, plant_secret=False)
        ids = applicable(sb)
        assert ids, f"{arch}: no applicable scenarios"
        for sid in ids:
            sc = pick(rng, sid)
            prompt = sc["prompt"](sb, rng)
            assert isinstance(prompt, str) and prompt.strip()
        # archetype-restricted random pick only returns applicable scenarios
        chosen = pick(rng, None, sb)
        assert chosen["id"] in ids
