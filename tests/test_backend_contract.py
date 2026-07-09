"""Contract tests for the WS0 backend ports — parametrized over every adapter.

One behavioral suite; each adapter (and the FallbackChain composite) must pass
all of it. New adapters get the full suite for free by joining ADAPTER_CASES.
The transport is faked by monkeypatching backends.http_post_json, so no live
server is needed.
"""

from __future__ import annotations

import pytest

import router.backends as backends
from router.backends import (
    LatticeBackend, OllamaBackend, OpenAICompatBackend, for_role, http_post_json,
)
from router.chain import FallbackChain

MESSAGES = [{"role": "system", "content": "judge difficulty"},
            {"role": "user", "content": "fix the failing test"}]


def ollama_chat_body(text="ok"):
    return {"message": {"content": text}}


def openai_chat_body(text="ok"):
    return {"choices": [{"message": {"content": text}}]}


def ollama_embed_body(n=1, dim=4):
    return {"embeddings": [[0.1] * dim for _ in range(n)]}


def openai_embed_body(n=1, dim=4):
    return {"data": [{"embedding": [0.1] * dim} for _ in range(n)]}


# (factory, chat_body_fn, embed_body_fn) — LatticeBackend is deliberately absent
# from the happy-path cases: it is an honest stub that abstains until wired.
ADAPTER_CASES = [
    (lambda: OllamaBackend("m"), ollama_chat_body, ollama_embed_body),
    (lambda: OpenAICompatBackend("m", base_url="http://localhost:1"),
     openai_chat_body, openai_embed_body),
]


@pytest.mark.parametrize("factory,chat_body,embed_body", ADAPTER_CASES)
def test_chat_happy_path(monkeypatch, factory, chat_body, embed_body):
    monkeypatch.setattr(backends, "http_post_json",
                        lambda endpoint, payload, timeout: chat_body("the answer"))
    assert factory().chat(MESSAGES) == "the answer"


@pytest.mark.parametrize("factory,chat_body,embed_body", ADAPTER_CASES)
def test_embed_happy_path(monkeypatch, factory, chat_body, embed_body):
    monkeypatch.setattr(backends, "http_post_json",
                        lambda endpoint, payload, timeout: embed_body(2))
    vectors = factory().embed(["a", "b"])
    assert vectors is not None and len(vectors) == 2 and len(vectors[0]) == 4


@pytest.mark.parametrize("factory,chat_body,embed_body", ADAPTER_CASES)
@pytest.mark.parametrize("bad_body", [None, {}, {"unexpected": 1},
                                      {"message": {}}, {"choices": []}])
def test_failures_return_none_never_raise(monkeypatch, factory, chat_body,
                                          embed_body, bad_body):
    monkeypatch.setattr(backends, "http_post_json",
                        lambda endpoint, payload, timeout: bad_body)
    adapter = factory()
    assert adapter.chat(MESSAGES) is None
    assert adapter.embed(["a"]) is None


@pytest.mark.parametrize("factory,chat_body,embed_body", ADAPTER_CASES)
def test_transport_exception_is_swallowed(monkeypatch, factory, chat_body, embed_body):
    def explode(endpoint, payload, timeout):
        raise RuntimeError("transport blew up")
    monkeypatch.setattr(backends, "http_post_json", explode)
    adapter = factory()
    assert adapter.chat(MESSAGES) is None       # fail_safe absorbs it
    assert adapter.embed(["a"]) is None


@pytest.mark.parametrize("factory,chat_body,embed_body", ADAPTER_CASES)
def test_embed_count_mismatch_rejected(monkeypatch, factory, chat_body, embed_body):
    monkeypatch.setattr(backends, "http_post_json",
                        lambda endpoint, payload, timeout: embed_body(1))
    assert factory().embed(["a", "b"]) is None  # 1 vector for 2 texts -> None


def test_lattice_stub_abstains_fail_safe():
    lattice = LatticeBackend()
    assert lattice.chat(MESSAGES) is None
    assert lattice.embed(["a"]) is None


def test_chain_falls_through_lattice_to_ollama(monkeypatch):
    monkeypatch.setattr(backends, "http_post_json",
                        lambda endpoint, payload, timeout: ollama_chat_body("served"))
    chain = FallbackChain([LatticeBackend(), OllamaBackend("m")])
    assert chain.chat(MESSAGES) == "served"     # stub abstains, ollama answers


def test_chain_all_fail_returns_none(monkeypatch):
    monkeypatch.setattr(backends, "http_post_json",
                        lambda endpoint, payload, timeout: None)
    chain = FallbackChain([LatticeBackend(), OllamaBackend("m")])
    assert chain.chat(MESSAGES) is None


def test_chain_member_raising_is_survived():
    class Hostile:
        def chat(self, *a, **k):
            raise RuntimeError("bad member")

    class Good:
        def chat(self, *a, **k):
            return "fine"

    assert FallbackChain([Hostile(), Good()]).chat(MESSAGES) == "fine"


def test_for_role_defaults_and_config_swap(tmp_path):
    # unknown/missing config file -> ROLE_DEFAULTS (ollama classifier)
    adapter = for_role("classifier", config_path=tmp_path / "missing.yaml")
    assert isinstance(adapter, OllamaBackend)
    assert adapter.model == "qwen2.5:3b-instruct"

    # config-only swap: same role, now an OpenAI-compat endpoint (llama.cpp/MLX)
    cfg = tmp_path / "backends.yaml"
    cfg.write_text(
        "classifier:\n  adapter: openai_compat\n  model: some-gguf\n"
        "  base_url: http://localhost:8080\n  timeout: 3.0\n")
    swapped = for_role("classifier", config_path=cfg)
    assert isinstance(swapped, OpenAICompatBackend)
    assert swapped.base_url == "http://localhost:8080" and swapped.model == "some-gguf"


def test_for_role_with_fallback_chain(tmp_path):
    cfg = tmp_path / "backends.yaml"
    cfg.write_text(
        "classifier:\n  adapter: lattice\n  model: x\n"
        "  fallbacks:\n    - {adapter: ollama, model: qwen2.5:3b-instruct}\n")
    chain = for_role("classifier", config_path=cfg)
    assert isinstance(chain, FallbackChain)
    assert isinstance(chain.members[0], LatticeBackend)
    assert isinstance(chain.members[1], OllamaBackend)


def test_for_role_never_raises_on_garbage_config(tmp_path):
    cfg = tmp_path / "backends.yaml"
    cfg.write_text("classifier: [this, is, not, a, mapping\n")  # malformed yaml
    adapter = for_role("classifier", config_path=cfg)
    assert isinstance(adapter, OllamaBackend)   # falls back to defaults


def test_classifier_through_backend_port(monkeypatch):
    """Acceptance: classify_intent_llm consumes the port (WS0)."""
    from router.llm_classifier import classify_intent_llm

    monkeypatch.setattr(
        backends, "http_post_json",
        lambda endpoint, payload, timeout: ollama_chat_body(
            '{"tier":"standard","needs_frontier":false,"reason":"single file"}'))
    verdict = classify_intent_llm("fix the failing test",
                                  backend=OllamaBackend("qwen2.5:3b-instruct"))
    assert verdict is not None and verdict.tier == "standard" and not verdict.needs_frontier


def test_classifier_backend_failure_falls_back_to_none(monkeypatch):
    from router.llm_classifier import classify_intent_llm

    monkeypatch.setattr(backends, "http_post_json",
                        lambda endpoint, payload, timeout: None)
    assert classify_intent_llm("fix it", backend=OllamaBackend("m")) is None


def test_for_role_wrong_shape_config_falls_back(tmp_path):
    """Finding #2: a valid-YAML but wrong-shape role (list/str) must not raise."""
    cfg = tmp_path / "backends.yaml"
    cfg.write_text("classifier: [this, is, wrong]\n")
    adapter = for_role("classifier", config_path=cfg)
    assert isinstance(adapter, OllamaBackend)   # defaults, no exception
