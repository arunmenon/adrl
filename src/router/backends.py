"""Backend ports — the router's model-serving contract (WS0).

The router never talks to a serving framework directly. It states its needs as
two deliberately tiny ports and consumes whichever adapter the config selects:

    GenerationBackend.chat(messages, options)  -> assistant text | None
    EmbeddingBackend.embed(texts)              -> list[vector]   | None

**Fail-safe is the contract.** Every adapter method returns ``None`` on ANY
failure (refused connection, timeout, non-200, malformed body, any exception)
and never raises — these calls sit on the live routing hot path, and a serving
hiccup must degrade a decision, never break a turn. The shared ``fail_safe``
decorator is the single implementation of that promise (previously copy-pasted
per module).

Adapters:
    LatticeBackend     — the user's own wrapper over serving frameworks; the
                         intended primary. Interface shim pending (TODO below).
    OpenAICompatBackend— /v1/chat/completions + /v1/embeddings; one adapter
                         covers llama.cpp server, mlx_lm.server, vLLM, and
                         ollama's own OpenAI-compatible endpoint.
    OllamaBackend      — ollama-native /api/chat + /api/embed (today's default).

Selection is by config (config/backends.yaml, per role: classifier / embedder /
local_small / local_code), never by import. Stdlib-only HTTP (urllib).
"""

from __future__ import annotations

import functools
import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional

DEFAULT_CONFIG_PATH = Path("config/backends.yaml")
DEFAULT_TIMEOUT = 5.0


def fail_safe(fn: Callable) -> Callable:
    """Any exception -> None. The one shared implementation of the hot-path
    fail-safe promise (see module docstring)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            return None

    return wrapper


def http_post_json(endpoint: str, payload: dict, timeout: float) -> Optional[dict]:
    """POST JSON, return the decoded JSON body, or None on any failure.

    The single low-level HTTP seam shared by every adapter (and by
    llm_classifier / eval_classifier, which previously each hand-rolled this).
    Monkeypatch ``urllib.request.urlopen`` or inject at the adapter level in
    tests.
    """
    try:
        data = json.dumps(payload).encode()
        request = urllib.request.Request(
            endpoint, data=data, method="POST",
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            OSError, ValueError, json.JSONDecodeError):
        return None


# ── Ports ────────────────────────────────────────────────────────────────────


class GenerationBackend(ABC):
    """Chat-completion port. ``messages`` is an OpenAI/anthropic-shaped list of
    {role, content}; ``options`` carries sampling knobs (temperature,
    max_tokens/num_predict, keep_alive where supported)."""

    @abstractmethod
    def chat(self, messages: list[dict], options: Optional[dict] = None) -> Optional[str]:
        """Return the assistant message text, or None on any failure."""


class EmbeddingBackend(ABC):
    """Embedding port. Returns one vector per input text, or None on failure."""

    @abstractmethod
    def embed(self, texts: list[str]) -> Optional[list[list[float]]]:
        """Return embeddings aligned with ``texts``, or None on any failure."""


# ── Adapters ─────────────────────────────────────────────────────────────────


class OllamaBackend(GenerationBackend, EmbeddingBackend):
    """ollama-native adapter: /api/chat and /api/embed."""

    def __init__(self, model: str, *, base_url: str = "http://localhost:11434",
                 timeout: float = DEFAULT_TIMEOUT,
                 keep_alive: str | int = "10m"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive

    @fail_safe
    def chat(self, messages: list[dict], options: Optional[dict] = None) -> Optional[str]:
        options = dict(options or {})
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": options.get("temperature", 0),
                "num_predict": options.get("num_predict",
                                           options.get("max_tokens", 256)),
            },
            "keep_alive": options.get("keep_alive", self.keep_alive),
        }
        body = http_post_json(f"{self.base_url}/api/chat", payload, self.timeout)
        if body is None:
            return None
        content = (body.get("message") or {}).get("content")
        if not isinstance(content, str) or not content.strip():
            return None
        return content

    @fail_safe
    def embed(self, texts: list[str]) -> Optional[list[list[float]]]:
        payload = {"model": self.model, "input": texts,
                   "keep_alive": self.keep_alive}
        body = http_post_json(f"{self.base_url}/api/embed", payload, self.timeout)
        if body is None:
            return None
        vectors = body.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            return None
        return vectors


class OpenAICompatBackend(GenerationBackend, EmbeddingBackend):
    """OpenAI-compatible adapter: /v1/chat/completions and /v1/embeddings.

    One adapter covers llama.cpp server, mlx_lm.server, vLLM, LM Studio, and
    ollama's own /v1 endpoint — anything OpenAI-shaped.
    """

    def __init__(self, model: str, *, base_url: str,
                 timeout: float = DEFAULT_TIMEOUT, api_key: str = ""):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key  # most local servers ignore it; kept for parity

    @fail_safe
    def chat(self, messages: list[dict], options: Optional[dict] = None) -> Optional[str]:
        options = dict(options or {})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": options.get("temperature", 0),
            "max_tokens": options.get("max_tokens",
                                      options.get("num_predict", 256)),
            "stream": False,
        }
        body = http_post_json(f"{self.base_url}/v1/chat/completions", payload,
                              self.timeout)
        if body is None:
            return None
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        content = ((choices[0] or {}).get("message") or {}).get("content")
        if not isinstance(content, str) or not content.strip():
            return None
        return content

    @fail_safe
    def embed(self, texts: list[str]) -> Optional[list[list[float]]]:
        payload = {"model": self.model, "input": texts}
        body = http_post_json(f"{self.base_url}/v1/embeddings", payload,
                              self.timeout)
        if body is None:
            return None
        rows = body.get("data")
        if not isinstance(rows, list) or len(rows) != len(texts):
            return None
        vectors = [r.get("embedding") for r in rows if isinstance(r, dict)]
        if len(vectors) != len(texts) or any(not isinstance(v, list) for v in vectors):
            return None
        return vectors


class LatticeBackend(GenerationBackend, EmbeddingBackend):
    """Shim onto lattice — the user's wrapper framework over serving backends
    (ollama / llama.cpp / MLX). Intended as the PRIMARY adapter.

    TODO(lattice): implement against lattice's real interface once its repo /
    endpoint and call signature are shared (plan WS0 input). Until then this is
    an honest stub: it satisfies the port contract by fail-safe-returning None,
    so a chain configured [lattice, ollama] simply falls through to the next
    adapter. No mock behavior.
    """

    def __init__(self, model: str = "", *, base_url: str = "",
                 timeout: float = DEFAULT_TIMEOUT):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    @fail_safe
    def chat(self, messages: list[dict], options: Optional[dict] = None) -> Optional[str]:
        # TODO(lattice): translate the port call onto lattice's API.
        return None

    @fail_safe
    def embed(self, texts: list[str]) -> Optional[list[list[float]]]:
        # TODO(lattice): translate the port call onto lattice's API.
        return None


ADAPTERS: dict[str, type] = {
    "lattice": LatticeBackend,
    "openai_compat": OpenAICompatBackend,
    "ollama": OllamaBackend,
}


# ── Config-driven construction ───────────────────────────────────────────────


def _load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read config/backends.yaml; missing or malformed file -> {} (defaults)."""
    try:
        import yaml
        with open(path) as fh:
            loaded = yaml.safe_load(fh)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


# Built-in defaults per role — what runs on this machine today. The config file
# overrides these; code never needs to change to move a role to a new backend.
ROLE_DEFAULTS: dict[str, dict[str, Any]] = {
    "classifier": {"adapter": "ollama", "model": "qwen2.5:3b-instruct",
                   "timeout": 5.0},
    "embedder": {"adapter": "ollama", "model": "nomic-embed-text",
                 "timeout": 10.0},
    "local_small": {"adapter": "ollama", "model": "llama3.2:latest",
                    "timeout": 45.0},
    "local_code": {"adapter": "ollama", "model": "qwen2.5:7b-instruct-q4_K_M",
                   "timeout": 90.0},
}


def for_role(role: str, *, config_path: Path = DEFAULT_CONFIG_PATH):
    """Build the configured adapter (or chain) for a role.

    Config shape per role (all keys optional; defaults from ROLE_DEFAULTS):
        classifier:
          adapter: ollama | openai_compat | lattice
          model: qwen2.5:3b-instruct
          base_url: http://localhost:11434
          timeout: 5.0
          fallbacks:            # optional chain, tried in order after primary
            - {adapter: ollama, model: ...}

    Returns a single adapter, or a FallbackChain when fallbacks are configured.
    Unknown role or adapter names fall back to ROLE_DEFAULTS / ollama — this
    function must never raise on bad config (fail-safe extends to setup).
    """
    spec = dict(ROLE_DEFAULTS.get(role, ROLE_DEFAULTS["classifier"]))
    spec.update((_load_config(config_path).get(role) or {}))

    def build(one: dict[str, Any]):
        cls = ADAPTERS.get(str(one.get("adapter", "ollama")), OllamaBackend)
        kwargs: dict[str, Any] = {"timeout": float(one.get("timeout", DEFAULT_TIMEOUT))}
        if one.get("base_url"):
            kwargs["base_url"] = str(one["base_url"])
        elif cls is OpenAICompatBackend:
            kwargs["base_url"] = "http://localhost:11434"  # ollama's /v1 works too
        return cls(str(one.get("model", "")), **kwargs)

    primary = build(spec)
    fallbacks = spec.get("fallbacks") or []
    if not isinstance(fallbacks, list) or not fallbacks:
        return primary
    from router.chain import FallbackChain
    return FallbackChain([primary] + [build(f) for f in fallbacks if isinstance(f, dict)])
