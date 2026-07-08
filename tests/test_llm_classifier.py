"""Unit tests for the production LLM difficulty classifier.

The HTTP call is always mocked — these tests never hit a live ollama. Two seams
are exercised: injecting a ``_sender`` callable, and monkeypatching the module's
``urllib.request.urlopen`` for the transport-level failure paths.

The contract under test is fail-safety: valid input parses correctly, and every
failure mode returns None without an exception escaping.
"""

from __future__ import annotations

import io
import json
import socket
import urllib.error

import pytest

from router import llm_classifier as lc
from router.llm_classifier import LlmVerdict, classify_intent_llm


# ── helpers ─────────────────────────────────────────────────────────────────

def _sender_returning(content):
    """Build a _sender that ignores its args and returns a fixed content str."""
    def _send(endpoint, payload, timeout):
        return content
    return _send


class _FakeResponse:
    """Minimal context-manager stand-in for urllib's urlopen return value."""

    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def _ollama_envelope(content: str) -> bytes:
    return json.dumps({"message": {"role": "assistant", "content": content}}).encode()


# ── valid JSON -> correct verdict + score mapping ────────────────────────────

def test_valid_json_hard_maps_to_score_085():
    content = '{"tier":"hard","needs_frontier":true,"reason":"refactor across modules"}'
    verdict = classify_intent_llm("refactor the auth layer", _sender=_sender_returning(content))
    assert isinstance(verdict, LlmVerdict)
    assert verdict.tier == "hard"
    assert verdict.needs_frontier is True
    assert verdict.score == 0.85
    assert verdict.reason == "refactor across modules"


def test_valid_json_trivial_maps_to_score_015():
    content = '{"tier":"trivial","needs_frontier":false,"reason":"just a rename"}'
    verdict = classify_intent_llm("rename this var", _sender=_sender_returning(content))
    assert verdict is not None
    assert verdict.tier == "trivial"
    assert verdict.needs_frontier is False
    assert verdict.score == 0.15


def test_valid_json_standard_maps_to_score_050():
    content = '{"tier":"standard","needs_frontier":false,"reason":"single file fix"}'
    verdict = classify_intent_llm("fix the off-by-one", _sender=_sender_returning(content))
    assert verdict is not None
    assert verdict.tier == "standard"
    assert verdict.needs_frontier is False
    assert verdict.score == 0.50


# ── slightly-malformed JSON -> lenient parse still works ─────────────────────

def test_prose_wrapped_json_is_recovered():
    content = 'Sure! Here is my verdict: {"tier":"hard","needs_frontier":true,"reason":"x"} done.'
    verdict = classify_intent_llm("redesign the pipeline", _sender=_sender_returning(content))
    assert verdict is not None
    assert verdict.tier == "hard"
    assert verdict.score == 0.85


def test_trailing_comma_json_recovered_via_regex_fallback():
    # json.loads chokes on the trailing comma; regex fallback still finds tier.
    content = '{"tier":"standard","needs_frontier":false,}'
    verdict = classify_intent_llm("add a helper", _sender=_sender_returning(content))
    assert verdict is not None
    assert verdict.tier == "standard"
    assert verdict.needs_frontier is False
    assert verdict.score == 0.50


# ── tier-only response -> needs_frontier derived ─────────────────────────────

def test_tier_only_hard_derives_needs_frontier_true():
    content = '{"tier":"hard"}'
    verdict = classify_intent_llm("overhaul everything", _sender=_sender_returning(content))
    assert verdict is not None
    assert verdict.tier == "hard"
    assert verdict.needs_frontier is True  # derived: tier == "hard"
    assert verdict.score == 0.85


def test_tier_only_trivial_derives_needs_frontier_false():
    content = 'tier: trivial'
    verdict = classify_intent_llm("fix typo", _sender=_sender_returning(content))
    assert verdict is not None
    assert verdict.tier == "trivial"
    assert verdict.needs_frontier is False
    assert verdict.score == 0.15


# ── garbage / empty body -> None ─────────────────────────────────────────────

def test_garbage_body_returns_none():
    verdict = classify_intent_llm("do the thing", _sender=_sender_returning("lorem ipsum no verdict here"))
    assert verdict is None


def test_no_valid_tier_in_json_returns_none():
    content = '{"tier":"mega-hard","needs_frontier":true}'
    verdict = classify_intent_llm("x", _sender=_sender_returning(content))
    assert verdict is None


def test_empty_content_returns_none():
    assert classify_intent_llm("x", _sender=_sender_returning("")) is None


def test_sender_returning_none_returns_none():
    assert classify_intent_llm("x", _sender=_sender_returning(None)) is None


def test_blank_input_text_returns_none_without_calling_sender():
    calls = []

    def _send(endpoint, payload, timeout):
        calls.append(1)
        return '{"tier":"hard"}'

    assert classify_intent_llm("   ", _sender=_send) is None
    assert calls == []  # short-circuits before any HTTP work


# ── raised connection error -> None (no exception escapes) ───────────────────

def test_connection_refused_returns_none():
    def _send(endpoint, payload, timeout):
        raise urllib.error.URLError("Connection refused")

    assert classify_intent_llm("x", _sender=_send) is None


def test_arbitrary_exception_in_sender_returns_none():
    def _send(endpoint, payload, timeout):
        raise RuntimeError("boom")

    assert classify_intent_llm("x", _sender=_send) is None


# ── timeout -> None ──────────────────────────────────────────────────────────

def test_timeout_returns_none():
    def _send(endpoint, payload, timeout):
        raise socket.timeout("timed out")

    assert classify_intent_llm("x", _sender=_send) is None


# ── transport seam: monkeypatch urllib directly (default sender path) ────────

def test_urlopen_monkeypatched_valid_response(monkeypatch):
    content = '{"tier":"hard","needs_frontier":true,"reason":"multi-file"}'

    def fake_urlopen(request, timeout=None):
        return _FakeResponse(_ollama_envelope(content))

    monkeypatch.setattr(lc.urllib.request, "urlopen", fake_urlopen)
    verdict = classify_intent_llm("refactor auth")  # no _sender -> real _http_post
    assert verdict is not None
    assert verdict.tier == "hard"
    assert verdict.score == 0.85


def test_urlopen_monkeypatched_http_error_returns_none(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            url="http://x", code=500, msg="err", hdrs=None, fp=io.BytesIO(b""))

    monkeypatch.setattr(lc.urllib.request, "urlopen", fake_urlopen)
    assert classify_intent_llm("x") is None


def test_urlopen_monkeypatched_empty_content_returns_none(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return _FakeResponse(_ollama_envelope("   "))

    monkeypatch.setattr(lc.urllib.request, "urlopen", fake_urlopen)
    assert classify_intent_llm("x") is None


def test_urlopen_monkeypatched_non_json_body_returns_none(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return _FakeResponse(b"<html>502 Bad Gateway</html>")

    monkeypatch.setattr(lc.urllib.request, "urlopen", fake_urlopen)
    assert classify_intent_llm("x") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
