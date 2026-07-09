"""WS2 UNIT A — the live_router decision module (the brain the proxy calls).

This is the PURE decision layer that the routing proxy consults on every wire
request. It owns *no* I/O: no aiohttp, no live model calls, no sockets. Every
dependency the decision needs — the shared session store, the transaction
memory recorder, and the LLM difficulty classifier — is *injected*, so this
module stays fast, deterministic, and unit-testable without a live server.

It answers one question — "where does this request go, and where does it fall
open to?" — as a ``RoutePlan``, and never raises: any failure whatsoever
collapses to a plain Anthropic passthrough plan (the same fail-safe philosophy
as ``proxy.capture_proxy.plan_route`` and the router hook).

DISPATCH MAP (label -> rung -> upstream/model rewrite):

  user_turn    -> discriminator.session_key -> features.extract -> route_turn:
     rung local        -> model "local-code" on LiteLLM, fallback = original body -> Anthropic
     rung cheap_cloud  -> model "claude-haiku-4-5" on Anthropic (subscription auth), no fallback
     rung frontier     -> original body on Anthropic, no rewrite
  continuation -> STICK to the session's sticky route (the escalation store),
                  dispatched exactly like its rung; route_turn is NOT re-run.
  utility:*    -> model "local-small" on LiteLLM, fallback = original body -> Anthropic
  passthrough:* / subagent / anything else -> Anthropic passthrough, no rewrite.

CLOUD dispatch reuses the client's inbound (subscription) auth by relaying to
Anthropic with only the model field changed; only LOCAL dispatch swaps the
upstream to LiteLLM. Fail-open ALWAYS lands on the client's original Anthropic
request (``fallback_model is None`` => forward the original body unchanged).

Recording (WS1 flywheel): on a user_turn boundary, if a ``RoutingRecorder`` is
injected, the decision is written to the transaction memory (``new_turn`` closes
the prior turn, ``record`` mints a ``route_id`` and stamps it on the plan). RAW
instruction text is passed to the recorder — the facade hashes/scrubs it; text
is never stored here.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Optional

from . import discriminator
from .features import extract
from .policy import route_turn
from .state import DictSessionStore, SessionStore

DEFAULT_LITELLM_URL = "http://localhost:4001"
DEFAULT_ANTHROPIC_URL = "https://api.anthropic.com"

# Model names each rung rewrites to (LiteLLM aliases for local, Anthropic model
# ids for cloud). Verified live: "local-code" -> qwen2.5:7b, "local-small" ->
# llama3.2 on the LiteLLM :4001 instance.
LOCAL_CODE_MODEL = "local-code"
LOCAL_SMALL_MODEL = "local-small"
CHEAP_CLOUD_MODEL = "claude-haiku-4-5"

# ~4 characters per token — a cheap context-size estimate for the feasibility
# gate, good enough to tell a 2k-token turn from a 200k-token one.
CHARS_PER_TOKEN = 4

INTERRUPT_MARKER = "[Request interrupted by user]"

# policy rung -> the escalation store's rung vocabulary. The sticky route MUST be
# stored in the dash-form because EscalationController.NEXT_RUNG keys on
# 'local-code'/'cheap-cloud'; storing raw 'local'/'cheap_cloud' would break the
# escalation ladder. _normalize_rung accepts both forms for dispatch.
_STORE_RUNG = {"local": "local-code", "cheap_cloud": "cheap-cloud",
               "frontier": "frontier"}


def _prev_turn_interrupted(body: dict) -> bool:
    """True when a recent user message carries the interrupt marker — the signal
    that the previous turn was cut off and this is the retry (S6). Feeds both the
    difficulty escalate-on-retry boost and the memory's Wave-2 finalize."""
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        return False
    for message in messages[-3:]:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(b.get("text", "") for b in content
                            if isinstance(b, dict))
        else:
            text = ""
        if INTERRUPT_MARKER in text:
            return True
    return False


@dataclass
class RoutePlan:
    """Where one request goes, and where it falls open to.

    ``primary_model``/``fallback_model`` are ``None`` when the body must be
    forwarded UNCHANGED (frontier / passthrough / the original-body fail-open);
    a string means rewrite the ``model`` field to it before forwarding.
    ``fallback_upstream is None`` marks a plan with no fail-open (already on
    Anthropic) — byte-identical to a plain relay.
    """

    primary_model: str | None
    primary_upstream: str
    fallback_model: str | None
    fallback_upstream: str | None
    rung: str
    label: str
    route_id: str | None
    layer: str


def _last_user_text(body: dict) -> str:
    """The isolated human instruction: the text blocks of the LAST user message,
    joined. This is exactly what the difficulty classifier is allowed to see —
    never the surrounding context."""
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            joined = " ".join(part for part in parts if part)
            if joined:
                return joined
    return ""


def _estimate_context_tokens(body: dict) -> int:
    """Rough token estimate of the whole working set (system + all messages,
    including tool_result payloads) for the policy feasibility gate."""
    total_chars = 0
    system = body.get("system")
    if isinstance(system, str):
        total_chars += len(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                total_chars += len(block.get("text", "") or "")
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                total_chars += len(block.get("text", "") or "")
                tool_content = block.get("content")
                if isinstance(tool_content, str):
                    total_chars += len(tool_content)
                elif isinstance(tool_content, list):
                    for inner in tool_content:
                        if isinstance(inner, dict):
                            total_chars += len(inner.get("text", "") or "")
    return total_chars // CHARS_PER_TOKEN


def _fallback_sid(body: dict) -> str:
    """A stable session key when ``metadata.user_id`` is absent: hash the system
    prompt (or the first message) so the same conversation keeps one key across
    requests rather than collapsing all anonymous traffic into one bucket."""
    basis = ""
    system = body.get("system")
    if isinstance(system, str):
        basis = system[:500]
    elif isinstance(system, list):
        basis = " ".join(
            block.get("text", "")
            for block in system
            if isinstance(block, dict)
        )[:500]
    if not basis:
        messages = body.get("messages") or []
        if isinstance(messages, list) and messages:
            basis = str(messages[0])[:500]
    digest = hashlib.sha1(basis.encode("utf-8", "replace")).hexdigest()[:16]
    return "anon:" + digest


class LiveRouter:
    """The injected, pure decision layer. ``plan()`` never raises."""

    def __init__(self, *, store: Optional[SessionStore] = None,
                 memory: Any = None, classifier: Any = None,
                 litellm_url: str = DEFAULT_LITELLM_URL,
                 anthropic_url: str = DEFAULT_ANTHROPIC_URL,
                 policy_version: str = "v1",
                 source: str = "organic"):
        # store: the SessionStore shared with the escalation controller (so the
        # sticky route the controller flips is the same one continuations read).
        self.store: SessionStore = store or DictSessionStore()
        self.memory = memory                    # RoutingRecorder | None
        self.classifier = classifier            # classify_intent_llm | None
        self.litellm_url = litellm_url
        self.anthropic_url = anthropic_url
        self.policy_version = policy_version
        # Instance-level provenance stamped on every recorded decision. A routing
        # instance carries ONE traffic kind: 'simulator' when the synthetic driver
        # points at it (fuel), 'organic' for a user's own opt-in sessions. Keeping
        # them separable is what lets WS4 firewall synthetic votes out of the
        # organic pool and WS3 avoid mixing synthetic outcomes into rule health.
        self.source = source

    # ── dispatch primitives ──────────────────────────────────────────────────

    def _passthrough(self, *, label: str, layer: str = "passthrough") -> RoutePlan:
        """Anthropic relay, no rewrite, no fallback — the universal fail-safe."""
        return RoutePlan(
            primary_model=None, primary_upstream=self.anthropic_url,
            fallback_model=None, fallback_upstream=None,
            rung="passthrough", label=label, route_id=None, layer=layer)

    def _dispatch(self, rung: str, *, label: str, layer: str) -> RoutePlan:
        """Map a rung (policy vocabulary 'local'|'cheap_cloud'|'frontier' OR the
        escalation store's 'local-code'|'local-small'|'cheap-cloud'|'frontier')
        to a concrete forward + fail-open plan."""
        normalized = self._normalize_rung(rung)
        if normalized == "local":
            # Local execution on LiteLLM; fail open to the ORIGINAL body on Anthropic.
            return RoutePlan(
                primary_model=LOCAL_CODE_MODEL, primary_upstream=self.litellm_url,
                fallback_model=None, fallback_upstream=self.anthropic_url,
                rung=rung, label=label, route_id=None, layer=layer)
        if normalized == "local_small":
            return RoutePlan(
                primary_model=LOCAL_SMALL_MODEL, primary_upstream=self.litellm_url,
                fallback_model=None, fallback_upstream=self.anthropic_url,
                rung=rung, label=label, route_id=None, layer=layer)
        if normalized == "cheap_cloud":
            # Cloud on the client's subscription auth — model swap only, no
            # upstream change (no LiteLLM; its key is credit-less). Fail open to
            # the ORIGINAL model on Anthropic (review finding): a rewrite to a
            # model the subscription can't access (model_not_found) must retry
            # the un-routed request rather than fail a turn that would have
            # succeeded — and this rung is sticky after an escalation.
            return RoutePlan(
                primary_model=CHEAP_CLOUD_MODEL, primary_upstream=self.anthropic_url,
                fallback_model=None, fallback_upstream=self.anthropic_url,
                rung=rung, label=label, route_id=None, layer=layer)
        if normalized == "frontier":
            return RoutePlan(
                primary_model=None, primary_upstream=self.anthropic_url,
                fallback_model=None, fallback_upstream=None,
                rung=rung, label=label, route_id=None, layer=layer)
        # Unknown rung -> safest thing is a plain passthrough.
        return self._passthrough(label=label, layer=layer)

    @staticmethod
    def _normalize_rung(rung: str) -> str:
        """Collapse both rung vocabularies to one dispatch key."""
        if rung in ("local", "local-code"):
            return "local"
        if rung == "local-small":
            return "local_small"
        if rung in ("cheap_cloud", "cheap-cloud"):
            return "cheap_cloud"
        if rung == "frontier":
            return "frontier"
        return "unknown"

    # ── the decision ─────────────────────────────────────────────────────────

    def plan(self, method: str, path: str, body: dict) -> RoutePlan:
        """One wire request -> a RoutePlan. NEVER raises: any failure collapses
        to an Anthropic passthrough (fail-safe like plan_route)."""
        try:
            return self._plan(method, path, body)
        except Exception:
            return self._passthrough(label="error", layer="error")

    def _plan(self, method: str, path: str, body: dict) -> RoutePlan:
        if not isinstance(body, dict):
            body = {}
        label = discriminator.classify(method, path, body)
        session_id = discriminator.session_key(body) or _fallback_sid(body)

        if label == "user_turn":
            return self._plan_user_turn(body, session_id, label)

        if label == "continuation":
            # Stick to whatever rung the session is already on (default local);
            # do NOT re-run route_turn — the sticky route is cache-safe.
            session = self.store.get_session(session_id)
            sticky = session.route or "local"
            return self._dispatch(sticky, label=label, layer="continuation")

        if label.startswith("utility"):
            return self._dispatch("local-small", label=label, layer="utility")

        # passthrough:*, subagent, or any unrecognized label -> Anthropic, no routing.
        return self._passthrough(label=label, layer="passthrough")

    def _plan_user_turn(self, body: dict, session_id: str, label: str) -> RoutePlan:
        instruction_text = _last_user_text(body)
        context_tokens = _estimate_context_tokens(body)
        prev_interrupted = _prev_turn_interrupted(body)
        session = self.store.get_session(session_id)
        features = extract(
            instruction_text, context_tokens=context_tokens,
            turn_index=session.turn_count,
            prev_turn_interrupted=prev_interrupted,   # S6 escalate-on-retry
            privacy_pinned=session.privacy_pinned,
            escalated_this_episode=session.escalated_this_episode,
        )
        started = time.perf_counter()
        route = route_turn(features, session, classifier=self.classifier)
        decision_ms = (time.perf_counter() - started) * 1000.0

        plan = self._dispatch(route.rung, label=label, layer=route.layer)

        # CRITICAL (review finding): persist the sticky route so this turn's
        # follow-up continuations stick to the chosen rung (not the default
        # local) and the escalation ladder can key off it. Store the dash-form.
        try:
            self.store.set_route(session_id, _STORE_RUNG.get(route.rung, route.rung))
        except Exception:
            pass

        # RECORD the decision to the transaction memory (off the critical path;
        # any failure is swallowed — a memory outage never blocks routing).
        if self.memory is not None:
            try:
                # Wave-2: the retry/interrupt signal finalizes the PRIOR turn.
                self.memory.new_turn(
                    session_id, prev_interrupted=prev_interrupted,
                    prev_retried=prev_interrupted,
                )
                route_id = self.memory.record(
                    instruction_text, features, route, session_id=session_id,
                    turn_index=session.turn_count, decision_ms=decision_ms,
                    source=self.source,
                )
                plan.route_id = route_id
            except Exception:
                pass

        # Advance the per-session turn counter so the NEXT user_turn orders after
        # this one (turn_index in the ledger; finalize_turn ORDER BY). Only
        # user_turns increment — the live path records only on user_turn.
        try:
            self.store.incr_turn(session_id)
        except Exception:
            pass
        return plan

    # ── forwarding helper ────────────────────────────────────────────────────

    def build_forward_body(self, body: dict, model: str | None) -> bytes:
        """Serialize the body to forward. ``model is None`` forwards the body
        unchanged (frontier / passthrough / original-body fail-open); a string
        rewrites the ``model`` field first."""
        if model is None:
            try:
                return json.dumps(body).encode()
            except Exception:
                return b"{}"
        try:
            rewritten = dict(body)
            rewritten["model"] = model
            return json.dumps(rewritten).encode()
        except Exception:
            return json.dumps({"model": model}).encode()
