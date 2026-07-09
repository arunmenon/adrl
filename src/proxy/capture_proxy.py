"""B2 — transparent wire-capture proxy for Claude Code traffic.

Sits between Claude Code and api.anthropic.com, logging every request and
response verbatim (headers redacted, bodies raw per decision D5) while
changing nothing. This is the data source for the wire-level Phase 0
questions: sidecar-call fingerprints (S1), metadata.user_id keying (B4),
tool-ID re-minting (B5), and the live shadow run (B7/B8).

Usage:
    .venv/bin/python -m proxy.capture_proxy --port 4000
    ANTHROPIC_BASE_URL=http://localhost:4000 claude

Kill switch: unset ANTHROPIC_BASE_URL — Claude Code goes direct again.
Fail-open is not possible for a transport proxy (if this dies, requests
fail), so it does the minimum: no parsing on the hot path, capture I/O
happens after the response has been fully relayed.

P1-A live utility pinning adds a *routing* fail-open on top of that: a
utility call pinned to the local rung that fails (connection error, timeout,
or 5xx from LiteLLM) is transparently retried against the ORIGINAL Anthropic
upstream with the ORIGINAL unrewritten body, so the user never sees a 502.
Every non-routed path stays byte-identical to the plain capture relay.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

import os

# Router hook (P1-A). If src/router isn't importable, the proxy still runs as a
# pure capture relay.
try:
    from router.hook import apply as _hook_apply
except Exception:  # pragma: no cover
    def _hook_apply(method, path, body):  # type: ignore
        return body, None

UPSTREAM = os.environ.get("CAPTURE_UPSTREAM", "https://api.anthropic.com")
REDACT_HEADERS = {"authorization", "x-api-key", "cookie", "set-cookie"}
# Hop-by-hop headers must not be forwarded either direction.
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
    "accept-encoding",  # ask upstream for identity so captured bodies are plaintext
}

_seq = 0
# Time-to-completion cap for a locally-pinned (utility) attempt. Bounds a hung
# LiteLLM so fail-open can trigger; generous enough for a warm/cold local model.
LOCAL_ATTEMPT_TIMEOUT = 45


def _capture_dir(root: Path) -> Path:
    d = root / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _redacted(headers) -> dict[str, str]:
    return {
        k: ("<redacted>" if k.lower() in REDACT_HEADERS else v)
        for k, v in headers.items()
    }


# ── P1-A routing decision (pure; unit-tested in tests/test_hook_failopen.py) ──
@dataclass
class RoutePlan:
    """The routing decision for one request: where to send it first, and — for a
    rewritten (locally-pinned) request — the fail-open fallback to Anthropic.

    ``fallback_upstream is None`` marks the plain relay path (no rewrite, nothing
    to fall back to); the caller treats it byte-identically to the old
    capture-only proxy.
    """

    primary_upstream: str
    primary_body: bytes | None
    fallback_upstream: str | None
    fallback_body: bytes | None
    routed_label: str | None


def plan_route(
    method: str,
    path: str,
    req_body: bytes | None,
    *,
    route_utility: bool,
    anthropic_upstream: str,
    local_upstream: str,
    hook_apply=_hook_apply,
    log=None,
) -> RoutePlan:
    """Decide the primary upstream + body and the fail-open fallback — pure, no I/O.

    Utility housekeeping (per the router hook) is pinned to ``local_upstream``
    with the ORIGINAL Anthropic request preserved as the fallback, so a local
    outage transparently degrades to a normal Anthropic response. Every other
    request — and any hook/parse error — yields the plain Anthropic relay with no
    fallback. Never raises: a routing bug must never fail a user's request.
    """
    if route_utility and req_body:
        try:
            parsed = json.loads(req_body)
            new_body, label = hook_apply(method, path, parsed)
            if label is not None:
                return RoutePlan(
                    primary_upstream=local_upstream,
                    primary_body=json.dumps(new_body).encode(),
                    fallback_upstream=anthropic_upstream,
                    fallback_body=req_body,          # ORIGINAL, unrewritten body+model
                    routed_label=label,
                )
        except Exception as exc:  # fail-open: any hook/parse error -> normal relay
            if log is not None:
                log(f"{exc}; passing through unrouted")
    return RoutePlan(
        primary_upstream=anthropic_upstream,
        primary_body=req_body,
        fallback_upstream=None,
        fallback_body=None,
        routed_label=None,
    )


def attempts_for(plan: RoutePlan) -> list[tuple[str, bytes | None]]:
    """Ordered (upstream, body) attempts: primary first, then the Anthropic
    fail-open fallback when the request was locally pinned."""
    attempts = [(plan.primary_upstream, plan.primary_body)]
    if plan.fallback_upstream is not None:
        attempts.append((plan.fallback_upstream, plan.fallback_body))
    return attempts


@dataclass
class AcquireResult:
    response: object | None       # live upstream response to stream, or None
    used_fallback: bool           # True iff a local failure fell open to Anthropic
    error: BaseException | None   # last connection error when every attempt failed
    sent_body: bytes | None       # body that produced ``response`` (for the capture)


async def acquire_upstream(attempts, send) -> AcquireResult:
    """Try each (upstream, body) in order, falling through on a local failure — a
    connection error/timeout (raised by ``send``) or a 5xx status — to the next
    attempt, except on the last one. ``send(upstream, body)`` is an injected
    coroutine returning a response with ``.status`` and an awaitable
    ``.release()``; all network I/O lives there so this decision loop is
    unit-testable without a live server. Returns the response to relay (or None
    when every attempt failed to connect).
    """
    used_fallback = False
    error = None
    for idx, (upstream, body) in enumerate(attempts):
        is_last = idx == len(attempts) - 1
        try:
            resp = await send(upstream, body)
        except Exception as exc:  # connection error / timeout
            error = exc
            if is_last:
                break
            used_fallback = True   # local unreachable -> fall open to Anthropic
            continue
        if resp.status >= 400 and not is_last:
            # Any 4xx/5xx from a non-last attempt is the LOCAL/rewritten rung
            # (the Anthropic fallback is always last). Fall open with the
            # original body rather than relay a local error to the user.
            await resp.release()
            used_fallback = True
            error = None
            continue
        return AcquireResult(resp, used_fallback, None, body)
    return AcquireResult(None, used_fallback, error, None)


# ── WS2 routing mode helpers (pure; unit-tested in tests/test_router_proxy.py) ─
# These only run when app["route_user_turns"] is True. The capture-only path
# (:4000) never touches any of this — it is gated behind the flag in handle().


def liveplan_attempts(live_router, plan, body: dict) -> list[tuple[str, bytes | None]]:
    """Turn a ``router.live_router.RoutePlan`` into the ordered (upstream, body)
    attempt list ``acquire_upstream`` consumes: the primary rung first, then the
    Anthropic fail-open (present only for local rungs). Bodies are built with the
    router's own model-rewrite helper, so a ``None`` model forwards the ORIGINAL
    body unchanged (frontier / passthrough / the local fail-open)."""
    attempts = [(
        plan.primary_upstream,
        live_router.build_forward_body(body, plan.primary_model),
    )]
    if plan.fallback_upstream is not None:
        attempts.append((
            plan.fallback_upstream,
            live_router.build_forward_body(body, plan.fallback_model),
        ))
    return attempts


def session_id_for(body: dict) -> str:
    """The session key EXACTLY as ``LiveRouter`` computes it, so the escalation
    controller and the memory recorder land on the same session the router
    recorded under: ``discriminator.session_key`` first, else the router's
    ``_fallback_sid``. Fail-safe to a constant so a bad body never breaks the
    relay."""
    try:
        from router import discriminator
        sid = discriminator.session_key(body)
        if sid:
            return sid
    except Exception:
        pass
    try:
        from router.live_router import _fallback_sid
        return _fallback_sid(body)
    except Exception:
        return "anon:unknown"


def last_user_content(body: dict):
    """The content of the last user message — the inbound tool_results a
    continuation carries, fed to the edit/no-progress/interrupt trip-wires."""
    messages = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return message.get("content")
    return None


def reconstruct_blocks(raw: bytes) -> tuple[list[dict], int]:
    """Reconstruct the assistant content blocks + output_tokens from a fully
    accumulated ``/v1/messages`` response — an SSE stream OR a plain JSON body.
    This is what the escalation controller observes. Fail-safe: any error yields
    ``([], 0)`` (the trip-wires simply see nothing and never escalate)."""
    if not raw:
        return [], 0
    text = raw.decode("utf-8", errors="replace")
    stripped = text.lstrip()
    if stripped.startswith("{"):                      # non-streaming JSON response
        try:
            payload = json.loads(stripped)
            content = payload.get("content")
            blocks = [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []
            usage = payload.get("usage")
            out = int(usage.get("output_tokens") or 0) if isinstance(usage, dict) else 0
            return blocks, out
        except Exception:
            return [], 0
    return _reconstruct_sse(text)


def _reconstruct_sse(text: str) -> tuple[list[dict], int]:
    """Rebuild content blocks from an Anthropic SSE stream. tool_use inputs are
    assembled from their ``input_json_delta`` partials; if the accumulated JSON
    is malformed (a local model emitting bad tool calls) the raw string is kept
    as ``input`` so the parse_schema trip-wire sees a non-dict and strikes."""
    blocks_by_index: dict[int, dict] = {}
    json_buffers: dict[int, list[str]] = {}
    output_tokens = 0
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            evt = json.loads(payload)
        except Exception:
            continue
        etype = evt.get("type")
        if etype == "message_start":
            usage = (evt.get("message") or {}).get("usage") or {}
            output_tokens = max(output_tokens, int(usage.get("output_tokens") or 0))
        elif etype == "content_block_start":
            idx, cb = evt.get("index"), evt.get("content_block")
            if isinstance(idx, int) and isinstance(cb, dict):
                blocks_by_index[idx] = dict(cb)
                json_buffers[idx] = []
        elif etype == "content_block_delta":
            idx = evt.get("index")
            if not isinstance(idx, int):
                continue
            delta = evt.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                blk = blocks_by_index.get(idx)
                if blk is not None:
                    blk["text"] = (blk.get("text") or "") + str(delta.get("text") or "")
            elif dtype == "input_json_delta":
                json_buffers.setdefault(idx, []).append(str(delta.get("partial_json") or ""))
        elif etype == "message_delta":
            usage = evt.get("usage") or {}
            output_tokens = max(output_tokens, int(usage.get("output_tokens") or 0))
    for idx, blk in blocks_by_index.items():
        if blk.get("type") != "tool_use":
            continue
        raw_json = "".join(json_buffers.get(idx, []))
        if raw_json.strip():
            try:
                blk["input"] = json.loads(raw_json)
            except Exception:
                blk["input"] = raw_json           # malformed -> parse trip-wire fires
        elif "input" not in blk:
            blk["input"] = {}
    return [blocks_by_index[i] for i in sorted(blocks_by_index)], output_tokens


async def handle(request: web.Request) -> web.StreamResponse:
    global _seq
    _seq += 1
    seq = _seq
    started = time.time()
    app = request.app

    req_body = await request.read()
    fwd_headers = {
        k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP
    }
    # Force plaintext from upstream (aiohttp would otherwise inject gzip and the
    # captured bodies become binary). Identity is always acceptable to clients.
    fwd_headers["Accept-Encoding"] = "identity"

    # Capture-fields for the routing (WS2) path; None in capture-only mode.
    routed_rung = route_layer = route_id = None
    live_plan = None
    parsed_body: dict = {}
    session_id = None

    if app["route_user_turns"]:
        # ── WS2 LIVE routing: user_turn/continuation -> local/cheap/frontier ──
        # A dedicated routing-proxy instance ONLY (flag-gated). The LiveRouter is
        # the pure decision layer; acquire_upstream provides the SAME fail-open
        # relay as P1-A (local failure -> ORIGINAL body -> Anthropic).
        live_router = app["live_router"]
        try:
            parsed_body = json.loads(req_body) if req_body else {}
            if not isinstance(parsed_body, dict):
                parsed_body = {}
        except Exception:
            parsed_body = {}
        live_plan = live_router.plan(request.method, str(request.rel_url), parsed_body)
        routed_rung = live_plan.rung
        route_layer = live_plan.layer
        route_id = live_plan.route_id          # minted by the router's recorder
        routed_label = None                    # P1-A field is N/A in routing mode
        attempts = liveplan_attempts(live_router, live_plan, parsed_body)
        session_id = session_id_for(parsed_body)
        local_upstream = live_router.litellm_url
        # A user turn starts a fresh trip-wire window (per-turn strike reset).
        if live_plan.label == "user_turn":
            try:
                app["escalation"].new_turn(session_id)
            except Exception:
                pass
    else:
        # ── P1-A live utility pinning (TRUE fail-open) ──────────────────────
        # Only when explicitly enabled AND the router hook fires. Rewrites utility
        # housekeeping to the local rung via LiteLLM; on any local failure the
        # ORIGINAL Anthropic request is retried transparently. Everything else is
        # byte-identical to the plain capture relay.
        plan = plan_route(
            request.method, str(request.rel_url), req_body,
            route_utility=app["route_utility"],
            anthropic_upstream=UPSTREAM,
            local_upstream=app["local_upstream"],
            hook_apply=_hook_apply,
            log=lambda msg: print(f"[hook-error seq={seq}] {msg}"),
        )
        routed_label = plan.routed_label
        attempts = attempts_for(plan)
        local_upstream = app["local_upstream"]

    primary_body_for_capture = attempts[0][1]
    client: ClientSession = app["client"]

    async def _send(upstream_base: str, body: bytes | None):
        upstream_url = upstream_base + request.rel_url.raw_path
        if request.rel_url.query_string:
            upstream_url += "?" + request.rel_url.query_string
        kwargs: dict = {}
        # Bound the LOCAL attempt so a hung-but-listening LiteLLM (accepts TCP,
        # never answers) raises -> fail-open fallback, instead of hanging forever.
        # Utility calls are tiny and the model is pre-warmed, so LOCAL_ATTEMPT_TIMEOUT
        # is generous. Anthropic (and all non-routed) traffic keeps the unbounded
        # stream timeout — streams run long.
        if upstream_base == local_upstream:
            kwargs["timeout"] = ClientTimeout(total=LOCAL_ATTEMPT_TIMEOUT, connect=10)
        return await client.request(
            request.method, upstream_url, headers=fwd_headers,
            data=body if body else None, allow_redirects=False, **kwargs,
        )

    result = await acquire_upstream(attempts, _send)
    local_fallback = result.used_fallback
    if local_fallback:
        print(f"[local-fallback seq={seq}] local rung failed -> Anthropic (original body)")

    upstream_resp = result.response
    chunks: list[bytes] = []
    resp_headers: dict[str, str] = {}
    try:
        if upstream_resp is None:  # every attempt (incl. any fallback) failed to connect
            raise result.error if result.error else RuntimeError("no upstream response")
        resp_headers = {
            k: v for k, v in upstream_resp.headers.items() if k.lower() not in HOP_BY_HOP
        }
        response = web.StreamResponse(status=upstream_resp.status, headers=resp_headers)
        await response.prepare(request)
        # Relay verbatim, chunk by chunk — SSE streams stay live.
        async for chunk in upstream_resp.content.iter_any():
            chunks.append(chunk)
            await response.write(chunk)
        await response.write_eof()
        status = upstream_resp.status
    except Exception as exc:  # upstream failure — surface honestly, log it
        status = 502
        response = web.json_response(
            {"type": "error", "error": {"type": "proxy_upstream_error", "message": str(exc)}},
            status=502,
        )
        chunks = [json.dumps({"proxy_error": str(exc)}).encode()]
    finally:
        if upstream_resp is not None:
            await upstream_resp.release()

    # ── WS2 post-call: observe the outcome + record it to the flywheel ───────
    # For a user_turn/continuation, reconstruct the assistant blocks from the
    # bytes we already relayed, feed the escalation trip-wires (which may flip
    # the session's sticky route for the NEXT request), and attach the outcome
    # to the route_id the router minted. All wrapped — a recording failure never
    # touches the response the client already has.
    if (app["route_user_turns"] and live_plan is not None
            and live_plan.label in ("user_turn", "continuation")):
        latency_ms = (time.time() - started) * 1000.0
        try:
            blocks, out_tokens = reconstruct_blocks(b"".join(chunks))
        except Exception:
            blocks, out_tokens = [], 0
        decision = None
        try:
            escalation = app["escalation"]
            inbound = last_user_content(parsed_body)   # tool_results on a continuation
            if inbound:
                escalation.observe_tool_results(session_id, inbound)
            decision = escalation.observe_response(
                session_id, blocks, output_tokens=out_tokens)
        except Exception:
            decision = None
        try:
            outcome_cls = app["outcome_cls"]
            if outcome_cls is not None:
                # Derive `escalated` from the ACCUMULATED session state, not the
                # single observation (review finding): once any continuation in
                # this turn trips a wire, escalated_this_episode stays True, so a
                # mid-turn escalation survives to the final outcome row. attach_
                # outcome now allows same-rank last-write-wins, so each
                # continuation refreshes the working row rather than being dropped.
                try:
                    accumulated = bool(app["escalation"].store.get_session(
                        session_id).escalated_this_episode)
                except Exception:
                    accumulated = False
                escalated = accumulated or bool(getattr(decision, "escalate", False))
                outcome = outcome_cls(
                    status="closed_turn",
                    escalated=escalated,
                    tripwire_name=getattr(decision, "tripwire", None),
                    tripwire_type=getattr(decision, "tripwire_type", None),
                    output_tokens=int(out_tokens),
                    latency_ms=float(latency_ms),
                )
                app["recorder"].attach(session_id, outcome)
        except Exception:
            pass

    # Capture AFTER the client has its bytes — logging never delays the session.
    # request_body reflects what actually produced the response (rewritten body on
    # a local hit, ORIGINAL body when we fell open to Anthropic).
    captured_body = result.sent_body if result.sent_body is not None else primary_body_for_capture
    try:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "seq": seq,
            "method": request.method,
            "path": str(request.rel_url),
            "status": status,
            "latency_ms": round((time.time() - started) * 1000),
            "request_headers": _redacted(request.headers),
            "request_body": (captured_body or b"").decode("utf-8", errors="replace"),
            "response_headers": _redacted(resp_headers) if status != 502 else {},
            "response_body": b"".join(chunks).decode("utf-8", errors="replace"),
            "routed_to_local": routed_label,   # P1-A: which utility label was pinned, or null
            "local_fallback": local_fallback,  # P1-A / WS2: True iff local failed -> Anthropic fallback
            "routed_rung": routed_rung,        # WS2: rung the LiveRouter chose (null in capture-only)
            "route_layer": route_layer,        # WS2: which policy layer decided
            "route_id": route_id,              # WS2: memory route_id for this decision
        }
        out = _capture_dir(app["capture_root"]) / f"{int(started * 1000)}-{seq:06d}.json"
        out.write_text(json.dumps(record))
    except Exception as exc:
        print(f"[capture-error] seq={seq}: {exc}")

    return response


async def make_app(capture_root: Path, route_utility: bool = False,
                   local_upstream: str = "http://localhost:4001",
                   route_user_turns: bool = False,
                   memory_db: str = "data/router-memory.db",
                   decision_source: str = "organic",
                   live_router=None, escalation=None,
                   recorder=None) -> web.Application:
    """Build the proxy app. ``route_user_turns`` is the WS2 flag: OFF (default,
    :4000) keeps every path byte-identical to the capture-only proxy and never
    constructs any router live-decision object. ON (the dedicated routing-proxy
    instance, :4002) wires ONE shared LiveRouter + EscalationController +
    RoutingRecorder over ONE DictSessionStore. The live objects may be injected
    (tests); otherwise they are built here from the real modules."""
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app["capture_root"] = capture_root
    app["route_utility"] = route_utility     # P1-A live toggle (off by default)
    app["local_upstream"] = local_upstream
    app["route_user_turns"] = route_user_turns  # WS2 live routing (off by default)
    app["live_router"] = None
    app["escalation"] = None
    app["recorder"] = None
    app["session_store"] = None
    app["outcome_cls"] = None

    if route_user_turns:
        # Build (or accept injected) the ONE-per-process routing singletons. All
        # three share ONE session store so the router's continuation-stickiness,
        # the controller's escalation flips, and the memory recording agree.
        if live_router is None or escalation is None or recorder is None:
            from router.live_router import LiveRouter
            from router.escalation_controller import EscalationController
            from router.memory_facade import RouterMemory, RoutingRecorder
            from router.state import DictSessionStore
            try:
                from router.llm_classifier import classify_intent_llm as classifier
            except Exception:
                classifier = None
            try:
                from router.memory_sqlite import SqliteProvider
                from router.memory_null import NullProvider
                providers = [SqliteProvider(Path(memory_db)), NullProvider()]
            except Exception:
                providers = None
            store = DictSessionStore()
            recorder = RoutingRecorder(RouterMemory(providers))
            escalation = EscalationController(store)
            live_router = LiveRouter(store=store, memory=recorder,
                                     classifier=classifier, source=decision_source)
            app["session_store"] = store
        app["live_router"] = live_router
        app["escalation"] = escalation
        app["recorder"] = recorder
        try:
            from router.memory_ports import OutcomeEvent
            app["outcome_cls"] = OutcomeEvent
        except Exception:
            app["outcome_cls"] = None

    app["client"] = ClientSession(
        connector=TCPConnector(limit=64),
        timeout=ClientTimeout(total=None, connect=30),  # streams run long — no total cap
        auto_decompress=False,
    )

    async def close_client(app):
        await app["client"].close()

    app.on_cleanup.append(close_client)
    app.router.add_route("*", "/{tail:.*}", handle)
    return app


def main() -> None:
    global UPSTREAM
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=4000)
    ap.add_argument("--captures", type=Path, default=Path("data/captures"))
    ap.add_argument("--upstream", default=UPSTREAM,
                    help="where to relay (default Anthropic; set to the LiteLLM proxy for local-rung capture)")
    ap.add_argument("--route-utility", action="store_true",
                    help="P1-A LIVE: pin utility housekeeping to the local rung (fail-open)")
    ap.add_argument("--local-upstream", default="http://localhost:4001",
                    help="LiteLLM execution layer for local-routed utility calls")
    ap.add_argument("--route-user-turns", action="store_true",
                    help="WS2 LIVE: route user_turns/continuations across local/cheap/frontier "
                         "with live escalation + outcome recording (dedicated routing-proxy)")
    ap.add_argument("--memory-db", default="data/router-memory.db",
                    help="WS2 routing-decision memory (SqliteProvider) path")
    ap.add_argument("--decision-source", default="organic",
                    choices=["organic", "simulator"],
                    help="provenance stamped on every recorded decision from this "
                         "instance: 'simulator' when the synthetic driver points here "
                         "(fuel), 'organic' for a user's own opt-in sessions")
    args = ap.parse_args()
    UPSTREAM = args.upstream
    args.captures.mkdir(parents=True, exist_ok=True)
    if args.route_user_turns:
        mode = "LIVE route user_turns"
    elif args.route_utility:
        mode = "LIVE utility->local"
    else:
        mode = "capture-only"
    print(f"capture proxy :{args.port} -> {UPSTREAM}  [{mode}]  (captures -> {args.captures}/)")
    if args.route_utility and not args.route_user_turns:
        print(f"  utility housekeeping -> {args.local_upstream} (LiteLLM, cloud-fallback); "
              "kill switch: unset ANTHROPIC_BASE_URL or restart without --route-utility")
    if args.route_user_turns:
        print(f"  user_turns -> local rung on {args.local_upstream} (fail-open to Anthropic); "
              f"cheap/frontier -> Anthropic; memory -> {args.memory_db}")
        print("  kill switch: unset ANTHROPIC_BASE_URL or restart without --route-user-turns")
    print(f"use:  ANTHROPIC_BASE_URL=http://localhost:{args.port} claude")
    web.run_app(make_app(args.captures, route_utility=args.route_utility,
                         local_upstream=args.local_upstream,
                         route_user_turns=args.route_user_turns,
                         memory_db=args.memory_db,
                         decision_source=args.decision_source),
                port=args.port, print=None)


if __name__ == "__main__":
    main()
