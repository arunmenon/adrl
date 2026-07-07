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

    # ── P1-A live utility pinning (TRUE fail-open) ──────────────────────────
    # Only when explicitly enabled AND the router hook fires. Rewrites utility
    # housekeeping (titles, sidecar classifiers) to the local rung via LiteLLM;
    # on any local failure the ORIGINAL Anthropic request is retried transparently.
    # Everything else is byte-identical to the plain relay.
    plan = plan_route(
        request.method, str(request.rel_url), req_body,
        route_utility=app["route_utility"],
        anthropic_upstream=UPSTREAM,
        local_upstream=app["local_upstream"],
        hook_apply=_hook_apply,
        log=lambda msg: print(f"[hook-error seq={seq}] {msg}"),
    )
    routed_label = plan.routed_label

    client: ClientSession = app["client"]
    local_upstream = app["local_upstream"]

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

    result = await acquire_upstream(attempts_for(plan), _send)
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

    # Capture AFTER the client has its bytes — logging never delays the session.
    # request_body reflects what actually produced the response (rewritten body on
    # a local hit, ORIGINAL body when we fell open to Anthropic).
    captured_body = result.sent_body if result.sent_body is not None else plan.primary_body
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
            "local_fallback": local_fallback,  # P1-A: True iff local failed -> Anthropic fallback
        }
        out = _capture_dir(app["capture_root"]) / f"{int(started * 1000)}-{seq:06d}.json"
        out.write_text(json.dumps(record))
    except Exception as exc:
        print(f"[capture-error] seq={seq}: {exc}")

    return response


async def make_app(capture_root: Path, route_utility: bool = False,
                   local_upstream: str = "http://localhost:4001") -> web.Application:
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app["capture_root"] = capture_root
    app["route_utility"] = route_utility     # P1-A live toggle (off by default)
    app["local_upstream"] = local_upstream
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
    args = ap.parse_args()
    UPSTREAM = args.upstream
    args.captures.mkdir(parents=True, exist_ok=True)
    mode = "LIVE utility->local" if args.route_utility else "capture-only"
    print(f"capture proxy :{args.port} -> {UPSTREAM}  [{mode}]  (captures -> {args.captures}/)")
    if args.route_utility:
        print(f"  utility housekeeping -> {args.local_upstream} (LiteLLM, cloud-fallback); "
              "kill switch: unset ANTHROPIC_BASE_URL or restart without --route-utility")
    print(f"use:  ANTHROPIC_BASE_URL=http://localhost:{args.port} claude")
    web.run_app(make_app(args.captures, route_utility=args.route_utility,
                         local_upstream=args.local_upstream), port=args.port, print=None)


if __name__ == "__main__":
    main()
