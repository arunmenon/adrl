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
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

UPSTREAM = "https://api.anthropic.com"
REDACT_HEADERS = {"authorization", "x-api-key", "cookie", "set-cookie"}
# Hop-by-hop headers must not be forwarded either direction.
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
    "accept-encoding",  # ask upstream for identity so captured bodies are plaintext
}

_seq = 0


def _capture_dir(root: Path) -> Path:
    d = root / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _redacted(headers) -> dict[str, str]:
    return {
        k: ("<redacted>" if k.lower() in REDACT_HEADERS else v)
        for k, v in headers.items()
    }


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

    upstream_url = UPSTREAM + request.rel_url.raw_path
    if request.rel_url.query_string:
        upstream_url += "?" + request.rel_url.query_string

    client: ClientSession = app["client"]
    chunks: list[bytes] = []
    try:
        async with client.request(
            request.method, upstream_url, headers=fwd_headers,
            data=req_body if req_body else None, allow_redirects=False,
        ) as upstream:
            resp_headers = {
                k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP
            }
            response = web.StreamResponse(status=upstream.status, headers=resp_headers)
            await response.prepare(request)
            # Relay verbatim, chunk by chunk — SSE streams stay live.
            async for chunk in upstream.content.iter_any():
                chunks.append(chunk)
                await response.write(chunk)
            await response.write_eof()
            status = upstream.status
    except Exception as exc:  # upstream failure — surface honestly, log it
        status = 502
        response = web.json_response(
            {"type": "error", "error": {"type": "proxy_upstream_error", "message": str(exc)}},
            status=502,
        )
        chunks = [json.dumps({"proxy_error": str(exc)}).encode()]

    # Capture AFTER the client has its bytes — logging never delays the session.
    try:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "seq": seq,
            "method": request.method,
            "path": str(request.rel_url),
            "status": status,
            "latency_ms": round((time.time() - started) * 1000),
            "request_headers": _redacted(request.headers),
            "request_body": req_body.decode("utf-8", errors="replace"),
            "response_headers": _redacted(resp_headers) if status != 502 else {},
            "response_body": b"".join(chunks).decode("utf-8", errors="replace"),
        }
        out = _capture_dir(app["capture_root"]) / f"{int(started * 1000)}-{seq:06d}.json"
        out.write_text(json.dumps(record))
    except Exception as exc:
        print(f"[capture-error] seq={seq}: {exc}")

    return response


async def make_app(capture_root: Path) -> web.Application:
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app["capture_root"] = capture_root
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=4000)
    ap.add_argument("--captures", type=Path, default=Path("data/captures"))
    args = ap.parse_args()
    args.captures.mkdir(parents=True, exist_ok=True)
    print(f"capture proxy :{args.port} -> {UPSTREAM}  (captures -> {args.captures}/)")
    print(f"use:  ANTHROPIC_BASE_URL=http://localhost:{args.port} claude")
    web.run_app(make_app(args.captures), port=args.port, print=None)


if __name__ == "__main__":
    main()
