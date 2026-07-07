#!/bin/bash
# B3 — start the wire-capture proxy detached (survives terminal close).
# Then launch Claude Code sessions with:
#   ANTHROPIC_BASE_URL=http://localhost:4000 claude
# Kill switch: unset ANTHROPIC_BASE_URL (sessions go direct); stop proxy with:
#   kill $(cat data/proxy.pid)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-4000}"

# Health-check the actual listening PORT, not just the pid. A wedged-but-alive
# process (pid up, event loop hung) still holds the pid file and passes kill -0,
# but never answers on the port — so we curl the port and require a completed
# HTTP response before trusting an existing proxy. If the port doesn't answer,
# the stale process is killed and a fresh one started.
port_answers() {
  curl -s -o /dev/null -m 4 "http://localhost:${PORT}/" >/dev/null 2>&1
}

if [ -f data/proxy.pid ] && kill -0 "$(cat data/proxy.pid)" 2>/dev/null; then
  if port_answers; then
    echo "proxy already running (pid $(cat data/proxy.pid)), port ${PORT} answering"
    exit 0
  fi
  STALE_PID="$(cat data/proxy.pid)"
  echo "proxy pid ${STALE_PID} alive but port ${PORT} not answering — recycling" >&2
  kill "${STALE_PID}" 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 "${STALE_PID}" 2>/dev/null || break
    sleep 0.3
  done
  kill -9 "${STALE_PID}" 2>/dev/null || true
  rm -f data/proxy.pid
fi

mkdir -p data
# Capture-only by default — NO --route-utility. Live utility pinning (P1-A) is
# opt-in and must be enabled deliberately; the default here never rewrites traffic.
PYTHONPATH=src nohup .venv/bin/python -m proxy.capture_proxy --port "${PORT}" \
  > data/proxy.log 2>&1 &
echo $! > data/proxy.pid
sleep 1
if kill -0 "$(cat data/proxy.pid)" 2>/dev/null; then
  echo "proxy running (pid $(cat data/proxy.pid)), log: data/proxy.log"
  echo "launch sessions with: ANTHROPIC_BASE_URL=http://localhost:${PORT} claude"
else
  echo "proxy failed to start — see data/proxy.log" >&2
  exit 1
fi
