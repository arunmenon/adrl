#!/bin/bash
# B3 — start the wire-capture proxy detached (survives terminal close).
# Then launch Claude Code sessions with:
#   ANTHROPIC_BASE_URL=http://localhost:4000 claude
# Kill switch: unset ANTHROPIC_BASE_URL (sessions go direct); stop proxy with:
#   kill $(cat data/proxy.pid)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f data/proxy.pid ] && kill -0 "$(cat data/proxy.pid)" 2>/dev/null; then
  echo "proxy already running (pid $(cat data/proxy.pid))"
  exit 0
fi

mkdir -p data
PYTHONPATH=src nohup .venv/bin/python -m proxy.capture_proxy --port "${PORT:-4000}" \
  > data/proxy.log 2>&1 &
echo $! > data/proxy.pid
sleep 1
if kill -0 "$(cat data/proxy.pid)" 2>/dev/null; then
  echo "proxy running (pid $(cat data/proxy.pid)), log: data/proxy.log"
  echo "launch sessions with: ANTHROPIC_BASE_URL=http://localhost:${PORT:-4000} claude"
else
  echo "proxy failed to start — see data/proxy.log" >&2
  exit 1
fi
