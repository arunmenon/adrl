#!/bin/bash
# Start the LiteLLM execution layer detached on :4001.
# Local rung served by ollama (must be running — tools/run_ollama.sh);
# cloud rungs need data/anthropic-key.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f data/litellm.pid ] && kill -0 "$(cat data/litellm.pid)" 2>/dev/null; then
  echo "litellm already running (pid $(cat data/litellm.pid))"; exit 0
fi

# ollama must be up for the local rung
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "ollama not responding on :11434 — run tools/run_ollama.sh first" >&2; exit 1
fi

# cloud rungs read the key from the gitignored file
if [ -f data/anthropic-key ]; then
  export ANTHROPIC_API_KEY="$(cat data/anthropic-key)"
fi

mkdir -p data
PYTHONPATH=src nohup .venv/bin/litellm \
  --config config/litellm-local.yaml --port "${LITELLM_PORT:-4001}" \
  > data/litellm.log 2>&1 &
echo $! > data/litellm.pid
sleep 6
if kill -0 "$(cat data/litellm.pid)" 2>/dev/null && \
   curl -s "http://localhost:${LITELLM_PORT:-4001}/health/liveliness" >/dev/null 2>&1; then
  echo "litellm running (pid $(cat data/litellm.pid)) on :${LITELLM_PORT:-4001}, log: data/litellm.log"
else
  echo "litellm failed to come up — see data/litellm.log" >&2
  tail -5 data/litellm.log >&2 || true
  exit 1
fi
