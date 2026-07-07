#!/bin/bash
# Start the ollama server detached (serves the local rung on :11434) and
# pre-warm the local-small model so the first real utility call isn't a cold
# weight-load stall.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

WARM_MODEL="${WARM_MODEL:-llama3.2:latest}"

prewarm() {
  # Load the local-small model into memory with a tiny generate (num_predict=1)
  # so the first P1-A utility call hits a warm model instead of a cold load.
  # Non-fatal: a warm failure must never fail startup.
  echo "pre-warming ${WARM_MODEL} ..."
  if curl -s -m 60 http://localhost:11434/api/generate \
      -d "{\"model\":\"${WARM_MODEL}\",\"prompt\":\"ok\",\"stream\":false,\"options\":{\"num_predict\":1}}" \
      >/dev/null 2>&1; then
    echo "  ${WARM_MODEL} warm"
  else
    echo "  warn: pre-warm of ${WARM_MODEL} failed (non-fatal)" >&2
  fi
}

if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "ollama already serving on :11434"
  prewarm
  exit 0
fi
mkdir -p data
nohup ollama serve > data/ollama.log 2>&1 &
echo $! > data/ollama.pid
sleep 3
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "ollama serving on :11434 (pid $(cat data/ollama.pid))"
  curl -s http://localhost:11434/api/tags | \
    .venv/bin/python -c "import json,sys; print('models:', [m['name'] for m in json.load(sys.stdin).get('models',[])])" 2>/dev/null || true
  prewarm
else
  echo "ollama failed to start — see data/ollama.log" >&2; exit 1
fi
