#!/bin/bash
# Start the ollama server detached (serves the local rung on :11434).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "ollama already serving on :11434"; exit 0
fi
mkdir -p data
nohup ollama serve > data/ollama.log 2>&1 &
echo $! > data/ollama.pid
sleep 3
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "ollama serving on :11434 (pid $(cat data/ollama.pid))"
  curl -s http://localhost:11434/api/tags | \
    .venv/bin/python -c "import json,sys; print('models:', [m['name'] for m in json.load(sys.stdin).get('models',[])])" 2>/dev/null || true
else
  echo "ollama failed to start — see data/ollama.log" >&2; exit 1
fi
