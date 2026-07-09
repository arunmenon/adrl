#!/bin/bash
# Flywheel: run the simulator THROUGH the live routing proxy so the transaction
# memory fills with correctly-tagged (source=simulator) decisions + outcomes.
#
# This exists because the fuel run needs TWO settings to agree that live on
# separate processes: the proxy's decision provenance (DECISION_SOURCE) and the
# simulator's target (--proxy). Launched by hand they drift, and synthetic fuel
# gets mis-recorded as organic - defeating WS3/WS4's source separation. This
# script pins both, so they can't disagree.
#
# Budget: the simulator enforces its own $25 hard cap from data/sim-ledger.jsonl.
# Usage: tools/run_flywheel.sh                 # 1 stochastic episode
#        RUNS=3 tools/run_flywheel.sh          # 3 episodes
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-4002}"
RUNS="${RUNS:-1}"
EMBED_URL="${EMBED_URL:-http://localhost:11434/v1/embeddings}"

# The retrieval flywheel is worthless without embeddings, and the embedder
# degrades SILENTLY (fail-open) when the OpenAI-compat /v1/embeddings route is
# missing - decisions would record with no vector and WS4 would quietly have no
# neighbors. Probe once up front and refuse loudly rather than log nothing.
if ! curl -s -m 8 "${EMBED_URL}" \
      -H 'content-type: application/json' \
      -d '{"model":"nomic-embed-text","input":["probe"]}' \
      | grep -q '"embedding"'; then
  echo "embedder probe FAILED at ${EMBED_URL}: decisions would record with no vector." >&2
  echo "start ollama (or point EMBED_URL at your /v1/embeddings server) and retry." >&2
  exit 1
fi
echo "embedder OK at ${EMBED_URL}"

# Start (or reuse) the routing proxy, TAGGED simulator. run_router_proxy.sh is
# idempotent and refuses to start without the local execution stack (:4001).
DECISION_SOURCE=simulator PORT="${PORT}" bash tools/run_router_proxy.sh

# Run the simulator AGAINST that proxy. --proxy is pinned to the routing port
# here (not the default :4000 capture-only proxy), so the pairing is guaranteed.
echo "running ${RUNS} simulator episode(s) -> :${PORT} (source=simulator)"
PYTHONPATH=src .venv/bin/python -m simulator.run_session \
  --episode stochastic --runs "${RUNS}" \
  --proxy "http://localhost:${PORT}" --models sonnet "$@"
