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

# The retrieval flywheel is worthless without embeddings, and the embedder
# degrades SILENTLY (fail-open) if its endpoint is unreachable - decisions would
# record with no vector and WS4 would quietly have no neighbors. Probe the ACTUAL
# embedder the recorder uses (config/backends.yaml, via for_role) rather than a
# side URL, so the probe can never pass while recording silently stores nothing.
if ! PYTHONPATH=src .venv/bin/python -c \
      "import sys; from router import backends; sys.exit(0 if backends.for_role('embedder').embed(['probe']) else 1)"; then
  echo "embedder probe FAILED: for_role('embedder').embed() returned nothing - decisions" >&2
  echo "would record with no vector. Check config/backends.yaml [embedder] and that its server is up." >&2
  exit 1
fi
echo "embedder OK (config/backends.yaml embedder answered)"

# Start (or reuse) the routing proxy, TAGGED simulator. run_router_proxy.sh is
# idempotent and refuses to start without the local execution stack (:4001).
DECISION_SOURCE=simulator PORT="${PORT}" bash tools/run_router_proxy.sh

# Run the simulator AGAINST that proxy. --proxy is pinned to the routing port
# here (not the default :4000 capture-only proxy), so the pairing is guaranteed.
echo "running ${RUNS} simulator episode(s) -> :${PORT} (source=simulator)"
PYTHONPATH=src .venv/bin/python -m simulator.run_session \
  --episode stochastic --runs "${RUNS}" \
  --proxy "http://localhost:${PORT}" --models sonnet "$@"
