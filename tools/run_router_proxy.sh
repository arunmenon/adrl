#!/bin/bash
# WS2 — start the LIVE routing proxy on :4002 (a SECOND, dedicated instance),
# detached (survives terminal close). This one ROUTES: user_turns/continuations
# go local->cheap->frontier with live escalation + outcome recording. The
# capture-only proxy on :4000 (tools/run_proxy.sh) is left untouched.
#
# Launch a routed Claude Code session with:
#   ANTHROPIC_BASE_URL=http://localhost:4002 claude
# Kill switch: unset ANTHROPIC_BASE_URL (sessions go direct); stop with:
#   kill $(cat data/router-proxy.pid)
#
# Local routing needs the execution stack (ollama + LiteLLM :4001). The proxy
# fails open to Anthropic if that stack is unreachable, but we refuse to *start*
# in routing mode without it so the flip is deliberate, not silent.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-4002}"
CAPTURES="${CAPTURES:-data/captures-routed}"
MEMORY_DB="${MEMORY_DB:-data/router-memory.db}"
LOCAL_UPSTREAM="${LOCAL_UPSTREAM:-http://localhost:4001}"
# Provenance stamped on every decision this instance records. Default 'organic'
# (a user's own opt-in sessions); launch with DECISION_SOURCE=simulator when the
# synthetic driver points here so sim fuel stays separable from the real pool.
DECISION_SOURCE="${DECISION_SOURCE:-organic}"
# Validate at the operator entry point: a typo'd source would silently label
# rows into a cohort WS3/WS4 filters neither match nor error on.
case "${DECISION_SOURCE}" in
  organic|simulator) ;;
  *) echo "DECISION_SOURCE must be 'organic' or 'simulator' (got '${DECISION_SOURCE}')" >&2
     exit 1 ;;
esac

# Health-check the actual listening PORT, not just the pid. A wedged-but-alive
# process (pid up, event loop hung) still holds the pid file and passes kill -0
# but never answers on the port — so we curl the port and require a completed
# HTTP response before trusting an existing proxy.
port_answers() {
  curl -s -o /dev/null -m 4 "http://localhost:${PORT}/" >/dev/null 2>&1
}

if [ -f data/router-proxy.pid ] && kill -0 "$(cat data/router-proxy.pid)" 2>/dev/null; then
  RUNNING_PID="$(cat data/router-proxy.pid)"
  RUNNING_SOURCE="$(cat data/router-proxy.source 2>/dev/null || echo unknown)"
  # Reuse ONLY when the live proxy is tagged with the SAME source requested.
  # Reusing a proxy tagged differently (e.g. an organic :4002 while the flywheel
  # asks for simulator) would silently mis-record decisions under the wrong
  # provenance - the exact hazard the source tag exists to prevent - so a
  # mismatch recycles instead of short-circuiting.
  if port_answers && [ "${RUNNING_SOURCE}" = "${DECISION_SOURCE}" ]; then
    echo "router proxy already running (pid ${RUNNING_PID}), port ${PORT} answering, source ${RUNNING_SOURCE}"
    exit 0
  fi
  if port_answers; then
    echo "proxy on ${PORT} is source='${RUNNING_SOURCE}' but '${DECISION_SOURCE}' requested - recycling" >&2
  else
    echo "router proxy pid ${RUNNING_PID} alive but port ${PORT} not answering - recycling" >&2
  fi
  kill "${RUNNING_PID}" 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 "${RUNNING_PID}" 2>/dev/null || break
    sleep 0.3
  done
  kill -9 "${RUNNING_PID}" 2>/dev/null || true
  rm -f data/router-proxy.pid data/router-proxy.source
fi

# Refuse to start in routing mode without the local execution stack answering.
if ! curl -s -o /dev/null -m 4 "${LOCAL_UPSTREAM}/health/liveliness" 2>/dev/null; then
  echo "LiteLLM ${LOCAL_UPSTREAM} not answering — start tools/run_litellm.sh first" >&2
  exit 1
fi

mkdir -p data "${CAPTURES}"
echo "LIVE user-turn routing ENABLED (user_turns -> local rung, fail-open to Anthropic)"
echo "decision provenance: ${DECISION_SOURCE}"
PYTHONPATH=src nohup .venv/bin/python -m proxy.capture_proxy \
  --port "${PORT}" --captures "${CAPTURES}" \
  --route-user-turns --local-upstream "${LOCAL_UPSTREAM}" --memory-db "${MEMORY_DB}" \
  --decision-source "${DECISION_SOURCE}" \
  > data/router-proxy.log 2>&1 &
echo $! > data/router-proxy.pid
echo "${DECISION_SOURCE}" > data/router-proxy.source   # for the reuse source-match check
sleep 1
if kill -0 "$(cat data/router-proxy.pid)" 2>/dev/null; then
  echo "router proxy running (pid $(cat data/router-proxy.pid)), log: data/router-proxy.log"
  echo "launch routed sessions with: ANTHROPIC_BASE_URL=http://localhost:${PORT} claude"
else
  echo "router proxy failed to start — see data/router-proxy.log" >&2
  exit 1
fi
