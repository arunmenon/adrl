# P1-A live telemetry finding: utility pinning is largely infeasible for long sessions

**Date:** 2026-07-07 · **How found:** telemetry over live-routing captures (the reason to instrument)

## What live routing revealed

Flipped P1-A live; telemetry over the current period showed **120 utility calls, 0 served locally, 100% fell open to cloud**. Fail-open worked perfectly (users got answers), but the local rung served nothing — so routing was net-NEGATIVE: every utility call wasted ~45s attempting local before falling back.

## Root cause

Real `utility:sidecar` calls are **small-output but huge-input**. A title/topic/summary sidecar on a long session carries the entire conversation: observed **652KB (~160k tokens)** with `max_tokens=64`, 2 messages. The discriminator labels them by SHAPE (no tools, tiny max_tokens, ≤2 messages) — correctly — but that shape says nothing about input size. LiteLLM's pre-call check rejects the oversized request (`RouterRateLimitError: No deployments available for local-small`), the call burns the 45s local timeout, then falls back.

Measured on a recent 2,000-call window: **0% of utility:sidecar calls fit** the local rung's budget. Across all history only ~0.7% do (early-session title calls on short conversations).

## Fix

Added a **feasibility gate** to the hook (design §5.3 principle the simple P1-A hook skipped): only pin a utility call to local if its input fits the local rung (~48KB budget for local-small). Oversized calls pass through to cloud with no wasted local attempt. Tests: tests/test_hook_feasibility.py.

## The finding that matters more than the fix

**Utility-call pinning — the design's original "zero-risk, instant-savings" Phase-1 win (§4) — is largely ineffective for real long-session usage.** The premise "utility = cheap = small" is wrong: utility calls have tiny *output* but can have enormous *input*. On a 16GB Mac, a small local model cannot serve 160k-token inputs at all.

This **reinforces the Phase-0 economics finding**: the real, *feasible* lever is subagent traffic (S13 / P1-D), whose calls have genuinely small, fresh context (bounded read-only tasks) — not utility calls, which are deceptively large-context. Subagents are both the bigger dollar lever AND the feasible one.

## Recommendation

- Keep the feasibility gate (prevents the net-negative). With it, P1-A live is safe but nearly dormant for long sessions — it routes only the rare small early-session utility call. Little reason to run it live; leave capture-only.
- **Pivot Phase-1 effort to P1-D (subagent-local pilot)** — the feasible, higher-value lever. The post-call path (P1-C) it needs is already built (shadow-verified).
- Serving large-input utility calls locally would need a big-context local model + far more RAM than 16GB — out of scope here; revisit on the office M4 Max / 64GB if utility savings ever matter.
