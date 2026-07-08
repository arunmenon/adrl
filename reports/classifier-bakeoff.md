# Classifier Bake-off: Local Routing Gate (16GB box)

**Date:** 2026-07-08
**Question:** Which small local model should serve as the ADRL routing classifier — the binary "local-eligible vs. needs-frontier" gate on the hot path?
**Benchmark:** `data/classifier-bench.jsonl` — 45 hand-labeled items (trivial=15, standard=20, hard=10; `needs_frontier` = 10 hard), single reference labeler, gitignored. Each model run 45 items × 3 repeats, one model resident at a time (ollama unloads previous on each load; `keep_alive:0` + `ollama stop` between runs). Metrics only below — no instruction payloads, safe to commit.

## Metrics

| Model | Params | Size (GB) | **Binary frontier acc** | Tier acc (4-way) | Self-consistency | Format valid | p50 (ms) | p90 (ms) | Recall (hard) | Precision |
|---|---|---|---|---|---|---|---|---|---|---|
| **qwen2.5:3b-instruct** | 3.0B | 1.9 | **0.8444** | 0.5778 | **1.000** | **1.000** | 1718 | 1899 | 0.50 | 0.714 |
| llama3.2:latest | 3.2B | 2.0 | 0.8222 | 0.4222 | 0.9778 | 0.9778 | 1925 | 2355 | — | — |
| qwen2.5:1.5b-instruct | 1.5B | 1.0 | 0.7556 | 0.4222 | 0.9778 | 0.9778 | **1139** | **1398** | 0.20 | — |
| gemma2:2b | 2.6B | 1.6 | 0.7556 | 0.5778 | 0.9111 | 0.9111 | 1733 | 2195 | 0.60 | — |

Ranked by the weighted priority order: (1) binary_frontier_accuracy, (2) self_consistency + format_valid_pct, (3) latency_p50, (4) RAM fit.

## Winner (16GB): `qwen2.5:3b-instruct`

It wins the metric that is the entire job. Binary frontier accuracy 0.8444 is the best in the field and clears the runner-up (llama3.2, 0.8222) — and, more decisively, it is the **only** model that pairs top accuracy with a perfect 1.000 self-consistency and 1.000 format validity across all 135 calls (zero malformed outputs, zero flips across repeats). For a component on the routing hot path, a deterministic, always-parseable verdict is worth as much as the raw accuracy: the other three each emitted 3–12 malformed or inconsistent responses, any of which is an undefined routing decision in production. It fits the RAM envelope comfortably (1.9GB, under the 3B/2.2GB cap, ran without OOM on the ~3.3GB free).

**Tradeoff vs. runner-up (llama3.2:latest, already local):** llama is within 2.2 points on the binary call (0.8222) and is already pulled, so it is a zero-download fallback. But it is strictly worse on every tiebreak: lower self-consistency (0.9778) and format validity (0.9778) mean occasional undefined verdicts, its 4-way tier accuracy collapses (0.4222 — it folds `standard` into `trivial`, so it cannot support any future finer-grained routing), and it is the slowest in the field (p50 1925ms / p90 2355ms). qwen2.5:3b is faster, cleaner, and more accurate at a near-identical footprint. The only reason to prefer llama is offline resilience if a fresh pull is impossible.

**Why not the others:** gemma2:2b has the best hard-task recall (0.60) but the worst format validity (0.9111, 12 malformed) and worst self-consistency, and ties for the lowest binary accuracy (0.7556) — a high-recall but unreliable gate. qwen2.5:1.5b is the latency floor (p50 1139ms) but its recall is only 0.20 (8 of 10 hard tasks missed → silently routed to local), which is disqualifying for a frontier gate; a fast gate that misroutes most hard work is worse than no gate.

**Latency caveat:** the stated preference is sub-200ms on the hot path. **No candidate meets it** — the field spans 1139–1925ms p50 on this box. Latency therefore does not separate the winner from the pack; it only rules the 1.5B's speed advantage moot since that model fails on accuracy. If sub-200ms becomes a hard requirement, the answer is not a different model at this quant on this hardware — it is quantization/hardware changes (smaller quant, GPU/Metal offload, or a cached-classification path), evaluated separately.

## Recommendation (64GB office M4 Max)

**Test `qwen2.5:7b-instruct` first, then `qwen2.5:14b-instruct`; use `gemma2:9b` as the cross-family check.**

Rationale:
- Keep Qwen2.5 as the spine. Two of the local rungs (1.5B, 3B) are Qwen2.5, and the 3B already showed a clean 1.5B→3B scaling curve (binary 0.7556 → 0.8444, format/consistency 0.98/0.91 → 1.00/1.00). Extending the same family to 7B and 14B isolates capacity from architecture and gives a continuous 1.5B→14B curve — the cleanest read on how much accuracy headroom is actually left.
- `qwen2.5:7b-instruct-q4_K_M` (4.7GB) is **already local on the office box** — the cheapest next data point. Benchmark it there, not on this 16GB machine (it exceeds the ~3.3GB free-RAM cap and would thrash).
- `gemma2:9b` guards against a family-specific artifact: gemma2:2b had the best recall here, so a 9B Gemma is the fair cross-family test of whether Qwen's format-discipline lead holds at scale or whether Gemma's recall advantage compounds.

**Will the 16GB winner still be best?** Possibly not on raw accuracy — 7B/14B should push binary accuracy and hard-recall higher, and on the M4 Max latency stops being the constraint the 16GB box imposes. But bigger is not automatically better for a hot-path gate: qwen2.5:3b already hit perfect format validity and self-consistency, so the larger models can only match (not beat) it on reliability, and they cost more latency and memory per call. The decision on the M4 Max should weight the accuracy gain of 7B/14B against their per-call cost; if 7B only marginally beats 3B on the binary call, the 3B remains the better production gate even where 64GB is available. Re-run this exact bake-off harness on the M4 Max before committing.

## Caveats (read before trusting these numbers)

- **Single-labeler, ~45-item gold set.** All labels are one reference annotator's judgment per the shared rubric. 45 items with only 10 hard positives means each hard task is worth 10 percentage points of recall — confidence intervals are wide and a handful of relabels could reorder the close pair (qwen3b vs. llama). Treat rankings as directional, not precise.
- **Weak outcome proxy.** `outcome_proxy_hard` agrees with the gold `needs_frontier` label on only 30/45 (67%): TP=2, FP=7, FN=8. It tracks execution friction (edit failures, interrupts, ≥10 continuations) rather than intrinsic difficulty — 8 of 10 hard rows are planning/research asks that completed without edit errors. It is a non-validating signal and was **not** used to score models; it only flags that the gold labels are not independently corroborated.
- **All-Opus corpus.** The source session log was produced entirely by a frontier model (an ML-infra / video-gen / Pilates-business build). Canonical coding instructions are sparse; tiers were assigned by analogy to the reasoning a small local model vs. a frontier model would need. The distribution may not match ADRL's real production traffic — revalidate on live routed instructions once available.
- **Latency is box-specific.** All p50/p90 figures are from this 16GB machine under one-model-at-a-time loading. They will not transfer to the M4 Max or to a GPU-served deployment; re-measure per target.
- **Metrics are reproducible.** Per-model raw metrics in `data/classifier-eval/{qwen2.5_1.5b_instruct,gemma2_2b,qwen2.5_3b_instruct,llama3.2_latest}.json`.
