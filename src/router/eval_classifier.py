"""Score ONE local model as the ADRL routing classifier against the benchmark.

The classifier's job (shared rubric, identical to the benchmark labels):
  - trivial : mechanical, any small model nails it (commit msg, rename, typo,
              format/lint, one-line "what does X do").
  - standard: a normal single-file change or clear bug fix (add a function, fix
              an obvious error, small edit). A competent small model can do it.
  - hard    : multi-file / architectural / ambiguous / cross-cutting (refactor
              across the codebase, redesign, subtle multi-module debugging).
              Needs a frontier model.
  needs_frontier := (tier == "hard").

Routing maps trivial+standard -> local-eligible and hard -> frontier, so the
load-bearing metric is the BINARY needs_frontier call, not 4-way exactness.

Each benchmark item is classified `--repeats` times. Every request hits ollama
POST /api/chat with keep_alive=0 so the model UNLOADS after each call — only one
model stays resident at a time (16GB box). Metrics land in
data/classifier-eval/<sanitized-model>.json.

Usage:
  PYTHONPATH=src .venv/bin/python -m router.eval_classifier --model llama3.2:latest
  PYTHONPATH=src .venv/bin/python -m router.eval_classifier --model qwen2.5:7b-instruct-q4_K_M --repeats 3
  # smoke test on the first few items:
  PYTHONPATH=src .venv/bin/python -m router.eval_classifier --model llama3.2:latest --limit 3 --repeats 1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_BENCH = Path("data/classifier-bench.jsonl")
DEFAULT_OUT_DIR = Path("data/classifier-eval")
REQUEST_TIMEOUT_S = 40.0
NUM_PREDICT = 120
VALID_TIERS = ("trivial", "standard", "hard")

# The classifier prompt. The rubric text is verbatim from the shared COMMON
# definitions so the model judges by exactly the standard the gold labels use.
SYSTEM_PROMPT = """You are a routing classifier for coding-agent instructions. \
Read ONLY the instruction the user gives you and judge how hard it is for a small \
local model to complete correctly. Do not attempt the task itself.

Use these exact tier definitions:
- "trivial": mechanical, any small model nails it — commit message, rename, typo, \
format/lint, one-line "what does X do".
- "standard": a normal single-file change or clear bug fix — add a function, fix an \
obvious error, small edit. A competent small model can do it.
- "hard": multi-file / architectural / ambiguous / cross-cutting — refactor across \
the codebase, redesign, subtle multi-module debugging. Needs a frontier model.

needs_frontier is true if and only if the tier is "hard".

Reply with STRICT JSON on a single line and nothing else, in exactly this shape:
{"tier":"trivial|standard|hard","needs_frontier":true|false,"reason":"<=12 words"}"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.S)
_TIER_RE = re.compile(r'"?tier"?\s*[:=]\s*"?(trivial|standard|hard)"?', re.I)
_FRONTIER_RE = re.compile(r'"?needs_frontier"?\s*[:=]\s*"?(true|false)"?', re.I)


def sanitize_model(model: str) -> str:
    """Turn an ollama tag into a safe filename stem: llama3.2:latest -> llama3.2_latest."""
    return re.sub(r"[^A-Za-z0-9.]+", "_", model).strip("_")


def load_bench(path: Path, limit: int) -> list[dict]:
    """Read the JSONL benchmark; keep only the fields the contract guarantees."""
    items: list[dict] = []
    with path.open() as fh:
        for line_number, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"skip malformed bench line {line_number}: {exc}", file=sys.stderr)
                continue
            if "instruction" not in record or "gold_tier" not in record:
                print(f"skip bench line {line_number}: missing required fields", file=sys.stderr)
                continue
            items.append(record)
            if limit and len(items) >= limit:
                break
    return items


def parse_classification(text: str | None) -> tuple[str | None, bool | None]:
    """Leniently extract (tier, needs_frontier) from a model reply.

    Returns tier=None when nothing usable could be recovered (format-invalid).
    When a tier is found but needs_frontier is absent, derive it from the tier
    (needs_frontier := tier == "hard") rather than discarding the item.
    """
    if not text:
        return None, None
    tier: str | None = None
    frontier: bool | None = None

    block = _JSON_BLOCK_RE.search(text)
    if block:
        try:
            obj = json.loads(block.group(0))
        except (json.JSONDecodeError, ValueError):
            obj = None
        if isinstance(obj, dict):
            candidate = str(obj.get("tier", "")).strip().lower()
            if candidate in VALID_TIERS:
                tier = candidate
            flag = obj.get("needs_frontier")
            if isinstance(flag, bool):
                frontier = flag
            elif isinstance(flag, str) and flag.strip().lower() in ("true", "false"):
                frontier = flag.strip().lower() == "true"

    if tier is None:
        match = _TIER_RE.search(text)
        if match:
            tier = match.group(1).lower()
    if frontier is None:
        match = _FRONTIER_RE.search(text)
        if match:
            frontier = match.group(1).lower() == "true"

    if frontier is None and tier is not None:
        frontier = tier == "hard"
    return tier, frontier


def call_ollama(model: str, instruction: str, timeout: float) -> tuple[str | None, float, str | None]:
    """POST one classification request. Returns (content, elapsed_ms, error).

    content is None on any failure; error carries a short reason. The broad
    exception net is deliberate — a hung or garbage upstream must be counted as
    format-invalid, never crash the run.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": NUM_PREDICT},
        "keep_alive": 0,
    }
    from router.backends import http_post_json  # shared transport (WS0)

    start = time.perf_counter()
    body = http_post_json(OLLAMA_URL, payload, timeout)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if body is None:
        return None, elapsed_ms, "transport failure (connection/timeout/non-200)"
    content = (body.get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        return None, elapsed_ms, "empty content"
    return content, elapsed_ms, None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((pct / 100.0) * (len(ordered) - 1)))
    index = max(0, min(len(ordered) - 1, index))
    return round(ordered[index], 1)


def score(items: list[dict], predictions: list[list[tuple]], latencies: list[float],
          repeats: int) -> dict:
    """Aggregate per-item repeat predictions into the metrics contract.

    predictions[i] is a list (length == repeats) of (tier, frontier) tuples; a
    (None, None) entry means that repeat was format-invalid.
    """
    n_items = len(items)
    tier_correct = 0
    binary_correct = 0
    consistent = 0
    true_pos = false_pos = false_neg = 0
    valid_responses = 0

    for item, repeat_preds in zip(items, predictions):
        valid_tiers = [tier for tier, _ in repeat_preds if tier is not None]
        valid_frontiers = [frontier for tier, frontier in repeat_preds
                           if tier is not None and frontier is not None]
        valid_responses += len(valid_tiers)

        majority_tier = Counter(valid_tiers).most_common(1)[0][0] if valid_tiers else None
        majority_frontier = (Counter(valid_frontiers).most_common(1)[0][0]
                             if valid_frontiers else None)

        if majority_tier == item["gold_tier"]:
            tier_correct += 1

        gold_frontier = bool(item.get("gold_needs_frontier",
                                      item["gold_tier"] == "hard"))
        if majority_frontier is not None:
            if majority_frontier == gold_frontier:
                binary_correct += 1
            if gold_frontier and majority_frontier:
                true_pos += 1
            elif not gold_frontier and majority_frontier:
                false_pos += 1
            elif gold_frontier and not majority_frontier:
                false_neg += 1

        # Consistent only if every repeat parsed AND agreed on one tier.
        if len(valid_tiers) == repeats and len(set(valid_tiers)) == 1:
            consistent += 1

    total_responses = n_items * repeats
    precision_denom = true_pos + false_pos
    recall_denom = true_pos + false_neg

    return {
        "model": None,  # filled by caller
        "n_items": n_items,
        "repeats": repeats,
        "tier_accuracy": round(tier_correct / n_items, 4) if n_items else 0.0,
        "binary_frontier_accuracy": round(binary_correct / n_items, 4) if n_items else 0.0,
        "binary_precision": round(true_pos / precision_denom, 4) if precision_denom else None,
        "binary_recall": round(true_pos / recall_denom, 4) if recall_denom else None,
        "self_consistency": round(consistent / n_items, 4) if n_items else 0.0,
        "format_valid_pct": round(valid_responses / total_responses, 4) if total_responses else 0.0,
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p90_ms": _percentile(latencies, 90),
        "confusion": {"tp": true_pos, "fp": false_pos, "fn": false_neg,
                      "n_scored": binary_correct + (precision_denom + recall_denom - 2 * true_pos)},
        "errors": None,  # filled by caller
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score one local model as the ADRL routing classifier.")
    parser.add_argument("--model", required=True, help="ollama tag, e.g. llama3.2:latest")
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0,
                        help="classify only the first N items (0 = all; use for smoke tests)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT_S)
    args = parser.parse_args()

    if args.repeats < 1:
        print("--repeats must be >= 1", file=sys.stderr)
        return 2
    if not args.bench.exists():
        print(f"benchmark not found: {args.bench}", file=sys.stderr)
        return 1

    items = load_bench(args.bench, args.limit)
    if not items:
        print(f"no usable benchmark items in {args.bench}", file=sys.stderr)
        return 1

    predictions: list[list[tuple]] = []
    latencies: list[float] = []
    errors = 0
    for item_index, item in enumerate(items, 1):
        repeat_preds: list[tuple] = []
        for _ in range(args.repeats):
            content, elapsed_ms, error = call_ollama(args.model, item["instruction"], args.timeout)
            if error is None:
                latencies.append(elapsed_ms)
                tier, frontier = parse_classification(content)
                if tier is None:
                    errors += 1
                repeat_preds.append((tier, frontier))
            else:
                errors += 1
                repeat_preds.append((None, None))
        predictions.append(repeat_preds)
        print(f"  [{item_index}/{len(items)}] {item['gold_tier']:<8} "
              f"-> {Counter(t for t, _ in repeat_preds if t)}", file=sys.stderr)

    metrics = score(items, predictions, latencies, args.repeats)
    metrics["model"] = args.model
    metrics["errors"] = errors

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{sanitize_model(args.model)}.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print(
        f"{args.model} | binary_acc={metrics['binary_frontier_accuracy']} "
        f"tier_acc={metrics['tier_accuracy']} "
        f"P={metrics['binary_precision']} R={metrics['binary_recall']} "
        f"self_consist={metrics['self_consistency']} fmt={metrics['format_valid_pct']} "
        f"p50={metrics['latency_p50_ms']}ms p90={metrics['latency_p90_ms']}ms "
        f"n={metrics['n_items']} x{metrics['repeats']} -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
