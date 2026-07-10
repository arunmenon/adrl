"""Model context-window table. Port of core/tokenLimits.ts.

Upstream normalizes the model id aggressively (strip provider prefixes,
dates, quantization suffixes) and then walks an ordered regex table —
first match wins. The compression trigger (client.py) divides history
tokens by this limit.
"""

from __future__ import annotations

import re

DEFAULT_TOKEN_LIMIT = 200_000
DEFAULT_OUTPUT_TOKEN_LIMIT = 32_000


def normalize(model: str) -> str:
    """Canonicalize a model id the way tokenLimits.ts does (abridged).

    'Qwen/Qwen3-Coder-480B:latest' -> 'qwen3-coder-480b' etc.
    """
    s = (model or "").strip().lower()
    s = s.split("/")[-1]  # strip provider prefix
    s = re.split(r"[|:]", s)[-1] or s  # last |/: segment (ollama tags)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-preview$", "", s)
    # date / version / quant suffixes: -20250101, -v1, -q4_k_m, -latest
    if not re.search(r"qwen-(plus|flash|vl-max)-latest", s):
        s = re.sub(r"-(latest|\d{6,8}|q\d[\w-]*)$", "", s)
    return s


# Ordered (pattern, limit) tables — first match wins, mirroring PATTERNS /
# OUTPUT_PATTERNS in tokenLimits.ts (subset covering the families this repo
# actually routes to, plus the qwen defaults).
_INPUT_PATTERNS: list[tuple[str, int]] = [
    (r"^gemini-", 1_048_576),
    (r"^gpt-5", 272_000),
    (r"^gpt-", 131_072),
    (r"^o\d", 200_000),
    (r"^claude-", 200_000),
    (r"^(qwen3-coder-plus|qwen3-coder-flash|qwen3\.\d|qwen-plus-latest|qwen-flash-latest|coder-model)", 1_000_000),
    (r"^qwen3-max", 262_144),
    (r"^qwen3-coder-", 262_144),
    (r"^qwen", 262_144),
    (r"^deepseek-v4", 1_000_000),
    (r"^deepseek", 131_072),
    (r"^kimi-", 262_144),
]

_OUTPUT_PATTERNS: list[tuple[str, int]] = [
    (r"^gemini-3", 65_536),
    (r"^gemini", 8_192),
    (r"^(gpt-5|o\d)", 131_072),
    (r"^gpt", 16_384),
    (r"^claude-", 65_536),
    (r"^(qwen3\.\d|coder-model)", 65_536),
    (r"^qwen", 32_768),
]


def token_limit(model: str, kind: str = "input") -> int:
    name = normalize(model)
    table = _INPUT_PATTERNS if kind == "input" else _OUTPUT_PATTERNS
    default = DEFAULT_TOKEN_LIMIT if kind == "input" else DEFAULT_OUTPUT_TOKEN_LIMIT
    for pattern, limit in table:
        if re.search(pattern, name):
            return limit
    return default
