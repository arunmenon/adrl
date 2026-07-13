"""Dependency-light secret detection for the live privacy pin.

The miner and live proxy share this module so replay precision and production
behavior cannot drift. It deliberately has no parser/dataframe dependencies.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


HIGH_CONFIDENCE: dict[str, re.Pattern] = {
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "bearer_token": re.compile(
        r"\b(?:sk-ant-[a-zA-Z0-9_-]{20,}|sk-[a-zA-Z0-9]{32,}|"
        r"ghp_[a-zA-Z0-9]{36}|gho_[a-zA-Z0-9]{36}|"
        r"xox[bp]-[a-zA-Z0-9-]{20,})"
    ),
}

_ENV_KEY = (
    r"[A-Z][A-Z0-9]*_?(?:API_?KEY|SECRET(?:_?KEY)?|PASSWORD|PASSWD|"
    r"ACCESS_?KEY|AUTH_?TOKEN|PRIVATE_?KEY|CLIENT_?SECRET)[A-Z0-9_]*"
)
ENTROPY_CANDIDATES: dict[str, re.Pattern] = {
    "env_assignment": re.compile(
        rf"\b{_ENV_KEY}\s*[=:]\s*['\"]?(?P<val>[A-Za-z0-9+/=_-]{{16,}})['\"]?"
    ),
    "connection_string_cred": re.compile(
        r"\b(?:postgres|postgresql|mysql|mongodb(?:\+srv)?|redis|amqp)://"
        r"[^\s/:@]+:(?P<val>[^\s@]+)@(?P<host>[^\s/:]+)"
    ),
}

PLACEHOLDER_CREDS = {
    "pass", "password", "passwd", "changeme", "secret", "example", "test",
    "user", "username", "foo", "bar", "xxx", "your_password", "placeholder",
    "admin", "root", "123456", "postgres", "mysql", "redis",
}
PLACEHOLDER_HOSTS = {
    "localhost", "127.0.0.1", "example.com", "host", "db", "database",
}
ENTROPY_MIN = 3.0
SECRET_PATTERNS = {**HIGH_CONFIDENCE, **ENTROPY_CANDIDATES}


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return -sum((count / length) * math.log2(count / length)
                for count in counts.values())


def scan_text(text: str) -> list[str]:
    """Return matched secret-pattern names without retaining the matched value."""
    hits: list[str] = []
    for name, pattern in HIGH_CONFIDENCE.items():
        if pattern.search(text):
            hits.append(name)
    for name, pattern in ENTROPY_CANDIDATES.items():
        for match in pattern.finditer(text):
            value = match.group("val")
            if name == "connection_string_cred":
                host = (match.groupdict().get("host") or "").lower()
                if value.lower() in PLACEHOLDER_CREDS or host in PLACEHOLDER_HOSTS:
                    continue
            if len(value) >= 12 and shannon_entropy(value) >= ENTROPY_MIN:
                hits.append(name)
                break
    return hits


def scan_content(content: Any) -> list[str]:
    """Scan nested message content while returning pattern names only."""
    hits: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, str):
            hits.extend(scan_text(value))
        elif isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)

    visit(content)
    return list(dict.fromkeys(hits))
