"""Small redaction helpers for logs, prompts, and exported artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_SECRET_KEYS = re.compile(r"(credential|secret|token|password|private[_-]?key)", re.IGNORECASE)


def redact(value: Any) -> Any:
    """Recursively redact secret-like keys and absolute local paths."""
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _SECRET_KEYS.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str):
        if value.startswith(("/Users/", "/home/")):
            return Path(value).name
        return value
    return value
