"""Compress Agent state into short text suitable for Laya.

**Unstable**. See ``API.md``.
"""

from __future__ import annotations

import json
import re
from typing import Any


_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def sanitize_text(text: str, max_chars: int | None = None) -> str:
    cleaned = _ILLEGAL.sub("", text).strip()
    if max_chars is not None and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned


def compress_context(
    query: str,
    context: str | dict[str, Any] | list[Any] | None = None,
    *,
    max_chars: int = 1500,
    keep_tail: bool = True,
    tool_result: str | None = None,
) -> str:
    """Build a compact state string for Laya predict()."""
    parts: list[str] = [f"User query: {sanitize_text(query, 500)}"]

    if context is not None:
        ctx = _stringify(context)
        parts.append(f"Context: {ctx}")

    if tool_result:
        result = sanitize_text(tool_result, 800)
        failed = bool(
            re.search(r"(?i)\berror\b|失败|exception|quota exceeded", result)
        )
        status = "FAILED" if failed else "AVAILABLE"
        parts.append(
            f"Tool result status: {status}\n"
            f"Tool result: {result}"
        )

    state = "\n".join(parts)
    state = sanitize_text(state)

    if len(state) <= max_chars:
        return state

    if keep_tail:
        header = parts[0] + "\n"
        budget = max_chars - len(header) - 20
        if budget < 100:
            return state[:max_chars]
        return header + "...[truncated]...\n" + state[-budget:]
    return state[:max_chars]


def _stringify(context: str | dict[str, Any] | list[Any]) -> str:
    if isinstance(context, str):
        return sanitize_text(context, 1200)
    try:
        return sanitize_text(json.dumps(context, ensure_ascii=False), 1200)
    except (TypeError, ValueError):
        return sanitize_text(str(context), 1200)
