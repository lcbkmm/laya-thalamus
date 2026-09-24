"""Decision result cache keyed by query+tools fingerprint.

**Unstable**. Configure via ``RouterConfig.cache``. See ``API.md`` / ``POLICY.md``.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Sequence

from laya_thalamus.schemas import RouteDecision, ToolSpec


@dataclass
class _Entry:
    value: RouteDecision
    stored_at: float


def fingerprint(
    query: str,
    tools: Sequence[ToolSpec],
    *,
    tool_result: str | None = None,
    context: Any = None,
) -> str:
    """Stable hash of routing inputs (query + tool criteria + optional result/context)."""
    payload = {
        "q": query,
        "tools": [(t.name, t.description) for t in tools],
        "tr": tool_result or "",
        "ctx": "" if context is None else str(context)[:800],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DecisionCache:
    """Bounded LRU + TTL cache for :class:`RouteDecision`."""

    def __init__(self, *, max_size: int = 1024, ttl_s: float | None = 30.0) -> None:
        self.max_size = max_size
        self.ttl_s = ttl_s
        self._data: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> RouteDecision | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            if self.ttl_s is not None and (time.time() - entry.stored_at) > self.ttl_s:
                self._data.pop(key, None)
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return entry.value

    def put(self, key: str, value: RouteDecision) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = _Entry(value=value, stored_at=time.time())
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0
