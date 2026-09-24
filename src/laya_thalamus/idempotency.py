"""In-memory idempotency cache keyed by (session_id, idempotency_key).

**Unstable**. Prefer ``idempotency_key`` on ``AgentRouter.route``. See ``API.md``.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class _Entry(Generic[T]):
    value: T
    stored_at: float


class IdempotencyCache(Generic[T]):
    """Bounded LRU cache; entries optionally expire after ``ttl_s`` seconds."""

    def __init__(self, *, max_size: int = 1024, ttl_s: float | None = 600.0) -> None:
        self.max_size = max_size
        self.ttl_s = ttl_s
        self._data: OrderedDict[tuple[str, str], _Entry[T]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: str, key: str) -> T | None:
        k = (session_id, key)
        with self._lock:
            entry = self._data.get(k)
            if entry is None:
                return None
            if self.ttl_s is not None and (time.time() - entry.stored_at) > self.ttl_s:
                self._data.pop(k, None)
                return None
            self._data.move_to_end(k)
            return entry.value

    def put(self, session_id: str, key: str, value: T) -> None:
        k = (session_id, key)
        with self._lock:
            if k in self._data:
                self._data.move_to_end(k)
            self._data[k] = _Entry(value=value, stored_at=time.time())
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

    def clear(self, session_id: str | None = None) -> None:
        with self._lock:
            if session_id is None:
                self._data.clear()
                return
            for k in [k for k in self._data if k[0] == session_id]:
                self._data.pop(k, None)
