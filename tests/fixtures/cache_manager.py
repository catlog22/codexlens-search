"""LRU cache implementation with TTL support."""
import time
from collections import OrderedDict
from typing import Any


class LRUCache:
    """Least Recently Used cache with optional TTL expiration."""

    def __init__(self, capacity: int = 100, ttl_seconds: int = 0):
        self._capacity = capacity
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        if key not in self._cache:
            self._misses += 1
            return None
        value, timestamp = self._cache[key]
        if self._ttl > 0 and time.time() - timestamp > self._ttl:
            del self._cache[key]
            self._misses += 1
            return None
        self._cache.move_to_end(key)
        self._hits += 1
        return value

    def put(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, time.time())
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)

    def delete(self, key: str) -> bool:
        try:
            del self._cache[key]
            return True
        except KeyError:
            return False

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0
