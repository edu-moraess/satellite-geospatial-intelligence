"""
Lightweight TTL cache for weather requests (master prompt item 16).

Deliberately independent of Streamlit: st.cache_data only caches
within a single Streamlit session/process and cannot be unit-tested
without the streamlit runtime. This cache is a plain dict with
expiry, so src/weather/ stays importable and testable on its own.
"""

from __future__ import annotations

import time


class TTLCache:
    """A tiny in-memory cache with per-entry time-to-live and an entry cap."""

    def __init__(self, ttl_seconds: float = 900.0, max_entries: int = 256):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._store: dict = {}

    def get(self, key):
        entry = self._store.get(key)
        if entry is None:
            return None

        expires_at, value = entry
        if time.monotonic() >= expires_at:
            self._store.pop(key, None)
            return None

        return value

    def set(self, key, value) -> None:
        if len(self._store) >= self.max_entries and key not in self._store:
            oldest_key = min(self._store, key=lambda k: self._store[k][0])
            self._store.pop(oldest_key, None)

        self._store[key] = (time.monotonic() + self.ttl_seconds, value)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


def cached_call(cache: TTLCache, key, fn):
    """Return cache[key] if fresh; otherwise call fn(), store it, and return it."""
    hit = cache.get(key)
    if hit is not None:
        return hit

    value = fn()
    cache.set(key, value)
    return value
