from __future__ import annotations

import json
import os
import tempfile
import time
from collections import OrderedDict
from typing import Any


class LRUCache:
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        if key not in self._cache:
            return None
        value, _ = self._cache.pop(key)
        self._cache[key] = (value, time.time())
        return value

    def put(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._cache.pop(key)
        elif len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
        self._cache[key] = (value, time.time())

    def clear(self) -> None:
        self._cache.clear()


class DiskCache:
    def __init__(self, cache_dir: str | None = None):
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), "cv_dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

    def get(self, key: str) -> Any | None:
        path = os.path.join(self.cache_dir, _sanitize(key) + ".json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, key: str, value: Any) -> None:
        path = os.path.join(self.cache_dir, _sanitize(key) + ".json")
        try:
            with open(path, "w") as f:
                json.dump(value, f)
        except (TypeError, OSError):
            # Skip non-serializable values
            pass

    def clear(self) -> None:
        for f in os.listdir(self.cache_dir):
            if f.endswith(".json"):
                os.remove(os.path.join(self.cache_dir, f))


def _sanitize(key: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in key)
