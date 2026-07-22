from __future__ import annotations

import time
from typing import Any, Callable

_TQDM_AVAILABLE = False
try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    pass


class ProgressTracker:
    def __init__(self, total: int, desc: str = "Processing", silent: bool = False):
        self.total = total
        self.current = 0
        self.desc = desc
        self.silent = silent
        self.start_time = time.time()
        self._tqdm_bar = None
        if _TQDM_AVAILABLE and not silent:
            self._tqdm_bar = tqdm(total=total, desc=desc, unit="img")

    def update(self, n: int = 1, message: str | None = None) -> None:
        self.current += n
        if self._tqdm_bar is not None:
            self._tqdm_bar.update(n)
            if message:
                self._tqdm_bar.set_postfix_str(message)

    def set_message(self, message: str) -> None:
        if self._tqdm_bar is not None:
            self._tqdm_bar.set_postfix_str(message)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def progress(self) -> float:
        return self.current / max(self.total, 1) * 100.0

    def close(self) -> dict[str, Any]:
        elapsed = self.elapsed
        if self._tqdm_bar is not None:
            self._tqdm_bar.close()
        return {
            "total": self.total,
            "processed": self.current,
            "elapsed_seconds": round(elapsed, 2),
            "progress_percent": round(self.progress, 1),
        }


def track_progress(
    items: list,
    func: Callable,
    desc: str = "Processing",
    silent: bool = False,
) -> tuple[list, dict[str, Any]]:
    tracker = ProgressTracker(len(items), desc=desc, silent=silent)
    results: list = []

    for i, item in enumerate(items):
        try:
            result = func(item)
            results.append(result)
        except Exception:
            results.append(None)
        tracker.update(1, f"{i + 1}/{len(items)}")

    stats = tracker.close()
    return results, stats
