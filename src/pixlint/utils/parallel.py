from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable


def get_num_workers(fraction: float = 0.8) -> int:
    return max(1, int(multiprocessing.cpu_count() * fraction))


def parallel_map(
    func: Callable,
    items: list,
    *,
    max_workers: int | None = None,
    use_processes: bool = True,
    chunksize: int = 1,
) -> list:
    if max_workers is None:
        max_workers = get_num_workers()
    executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    with executor_cls(max_workers=max_workers) as executor:
        return list(executor.map(func, items, chunksize=chunksize))
