from __future__ import annotations

import threading
from pathlib import Path

from pixlint.utils.schemas import CVDatasetInfo

_dataset_registry: dict[str, CVDatasetInfo] = {}
_registry_lock = threading.Lock()


def register_dataset(info: CVDatasetInfo) -> str:
    with _registry_lock:
        _dataset_registry[info.dataset_id] = info
    return info.dataset_id


def list_datasets() -> list[CVDatasetInfo]:
    with _registry_lock:
        return list(_dataset_registry.values())


def get_dataset_info(dataset_id: str) -> CVDatasetInfo | None:
    with _registry_lock:
        return _dataset_registry.get(dataset_id)


def remove_dataset(dataset_id: str) -> bool:
    with _registry_lock:
        return _dataset_registry.pop(dataset_id, None) is not None


def generate_dataset_id(path: str) -> str:
    p = Path(path).resolve()
    suffix = str(p.stat().st_ino % 100000) if p.exists() else str(abs(hash(path)) % 100000)
    return p.name + "_" + suffix
