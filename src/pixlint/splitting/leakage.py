from __future__ import annotations

from hashlib import md5

from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import LeakageReport


def detect_leakage(
    dataset: CVDataset,
    split_result: dict[str, list[str]] | None = None,
    methods: list[str] | None = None,
) -> LeakageReport:
    if methods is None:
        methods = ["exact"]

    leakage_pairs: list[dict[str, str]] = []

    if "exact" in methods:
        hash_to_ids: dict[str, list[str]] = {}
        for img in dataset.images:
            h = _fast_md5(img.path)
            if h not in hash_to_ids:
                hash_to_ids[h] = []
            hash_to_ids[h].append(img.image_id)

        for h, ids in hash_to_ids.items():
            if len(ids) > 1:
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        leakage_pairs.append({
                            "image_a": ids[i],
                            "image_b": ids[j],
                            "method": "exact",
                            "similarity": "1.0",
                        })

    if split_result:
        train_set = set(split_result.get("train", []))
        test_set = set(split_result.get("test", []))
        overlap = train_set & test_set
        for img_id in overlap:
            leakage_pairs.append({
                "image_a": img_id,
                "image_b": img_id,
                "method": "train_test_overlap",
                "similarity": "1.0",
            })

    return LeakageReport(
        dataset_id=dataset.dataset_id,
        total_leaks=len(leakage_pairs),
        leakage_pairs=leakage_pairs,
        methods_used=methods,
    )


def _fast_md5(filepath: str, chunk_size: int = 4096) -> str:
    h = md5(usedforsecurity=False)
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except (FileNotFoundError, OSError):
        return ""
