from __future__ import annotations

import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from pixlint.utils.hashing import (
    compute_ahash,
    compute_dhash,
    compute_phash,
    hamming_distance,
)
from pixlint.utils.image_io import read_image


class ImageIndex:
    def __init__(self, db_path: str | None = None):
        self._phash_index: dict[str, list[str]] = defaultdict(list)
        self._dhash_index: dict[str, list[str]] = defaultdict(list)
        self._ahash_index: dict[str, list[str]] = defaultdict(list)
        self._image_paths: dict[str, str] = {}
        self._image_sizes: dict[str, tuple[int, int]] = {}

        if db_path is None:
            db_path = str(Path(tempfile.gettempdir()) / "cv_dataset_index.db")
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS image_index (
                    image_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    phash TEXT,
                    dhash TEXT,
                    ahash TEXT,
                    width INTEGER,
                    height INTEGER,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embedding_index (
                    image_id TEXT PRIMARY KEY,
                    embedding BLOB,
                    model TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_phash ON image_index(phash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dhash ON image_index(dhash)")
            conn.commit()

    def build(
        self,
        image_ids: list[str],
        image_paths: list[str],
        read_fn: Callable | None = None,
    ) -> None:
        with sqlite3.connect(self._db_path) as conn:
            for img_id, path in zip(image_ids, image_paths):
                self._image_paths[img_id] = path
                img = read_image(path)
                if img is None:
                    continue
                ph = compute_phash(img)
                self._phash_index[ph].append(img_id)
                dh = compute_dhash(img)
                self._dhash_index[dh].append(img_id)
                ah = compute_ahash(img)
                self._ahash_index[ah].append(img_id)
                h, w = img.shape[:2]
                self._image_sizes[img_id] = (w, h)
                conn.execute(
                    "INSERT OR REPLACE INTO image_index (image_id, path, phash, dhash, ahash, width, height) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (img_id, path, ph, dh, ah, w, h),
                )
            conn.commit()

    def find_similar_phash(
        self, image_id: str, threshold: int = 8
    ) -> list[str]:
        path = self._image_paths.get(image_id)
        if not path:
            return []
        img = read_image(path)
        if img is None:
            return []
        query_hash = compute_phash(img)
        similar: list[str] = []
        for hash_val, ids in self._phash_index.items():
            if hamming_distance(query_hash, hash_val) <= threshold:
                similar.extend(ids)
        return list(set(similar))

    def get_path(self, image_id: str) -> str | None:
        return self._image_paths.get(image_id)

    def get_size(self, image_id: str) -> tuple[int, int] | None:
        return self._image_sizes.get(image_id)

    def search_by_path(self, path_pattern: str) -> list[dict[str, Any]]:
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "SELECT image_id, path, width, height FROM image_index WHERE path LIKE ?",
                (f"%{path_pattern}%",),
            )
            results = []
            for row in cursor.fetchall():
                results.append({
                    "image_id": row[0],
                    "path": row[1],
                    "width": row[2],
                    "height": row[3],
                })
            return results

    def save(self, path: str) -> None:
        import shutil
        shutil.copy2(self._db_path, path)

    def load(self, path: str) -> None:
        import shutil
        shutil.copy2(path, self._db_path)
        self._rebuild_from_db()

    def _rebuild_from_db(self) -> None:
        self._phash_index.clear()
        self._dhash_index.clear()
        self._ahash_index.clear()
        self._image_paths.clear()
        self._image_sizes.clear()

        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "SELECT image_id, path, phash, dhash, ahash, width, height FROM image_index"
            )
            for row in cursor.fetchall():
                img_id, path, ph, dh, ah, w, h = row
                self._image_paths[img_id] = path
                if ph:
                    self._phash_index[ph].append(img_id)
                if dh:
                    self._dhash_index[dh].append(img_id)
                if ah:
                    self._ahash_index[ah].append(img_id)
                if w and h:
                    self._image_sizes[img_id] = (w, h)

    def store_embedding(self, image_id: str, embedding: list[float], model: str) -> None:
        import struct
        blob = struct.pack(f"{len(embedding)}f", *embedding)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embedding_index (image_id, embedding, model) VALUES (?, ?, ?)",
                (image_id, blob, model),
            )
            conn.commit()

    def get_embedding(self, image_id: str) -> list[float] | None:
        import struct
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "SELECT embedding FROM embedding_index WHERE image_id = ?", (image_id,)
            )
            row = cursor.fetchone()
            if row and row[0]:
                n = len(row[0]) // 4
                return list(struct.unpack(f"{n}f", row[0]))
            return None

    @property
    def size(self) -> int:
        return len(self._image_paths)
