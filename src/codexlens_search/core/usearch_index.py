from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from codexlens_search.config import Config
from codexlens_search.core.base import BaseANNIndex

logger = logging.getLogger(__name__)

try:
    from usearch.index import Index as USearchIndex
    _USEARCH_AVAILABLE = True
except ImportError:
    _USEARCH_AVAILABLE = False


class UsearchANNIndex(BaseANNIndex):
    """USearch HNSW-based approximate nearest neighbor index.

    Uses cosine metric natively (no manual L2 normalization needed).
    Thread-safe via RLock.
    """

    def __init__(self, path: str | Path, dim: int, config: Config) -> None:
        if not _USEARCH_AVAILABLE:
            raise ImportError(
                "usearch is required. Install with: pip install usearch"
            )

        self._path = Path(path)
        self._index_path = self._path / "usearch_ann.index"
        self._dim = dim
        self._config = config
        self._lock = threading.RLock()
        self._index: USearchIndex | None = None

    def _ensure_loaded(self) -> None:
        if self._index is not None:
            return
        self.load()

    def load(self) -> None:
        with self._lock:
            idx = USearchIndex(
                ndim=self._dim,
                metric="cos",
                connectivity=self._config.hnsw_M,
                expansion_add=self._config.hnsw_ef_construction,
                expansion_search=self._config.hnsw_ef,
                dtype="f32",
            )
            if self._index_path.exists():
                idx.load(str(self._index_path))
                logger.debug(
                    "Loaded USearch index from %s (%d items)",
                    self._index_path, len(idx),
                )
            else:
                logger.debug(
                    "Initialized fresh USearch index (dim=%d, M=%d)",
                    self._dim, self._config.hnsw_M,
                )
            self._index = idx

    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return

        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        keys = np.ascontiguousarray(ids, dtype=np.int64)

        with self._lock:
            self._ensure_loaded()
            self._index.add(keys=keys, vectors=vecs)

    def fine_search(
        self, query_vec: np.ndarray, top_k: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        k = top_k if top_k is not None else self._config.ann_top_k

        with self._lock:
            self._ensure_loaded()

            count = len(self._index)
            if count == 0:
                return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

            k = min(k, count)
            q = np.ascontiguousarray(query_vec, dtype=np.float32).reshape(1, -1)
            matches = self._index.search(q, count=k)
            return (
                np.asarray(matches.keys, dtype=np.int64).ravel(),
                np.asarray(matches.distances, dtype=np.float32).ravel(),
            )

    def save(self) -> None:
        with self._lock:
            if self._index is None:
                return
            self._path.mkdir(parents=True, exist_ok=True)
            self._index.save(str(self._index_path))

    def __len__(self) -> int:
        with self._lock:
            if self._index is None:
                return 0
            return len(self._index)
