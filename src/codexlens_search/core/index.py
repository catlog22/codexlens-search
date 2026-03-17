from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from codexlens_search.config import Config
from codexlens_search.core.base import BaseANNIndex

logger = logging.getLogger(__name__)

try:
    import hnswlib
    _HNSWLIB_AVAILABLE = True
except ImportError:
    _HNSWLIB_AVAILABLE = False


class ANNIndex(BaseANNIndex):
    """HNSW-based approximate nearest neighbor index.

    Lazy-loads on first use, thread-safe via RLock.
    """

    def __init__(self, path: str | Path, dim: int, config: Config) -> None:
        if not _HNSWLIB_AVAILABLE:
            raise ImportError("hnswlib is required. Install with: pip install hnswlib")

        self._path = Path(path)
        self._hnsw_path = self._path / "ann_index.hnsw"
        self._dim = dim
        self._config = config
        self._lock = threading.RLock()
        self._index: hnswlib.Index | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load or initialize the index (caller holds lock)."""
        if self._index is not None:
            return
        self.load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load index from disk or initialize a fresh one."""
        with self._lock:
            idx = hnswlib.Index(space="cosine", dim=self._dim)
            if self._hnsw_path.exists():
                idx.load_index(str(self._hnsw_path), max_elements=0)
                idx.set_ef(self._config.hnsw_ef)
                logger.debug("Loaded HNSW index from %s (%d items)", self._hnsw_path, idx.get_current_count())
            else:
                idx.init_index(
                    max_elements=1000,
                    ef_construction=self._config.hnsw_ef_construction,
                    M=self._config.hnsw_M,
                )
                idx.set_ef(self._config.hnsw_ef)
                logger.debug("Initialized fresh HNSW index (dim=%d)", self._dim)
            self._index = idx

    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        """Add float32 vectors.

        Does NOT call save() internally -- callers must call save()
        explicitly after batch indexing.

        Args:
            ids: shape (N,) int64
            vectors: shape (N, dim) float32
        """
        if len(ids) == 0:
            return

        vecs = np.ascontiguousarray(vectors, dtype=np.float32)

        with self._lock:
            self._ensure_loaded()
            # Expand capacity if needed
            current = self._index.get_current_count()
            max_el = self._index.get_max_elements()
            needed = current + len(ids)
            if needed > max_el:
                new_cap = max(max_el * 2, needed + 100)
                self._index.resize_index(new_cap)
            self._index.add_items(vecs, ids.astype(np.int64))

    def fine_search(
        self, query_vec: np.ndarray, top_k: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors.

        Args:
            query_vec: float32 vector of shape (dim,)
            top_k: number of results; defaults to config.ann_top_k

        Returns:
            (ids, distances) as numpy arrays
        """
        k = top_k if top_k is not None else self._config.ann_top_k

        with self._lock:
            self._ensure_loaded()

            count = self._index.get_current_count()
            if count == 0:
                return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

            k = min(k, count)
            self._index.set_ef(max(self._config.hnsw_ef, k))

            q = np.ascontiguousarray(query_vec, dtype=np.float32).reshape(1, -1)
            labels, distances = self._index.knn_query(q, k=k)
            return labels[0].astype(np.int64), distances[0].astype(np.float32)

    def save(self) -> None:
        """Save index to disk (caller may or may not hold lock)."""
        with self._lock:
            if self._index is None:
                return
            self._path.mkdir(parents=True, exist_ok=True)
            self._index.save_index(str(self._hnsw_path))

    def __len__(self) -> int:
        with self._lock:
            if self._index is None:
                return 0
            return self._index.get_current_count()
