from __future__ import annotations

import logging
import math
import threading
from pathlib import Path

import numpy as np

from codexlens_search.config import Config
from codexlens_search.core.base import BaseANNIndex, BaseBinaryIndex

logger = logging.getLogger(__name__)

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    faiss = None  # type: ignore[assignment]
    _FAISS_AVAILABLE = False


def _try_gpu_index(index: "faiss.Index") -> "faiss.Index":
    """Transfer a FAISS index to GPU if faiss-gpu is available.

    Returns the GPU index on success, or the original CPU index on failure.
    """
    try:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        logger.info("FAISS index transferred to GPU 0")
        return gpu_index
    except (AttributeError, RuntimeError) as exc:
        logger.debug("GPU transfer unavailable, staying on CPU: %s", exc)
        return index


def _to_cpu_for_save(index: "faiss.Index") -> "faiss.Index":
    """Convert a GPU index back to CPU for serialization."""
    try:
        return faiss.index_gpu_to_cpu(index)
    except (AttributeError, RuntimeError):
        return index


class FAISSANNIndex(BaseANNIndex):
    """FAISS-based ANN index using IndexHNSWFlat with optional GPU.

    Uses Inner Product space with L2-normalized vectors for cosine similarity.
    Thread-safe via RLock.
    """

    def __init__(self, path: str | Path, dim: int, config: Config) -> None:
        if not _FAISS_AVAILABLE:
            raise ImportError(
                "faiss is required. Install with: pip install faiss-cpu "
                "or pip install faiss-gpu"
            )

        self._path = Path(path)
        self._index_path = self._path / "faiss_ann.index"
        self._dim = dim
        self._config = config
        self._lock = threading.RLock()
        self._index: faiss.Index | None = None

    def _ensure_loaded(self) -> None:
        """Load or initialize the index (caller holds lock)."""
        if self._index is not None:
            return
        self.load()

    def load(self) -> None:
        """Load index from disk or initialize a fresh one."""
        with self._lock:
            if self._index_path.exists():
                idx = faiss.read_index(str(self._index_path))
                logger.debug(
                    "Loaded FAISS ANN index from %s (%d items)",
                    self._index_path, idx.ntotal,
                )
            else:
                # HNSW with flat storage, M=32 by default
                m = self._config.hnsw_M
                idx = faiss.IndexHNSWFlat(self._dim, m, faiss.METRIC_INNER_PRODUCT)
                idx.hnsw.efConstruction = self._config.hnsw_ef_construction
                idx.hnsw.efSearch = self._config.hnsw_ef
                logger.debug(
                    "Initialized fresh FAISS HNSW index (dim=%d, M=%d)",
                    self._dim, m,
                )
            self._index = _try_gpu_index(idx)

    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        """Add L2-normalized float32 vectors.

        Vectors are normalized before insertion so that Inner Product
        distance equals cosine similarity.

        Args:
            ids: shape (N,) int64 -- currently unused by FAISS flat index
                 but kept for API compatibility. FAISS uses sequential IDs.
            vectors: shape (N, dim) float32
        """
        if len(ids) == 0:
            return

        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        # Normalize for cosine similarity via Inner Product
        faiss.normalize_L2(vecs)

        with self._lock:
            self._ensure_loaded()
            self._index.add(vecs)

    def fine_search(
        self, query_vec: np.ndarray, top_k: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors.

        Args:
            query_vec: float32 vector of shape (dim,)
            top_k: number of results; defaults to config.ann_top_k

        Returns:
            (ids, distances) as numpy arrays. For IP space, higher = more
            similar, but distances are returned as-is for consumer handling.
        """
        k = top_k if top_k is not None else self._config.ann_top_k

        with self._lock:
            self._ensure_loaded()

            count = self._index.ntotal
            if count == 0:
                return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

            k = min(k, count)
            # Set efSearch for HNSW accuracy
            try:
                self._index.hnsw.efSearch = max(self._config.hnsw_ef, k)
            except AttributeError:
                pass  # GPU index may not expose hnsw attribute directly

            q = np.ascontiguousarray(query_vec, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(q)
            distances, labels = self._index.search(q, k)
            return labels[0].astype(np.int64), distances[0].astype(np.float32)

    def save(self) -> None:
        """Save index to disk."""
        with self._lock:
            if self._index is None:
                return
            self._path.mkdir(parents=True, exist_ok=True)
            cpu_index = _to_cpu_for_save(self._index)
            faiss.write_index(cpu_index, str(self._index_path))

    def __len__(self) -> int:
        with self._lock:
            if self._index is None:
                return 0
            return self._index.ntotal


class FAISSBinaryIndex(BaseBinaryIndex):
    """FAISS-based binary index using IndexBinaryFlat for Hamming distance.

    Vectors are binary-quantized (sign bit) before insertion.
    Thread-safe via RLock.
    """

    def __init__(self, path: str | Path, dim: int, config: Config) -> None:
        if not _FAISS_AVAILABLE:
            raise ImportError(
                "faiss is required. Install with: pip install faiss-cpu "
                "or pip install faiss-gpu"
            )

        self._path = Path(path)
        self._index_path = self._path / "faiss_binary.index"
        self._dim = dim
        self._config = config
        self._packed_bytes = math.ceil(dim / 8)
        self._lock = threading.RLock()
        self._index: faiss.IndexBinary | None = None

    def _ensure_loaded(self) -> None:
        if self._index is not None:
            return
        self.load()

    def _quantize(self, vectors: np.ndarray) -> np.ndarray:
        """Convert float32 vectors (N, dim) to packed uint8 (N, packed_bytes)."""
        binary = (vectors > 0).astype(np.uint8)
        return np.packbits(binary, axis=1)

    def _quantize_single(self, vec: np.ndarray) -> np.ndarray:
        """Convert a single float32 vector (dim,) to packed uint8 (1, packed_bytes)."""
        binary = (vec > 0).astype(np.uint8)
        return np.packbits(binary).reshape(1, -1)

    def load(self) -> None:
        """Load binary index from disk or initialize a fresh one."""
        with self._lock:
            if self._index_path.exists():
                idx = faiss.read_index_binary(str(self._index_path))
                logger.debug(
                    "Loaded FAISS binary index from %s (%d items)",
                    self._index_path, idx.ntotal,
                )
            else:
                # IndexBinaryFlat takes dimension in bits
                idx = faiss.IndexBinaryFlat(self._dim)
                logger.debug(
                    "Initialized fresh FAISS binary index (dim_bits=%d)", self._dim,
                )
            self._index = idx

    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        """Add float32 vectors (binary-quantized internally).

        Args:
            ids: shape (N,) int64 -- kept for API compatibility
            vectors: shape (N, dim) float32
        """
        if len(ids) == 0:
            return

        packed = self._quantize(vectors)
        packed = np.ascontiguousarray(packed, dtype=np.uint8)

        with self._lock:
            self._ensure_loaded()
            self._index.add(packed)

    def coarse_search(
        self, query_vec: np.ndarray, top_k: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search by Hamming distance.

        Args:
            query_vec: float32 vector of shape (dim,)
            top_k: number of results; defaults to config.binary_top_k

        Returns:
            (ids, distances) sorted ascending by Hamming distance
        """
        with self._lock:
            self._ensure_loaded()

            if self._index.ntotal == 0:
                return np.array([], dtype=np.int64), np.array([], dtype=np.int32)

            k = top_k if top_k is not None else self._config.binary_top_k
            k = min(k, self._index.ntotal)

            q = self._quantize_single(query_vec)
            q = np.ascontiguousarray(q, dtype=np.uint8)
            distances, labels = self._index.search(q, k)
            return labels[0].astype(np.int64), distances[0].astype(np.int32)

    def save(self) -> None:
        """Save binary index to disk."""
        with self._lock:
            if self._index is None:
                return
            self._path.mkdir(parents=True, exist_ok=True)
            faiss.write_index_binary(self._index, str(self._index_path))

    def __len__(self) -> int:
        with self._lock:
            if self._index is None:
                return 0
            return self._index.ntotal
