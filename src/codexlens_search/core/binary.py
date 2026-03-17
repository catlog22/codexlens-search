from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from codexlens_search.config import Config
from codexlens_search.core.base import BaseBinaryIndex

logger = logging.getLogger(__name__)


class BinaryStore(BaseBinaryIndex):
    """Persistent binary vector store using numpy memmap.

    Stores binary-quantized float32 vectors as packed uint8 arrays on disk.
    Supports fast coarse search via XOR + popcount Hamming distance.
    """

    def __init__(self, path: str | Path, dim: int, config: Config) -> None:
        self._dir = Path(path)
        self._dim = dim
        self._config = config
        self._packed_bytes = math.ceil(dim / 8)

        self._bin_path = self._dir / "binary_store.bin"
        self._ids_path = self._dir / "binary_store_ids.npy"

        self._matrix: np.ndarray | None = None  # shape (N, packed_bytes), uint8
        self._ids: np.ndarray | None = None      # shape (N,), int64
        self._count: int = 0

        if self._bin_path.exists() and self._ids_path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _quantize(self, vectors: np.ndarray) -> np.ndarray:
        """Convert float32 vectors (N, dim) to packed uint8 (N, packed_bytes)."""
        binary = (vectors > 0).astype(np.uint8)
        packed = np.packbits(binary, axis=1)
        return packed

    def _quantize_single(self, vec: np.ndarray) -> np.ndarray:
        """Convert a single float32 vector (dim,) to packed uint8 (packed_bytes,)."""
        binary = (vec > 0).astype(np.uint8)
        return np.packbits(binary)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _ensure_capacity(self, needed: int) -> None:
        """Grow pre-allocated matrix/ids arrays to fit *needed* total items."""
        if self._matrix is not None and self._matrix.shape[0] >= needed:
            return

        new_cap = max(1024, needed)
        # Double until large enough
        if self._matrix is not None:
            cur_cap = self._matrix.shape[0]
            new_cap = max(cur_cap, 1024)
            while new_cap < needed:
                new_cap *= 2

        new_matrix = np.zeros((new_cap, self._packed_bytes), dtype=np.uint8)
        new_ids = np.zeros(new_cap, dtype=np.int64)

        if self._matrix is not None and self._count > 0:
            new_matrix[: self._count] = self._matrix[: self._count]
            new_ids[: self._count] = self._ids[: self._count]

        self._matrix = new_matrix
        self._ids = new_ids

    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        """Add float32 vectors and their ids.

        Does NOT call save() internally -- callers must call save()
        explicitly after batch indexing.

        Args:
            ids: shape (N,) int64
            vectors: shape (N, dim) float32
        """
        if len(ids) == 0:
            return

        packed = self._quantize(vectors)  # (N, packed_bytes)
        n = len(ids)

        self._ensure_capacity(self._count + n)
        self._matrix[self._count : self._count + n] = packed
        self._ids[self._count : self._count + n] = ids.astype(np.int64)
        self._count += n

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
        if self._matrix is None or self._count == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int32)

        k = top_k if top_k is not None else self._config.binary_top_k
        k = min(k, self._count)

        query_bin = self._quantize_single(query_vec)  # (packed_bytes,)

        # Slice to active region (matrix may be pre-allocated larger)
        active_matrix = self._matrix[: self._count]
        active_ids = self._ids[: self._count]

        # XOR then popcount via unpackbits
        xor = np.bitwise_xor(active_matrix, query_bin[np.newaxis, :])  # (N, packed_bytes)
        dists = np.unpackbits(xor, axis=1).sum(axis=1).astype(np.int32)  # (N,)

        if k >= self._count:
            order = np.argsort(dists)
        else:
            part = np.argpartition(dists, k)[:k]
            order = part[np.argsort(dists[part])]

        return active_ids[order], dists[order]

    def save(self) -> None:
        """Flush binary store to disk."""
        if self._matrix is None or self._count == 0:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        # Write only the occupied portion of the pre-allocated matrix
        active_matrix = self._matrix[: self._count]
        mm = np.memmap(
            str(self._bin_path),
            dtype=np.uint8,
            mode="w+",
            shape=active_matrix.shape,
        )
        mm[:] = active_matrix
        mm.flush()
        del mm
        np.save(str(self._ids_path), self._ids[: self._count])

    def load(self) -> None:
        """Reload binary store from disk."""
        ids = np.load(str(self._ids_path))
        n = len(ids)
        if n == 0:
            return
        mm = np.memmap(
            str(self._bin_path),
            dtype=np.uint8,
            mode="r",
            shape=(n, self._packed_bytes),
        )
        self._matrix = np.array(mm)  # copy into RAM for mutation support
        del mm
        self._ids = ids.astype(np.int64)
        self._count = n

    def __len__(self) -> int:
        return self._count
