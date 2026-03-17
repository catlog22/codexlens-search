from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseANNIndex(ABC):
    """Abstract base class for approximate nearest neighbor indexes."""

    @abstractmethod
    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        """Add float32 vectors with corresponding IDs.

        Args:
            ids: shape (N,) int64
            vectors: shape (N, dim) float32
        """

    @abstractmethod
    def fine_search(
        self, query_vec: np.ndarray, top_k: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors.

        Args:
            query_vec: float32 vector of shape (dim,)
            top_k: number of results

        Returns:
            (ids, distances) as numpy arrays
        """

    @abstractmethod
    def save(self) -> None:
        """Persist index to disk."""

    @abstractmethod
    def load(self) -> None:
        """Load index from disk."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of indexed items."""


class BaseBinaryIndex(ABC):
    """Abstract base class for binary vector indexes (Hamming distance)."""

    @abstractmethod
    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        """Add float32 vectors (will be binary-quantized internally).

        Args:
            ids: shape (N,) int64
            vectors: shape (N, dim) float32
        """

    @abstractmethod
    def coarse_search(
        self, query_vec: np.ndarray, top_k: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search by Hamming distance.

        Args:
            query_vec: float32 vector of shape (dim,)
            top_k: number of results

        Returns:
            (ids, distances) sorted ascending by distance
        """

    @abstractmethod
    def save(self) -> None:
        """Persist store to disk."""

    @abstractmethod
    def load(self) -> None:
        """Load store from disk."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of stored items."""
