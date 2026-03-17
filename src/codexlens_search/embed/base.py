from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text, returns float32 ndarray shape (dim,)."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a list of texts, returns list of float32 ndarrays."""
