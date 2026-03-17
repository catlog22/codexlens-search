from __future__ import annotations
from abc import ABC, abstractmethod


class BaseReranker(ABC):
    @abstractmethod
    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        """Score (query, doc) pairs. Returns list of floats same length as documents."""
