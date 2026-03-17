from __future__ import annotations

from codexlens_search.config import Config
from .base import BaseReranker


class FastEmbedReranker(BaseReranker):
    """Local reranker backed by fastembed TextCrossEncoder."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            from .. import model_manager
            model_manager.ensure_model(self._config.reranker_model, self._config)

            from fastembed.rerank.cross_encoder import TextCrossEncoder
            cache_kwargs = model_manager.get_cache_kwargs(self._config)
            self._model = TextCrossEncoder(
                model_name=self._config.reranker_model,
                **cache_kwargs,
            )

    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        self._load()
        results = list(self._model.rerank(query, documents))
        if not results:
            return [0.0] * len(documents)
        # fastembed may return list[float] or list[RerankResult] depending on version
        first = results[0]
        if isinstance(first, (int, float)):
            return [float(s) for s in results]
        # Older format: objects with .index and .score
        scores = [0.0] * len(documents)
        for r in results:
            scores[r.index] = float(r.score)
        return scores
