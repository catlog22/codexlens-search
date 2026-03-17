from __future__ import annotations

import numpy as np

from ..config import Config
from .base import BaseEmbedder

EMBED_PROFILES = {
    "small": "BAAI/bge-small-en-v1.5",                    # 384d
    "base": "BAAI/bge-base-en-v1.5",                      # 768d
    "large": "BAAI/bge-large-en-v1.5",                    # 1024d
    "code": "jinaai/jina-embeddings-v2-base-code",         # 768d
}


class FastEmbedEmbedder(BaseEmbedder):
    """Embedder backed by fastembed.TextEmbedding with lazy model loading."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._model = None

    def _load(self) -> None:
        """Lazy-load the fastembed TextEmbedding model on first use."""
        if self._model is not None:
            return
        from .. import model_manager
        model_manager.ensure_model(self._config.embed_model, self._config)

        from fastembed import TextEmbedding
        providers = self._config.resolve_embed_providers()
        cache_kwargs = model_manager.get_cache_kwargs(self._config)
        try:
            self._model = TextEmbedding(
                model_name=self._config.embed_model,
                providers=providers,
                **cache_kwargs,
            )
        except TypeError:
            self._model = TextEmbedding(
                model_name=self._config.embed_model,
                **cache_kwargs,
            )

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text, returns float32 ndarray of shape (dim,)."""
        self._load()
        result = list(self._model.embed([text]))
        return result[0].astype(np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a list of texts in batches, returns list of float32 ndarrays."""
        self._load()
        batch_size = self._config.embed_batch_size
        results: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            for vec in self._model.embed(batch):
                results.append(vec.astype(np.float32))
        return results
