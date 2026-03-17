from __future__ import annotations

import logging
import time

import httpx

from codexlens_search.config import Config
from .base import BaseReranker

logger = logging.getLogger(__name__)


class APIReranker(BaseReranker):
    """Reranker backed by a remote HTTP API (SiliconFlow/Cohere/Jina format)."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {config.reranker_api_key}",
                "Content-Type": "application/json",
            },
        )

    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        max_tokens = self._config.reranker_api_max_tokens_per_batch
        batches = self._split_batches(documents, max_tokens)
        scores = [0.0] * len(documents)
        for batch in batches:
            batch_scores = self._call_api_with_retry(query, batch)
            for orig_idx, score in batch_scores.items():
                scores[orig_idx] = score
        return scores

    def _split_batches(
        self, documents: list[str], max_tokens: int
    ) -> list[list[tuple[int, str]]]:
        batches: list[list[tuple[int, str]]] = []
        current_batch: list[tuple[int, str]] = []
        current_tokens = 0

        for idx, text in enumerate(documents):
            doc_tokens = len(text) // 4
            if current_tokens + doc_tokens > max_tokens and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append((idx, text))
            current_tokens += doc_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    def _call_api_with_retry(
        self,
        query: str,
        docs: list[tuple[int, str]],
        max_retries: int = 3,
    ) -> dict[int, float]:
        url = self._config.reranker_api_url.rstrip("/") + "/rerank"
        payload = {
            "model": self._config.reranker_api_model,
            "query": query,
            "documents": [t for _, t in docs],
        }

        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self._client.post(url, json=payload)
            except Exception as exc:
                last_exc = exc
                time.sleep((2 ** attempt) * 0.5)
                continue

            if response.status_code in (429, 503):
                logger.warning(
                    "API reranker returned HTTP %s (attempt %d/%d), retrying...",
                    response.status_code,
                    attempt + 1,
                    max_retries,
                )
                time.sleep((2 ** attempt) * 0.5)
                continue

            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            scores: dict[int, float] = {}
            for item in results:
                local_idx = int(item["index"])
                orig_idx = docs[local_idx][0]
                scores[orig_idx] = float(item["relevance_score"])
            return scores

        raise RuntimeError(
            f"API reranker failed after {max_retries} attempts. Last error: {last_exc}"
        )
