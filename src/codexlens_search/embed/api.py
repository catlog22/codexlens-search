from __future__ import annotations

import itertools
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import numpy as np

from ..config import Config
from .base import BaseEmbedder

logger = logging.getLogger(__name__)


class _Endpoint:
    """A single API endpoint with its own client and rate-limit tracking."""

    __slots__ = ("url", "key", "model", "client", "failures", "lock")

    def __init__(self, url: str, key: str, model: str) -> None:
        self.url = url.rstrip("/")
        if not self.url.endswith("/embeddings"):
            self.url += "/embeddings"
        self.key = key
        self.model = model
        self.client = httpx.Client(
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
        self.failures = 0
        self.lock = threading.Lock()


class APIEmbedder(BaseEmbedder):
    """Embedder backed by remote HTTP API (OpenAI /v1/embeddings format).

    Features:
    - Token packing: packs small chunks into batches up to max_tokens_per_batch
    - Multi-endpoint: round-robins across multiple (url, key) pairs
    - Concurrent dispatch: parallel API calls via ThreadPoolExecutor
    - Per-endpoint failure tracking and retry with backoff
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._endpoints = self._build_endpoints(config)
        self._cycler = itertools.cycle(range(len(self._endpoints)))
        self._cycler_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=min(config.embed_api_concurrency, len(self._endpoints) * 2),
        )

    @staticmethod
    def _build_endpoints(config: Config) -> list[_Endpoint]:
        """Build endpoint list from config. Supports both single and multi configs."""
        endpoints: list[_Endpoint] = []

        # Multi-endpoint config takes priority
        if config.embed_api_endpoints:
            for ep in config.embed_api_endpoints:
                endpoints.append(_Endpoint(
                    url=ep.get("url", config.embed_api_url),
                    key=ep.get("key", config.embed_api_key),
                    model=ep.get("model", config.embed_api_model),
                ))

        # Fallback: single endpoint from top-level config
        if not endpoints and config.embed_api_url:
            endpoints.append(_Endpoint(
                url=config.embed_api_url,
                key=config.embed_api_key,
                model=config.embed_api_model,
            ))

        if not endpoints:
            raise ValueError("No API embedding endpoints configured")

        return endpoints

    def _next_endpoint(self) -> _Endpoint:
        with self._cycler_lock:
            idx = next(self._cycler)
        return self._endpoints[idx]

    # -- Token packing ------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token for code."""
        return max(1, len(text) // 4)

    def _pack_batches(
        self, texts: list[str]
    ) -> list[list[tuple[int, str]]]:
        """Pack texts into batches respecting max_tokens_per_batch.

        Returns list of batches, each batch is list of (original_index, text).
        Also respects embed_batch_size as max items per batch.
        """
        max_tokens = self._config.embed_api_max_tokens_per_batch
        max_items = self._config.embed_batch_size
        batches: list[list[tuple[int, str]]] = []
        current: list[tuple[int, str]] = []
        current_tokens = 0

        for i, text in enumerate(texts):
            tokens = self._estimate_tokens(text)
            # Start new batch if adding this text would exceed limits
            if current and (
                current_tokens + tokens > max_tokens
                or len(current) >= max_items
            ):
                batches.append(current)
                current = []
                current_tokens = 0
            current.append((i, text))
            current_tokens += tokens

        if current:
            batches.append(current)

        return batches

    # -- API call with retry ------------------------------------------

    def _call_api(
        self,
        texts: list[str],
        endpoint: _Endpoint,
        max_retries: int = 3,
    ) -> list[np.ndarray]:
        """Call a single endpoint with retry logic."""
        payload: dict = {"input": texts}
        if endpoint.model:
            payload["model"] = endpoint.model

        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = endpoint.client.post(endpoint.url, json=payload)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "API embed %s failed (attempt %d/%d): %s",
                    endpoint.url, attempt + 1, max_retries, exc,
                )
                time.sleep((2 ** attempt) * 0.5)
                continue

            if response.status_code in (429, 503):
                logger.warning(
                    "API embed %s returned HTTP %s (attempt %d/%d), retrying...",
                    endpoint.url, response.status_code, attempt + 1, max_retries,
                )
                time.sleep((2 ** attempt) * 0.5)
                continue

            response.raise_for_status()
            data = response.json()

            items = data.get("data", [])
            items.sort(key=lambda x: x["index"])
            vectors = [
                np.array(item["embedding"], dtype=np.float32)
                for item in items
            ]

            # Reset failure counter on success
            with endpoint.lock:
                endpoint.failures = 0

            return vectors

        # Track failures
        with endpoint.lock:
            endpoint.failures += 1

        raise RuntimeError(
            f"API embed failed at {endpoint.url} after {max_retries} attempts. "
            f"Last error: {last_exc}"
        )

    # -- Public interface ---------------------------------------------

    def embed_single(self, text: str) -> np.ndarray:
        endpoint = self._next_endpoint()
        vecs = self._call_api([text], endpoint)
        return vecs[0]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []

        # 1. Pack into token-aware batches
        packed = self._pack_batches(texts)

        if len(packed) == 1:
            # Single batch — no concurrency overhead needed
            batch_texts = [t for _, t in packed[0]]
            batch_indices = [i for i, _ in packed[0]]
            endpoint = self._next_endpoint()
            vecs = self._call_api(batch_texts, endpoint)
            results: dict[int, np.ndarray] = {}
            for idx, vec in zip(batch_indices, vecs):
                results[idx] = vec
            return [results[i] for i in range(len(texts))]

        # 2. Dispatch batches concurrently across endpoints
        results: dict[int, np.ndarray] = {}
        futures = []
        batch_index_map: list[list[int]] = []

        for batch in packed:
            batch_texts = [t for _, t in batch]
            batch_indices = [i for i, _ in batch]
            endpoint = self._next_endpoint()
            future = self._executor.submit(self._call_api, batch_texts, endpoint)
            futures.append(future)
            batch_index_map.append(batch_indices)

        for future, indices in zip(futures, batch_index_map):
            vecs = future.result()  # propagates exceptions
            for idx, vec in zip(indices, vecs):
                results[idx] = vec

        return [results[i] for i in range(len(texts))]
