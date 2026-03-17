from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..core import ANNIndex, BinaryStore
from ..embed import BaseEmbedder
from ..indexing.metadata import MetadataStore
from ..rerank import BaseReranker
from .fts import FTSEngine
from .fusion import (
    DEFAULT_WEIGHTS,
    detect_query_intent,
    get_adaptive_weights,
    reciprocal_rank_fusion,
)

_log = logging.getLogger(__name__)


@dataclass
class SearchResult:
    id: int
    path: str
    score: float
    snippet: str = ""
    line: int = 0
    end_line: int = 0
    content: str = ""


class SearchPipeline:
    def __init__(
        self,
        embedder: BaseEmbedder,
        binary_store: BinaryStore,
        ann_index: ANNIndex,
        reranker: BaseReranker,
        fts: FTSEngine,
        config: Config,
        metadata_store: MetadataStore | None = None,
    ) -> None:
        self._embedder = embedder
        self._binary_store = binary_store
        self._ann_index = ann_index
        self._reranker = reranker
        self._fts = fts
        self._config = config
        self._metadata_store = metadata_store

    # -- Helper: vector search (binary coarse + ANN fine) -----------------

    def _vector_search(
        self, query_vec: np.ndarray
    ) -> list[tuple[int, float]]:
        """Run binary coarse search then ANN fine search and intersect."""
        cfg = self._config

        # Binary coarse search -> candidate_ids set
        candidate_ids_list, _ = self._binary_store.coarse_search(
            query_vec, top_k=cfg.binary_top_k
        )
        candidate_ids = set(candidate_ids_list)

        # ANN fine search on full index, then intersect with binary candidates
        ann_ids, ann_scores = self._ann_index.fine_search(
            query_vec, top_k=cfg.ann_top_k
        )
        # Keep only results that appear in binary candidates (2-stage funnel)
        vector_results: list[tuple[int, float]] = [
            (int(doc_id), float(score))
            for doc_id, score in zip(ann_ids, ann_scores)
            if int(doc_id) in candidate_ids
        ]
        # Fall back to full ANN results if intersection is empty
        if not vector_results:
            vector_results = [
                (int(doc_id), float(score))
                for doc_id, score in zip(ann_ids, ann_scores)
            ]
        return vector_results

    # -- Helper: FTS search (exact + fuzzy) ------------------------------

    def _fts_search(
        self, query: str
    ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        """Run exact and fuzzy full-text search."""
        cfg = self._config
        exact_results = self._fts.exact_search(query, top_k=cfg.fts_top_k)
        fuzzy_results = self._fts.fuzzy_search(query, top_k=cfg.fts_top_k)
        return exact_results, fuzzy_results

    # -- Main search entry point -----------------------------------------

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        cfg = self._config
        final_top_k = top_k if top_k is not None else cfg.reranker_top_k

        # 1. Detect intent -> adaptive weights
        intent = detect_query_intent(query)
        weights = get_adaptive_weights(intent, cfg.fusion_weights)

        # 2. Embed query
        query_vec = self._embedder.embed_single(query)

        # 3. Parallel vector + FTS search
        vector_results: list[tuple[int, float]] = []
        exact_results: list[tuple[int, float]] = []
        fuzzy_results: list[tuple[int, float]] = []

        with ThreadPoolExecutor(max_workers=2) as pool:
            vec_future = pool.submit(self._vector_search, query_vec)
            fts_future = pool.submit(self._fts_search, query)

            # Collect vector results
            try:
                vector_results = vec_future.result()
            except Exception:
                _log.warning("Vector search failed, using empty results", exc_info=True)

            # Collect FTS results
            try:
                exact_results, fuzzy_results = fts_future.result()
            except Exception:
                _log.warning("FTS search failed, using empty results", exc_info=True)

        # 4. RRF fusion
        fusion_input: dict[str, list[tuple[int, float]]] = {}
        if vector_results:
            fusion_input["vector"] = vector_results
        if exact_results:
            fusion_input["exact"] = exact_results
        if fuzzy_results:
            fusion_input["fuzzy"] = fuzzy_results

        if not fusion_input:
            return []

        fused = reciprocal_rank_fusion(fusion_input, weights=weights, k=cfg.fusion_k)

        # 4b. Filter out deleted IDs (tombstone filtering)
        if self._metadata_store is not None:
            deleted_ids = self._metadata_store.get_deleted_ids()
            if deleted_ids:
                fused = [
                    (doc_id, score)
                    for doc_id, score in fused
                    if doc_id not in deleted_ids
                ]

        # 5. Rerank top candidates
        rerank_ids = [doc_id for doc_id, _ in fused[:50]]
        contents = [self._fts.get_content(doc_id) for doc_id in rerank_ids]
        rerank_scores = self._reranker.score_pairs(query, contents)

        # 6. Sort by rerank score, build SearchResult list
        ranked = sorted(
            zip(rerank_ids, rerank_scores), key=lambda x: x[1], reverse=True
        )

        results: list[SearchResult] = []
        for doc_id, score in ranked[:final_top_k]:
            path, start_line, end_line = self._fts.get_doc_meta(doc_id)
            full_content = self._fts.get_content(doc_id)
            results.append(
                SearchResult(
                    id=doc_id,
                    path=path,
                    score=float(score),
                    snippet=full_content[:200],
                    line=start_line,
                    end_line=end_line,
                    content=full_content,
                )
            )
        return results
