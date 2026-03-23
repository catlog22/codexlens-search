from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..core.base import BaseANNIndex, BaseBinaryIndex
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
from .expansion import QueryExpander
from .graph import GraphSearcher

_log = logging.getLogger(__name__)

_VALID_QUALITIES = ("fast", "balanced", "thorough", "auto")


@dataclass
class SearchResult:
    id: int
    path: str
    score: float
    snippet: str = ""
    line: int = 0
    end_line: int = 0
    content: str = ""
    language: str = ""


class SearchPipeline:
    def __init__(
        self,
        embedder: BaseEmbedder,
        binary_store: BaseBinaryIndex,
        ann_index: BaseANNIndex,
        reranker: BaseReranker,
        fts: FTSEngine,
        config: Config,
        metadata_store: MetadataStore | None = None,
        graph_searcher: GraphSearcher | None = None,
        query_expander: QueryExpander | None = None,
    ) -> None:
        self._embedder = embedder
        self._binary_store = binary_store
        self._ann_index = ann_index
        self._reranker = reranker
        self._fts = fts
        self._config = config
        self._metadata_store = metadata_store
        self._graph_searcher = graph_searcher
        self._query_expander = query_expander

    def close(self) -> None:
        """Close owned FTS and metadata connections."""
        if self._fts is not None:
            self._fts.close()
        if self._metadata_store is not None:
            self._metadata_store.close()

    # -- Helper: check if vector index has data ----------------------------

    def _has_vector_index(self) -> bool:
        """Check if the binary store has any indexed entries.

        Triggers lazy-load if needed so that a freshly-created
        SearchPipeline correctly detects an on-disk index.
        """
        try:
            if hasattr(self._binary_store, "_ensure_loaded"):
                self._binary_store._ensure_loaded()
            return len(self._binary_store) > 0
        except Exception:
            return False

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

    # -- Helper: binary coarse search only --------------------------------

    def _binary_coarse_search(
        self, query_vec: np.ndarray
    ) -> list[tuple[int, float]]:
        """Run binary coarse search only (no ANN fine search)."""
        cfg = self._config
        candidate_ids, distances = self._binary_store.coarse_search(
            query_vec, top_k=cfg.binary_top_k
        )
        return [
            (int(doc_id), float(dist))
            for doc_id, dist in zip(candidate_ids, distances)
        ]

    # -- Helper: FTS search (exact + fuzzy) ------------------------------

    def _fts_search(
        self, query: str
    ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        """Run exact and fuzzy full-text search."""
        cfg = self._config
        exact_results = self._fts.exact_search(query, top_k=cfg.fts_top_k)
        fuzzy_results = self._fts.fuzzy_search(query, top_k=cfg.fts_top_k)
        return exact_results, fuzzy_results

    # -- Helper: filter deleted IDs ---------------------------------------

    def _filter_deleted(
        self, fused: list[tuple[int, float]]
    ) -> list[tuple[int, float]]:
        """Remove tombstoned chunk IDs from results."""
        if self._metadata_store is not None:
            deleted_ids = self._metadata_store.get_deleted_ids()
            if deleted_ids:
                fused = [
                    (doc_id, score)
                    for doc_id, score in fused
                    if doc_id not in deleted_ids
                ]
        return fused

    # -- Helper: rerank and build results ---------------------------------

    def _rerank_and_build(
        self,
        query: str,
        fused: list[tuple[int, float]],
        final_top_k: int,
        use_reranker: bool = True,
    ) -> list[SearchResult]:
        """Rerank candidates (optionally) and build SearchResult list."""
        if not fused:
            return []

        if use_reranker:
            rerank_ids = [doc_id for doc_id, _ in fused[:50]]
            contents = [self._fts.get_content(doc_id) for doc_id in rerank_ids]
            rerank_scores = self._reranker.score_pairs(query, contents)
            ranked = sorted(
                zip(rerank_ids, rerank_scores), key=lambda x: x[1], reverse=True
            )
        else:
            ranked = fused

        results: list[SearchResult] = []
        for doc_id, score in ranked[:final_top_k]:
            path, start_line, end_line, language = self._fts.get_doc_meta(doc_id)
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
                    language=language,
                )
            )
        return results

    # -- Helper: record access for tier tracking --------------------------

    def _record_access(self, results: list[SearchResult]) -> None:
        """Record file access for data tier tracking."""
        if results and self._metadata_store is not None:
            unique_paths = list({r.path for r in results})
            try:
                self._metadata_store.record_access_batch(unique_paths)
            except Exception:
                _log.debug("Failed to record access for tier tracking", exc_info=True)

    # -- Helper: collect top chunk IDs for graph seeding ------------------

    @staticmethod
    def _collect_top_chunk_ids(
        fusion_input: dict[str, list[tuple[int, float]]],
        max_seeds: int = 10,
    ) -> list[int]:
        """Extract top chunk IDs from vector/FTS results for graph seed discovery."""
        scored: dict[int, float] = {}
        for results in fusion_input.values():
            for doc_id, score in results:
                if doc_id not in scored or score > scored[doc_id]:
                    scored[doc_id] = score
        ranked = sorted(scored, key=scored.get, reverse=True)  # type: ignore[arg-type]
        return ranked[:max_seeds]

    # -- Quality-routed search methods ------------------------------------

    def _search_fast(
        self, query: str, final_top_k: int
    ) -> list[SearchResult]:
        """FTS-only search with reranking. No embedding needed."""
        exact_results, fuzzy_results = self._fts_search(query)

        fusion_input: dict[str, list[tuple[int, float]]] = {}
        if exact_results:
            fusion_input["exact"] = exact_results
        if fuzzy_results:
            fusion_input["fuzzy"] = fuzzy_results

        if not fusion_input:
            return []

        fused = reciprocal_rank_fusion(
            fusion_input, weights={"exact": 0.7, "fuzzy": 0.3},
            k=self._config.fusion_k,
        )
        fused = self._filter_deleted(fused)
        return self._rerank_and_build(query, fused, final_top_k, use_reranker=True)

    def _search_balanced(
        self, query: str, final_top_k: int
    ) -> list[SearchResult]:
        """FTS + binary coarse search with RRF fusion and reranking.

        Embeds the query for binary coarse search but skips ANN fine search.
        """
        intent = detect_query_intent(query)
        weights = get_adaptive_weights(intent, self._config.fusion_weights)

        query_vec = self._embedder.embed_single(query)

        # Parallel: binary coarse + FTS
        coarse_results: list[tuple[int, float]] = []
        exact_results: list[tuple[int, float]] = []
        fuzzy_results: list[tuple[int, float]] = []

        with ThreadPoolExecutor(max_workers=2) as pool:
            coarse_future = pool.submit(self._binary_coarse_search, query_vec)
            fts_future = pool.submit(self._fts_search, query)

            try:
                coarse_results = coarse_future.result()
            except Exception:
                _log.warning("Binary coarse search failed", exc_info=True)

            try:
                exact_results, fuzzy_results = fts_future.result()
            except Exception:
                _log.warning("FTS search failed", exc_info=True)

        fusion_input: dict[str, list[tuple[int, float]]] = {}
        if coarse_results:
            fusion_input["vector"] = coarse_results
        if exact_results:
            fusion_input["exact"] = exact_results
        if fuzzy_results:
            fusion_input["fuzzy"] = fuzzy_results

        # Graph search: seed from top vector/FTS chunks
        if self._graph_searcher is not None:
            try:
                seed_ids = self._collect_top_chunk_ids(fusion_input)
                if seed_ids:
                    graph_results = self._graph_searcher.search_from_chunks(seed_ids)
                    if graph_results:
                        fusion_input["graph"] = graph_results
            except Exception:
                _log.warning("Graph search failed", exc_info=True)

        if not fusion_input:
            return []

        fused = reciprocal_rank_fusion(fusion_input, weights=weights, k=self._config.fusion_k)
        fused = self._filter_deleted(fused)
        return self._rerank_and_build(query, fused, final_top_k, use_reranker=True)

    def _search_thorough(
        self, query: str, final_top_k: int
    ) -> list[SearchResult]:
        """Full 2-stage vector + FTS + reranking pipeline (original behavior)."""
        cfg = self._config

        intent = detect_query_intent(query)
        weights = get_adaptive_weights(intent, cfg.fusion_weights)

        query_vec = self._embedder.embed_single(query)

        # Parallel vector + FTS search
        vector_results: list[tuple[int, float]] = []
        exact_results: list[tuple[int, float]] = []
        fuzzy_results: list[tuple[int, float]] = []

        with ThreadPoolExecutor(max_workers=2) as pool:
            vec_future = pool.submit(self._vector_search, query_vec)
            fts_future = pool.submit(self._fts_search, query)

            try:
                vector_results = vec_future.result()
            except Exception:
                _log.warning("Vector search failed, using empty results", exc_info=True)

            try:
                exact_results, fuzzy_results = fts_future.result()
            except Exception:
                _log.warning("FTS search failed, using empty results", exc_info=True)

        fusion_input: dict[str, list[tuple[int, float]]] = {}
        if vector_results:
            fusion_input["vector"] = vector_results
        if exact_results:
            fusion_input["exact"] = exact_results
        if fuzzy_results:
            fusion_input["fuzzy"] = fuzzy_results

        # Graph search: seed from top vector/FTS chunks
        if self._graph_searcher is not None:
            try:
                seed_ids = self._collect_top_chunk_ids(fusion_input)
                if seed_ids:
                    graph_results = self._graph_searcher.search_from_chunks(seed_ids)
                    if graph_results:
                        fusion_input["graph"] = graph_results
            except Exception:
                _log.warning("Graph search failed", exc_info=True)

        if not fusion_input:
            return []

        fused = reciprocal_rank_fusion(fusion_input, weights=weights, k=cfg.fusion_k)
        fused = self._filter_deleted(fused)
        return self._rerank_and_build(query, fused, final_top_k, use_reranker=True)

    # -- Main search entry point -----------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        quality: str | None = None,
    ) -> list[SearchResult]:
        """Search with quality-based routing.

        Args:
            query: Search query string.
            top_k: Maximum results to return.
            quality: Search quality tier:
                - 'fast': FTS-only + rerank (no embedding, no vector search)
                - 'balanced': FTS + binary coarse + rerank (no ANN fine search)
                - 'thorough': Full 2-stage vector + FTS + reranking
                - 'auto': Selects 'thorough' if vectors exist, else 'fast'
                - None: Uses config.default_search_quality

        Returns:
            List of SearchResult ordered by relevance.
        """
        cfg = self._config
        final_top_k = top_k if top_k is not None else cfg.reranker_top_k

        # Query expansion: inject code tokens for natural language queries
        if self._query_expander is not None and cfg.expansion_enabled:
            try:
                query = self._query_expander.expand(query)
            except Exception:
                _log.warning("Query expansion failed", exc_info=True)

        # Resolve quality tier
        effective_quality = quality or cfg.default_search_quality
        if effective_quality not in _VALID_QUALITIES:
            _log.warning(
                "Invalid search quality '%s', falling back to 'auto'",
                effective_quality,
            )
            effective_quality = "auto"

        # Auto-detect: use thorough if vector index has data, else fast
        if effective_quality == "auto":
            effective_quality = "thorough" if self._has_vector_index() else "fast"

        if effective_quality == "fast":
            results = self._search_fast(query, final_top_k)
        elif effective_quality == "balanced":
            results = self._search_balanced(query, final_top_k)
        else:
            results = self._search_thorough(query, final_top_k)

        self._record_access(results)
        return results
