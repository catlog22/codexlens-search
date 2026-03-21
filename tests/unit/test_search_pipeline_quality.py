"""Unit tests for SearchPipeline quality-routed search paths."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from codexlens_search.config import Config
from codexlens_search.search.fts import FTSEngine
from codexlens_search.search.pipeline import SearchPipeline, SearchResult


def _make_fts(docs: list[tuple[int, str, str]] | None = None) -> FTSEngine:
    engine = FTSEngine(":memory:")
    if docs:
        engine.add_documents(docs)
    return engine


def _make_pipeline(
    fts: FTSEngine,
    *,
    has_vectors: bool = True,
    quality: str = "auto",
    metadata_store=None,
    graph_searcher=None,
) -> SearchPipeline:
    cfg = Config.small()
    cfg.default_search_quality = quality

    embedder = MagicMock()
    embedder.embed_single.return_value = np.random.randn(cfg.embed_dim).astype(np.float32)

    binary_store = MagicMock()
    binary_store.__len__ = MagicMock(return_value=100 if has_vectors else 0)
    binary_store.coarse_search.return_value = (
        np.array([1, 2, 3], dtype=np.int64),
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )

    ann_index = MagicMock()
    ann_index.fine_search.return_value = (
        np.array([1, 2, 3], dtype=np.int64),
        np.array([0.9, 0.8, 0.7], dtype=np.float32),
    )

    reranker = MagicMock()
    reranker.score_pairs.side_effect = lambda q, contents: [
        0.9 - i * 0.1 for i in range(len(contents))
    ]

    return SearchPipeline(
        embedder=embedder,
        binary_store=binary_store,
        ann_index=ann_index,
        reranker=reranker,
        fts=fts,
        config=cfg,
        metadata_store=metadata_store,
        graph_searcher=graph_searcher,
    )


DOCS = [
    (1, "a.py", "def authenticate user password login"),
    (2, "b.py", "connect to database with credentials"),
    (3, "c.py", "render template html response"),
]


class TestQualityRouting:
    """Test that quality parameter routes to correct search method."""

    def test_auto_with_vectors_uses_thorough(self) -> None:
        fts = _make_fts(DOCS)
        pipeline = _make_pipeline(fts, has_vectors=True, quality="auto")
        results = pipeline.search("authenticate")
        # Should use thorough path -> embedder.embed_single called
        assert pipeline._embedder.embed_single.called

    def test_auto_without_vectors_uses_fast(self) -> None:
        fts = _make_fts(DOCS)
        pipeline = _make_pipeline(fts, has_vectors=False, quality="auto")
        results = pipeline.search("authenticate")
        # Fast path should NOT call embedder
        assert not pipeline._embedder.embed_single.called

    def test_fast_never_embeds(self) -> None:
        fts = _make_fts(DOCS)
        pipeline = _make_pipeline(fts, has_vectors=True)
        results = pipeline.search("authenticate", quality="fast")
        assert not pipeline._embedder.embed_single.called

    def test_balanced_embeds_but_no_ann(self) -> None:
        fts = _make_fts(DOCS)
        pipeline = _make_pipeline(fts, has_vectors=True)
        results = pipeline.search("authenticate", quality="balanced")
        assert pipeline._embedder.embed_single.called
        # Balanced uses binary coarse but not ANN fine
        assert pipeline._binary_store.coarse_search.called
        assert not pipeline._ann_index.fine_search.called

    def test_thorough_uses_both_stages(self) -> None:
        fts = _make_fts(DOCS)
        pipeline = _make_pipeline(fts, has_vectors=True)
        results = pipeline.search("authenticate", quality="thorough")
        assert pipeline._embedder.embed_single.called
        assert pipeline._binary_store.coarse_search.called
        assert pipeline._ann_index.fine_search.called

    def test_invalid_quality_falls_back_to_auto(self) -> None:
        fts = _make_fts(DOCS)
        pipeline = _make_pipeline(fts, has_vectors=False)
        # Invalid quality should fall back to auto -> fast (no vectors)
        results = pipeline.search("authenticate", quality="invalid_tier")
        assert not pipeline._embedder.embed_single.called

    def test_none_quality_uses_config_default(self) -> None:
        fts = _make_fts(DOCS)
        pipeline = _make_pipeline(fts, has_vectors=True, quality="fast")
        results = pipeline.search("authenticate", quality=None)
        # Config default is "fast", so no embedding
        assert not pipeline._embedder.embed_single.called


class TestFilterDeleted:
    """Test _filter_deleted removes tombstoned IDs."""

    def test_filter_removes_deleted_ids(self) -> None:
        fts = _make_fts(DOCS)
        metadata = MagicMock()
        metadata.get_deleted_ids.return_value = {2}
        pipeline = _make_pipeline(fts, metadata_store=metadata)
        fused = [(1, 0.9), (2, 0.8), (3, 0.7)]
        filtered = pipeline._filter_deleted(fused)
        assert [id_ for id_, _ in filtered] == [1, 3]

    def test_filter_no_metadata_returns_all(self) -> None:
        fts = _make_fts(DOCS)
        pipeline = _make_pipeline(fts, metadata_store=None)
        fused = [(1, 0.9), (2, 0.8)]
        filtered = pipeline._filter_deleted(fused)
        assert len(filtered) == 2


class TestRecordAccess:
    """Test _record_access for tier tracking."""

    def test_record_access_calls_metadata(self) -> None:
        fts = _make_fts(DOCS)
        metadata = MagicMock()
        pipeline = _make_pipeline(fts, metadata_store=metadata)
        results = [
            SearchResult(id=1, path="a.py", score=0.9),
            SearchResult(id=2, path="a.py", score=0.8),
            SearchResult(id=3, path="b.py", score=0.7),
        ]
        pipeline._record_access(results)
        metadata.record_access_batch.assert_called_once()
        paths = metadata.record_access_batch.call_args[0][0]
        assert set(paths) == {"a.py", "b.py"}

    def test_record_access_empty_results(self) -> None:
        fts = _make_fts(DOCS)
        metadata = MagicMock()
        pipeline = _make_pipeline(fts, metadata_store=metadata)
        pipeline._record_access([])
        metadata.record_access_batch.assert_not_called()


class TestRerankAndBuild:
    """Test _rerank_and_build."""

    def test_empty_fused_returns_empty(self) -> None:
        fts = _make_fts(DOCS)
        pipeline = _make_pipeline(fts)
        results = pipeline._rerank_and_build("test", [], 10)
        assert results == []

    def test_no_reranker_preserves_order(self) -> None:
        fts = _make_fts(DOCS)
        pipeline = _make_pipeline(fts)
        fused = [(1, 0.9), (2, 0.5)]
        results = pipeline._rerank_and_build("test", fused, 10, use_reranker=False)
        assert results[0].id == 1
        assert results[1].id == 2

    def test_top_k_limits_results(self) -> None:
        fts = _make_fts(DOCS)
        pipeline = _make_pipeline(fts)
        fused = [(1, 0.9), (2, 0.8), (3, 0.7)]
        results = pipeline._rerank_and_build("test", fused, 1, use_reranker=False)
        assert len(results) == 1


class TestVectorSearch:
    """Test _vector_search two-stage funnel."""

    def test_intersection_filters(self) -> None:
        fts = _make_fts(DOCS)
        pipeline = _make_pipeline(fts)
        # Binary returns {1,2}, ANN returns [1,2,3] -> intersection = [1,2]
        pipeline._binary_store.coarse_search.return_value = (
            np.array([1, 2], dtype=np.int64),
            np.array([1.0, 2.0], dtype=np.float32),
        )
        pipeline._ann_index.fine_search.return_value = (
            np.array([1, 2, 3], dtype=np.int64),
            np.array([0.9, 0.8, 0.7], dtype=np.float32),
        )
        query_vec = np.random.randn(384).astype(np.float32)
        results = pipeline._vector_search(query_vec)
        ids = [r[0] for r in results]
        assert 3 not in ids
        assert 1 in ids and 2 in ids

    def test_empty_intersection_falls_back(self) -> None:
        fts = _make_fts(DOCS)
        pipeline = _make_pipeline(fts)
        # Binary returns {10,11}, ANN returns [1,2] -> no overlap -> fallback to ANN
        pipeline._binary_store.coarse_search.return_value = (
            np.array([10, 11], dtype=np.int64),
            np.array([1.0, 2.0], dtype=np.float32),
        )
        pipeline._ann_index.fine_search.return_value = (
            np.array([1, 2], dtype=np.int64),
            np.array([0.9, 0.8], dtype=np.float32),
        )
        query_vec = np.random.randn(384).astype(np.float32)
        results = pipeline._vector_search(query_vec)
        ids = [r[0] for r in results]
        assert 1 in ids and 2 in ids


class TestGraphSearcherIntegration:
    """Test graph searcher integration in balanced/thorough paths."""

    def test_balanced_includes_graph_results(self) -> None:
        fts = _make_fts(DOCS)
        graph = MagicMock()
        graph.search_from_chunks.return_value = [(1, 0.5)]
        pipeline = _make_pipeline(fts, graph_searcher=graph)
        results = pipeline.search("authenticate", quality="balanced")
        assert graph.search_from_chunks.called

    def test_graph_failure_does_not_crash(self) -> None:
        fts = _make_fts(DOCS)
        graph = MagicMock()
        graph.search_from_chunks.side_effect = RuntimeError("graph broken")
        pipeline = _make_pipeline(fts, graph_searcher=graph)
        # Should not raise
        results = pipeline.search("authenticate", quality="thorough")
        assert isinstance(results, list)
