"""Unit tests for core/usearch_index.py — UsearchANNIndex backend."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from codexlens_search.config import Config

DIM = 32


@pytest.fixture
def cfg():
    return Config.small()


@pytest.fixture
def usearch_index(tmp_path, cfg):
    from codexlens_search.core.usearch_index import UsearchANNIndex
    return UsearchANNIndex(tmp_path / "usearch", DIM, cfg)


class TestUsearchANNIndex:
    """Core functionality tests for UsearchANNIndex."""

    def test_init_creates_empty_index(self, usearch_index):
        assert len(usearch_index) == 0

    def test_add_increases_count(self, usearch_index):
        ids = np.arange(5, dtype=np.int64)
        vecs = np.random.default_rng(42).standard_normal((5, DIM)).astype(np.float32)
        usearch_index.add(ids, vecs)
        assert len(usearch_index) == 5

    def test_add_empty_is_noop(self, usearch_index):
        usearch_index.add(np.array([], dtype=np.int64), np.empty((0, DIM), dtype=np.float32))
        assert len(usearch_index) == 0

    def test_fine_search_empty_returns_empty(self, usearch_index):
        q = np.random.default_rng(0).standard_normal(DIM).astype(np.float32)
        ids, dists = usearch_index.fine_search(q, top_k=5)
        assert len(ids) == 0
        assert len(dists) == 0

    def test_fine_search_finds_self(self, usearch_index):
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((10, DIM)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
        ids = np.arange(10, dtype=np.int64)
        usearch_index.add(ids, vecs)

        result_ids, result_dists = usearch_index.fine_search(vecs[3], top_k=3)
        assert len(result_ids) == 3
        assert 3 in result_ids
        # Self-match should be first (smallest distance)
        assert result_ids[0] == 3

    def test_fine_search_top_k_clamps_to_count(self, usearch_index):
        ids = np.arange(3, dtype=np.int64)
        vecs = np.random.default_rng(42).standard_normal((3, DIM)).astype(np.float32)
        usearch_index.add(ids, vecs)

        result_ids, _ = usearch_index.fine_search(vecs[0], top_k=100)
        assert len(result_ids) == 3

    def test_fine_search_uses_config_default_k(self, tmp_path, cfg):
        cfg.ann_top_k = 2
        from codexlens_search.core.usearch_index import UsearchANNIndex
        idx = UsearchANNIndex(tmp_path / "usearch_k", DIM, cfg)

        ids = np.arange(10, dtype=np.int64)
        vecs = np.random.default_rng(42).standard_normal((10, DIM)).astype(np.float32)
        idx.add(ids, vecs)

        result_ids, _ = idx.fine_search(vecs[0])
        assert len(result_ids) == 2

    def test_add_multiple_batches(self, usearch_index):
        rng = np.random.default_rng(42)
        ids1 = np.arange(5, dtype=np.int64)
        vecs1 = rng.standard_normal((5, DIM)).astype(np.float32)
        usearch_index.add(ids1, vecs1)

        ids2 = np.arange(5, 10, dtype=np.int64)
        vecs2 = rng.standard_normal((5, DIM)).astype(np.float32)
        usearch_index.add(ids2, vecs2)

        assert len(usearch_index) == 10


class TestUsearchPersistence:
    """Save/load persistence tests."""

    def test_save_and_load(self, tmp_path, cfg):
        from codexlens_search.core.usearch_index import UsearchANNIndex

        path = tmp_path / "usearch_persist"
        idx = UsearchANNIndex(path, DIM, cfg)

        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((10, DIM)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
        ids = np.arange(10, dtype=np.int64)
        idx.add(ids, vecs)
        idx.save()

        # Reload into new instance
        idx2 = UsearchANNIndex(path, DIM, cfg)
        idx2.load()
        assert len(idx2) == 10

        # Search should still work
        result_ids, _ = idx2.fine_search(vecs[0], top_k=3)
        assert 0 in result_ids

    def test_save_empty_noop(self, tmp_path, cfg):
        from codexlens_search.core.usearch_index import UsearchANNIndex
        idx = UsearchANNIndex(tmp_path / "usearch_empty", DIM, cfg)
        # No load, no add — save should not crash
        idx.save()

    def test_load_creates_fresh_when_no_file(self, tmp_path, cfg):
        from codexlens_search.core.usearch_index import UsearchANNIndex
        idx = UsearchANNIndex(tmp_path / "usearch_fresh", DIM, cfg)
        idx.load()
        assert len(idx) == 0


class TestUsearchImportGuard:
    """Test ImportError when usearch is not installed."""

    @patch("codexlens_search.core.usearch_index._USEARCH_AVAILABLE", False)
    def test_raises_import_error(self, tmp_path, cfg):
        from codexlens_search.core.usearch_index import UsearchANNIndex
        with pytest.raises(ImportError, match="usearch"):
            UsearchANNIndex(tmp_path / "no_usearch", DIM, cfg)


class TestUsearchFactoryIntegration:
    """Test usearch via factory create_ann_index."""

    def test_explicit_usearch_backend(self, tmp_path, cfg):
        cfg.ann_backend = "usearch"
        from codexlens_search.core.factory import create_ann_index
        from codexlens_search.core.usearch_index import UsearchANNIndex
        idx = create_ann_index(tmp_path / "factory_usearch", DIM, cfg)
        assert isinstance(idx, UsearchANNIndex)

    def test_auto_selects_usearch_when_available(self, tmp_path, cfg):
        cfg.ann_backend = "auto"
        with patch("codexlens_search.core.factory._USEARCH_AVAILABLE", True):
            from codexlens_search.core.factory import create_ann_index
            from codexlens_search.core.usearch_index import UsearchANNIndex
            idx = create_ann_index(tmp_path / "factory_auto", DIM, cfg)
            assert isinstance(idx, UsearchANNIndex)

    @patch("codexlens_search.core.factory._USEARCH_AVAILABLE", False)
    @patch("codexlens_search.core.factory._FAISS_AVAILABLE", False)
    @patch("codexlens_search.core.factory._HNSWLIB_AVAILABLE", True)
    def test_auto_falls_through_to_hnswlib(self, tmp_path, cfg):
        cfg.ann_backend = "auto"
        from codexlens_search.core.factory import create_ann_index
        from codexlens_search.core.index import ANNIndex
        idx = create_ann_index(tmp_path / "factory_fallback", DIM, cfg)
        assert isinstance(idx, ANNIndex)

    def test_usearch_add_search_via_factory(self, tmp_path, cfg):
        cfg.ann_backend = "usearch"
        from codexlens_search.core.factory import create_ann_index

        idx = create_ann_index(tmp_path / "factory_e2e", DIM, cfg)
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((10, DIM)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
        ids = np.arange(10, dtype=np.int64)

        idx.add(ids, vecs)
        assert len(idx) == 10

        result_ids, result_dists = idx.fine_search(vecs[0], top_k=5)
        assert len(result_ids) == 5
        assert 0 in result_ids


class TestUsearchInSearchPipeline:
    """Test usearch backend works in full SearchPipeline."""

    def test_pipeline_with_usearch_backend(self, tmp_path):
        from codexlens_search.core.factory import create_ann_index, create_binary_index
        from codexlens_search.search.fts import FTSEngine
        from codexlens_search.search.pipeline import SearchPipeline

        # Import test fixtures from integration conftest
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from integration.conftest import MockEmbedder, MockReranker, TEST_DOCS

        config = Config(
            embed_dim=DIM,
            ann_backend="usearch",
            binary_backend="hnswlib",
            hnsw_ef=50,
            hnsw_M=16,
            binary_top_k=50,
            ann_top_k=20,
            reranker_top_k=10,
        )

        embedder = MockEmbedder()
        reranker = MockReranker()
        binary_store = create_binary_index(tmp_path / "bin", DIM, config)
        ann_index = create_ann_index(tmp_path / "ann", DIM, config)
        fts = FTSEngine(tmp_path / "fts.db")

        ids = np.array([d[0] for d in TEST_DOCS], dtype=np.int64)
        vectors = np.array(
            [embedder.embed_single(d[2]) for d in TEST_DOCS], dtype=np.float32
        )

        binary_store.add(ids, vectors)
        ann_index.add(ids, vectors)
        fts.add_documents(TEST_DOCS)

        pipeline = SearchPipeline(
            embedder=embedder,
            binary_store=binary_store,
            ann_index=ann_index,
            reranker=reranker,
            fts=fts,
            config=config,
        )

        results = pipeline.search("authenticate user")
        assert len(results) > 0
        pipeline.close()
