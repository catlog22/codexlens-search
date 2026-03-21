"""L2 integration tests for core/factory.py backend selection and fallback.

Tests create_ann_index and create_binary_index with real hnswlib backend
and mocked faiss availability.

Targets: core/factory.py coverage from 54% toward 80%+.
"""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from codexlens_search.config import Config
from codexlens_search.core.base import BaseANNIndex, BaseBinaryIndex
from codexlens_search.core.binary import BinaryStore
from codexlens_search.core.index import ANNIndex

from tests.integration.conftest import DIM


@pytest.fixture
def factory_config():
    return Config(embed_dim=DIM, hnsw_ef=50, hnsw_M=16, binary_top_k=50, ann_top_k=20)


class TestCreateANNIndex:
    """Test create_ann_index with various backend configs."""

    def test_hnswlib_explicit(self, tmp_path, factory_config):
        factory_config.ann_backend = "hnswlib"

        from codexlens_search.core.factory import create_ann_index
        idx = create_ann_index(tmp_path / "ann", DIM, factory_config)

        assert isinstance(idx, ANNIndex)

    def test_auto_selects_hnswlib_when_faiss_unavailable(self, tmp_path, factory_config):
        factory_config.ann_backend = "auto"

        with mock.patch("codexlens_search.core.factory._FAISS_AVAILABLE", False):
            with mock.patch("codexlens_search.core.factory._HNSWLIB_AVAILABLE", True):
                from codexlens_search.core.factory import create_ann_index
                idx = create_ann_index(tmp_path / "ann_auto", DIM, factory_config)
                assert isinstance(idx, ANNIndex)

    def test_auto_no_backend_raises(self, tmp_path, factory_config):
        factory_config.ann_backend = "auto"

        with mock.patch("codexlens_search.core.factory._FAISS_AVAILABLE", False):
            with mock.patch("codexlens_search.core.factory._HNSWLIB_AVAILABLE", False):
                from codexlens_search.core.factory import create_ann_index
                with pytest.raises(ImportError, match="No ANN backend"):
                    create_ann_index(tmp_path / "ann_none", DIM, factory_config)

    def test_faiss_explicit_when_available(self, tmp_path, factory_config):
        """When faiss is explicitly requested and available, should use FAISSANNIndex."""
        factory_config.ann_backend = "faiss"

        try:
            import faiss  # noqa: F401
        except ImportError:
            pytest.skip("faiss not installed")

        from codexlens_search.core.factory import create_ann_index
        from codexlens_search.core.faiss_index import FAISSANNIndex

        idx = create_ann_index(tmp_path / "ann_faiss", DIM, factory_config)
        assert isinstance(idx, FAISSANNIndex)

    def test_hnswlib_ann_index_add_and_search(self, tmp_path, factory_config):
        """Integration: create via factory, add vectors, search."""
        factory_config.ann_backend = "hnswlib"

        from codexlens_search.core.factory import create_ann_index
        idx = create_ann_index(tmp_path / "ann_int", DIM, factory_config)

        # Add vectors
        rng = np.random.default_rng(42)
        ids = np.arange(10, dtype=np.int64)
        vecs = rng.standard_normal((10, DIM)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms

        idx.add(ids, vecs)

        # Search
        query = vecs[0]
        result_ids, scores = idx.fine_search(query, top_k=5)
        assert len(result_ids) > 0
        assert 0 in result_ids  # should find itself


class TestCreateBinaryIndex:
    """Test create_binary_index with various backend configs."""

    def test_hnswlib_explicit(self, tmp_path, factory_config):
        factory_config.binary_backend = "hnswlib"

        from codexlens_search.core.factory import create_binary_index
        idx = create_binary_index(tmp_path / "bin_hnsw", DIM, factory_config)

        assert isinstance(idx, BinaryStore)

    def test_auto_falls_back_to_binarystore_without_faiss(self, tmp_path, factory_config):
        factory_config.binary_backend = "auto"

        with mock.patch("codexlens_search.core.factory._FAISS_AVAILABLE", False):
            from codexlens_search.core.factory import create_binary_index
            with pytest.warns(DeprecationWarning, match="BinaryStore"):
                idx = create_binary_index(tmp_path / "bin_auto", DIM, factory_config)
            assert isinstance(idx, BinaryStore)

    def test_faiss_explicit_but_unavailable_warns(self, tmp_path, factory_config):
        """When faiss is explicitly requested but not available, should warn and fallback."""
        factory_config.binary_backend = "faiss"

        with mock.patch("codexlens_search.core.factory._FAISS_AVAILABLE", False):
            from codexlens_search.core.factory import create_binary_index
            with pytest.warns(DeprecationWarning, match="faiss"):
                idx = create_binary_index(tmp_path / "bin_faiss_fb", DIM, factory_config)
            assert isinstance(idx, BinaryStore)

    def test_faiss_explicit_when_available(self, tmp_path, factory_config):
        """When faiss is available, should use FAISSBinaryIndex."""
        factory_config.binary_backend = "faiss"

        try:
            import faiss  # noqa: F401
        except ImportError:
            pytest.skip("faiss not installed")

        from codexlens_search.core.factory import create_binary_index
        from codexlens_search.core.faiss_index import FAISSBinaryIndex

        idx = create_binary_index(tmp_path / "bin_faiss", DIM, factory_config)
        assert isinstance(idx, FAISSBinaryIndex)

    def test_binary_store_add_and_search(self, tmp_path, factory_config):
        """Integration: create via factory, add vectors, coarse search."""
        factory_config.binary_backend = "hnswlib"

        from codexlens_search.core.factory import create_binary_index
        idx = create_binary_index(tmp_path / "bin_int", DIM, factory_config)

        rng = np.random.default_rng(42)
        ids = np.arange(10, dtype=np.int64)
        vecs = rng.standard_normal((10, DIM)).astype(np.float32)

        idx.add(ids, vecs)

        result_ids, dists = idx.coarse_search(vecs[0], top_k=5)
        assert len(result_ids) > 0
        assert 0 in result_ids


class TestFactoryWithPipeline:
    """Integration: factory-created backends used in full search pipeline."""

    def test_factory_backends_in_search_pipeline(self, tmp_path):
        """Create pipeline with factory backends, index docs, search."""
        config = Config(
            embed_dim=DIM,
            ann_backend="hnswlib",
            binary_backend="hnswlib",
            hnsw_ef=50,
            hnsw_M=16,
            binary_top_k=50,
            ann_top_k=20,
            reranker_top_k=10,
        )

        from codexlens_search.core.factory import create_ann_index, create_binary_index
        from codexlens_search.search.fts import FTSEngine
        from codexlens_search.search.pipeline import SearchPipeline

        from tests.integration.conftest import MockEmbedder, MockReranker, TEST_DOCS

        embedder = MockEmbedder()
        reranker = MockReranker()
        binary_store = create_binary_index(tmp_path / "bin", DIM, config)
        ann_index = create_ann_index(tmp_path / "ann", DIM, config)
        fts = FTSEngine(tmp_path / "fts.db")

        # Index test docs
        ids = np.array([d[0] for d in TEST_DOCS], dtype=np.int64)
        vectors = np.array([embedder.embed_single(d[2]) for d in TEST_DOCS], dtype=np.float32)

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
