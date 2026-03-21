"""Unit tests for core/faiss_index.py — FAISSANNIndex and FAISSBinaryIndex with mocked faiss."""
from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from codexlens_search.config import Config


DIM = 32
RNG = np.random.default_rng(99)


def make_vectors(n: int, dim: int = DIM) -> np.ndarray:
    return RNG.standard_normal((n, dim)).astype(np.float32)


def make_ids(n: int, start: int = 0) -> np.ndarray:
    return np.arange(start, start + n, dtype=np.int64)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

class TestModuleHelpers:
    def test_try_gpu_index_falls_back_on_cpu(self):
        mock_faiss = MagicMock()
        mock_faiss.StandardGpuResources.side_effect = AttributeError("no GPU")
        mock_index = MagicMock()

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from codexlens_search.core.faiss_index import _try_gpu_index
            result = _try_gpu_index(mock_index)
            # Should return original index on failure
            assert result is mock_index

    def test_to_cpu_for_save_passthrough(self):
        mock_faiss = MagicMock()
        mock_faiss.index_gpu_to_cpu.side_effect = AttributeError("not gpu")
        mock_index = MagicMock()

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from codexlens_search.core.faiss_index import _to_cpu_for_save
            result = _to_cpu_for_save(mock_index)
            assert result is mock_index


# ---------------------------------------------------------------------------
# FAISSANNIndex
# ---------------------------------------------------------------------------

class TestFAISSANNIndex:
    @pytest.fixture
    def mock_faiss(self):
        """Create a mock faiss module."""
        mock = MagicMock()
        mock.IO_FLAG_MMAP = 1
        mock.METRIC_INNER_PRODUCT = 0
        mock_index = MagicMock()
        mock_index.ntotal = 0
        mock_index.hnsw = MagicMock()
        mock.IndexHNSWFlat.return_value = mock_index
        mock.normalize_L2 = MagicMock()  # In-place, returns None
        return mock

    @pytest.fixture
    def ann_index(self, tmp_path, mock_faiss):
        """Create a FAISSANNIndex with mocked faiss."""
        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            with patch("codexlens_search.core.faiss_index._FAISS_AVAILABLE", True):
                with patch("codexlens_search.core.faiss_index.faiss", mock_faiss):
                    from codexlens_search.core.faiss_index import FAISSANNIndex
                    cfg = Config.small()
                    idx = FAISSANNIndex(tmp_path, DIM, cfg)
                    yield idx, mock_faiss

    def test_init(self, ann_index):
        idx, mock_faiss = ann_index
        assert idx._dim == DIM
        assert idx._index is None

    def test_load_creates_fresh_index(self, ann_index):
        idx, mock_faiss = ann_index
        idx.load()
        mock_faiss.IndexHNSWFlat.assert_called_once()
        assert idx._index is not None

    def test_load_from_disk(self, ann_index, tmp_path):
        idx, mock_faiss = ann_index
        # Create fake index file
        idx._index_path.parent.mkdir(parents=True, exist_ok=True)
        idx._index_path.touch()

        loaded_idx = MagicMock()
        loaded_idx.ntotal = 100
        mock_faiss.read_index.return_value = loaded_idx

        idx.load()
        mock_faiss.read_index.assert_called()

    def test_load_mmap_fallback(self, ann_index, tmp_path):
        idx, mock_faiss = ann_index
        idx._index_path.parent.mkdir(parents=True, exist_ok=True)
        idx._index_path.touch()

        # First call with MMAP fails, second without MMAP succeeds
        loaded_idx = MagicMock()
        loaded_idx.ntotal = 50
        mock_faiss.read_index.side_effect = [RuntimeError("mmap fail"), loaded_idx]

        idx.load()
        assert mock_faiss.read_index.call_count == 2

    def test_add_empty_ids_no_op(self, ann_index):
        idx, mock_faiss = ann_index
        idx.add(np.array([], dtype=np.int64), np.array([]).reshape(0, DIM).astype(np.float32))
        # Should not call _ensure_loaded
        assert idx._index is None

    def test_add_normalizes_and_adds(self, ann_index):
        idx, mock_faiss = ann_index
        idx.load()
        vecs = make_vectors(5)
        ids = make_ids(5)

        idx.add(ids, vecs)
        mock_faiss.normalize_L2.assert_called()
        idx._index.add.assert_called_once()

    def test_fine_search_empty_index(self, ann_index):
        idx, mock_faiss = ann_index
        idx.load()
        idx._index.ntotal = 0

        result_ids, result_dists = idx.fine_search(make_vectors(1)[0])
        assert len(result_ids) == 0
        assert len(result_dists) == 0

    def test_fine_search_returns_results(self, ann_index):
        idx, mock_faiss = ann_index
        idx.load()
        idx._index.ntotal = 10
        idx._index.search.return_value = (
            np.array([[0, 1, 2]], dtype=np.int64),
            np.array([[0.9, 0.8, 0.7]], dtype=np.float32),
        )

        result_ids, result_dists = idx.fine_search(make_vectors(1)[0], top_k=3)
        assert len(result_ids) == 3
        assert len(result_dists) == 3

    def test_fine_search_clamps_k(self, ann_index):
        idx, mock_faiss = ann_index
        idx.load()
        idx._index.ntotal = 2
        idx._index.search.return_value = (
            np.array([[0, 1]], dtype=np.int64),
            np.array([[0.9, 0.8]], dtype=np.float32),
        )

        # Request more than available
        result_ids, _ = idx.fine_search(make_vectors(1)[0], top_k=100)
        # search should be called with k=2 (clamped to ntotal)
        call_k = idx._index.search.call_args[0][1]
        assert call_k == 2

    def test_save_no_index_no_op(self, ann_index):
        idx, mock_faiss = ann_index
        idx.save()  # Should not raise
        mock_faiss.write_index.assert_not_called()

    def test_save_writes_to_disk(self, ann_index):
        idx, mock_faiss = ann_index
        idx.load()
        mock_faiss.index_gpu_to_cpu.side_effect = AttributeError("not gpu")
        idx.save()
        mock_faiss.write_index.assert_called_once()

    def test_len_no_index(self, ann_index):
        idx, _ = ann_index
        assert len(idx) == 0

    def test_len_with_index(self, ann_index):
        idx, _ = ann_index
        idx.load()
        idx._index.ntotal = 42
        assert len(idx) == 42


# ---------------------------------------------------------------------------
# FAISSBinaryIndex
# ---------------------------------------------------------------------------

class TestFAISSBinaryIndex:
    @pytest.fixture
    def mock_faiss(self):
        mock = MagicMock()
        mock.IO_FLAG_MMAP = 1
        mock_index = MagicMock()
        mock_index.ntotal = 0
        mock.IndexBinaryFlat.return_value = mock_index
        return mock

    @pytest.fixture
    def binary_index(self, tmp_path, mock_faiss):
        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            with patch("codexlens_search.core.faiss_index._FAISS_AVAILABLE", True):
                with patch("codexlens_search.core.faiss_index.faiss", mock_faiss):
                    from codexlens_search.core.faiss_index import FAISSBinaryIndex
                    cfg = Config.small()
                    idx = FAISSBinaryIndex(tmp_path, DIM, cfg)
                    yield idx, mock_faiss

    def test_init(self, binary_index):
        idx, _ = binary_index
        assert idx._dim == DIM
        assert idx._packed_bytes == math.ceil(DIM / 8)
        assert idx._index is None

    def test_load_creates_fresh(self, binary_index):
        idx, mock_faiss = binary_index
        idx.load()
        mock_faiss.IndexBinaryFlat.assert_called_once_with(DIM)

    def test_load_from_disk(self, binary_index, tmp_path):
        idx, mock_faiss = binary_index
        idx._index_path.parent.mkdir(parents=True, exist_ok=True)
        idx._index_path.touch()

        loaded = MagicMock()
        loaded.ntotal = 50
        mock_faiss.read_index_binary.return_value = loaded

        idx.load()
        mock_faiss.read_index_binary.assert_called()

    def test_quantize(self, binary_index):
        idx, _ = binary_index
        vecs = np.array([[1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5] * (DIM // 8)], dtype=np.float32)
        packed = idx._quantize(vecs)
        assert packed.dtype == np.uint8
        assert packed.shape[0] == 1

    def test_quantize_single(self, binary_index):
        idx, _ = binary_index
        vec = make_vectors(1)[0]
        packed = idx._quantize_single(vec)
        assert packed.dtype == np.uint8
        assert packed.shape[0] == 1

    def test_add_empty_no_op(self, binary_index):
        idx, _ = binary_index
        idx.add(np.array([], dtype=np.int64), np.array([]).reshape(0, DIM).astype(np.float32))
        assert idx._index is None

    def test_add_quantizes_and_adds(self, binary_index):
        idx, mock_faiss = binary_index
        idx.load()
        vecs = make_vectors(5)
        ids = make_ids(5)

        idx.add(ids, vecs)
        idx._index.add.assert_called_once()

    def test_coarse_search_empty(self, binary_index):
        idx, mock_faiss = binary_index
        idx.load()
        idx._index.ntotal = 0

        result_ids, result_dists = idx.coarse_search(make_vectors(1)[0])
        assert len(result_ids) == 0
        assert len(result_dists) == 0

    def test_coarse_search_returns_results(self, binary_index):
        idx, mock_faiss = binary_index
        idx.load()
        idx._index.ntotal = 10
        idx._index.search.return_value = (
            np.array([[0, 1, 2]], dtype=np.int64),
            np.array([[5, 8, 12]], dtype=np.int32),
        )

        result_ids, result_dists = idx.coarse_search(make_vectors(1)[0], top_k=3)
        assert len(result_ids) == 3
        assert result_dists.dtype == np.int32

    def test_save_no_index_no_op(self, binary_index):
        idx, mock_faiss = binary_index
        idx.save()
        mock_faiss.write_index_binary.assert_not_called()

    def test_save_writes(self, binary_index):
        idx, mock_faiss = binary_index
        idx.load()
        idx.save()
        mock_faiss.write_index_binary.assert_called_once()

    def test_len_no_index(self, binary_index):
        idx, _ = binary_index
        assert len(idx) == 0

    def test_len_with_index(self, binary_index):
        idx, _ = binary_index
        idx.load()
        idx._index.ntotal = 99
        assert len(idx) == 99


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

class TestImportGuard:
    def test_faiss_not_available_raises(self, tmp_path):
        with patch("codexlens_search.core.faiss_index._FAISS_AVAILABLE", False):
            from codexlens_search.core.faiss_index import FAISSANNIndex, FAISSBinaryIndex
            cfg = Config.small()

            with pytest.raises(ImportError, match="faiss is required"):
                FAISSANNIndex(tmp_path, DIM, cfg)

            with pytest.raises(ImportError, match="faiss is required"):
                FAISSBinaryIndex(tmp_path, DIM, cfg)
