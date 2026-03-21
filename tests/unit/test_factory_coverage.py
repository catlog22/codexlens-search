"""Additional factory.py coverage tests — FAISS auto-detection, _has_faiss_gpu."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codexlens_search.config import Config


# ---------------------------------------------------------------------------
# _has_faiss_gpu (covers lines 27-34)
# ---------------------------------------------------------------------------

class TestHasFaissGpu:
    @patch("codexlens_search.core.factory._FAISS_AVAILABLE", False)
    def test_no_faiss_returns_false(self):
        from codexlens_search.core.factory import _has_faiss_gpu
        assert _has_faiss_gpu() is False

    @patch("codexlens_search.core.factory._FAISS_AVAILABLE", True)
    def test_faiss_no_gpu_returns_false(self):
        from codexlens_search.core.factory import _has_faiss_gpu
        with patch.dict("sys.modules", {"faiss": MagicMock(StandardGpuResources=MagicMock(side_effect=AttributeError))}):
            assert _has_faiss_gpu() is False

    @patch("codexlens_search.core.factory._FAISS_AVAILABLE", True)
    def test_faiss_with_gpu_returns_true(self):
        from codexlens_search.core.factory import _has_faiss_gpu
        mock_faiss = MagicMock()
        mock_faiss.StandardGpuResources.return_value = MagicMock()
        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            assert _has_faiss_gpu() is True


# ---------------------------------------------------------------------------
# create_ann_index with FAISS auto (covers lines 64-68)
# ---------------------------------------------------------------------------

class TestCreateAnnIndexFaissAuto:
    @patch("codexlens_search.core.factory._USEARCH_AVAILABLE", False)
    @patch("codexlens_search.core.factory._FAISS_AVAILABLE", True)
    @patch("codexlens_search.core.factory._has_faiss_gpu", return_value=False)
    def test_auto_selects_faiss_cpu(self, mock_gpu, tmp_path):
        cfg = Config.small()
        cfg.ann_backend = "auto"
        mock_faiss_idx = MagicMock()
        with patch("codexlens_search.core.faiss_index.FAISSANNIndex", return_value=mock_faiss_idx):
            from codexlens_search.core.factory import create_ann_index
            result = create_ann_index(tmp_path, 32, cfg)
            assert result is mock_faiss_idx

    @patch("codexlens_search.core.factory._USEARCH_AVAILABLE", False)
    @patch("codexlens_search.core.factory._FAISS_AVAILABLE", True)
    @patch("codexlens_search.core.factory._has_faiss_gpu", return_value=True)
    def test_auto_selects_faiss_gpu(self, mock_gpu, tmp_path):
        cfg = Config.small()
        cfg.ann_backend = "auto"
        mock_faiss_idx = MagicMock()
        with patch("codexlens_search.core.faiss_index.FAISSANNIndex", return_value=mock_faiss_idx):
            from codexlens_search.core.factory import create_ann_index
            result = create_ann_index(tmp_path, 32, cfg)
            assert result is mock_faiss_idx


# ---------------------------------------------------------------------------
# create_binary_index with FAISS auto (covers lines 124-127)
# ---------------------------------------------------------------------------

class TestCreateBinaryIndexFaissAuto:
    @patch("codexlens_search.core.factory._FAISS_AVAILABLE", True)
    def test_auto_selects_faiss_binary(self, tmp_path):
        cfg = Config.small()
        cfg.binary_backend = "auto"
        mock_faiss_bin = MagicMock()
        with patch("codexlens_search.core.faiss_index.FAISSBinaryIndex", return_value=mock_faiss_bin):
            from codexlens_search.core.factory import create_binary_index
            result = create_binary_index(tmp_path, 32, cfg)
            assert result is mock_faiss_bin


# ---------------------------------------------------------------------------
# create_ann_index explicit faiss (covers lines 56-57)
# ---------------------------------------------------------------------------

class TestCreateAnnIndexExplicitFaiss:
    @patch("codexlens_search.core.factory._FAISS_AVAILABLE", True)
    def test_explicit_faiss_backend(self, tmp_path):
        cfg = Config.small()
        cfg.ann_backend = "faiss"
        mock_faiss_idx = MagicMock()
        with patch("codexlens_search.core.faiss_index.FAISSANNIndex", return_value=mock_faiss_idx):
            from codexlens_search.core.factory import create_ann_index
            result = create_ann_index(tmp_path, 32, cfg)
            assert result is mock_faiss_idx


# ---------------------------------------------------------------------------
# create_binary_index explicit faiss available (covers lines 102-103)
# ---------------------------------------------------------------------------

class TestCreateBinaryIndexExplicitFaissAvailable:
    @patch("codexlens_search.core.factory._FAISS_AVAILABLE", True)
    def test_explicit_faiss_with_faiss_installed(self, tmp_path):
        cfg = Config.small()
        cfg.binary_backend = "faiss"
        mock_faiss_bin = MagicMock()
        with patch("codexlens_search.core.faiss_index.FAISSBinaryIndex", return_value=mock_faiss_bin):
            from codexlens_search.core.factory import create_binary_index
            result = create_binary_index(tmp_path, 32, cfg)
            assert result is mock_faiss_bin
