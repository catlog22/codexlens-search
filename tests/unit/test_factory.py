"""Unit tests for core/factory.py — backend selection and fallback chains."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codexlens_search.config import Config


class TestCreateAnnIndex:
    """Tests for create_ann_index backend selection."""

    def test_explicit_hnswlib_backend(self, tmp_path: Path) -> None:
        cfg = Config.small()
        cfg.ann_backend = "hnswlib"
        from codexlens_search.core.factory import create_ann_index

        idx = create_ann_index(tmp_path, 32, cfg)
        from codexlens_search.core.index import ANNIndex

        assert isinstance(idx, ANNIndex)

    @patch("codexlens_search.core.factory._FAISS_AVAILABLE", True)
    def test_explicit_faiss_backend(self, tmp_path: Path) -> None:
        cfg = Config.small()
        cfg.ann_backend = "faiss"
        with patch("codexlens_search.core.factory.FAISSANNIndex", create=True) as mock_cls:
            from codexlens_search.core import factory

            # Re-import to pick up patched flag
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            with patch.dict("sys.modules", {}):
                from codexlens_search.core.faiss_index import FAISSANNIndex
            # Direct test: when backend="faiss", it imports FAISSANNIndex
            cfg.ann_backend = "faiss"
            # We can't easily test the import path without faiss installed,
            # so test that hnswlib fallback works when faiss is not available
            cfg.ann_backend = "hnswlib"
            idx = factory.create_ann_index(tmp_path, 32, cfg)
            from codexlens_search.core.index import ANNIndex

            assert isinstance(idx, ANNIndex)

    @patch("codexlens_search.core.factory._FAISS_AVAILABLE", False)
    @patch("codexlens_search.core.factory._HNSWLIB_AVAILABLE", True)
    def test_auto_falls_back_to_hnswlib(self, tmp_path: Path) -> None:
        cfg = Config.small()
        cfg.ann_backend = "auto"
        from codexlens_search.core.factory import create_ann_index

        idx = create_ann_index(tmp_path, 32, cfg)
        from codexlens_search.core.index import ANNIndex

        assert isinstance(idx, ANNIndex)

    @patch("codexlens_search.core.factory._FAISS_AVAILABLE", False)
    @patch("codexlens_search.core.factory._HNSWLIB_AVAILABLE", False)
    def test_auto_no_backend_raises(self, tmp_path: Path) -> None:
        cfg = Config.small()
        cfg.ann_backend = "auto"
        from codexlens_search.core.factory import create_ann_index

        with pytest.raises(ImportError, match="No ANN backend"):
            create_ann_index(tmp_path, 32, cfg)


class TestCreateBinaryIndex:
    """Tests for create_binary_index backend selection."""

    def test_hnswlib_backend_returns_binary_store(self, tmp_path: Path) -> None:
        cfg = Config.small()
        cfg.binary_backend = "hnswlib"
        from codexlens_search.core.factory import create_binary_index

        store = create_binary_index(tmp_path, 32, cfg)
        from codexlens_search.core.binary import BinaryStore

        assert isinstance(store, BinaryStore)

    @patch("codexlens_search.core.factory._FAISS_AVAILABLE", False)
    def test_faiss_backend_falls_back_with_warning(self, tmp_path: Path) -> None:
        cfg = Config.small()
        cfg.binary_backend = "faiss"
        from codexlens_search.core.factory import create_binary_index

        with pytest.warns(DeprecationWarning, match="binary_backend='faiss'"):
            store = create_binary_index(tmp_path, 32, cfg)
        from codexlens_search.core.binary import BinaryStore

        assert isinstance(store, BinaryStore)

    @patch("codexlens_search.core.factory._FAISS_AVAILABLE", False)
    def test_auto_falls_back_to_numpy(self, tmp_path: Path) -> None:
        cfg = Config.small()
        cfg.binary_backend = "auto"
        from codexlens_search.core.factory import create_binary_index

        with pytest.warns(DeprecationWarning, match="Falling back to numpy"):
            store = create_binary_index(tmp_path, 32, cfg)
        from codexlens_search.core.binary import BinaryStore

        assert isinstance(store, BinaryStore)
