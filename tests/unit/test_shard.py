"""Unit tests for core/shard.py — Shard lifecycle."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codexlens_search.config import Config
from codexlens_search.core.shard import Shard


def _mock_embedder():
    return MagicMock()


def _mock_reranker():
    return MagicMock()


# Patch targets are in core.factory since Shard._ensure_loaded does:
#   from codexlens_search.core.factory import create_ann_index, create_binary_index
_PATCH_ANN = "codexlens_search.core.factory.create_ann_index"
_PATCH_BIN = "codexlens_search.core.factory.create_binary_index"


class TestShardLifecycle:
    """Test Shard lazy loading and unloading."""

    def test_new_shard_is_not_loaded(self, tmp_path: Path) -> None:
        cfg = Config.small()
        shard = Shard(0, tmp_path, cfg)
        assert not shard.is_loaded
        assert shard.shard_id == 0

    @patch(_PATCH_ANN)
    @patch(_PATCH_BIN)
    def test_load_creates_components(self, mock_binary, mock_ann, tmp_path: Path) -> None:
        cfg = Config.small()
        mock_binary.return_value = MagicMock()
        mock_ann.return_value = MagicMock()
        shard = Shard(0, tmp_path, cfg)
        shard.load(_mock_embedder(), _mock_reranker())
        assert shard.is_loaded
        assert shard._fts is not None
        assert shard._search is not None
        assert shard._indexing is not None

    @patch(_PATCH_ANN)
    @patch(_PATCH_BIN)
    def test_load_is_idempotent(self, mock_binary, mock_ann, tmp_path: Path) -> None:
        cfg = Config.small()
        mock_binary.return_value = MagicMock()
        mock_ann.return_value = MagicMock()
        shard = Shard(0, tmp_path, cfg)
        embedder, reranker = _mock_embedder(), _mock_reranker()
        shard.load(embedder, reranker)
        shard.load(embedder, reranker)  # Second call should be no-op
        assert mock_binary.call_count == 1

    @patch(_PATCH_ANN)
    @patch(_PATCH_BIN)
    def test_unload_releases_components(self, mock_binary, mock_ann, tmp_path: Path) -> None:
        cfg = Config.small()
        mock_binary.return_value = MagicMock()
        mock_ann.return_value = MagicMock()
        shard = Shard(0, tmp_path, cfg)
        shard.load(_mock_embedder(), _mock_reranker())
        assert shard.is_loaded
        shard.unload()
        assert not shard.is_loaded
        assert shard._fts is None
        assert shard._search is None

    def test_unload_when_not_loaded_is_noop(self, tmp_path: Path) -> None:
        cfg = Config.small()
        shard = Shard(0, tmp_path, cfg)
        shard.unload()  # Should not raise
        assert not shard.is_loaded

    @patch(_PATCH_ANN)
    @patch(_PATCH_BIN)
    def test_save_persists_stores(self, mock_binary, mock_ann, tmp_path: Path) -> None:
        cfg = Config.small()
        binary_mock = MagicMock()
        ann_mock = MagicMock()
        mock_binary.return_value = binary_mock
        mock_ann.return_value = ann_mock
        shard = Shard(0, tmp_path, cfg)
        shard.load(_mock_embedder(), _mock_reranker())
        shard.save()
        binary_mock.save.assert_called_once()
        ann_mock.save.assert_called_once()

    def test_save_when_not_loaded_is_noop(self, tmp_path: Path) -> None:
        cfg = Config.small()
        shard = Shard(0, tmp_path, cfg)
        shard.save()  # Should not raise


class TestShardSearch:
    """Test Shard.search delegation."""

    @patch(_PATCH_ANN)
    @patch(_PATCH_BIN)
    def test_search_delegates_to_pipeline(self, mock_binary, mock_ann, tmp_path: Path) -> None:
        cfg = Config.small()
        mock_binary.return_value = MagicMock()
        mock_ann.return_value = MagicMock()
        shard = Shard(0, tmp_path, cfg)
        embedder, reranker = _mock_embedder(), _mock_reranker()
        shard.load(embedder, reranker)
        # Mock the search pipeline
        shard._search = MagicMock()
        shard._search.search.return_value = []
        results = shard.search("test", embedder, reranker, quality="fast")
        shard._search.search.assert_called_once_with("test", top_k=None, quality="fast")

    @patch(_PATCH_ANN)
    @patch(_PATCH_BIN)
    def test_search_lazy_loads(self, mock_binary, mock_ann, tmp_path: Path) -> None:
        cfg = Config.small()
        mock_binary.return_value = MagicMock()
        mock_ann.return_value = MagicMock()
        shard = Shard(0, tmp_path, cfg)
        embedder, reranker = _mock_embedder(), _mock_reranker()
        assert not shard.is_loaded
        # search should trigger lazy load
        shard.search("test", embedder, reranker)
        assert shard.is_loaded


class TestShardSync:
    """Test Shard.sync delegation."""

    @patch(_PATCH_ANN)
    @patch(_PATCH_BIN)
    def test_sync_delegates_to_indexing(self, mock_binary, mock_ann, tmp_path: Path) -> None:
        cfg = Config.small()
        mock_binary.return_value = MagicMock()
        mock_ann.return_value = MagicMock()
        shard = Shard(0, tmp_path, cfg)
        embedder, reranker = _mock_embedder(), _mock_reranker()
        shard.load(embedder, reranker)
        shard._indexing = MagicMock()
        from codexlens_search.indexing.pipeline import IndexStats
        shard._indexing.sync.return_value = IndexStats(files_processed=1, chunks_created=5)
        files = [tmp_path / "test.py"]
        stats = shard.sync(files, root=tmp_path, embedder=embedder, reranker=reranker)
        shard._indexing.sync.assert_called_once()
        assert stats.files_processed == 1
