"""Unit tests for core/shard_manager.py — ShardManager routing, LRU, parallel search."""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codexlens_search.config import Config
from codexlens_search.core.shard_manager import ShardManager
from codexlens_search.indexing.pipeline import IndexStats
from codexlens_search.search.pipeline import SearchResult


def _mock_embedder():
    return MagicMock()


def _mock_reranker():
    return MagicMock()


def _make_mock_shard(shard_id: int) -> MagicMock:
    """Create a mock Shard with controllable is_loaded state."""
    m = MagicMock()
    m.shard_id = shard_id
    m._is_loaded = False

    def _get_loaded():
        return m._is_loaded

    type(m).is_loaded = property(lambda self: self._is_loaded)

    def load(*a, **k):
        m._is_loaded = True
    m.load.side_effect = load

    def unload():
        m._is_loaded = False
    m.unload.side_effect = unload

    return m


def _make_manager(
    num_shards: int, tmp_path: Path, max_loaded: int = 4
) -> ShardManager:
    """Create a ShardManager with mock shards (bypassing real Shard I/O)."""
    cfg = Config.small()
    cfg.max_loaded_shards = max_loaded

    mgr = ShardManager.__new__(ShardManager)
    mgr._num_shards = num_shards
    mgr._db_path = tmp_path
    mgr._config = cfg
    mgr._embedder = _mock_embedder()
    mgr._reranker = _mock_reranker()
    mgr._max_loaded = max_loaded
    mgr._shards = {i: _make_mock_shard(i) for i in range(num_shards)}
    mgr._loaded_order = OrderedDict()
    mgr._lru_lock = threading.Lock()
    return mgr


class TestInit:
    """Test ShardManager initialization."""

    def test_creates_correct_number_of_shards(self, tmp_path: Path) -> None:
        cfg = Config.small()
        cfg.max_loaded_shards = 4
        mgr = ShardManager(3, tmp_path, cfg, _mock_embedder(), _mock_reranker())
        assert mgr.num_shards == 3
        assert len(mgr._shards) == 3

    def test_zero_shards_raises(self, tmp_path: Path) -> None:
        cfg = Config.small()
        with pytest.raises(ValueError, match="num_shards must be >= 1"):
            ShardManager(0, tmp_path, cfg, _mock_embedder(), _mock_reranker())

    def test_negative_shards_raises(self, tmp_path: Path) -> None:
        cfg = Config.small()
        with pytest.raises(ValueError):
            ShardManager(-1, tmp_path, cfg, _mock_embedder(), _mock_reranker())


class TestRouteFile:
    """Test deterministic file routing."""

    def test_same_path_same_shard(self, tmp_path: Path) -> None:
        mgr = _make_manager(4, tmp_path)
        s1 = mgr.route_file("src/main.py")
        s2 = mgr.route_file("src/main.py")
        assert s1 == s2

    def test_route_within_range(self, tmp_path: Path) -> None:
        mgr = _make_manager(4, tmp_path)
        for path in ["a.py", "b.py", "c.py", "d/e.py", "f/g/h.py"]:
            shard_id = mgr.route_file(path)
            assert 0 <= shard_id < 4

    def test_distribution_not_all_same(self, tmp_path: Path) -> None:
        mgr = _make_manager(4, tmp_path)
        shard_ids = {mgr.route_file(f"file_{i}.py") for i in range(100)}
        assert len(shard_ids) > 1


class TestGetShard:
    """Test get_shard validation."""

    def test_valid_shard_id(self, tmp_path: Path) -> None:
        mgr = _make_manager(3, tmp_path)
        shard = mgr.get_shard(0)
        assert shard.shard_id == 0

    def test_invalid_shard_id_raises(self, tmp_path: Path) -> None:
        mgr = _make_manager(3, tmp_path)
        with pytest.raises(ValueError, match="Invalid shard_id"):
            mgr.get_shard(5)


class TestLRUEviction:
    """Test LRU eviction policy."""

    def test_evicts_lru_when_over_limit(self, tmp_path: Path) -> None:
        mgr = _make_manager(3, tmp_path, max_loaded=2)

        mgr._ensure_loaded(0)
        mgr._ensure_loaded(1)
        assert len(mgr._loaded_order) == 2

        # Load shard 2 -> should evict shard 0 (LRU)
        mgr._ensure_loaded(2)
        assert 0 not in mgr._loaded_order
        mgr._shards[0].unload.assert_called()

    def test_access_refreshes_lru_position(self, tmp_path: Path) -> None:
        mgr = _make_manager(3, tmp_path, max_loaded=2)

        mgr._ensure_loaded(0)
        mgr._ensure_loaded(1)
        # Access 0 again to refresh it
        mgr._ensure_loaded(0)
        # Load 2 -> should evict 1 (now LRU), not 0
        mgr._ensure_loaded(2)
        assert 0 in mgr._loaded_order
        assert 1 not in mgr._loaded_order


class TestSync:
    """Test ShardManager.sync file routing."""

    def test_sync_routes_files_to_shards(self, tmp_path: Path) -> None:
        mgr = _make_manager(2, tmp_path)

        for shard in mgr._shards.values():
            shard.sync.return_value = IndexStats(
                files_processed=1, chunks_created=3, duration_seconds=0.1
            )

        files = [tmp_path / "a.py", tmp_path / "b.py"]
        for f in files:
            f.touch()

        stats = mgr.sync(files, root=tmp_path)
        assert stats.files_processed >= 0
        assert stats.duration_seconds >= 0

    def test_sync_empty_files(self, tmp_path: Path) -> None:
        mgr = _make_manager(2, tmp_path)
        stats = mgr.sync([], root=tmp_path)
        assert stats.files_processed == 0


class TestSearch:
    """Test ShardManager.search parallel search and RRF merge."""

    def test_search_returns_merged_results(self, tmp_path: Path) -> None:
        mgr = _make_manager(2, tmp_path)

        for i, shard in mgr._shards.items():
            shard.search.return_value = [
                SearchResult(id=i * 10 + 1, path=f"s{i}_a.py", score=0.9),
                SearchResult(id=i * 10 + 2, path=f"s{i}_b.py", score=0.8),
            ]

        results = mgr.search("test query")
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_single_shard_no_rrf(self, tmp_path: Path) -> None:
        mgr = _make_manager(2, tmp_path)

        mgr._shards[0].search.return_value = [
            SearchResult(id=1, path="a.py", score=0.9),
        ]
        mgr._shards[1].search.return_value = []

        results = mgr.search("test query")
        assert len(results) == 1
        assert results[0].path == "a.py"

    def test_search_empty_results(self, tmp_path: Path) -> None:
        mgr = _make_manager(2, tmp_path)

        for shard in mgr._shards.values():
            shard.search.return_value = []

        results = mgr.search("nothing")
        assert results == []

    def test_search_respects_top_k(self, tmp_path: Path) -> None:
        mgr = _make_manager(2, tmp_path)

        for i, shard in mgr._shards.items():
            shard.search.return_value = [
                SearchResult(id=j, path=f"s{i}_{j}.py", score=1.0 - j * 0.1)
                for j in range(10)
            ]

        results = mgr.search("test", top_k=3)
        assert len(results) <= 3
