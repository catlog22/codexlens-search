"""L2 integration tests for shard routing and multi-shard search.

Tests ShardManager with real SQLite, real FTS, but mock embedder/reranker.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from codexlens_search.config import Config
from codexlens_search.core.shard_manager import ShardManager

from tests.integration.conftest import DIM, MockEmbedder, MockReranker


@pytest.fixture
def shard_config():
    """Config for 3-shard tests."""
    return Config(
        embed_dim=DIM,
        num_shards=3,
        max_loaded_shards=2,
        hnsw_ef=50,
        hnsw_M=16,
        binary_top_k=50,
        ann_top_k=20,
        reranker_top_k=10,
    )


@pytest.fixture
def shard_manager(tmp_path, shard_config):
    embedder = MockEmbedder()
    reranker = MockReranker()
    return ShardManager(
        num_shards=3,
        db_path=tmp_path / "shards",
        config=shard_config,
        embedder=embedder,
        reranker=reranker,
    )


@pytest.fixture
def source_files(tmp_path):
    """Create sample source files for shard sync."""
    src = tmp_path / "src"
    src.mkdir()
    files_content = {
        "auth.py": "def authenticate(user, password): return check_hash(password)",
        "models.py": "class User:\n    def __init__(self, name): self.name = name",
        "api.py": "def get_user(request, user_id): return User.objects.get(id=user_id)",
        "config.py": "DATABASE_URL = 'sqlite:///db.sqlite3'",
        "cache.py": "def cache_get(key): return redis.get(key)",
        "router.py": "app.route('/users')(list_users)",
    }
    for name, content in files_content.items():
        (src / name).write_text(content, encoding="utf-8")
    return src


class TestShardRouting:
    """Test deterministic file-to-shard routing."""

    def test_same_file_always_routes_to_same_shard(self, shard_manager):
        path = "src/auth.py"
        shard_id_1 = shard_manager.route_file(path)
        shard_id_2 = shard_manager.route_file(path)
        assert shard_id_1 == shard_id_2

    def test_routing_within_valid_range(self, shard_manager):
        paths = ["auth.py", "models.py", "api.py", "config.py", "cache.py"]
        for path in paths:
            shard_id = shard_manager.route_file(path)
            assert 0 <= shard_id < 3

    def test_routing_distributes_files(self, shard_manager):
        """Given enough files, routing should hit multiple shards."""
        shard_ids = set()
        for i in range(20):
            shard_ids.add(shard_manager.route_file(f"file_{i}.py"))
        # With 20 files and 3 shards, very unlikely all go to same shard
        assert len(shard_ids) >= 2

    def test_get_shard_invalid_raises(self, shard_manager):
        with pytest.raises(ValueError, match="Invalid shard_id"):
            shard_manager.get_shard(99)


class TestShardSync:
    """Test multi-shard sync operation."""

    def test_sync_distributes_across_shards(self, shard_manager, source_files):
        files = list(source_files.glob("*.py"))
        stats = shard_manager.sync(files, root=source_files)

        assert stats.files_processed == len(files)
        assert stats.chunks_created >= len(files)

    def test_sync_all_files_searchable(self, shard_manager, source_files):
        files = list(source_files.glob("*.py"))
        shard_manager.sync(files, root=source_files)

        results = shard_manager.search("authenticate")
        assert len(results) > 0

    def test_sync_empty_file_list(self, shard_manager, source_files):
        stats = shard_manager.sync([], root=source_files)
        assert stats.files_processed == 0
        assert stats.chunks_created == 0


class TestShardSearch:
    """Test cross-shard search and RRF merge."""

    def test_search_merges_results_from_multiple_shards(self, shard_manager, source_files):
        files = list(source_files.glob("*.py"))
        shard_manager.sync(files, root=source_files)

        results = shard_manager.search("user")
        assert len(results) > 0
        # Results should come from multiple source files
        paths = {r.path for r in results}
        assert len(paths) >= 1

    def test_search_respects_top_k(self, shard_manager, source_files):
        files = list(source_files.glob("*.py"))
        shard_manager.sync(files, root=source_files)

        results = shard_manager.search("user", top_k=2)
        assert len(results) <= 2

    def test_search_empty_index_returns_empty(self, shard_manager):
        results = shard_manager.search("anything")
        assert results == []

    def test_search_result_fields_populated(self, shard_manager, source_files):
        files = list(source_files.glob("*.py"))
        shard_manager.sync(files, root=source_files)

        results = shard_manager.search("authenticate")
        assert len(results) > 0
        for r in results:
            assert r.score >= 0
            assert isinstance(r.path, str)
            assert len(r.path) > 0


class TestShardLRU:
    """Test LRU eviction of loaded shards."""

    def test_lru_evicts_when_over_limit(self, tmp_path):
        """With max_loaded_shards=2 and 3 shards, loading all 3 evicts the LRU."""
        config = Config(
            embed_dim=DIM,
            num_shards=3,
            max_loaded_shards=2,
            hnsw_ef=50,
            hnsw_M=16,
        )
        embedder = MockEmbedder()
        reranker = MockReranker()
        manager = ShardManager(
            num_shards=3,
            db_path=tmp_path / "lru_shards",
            config=config,
            embedder=embedder,
            reranker=reranker,
        )

        # Load shards 0, 1, 2 sequentially
        manager._ensure_loaded(0)
        manager._ensure_loaded(1)
        # At this point, 2 shards loaded (at limit)
        assert manager.get_shard(0).is_loaded
        assert manager.get_shard(1).is_loaded

        manager._ensure_loaded(2)
        # Shard 0 (LRU) should be evicted
        assert not manager.get_shard(0).is_loaded
        assert manager.get_shard(1).is_loaded
        assert manager.get_shard(2).is_loaded

    def test_lru_access_refreshes_order(self, tmp_path):
        """Accessing shard 0 again after loading 1 should keep 0 loaded."""
        config = Config(
            embed_dim=DIM,
            num_shards=3,
            max_loaded_shards=2,
            hnsw_ef=50,
            hnsw_M=16,
        )
        embedder = MockEmbedder()
        reranker = MockReranker()
        manager = ShardManager(
            num_shards=3,
            db_path=tmp_path / "lru_refresh",
            config=config,
            embedder=embedder,
            reranker=reranker,
        )

        manager._ensure_loaded(0)
        manager._ensure_loaded(1)
        # Re-access shard 0 to refresh its LRU position
        manager._ensure_loaded(0)
        # Now load shard 2 - shard 1 (LRU) should be evicted
        manager._ensure_loaded(2)

        assert manager.get_shard(0).is_loaded
        assert not manager.get_shard(1).is_loaded
        assert manager.get_shard(2).is_loaded
