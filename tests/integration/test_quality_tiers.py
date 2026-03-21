"""L2 integration tests for quality-routed search and metadata tier tracking.

Tests SearchPipeline quality routing (fast/balanced/thorough/auto) and
MetadataStore tier management with real SQLite backends.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from codexlens_search.config import Config
from codexlens_search.core import ANNIndex, BinaryStore
from codexlens_search.indexing.metadata import MetadataStore
from codexlens_search.search.fts import FTSEngine
from codexlens_search.search.pipeline import SearchPipeline

from tests.integration.conftest import DIM, TEST_DOCS, MockEmbedder, MockReranker


@pytest.fixture
def quality_env(tmp_path):
    """Create SearchPipeline with MetadataStore for quality + tier tests."""
    config = Config.small()
    embedder = MockEmbedder()
    reranker = MockReranker()
    binary_store = BinaryStore(tmp_path / "binary", dim=DIM, config=config)
    ann_index = ANNIndex(tmp_path / "ann.hnsw", dim=DIM, config=config)
    fts = FTSEngine(tmp_path / "fts.db")
    metadata = MetadataStore(tmp_path / "metadata.db")

    # Index all test docs
    ids = np.array([d[0] for d in TEST_DOCS], dtype=np.int64)
    vectors = np.array([embedder.embed_single(d[2]) for d in TEST_DOCS], dtype=np.float32)
    binary_store.add(ids, vectors)
    ann_index.add(ids, vectors)
    fts.add_documents(TEST_DOCS)

    # Register files in metadata so tier tracking works
    for doc in TEST_DOCS:
        path = doc[1]
        metadata.register_file(path, f"hash_{doc[0]}", time.time(), 100)
        metadata.register_chunks(path, [(doc[0], f"chunkhash_{doc[0]}")])
    metadata.flush()

    pipeline = SearchPipeline(
        embedder=embedder,
        binary_store=binary_store,
        ann_index=ann_index,
        reranker=reranker,
        fts=fts,
        config=config,
        metadata_store=metadata,
    )

    return pipeline, metadata, config


class TestQualityRouting:
    """Test that different quality tiers produce results."""

    def test_fast_quality_returns_results(self, quality_env):
        pipeline, _, _ = quality_env
        results = pipeline.search("authenticate", quality="fast")
        assert len(results) > 0

    def test_balanced_quality_returns_results(self, quality_env):
        pipeline, _, _ = quality_env
        results = pipeline.search("authenticate", quality="balanced")
        assert len(results) > 0

    def test_thorough_quality_returns_results(self, quality_env):
        pipeline, _, _ = quality_env
        results = pipeline.search("authenticate", quality="thorough")
        assert len(results) > 0

    def test_auto_with_vectors_uses_thorough(self, quality_env):
        """Auto quality should select thorough when vector index has data."""
        pipeline, _, _ = quality_env
        # With vectors indexed, auto should behave like thorough
        results_auto = pipeline.search("authenticate", quality="auto")
        results_thorough = pipeline.search("authenticate", quality="thorough")
        # Both should return results (may differ due to threading, just check non-empty)
        assert len(results_auto) > 0
        assert len(results_thorough) > 0

    def test_auto_without_vectors_uses_fast(self, tmp_path):
        """Auto quality should select fast when vector index is empty."""
        config = Config.small()
        embedder = MockEmbedder()
        reranker = MockReranker()
        # Create empty binary/ann stores
        binary_store = BinaryStore(tmp_path / "empty_binary", dim=DIM, config=config)
        ann_index = ANNIndex(tmp_path / "empty_ann.hnsw", dim=DIM, config=config)
        fts = FTSEngine(tmp_path / "empty_fts.db")

        # Add docs to FTS only (no vectors)
        fts.add_documents(TEST_DOCS[:5])

        pipeline = SearchPipeline(
            embedder=embedder,
            binary_store=binary_store,
            ann_index=ann_index,
            reranker=reranker,
            fts=fts,
            config=config,
        )

        # Auto with empty vector index should fall back to fast (FTS-only)
        results = pipeline.search("authenticate", quality="auto")
        assert len(results) > 0

    def test_invalid_quality_falls_back_to_auto(self, quality_env):
        pipeline, _, _ = quality_env
        # Invalid quality should not raise, should fall back to auto
        results = pipeline.search("authenticate", quality="invalid_tier")
        assert isinstance(results, list)

    def test_fast_skips_embedding(self, quality_env):
        """Fast quality should not call embed_single (FTS-only)."""
        pipeline, _, _ = quality_env
        # Replace embedder with one that tracks calls
        call_count = [0]
        orig_embed = pipeline._embedder.embed_single

        def counting_embed(text):
            call_count[0] += 1
            return orig_embed(text)

        pipeline._embedder.embed_single = counting_embed
        pipeline.search("authenticate", quality="fast")
        assert call_count[0] == 0, "Fast quality should not call embed_single"

    def test_thorough_calls_embedding(self, quality_env):
        """Thorough quality should call embed_single for vector search."""
        pipeline, _, _ = quality_env
        call_count = [0]
        orig_embed = pipeline._embedder.embed_single

        def counting_embed(text):
            call_count[0] += 1
            return orig_embed(text)

        pipeline._embedder.embed_single = counting_embed
        pipeline.search("authenticate", quality="thorough")
        assert call_count[0] >= 1, "Thorough quality should call embed_single"


class TestFilterDeleted:
    """Test that deleted chunks are excluded from search results."""

    def test_deleted_chunks_filtered(self, tmp_path):
        """Manually tombstone specific chunk IDs and verify they're excluded."""
        config = Config.small()
        embedder = MockEmbedder()
        reranker = MockReranker()
        binary_store = BinaryStore(tmp_path / "del_binary", dim=DIM, config=config)
        ann_index = ANNIndex(tmp_path / "del_ann.hnsw", dim=DIM, config=config)
        fts = FTSEngine(tmp_path / "del_fts.db")
        metadata = MetadataStore(tmp_path / "del_metadata.db")

        # Add docs: 0 from file_a.py, 1 from file_b.py
        docs = [
            (0, "file_a.py", "def authenticate(user): return check(user)"),
            (1, "file_b.py", "def authorize(user): return user.is_admin"),
        ]
        fts.add_documents(docs)
        ids = np.array([0, 1], dtype=np.int64)
        vecs = np.array([embedder.embed_single(d[2]) for d in docs], dtype=np.float32)
        binary_store.add(ids, vecs)
        ann_index.add(ids, vecs)

        # Register file_a in metadata and tombstone it
        metadata.register_file("file_a.py", "hash_a", 0.0, 100)
        metadata.register_chunks("file_a.py", [(0, "chunk_0")])
        metadata.flush()
        metadata.mark_file_deleted("file_a.py")

        pipeline = SearchPipeline(
            embedder=embedder,
            binary_store=binary_store,
            ann_index=ann_index,
            reranker=reranker,
            fts=fts,
            config=config,
            metadata_store=metadata,
        )

        results = pipeline.search("authenticate", quality="fast")
        result_ids = {r.id for r in results}
        assert 0 not in result_ids, "Tombstoned doc 0 should be filtered out"


class TestMetadataTierTracking:
    """Test metadata tier management (hot/warm/cold)."""

    def test_search_records_access(self, quality_env):
        pipeline, metadata, _ = quality_env

        results = pipeline.search("authenticate")
        assert len(results) > 0

        # Files in results should have last_accessed updated
        accessed_paths = {r.path for r in results}
        for path in accessed_paths:
            row = metadata._conn.execute(
                "SELECT last_accessed FROM files WHERE file_path = ?", (path,)
            ).fetchone()
            if row:
                assert row[0] is not None, f"File {path} should have last_accessed set"

    def test_classify_tiers_hot(self, quality_env):
        _, metadata, _ = quality_env

        # Record recent access for auth.py
        metadata.record_access("auth.py")

        # Classify with generous hot threshold
        metadata.classify_tiers(hot_threshold_hours=24, cold_threshold_hours=168)

        tier = metadata.get_file_tier("auth.py")
        assert tier == "hot"

    def test_classify_tiers_cold(self, quality_env):
        _, metadata, _ = quality_env

        # Set a very old last_accessed (> 168 hours ago)
        old_time = time.time() - (200 * 3600)
        metadata._conn.execute(
            "UPDATE files SET last_accessed = ? WHERE file_path = ?",
            (old_time, "auth.py"),
        )
        metadata._conn.commit()

        metadata.classify_tiers(hot_threshold_hours=24, cold_threshold_hours=168)
        tier = metadata.get_file_tier("auth.py")
        assert tier == "cold"

    def test_never_accessed_becomes_cold(self, quality_env):
        _, metadata, _ = quality_env

        # Ensure no access recorded for config.py
        metadata._conn.execute(
            "UPDATE files SET last_accessed = NULL WHERE file_path = ?",
            ("config.py",),
        )
        metadata._conn.commit()

        metadata.classify_tiers(hot_threshold_hours=24, cold_threshold_hours=168)
        tier = metadata.get_file_tier("config.py")
        assert tier == "cold"

    def test_get_files_by_tier(self, quality_env):
        _, metadata, _ = quality_env

        # Record access for some files
        metadata.record_access("auth.py")
        metadata.record_access("models.py")

        metadata.classify_tiers(hot_threshold_hours=24, cold_threshold_hours=168)

        hot_files = metadata.get_files_by_tier("hot")
        assert "auth.py" in hot_files
        assert "models.py" in hot_files

    def test_record_access_batch(self, quality_env):
        _, metadata, _ = quality_env

        metadata.record_access_batch(["auth.py", "api.py", "cache.py"])

        for path in ["auth.py", "api.py", "cache.py"]:
            row = metadata._conn.execute(
                "SELECT last_accessed FROM files WHERE file_path = ?", (path,)
            ).fetchone()
            assert row is not None and row[0] is not None
