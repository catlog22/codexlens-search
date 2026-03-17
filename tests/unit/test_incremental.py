"""Unit tests for IndexingPipeline incremental API (index_file, remove_file, sync, compact)."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from codexlens_search.config import Config
from codexlens_search.core.binary import BinaryStore
from codexlens_search.core.index import ANNIndex
from codexlens_search.embed.base import BaseEmbedder
from codexlens_search.indexing.metadata import MetadataStore
from codexlens_search.indexing.pipeline import IndexingPipeline, IndexStats
from codexlens_search.search.fts import FTSEngine


DIM = 32


class FakeEmbedder(BaseEmbedder):
    """Deterministic embedder for testing."""

    def __init__(self) -> None:
        pass

    def embed_single(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(hash(text) % (2**31))
        return rng.standard_normal(DIM).astype(np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed_single(t) for t in texts]


@pytest.fixture
def workspace(tmp_path: Path):
    """Create workspace with stores, metadata, and pipeline."""
    cfg = Config.small()
    # Override embed_dim to match our test dim
    cfg.embed_dim = DIM

    store_dir = tmp_path / "stores"
    store_dir.mkdir()

    binary_store = BinaryStore(store_dir, DIM, cfg)
    ann_index = ANNIndex(store_dir, DIM, cfg)
    fts = FTSEngine(str(store_dir / "fts.db"))
    metadata = MetadataStore(str(store_dir / "metadata.db"))
    embedder = FakeEmbedder()

    pipeline = IndexingPipeline(
        embedder=embedder,
        binary_store=binary_store,
        ann_index=ann_index,
        fts=fts,
        config=cfg,
        metadata=metadata,
    )

    # Create sample source files
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    return {
        "pipeline": pipeline,
        "metadata": metadata,
        "binary_store": binary_store,
        "ann_index": ann_index,
        "fts": fts,
        "src_dir": src_dir,
        "store_dir": store_dir,
        "config": cfg,
    }


def _write_file(src_dir: Path, name: str, content: str) -> Path:
    """Write a file and return its path."""
    p = src_dir / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# MetadataStore helper method tests
# ---------------------------------------------------------------------------


class TestMetadataHelpers:
    def test_get_all_files_empty(self, workspace):
        meta = workspace["metadata"]
        assert meta.get_all_files() == {}

    def test_get_all_files_after_register(self, workspace):
        meta = workspace["metadata"]
        meta.register_file("a.py", "hash_a", 1000.0)
        meta.register_file("b.py", "hash_b", 2000.0)
        result = meta.get_all_files()
        assert result == {"a.py": "hash_a", "b.py": "hash_b"}

    def test_max_chunk_id_empty(self, workspace):
        meta = workspace["metadata"]
        assert meta.max_chunk_id() == -1

    def test_max_chunk_id_with_chunks(self, workspace):
        meta = workspace["metadata"]
        meta.register_file("a.py", "hash_a", 1000.0)
        meta.register_chunks("a.py", [(0, "h0"), (1, "h1"), (5, "h5")])
        assert meta.max_chunk_id() == 5

    def test_max_chunk_id_includes_deleted(self, workspace):
        meta = workspace["metadata"]
        meta.register_file("a.py", "hash_a", 1000.0)
        meta.register_chunks("a.py", [(0, "h0"), (3, "h3")])
        meta.mark_file_deleted("a.py")
        # Chunks moved to deleted_chunks, max should still be 3
        assert meta.max_chunk_id() == 3


# ---------------------------------------------------------------------------
# index_file tests
# ---------------------------------------------------------------------------


class TestIndexFile:
    def test_index_file_basic(self, workspace):
        pipeline = workspace["pipeline"]
        meta = workspace["metadata"]
        src_dir = workspace["src_dir"]

        f = _write_file(src_dir, "hello.py", "print('hello world')\n")
        stats = pipeline.index_file(f, root=src_dir)

        assert stats.files_processed == 1
        assert stats.chunks_created >= 1
        assert meta.get_file_hash("hello.py") is not None
        assert len(meta.get_chunk_ids_for_file("hello.py")) >= 1

    def test_index_file_skips_unchanged(self, workspace):
        pipeline = workspace["pipeline"]
        src_dir = workspace["src_dir"]

        f = _write_file(src_dir, "same.py", "x = 1\n")
        stats1 = pipeline.index_file(f, root=src_dir)
        assert stats1.files_processed == 1

        stats2 = pipeline.index_file(f, root=src_dir)
        assert stats2.files_processed == 0
        assert stats2.chunks_created == 0

    def test_index_file_force_reindex(self, workspace):
        pipeline = workspace["pipeline"]
        src_dir = workspace["src_dir"]

        f = _write_file(src_dir, "force.py", "x = 1\n")
        pipeline.index_file(f, root=src_dir)

        stats = pipeline.index_file(f, root=src_dir, force=True)
        assert stats.files_processed == 1
        assert stats.chunks_created >= 1

    def test_index_file_updates_changed_file(self, workspace):
        pipeline = workspace["pipeline"]
        meta = workspace["metadata"]
        src_dir = workspace["src_dir"]

        f = _write_file(src_dir, "changing.py", "version = 1\n")
        pipeline.index_file(f, root=src_dir)
        old_chunks = meta.get_chunk_ids_for_file("changing.py")

        # Modify file
        f.write_text("version = 2\nmore code\n", encoding="utf-8")
        stats = pipeline.index_file(f, root=src_dir)
        assert stats.files_processed == 1

        new_chunks = meta.get_chunk_ids_for_file("changing.py")
        # Old chunks should have been tombstoned, new ones assigned
        assert set(old_chunks) != set(new_chunks)

    def test_index_file_registers_in_metadata(self, workspace):
        pipeline = workspace["pipeline"]
        meta = workspace["metadata"]
        fts = workspace["fts"]
        src_dir = workspace["src_dir"]

        f = _write_file(src_dir, "meta_test.py", "def foo(): pass\n")
        pipeline.index_file(f, root=src_dir)

        # MetadataStore has file registered
        assert meta.get_file_hash("meta_test.py") is not None
        chunk_ids = meta.get_chunk_ids_for_file("meta_test.py")
        assert len(chunk_ids) >= 1

        # FTS has the content
        fts_ids = fts.get_chunk_ids_by_path("meta_test.py")
        assert len(fts_ids) >= 1

    def test_index_file_no_metadata_raises(self, workspace):
        cfg = workspace["config"]
        pipeline_no_meta = IndexingPipeline(
            embedder=FakeEmbedder(),
            binary_store=workspace["binary_store"],
            ann_index=workspace["ann_index"],
            fts=workspace["fts"],
            config=cfg,
        )
        f = _write_file(workspace["src_dir"], "no_meta.py", "x = 1\n")
        with pytest.raises(RuntimeError, match="MetadataStore is required"):
            pipeline_no_meta.index_file(f)


# ---------------------------------------------------------------------------
# remove_file tests
# ---------------------------------------------------------------------------


class TestRemoveFile:
    def test_remove_file_tombstones_and_fts(self, workspace):
        pipeline = workspace["pipeline"]
        meta = workspace["metadata"]
        fts = workspace["fts"]
        src_dir = workspace["src_dir"]

        f = _write_file(src_dir, "to_remove.py", "data = [1, 2, 3]\n")
        pipeline.index_file(f, root=src_dir)

        chunk_ids = meta.get_chunk_ids_for_file("to_remove.py")
        assert len(chunk_ids) >= 1

        pipeline.remove_file("to_remove.py")

        # File should be gone from metadata
        assert meta.get_file_hash("to_remove.py") is None
        assert meta.get_chunk_ids_for_file("to_remove.py") == []

        # Chunks should be in deleted_chunks
        deleted = meta.get_deleted_ids()
        for cid in chunk_ids:
            assert cid in deleted

        # FTS should be cleared
        assert fts.get_chunk_ids_by_path("to_remove.py") == []

    def test_remove_nonexistent_file(self, workspace):
        pipeline = workspace["pipeline"]
        # Should not raise
        pipeline.remove_file("nonexistent.py")


# ---------------------------------------------------------------------------
# sync tests
# ---------------------------------------------------------------------------


class TestSync:
    def test_sync_indexes_new_files(self, workspace):
        pipeline = workspace["pipeline"]
        meta = workspace["metadata"]
        src_dir = workspace["src_dir"]

        f1 = _write_file(src_dir, "a.py", "a = 1\n")
        f2 = _write_file(src_dir, "b.py", "b = 2\n")

        stats = pipeline.sync([f1, f2], root=src_dir)
        assert stats.files_processed == 2
        assert meta.get_file_hash("a.py") is not None
        assert meta.get_file_hash("b.py") is not None

    def test_sync_removes_missing_files(self, workspace):
        pipeline = workspace["pipeline"]
        meta = workspace["metadata"]
        src_dir = workspace["src_dir"]

        f1 = _write_file(src_dir, "keep.py", "keep = True\n")
        f2 = _write_file(src_dir, "remove.py", "remove = True\n")

        pipeline.sync([f1, f2], root=src_dir)
        assert meta.get_file_hash("remove.py") is not None

        # Sync with only f1 -- f2 should be removed
        stats = pipeline.sync([f1], root=src_dir)
        assert meta.get_file_hash("remove.py") is None
        deleted = meta.get_deleted_ids()
        assert len(deleted) > 0

    def test_sync_detects_changed_files(self, workspace):
        pipeline = workspace["pipeline"]
        meta = workspace["metadata"]
        src_dir = workspace["src_dir"]

        f = _write_file(src_dir, "mutable.py", "v1\n")
        pipeline.sync([f], root=src_dir)
        old_hash = meta.get_file_hash("mutable.py")

        f.write_text("v2\n", encoding="utf-8")
        stats = pipeline.sync([f], root=src_dir)
        assert stats.files_processed == 1
        new_hash = meta.get_file_hash("mutable.py")
        assert old_hash != new_hash

    def test_sync_skips_unchanged(self, workspace):
        pipeline = workspace["pipeline"]
        src_dir = workspace["src_dir"]

        f = _write_file(src_dir, "stable.py", "stable = True\n")
        pipeline.sync([f], root=src_dir)

        # Second sync with same file, unchanged
        stats = pipeline.sync([f], root=src_dir)
        assert stats.files_processed == 0
        assert stats.chunks_created == 0


# ---------------------------------------------------------------------------
# compact tests
# ---------------------------------------------------------------------------


class TestCompact:
    def test_compact_removes_tombstoned_from_binary_store(self, workspace):
        pipeline = workspace["pipeline"]
        meta = workspace["metadata"]
        binary_store = workspace["binary_store"]
        src_dir = workspace["src_dir"]

        f1 = _write_file(src_dir, "alive.py", "alive = True\n")
        f2 = _write_file(src_dir, "dead.py", "dead = True\n")

        pipeline.index_file(f1, root=src_dir)
        pipeline.index_file(f2, root=src_dir)

        count_before = binary_store._count
        assert count_before >= 2

        pipeline.remove_file("dead.py")
        pipeline.compact()

        # BinaryStore should have fewer entries
        assert binary_store._count < count_before
        # deleted_chunks should be cleared
        assert meta.get_deleted_ids() == set()

    def test_compact_noop_when_no_deletions(self, workspace):
        pipeline = workspace["pipeline"]
        meta = workspace["metadata"]
        binary_store = workspace["binary_store"]
        src_dir = workspace["src_dir"]

        f = _write_file(src_dir, "solo.py", "solo = True\n")
        pipeline.index_file(f, root=src_dir)
        count_before = binary_store._count

        pipeline.compact()
        assert binary_store._count == count_before


# ---------------------------------------------------------------------------
# Backward compatibility: existing batch API still works
# ---------------------------------------------------------------------------


class TestBatchAPIUnchanged:
    def test_index_files_still_works(self, workspace):
        pipeline = workspace["pipeline"]
        src_dir = workspace["src_dir"]

        f1 = _write_file(src_dir, "batch1.py", "batch1 = 1\n")
        f2 = _write_file(src_dir, "batch2.py", "batch2 = 2\n")

        stats = pipeline.index_files([f1, f2], root=src_dir)
        assert stats.files_processed == 2
        assert stats.chunks_created >= 2

    def test_index_files_works_without_metadata(self, workspace):
        """Batch API should work even without MetadataStore."""
        cfg = workspace["config"]
        pipeline_no_meta = IndexingPipeline(
            embedder=FakeEmbedder(),
            binary_store=BinaryStore(workspace["store_dir"] / "no_meta", DIM, cfg),
            ann_index=ANNIndex(workspace["store_dir"] / "no_meta", DIM, cfg),
            fts=FTSEngine(str(workspace["store_dir"] / "no_meta_fts.db")),
            config=cfg,
        )
        src_dir = workspace["src_dir"]
        f = _write_file(src_dir, "no_meta_batch.py", "x = 1\n")
        stats = pipeline_no_meta.index_files([f], root=src_dir)
        assert stats.files_processed == 1
