"""L2 integration tests for incremental sync cycle.

Tests the index -> modify -> re-sync -> verify flow using real SQLite,
real FTSEngine, real MetadataStore, but mock embedder.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from codexlens_search.config import Config
from codexlens_search.core import ANNIndex, BinaryStore
from codexlens_search.indexing.metadata import MetadataStore
from codexlens_search.indexing.pipeline import IndexingPipeline
from codexlens_search.search.fts import FTSEngine
from codexlens_search.search.pipeline import SearchPipeline

from tests.integration.conftest import DIM, MockEmbedder, MockReranker


@pytest.fixture
def pipeline_env(tmp_path):
    """Create a full IndexingPipeline + SearchPipeline environment."""
    config = Config.small()
    embedder = MockEmbedder()
    reranker = MockReranker()

    binary_store = BinaryStore(tmp_path / "binary", dim=DIM, config=config)
    ann_index = ANNIndex(tmp_path / "ann.hnsw", dim=DIM, config=config)
    fts = FTSEngine(tmp_path / "fts.db")
    metadata = MetadataStore(tmp_path / "metadata.db")

    indexing = IndexingPipeline(
        embedder=embedder,
        binary_store=binary_store,
        ann_index=ann_index,
        fts=fts,
        config=config,
        metadata=metadata,
    )

    search = SearchPipeline(
        embedder=embedder,
        binary_store=binary_store,
        ann_index=ann_index,
        reranker=reranker,
        fts=fts,
        config=config,
        metadata_store=metadata,
    )

    return indexing, search, metadata, tmp_path


@pytest.fixture
def source_dir(tmp_path):
    """Create a temp source directory with sample files."""
    src = tmp_path / "src"
    src.mkdir()
    return src


def _write_file(path, content):
    """Write content and ensure mtime is updated."""
    path.write_text(content, encoding="utf-8")


class TestIncrementalSync:
    """Test incremental sync: index -> modify -> re-sync -> verify."""

    def test_initial_sync_indexes_all_files(self, pipeline_env, source_dir):
        indexing, search, metadata, _ = pipeline_env

        _write_file(source_dir / "auth.py", "def authenticate(user, password): pass")
        _write_file(source_dir / "models.py", "class User: name = ''")

        files = list(source_dir.glob("*.py"))
        stats = indexing.sync(files, root=source_dir)

        assert stats.files_processed == 2
        assert stats.chunks_created >= 2

        all_files = metadata.get_all_files()
        assert "auth.py" in all_files
        assert "models.py" in all_files

    def test_resync_unchanged_files_skipped(self, pipeline_env, source_dir):
        indexing, search, metadata, _ = pipeline_env

        _write_file(source_dir / "auth.py", "def authenticate(user, password): pass")
        files = list(source_dir.glob("*.py"))

        stats1 = indexing.sync(files, root=source_dir)
        assert stats1.files_processed == 1

        # Re-sync without changes: no files should be re-indexed
        stats2 = indexing.sync(files, root=source_dir)
        assert stats2.files_processed == 0

    def test_modified_file_reindexed(self, pipeline_env, source_dir):
        indexing, search, metadata, _ = pipeline_env

        auth_file = source_dir / "auth.py"
        _write_file(auth_file, "def authenticate(user, password): pass")

        files = list(source_dir.glob("*.py"))
        indexing.sync(files, root=source_dir)

        # Modify file content (need different mtime)
        time.sleep(0.05)
        _write_file(auth_file, "def authenticate(user, password, token): return True")

        stats = indexing.sync(files, root=source_dir)
        assert stats.files_processed == 1
        assert stats.chunks_created >= 1

    def test_modified_file_old_chunks_tombstoned(self, pipeline_env, source_dir):
        indexing, search, metadata, _ = pipeline_env

        auth_file = source_dir / "auth.py"
        _write_file(auth_file, "def authenticate(user, password): pass")

        files = list(source_dir.glob("*.py"))
        indexing.sync(files, root=source_dir)

        old_chunks = metadata.get_chunk_ids_for_file("auth.py")
        assert len(old_chunks) > 0

        # Modify file
        time.sleep(0.05)
        _write_file(auth_file, "def authenticate_v2(user, token): return validate(token)")

        indexing.sync(files, root=source_dir)

        # Old chunk IDs should be tombstoned
        deleted = metadata.get_deleted_ids()
        for old_id in old_chunks:
            assert old_id in deleted, f"Old chunk {old_id} should be tombstoned"

    def test_search_returns_updated_content(self, pipeline_env, source_dir):
        indexing, search, metadata, _ = pipeline_env

        auth_file = source_dir / "auth.py"
        _write_file(auth_file, "def old_function(): return None")

        files = list(source_dir.glob("*.py"))
        indexing.sync(files, root=source_dir)

        # Modify with new content
        time.sleep(0.05)
        _write_file(auth_file, "def new_function(): return authenticate_user()")

        indexing.sync(files, root=source_dir)

        # Search should find the new content
        results = search.search("new_function")
        assert len(results) > 0
        # At least one result should contain the new content
        contents = [r.content for r in results]
        assert any("new_function" in c for c in contents)

    def test_deleted_file_excluded_from_search(self, pipeline_env, source_dir):
        indexing, search, metadata, _ = pipeline_env

        _write_file(source_dir / "keep.py", "def keep_this(): return True")
        _write_file(source_dir / "remove.py", "def remove_this(): return False")

        files = list(source_dir.glob("*.py"))
        indexing.sync(files, root=source_dir)

        # Delete one file
        (source_dir / "remove.py").unlink()
        files = list(source_dir.glob("*.py"))
        indexing.sync(files, root=source_dir)

        # Search should not return removed content
        results = search.search("remove_this")
        result_contents = [r.content for r in results]
        assert not any("remove_this" in c for c in result_contents)

    def test_compact_clears_tombstones(self, pipeline_env, source_dir):
        indexing, search, metadata, _ = pipeline_env

        _write_file(source_dir / "temp.py", "def temporary(): pass")
        files = list(source_dir.glob("*.py"))
        indexing.sync(files, root=source_dir)

        # Delete file to create tombstones
        (source_dir / "temp.py").unlink()
        indexing.sync([], root=source_dir)

        deleted_before = metadata.get_deleted_ids()
        assert len(deleted_before) > 0

        # Compact should clear tombstones
        compacted = metadata.compact_deleted()
        assert compacted == deleted_before
        assert len(metadata.get_deleted_ids()) == 0
