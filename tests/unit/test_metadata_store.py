"""Unit tests for MetadataStore — SQLite file-to-chunk mapping + tombstone tracking."""
from __future__ import annotations

import pytest

from codexlens_search.indexing.metadata import MetadataStore


@pytest.fixture
def store(tmp_path):
    """Create a fresh MetadataStore backed by a temp db."""
    return MetadataStore(str(tmp_path / "meta.db"))


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

class TestTableCreation:
    def test_creates_three_tables(self, store):
        """MetadataStore should create files, chunks, deleted_chunks tables."""
        tables = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = {r[0] for r in tables}
        assert "files" in names
        assert "chunks" in names
        assert "deleted_chunks" in names

    def test_foreign_keys_enabled(self, store):
        """PRAGMA foreign_keys must be ON."""
        row = store._conn.execute("PRAGMA foreign_keys").fetchone()
        assert row[0] == 1

    def test_wal_mode(self, store):
        """journal_mode should be WAL for concurrency."""
        row = store._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0].lower() == "wal"


# ---------------------------------------------------------------------------
# register_file
# ---------------------------------------------------------------------------

class TestRegisterFile:
    def test_register_and_retrieve(self, store):
        store.register_file("src/main.py", "abc123", 1000.0)
        assert store.get_file_hash("src/main.py") == "abc123"

    def test_register_updates_existing(self, store):
        store.register_file("a.py", "hash1", 1000.0)
        store.register_file("a.py", "hash2", 2000.0)
        assert store.get_file_hash("a.py") == "hash2"

    def test_get_file_hash_returns_none_for_unknown(self, store):
        assert store.get_file_hash("nonexistent.py") is None


# ---------------------------------------------------------------------------
# register_chunks
# ---------------------------------------------------------------------------

class TestRegisterChunks:
    def test_register_and_retrieve_chunks(self, store):
        store.register_file("a.py", "h", 1.0)
        store.register_chunks("a.py", [(0, "c0"), (1, "c1"), (2, "c2")])
        ids = store.get_chunk_ids_for_file("a.py")
        assert sorted(ids) == [0, 1, 2]

    def test_empty_chunks_list(self, store):
        store.register_file("a.py", "h", 1.0)
        store.register_chunks("a.py", [])
        assert store.get_chunk_ids_for_file("a.py") == []

    def test_chunks_for_unknown_file(self, store):
        assert store.get_chunk_ids_for_file("unknown.py") == []


# ---------------------------------------------------------------------------
# mark_file_deleted
# ---------------------------------------------------------------------------

class TestMarkFileDeleted:
    def test_tombstones_chunks(self, store):
        store.register_file("a.py", "h", 1.0)
        store.register_chunks("a.py", [(10, "c10"), (11, "c11")])
        count = store.mark_file_deleted("a.py")
        assert count == 2
        assert store.get_deleted_ids() == {10, 11}

    def test_file_removed_after_delete(self, store):
        store.register_file("a.py", "h", 1.0)
        store.register_chunks("a.py", [(0, "c0")])
        store.mark_file_deleted("a.py")
        assert store.get_file_hash("a.py") is None

    def test_chunks_cascaded_after_delete(self, store):
        store.register_file("a.py", "h", 1.0)
        store.register_chunks("a.py", [(0, "c0")])
        store.mark_file_deleted("a.py")
        assert store.get_chunk_ids_for_file("a.py") == []

    def test_delete_nonexistent_file(self, store):
        count = store.mark_file_deleted("nonexistent.py")
        assert count == 0

    def test_delete_file_without_chunks(self, store):
        store.register_file("empty.py", "h", 1.0)
        count = store.mark_file_deleted("empty.py")
        assert count == 0
        assert store.get_file_hash("empty.py") is None


# ---------------------------------------------------------------------------
# file_needs_update
# ---------------------------------------------------------------------------

class TestFileNeedsUpdate:
    def test_new_file_needs_update(self, store):
        assert store.file_needs_update("new.py", "any_hash") is True

    def test_unchanged_file(self, store):
        store.register_file("a.py", "same_hash", 1.0)
        assert store.file_needs_update("a.py", "same_hash") is False

    def test_changed_file(self, store):
        store.register_file("a.py", "old_hash", 1.0)
        assert store.file_needs_update("a.py", "new_hash") is True


# ---------------------------------------------------------------------------
# get_deleted_ids / compact_deleted
# ---------------------------------------------------------------------------

class TestDeletedIdsAndCompact:
    def test_empty_deleted_ids(self, store):
        assert store.get_deleted_ids() == set()

    def test_compact_returns_and_clears(self, store):
        store.register_file("a.py", "h", 1.0)
        store.register_chunks("a.py", [(5, "c5"), (6, "c6")])
        store.mark_file_deleted("a.py")

        deleted = store.compact_deleted()
        assert deleted == {5, 6}
        assert store.get_deleted_ids() == set()

    def test_compact_noop_when_empty(self, store):
        deleted = store.compact_deleted()
        assert deleted == set()


# ---------------------------------------------------------------------------
# get_all_files / max_chunk_id
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_get_all_files(self, store):
        store.register_file("a.py", "h1", 1.0)
        store.register_file("b.py", "h2", 2.0)
        assert store.get_all_files() == {"a.py": "h1", "b.py": "h2"}

    def test_max_chunk_id_empty(self, store):
        assert store.max_chunk_id() == -1

    def test_max_chunk_id_active(self, store):
        store.register_file("a.py", "h", 1.0)
        store.register_chunks("a.py", [(0, "c"), (5, "c"), (3, "c")])
        assert store.max_chunk_id() == 5

    def test_max_chunk_id_includes_deleted(self, store):
        store.register_file("a.py", "h", 1.0)
        store.register_chunks("a.py", [(10, "c")])
        store.mark_file_deleted("a.py")
        assert store.max_chunk_id() == 10

    def test_max_chunk_id_mixed(self, store):
        store.register_file("a.py", "h", 1.0)
        store.register_chunks("a.py", [(3, "c")])
        store.register_file("b.py", "h2", 1.0)
        store.register_chunks("b.py", [(7, "c")])
        store.mark_file_deleted("a.py")
        # deleted has 3, active has 7
        assert store.max_chunk_id() == 7
