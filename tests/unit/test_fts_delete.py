"""Unit tests for FTSEngine delete_by_path and get_chunk_ids_by_path."""
from __future__ import annotations

import pytest

from codexlens_search.search.fts import FTSEngine


@pytest.fixture
def fts(tmp_path):
    return FTSEngine(str(tmp_path / "fts.db"))


class TestGetChunkIdsByPath:
    def test_empty(self, fts):
        assert fts.get_chunk_ids_by_path("a.py") == []

    def test_returns_matching_ids(self, fts):
        fts.add_documents([
            (0, "a.py", "hello world"),
            (1, "a.py", "foo bar"),
            (2, "b.py", "other content"),
        ])
        ids = fts.get_chunk_ids_by_path("a.py")
        assert sorted(ids) == [0, 1]

    def test_no_match(self, fts):
        fts.add_documents([(0, "a.py", "content")])
        assert fts.get_chunk_ids_by_path("b.py") == []


class TestDeleteByPath:
    def test_deletes_docs_and_meta(self, fts):
        fts.add_documents([
            (0, "target.py", "to be deleted"),
            (1, "target.py", "also deleted"),
            (2, "keep.py", "keep this"),
        ])
        count = fts.delete_by_path("target.py")
        assert count == 2

        # target.py gone from both tables
        assert fts.get_chunk_ids_by_path("target.py") == []
        assert fts.get_content(0) == ""
        assert fts.get_content(1) == ""

        # keep.py still there
        assert fts.get_chunk_ids_by_path("keep.py") == [2]
        assert fts.get_content(2) == "keep this"

    def test_delete_nonexistent_path(self, fts):
        count = fts.delete_by_path("nonexistent.py")
        assert count == 0

    def test_delete_then_search(self, fts):
        fts.add_documents([
            (0, "a.py", "unique searchable content"),
            (1, "b.py", "different content here"),
        ])
        fts.delete_by_path("a.py")
        results = fts.exact_search("unique searchable")
        assert len(results) == 0

        results = fts.exact_search("different")
        assert len(results) == 1
        assert results[0][0] == 1
