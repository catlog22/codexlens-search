"""Unit tests for FTSEngine symbol storage and retrieval."""
from __future__ import annotations

import pytest

from codexlens_search.search.fts import FTSEngine


@pytest.fixture
def fts(tmp_path):
    return FTSEngine(str(tmp_path / "fts.db"))


class TestSymbolsTableCreation:
    def test_symbols_table_exists(self, fts):
        tables = {
            row[0]
            for row in fts._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "symbols" in tables

    def test_symbols_table_columns(self, fts):
        cols = [
            row[1]
            for row in fts._conn.execute("PRAGMA table_info(symbols)").fetchall()
        ]
        expected = [
            "id", "chunk_id", "name", "kind",
            "start_line", "end_line", "parent_name", "signature", "language",
        ]
        assert cols == expected

    def test_symbols_indexes_exist(self, fts):
        indexes = {
            row[0]
            for row in fts._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='symbols'"
            ).fetchall()
        }
        assert "idx_symbols_chunk_id" in indexes
        assert "idx_symbols_name" in indexes
        assert "idx_symbols_kind" in indexes


class TestAddSymbols:
    def test_batch_insert(self, fts):
        symbols = [
            (0, "foo", "function", 1, 5, "", "def foo():", "python"),
            (0, "Bar", "class", 7, 20, "", "class Bar:", "python"),
            (1, "baz", "function", 1, 3, "Bar", "def baz(self):", "python"),
        ]
        fts.add_symbols(symbols)
        count = fts._conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        assert count == 3

    def test_empty_list(self, fts):
        fts.add_symbols([])
        count = fts._conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        assert count == 0


class TestGetSymbolsByName:
    def test_lookup_by_name(self, fts):
        fts.add_symbols([
            (0, "foo", "function", 1, 5, "", "def foo():", "python"),
            (1, "foo", "function", 10, 15, "", "def foo():", "python"),
            (0, "bar", "function", 6, 9, "", "def bar():", "python"),
        ])
        results = fts.get_symbols_by_name("foo")
        assert len(results) == 2
        assert all(r["name"] == "foo" for r in results)
        assert {r["chunk_id"] for r in results} == {0, 1}

    def test_lookup_by_name_and_kind(self, fts):
        fts.add_symbols([
            (0, "Foo", "class", 1, 20, "", "class Foo:", "python"),
            (0, "Foo", "function", 22, 25, "", "def Foo():", "python"),
        ])
        results = fts.get_symbols_by_name("Foo", kind="class")
        assert len(results) == 1
        assert results[0]["kind"] == "class"

    def test_lookup_nonexistent(self, fts):
        results = fts.get_symbols_by_name("nonexistent")
        assert results == []

    def test_result_dict_keys(self, fts):
        fts.add_symbols([
            (5, "test_fn", "function", 10, 15, "MyClass", "def test_fn():", "python"),
        ])
        results = fts.get_symbols_by_name("test_fn")
        assert len(results) == 1
        r = results[0]
        assert r["chunk_id"] == 5
        assert r["name"] == "test_fn"
        assert r["kind"] == "function"
        assert r["start_line"] == 10
        assert r["end_line"] == 15
        assert r["parent_name"] == "MyClass"
        assert r["signature"] == "def test_fn():"
        assert r["language"] == "python"


class TestGetSymbolsByChunk:
    def test_returns_chunk_symbols(self, fts):
        fts.add_symbols([
            (0, "foo", "function", 1, 5, "", "", "python"),
            (0, "bar", "function", 6, 9, "", "", "python"),
            (1, "baz", "function", 1, 3, "", "", "python"),
        ])
        results = fts.get_symbols_by_chunk(0)
        assert len(results) == 2
        names = {r["name"] for r in results}
        assert names == {"foo", "bar"}

    def test_empty_chunk(self, fts):
        results = fts.get_symbols_by_chunk(999)
        assert results == []


class TestDeleteSymbolsByChunkIds:
    def test_delete_by_chunk_ids(self, fts):
        fts.add_symbols([
            (0, "a", "function", 1, 5, "", "", "python"),
            (1, "b", "function", 1, 5, "", "", "python"),
            (2, "c", "function", 1, 5, "", "", "python"),
        ])
        deleted = fts.delete_symbols_by_chunk_ids([0, 1])
        assert deleted == 2
        # chunk 2 remains
        remaining = fts._conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        assert remaining == 1

    def test_delete_empty_list(self, fts):
        deleted = fts.delete_symbols_by_chunk_ids([])
        assert deleted == 0


class TestDeleteByPathCleansSymbols:
    def test_symbols_cleaned_on_path_delete(self, fts):
        """delete_by_path must remove symbols before removing chunks."""
        fts.add_documents([
            (0, "target.py", "def foo(): pass", 1, 1, "python"),
            (1, "target.py", "def bar(): pass", 2, 2, "python"),
            (2, "keep.py", "def baz(): pass", 1, 1, "python"),
        ])
        fts.add_symbols([
            (0, "foo", "function", 1, 1, "", "def foo(): pass", "python"),
            (1, "bar", "function", 2, 2, "", "def bar(): pass", "python"),
            (2, "baz", "function", 1, 1, "", "def baz(): pass", "python"),
        ])
        count = fts.delete_by_path("target.py")
        assert count == 2

        # Symbols for target.py chunks should be gone
        assert fts.get_symbols_by_chunk(0) == []
        assert fts.get_symbols_by_chunk(1) == []

        # Symbols for keep.py should remain
        assert len(fts.get_symbols_by_chunk(2)) == 1

    def test_delete_nonexistent_path_no_symbol_side_effects(self, fts):
        fts.add_symbols([
            (0, "x", "function", 1, 1, "", "", "python"),
        ])
        fts.delete_by_path("nonexistent.py")
        assert len(fts.get_symbols_by_chunk(0)) == 1
