"""Unit tests for parsers/chunker.py — AST-aware chunking with mocked tree-sitter."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _sub_chunk_lines
# ---------------------------------------------------------------------------

class TestSubChunkLines:
    def test_basic_splitting(self):
        from codexlens_search.parsers.chunker import _sub_chunk_lines

        lines = [f"line {i}\n" for i in range(20)]
        # Each line is ~7 chars. max_chars=50 → ~7 lines per chunk
        chunks = _sub_chunk_lines(lines, 1, 20, "test.py", 50, 0)
        assert len(chunks) > 1
        for text, path, start, end in chunks:
            assert path == "test.py"
            assert start >= 1
            assert end <= 20

    def test_with_overlap(self):
        from codexlens_search.parsers.chunker import _sub_chunk_lines

        lines = [f"x" * 30 + "\n" for _ in range(5)]
        # Each line ~31 chars. max_chars=60 → 2 lines per chunk
        chunks = _sub_chunk_lines(lines, 1, 5, "test.py", 60, 10)
        assert len(chunks) >= 2

    def test_single_line_no_split(self):
        from codexlens_search.parsers.chunker import _sub_chunk_lines

        lines = ["short line\n"]
        chunks = _sub_chunk_lines(lines, 1, 1, "test.py", 800, 0)
        assert len(chunks) == 1
        assert chunks[0][0] == "short line\n"
        assert chunks[0][2] == 1  # start_line
        assert chunks[0][3] == 1  # end_line

    def test_empty_content_skipped(self):
        from codexlens_search.parsers.chunker import _sub_chunk_lines

        lines = ["   \n", "  \n"]
        chunks = _sub_chunk_lines(lines, 1, 2, "test.py", 800, 0)
        # Whitespace-only should be skipped
        assert len(chunks) == 0


# ---------------------------------------------------------------------------
# chunk_by_ast
# ---------------------------------------------------------------------------

class TestChunkByAst:
    def _make_symbol(self, start_line, end_line, name="sym"):
        sym = MagicMock()
        sym.start_line = start_line
        sym.end_line = end_line
        sym.name = name
        return sym

    @patch("codexlens_search.parsers.chunker.extract_symbols")
    @patch("codexlens_search.parsers.chunker.ASTParser")
    def test_returns_empty_when_parse_fails(self, mock_parser_cls, mock_extract):
        from codexlens_search.parsers.chunker import chunk_by_ast

        mock_instance = MagicMock()
        mock_instance.parse.return_value = None
        mock_parser_cls.get_instance.return_value = mock_instance

        result = chunk_by_ast("def foo(): pass", "test.py", "python")
        assert result == []

    @patch("codexlens_search.parsers.chunker.extract_symbols")
    @patch("codexlens_search.parsers.chunker.ASTParser")
    def test_returns_empty_when_no_symbols(self, mock_parser_cls, mock_extract):
        from codexlens_search.parsers.chunker import chunk_by_ast

        mock_instance = MagicMock()
        mock_instance.parse.return_value = MagicMock()  # non-None tree
        mock_parser_cls.get_instance.return_value = mock_instance
        mock_extract.return_value = []

        result = chunk_by_ast("def foo(): pass", "test.py", "python")
        assert result == []

    @patch("codexlens_search.parsers.chunker.extract_symbols")
    @patch("codexlens_search.parsers.chunker.ASTParser")
    def test_returns_empty_for_empty_source(self, mock_parser_cls, mock_extract):
        from codexlens_search.parsers.chunker import chunk_by_ast

        mock_instance = MagicMock()
        mock_instance.parse.return_value = MagicMock()
        mock_parser_cls.get_instance.return_value = mock_instance
        mock_extract.return_value = [self._make_symbol(1, 1)]

        result = chunk_by_ast("", "test.py", "python")
        assert result == []

    @patch("codexlens_search.parsers.chunker.extract_symbols")
    @patch("codexlens_search.parsers.chunker.ASTParser")
    def test_single_symbol_one_chunk(self, mock_parser_cls, mock_extract):
        from codexlens_search.parsers.chunker import chunk_by_ast

        source = "def foo():\n    pass\n"
        mock_instance = MagicMock()
        mock_instance.parse.return_value = MagicMock()
        mock_parser_cls.get_instance.return_value = mock_instance
        mock_extract.return_value = [self._make_symbol(1, 2)]

        chunks = chunk_by_ast(source, "test.py", "python", max_chars=800)
        assert len(chunks) >= 1
        text, path, start, end = chunks[0]
        assert path == "test.py"
        assert start == 1
        assert "def foo" in text

    @patch("codexlens_search.parsers.chunker.extract_symbols")
    @patch("codexlens_search.parsers.chunker.ASTParser")
    def test_multiple_symbols_merged_when_small(self, mock_parser_cls, mock_extract):
        from codexlens_search.parsers.chunker import chunk_by_ast

        source = "def a():\n    pass\ndef b():\n    pass\n"
        mock_instance = MagicMock()
        mock_instance.parse.return_value = MagicMock()
        mock_parser_cls.get_instance.return_value = mock_instance
        mock_extract.return_value = [
            self._make_symbol(1, 2),
            self._make_symbol(3, 4),
        ]

        chunks = chunk_by_ast(source, "test.py", "python", max_chars=800)
        # Small enough to merge into one chunk
        assert len(chunks) == 1
        assert "def a" in chunks[0][0]
        assert "def b" in chunks[0][0]

    @patch("codexlens_search.parsers.chunker.extract_symbols")
    @patch("codexlens_search.parsers.chunker.ASTParser")
    def test_large_symbols_split(self, mock_parser_cls, mock_extract):
        from codexlens_search.parsers.chunker import chunk_by_ast

        # Create source with two large functions
        lines1 = ["def func_a():\n"] + [f"    line{i} = {i}\n" for i in range(30)]
        lines2 = ["def func_b():\n"] + [f"    line{i} = {i}\n" for i in range(30)]
        source = "".join(lines1 + lines2)

        mock_instance = MagicMock()
        mock_instance.parse.return_value = MagicMock()
        mock_parser_cls.get_instance.return_value = mock_instance
        mock_extract.return_value = [
            self._make_symbol(1, 31),
            self._make_symbol(32, 62),
        ]

        # Use small max_chars to force splitting
        chunks = chunk_by_ast(source, "test.py", "python", max_chars=100, overlap=20)
        assert len(chunks) > 2  # Should split into multiple chunks

    @patch("codexlens_search.parsers.chunker.extract_symbols")
    @patch("codexlens_search.parsers.chunker.ASTParser")
    def test_trailing_lines_appended(self, mock_parser_cls, mock_extract):
        from codexlens_search.parsers.chunker import chunk_by_ast

        # Source with trailing content after last symbol
        source = "def foo():\n    pass\n\n# trailing comment\n"
        mock_instance = MagicMock()
        mock_instance.parse.return_value = MagicMock()
        mock_parser_cls.get_instance.return_value = mock_instance
        mock_extract.return_value = [self._make_symbol(1, 2)]

        chunks = chunk_by_ast(source, "test.py", "python", max_chars=800)
        assert len(chunks) >= 1
        # Trailing lines should be included
        combined = "".join(c[0] for c in chunks)
        assert "trailing comment" in combined

    @patch("codexlens_search.parsers.chunker.extract_symbols")
    @patch("codexlens_search.parsers.chunker.ASTParser")
    def test_unsorted_symbols_sorted(self, mock_parser_cls, mock_extract):
        from codexlens_search.parsers.chunker import chunk_by_ast

        source = "a\nb\nc\nd\n"
        mock_instance = MagicMock()
        mock_instance.parse.return_value = MagicMock()
        mock_parser_cls.get_instance.return_value = mock_instance
        # Return symbols in reverse order
        mock_extract.return_value = [
            self._make_symbol(3, 4, "second"),
            self._make_symbol(1, 2, "first"),
        ]

        chunks = chunk_by_ast(source, "test.py", "python", max_chars=800)
        assert len(chunks) >= 1
        # Should process in order: first then second
        assert chunks[0][2] == 1  # start_line of first chunk
