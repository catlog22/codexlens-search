"""Additional pipeline.py coverage tests — chunking, metadata, sync, remove_file."""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from codexlens_search.config import Config
from codexlens_search.indexing.pipeline import (
    IndexingPipeline,
    IndexStats,
    _DEFAULT_MAX_CHUNK_CHARS,
    _DEFAULT_CHUNK_OVERLAP,
)


@pytest.fixture
def pipeline_mocks(tmp_path):
    """Create an IndexingPipeline with all mocked dependencies."""
    embedder = MagicMock()
    embedder.embed_batch.return_value = [np.zeros(32, dtype=np.float32)]

    binary_store = MagicMock()
    ann_index = MagicMock()
    fts = MagicMock()
    metadata = MagicMock()
    config = Config()
    config.embed_batch_size = 2
    config.index_workers = 1

    pipeline = IndexingPipeline(
        embedder=embedder,
        binary_store=binary_store,
        ann_index=ann_index,
        fts=fts,
        config=config,
        metadata=metadata,
    )
    return pipeline, embedder, binary_store, ann_index, fts, metadata


# ---------------------------------------------------------------------------
# _content_hash (covers line 762)
# ---------------------------------------------------------------------------

class TestContentHash:
    def test_content_hash(self, pipeline_mocks):
        pipeline = pipeline_mocks[0]
        result = pipeline._content_hash("hello world")
        expected = hashlib.sha256("hello world".encode("utf-8")).hexdigest()
        assert result == expected

    def test_content_hash_empty(self, pipeline_mocks):
        pipeline = pipeline_mocks[0]
        result = pipeline._content_hash("")
        assert len(result) == 64  # SHA-256 hex digest


# ---------------------------------------------------------------------------
# _require_metadata (covers lines 766-771)
# ---------------------------------------------------------------------------

class TestRequireMetadata:
    def test_raises_when_no_metadata(self):
        pipeline = IndexingPipeline(
            embedder=MagicMock(),
            binary_store=MagicMock(),
            ann_index=MagicMock(),
            fts=MagicMock(),
            config=Config(),
            metadata=None,
        )
        with pytest.raises(RuntimeError, match="MetadataStore is required"):
            pipeline._require_metadata()

    def test_returns_metadata_when_present(self, pipeline_mocks):
        pipeline, _, _, _, _, metadata = pipeline_mocks
        result = pipeline._require_metadata()
        assert result is metadata


# ---------------------------------------------------------------------------
# _next_chunk_id (covers lines 773-776)
# ---------------------------------------------------------------------------

class TestNextChunkId:
    def test_next_chunk_id(self, pipeline_mocks):
        pipeline, _, _, _, _, metadata = pipeline_mocks
        metadata.max_chunk_id.return_value = 42
        assert pipeline._next_chunk_id() == 43

    def test_next_chunk_id_empty_store(self, pipeline_mocks):
        pipeline, _, _, _, _, metadata = pipeline_mocks
        metadata.max_chunk_id.return_value = -1
        assert pipeline._next_chunk_id() == 0


# ---------------------------------------------------------------------------
# _chunk_text (covers lines 491-530)
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_empty_text(self):
        result = IndexingPipeline._chunk_text("", "test.py", 100, 10)
        assert result == []

    def test_whitespace_only(self):
        result = IndexingPipeline._chunk_text("   \n  \n  ", "test.py", 100, 10)
        assert result == []

    def test_single_chunk(self):
        text = "def hello():\n    return 'world'\n"
        result = IndexingPipeline._chunk_text(text, "test.py", 1000, 10)
        assert len(result) == 1
        assert result[0][1] == "test.py"
        assert result[0][2] == 1  # start_line
        assert result[0][3] == 2  # end_line

    def test_multiple_chunks_with_overlap(self):
        # Create text that needs multiple chunks
        lines = [f"line {i}\n" for i in range(20)]
        text = "".join(lines)
        result = IndexingPipeline._chunk_text(text, "big.py", 50, 10)
        assert len(result) > 1
        # All chunks should reference the same path
        for chunk in result:
            assert chunk[1] == "big.py"


# ---------------------------------------------------------------------------
# _chunk_code (covers lines 544-614)
# ---------------------------------------------------------------------------

class TestChunkCode:
    def test_no_boundaries_falls_back(self, pipeline_mocks):
        pipeline = pipeline_mocks[0]
        text = "x = 1\ny = 2\nz = 3\n"
        result = pipeline._chunk_code(text, "simple.py", 1000, 10)
        assert len(result) >= 1

    def test_code_with_boundaries(self, pipeline_mocks):
        pipeline = pipeline_mocks[0]
        text = (
            "import os\n"
            "\n"
            "def foo():\n"
            "    return 1\n"
            "\n"
            "def bar():\n"
            "    return 2\n"
            "\n"
            "class MyClass:\n"
            "    pass\n"
        )
        result = pipeline._chunk_code(text, "code.py", 1000, 10)
        assert len(result) >= 1

    def test_empty_code(self, pipeline_mocks):
        pipeline = pipeline_mocks[0]
        result = pipeline._chunk_code("", "empty.py", 100, 10)
        assert result == []


# ---------------------------------------------------------------------------
# _smart_chunk (covers lines 616-658)
# ---------------------------------------------------------------------------

class TestSmartChunk:
    def test_smart_chunk_plain_text(self, pipeline_mocks):
        pipeline = pipeline_mocks[0]
        text = "This is a readme\n" * 5
        result = pipeline._smart_chunk(text, "readme.txt", 1000, 10)
        assert len(result) >= 1
        # Language should be empty for .txt
        for chunk in result:
            assert chunk[4] == ""  # language

    def test_smart_chunk_python_file(self, pipeline_mocks):
        pipeline = pipeline_mocks[0]
        pipeline._config.code_aware_chunking = True
        text = "def hello():\n    pass\n"
        result = pipeline._smart_chunk(text, "test.py", 1000, 10)
        assert len(result) >= 1
        assert result[0][4] == "python"

    def test_smart_chunk_ast_chunking_fallback(self, pipeline_mocks):
        pipeline = pipeline_mocks[0]
        pipeline._config.ast_chunking = True
        text = "def hello():\n    pass\n"
        # Even if AST chunking fails, should fall back gracefully
        result = pipeline._smart_chunk(text, "test.py", 1000, 10)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# _should_extract_symbols (covers lines 714-724)
# ---------------------------------------------------------------------------

class TestShouldExtractSymbols:
    def test_disabled_when_no_ast_chunking(self, pipeline_mocks):
        pipeline = pipeline_mocks[0]
        pipeline._config.ast_chunking = False
        assert pipeline._should_extract_symbols("python") is False

    def test_disabled_for_empty_language(self, pipeline_mocks):
        pipeline = pipeline_mocks[0]
        pipeline._config.ast_chunking = True
        assert pipeline._should_extract_symbols("") is False

    def test_enabled_with_ast_languages_none(self, pipeline_mocks):
        pipeline = pipeline_mocks[0]
        pipeline._config.ast_chunking = True
        pipeline._config.ast_languages = None
        # Will be True if _HAS_AST_CHUNKER is True
        from codexlens_search.indexing.pipeline import _HAS_AST_CHUNKER
        assert pipeline._should_extract_symbols("python") == _HAS_AST_CHUNKER

    def test_enabled_with_matching_language(self, pipeline_mocks):
        pipeline = pipeline_mocks[0]
        pipeline._config.ast_chunking = True
        pipeline._config.ast_languages = {"python", "javascript"}
        from codexlens_search.indexing.pipeline import _HAS_AST_CHUNKER
        assert pipeline._should_extract_symbols("python") == _HAS_AST_CHUNKER

    def test_disabled_with_non_matching_language(self, pipeline_mocks):
        pipeline = pipeline_mocks[0]
        pipeline._config.ast_chunking = True
        pipeline._config.ast_languages = {"javascript"}
        assert pipeline._should_extract_symbols("python") is False


# ---------------------------------------------------------------------------
# remove_file (covers lines 1041-1057)
# ---------------------------------------------------------------------------

class TestRemoveFile:
    def test_remove_file(self, pipeline_mocks):
        pipeline, _, _, _, fts, metadata = pipeline_mocks
        metadata.mark_file_deleted.return_value = 3
        fts.delete_by_path.return_value = 3

        pipeline.remove_file("src/old.py")

        metadata.mark_file_deleted.assert_called_once_with("src/old.py")
        fts.delete_by_path.assert_called_once_with("src/old.py")


# ---------------------------------------------------------------------------
# index_files empty (covers line 241)
# ---------------------------------------------------------------------------

class TestIndexFilesEmpty:
    def test_empty_files_list(self, pipeline_mocks):
        pipeline = pipeline_mocks[0]
        result = pipeline.index_files([])
        assert result.files_processed == 0
        assert result.chunks_created == 0


# ---------------------------------------------------------------------------
# _get_gitignore_matcher (covers lines 205-216)
# ---------------------------------------------------------------------------

class TestGetGitignoreMatcher:
    def test_disabled_config(self, pipeline_mocks):
        pipeline = pipeline_mocks[0]
        pipeline._config.gitignore_filtering = False
        result = pipeline._get_gitignore_matcher(Path("/tmp"))
        assert result is None

    def test_no_root(self, pipeline_mocks):
        pipeline = pipeline_mocks[0]
        pipeline._config.gitignore_filtering = True
        result = pipeline._get_gitignore_matcher(None)
        assert result is None


# ---------------------------------------------------------------------------
# index_files_fts_only empty (covers line 800-801)
# ---------------------------------------------------------------------------

class TestIndexFilesFtsOnlyEmpty:
    def test_empty_files(self, pipeline_mocks):
        pipeline = pipeline_mocks[0]
        result = pipeline.index_files_fts_only([])
        assert result.files_processed == 0
