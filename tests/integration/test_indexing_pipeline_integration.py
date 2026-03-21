"""L2 integration tests for IndexingPipeline covering chunking modes,
file type detection, binary detection, generated code skip, and language detection.

Targets: indexing/pipeline.py coverage from 48% toward 60%+.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from codexlens_search.config import Config
from codexlens_search.core import ANNIndex, BinaryStore
from codexlens_search.indexing.metadata import MetadataStore
from codexlens_search.indexing.pipeline import (
    IndexingPipeline,
    IndexStats,
    detect_language,
    is_file_excluded,
)
from codexlens_search.search.fts import FTSEngine
from codexlens_search.search.pipeline import SearchPipeline

from tests.integration.conftest import DIM, MockEmbedder, MockReranker


@pytest.fixture
def pipeline_env(tmp_path):
    """Full pipeline environment with real stores."""
    config = Config.small()
    config.embed_dim = DIM
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

    return indexing, search, metadata, config, tmp_path


@pytest.fixture
def src_dir(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    return src


class TestLanguageDetection:
    """Test detect_language for various file extensions."""

    @pytest.mark.parametrize(
        "path, expected",
        [
            ("main.py", "python"),
            ("app.js", "javascript"),
            ("app.jsx", "javascript"),
            ("index.ts", "typescript"),
            ("index.tsx", "typescript"),
            ("main.go", "go"),
            ("Main.java", "java"),
            ("lib.rs", "rust"),
            ("main.cpp", "cpp"),
            ("main.c", "c"),
            ("header.h", "c"),
            ("header.hpp", "cpp"),
            ("app.rb", "ruby"),
            ("index.php", "php"),
            ("script.sh", "bash"),
            ("unknown.xyz", None),
            ("readme.md", None),
            ("data.json", None),
        ],
    )
    def test_detect_language(self, path, expected):
        assert detect_language(path) == expected


class TestFileExclusion:
    """Test is_file_excluded with various file types and configs."""

    def test_exclude_binary_file(self, src_dir):
        """Files with high null byte ratio should be excluded."""
        binary = src_dir / "image.dat"
        # Create file with >10% null bytes
        data = b"\x00" * 300 + b"some text" * 20
        binary.write_bytes(data)

        config = Config.small()
        reason = is_file_excluded(binary, config)
        assert reason is not None
        assert "binary" in reason.lower()

    def test_exclude_generated_code(self, src_dir):
        """Files with generated code markers should be excluded."""
        gen_file = src_dir / "generated.py"
        gen_file.write_text("# @generated\ndef foo(): pass\n", encoding="utf-8")

        config = Config.small()
        reason = is_file_excluded(gen_file, config)
        assert reason is not None
        assert "generated" in reason.lower()

    def test_exclude_large_file(self, src_dir):
        """Files exceeding max_file_size_bytes should be excluded."""
        large = src_dir / "large.py"
        large.write_text("x" * 2_000_000, encoding="utf-8")

        config = Config.small()
        config.max_file_size_bytes = 1_000_000
        reason = is_file_excluded(large, config)
        assert reason is not None
        assert "max size" in reason.lower()

    def test_exclude_empty_file(self, src_dir):
        """Empty files should be excluded."""
        empty = src_dir / "empty.py"
        empty.write_bytes(b"")

        config = Config.small()
        reason = is_file_excluded(empty, config)
        assert reason is not None
        assert "empty" in reason.lower()

    def test_exclude_by_extension(self, src_dir):
        """Files with excluded extensions should be excluded."""
        img = src_dir / "photo.png"
        img.write_bytes(b"fake png data " * 10)

        config = Config.small()
        reason = is_file_excluded(img, config)
        assert reason is not None
        assert "extension" in reason.lower()

    def test_normal_python_file_not_excluded(self, src_dir):
        """Normal Python file should not be excluded."""
        normal = src_dir / "normal.py"
        normal.write_text("def hello(): return 'world'\n", encoding="utf-8")

        config = Config.small()
        reason = is_file_excluded(normal, config)
        assert reason is None

    def test_exclude_with_content_param(self, src_dir):
        """When content bytes are provided, should use them instead of reading file."""
        f = src_dir / "check.py"
        f.write_text("def foo(): pass\n", encoding="utf-8")

        # Provide binary content that should trigger binary detection
        binary_content = b"\x00" * 500 + b"x" * 100
        config = Config.small()
        reason = is_file_excluded(f, config, content=binary_content)
        assert reason is not None
        assert "binary" in reason.lower()

    def test_generated_code_marker_do_not_edit(self, src_dir):
        """DO NOT EDIT marker should trigger exclusion."""
        f = src_dir / "auto.py"
        f.write_text("# DO NOT EDIT\n# Auto-generated\ndef foo(): pass\n", encoding="utf-8")

        config = Config.small()
        reason = is_file_excluded(f, config)
        assert reason is not None
        assert "generated" in reason.lower()


class TestIndexingPipelineDifferentFileTypes:
    """Test indexing pipeline with different file types (Python, JS, plain text)."""

    def test_index_python_file(self, pipeline_env, src_dir):
        indexing, search, metadata, _, _ = pipeline_env

        (src_dir / "auth.py").write_text(
            "def authenticate(user, password):\n    return check_hash(password, user.hash)\n",
            encoding="utf-8",
        )

        files = list(src_dir.glob("*.py"))
        stats = indexing.sync(files, root=src_dir)

        assert stats.files_processed == 1
        assert stats.chunks_created >= 1

        results = search.search("authenticate")
        assert len(results) > 0

    def test_index_javascript_file(self, pipeline_env, src_dir):
        indexing, search, metadata, _, _ = pipeline_env

        (src_dir / "app.js").write_text(
            "function handleRequest(req, res) {\n  return res.json({ ok: true });\n}\n",
            encoding="utf-8",
        )

        files = list(src_dir.glob("*.js"))
        stats = indexing.sync(files, root=src_dir)

        assert stats.files_processed == 1
        results = search.search("handleRequest")
        assert len(results) > 0

    def test_index_plain_text_file(self, pipeline_env, src_dir):
        indexing, search, metadata, _, _ = pipeline_env

        (src_dir / "notes.txt").write_text(
            "This is a plain text document about authentication patterns.\n"
            "It discusses various login mechanisms and security best practices.\n",
            encoding="utf-8",
        )

        files = list(src_dir.glob("*.txt"))
        stats = indexing.sync(files, root=src_dir)

        assert stats.files_processed == 1
        results = search.search("authentication")
        assert len(results) > 0

    def test_index_mixed_file_types(self, pipeline_env, src_dir):
        """Index mix of Python, JS, and text files."""
        indexing, search, metadata, _, _ = pipeline_env

        (src_dir / "server.py").write_text("def start_server(): pass\n", encoding="utf-8")
        (src_dir / "client.js").write_text("function connect() { return ws(); }\n", encoding="utf-8")
        (src_dir / "readme.txt").write_text("Server and client implementation\n", encoding="utf-8")

        files = list(src_dir.iterdir())
        stats = indexing.sync(files, root=src_dir)

        assert stats.files_processed == 3
        assert stats.chunks_created >= 3


class TestCodeAwareChunking:
    """Test code-aware chunking produces boundary-respecting chunks."""

    def test_code_aware_chunking_splits_at_functions(self, pipeline_env, src_dir):
        """Code with multiple functions should chunk at function boundaries."""
        indexing, search, metadata, config, _ = pipeline_env

        code = "\n".join([
            "def function_alpha():",
            "    return 'alpha result'",
            "",
            "def function_beta():",
            "    return 'beta result'",
            "",
            "def function_gamma():",
            "    return 'gamma result'",
            "",
            "class DataProcessor:",
            "    def process(self):",
            "        return self.data",
        ])
        (src_dir / "multi.py").write_text(code, encoding="utf-8")

        files = list(src_dir.glob("*.py"))
        stats = indexing.sync(files, root=src_dir)

        assert stats.files_processed == 1
        assert stats.chunks_created >= 1

        # Should be searchable by individual function names
        for name in ["function_alpha", "function_beta", "DataProcessor"]:
            results = search.search(name)
            assert len(results) > 0, f"Expected results for {name}"

    def test_plain_text_chunking_disabled(self, pipeline_env, src_dir):
        """With code_aware_chunking=False, should use plain text chunking."""
        indexing, search, metadata, config, _ = pipeline_env
        config.code_aware_chunking = False

        code = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        (src_dir / "simple.py").write_text(code, encoding="utf-8")

        files = list(src_dir.glob("*.py"))
        stats = indexing.sync(files, root=src_dir)
        assert stats.files_processed == 1
        assert stats.chunks_created >= 1


class TestIndexingPipelineSkipsExcluded:
    """Test that excluded files are properly skipped during indexing."""

    def test_binary_file_skipped(self, pipeline_env, src_dir):
        indexing, search, metadata, _, _ = pipeline_env

        # Normal file
        (src_dir / "good.py").write_text("def good(): pass\n", encoding="utf-8")
        # Binary file
        (src_dir / "bad.dat").write_bytes(b"\x00" * 500 + b"x" * 100)

        files = list(src_dir.iterdir())
        stats = indexing.sync(files, root=src_dir)

        # Only the .py file should be processed (the .dat has null bytes)
        assert stats.files_processed == 1

    def test_generated_file_skipped(self, pipeline_env, src_dir):
        indexing, search, metadata, _, _ = pipeline_env

        (src_dir / "real.py").write_text("def real(): pass\n", encoding="utf-8")
        (src_dir / "gen.py").write_text("# @generated\ndef gen(): pass\n", encoding="utf-8")

        files = list(src_dir.glob("*.py"))
        stats = indexing.sync(files, root=src_dir)

        assert stats.files_processed == 1

    def test_empty_files_skipped(self, pipeline_env, src_dir):
        indexing, search, metadata, _, _ = pipeline_env

        (src_dir / "real.py").write_text("def real(): pass\n", encoding="utf-8")
        (src_dir / "empty.py").write_bytes(b"")

        files = list(src_dir.glob("*.py"))
        stats = indexing.sync(files, root=src_dir)

        assert stats.files_processed == 1


class TestIndexFileSingle:
    """Test index_file() for single file indexing."""

    def test_index_single_file(self, pipeline_env, src_dir):
        indexing, search, metadata, _, _ = pipeline_env

        f = src_dir / "single.py"
        f.write_text("def unique_function(): return 42\n", encoding="utf-8")

        stats = indexing.index_file(f, root=src_dir)
        assert stats.files_processed == 1
        assert stats.chunks_created >= 1

        results = search.search("unique_function")
        assert len(results) > 0


class TestRemoveFile:
    """Test remove_file() marks content as deleted."""

    def test_remove_file_tombstones_chunks(self, pipeline_env, src_dir):
        indexing, search, metadata, _, _ = pipeline_env

        f = src_dir / "removable.py"
        f.write_text("def to_be_removed(): pass\n", encoding="utf-8")

        indexing.sync([f], root=src_dir)
        assert metadata.get_file_hash("removable.py") is not None

        indexing.remove_file("removable.py")
        deleted = metadata.get_deleted_ids()
        assert len(deleted) > 0


class TestFTSOnlyIndexing:
    """Test index_files_fts_only for FTS-only mode."""

    def test_fts_only_indexes_into_fts(self, pipeline_env, src_dir):
        indexing, search, metadata, _, _ = pipeline_env

        (src_dir / "fts_only.py").write_text(
            "def fts_only_function(): return 'fts'\n", encoding="utf-8"
        )

        files = list(src_dir.glob("*.py"))
        stats = indexing.index_files_fts_only(files, root=src_dir)

        assert stats.files_processed == 1
        assert stats.chunks_created >= 1

        # FTS search should find it
        results = search.search("fts_only_function", quality="fast")
        assert len(results) > 0


class TestEmptyInputs:
    """Test edge cases with empty inputs."""

    def test_sync_empty_file_list(self, pipeline_env, src_dir):
        indexing, search, metadata, _, _ = pipeline_env

        stats = indexing.sync([], root=src_dir)
        assert stats.files_processed == 0
        assert stats.chunks_created == 0

    def test_index_files_empty_list(self, pipeline_env, src_dir):
        indexing, _, _, _, _ = pipeline_env

        stats = indexing.index_files([], root=src_dir)
        assert stats.files_processed == 0
        assert stats.chunks_created == 0
