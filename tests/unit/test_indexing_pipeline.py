"""Unit tests for indexing/pipeline.py — language detection, file exclusion, chunking."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codexlens_search.config import Config
from codexlens_search.indexing.pipeline import (
    IndexStats,
    detect_language,
    is_file_excluded,
)


class TestDetectLanguage:
    """Test language detection from file extension."""

    @pytest.mark.parametrize("ext,expected", [
        (".py", "python"),
        (".js", "javascript"),
        (".jsx", "javascript"),
        (".ts", "typescript"),
        (".tsx", "typescript"),
        (".go", "go"),
        (".java", "java"),
        (".rs", "rust"),
        (".cpp", "cpp"),
        (".c", "c"),
        (".h", "c"),
        (".hpp", "cpp"),
        (".rb", "ruby"),
        (".php", "php"),
        (".scala", "scala"),
        (".kt", "kotlin"),
        (".swift", "swift"),
        (".cs", "csharp"),
        (".vue", "vue"),
        (".svelte", "svelte"),
        (".lua", "lua"),
        (".sh", "bash"),
        (".bash", "bash"),
        (".ps1", "powershell"),
        (".r", "r"),
        (".ex", "elixir"),
    ])
    def test_known_extensions(self, ext: str, expected: str) -> None:
        assert detect_language(f"test{ext}") == expected

    def test_unknown_extension(self) -> None:
        assert detect_language("file.xyz") is None

    def test_no_extension(self) -> None:
        assert detect_language("Makefile") is None

    def test_case_insensitive_via_suffix(self) -> None:
        # Path.suffix.lower() handles this
        assert detect_language("test.PY") == "python"


class TestIsFileExcluded:
    """Test file exclusion logic."""

    def test_excluded_extension(self, tmp_path: Path) -> None:
        cfg = Config()
        f = tmp_path / "image.png"
        f.write_bytes(b"fake image data")
        reason = is_file_excluded(f, cfg)
        assert reason is not None
        assert "excluded extension" in reason

    def test_exceeds_max_size(self, tmp_path: Path) -> None:
        cfg = Config()
        cfg.max_file_size_bytes = 100
        f = tmp_path / "big.py"
        f.write_bytes(b"x" * 200)
        reason = is_file_excluded(f, cfg)
        assert reason is not None
        assert "exceeds max size" in reason

    def test_empty_file(self, tmp_path: Path) -> None:
        cfg = Config()
        f = tmp_path / "empty.py"
        f.write_bytes(b"")
        reason = is_file_excluded(f, cfg)
        assert reason == "empty file"

    def test_binary_file_detected(self, tmp_path: Path) -> None:
        cfg = Config()
        cfg.binary_null_threshold = 0.10
        f = tmp_path / "binary.dat"
        # Write content with >10% null bytes
        content = b"\x00" * 30 + b"x" * 70
        f.write_bytes(content)
        reason = is_file_excluded(f, cfg)
        assert reason is not None
        assert "binary file" in reason

    def test_text_file_not_binary(self, tmp_path: Path) -> None:
        cfg = Config()
        f = tmp_path / "normal.py"
        f.write_text("def hello():\n    pass\n")
        reason = is_file_excluded(f, cfg)
        assert reason is None

    def test_generated_code_excluded(self, tmp_path: Path) -> None:
        cfg = Config()
        f = tmp_path / "generated.py"
        f.write_text("# @generated\ndef func(): pass\n")
        reason = is_file_excluded(f, cfg)
        assert reason is not None
        assert "generated code marker" in reason

    def test_generated_code_do_not_edit(self, tmp_path: Path) -> None:
        cfg = Config()
        f = tmp_path / "generated.py"
        f.write_text("# DO NOT EDIT - auto-generated\ndef func(): pass\n")
        reason = is_file_excluded(f, cfg)
        assert reason is not None

    def test_normal_python_file_passes(self, tmp_path: Path) -> None:
        cfg = Config()
        f = tmp_path / "normal.py"
        f.write_text("def hello():\n    return 'world'\n")
        reason = is_file_excluded(f, cfg)
        assert reason is None

    def test_content_parameter_skips_io(self, tmp_path: Path) -> None:
        cfg = Config()
        f = tmp_path / "test.py"
        f.write_text("normal content")
        # Pass content directly - should use it instead of reading file
        content = b"\x00" * 300  # binary content
        reason = is_file_excluded(f, cfg, content=content)
        assert reason is not None
        assert "binary file" in reason

    def test_gitignore_exclusion(self, tmp_path: Path) -> None:
        cfg = Config()
        matcher = MagicMock()
        matcher.is_excluded.return_value = True
        f = tmp_path / "ignored.py"
        f.write_text("code")
        reason = is_file_excluded(f, cfg, gitignore_matcher=matcher)
        assert reason == "excluded by .gitignore"

    def test_compound_extension_min_js(self, tmp_path: Path) -> None:
        cfg = Config()
        f = tmp_path / "bundle.min.js"
        f.write_text("var x=1;")
        reason = is_file_excluded(f, cfg)
        assert reason is not None
        assert ".min.js" in reason


class TestIndexStats:
    def test_defaults(self) -> None:
        stats = IndexStats()
        assert stats.files_processed == 0
        assert stats.chunks_created == 0
        assert stats.duration_seconds == 0.0
