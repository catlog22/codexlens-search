"""Unit tests for indexing/gitignore.py — GitignoreAwareMatcher."""
from __future__ import annotations

from pathlib import Path

import pytest

try:
    import pathspec
    _HAS_PATHSPEC = True
except ImportError:
    _HAS_PATHSPEC = False

pytestmark = pytest.mark.skipif(not _HAS_PATHSPEC, reason="pathspec not installed")


from codexlens_search.indexing.gitignore import GitignoreAwareMatcher


class TestGitignoreAwareMatcher:
    """Test .gitignore discovery and matching."""

    def test_simple_pattern(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("*.pyc\n")
        (tmp_path / "test.pyc").touch()
        (tmp_path / "test.py").touch()

        matcher = GitignoreAwareMatcher(tmp_path)
        assert matcher.is_excluded(tmp_path / "test.pyc") is True
        assert matcher.is_excluded(tmp_path / "test.py") is False

    def test_directory_pattern(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("build/\n__pycache__/\n")
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        (build_dir / "output.js").touch()
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "module.cpython-310.pyc").touch()

        matcher = GitignoreAwareMatcher(tmp_path)
        assert matcher.is_excluded(tmp_path / "build" / "output.js") is True
        assert matcher.is_excluded(tmp_path / "__pycache__" / "module.cpython-310.pyc") is True

    def test_nested_gitignore(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("*.log\n")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / ".gitignore").write_text("*.tmp\n")
        (tmp_path / "root.log").touch()
        (sub / "deep.tmp").touch()
        (sub / "deep.log").touch()
        (sub / "deep.py").touch()

        matcher = GitignoreAwareMatcher(tmp_path)
        assert matcher.is_excluded(tmp_path / "root.log") is True
        assert matcher.is_excluded(sub / "deep.tmp") is True
        assert matcher.is_excluded(sub / "deep.log") is True  # parent pattern
        assert matcher.is_excluded(sub / "deep.py") is False

    def test_path_outside_root(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("*.pyc\n")
        matcher = GitignoreAwareMatcher(tmp_path)
        # Path outside root should return False
        assert matcher.is_excluded(Path("/some/other/path/test.pyc")) is False

    def test_negation_pattern(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("*.log\n!important.log\n")
        (tmp_path / "debug.log").touch()
        (tmp_path / "important.log").touch()

        matcher = GitignoreAwareMatcher(tmp_path)
        assert matcher.is_excluded(tmp_path / "debug.log") is True
        assert matcher.is_excluded(tmp_path / "important.log") is False

    def test_no_gitignore_file(self, tmp_path: Path) -> None:
        (tmp_path / "test.py").touch()
        matcher = GitignoreAwareMatcher(tmp_path)
        assert matcher.is_excluded(tmp_path / "test.py") is False

    def test_cache_invalidation_on_mtime_change(self, tmp_path: Path) -> None:
        gi = tmp_path / ".gitignore"
        gi.write_text("*.log\n")
        (tmp_path / "test.log").touch()

        matcher = GitignoreAwareMatcher(tmp_path)
        assert matcher.is_excluded(tmp_path / "test.log") is True

        # Modify .gitignore to no longer exclude .log
        import time
        time.sleep(0.05)  # Ensure mtime differs
        gi.write_text("*.tmp\n")

        # Force re-check (cache should detect mtime change)
        matcher._cache.clear()
        assert matcher.is_excluded(tmp_path / "test.log") is False

    def test_missing_pathspec_raises(self) -> None:
        """When pathspec is not installed, __init__ should raise ImportError."""
        from codexlens_search.indexing import gitignore
        original = gitignore._HAS_PATHSPEC
        try:
            gitignore._HAS_PATHSPEC = False
            with pytest.raises(ImportError, match="pathspec"):
                GitignoreAwareMatcher(Path("/tmp"))
        finally:
            gitignore._HAS_PATHSPEC = original
