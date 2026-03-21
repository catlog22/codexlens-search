"""L2 integration tests for indexing/gitignore.py.

Tests GitignoreAwareMatcher discovery, caching, invalidation, and exclusion
through the indexing pipeline with real filesystem and .gitignore files.

Targets: indexing/gitignore.py coverage from 20% toward 60%+.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

try:
    import pathspec  # noqa: F401
    _HAS_PATHSPEC = True
except ImportError:
    _HAS_PATHSPEC = False

pytestmark = pytest.mark.skipif(
    not _HAS_PATHSPEC, reason="pathspec not installed"
)


class TestGitignoreAwareMatcherDiscovery:
    """Test .gitignore file discovery and pattern parsing."""

    def test_discovers_root_gitignore(self, tmp_path):
        from codexlens_search.indexing.gitignore import GitignoreAwareMatcher

        (tmp_path / ".gitignore").write_text("*.log\nbuild/\n", encoding="utf-8")
        (tmp_path / "app.py").write_text("pass", encoding="utf-8")
        (tmp_path / "debug.log").write_text("log data", encoding="utf-8")

        matcher = GitignoreAwareMatcher(tmp_path)
        assert matcher.is_excluded(tmp_path / "debug.log") is True
        assert matcher.is_excluded(tmp_path / "app.py") is False

    def test_discovers_nested_gitignore(self, tmp_path):
        from codexlens_search.indexing.gitignore import GitignoreAwareMatcher

        (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / ".gitignore").write_text("*.tmp\n", encoding="utf-8")
        (sub / "data.tmp").write_text("temp", encoding="utf-8")
        (sub / "code.py").write_text("pass", encoding="utf-8")

        matcher = GitignoreAwareMatcher(tmp_path)
        # Nested gitignore should exclude .tmp files in sub/
        assert matcher.is_excluded(sub / "data.tmp") is True
        assert matcher.is_excluded(sub / "code.py") is False
        # Root gitignore should still work
        assert matcher.is_excluded(tmp_path / "error.log") is True

    def test_directory_pattern_excludes_contents(self, tmp_path):
        from codexlens_search.indexing.gitignore import GitignoreAwareMatcher

        (tmp_path / ".gitignore").write_text("dist/\n", encoding="utf-8")
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "bundle.js").write_text("var x=1;", encoding="utf-8")

        matcher = GitignoreAwareMatcher(tmp_path)
        assert matcher.is_excluded(dist / "bundle.js") is True


class TestGitignoreAwareMatcherCaching:
    """Test mtime-based cache invalidation."""

    def test_cache_invalidation_on_gitignore_edit(self, tmp_path):
        from codexlens_search.indexing.gitignore import GitignoreAwareMatcher
        import time

        gi_path = tmp_path / ".gitignore"
        gi_path.write_text("*.log\n", encoding="utf-8")
        (tmp_path / "data.csv").write_text("1,2,3", encoding="utf-8")

        matcher = GitignoreAwareMatcher(tmp_path)
        assert matcher.is_excluded(tmp_path / "data.csv") is False

        # Edit .gitignore to also exclude .csv
        time.sleep(0.05)  # ensure different mtime
        gi_path.write_text("*.log\n*.csv\n", encoding="utf-8")

        # Should pick up the new pattern
        assert matcher.is_excluded(tmp_path / "data.csv") is True

    def test_no_gitignore_directory_cached(self, tmp_path):
        from codexlens_search.indexing.gitignore import GitignoreAwareMatcher

        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "file.py").write_text("pass", encoding="utf-8")

        matcher = GitignoreAwareMatcher(tmp_path)
        # First check: no gitignore anywhere
        assert matcher.is_excluded(sub / "file.py") is False
        # Second check should use cached "no gitignore" knowledge
        assert matcher.is_excluded(sub / "file.py") is False
        assert tmp_path in matcher._no_gitignore or sub in matcher._no_gitignore

    def test_gitignore_deleted_clears_cache(self, tmp_path):
        from codexlens_search.indexing.gitignore import GitignoreAwareMatcher

        gi_path = tmp_path / ".gitignore"
        gi_path.write_text("*.log\n", encoding="utf-8")
        (tmp_path / "test.log").write_text("log", encoding="utf-8")

        matcher = GitignoreAwareMatcher(tmp_path)
        assert matcher.is_excluded(tmp_path / "test.log") is True

        # Delete .gitignore
        gi_path.unlink()
        # Internally _get_spec should handle missing file
        # Cache should be cleared on next is_excluded call
        assert matcher.is_excluded(tmp_path / "test.log") is False


class TestGitignoreAwareMatcherEdgeCases:
    """Test edge cases for path matching."""

    def test_path_outside_root_not_excluded(self, tmp_path):
        from codexlens_search.indexing.gitignore import GitignoreAwareMatcher

        root = tmp_path / "project"
        root.mkdir()
        (root / ".gitignore").write_text("*\n", encoding="utf-8")

        matcher = GitignoreAwareMatcher(root)
        outside = tmp_path / "other" / "file.py"
        assert matcher.is_excluded(outside) is False

    def test_negation_pattern(self, tmp_path):
        from codexlens_search.indexing.gitignore import GitignoreAwareMatcher

        (tmp_path / ".gitignore").write_text("*.pyc\n!important.pyc\n", encoding="utf-8")
        (tmp_path / "test.pyc").write_text("bytecode", encoding="utf-8")
        (tmp_path / "important.pyc").write_text("bytecode", encoding="utf-8")

        matcher = GitignoreAwareMatcher(tmp_path)
        assert matcher.is_excluded(tmp_path / "test.pyc") is True
        assert matcher.is_excluded(tmp_path / "important.pyc") is False

    def test_wildcard_patterns(self, tmp_path):
        from codexlens_search.indexing.gitignore import GitignoreAwareMatcher

        (tmp_path / ".gitignore").write_text("**/__pycache__/\n*.egg-info/\n", encoding="utf-8")

        pycache = tmp_path / "src" / "__pycache__"
        pycache.mkdir(parents=True)
        (pycache / "module.cpython-310.pyc").write_text("bc", encoding="utf-8")

        matcher = GitignoreAwareMatcher(tmp_path)
        assert matcher.is_excluded(pycache / "module.cpython-310.pyc") is True

    def test_empty_gitignore(self, tmp_path):
        from codexlens_search.indexing.gitignore import GitignoreAwareMatcher

        (tmp_path / ".gitignore").write_text("", encoding="utf-8")
        (tmp_path / "file.py").write_text("pass", encoding="utf-8")

        matcher = GitignoreAwareMatcher(tmp_path)
        assert matcher.is_excluded(tmp_path / "file.py") is False

    def test_comment_lines_ignored(self, tmp_path):
        from codexlens_search.indexing.gitignore import GitignoreAwareMatcher

        (tmp_path / ".gitignore").write_text(
            "# This is a comment\n*.log\n# Another comment\n", encoding="utf-8"
        )

        matcher = GitignoreAwareMatcher(tmp_path)
        assert matcher.is_excluded(tmp_path / "app.log") is True
        assert matcher.is_excluded(tmp_path / "app.py") is False


class TestGitignoreWithIndexingPipeline:
    """Test gitignore filtering through the full IndexingPipeline."""

    def test_gitignore_filtering_skips_excluded_files(self, tmp_path):
        from codexlens_search.config import Config
        from codexlens_search.core import ANNIndex, BinaryStore
        from codexlens_search.indexing.metadata import MetadataStore
        from codexlens_search.indexing.pipeline import IndexingPipeline
        from codexlens_search.search.fts import FTSEngine
        from tests.integration.conftest import DIM, MockEmbedder

        root = tmp_path / "project"
        root.mkdir()
        (root / ".gitignore").write_text("*.log\nbuild/\n", encoding="utf-8")
        (root / "app.py").write_text("def main(): pass\n", encoding="utf-8")
        (root / "debug.log").write_text("log content\n", encoding="utf-8")
        build = root / "build"
        build.mkdir()
        (build / "output.js").write_text("compiled\n", encoding="utf-8")

        config = Config.small()
        config.embed_dim = DIM
        config.gitignore_filtering = True

        db = tmp_path / "db"
        db.mkdir()
        indexing = IndexingPipeline(
            embedder=MockEmbedder(),
            binary_store=BinaryStore(db / "binary", dim=DIM, config=config),
            ann_index=ANNIndex(db / "ann.hnsw", dim=DIM, config=config),
            fts=FTSEngine(db / "fts.db"),
            config=config,
            metadata=MetadataStore(db / "metadata.db"),
        )

        # Sync all files under root
        all_files = [
            root / "app.py",
            root / "debug.log",
            build / "output.js",
        ]
        stats = indexing.sync(all_files, root=root)

        # Only app.py should be processed (debug.log and build/* excluded)
        assert stats.files_processed == 1

    def test_gitignore_filtering_disabled_includes_all(self, tmp_path):
        from codexlens_search.config import Config
        from codexlens_search.core import ANNIndex, BinaryStore
        from codexlens_search.indexing.metadata import MetadataStore
        from codexlens_search.indexing.pipeline import IndexingPipeline
        from codexlens_search.search.fts import FTSEngine
        from tests.integration.conftest import DIM, MockEmbedder

        root = tmp_path / "project"
        root.mkdir()
        (root / ".gitignore").write_text("*.txt\n", encoding="utf-8")
        (root / "code.py").write_text("def foo(): pass\n", encoding="utf-8")
        (root / "notes.txt").write_text("some notes here about code\n", encoding="utf-8")

        config = Config.small()
        config.embed_dim = DIM
        config.gitignore_filtering = False  # disabled

        db = tmp_path / "db"
        db.mkdir()
        indexing = IndexingPipeline(
            embedder=MockEmbedder(),
            binary_store=BinaryStore(db / "binary", dim=DIM, config=config),
            ann_index=ANNIndex(db / "ann.hnsw", dim=DIM, config=config),
            fts=FTSEngine(db / "fts.db"),
            config=config,
            metadata=MetadataStore(db / "metadata.db"),
        )

        all_files = [root / "code.py", root / "notes.txt"]
        stats = indexing.sync(all_files, root=root)
        # Both files should be processed since gitignore filtering is off
        assert stats.files_processed == 2


class TestGitignoreImportError:
    """Test behavior when pathspec is not available."""

    def test_raises_import_error_without_pathspec(self, tmp_path):
        import codexlens_search.indexing.gitignore as gi_mod

        original = gi_mod._HAS_PATHSPEC
        try:
            gi_mod._HAS_PATHSPEC = False
            with pytest.raises(ImportError, match="pathspec"):
                gi_mod.GitignoreAwareMatcher(tmp_path)
        finally:
            gi_mod._HAS_PATHSPEC = original
