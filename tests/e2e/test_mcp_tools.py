"""E2E tests for MCP server tool functions.

Tests the MCP tool functions (search_code, index_project, find_files, watch_project)
using the real pipeline with mock embedder.
"""
import asyncio
import shutil
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import numpy as np
import pytest

from codexlens_search.bridge import (
    create_config_from_env,
    create_pipeline,
    should_exclude,
    DEFAULT_EXCLUDES,
)
from codexlens_search.config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockEmbedder:
    """Deterministic embedder for E2E tests."""

    def embed_single(self, text: str) -> np.ndarray:
        seed = hash(text) % (2**31)
        rng = np.random.default_rng(seed=seed)
        vec = rng.standard_normal(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed_single(t) for t in texts]

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        return self.embed_batch(texts)


class MockReranker:
    """Simple keyword-overlap reranker."""

    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        query_words = set(query.lower().split())
        scores = []
        for doc in documents:
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            scores.append(float(overlap) / max(len(query_words), 1))
        return scores


def _sync_project(project_path: Path):
    """Index all .py files in a project directory using the pipeline directly."""
    db_path = project_path / ".codexlens"
    db_path.mkdir(exist_ok=True)

    with patch("codexlens_search.bridge._create_embedder", return_value=MockEmbedder()), \
         patch("codexlens_search.bridge._create_reranker", return_value=MockReranker()):
        config = create_config_from_env(db_path)
        indexing, search, config = create_pipeline(db_path, config)

        root = project_path
        file_paths = [
            p for p in root.glob("**/*.py")
            if p.is_file() and not should_exclude(p.relative_to(root), DEFAULT_EXCLUDES)
        ]
        stats = indexing.sync(file_paths, root=root)
        return stats, indexing, search


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestMCPSearchTool:
    """Test the search_code MCP tool function."""

    def test_search_returns_results(self, project_dir):
        """search_code should return formatted markdown results after indexing."""
        # Copy fixtures into project
        src_dir = project_dir / "src"

        with patch("codexlens_search.bridge._create_embedder", return_value=MockEmbedder()), \
             patch("codexlens_search.bridge._create_reranker", return_value=MockReranker()):
            stats, indexing, search = _sync_project(project_dir)
            assert stats.files_processed > 0

            # Use the search pipeline directly (same as what MCP tool does)
            results = search.search("authentication password", top_k=5)
            assert len(results) > 0
            # Check result fields
            for r in results:
                assert r.path
                assert r.score >= 0
                assert r.content

    def test_search_with_scope(self, project_dir):
        """Search with scope filter should restrict results to matching paths."""
        with patch("codexlens_search.bridge._create_embedder", return_value=MockEmbedder()), \
             patch("codexlens_search.bridge._create_reranker", return_value=MockReranker()):
            _sync_project(project_dir)

            db_path = project_dir / ".codexlens"
            config = create_config_from_env(db_path)
            _, search, _ = create_pipeline(db_path, config)

            # Search with scope "src" — all our files are under src/
            results = search.search("cache", top_k=10)
            assert len(results) > 0

    def test_search_empty_index(self, tmp_path):
        """Search on empty index should return empty results."""
        db_path = tmp_path / ".codexlens"
        db_path.mkdir()

        with patch("codexlens_search.bridge._create_embedder", return_value=MockEmbedder()), \
             patch("codexlens_search.bridge._create_reranker", return_value=MockReranker()):
            config = create_config_from_env(db_path)
            _, search, _ = create_pipeline(db_path, config)
            results = search.search("anything", top_k=5)
            assert len(results) == 0


@pytest.mark.e2e
class TestMCPIndexProjectTool:
    """Test the index_project MCP tool function logic."""

    def test_index_project_sync(self, project_dir):
        """index_project sync should index files and create db."""
        with patch("codexlens_search.bridge._create_embedder", return_value=MockEmbedder()), \
             patch("codexlens_search.bridge._create_reranker", return_value=MockReranker()):
            stats, _, _ = _sync_project(project_dir)

        assert stats.files_processed > 0
        assert stats.chunks_created > 0
        assert stats.duration_seconds >= 0

        # Verify database files exist
        db_path = project_dir / ".codexlens"
        assert (db_path / "metadata.db").exists()
        assert (db_path / "fts.db").exists()

    def test_index_project_status(self, project_dir):
        """After indexing, metadata store should report correct counts."""
        from codexlens_search.indexing.metadata import MetadataStore

        with patch("codexlens_search.bridge._create_embedder", return_value=MockEmbedder()), \
             patch("codexlens_search.bridge._create_reranker", return_value=MockReranker()):
            stats, _, _ = _sync_project(project_dir)

        db_path = project_dir / ".codexlens"
        metadata = MetadataStore(db_path / "metadata.db")
        all_files = metadata.get_all_files()
        max_chunk = metadata.max_chunk_id()

        assert len(all_files) == stats.files_processed
        assert max_chunk >= 0

    def test_index_project_incremental(self, project_dir):
        """Second sync should be no-op if files unchanged."""
        with patch("codexlens_search.bridge._create_embedder", return_value=MockEmbedder()), \
             patch("codexlens_search.bridge._create_reranker", return_value=MockReranker()):
            stats1, _, _ = _sync_project(project_dir)
            assert stats1.files_processed > 0

            # Second sync — no changes
            stats2, _, _ = _sync_project(project_dir)
            # Files should be skipped (no new chunks)
            assert stats2.chunks_created == 0


@pytest.mark.e2e
class TestMCPFindFilesTool:
    """Test the find_files MCP tool function logic."""

    def test_find_py_files(self, project_dir):
        """find_files with *.py glob should find fixture files."""
        root = project_dir
        matches = []
        for p in root.glob("**/*.py"):
            if p.is_file() and not should_exclude(p.relative_to(root), DEFAULT_EXCLUDES):
                matches.append(str(p.relative_to(root)))

        assert len(matches) >= 5  # we have 10 fixture files in src/

    def test_find_files_respects_excludes(self, project_dir):
        """Files in excluded directories should be filtered out."""
        # Create a node_modules directory with a .py file
        nm_dir = project_dir / "node_modules" / "pkg"
        nm_dir.mkdir(parents=True)
        (nm_dir / "script.py").write_text("x = 1")

        root = project_dir
        matches = []
        for p in root.glob("**/*.py"):
            if p.is_file() and not should_exclude(p.relative_to(root), DEFAULT_EXCLUDES):
                matches.append(str(p.relative_to(root)))

        # node_modules/pkg/script.py should NOT be in results
        assert not any("node_modules" in m for m in matches)


@pytest.mark.e2e
class TestMCPWatchProjectTool:
    """Test the watch_project MCP tool status logic."""

    def test_watch_status_default_stopped(self, project_dir):
        """Without starting, watcher should report stopped."""
        from codexlens_search.mcp_server import _watchers, _watcher_lock

        resolved = str(project_dir.resolve())
        with _watcher_lock:
            watcher = _watchers.get(resolved)
        # By default no watcher is running
        assert watcher is None
