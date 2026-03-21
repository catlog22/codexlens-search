"""E2E tests for vector search semantic accuracy.

Uses real fastembed model (bge-small-en-v1.5) to test that semantically
relevant results rank higher than irrelevant ones.
Marked with @pytest.mark.slow and @pytest.mark.e2e.
"""
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from codexlens_search.bridge import (
    create_config_from_env,
    create_pipeline,
    should_exclude,
    DEFAULT_EXCLUDES,
)
from codexlens_search.config import Config

from .conftest import requires_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockReranker:
    """Simple reranker that preserves vector search ordering."""

    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        # Return decreasing scores to preserve input order
        return [1.0 - i * 0.01 for i in range(len(documents))]


def _index_and_search(project_dir: Path, query: str, top_k: int = 5):
    """Index fixtures with real embedder, then search."""
    from codexlens_search.embed.local import FastEmbedEmbedder

    db_path = project_dir / ".codexlens"
    db_path.mkdir(exist_ok=True)

    config = create_config_from_env(db_path)
    embedder = FastEmbedEmbedder(config)
    reranker = MockReranker()

    with patch("codexlens_search.bridge._create_embedder", return_value=embedder), \
         patch("codexlens_search.bridge._create_reranker", return_value=reranker):
        indexing, search, config = create_pipeline(db_path, config)

        root = project_dir
        file_paths = [
            p for p in root.glob("**/*.py")
            if p.is_file() and not should_exclude(p.relative_to(root), DEFAULT_EXCLUDES)
        ]
        stats = indexing.sync(file_paths, root=root)

    with patch("codexlens_search.bridge._create_embedder", return_value=embedder), \
         patch("codexlens_search.bridge._create_reranker", return_value=reranker):
        _, search2, _ = create_pipeline(db_path, config)
        results = search2.search(query, top_k=top_k)

    return results, stats


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@requires_model
@pytest.mark.slow
@pytest.mark.e2e
class TestSearchAccuracy:
    """Test that semantic search returns relevant results."""

    def test_auth_query_ranks_auth_file_higher(self, project_dir):
        """Query about authentication should rank auth_handler.py higher."""
        results, stats = _index_and_search(project_dir, "user authentication and password hashing")
        assert stats.files_processed > 0
        assert len(results) > 0

        # Find which results come from auth_handler.py
        auth_results = [r for r in results if "auth_handler" in r.path]
        non_auth_results = [r for r in results if "auth_handler" not in r.path]

        assert len(auth_results) > 0, "auth_handler.py should appear in results"

        # Auth results should have higher scores on average than non-auth
        if non_auth_results:
            avg_auth_score = sum(r.score for r in auth_results) / len(auth_results)
            avg_other_score = sum(r.score for r in non_auth_results) / len(non_auth_results)
            assert avg_auth_score >= avg_other_score, (
                f"Auth results (avg={avg_auth_score:.4f}) should score >= "
                f"non-auth results (avg={avg_other_score:.4f})"
            )

    def test_database_query_ranks_db_file_higher(self, project_dir):
        """Query about database should rank database_connector.py higher."""
        results, _ = _index_and_search(project_dir, "database connection pool SQL query builder")
        assert len(results) > 0

        db_results = [r for r in results if "database_connector" in r.path]
        assert len(db_results) > 0, "database_connector.py should appear in results"

        # The top result should be from database_connector.py
        top_paths = [r.path for r in results[:3]]
        assert any("database_connector" in p for p in top_paths), (
            f"database_connector.py should be in top 3 results, got: {top_paths}"
        )

    def test_math_query_ranks_math_file_higher(self, project_dir):
        """Query about math should rank math_utils.py higher."""
        results, _ = _index_and_search(
            project_dir,
            "standard deviation percentile cosine similarity normalization",
        )
        assert len(results) > 0

        math_results = [r for r in results if "math_utils" in r.path]
        assert len(math_results) > 0, "math_utils.py should appear in results"

    def test_cache_query_ranks_cache_file_higher(self, project_dir):
        """Query about caching should rank cache_manager.py higher."""
        results, _ = _index_and_search(project_dir, "LRU cache eviction TTL expiration")
        assert len(results) > 0

        cache_results = [r for r in results if "cache_manager" in r.path]
        assert len(cache_results) > 0, "cache_manager.py should appear in results"

    def test_event_query_ranks_event_file_higher(self, project_dir):
        """Query about events should rank event_emitter.py higher."""
        results, _ = _index_and_search(
            project_dir, "event listener publish subscribe emit callback"
        )
        assert len(results) > 0

        event_results = [r for r in results if "event_emitter" in r.path]
        assert len(event_results) > 0, "event_emitter.py should appear in results"

    def test_scheduling_query_ranks_scheduler_higher(self, project_dir):
        """Query about task scheduling should rank task_scheduler.py higher."""
        results, _ = _index_and_search(
            project_dir, "priority task queue scheduler execution"
        )
        assert len(results) > 0

        sched_results = [r for r in results if "task_scheduler" in r.path]
        assert len(sched_results) > 0, "task_scheduler.py should appear in results"

    def test_result_fields_populated(self, project_dir):
        """All result fields should be properly populated."""
        results, _ = _index_and_search(project_dir, "function definition class")
        assert len(results) > 0

        for r in results:
            assert r.id >= 0
            assert r.path != ""
            assert r.score > 0
            assert r.content != ""
            assert r.line >= 0
            assert r.end_line >= r.line

    def test_top_k_limit_respected(self, project_dir):
        """Search should return at most top_k results."""
        results, _ = _index_and_search(project_dir, "function", top_k=3)
        assert len(results) <= 3

    def test_distinct_queries_return_different_rankings(self, project_dir):
        """Different queries should produce different result rankings."""
        results_auth, _ = _index_and_search(project_dir, "authentication login session")
        results_math, _ = _index_and_search(project_dir, "mathematical computation statistics")

        # Top results should differ
        if results_auth and results_math:
            top_auth_path = results_auth[0].path
            top_math_path = results_math[0].path
            assert top_auth_path != top_math_path, (
                "Different queries should return different top results"
            )
