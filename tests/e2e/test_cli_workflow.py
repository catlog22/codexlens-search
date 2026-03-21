"""E2E tests for CLI bridge workflows.

Tests the full pipeline from CLI entry points through indexing and search,
using real SQLite and mock embedder (real fastembed only for accuracy test).
"""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from codexlens_search.bridge import (
    cmd_init,
    cmd_index_file,
    cmd_remove_file,
    cmd_search,
    cmd_status,
    cmd_sync,
    create_config_from_env,
    create_pipeline,
    should_exclude,
)
from codexlens_search.config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockEmbedder:
    """Deterministic embedder for fast E2E tests."""

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
    """Simple keyword-overlap reranker for E2E tests."""

    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        query_words = set(query.lower().split())
        scores = []
        for doc in documents:
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            scores.append(float(overlap) / max(len(query_words), 1))
        return scores


def _make_args(**kwargs) -> argparse.Namespace:
    """Build argparse.Namespace with defaults for CLI commands."""
    defaults = {
        "db_path": "",
        "verbose": False,
        "embed_api_url": "",
        "embed_api_key": "",
        "embed_api_model": "",
        "embed_model": "",
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _capture_json_output(func, args):
    """Capture JSON output from a CLI command that prints to stdout."""
    captured = []
    original_print = print

    def mock_print(*a, **kw):
        captured.append(a[0] if a else "")

    with patch("builtins.print", mock_print):
        try:
            func(args)
        except SystemExit:
            pass

    results = []
    for line in captured:
        if isinstance(line, str):
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestCliInitSyncSearch:
    """Test CLI init -> sync -> search workflow."""

    def test_init_creates_databases(self, db_path):
        """cmd_init should create metadata.db and fts.db."""
        args = _make_args(db_path=str(db_path))
        output = _capture_json_output(cmd_init, args)

        assert len(output) == 1
        assert output[0]["status"] == "initialized"
        assert (db_path / "metadata.db").exists()
        assert (db_path / "fts.db").exists()

    def test_sync_indexes_fixture_files(self, project_dir, db_path):
        """cmd_sync should index all fixture files and report stats."""
        src_dir = project_dir / "src"

        # Patch embedder and reranker to avoid real model download
        with patch("codexlens_search.bridge._create_embedder", return_value=MockEmbedder()), \
             patch("codexlens_search.bridge._create_reranker", return_value=MockReranker()):
            args = _make_args(
                db_path=str(db_path),
                root=str(src_dir),
                glob="**/*.py",
                exclude=None,
            )
            output = _capture_json_output(cmd_sync, args)

        assert len(output) == 1
        result = output[0]
        assert result["status"] == "synced"
        assert result["files_processed"] >= 5  # at least some fixture files indexed
        assert result["chunks_created"] > 0

    def test_full_init_sync_search_workflow(self, project_dir, db_path):
        """Full workflow: init -> sync -> search -> verify results."""
        src_dir = project_dir / "src"

        with patch("codexlens_search.bridge._create_embedder", return_value=MockEmbedder()), \
             patch("codexlens_search.bridge._create_reranker", return_value=MockReranker()):
            # Step 1: init
            init_out = _capture_json_output(cmd_init, _make_args(db_path=str(db_path)))
            assert init_out[0]["status"] == "initialized"

            # Step 2: sync
            sync_args = _make_args(
                db_path=str(db_path),
                root=str(src_dir),
                glob="**/*.py",
                exclude=None,
            )
            sync_out = _capture_json_output(cmd_sync, sync_args)
            assert sync_out[0]["status"] == "synced"
            assert sync_out[0]["files_processed"] > 0

            # Step 3: search
            search_args = _make_args(
                db_path=str(db_path),
                query="authentication password hash",
                top_k=5,
            )
            search_out = _capture_json_output(cmd_search, search_args)

            assert len(search_out) == 1
            results = search_out[0]
            assert isinstance(results, list)
            assert len(results) > 0
            # Results should have expected fields
            for r in results:
                assert "path" in r
                assert "score" in r
                assert "content" in r


@pytest.mark.e2e
class TestCliStatus:
    """Test CLI status command after indexing."""

    def test_status_not_initialized(self, tmp_path):
        """cmd_status on empty db_path reports not_initialized."""
        db_path = tmp_path / ".codexlens"
        db_path.mkdir()
        args = _make_args(db_path=str(db_path))
        output = _capture_json_output(cmd_status, args)

        assert len(output) == 1
        assert output[0]["status"] == "not_initialized"

    def test_status_after_sync(self, project_dir, db_path):
        """cmd_status after sync reports file and chunk counts."""
        src_dir = project_dir / "src"

        with patch("codexlens_search.bridge._create_embedder", return_value=MockEmbedder()), \
             patch("codexlens_search.bridge._create_reranker", return_value=MockReranker()):
            # Sync first
            sync_args = _make_args(
                db_path=str(db_path),
                root=str(src_dir),
                glob="**/*.py",
                exclude=None,
            )
            _capture_json_output(cmd_sync, sync_args)

            # Check status
            status_args = _make_args(db_path=str(db_path))
            output = _capture_json_output(cmd_status, status_args)

        assert len(output) == 1
        result = output[0]
        assert result["status"] == "ok"
        assert result["files_tracked"] > 0
        assert result["total_chunks_approx"] > 0


@pytest.mark.e2e
class TestCliIndexRemove:
    """Test CLI index-file and remove-file commands."""

    def test_index_single_file_then_remove(self, project_dir, db_path):
        """Index a single file, search it, remove it, search returns empty."""
        src_dir = project_dir / "src"
        target_file = src_dir / "auth_handler.py"

        with patch("codexlens_search.bridge._create_embedder", return_value=MockEmbedder()), \
             patch("codexlens_search.bridge._create_reranker", return_value=MockReranker()):
            # Init
            _capture_json_output(cmd_init, _make_args(db_path=str(db_path)))

            # Index single file
            index_args = _make_args(
                db_path=str(db_path),
                file=str(target_file),
                root=str(src_dir),
            )
            index_out = _capture_json_output(cmd_index_file, index_args)
            assert len(index_out) == 1
            assert index_out[0]["status"] == "indexed"
            assert index_out[0]["chunks_created"] > 0

            # Search should find content
            search_args = _make_args(
                db_path=str(db_path),
                query="authentication session token",
                top_k=5,
            )
            search_out = _capture_json_output(cmd_search, search_args)
            assert len(search_out) == 1
            assert len(search_out[0]) > 0

            # Remove the file
            rel_path = str(target_file.relative_to(src_dir))
            remove_args = _make_args(
                db_path=str(db_path),
                file=rel_path,
            )
            remove_out = _capture_json_output(cmd_remove_file, remove_args)
            assert len(remove_out) == 1
            assert remove_out[0]["status"] == "removed"

            # Search after removal: results should be empty (tombstoned)
            search_out2 = _capture_json_output(cmd_search, search_args)
            assert len(search_out2) == 1
            # All results should be filtered out (tombstoned)
            assert len(search_out2[0]) == 0


@pytest.mark.e2e
class TestCreateConfigFromEnv:
    """Test create_config_from_env with environment variables."""

    def test_default_config(self, tmp_path):
        """Default config without env vars."""
        config = create_config_from_env(tmp_path)
        assert config.embed_model == "BAAI/bge-small-en-v1.5"
        assert config.embed_dim == 384

    def test_env_var_override(self, tmp_path, monkeypatch):
        """Environment variables should override defaults."""
        monkeypatch.setenv("CODEXLENS_EMBED_DIM", "768")
        monkeypatch.setenv("CODEXLENS_EMBED_BATCH_SIZE", "64")
        config = create_config_from_env(tmp_path)
        assert config.embed_dim == 768
        assert config.embed_batch_size == 64

    def test_explicit_overrides_take_priority(self, tmp_path, monkeypatch):
        """Explicit kwargs override env vars."""
        monkeypatch.setenv("CODEXLENS_EMBED_API_URL", "https://env-url.example.com")
        config = create_config_from_env(
            tmp_path, embed_api_url="https://override.example.com"
        )
        assert config.embed_api_url == "https://override.example.com"


@pytest.mark.e2e
class TestShouldExclude:
    """Test the should_exclude helper."""

    def test_excludes_node_modules(self):
        assert should_exclude(Path("node_modules/pkg/index.js"), frozenset({"node_modules"}))

    def test_excludes_git(self):
        assert should_exclude(Path(".git/objects/abc"), frozenset({".git"}))

    def test_allows_normal_path(self):
        assert not should_exclude(Path("src/main.py"), frozenset({"node_modules", ".git"}))

    def test_nested_exclude(self):
        assert should_exclude(
            Path("foo/node_modules/bar/baz.js"),
            frozenset({"node_modules"}),
        )
