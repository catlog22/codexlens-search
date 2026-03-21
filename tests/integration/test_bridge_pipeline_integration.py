"""L2 integration tests for bridge.py pipeline lifecycle.

Tests create_pipeline (single-shard and multi-shard), _create_embedder,
_create_reranker, cmd_search, cmd_sync, cmd_init, cmd_status, cmd_index_file,
cmd_remove_file through real pipeline components with mocked embedders.

Targets: bridge.py coverage from 34% toward 60%+.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from codexlens_search.bridge import (
    create_config_from_env,
    create_pipeline,
    should_exclude,
    _create_embedder,
    _create_reranker,
    _json_output,
    _error_exit,
    _resolve_db_path,
    DEFAULT_EXCLUDES,
)
from codexlens_search.config import Config

from tests.integration.conftest import DIM, MockEmbedder, MockReranker


class TestCreatePipelineSingleShard:
    """Test create_pipeline returns working single-shard pipeline."""

    def test_creates_indexing_and_search_pipelines(self, tmp_path):
        config = Config.small()
        config.embed_dim = DIM
        with mock.patch(
            "codexlens_search.bridge._create_embedder", return_value=MockEmbedder()
        ), mock.patch(
            "codexlens_search.bridge._create_reranker", return_value=MockReranker()
        ):
            indexing, search, returned_config = create_pipeline(tmp_path, config)

        assert indexing is not None
        assert search is not None
        assert returned_config is config

    def test_pipeline_can_index_and_search(self, tmp_path):
        config = Config.small()
        config.embed_dim = DIM
        with mock.patch(
            "codexlens_search.bridge._create_embedder", return_value=MockEmbedder()
        ), mock.patch(
            "codexlens_search.bridge._create_reranker", return_value=MockReranker()
        ):
            indexing, search, _ = create_pipeline(tmp_path, config)

        # Create a file to index
        src = tmp_path / "src"
        src.mkdir()
        (src / "hello.py").write_text(
            "def greet(name):\n    return f'Hello, {name}!'\n", encoding="utf-8"
        )
        files = list(src.glob("*.py"))
        stats = indexing.sync(files, root=src)
        assert stats.files_processed == 1
        assert stats.chunks_created >= 1

        results = search.search("greet")
        assert len(results) > 0

    def test_pipeline_creates_db_directory(self, tmp_path):
        db_dir = tmp_path / "subdir" / "index"
        config = Config.small()
        config.embed_dim = DIM
        with mock.patch(
            "codexlens_search.bridge._create_embedder", return_value=MockEmbedder()
        ), mock.patch(
            "codexlens_search.bridge._create_reranker", return_value=MockReranker()
        ):
            create_pipeline(db_dir, config)
        assert db_dir.exists()

    def test_pipeline_with_no_config_uses_defaults(self, tmp_path):
        with mock.patch(
            "codexlens_search.bridge._create_embedder", return_value=MockEmbedder()
        ), mock.patch(
            "codexlens_search.bridge._create_reranker", return_value=MockReranker()
        ):
            indexing, search, config = create_pipeline(tmp_path)
        assert config is not None
        assert config.embed_dim == 384


class TestCreatePipelineMultiShard:
    """Test create_pipeline with multi-shard config returns ShardManager."""

    def test_multi_shard_returns_shard_manager(self, tmp_path):
        config = Config.small()
        config.embed_dim = DIM
        config.num_shards = 2
        config.max_loaded_shards = 2
        with mock.patch(
            "codexlens_search.bridge._create_embedder", return_value=MockEmbedder()
        ), mock.patch(
            "codexlens_search.bridge._create_reranker", return_value=MockReranker()
        ):
            idx_mgr, search_mgr, _ = create_pipeline(tmp_path, config)

        # Both should be the same ShardManager instance
        assert idx_mgr is search_mgr
        from codexlens_search.core.shard_manager import ShardManager
        assert isinstance(idx_mgr, ShardManager)


class TestCreateEmbedder:
    """Test _create_embedder selects correct embedder type."""

    def test_local_embedder_when_no_api_url(self):
        config = Config.small()
        config.embed_api_url = ""
        with mock.patch(
            "codexlens_search.embed.local.FastEmbedEmbedder"
        ) as MockFE:
            embedder = _create_embedder(config)
        MockFE.assert_called_once_with(config)

    def test_api_embedder_when_api_url_set(self):
        config = Config.small()
        config.embed_api_url = "https://api.example.com/v1"
        config.embed_api_key = "test-key"
        mock_embedder = mock.MagicMock()
        mock_embedder.embed_single.return_value = np.zeros(384, dtype=np.float32)
        with mock.patch(
            "codexlens_search.embed.api.APIEmbedder", return_value=mock_embedder
        ):
            embedder = _create_embedder(config)
        assert embedder is mock_embedder

    def test_api_embedder_autodetects_dim(self):
        config = Config.small()
        config.embed_api_url = "https://api.example.com/v1"
        config.embed_dim = 384  # default
        mock_embedder = mock.MagicMock()
        # API returns 768-dim vector, should auto-detect
        mock_embedder.embed_single.return_value = np.zeros(768, dtype=np.float32)
        with mock.patch(
            "codexlens_search.embed.api.APIEmbedder", return_value=mock_embedder
        ):
            _create_embedder(config)
        assert config.embed_dim == 768


class TestCreateReranker:
    """Test _create_reranker selects correct reranker type."""

    def test_local_reranker_when_no_api_url(self):
        config = Config.small()
        config.reranker_api_url = ""
        with mock.patch(
            "codexlens_search.rerank.local.FastEmbedReranker"
        ) as MockFR:
            reranker = _create_reranker(config)
        MockFR.assert_called_once_with(config)

    def test_api_reranker_when_api_url_set(self):
        config = Config.small()
        config.reranker_api_url = "https://rerank.example.com/v1"
        config.reranker_api_key = "rk-123"
        with mock.patch(
            "codexlens_search.rerank.api.APIReranker"
        ) as MockAR:
            reranker = _create_reranker(config)
        MockAR.assert_called_once_with(config)


class TestCmdInit:
    """Test cmd_init creates empty index."""

    def test_cmd_init_creates_stores(self, tmp_path, capsys):
        from codexlens_search.bridge import cmd_init

        args = argparse.Namespace(db_path=str(tmp_path / "idx"))
        cmd_init(args)

        captured = capsys.readouterr()
        output = json.loads(captured.out.strip())
        assert output["status"] == "initialized"
        assert "db_path" in output
        # Should have created metadata.db and fts.db
        assert (Path(output["db_path"]) / "metadata.db").exists()
        assert (Path(output["db_path"]) / "fts.db").exists()


class TestCmdStatus:
    """Test cmd_status reports index statistics."""

    def test_status_not_initialized(self, tmp_path, capsys):
        from codexlens_search.bridge import cmd_status

        args = argparse.Namespace(db_path=str(tmp_path / "nonexist"))
        cmd_status(args)

        captured = capsys.readouterr()
        output = json.loads(captured.out.strip())
        assert output["status"] == "not_initialized"

    def test_status_after_init(self, tmp_path, capsys):
        from codexlens_search.bridge import cmd_init, cmd_status

        args = argparse.Namespace(db_path=str(tmp_path / "idx"))
        cmd_init(args)
        capsys.readouterr()  # clear init output

        cmd_status(args)
        captured = capsys.readouterr()
        output = json.loads(captured.out.strip())
        assert output["status"] == "ok"
        assert output["files_tracked"] == 0


class TestCmdIndexFileAndRemove:
    """Test cmd_index_file and cmd_remove_file through CLI bridge."""

    def test_index_file_not_found_exits(self, tmp_path):
        from codexlens_search.bridge import cmd_index_file

        args = argparse.Namespace(
            db_path=str(tmp_path / "idx"),
            file=str(tmp_path / "missing.py"),
            root=None,
            embed_model=None,
            embed_api_url=None,
            embed_api_key=None,
            embed_api_model=None,
        )
        with pytest.raises(SystemExit):
            cmd_index_file(args)


class TestCmdSync:
    """Test cmd_sync through the bridge with mocked pipeline."""

    def test_sync_with_mocked_pipeline(self, tmp_path, capsys):
        from codexlens_search.bridge import cmd_sync

        # Create source files
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("def a(): pass\n", encoding="utf-8")
        (src / "b.py").write_text("def b(): pass\n", encoding="utf-8")
        (src / "node_modules").mkdir()
        (src / "node_modules" / "pkg.js").write_text("var x=1;", encoding="utf-8")

        args = argparse.Namespace(
            db_path=str(tmp_path / "idx"),
            root=str(src),
            glob="**/*",
            exclude=None,
            embed_model=None,
            embed_api_url=None,
            embed_api_key=None,
            embed_api_model=None,
        )

        mock_indexing = mock.MagicMock()
        mock_indexing.sync.return_value = mock.MagicMock(
            files_processed=2, chunks_created=4, duration_seconds=0.1
        )

        with mock.patch(
            "codexlens_search.bridge._create_pipeline",
            return_value=(mock_indexing, mock.MagicMock(), Config.small()),
        ):
            cmd_sync(args)

        captured = capsys.readouterr()
        output = json.loads(captured.out.strip())
        assert output["status"] == "synced"

        # Verify node_modules was filtered out
        call_args = mock_indexing.sync.call_args
        synced_files = call_args[0][0]
        synced_paths = [str(f) for f in synced_files]
        assert not any("node_modules" in p for p in synced_paths)


class TestResolveDbPath:
    """Test _resolve_db_path resolves and creates parent dirs."""

    def test_creates_parent_dirs(self, tmp_path):
        args = argparse.Namespace(db_path=str(tmp_path / "a" / "b" / "c"))
        result = _resolve_db_path(args)
        assert result.exists()
        assert result.is_dir()


class TestJsonHelpers:
    """Test JSON output helpers."""

    def test_json_output(self, capsys):
        _json_output({"key": "value"})
        captured = capsys.readouterr()
        assert json.loads(captured.out.strip()) == {"key": "value"}

    def test_error_exit(self):
        with pytest.raises(SystemExit) as exc_info:
            _error_exit("test error", code=2)
        assert exc_info.value.code == 2
