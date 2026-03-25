"""Unit tests for bridge.py CLI — argparse parsing, JSON protocol, error handling."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codexlens_search.bridge import (
    DEFAULT_EXCLUDES,
    _build_parser,
    _json_output,
    _error_exit,
    should_exclude,
)


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

class TestParser:
    @pytest.fixture(autouse=True)
    def _parser(self):
        self.parser = _build_parser()

    def test_all_subcommands_exist(self):
        expected = {
            "init", "search", "search-files", "locate", "index-file", "remove-file",
            "sync", "watch", "download-models", "traverse", "status",
        }
        # parse each subcommand with minimal required args to verify it exists
        for cmd in expected:
            if cmd == "search":
                args = self.parser.parse_args(["search", "--query", "test"])
            elif cmd == "search-files":
                args = self.parser.parse_args(["search-files", "--query", "test"])
            elif cmd == "locate":
                args = self.parser.parse_args(["locate", "--query", "test"])
            elif cmd == "traverse":
                args = self.parser.parse_args(["traverse", "MySymbol"])
            elif cmd == "index-file":
                args = self.parser.parse_args(["index-file", "--file", "x.py"])
            elif cmd == "remove-file":
                args = self.parser.parse_args(["remove-file", "--file", "x.py"])
            elif cmd == "sync":
                args = self.parser.parse_args(["sync", "--root", "/tmp"])
            elif cmd == "watch":
                args = self.parser.parse_args(["watch", "--root", "/tmp"])
            else:
                args = self.parser.parse_args([cmd])
            assert args.command == cmd

    def test_global_db_path_default(self):
        args = self.parser.parse_args(["status"])
        assert args.db_path  # has a default

    def test_global_db_path_override(self):
        args = self.parser.parse_args(["--db-path", "/custom/path", "status"])
        assert args.db_path == "/custom/path"

    def test_search_args(self):
        args = self.parser.parse_args(["search", "-q", "hello", "-k", "5"])
        assert args.query == "hello"
        assert args.top_k == 5

    def test_search_default_top_k(self):
        args = self.parser.parse_args(["search", "--query", "test"])
        assert args.top_k == 10

    def test_sync_glob_default(self):
        args = self.parser.parse_args(["sync", "--root", "/tmp"])
        assert args.glob == "**/*"

    def test_watch_debounce_default(self):
        args = self.parser.parse_args(["watch", "--root", "/tmp"])
        assert args.debounce_ms == 500

    def test_no_command_returns_none(self):
        args = self.parser.parse_args([])
        assert args.command is None

    def test_default_excludes_include_codexlens(self):
        assert ".codexlens" in DEFAULT_EXCLUDES

    def test_should_exclude_codexlens_directory(self):
        assert should_exclude(Path(".codexlens") / "metadata.db", DEFAULT_EXCLUDES) is True


# ---------------------------------------------------------------------------
# JSON output helpers
# ---------------------------------------------------------------------------

class TestJsonHelpers:
    def test_json_output(self, capsys):
        _json_output({"key": "value"})
        out = capsys.readouterr().out.strip()
        parsed = json.loads(out)
        assert parsed == {"key": "value"}

    def test_json_output_list(self, capsys):
        _json_output([1, 2, 3])
        out = capsys.readouterr().out.strip()
        assert json.loads(out) == [1, 2, 3]

    def test_json_output_unicode(self, capsys):
        _json_output({"msg": "中文测试"})
        out = capsys.readouterr().out.strip()
        parsed = json.loads(out)
        assert parsed["msg"] == "中文测试"

    def test_error_exit(self):
        with pytest.raises(SystemExit) as exc_info:
            _error_exit("something broke")
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# cmd_init (lightweight, no model loading)
# ---------------------------------------------------------------------------

class TestCmdInit:
    def test_init_creates_databases(self, tmp_path):
        """Init should create metadata.db and fts.db."""
        from codexlens_search.bridge import cmd_init
        import argparse

        db_path = str(tmp_path / "test_idx")
        args = argparse.Namespace(db_path=db_path, verbose=False)
        cmd_init(args)

        assert (Path(db_path) / "metadata.db").exists()
        assert (Path(db_path) / "fts.db").exists()


# ---------------------------------------------------------------------------
# cmd_status (lightweight, no model loading)
# ---------------------------------------------------------------------------

class TestCmdStatus:
    def test_status_not_initialized(self, tmp_path, capsys):
        from codexlens_search.bridge import cmd_status
        import argparse

        db_path = str(tmp_path / "empty_idx")
        Path(db_path).mkdir()
        args = argparse.Namespace(db_path=db_path, verbose=False)
        cmd_status(args)

        out = json.loads(capsys.readouterr().out.strip())
        assert out["status"] == "not_initialized"

    def test_status_after_init(self, tmp_path, capsys):
        from codexlens_search.bridge import cmd_init, cmd_status
        import argparse

        db_path = str(tmp_path / "idx")
        args = argparse.Namespace(db_path=db_path, verbose=False)
        cmd_init(args)

        # Re-capture after init output
        capsys.readouterr()

        cmd_status(args)
        out = json.loads(capsys.readouterr().out.strip())
        assert out["status"] == "ok"
        assert out["files_tracked"] == 0
        assert out["deleted_chunks"] == 0


# ---------------------------------------------------------------------------
# create_config_from_env
# ---------------------------------------------------------------------------

class TestCreateConfigFromEnv:
    def test_basic_defaults(self, tmp_path):
        from codexlens_search.bridge import create_config_from_env
        config = create_config_from_env(tmp_path)
        assert config.metadata_db_path == str(tmp_path.resolve() / "metadata.db")

    def test_env_vars_override(self, tmp_path):
        from codexlens_search.bridge import create_config_from_env
        env = {
            "CODEXLENS_EMBED_API_URL": "https://example.com/v1",
            "CODEXLENS_EMBED_API_KEY": "test-key",
            "CODEXLENS_EMBED_API_MODEL": "test-model",
            "CODEXLENS_EMBED_DIM": "768",
        }
        with patch.dict(os.environ, env):
            config = create_config_from_env(tmp_path)
            assert config.embed_api_url == "https://example.com/v1"
            assert config.embed_api_key == "test-key"
            assert config.embed_api_model == "test-model"
            assert config.embed_dim == 768

    def test_explicit_overrides_take_priority(self, tmp_path):
        from codexlens_search.bridge import create_config_from_env
        env = {"CODEXLENS_EMBED_API_URL": "https://env.com"}
        with patch.dict(os.environ, env):
            config = create_config_from_env(
                tmp_path, embed_api_url="https://override.com"
            )
            assert config.embed_api_url == "https://override.com"

    def test_multi_endpoint_parsing(self, tmp_path):
        from codexlens_search.bridge import create_config_from_env
        env = {
            "CODEXLENS_EMBED_API_ENDPOINTS": "url1|key1|model1,url2|key2"
        }
        with patch.dict(os.environ, env):
            config = create_config_from_env(tmp_path)
            assert len(config.embed_api_endpoints) == 2
            assert config.embed_api_endpoints[0]["url"] == "url1"
            assert config.embed_api_endpoints[0]["model"] == "model1"
            assert config.embed_api_endpoints[1]["url"] == "url2"
            assert "model" not in config.embed_api_endpoints[1]

    def test_ast_chunking_env(self, tmp_path):
        from codexlens_search.bridge import create_config_from_env
        with patch.dict(os.environ, {"CODEXLENS_AST_CHUNKING": "true"}):
            config = create_config_from_env(tmp_path)
            assert config.ast_chunking is True

    def test_search_params_env(self, tmp_path):
        from codexlens_search.bridge import create_config_from_env
        env = {
            "CODEXLENS_BINARY_TOP_K": "200",
            "CODEXLENS_ANN_TOP_K": "50",
            "CODEXLENS_FTS_TOP_K": "30",
            "CODEXLENS_FUSION_K": "60",
        }
        with patch.dict(os.environ, env):
            config = create_config_from_env(tmp_path)
            assert config.binary_top_k == 200
            assert config.ann_top_k == 50
            assert config.fts_top_k == 30
            assert config.fusion_k == 60

    def test_agent_env_vars(self, tmp_path):
        from codexlens_search.bridge import create_config_from_env
        env = {
            "CODEXLENS_AGENT_LLM_MODEL": "gpt-4o-mini",
            "CODEXLENS_AGENT_LLM_API_KEY": "test-key",
            "CODEXLENS_AGENT_MAX_ITERATIONS": "7",
        }
        with patch.dict(os.environ, env):
            config = create_config_from_env(tmp_path)
            assert config.agent_llm_model == "gpt-4o-mini"
            assert config.agent_llm_api_key == "test-key"
            assert config.agent_max_iterations == 7


# ---------------------------------------------------------------------------
# cmd_search (with mocked pipeline)
# ---------------------------------------------------------------------------

class TestCmdSearch:
    def test_search_outputs_json(self, tmp_path, capsys):
        from codexlens_search.bridge import cmd_search

        mock_result = MagicMock()
        mock_result.path = "src/main.py"
        mock_result.score = 0.95
        mock_result.line = 10
        mock_result.end_line = 20
        mock_result.snippet = "def main():"
        mock_result.content = "def main():\n    pass"

        mock_search = MagicMock()
        mock_search.search.return_value = [mock_result]

        with patch("codexlens_search.bridge._create_pipeline") as mock_cp:
            mock_cp.return_value = (MagicMock(), mock_search, MagicMock())
            args = argparse.Namespace(
                db_path=str(tmp_path),
                verbose=False,
                query="main",
                top_k=10,
                embed_api_url="",
                embed_api_key="",
                embed_api_model="",
                embed_model=None,
            )
            cmd_search(args)

        out = json.loads(capsys.readouterr().out.strip())
        assert len(out) == 1
        assert out[0]["path"] == "src/main.py"
        assert out[0]["score"] == 0.95


class TestCmdSearchFiles:
    def test_search_files_outputs_json(self, tmp_path, capsys):
        from codexlens_search.bridge import cmd_search_files

        mock_result = MagicMock()
        mock_result.path = "src/main.py"
        mock_result.score = 0.95
        mock_result.best_chunk_id = 42
        mock_result.line = 10
        mock_result.end_line = 20
        mock_result.snippet = "def main():"
        mock_result.content = "def main():\n    pass"
        mock_result.chunk_ids = (1, 42)

        mock_search = MagicMock()
        mock_search.search_files.return_value = [mock_result]

        with patch("codexlens_search.bridge._create_pipeline") as mock_cp:
            mock_cp.return_value = (MagicMock(), mock_search, MagicMock())
            args = argparse.Namespace(
                db_path=str(tmp_path),
                verbose=False,
                query="main",
                top_k=10,
                embed_api_url="",
                embed_api_key="",
                embed_api_model="",
                embed_model=None,
            )
            cmd_search_files(args)

        out = json.loads(capsys.readouterr().out.strip())
        assert len(out) == 1
        assert out[0]["path"] == "src/main.py"
        assert out[0]["best_chunk_id"] == 42


# ---------------------------------------------------------------------------
# cmd_locate (with mocked agent)
# ---------------------------------------------------------------------------

class TestCmdLocate:
    def test_locate_outputs_json(self, tmp_path, capsys):
        from codexlens_search.bridge import cmd_locate

        mock_result = MagicMock()
        mock_result.path = "src/main.py"
        mock_result.score = 0.95
        mock_result.best_chunk_id = 42
        mock_result.line = 10
        mock_result.end_line = 20
        mock_result.snippet = "def main():"
        mock_result.content = "def main():\n    pass"
        mock_result.chunk_ids = (1, 42)

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = [mock_result]

        mock_search = MagicMock()
        mock_search._entity_graph = MagicMock()

        mock_config = MagicMock()
        mock_config.agent_max_iterations = 5

        with patch("codexlens_search.bridge._create_pipeline") as mock_cp, patch(
            "codexlens_search.bridge.create_agent"
        ) as mock_ca:
            mock_cp.return_value = (MagicMock(), mock_search, mock_config)
            mock_ca.return_value = mock_agent

            args = argparse.Namespace(
                db_path=str(tmp_path),
                verbose=False,
                query="main",
                top_k=10,
                max_iterations=None,
                embed_api_url="",
                embed_api_key="",
                embed_api_model="",
                embed_model=None,
            )
            cmd_locate(args)

        out = json.loads(capsys.readouterr().out.strip())
        assert len(out) == 1
        assert out[0]["path"] == "src/main.py"
        assert out[0]["best_chunk_id"] == 42


# ---------------------------------------------------------------------------
# cmd_index_file (with mocked pipeline)
# ---------------------------------------------------------------------------

class TestCmdIndexFile:
    def test_index_file_missing(self, tmp_path):
        from codexlens_search.bridge import cmd_index_file

        with patch("codexlens_search.bridge._create_pipeline") as mock_cp:
            mock_cp.return_value = (MagicMock(), MagicMock(), MagicMock())
            args = argparse.Namespace(
                db_path=str(tmp_path),
                verbose=False,
                file=str(tmp_path / "nonexistent.py"),
                root=None,
                embed_api_url="",
                embed_api_key="",
                embed_api_model="",
                embed_model=None,
            )
            with pytest.raises(SystemExit):
                cmd_index_file(args)

    def test_index_file_success(self, tmp_path, capsys):
        from codexlens_search.bridge import cmd_index_file

        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        mock_indexing = MagicMock()
        mock_indexing.index_file.return_value = MagicMock(
            files_processed=1, chunks_created=2, duration_seconds=0.1
        )

        with patch("codexlens_search.bridge._create_pipeline") as mock_cp:
            mock_cp.return_value = (mock_indexing, MagicMock(), MagicMock())
            args = argparse.Namespace(
                db_path=str(tmp_path / "db"),
                verbose=False,
                file=str(test_file),
                root=None,
                embed_api_url="",
                embed_api_key="",
                embed_api_model="",
                embed_model=None,
            )
            cmd_index_file(args)

        out = json.loads(capsys.readouterr().out.strip())
        assert out["status"] == "indexed"
        assert out["files_processed"] == 1


# ---------------------------------------------------------------------------
# cmd_remove_file (with mocked pipeline)
# ---------------------------------------------------------------------------

class TestCmdRemoveFile:
    def test_remove_file(self, tmp_path, capsys):
        from codexlens_search.bridge import cmd_remove_file

        mock_indexing = MagicMock()

        with patch("codexlens_search.bridge._create_pipeline") as mock_cp:
            mock_cp.return_value = (mock_indexing, MagicMock(), MagicMock())
            args = argparse.Namespace(
                db_path=str(tmp_path / "db"),
                verbose=False,
                file="src/old.py",
                embed_api_url="",
                embed_api_key="",
                embed_api_model="",
                embed_model=None,
            )
            cmd_remove_file(args)

        mock_indexing.remove_file.assert_called_once_with("src/old.py")
        out = json.loads(capsys.readouterr().out.strip())
        assert out["status"] == "removed"


# ---------------------------------------------------------------------------
# cmd_sync (with mocked pipeline)
# ---------------------------------------------------------------------------

class TestCmdSync:
    def test_sync_missing_root(self, tmp_path):
        from codexlens_search.bridge import cmd_sync

        with patch("codexlens_search.bridge._create_pipeline") as mock_cp:
            mock_cp.return_value = (MagicMock(), MagicMock(), MagicMock())
            args = argparse.Namespace(
                db_path=str(tmp_path / "db"),
                verbose=False,
                root=str(tmp_path / "nonexistent"),
                glob="**/*",
                exclude=None,
                embed_api_url="",
                embed_api_key="",
                embed_api_model="",
                embed_model=None,
            )
            with pytest.raises(SystemExit):
                cmd_sync(args)

    def test_sync_success(self, tmp_path, capsys):
        from codexlens_search.bridge import cmd_sync

        # Create some files to sync
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("x = 1")
        (tmp_path / "src" / "util.py").write_text("y = 2")

        mock_indexing = MagicMock()
        mock_indexing.sync.return_value = MagicMock(
            files_processed=2, chunks_created=4, duration_seconds=0.5
        )

        with patch("codexlens_search.bridge._create_pipeline") as mock_cp:
            mock_cp.return_value = (mock_indexing, MagicMock(), MagicMock())
            args = argparse.Namespace(
                db_path=str(tmp_path / "db"),
                verbose=False,
                root=str(tmp_path),
                glob="**/*",
                exclude=None,
                embed_api_url="",
                embed_api_key="",
                embed_api_model="",
                embed_model=None,
            )
            cmd_sync(args)

        out = json.loads(capsys.readouterr().out.strip())
        assert out["status"] == "synced"
        assert out["files_processed"] == 2

    def test_sync_custom_exclude(self, tmp_path, capsys):
        from codexlens_search.bridge import cmd_sync

        (tmp_path / "main.py").write_text("x = 1")
        (tmp_path / "skip_dir").mkdir()
        (tmp_path / "skip_dir" / "skipped.py").write_text("z = 1")

        mock_indexing = MagicMock()
        mock_indexing.sync.return_value = MagicMock(
            files_processed=1, chunks_created=1, duration_seconds=0.1
        )

        with patch("codexlens_search.bridge._create_pipeline") as mock_cp:
            mock_cp.return_value = (mock_indexing, MagicMock(), MagicMock())
            args = argparse.Namespace(
                db_path=str(tmp_path / "db"),
                verbose=False,
                root=str(tmp_path),
                glob="**/*",
                exclude=["skip_dir"],
                embed_api_url="",
                embed_api_key="",
                embed_api_model="",
                embed_model=None,
            )
            cmd_sync(args)

        # The sync call should have excluded skip_dir files
        call_args = mock_indexing.sync.call_args
        file_paths = call_args[0][0]
        file_names = [p.name for p in file_paths]
        assert "skipped.py" not in file_names


# ---------------------------------------------------------------------------
# should_exclude edge cases
# ---------------------------------------------------------------------------

class TestShouldExclude:
    def test_nested_exclude(self):
        assert should_exclude(Path("a/b/node_modules/c.js"), DEFAULT_EXCLUDES) is True

    def test_not_excluded(self):
        assert should_exclude(Path("src/main.py"), DEFAULT_EXCLUDES) is False

    def test_git_directory(self):
        assert should_exclude(Path(".git/objects/abc"), DEFAULT_EXCLUDES) is True

    def test_empty_path(self):
        # Path with no parts matching excludes
        assert should_exclude(Path("readme.md"), DEFAULT_EXCLUDES) is False


# ---------------------------------------------------------------------------
# _ensure_utf8_stdio
# ---------------------------------------------------------------------------

class TestEnsureUtf8:
    def test_reconfigure_on_windows(self):
        from codexlens_search.bridge import _ensure_utf8_stdio
        # Should not raise on any platform
        _ensure_utf8_stdio()


# ---------------------------------------------------------------------------
# _resolve_db_path
# ---------------------------------------------------------------------------

class TestResolveDbPath:
    def test_creates_directory(self, tmp_path):
        from codexlens_search.bridge import _resolve_db_path
        target = tmp_path / "new_db"
        args = argparse.Namespace(db_path=str(target))
        result = _resolve_db_path(args)
        assert result.exists()
        assert result.is_dir()
