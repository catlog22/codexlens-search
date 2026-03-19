"""Unit tests for bridge.py CLI — argparse parsing, JSON protocol, error handling."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

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
            "init", "search", "index-file", "remove-file",
            "sync", "watch", "download-models", "status",
        }
        # parse each subcommand with minimal required args to verify it exists
        for cmd in expected:
            if cmd == "search":
                args = self.parser.parse_args(["search", "--query", "test"])
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
