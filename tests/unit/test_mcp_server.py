"""Unit tests for mcp_server.py — tool functions, pipeline management, watcher lifecycle."""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# ---------------------------------------------------------------------------
# Module-level mocking: mock FastMCP before importing mcp_server
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_module_state():
    """Reset mcp_server module-level caches between tests."""
    import codexlens_search.mcp_server as mod
    mod._pipelines.clear()
    mod._bg_indexing.clear()
    mod._watchers.clear()
    yield
    mod._pipelines.clear()
    mod._bg_indexing.clear()
    mod._watchers.clear()


# ---------------------------------------------------------------------------
# _db_path_for_project
# ---------------------------------------------------------------------------

class TestDbPathForProject:
    def test_returns_codexlens_subdir(self, tmp_path):
        from codexlens_search.mcp_server import _db_path_for_project
        result = _db_path_for_project(str(tmp_path))
        assert result == tmp_path.resolve() / ".codexlens"

    def test_resolves_relative_path(self):
        from codexlens_search.mcp_server import _db_path_for_project
        result = _db_path_for_project(".")
        assert result.is_absolute()
        assert result.name == ".codexlens"


# ---------------------------------------------------------------------------
# _get_pipelines
# ---------------------------------------------------------------------------

class TestGetPipelines:
    @patch("codexlens_search.mcp_server.create_pipeline")
    @patch("codexlens_search.mcp_server.create_config_from_env")
    def test_caches_pipeline(self, mock_config, mock_create, tmp_path):
        from codexlens_search.mcp_server import _get_pipelines
        sentinel = (MagicMock(), MagicMock(), MagicMock())
        mock_create.return_value = sentinel

        result1 = _get_pipelines(str(tmp_path))
        result2 = _get_pipelines(str(tmp_path))

        assert result1 is result2
        assert mock_create.call_count == 1  # only called once

    @patch("codexlens_search.mcp_server.create_pipeline")
    @patch("codexlens_search.mcp_server.create_config_from_env")
    def test_force_recreates(self, mock_config, mock_create, tmp_path):
        from codexlens_search.mcp_server import _get_pipelines
        mock_create.return_value = (MagicMock(), MagicMock(), MagicMock())

        _get_pipelines(str(tmp_path))
        _get_pipelines(str(tmp_path), force=True)

        assert mock_create.call_count == 2


# ---------------------------------------------------------------------------
# _get_fts
# ---------------------------------------------------------------------------

class TestGetFts:
    def test_returns_none_when_no_fts_db(self, tmp_path):
        from codexlens_search.mcp_server import _get_fts
        result = _get_fts(str(tmp_path))
        assert result is None

    @patch("codexlens_search.mcp_server.FTSEngine", create=True)
    def test_returns_fts_when_db_exists(self, tmp_path):
        from codexlens_search.mcp_server import _get_fts
        codexlens_dir = tmp_path / ".codexlens"
        codexlens_dir.mkdir()
        fts_path = codexlens_dir / "fts.db"
        fts_path.touch()

        with patch("codexlens_search.mcp_server._db_path_for_project", return_value=codexlens_dir):
            with patch("codexlens_search.search.fts.FTSEngine") as mock_fts:
                mock_fts.return_value = MagicMock()
                result = _get_fts(str(tmp_path))
                assert result is not None


# ---------------------------------------------------------------------------
# _trigger_background_index
# ---------------------------------------------------------------------------

class TestTriggerBackgroundIndex:
    @patch("codexlens_search.mcp_server._get_pipelines")
    @patch("codexlens_search.mcp_server._ensure_watcher")
    @patch("codexlens_search.mcp_server.should_exclude", return_value=False)
    def test_returns_notice_string(self, mock_excl, mock_watcher, mock_pipes, tmp_path):
        from codexlens_search.mcp_server import _trigger_background_index
        mock_pipes.return_value = (MagicMock(), MagicMock(), MagicMock())

        result = _trigger_background_index(str(tmp_path))
        assert "background" in result.lower() or "Background" in result

    def test_returns_in_progress_when_already_running(self, tmp_path):
        import codexlens_search.mcp_server as mod
        from codexlens_search.mcp_server import _trigger_background_index

        # Fake a running thread
        resolved = str(tmp_path.resolve())
        fake_thread = MagicMock()
        fake_thread.is_alive.return_value = True
        mod._bg_indexing[resolved] = fake_thread

        result = _trigger_background_index(str(tmp_path))
        assert "background" in result.lower()
        assert "being built" in result.lower() or "in the background" in result.lower()


# ---------------------------------------------------------------------------
# _ensure_watcher / _stop_watcher / _cleanup_watchers
# ---------------------------------------------------------------------------

class TestWatcherManagement:
    def test_ensure_watcher_returns_none_when_env_not_set(self, tmp_path):
        from codexlens_search.mcp_server import _ensure_watcher
        import os
        with patch.dict(os.environ, {}, clear=True):
            # Remove CODEXLENS_AUTO_WATCH if set
            os.environ.pop("CODEXLENS_AUTO_WATCH", None)
            result = _ensure_watcher(str(tmp_path))
            assert result is None

    def test_stop_watcher_no_active(self, tmp_path):
        from codexlens_search.mcp_server import _stop_watcher
        result = _stop_watcher(str(tmp_path))
        assert "No active" in result or "no active" in result.lower()

    def test_stop_watcher_stops_existing(self, tmp_path):
        import codexlens_search.mcp_server as mod
        from codexlens_search.mcp_server import _stop_watcher

        resolved = str(tmp_path.resolve())
        mock_watcher = MagicMock()
        mod._watchers[resolved] = mock_watcher

        result = _stop_watcher(str(tmp_path))
        mock_watcher.stop.assert_called_once()
        assert "stopped" in result.lower()

    def test_cleanup_watchers_stops_all(self, tmp_path):
        import codexlens_search.mcp_server as mod
        from codexlens_search.mcp_server import _cleanup_watchers

        mock1 = MagicMock()
        mock2 = MagicMock()
        mod._watchers["path1"] = mock1
        mod._watchers["path2"] = mock2

        _cleanup_watchers()

        mock1.stop.assert_called_once()
        mock2.stop.assert_called_once()
        assert len(mod._watchers) == 0


# ---------------------------------------------------------------------------
# search_code tool
# ---------------------------------------------------------------------------

class TestSearchCode:
    @pytest.fixture
    def run_async(self):
        def _run(coro):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        return _run

    def test_returns_error_for_missing_project(self, run_async, tmp_path):
        from codexlens_search.mcp_server import search_code
        result = run_async(search_code(str(tmp_path / "nonexistent"), "query"))
        assert "Error" in result

    def test_mode_regex_delegates(self, run_async, tmp_path):
        from codexlens_search.mcp_server import search_code
        tmp_path.mkdir(exist_ok=True)
        with patch("codexlens_search.mcp_server._search_regex", new_callable=AsyncMock) as mock_rg:
            mock_rg.return_value = "No results found."
            result = run_async(search_code(str(tmp_path), "test", mode="regex"))
            mock_rg.assert_called_once()

    def test_mode_symbol_delegates(self, run_async, tmp_path):
        from codexlens_search.mcp_server import search_code
        tmp_path.mkdir(exist_ok=True)
        with patch("codexlens_search.mcp_server._search_symbol") as mock_sym:
            mock_sym.return_value = "No symbols found."
            result = run_async(search_code(str(tmp_path), "test", mode="symbol"))
            mock_sym.assert_called_once()

    def test_mode_refs_delegates(self, run_async, tmp_path):
        from codexlens_search.mcp_server import search_code
        tmp_path.mkdir(exist_ok=True)
        with patch("codexlens_search.mcp_server._search_refs") as mock_refs:
            mock_refs.return_value = "No references found."
            result = run_async(search_code(str(tmp_path), "test", mode="refs"))
            mock_refs.assert_called_once()


# ---------------------------------------------------------------------------
# _parse_regex_output
# ---------------------------------------------------------------------------

class TestParseRegexOutput:
    def test_empty_input(self):
        from codexlens_search.mcp_server import _parse_regex_output
        assert _parse_regex_output("") == []
        assert _parse_regex_output("Error: something") == []
        assert _parse_regex_output("No results found.") == []

    def test_parses_formatted_output(self):
        from codexlens_search.mcp_server import _parse_regex_output
        raw = (
            "## 1. src/main.py L42\n"
            "```\n"
            "def hello():\n"
            "```\n"
            "\n"
            "## 2. src/utils.py L10\n"
            "```\n"
            "import os\n"
            "```\n"
        )
        results = _parse_regex_output(raw)
        assert len(results) == 2
        assert results[0] == ("src/main.py", 42, "def hello():")
        assert results[1] == ("src/utils.py", 10, "import os")


# ---------------------------------------------------------------------------
# _search_symbol
# ---------------------------------------------------------------------------

class TestSearchSymbol:
    def test_no_fts_returns_error(self, tmp_path):
        from codexlens_search.mcp_server import _search_symbol
        with patch("codexlens_search.mcp_server._get_fts", return_value=None):
            result = _search_symbol(str(tmp_path), "MyClass", 10)
            assert "Error" in result

    def test_no_symbols_found(self, tmp_path):
        from codexlens_search.mcp_server import _search_symbol
        mock_fts = MagicMock()
        mock_fts.get_symbols_by_name.return_value = []
        mock_fts._conn.execute.return_value.fetchall.return_value = []

        with patch("codexlens_search.mcp_server._get_fts", return_value=mock_fts):
            result = _search_symbol(str(tmp_path), "Unknown", 10)
            assert "No symbols" in result


# ---------------------------------------------------------------------------
# _search_refs
# ---------------------------------------------------------------------------

class TestSearchRefs:
    def test_no_fts_returns_error(self, tmp_path):
        from codexlens_search.mcp_server import _search_refs
        with patch("codexlens_search.mcp_server._get_fts", return_value=None):
            result = _search_refs(str(tmp_path), "func")
            assert "Error" in result

    def test_no_refs_found(self, tmp_path):
        from codexlens_search.mcp_server import _search_refs
        mock_fts = MagicMock()
        mock_fts.get_refs_to.return_value = []
        mock_fts.get_refs_from.return_value = []

        with patch("codexlens_search.mcp_server._get_fts", return_value=mock_fts):
            result = _search_refs(str(tmp_path), "func")
            assert "No references" in result

    def test_refs_to_formatting(self, tmp_path):
        from codexlens_search.mcp_server import _search_refs
        mock_fts = MagicMock()
        mock_fts.get_refs_to.return_value = [
            {"ref_kind": "import", "from_name": "main", "from_path": "src/main.py", "line": 5}
        ]
        mock_fts.get_refs_from.return_value = []

        with patch("codexlens_search.mcp_server._get_fts", return_value=mock_fts):
            result = _search_refs(str(tmp_path), "func")
            assert "Referenced by" in result
            assert "main" in result


# ---------------------------------------------------------------------------
# find_files tool
# ---------------------------------------------------------------------------

class TestFindFiles:
    def test_missing_project(self, tmp_path):
        from codexlens_search.mcp_server import find_files
        result = find_files(str(tmp_path / "nonexistent"))
        assert "Error" in result

    def test_finds_files(self, tmp_path):
        from codexlens_search.mcp_server import find_files
        (tmp_path / "a.py").touch()
        (tmp_path / "b.txt").touch()
        result = find_files(str(tmp_path))
        assert "Found" in result
        assert "a.py" in result

    def test_no_matches(self, tmp_path):
        from codexlens_search.mcp_server import find_files
        result = find_files(str(tmp_path), pattern="*.xyz")
        assert "No files" in result

    def test_max_results_limit(self, tmp_path):
        from codexlens_search.mcp_server import find_files
        for i in range(5):
            (tmp_path / f"file{i}.py").touch()
        result = find_files(str(tmp_path), max_results=3)
        assert "limited" in result.lower()


# ---------------------------------------------------------------------------
# watch_project tool
# ---------------------------------------------------------------------------

class TestWatchProject:
    def test_status_stopped(self, tmp_path):
        from codexlens_search.mcp_server import watch_project
        result = watch_project(str(tmp_path), action="status")
        assert "STOPPED" in result

    def test_status_running(self, tmp_path):
        import codexlens_search.mcp_server as mod
        from codexlens_search.mcp_server import watch_project
        resolved = str(tmp_path.resolve())
        mock_w = MagicMock()
        mock_w.is_running = True
        mod._watchers[resolved] = mock_w
        result = watch_project(str(tmp_path), action="status")
        assert "RUNNING" in result

    def test_stop_action(self, tmp_path):
        from codexlens_search.mcp_server import watch_project
        with patch("codexlens_search.mcp_server._stop_watcher") as mock_stop:
            mock_stop.return_value = "stopped"
            result = watch_project(str(tmp_path), action="stop")
            mock_stop.assert_called_once()

    def test_start_action(self, tmp_path):
        from codexlens_search.mcp_server import watch_project
        with patch("codexlens_search.mcp_server._ensure_watcher") as mock_ensure:
            mock_ensure.return_value = "started"
            result = watch_project(str(tmp_path), action="start")
            assert result == "started"


# ---------------------------------------------------------------------------
# index_project tool
# ---------------------------------------------------------------------------

class TestIndexProject:
    @pytest.fixture
    def run_async(self):
        def _run(coro):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        return _run

    def test_missing_project(self, run_async, tmp_path):
        from codexlens_search.mcp_server import index_project
        result = run_async(index_project(str(tmp_path / "nonexistent")))
        assert "Error" in result

    def test_status_action_no_index(self, run_async, tmp_path):
        from codexlens_search.mcp_server import index_project
        result = run_async(index_project(str(tmp_path), action="status"))
        assert "No index" in result or "not found" in result.lower()

    def test_bad_scope(self, run_async, tmp_path):
        from codexlens_search.mcp_server import index_project
        result = run_async(index_project(str(tmp_path), scope="nonexistent_dir"))
        assert "Error" in result


# ---------------------------------------------------------------------------
# _index_status
# ---------------------------------------------------------------------------

class TestIndexStatus:
    def test_no_index(self, tmp_path):
        from codexlens_search.mcp_server import _index_status
        result = _index_status(str(tmp_path))
        assert "No index" in result

    def test_with_index(self, tmp_path):
        from codexlens_search.mcp_server import _index_status
        codexlens_dir = tmp_path / ".codexlens"
        codexlens_dir.mkdir()

        with patch("codexlens_search.mcp_server._db_path_for_project", return_value=codexlens_dir):
            meta_path = codexlens_dir / "metadata.db"
            meta_path.touch()

            mock_meta = MagicMock()
            mock_meta.get_all_files.return_value = ["a.py", "b.py"]
            mock_meta.get_deleted_ids.return_value = [1]
            mock_meta.max_chunk_id.return_value = 10

            with patch("codexlens_search.indexing.metadata.MetadataStore", return_value=mock_meta):
                result = _index_status(str(tmp_path))
                assert "Files:" in result or "2" in result


# ---------------------------------------------------------------------------
# _semantic_search
# ---------------------------------------------------------------------------

class TestSemanticSearch:
    def test_basic_search(self, tmp_path):
        from codexlens_search.mcp_server import _semantic_search
        mock_result = MagicMock()
        mock_result.path = "src/main.py"
        mock_result.line = 1
        mock_result.end_line = 10
        mock_result.score = 0.95
        mock_result.content = "def main():"
        mock_result.language = "python"

        mock_search = MagicMock()
        mock_search.search.return_value = [mock_result]

        with patch("codexlens_search.mcp_server._get_pipelines", return_value=(None, mock_search, None)):
            results = _semantic_search(str(tmp_path), "main function", 10, "")
            assert len(results) == 1
            assert results[0][0] == "src/main.py"

    def test_scope_filtering(self, tmp_path):
        from codexlens_search.mcp_server import _semantic_search

        r1 = MagicMock(path="src/auth/login.py", line=1, end_line=5, score=0.9, content="x", language="python")
        r2 = MagicMock(path="src/utils/helper.py", line=1, end_line=5, score=0.8, content="y", language="python")

        mock_search = MagicMock()
        mock_search.search.return_value = [r1, r2]

        with patch("codexlens_search.mcp_server._get_pipelines", return_value=(None, mock_search, None)):
            results = _semantic_search(str(tmp_path), "test", 10, "src/auth")
            assert len(results) == 1
            assert results[0][0] == "src/auth/login.py"
