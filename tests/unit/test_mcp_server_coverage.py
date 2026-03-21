"""Additional mcp_server.py coverage tests — search_auto, symbol/refs formatting, watcher, index_project."""
from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


@pytest.fixture(autouse=True)
def _reset_module_state():
    import codexlens_search.mcp_server as mod
    mod._pipelines.clear()
    mod._bg_indexing.clear()
    mod._watchers.clear()
    yield
    mod._pipelines.clear()
    mod._bg_indexing.clear()
    mod._watchers.clear()


@pytest.fixture
def run_async():
    def _run(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    return _run


# ---------------------------------------------------------------------------
# _search_auto: no index + no rg (covers line 243)
# ---------------------------------------------------------------------------

class TestSearchAutoNoBackend:
    def test_no_index_no_rg(self, run_async, tmp_path):
        from codexlens_search.mcp_server import _search_auto
        with patch("codexlens_search.mcp_server.shutil") as mock_shutil:
            mock_shutil.which.return_value = None
            result = run_async(_search_auto(str(tmp_path), "test", 10, ""))
            assert "Error" in result or "no index" in result.lower()


# ---------------------------------------------------------------------------
# _search_auto: has_index only (semantic only path, covers lines 273-275)
# ---------------------------------------------------------------------------

class TestSearchAutoSemanticOnly:
    def test_semantic_only_no_rg(self, run_async, tmp_path):
        from codexlens_search.mcp_server import _search_auto
        db_path = tmp_path / ".codexlens"
        db_path.mkdir()
        (db_path / "metadata.db").touch()

        mock_result = MagicMock(
            path="src/main.py", line=1, end_line=5, score=0.9,
            content="def main():", language="python"
        )
        mock_search = MagicMock()
        mock_search.search.return_value = [mock_result]

        with patch("codexlens_search.mcp_server._db_path_for_project", return_value=db_path):
            with patch("codexlens_search.mcp_server.shutil") as mock_shutil:
                mock_shutil.which.return_value = None
                with patch("codexlens_search.mcp_server._semantic_search", return_value=[
                    ("src/main.py", 1, 5, 0.9, "def main():", "python")
                ]):
                    result = run_async(_search_auto(str(tmp_path), "main", 10, ""))
                    assert "src/main.py" in result


# ---------------------------------------------------------------------------
# _search_auto: regex only (no index, covers lines 277-278)
# ---------------------------------------------------------------------------

class TestSearchAutoRegexOnly:
    def test_regex_only_no_index(self, run_async, tmp_path):
        from codexlens_search.mcp_server import _search_auto
        with patch("codexlens_search.mcp_server._db_path_for_project", return_value=tmp_path / ".codexlens"):
            with patch("codexlens_search.mcp_server.shutil") as mock_shutil:
                mock_shutil.which.return_value = "/usr/bin/rg"
                with patch("codexlens_search.mcp_server._trigger_background_index", return_value="Note: indexing..."):
                    with patch("codexlens_search.mcp_server._search_regex", new_callable=AsyncMock) as mock_rg:
                        mock_rg.return_value = "No results found."
                        result = run_async(_search_auto(str(tmp_path), "test", 10, ""))
                        assert "indexing" in result.lower() or "No results" in result


# ---------------------------------------------------------------------------
# _search_auto: no results (covers lines 304-307)
# ---------------------------------------------------------------------------

class TestSearchAutoNoResults:
    def test_no_results_with_indexing_notice(self, run_async, tmp_path):
        from codexlens_search.mcp_server import _search_auto
        with patch("codexlens_search.mcp_server._db_path_for_project", return_value=tmp_path / ".codexlens"):
            with patch("codexlens_search.mcp_server.shutil") as mock_shutil:
                mock_shutil.which.return_value = "/usr/bin/rg"
                with patch("codexlens_search.mcp_server._trigger_background_index", return_value="Note: indexing..."):
                    with patch("codexlens_search.mcp_server._search_regex", new_callable=AsyncMock) as mock_rg:
                        mock_rg.return_value = "No results found."
                        result = run_async(_search_auto(str(tmp_path), "xyznonexistent", 10, ""))
                        assert "No results" in result


# ---------------------------------------------------------------------------
# _search_symbol: with results (covers lines 389-400)
# ---------------------------------------------------------------------------

class TestSearchSymbolWithResults:
    def test_exact_match_with_results(self, tmp_path):
        from codexlens_search.mcp_server import _search_symbol
        mock_fts = MagicMock()
        mock_fts.get_symbols_by_name.return_value = [
            {
                "id": 1, "chunk_id": 10, "name": "MyClass",
                "kind": "class", "start_line": 5, "end_line": 20,
                "parent_name": "", "signature": "class MyClass:", "language": "python",
            }
        ]
        mock_fts.get_doc_meta.return_value = ("src/models.py", "content", 5, 20, "python")

        with patch("codexlens_search.mcp_server._get_fts", return_value=mock_fts):
            result = _search_symbol(str(tmp_path), "MyClass", 10)
            assert "MyClass" in result
            assert "class" in result
            assert "src/models.py" in result

    def test_fuzzy_match(self, tmp_path):
        from codexlens_search.mcp_server import _search_symbol
        mock_fts = MagicMock()
        mock_fts.get_symbols_by_name.return_value = []
        # Fuzzy match via LIKE
        mock_fts._conn.execute.return_value.fetchall.return_value = [
            (1, 10, "my_helper", "function", 5, 15, "", "def my_helper():", "python")
        ]
        mock_fts.get_doc_meta.return_value = ("src/utils.py", "content", 5, 15, "python")

        with patch("codexlens_search.mcp_server._get_fts", return_value=mock_fts):
            result = _search_symbol(str(tmp_path), "helper", 10)
            assert "my_helper" in result

    def test_symbol_with_parent(self, tmp_path):
        from codexlens_search.mcp_server import _search_symbol
        mock_fts = MagicMock()
        mock_fts.get_symbols_by_name.return_value = [
            {
                "id": 1, "chunk_id": 10, "name": "run",
                "kind": "method", "start_line": 10, "end_line": 25,
                "parent_name": "MyClass", "signature": "def run(self):", "language": "python",
            }
        ]
        mock_fts.get_doc_meta.return_value = ("src/app.py", "content", 10, 25, "python")

        with patch("codexlens_search.mcp_server._get_fts", return_value=mock_fts):
            result = _search_symbol(str(tmp_path), "run", 10)
            assert "MyClass" in result


# ---------------------------------------------------------------------------
# _search_refs: with refs_from (covers lines 425-430)
# ---------------------------------------------------------------------------

class TestSearchRefsFrom:
    def test_refs_from_formatting(self, tmp_path):
        from codexlens_search.mcp_server import _search_refs
        mock_fts = MagicMock()
        mock_fts.get_refs_to.return_value = []
        mock_fts.get_refs_from.return_value = [
            {"ref_kind": "call", "from_name": "main", "to_name": "helper", "line": 10}
        ]

        with patch("codexlens_search.mcp_server._get_fts", return_value=mock_fts):
            result = _search_refs(str(tmp_path), "main")
            assert "References from" in result
            assert "helper" in result


# ---------------------------------------------------------------------------
# _ensure_watcher: env set (covers lines 117-144)
# ---------------------------------------------------------------------------

class TestEnsureWatcherWithEnv:
    def test_ensure_watcher_already_running(self, tmp_path):
        import codexlens_search.mcp_server as mod
        from codexlens_search.mcp_server import _ensure_watcher

        resolved = str(tmp_path.resolve())
        mock_w = MagicMock()
        mock_w.is_running = True
        mod._watchers[resolved] = mock_w

        with patch.dict(os.environ, {"CODEXLENS_AUTO_WATCH": "true"}):
            result = _ensure_watcher(str(tmp_path))
            assert result is None

    def test_ensure_watcher_exception_in_setup(self, tmp_path):
        from codexlens_search.mcp_server import _ensure_watcher
        with patch.dict(os.environ, {"CODEXLENS_AUTO_WATCH": "true"}):
            with patch("codexlens_search.mcp_server._get_pipelines", side_effect=Exception("setup failed")):
                # _ensure_watcher catches exceptions and returns None
                result = _ensure_watcher(str(tmp_path))
                assert result is None


# ---------------------------------------------------------------------------
# _trigger_background_index: completed thread cleanup (covers line 77)
# ---------------------------------------------------------------------------

class TestTriggerBackgroundIndexCompleted:
    def test_completed_thread_cleanup(self, tmp_path):
        import codexlens_search.mcp_server as mod
        from codexlens_search.mcp_server import _trigger_background_index

        resolved = str(tmp_path.resolve())
        fake_thread = MagicMock()
        fake_thread.is_alive.return_value = False  # thread is done
        mod._bg_indexing[resolved] = fake_thread

        with patch("codexlens_search.mcp_server._get_pipelines") as mock_pipes:
            mock_indexing = MagicMock()
            mock_indexing.sync.return_value = MagicMock()
            mock_pipes.return_value = (mock_indexing, MagicMock(), MagicMock())
            with patch("codexlens_search.mcp_server._ensure_watcher"):
                result = _trigger_background_index(str(tmp_path))
                # Should have cleaned up the completed thread and started new one
                assert "background" in result.lower()


# ---------------------------------------------------------------------------
# _cleanup_watchers: exception in stop (covers lines 165-166)
# ---------------------------------------------------------------------------

class TestCleanupWatchersException:
    def test_cleanup_handles_stop_exception(self):
        import codexlens_search.mcp_server as mod
        from codexlens_search.mcp_server import _cleanup_watchers

        mock_w = MagicMock()
        mock_w.stop.side_effect = RuntimeError("stop failed")
        mod._watchers["path1"] = mock_w

        _cleanup_watchers()  # Should not raise
        assert len(mod._watchers) == 0


# ---------------------------------------------------------------------------
# index_project: sync with progress (covers lines 530-569)
# ---------------------------------------------------------------------------

class TestIndexProjectSync:
    def test_sync_with_files(self, run_async, tmp_path):
        from codexlens_search.mcp_server import index_project

        (tmp_path / "main.py").write_text("x = 1")

        mock_indexing = MagicMock()
        mock_indexing.sync.return_value = MagicMock(
            files_processed=1, chunks_created=3, duration_seconds=0.5
        )

        with patch("codexlens_search.mcp_server._get_pipelines", return_value=(mock_indexing, MagicMock(), MagicMock())):
            with patch("codexlens_search.mcp_server._ensure_watcher"):
                result = run_async(index_project(str(tmp_path), action="sync"))
                assert "Indexed" in result or "1 files" in result

    def test_rebuild_with_force(self, run_async, tmp_path):
        from codexlens_search.mcp_server import index_project

        (tmp_path / "main.py").write_text("x = 1")

        mock_indexing = MagicMock()
        mock_indexing.sync.return_value = MagicMock(
            files_processed=1, chunks_created=2, duration_seconds=0.3
        )

        with patch("codexlens_search.mcp_server._get_pipelines", return_value=(mock_indexing, MagicMock(), MagicMock())):
            with patch("codexlens_search.mcp_server._ensure_watcher"):
                result = run_async(index_project(str(tmp_path), force=True))
                assert "Indexed" in result


# ---------------------------------------------------------------------------
# _index_status with symbols (covers lines 595-611)
# ---------------------------------------------------------------------------

class TestIndexStatusWithSymbols:
    def test_status_with_symbols(self, tmp_path):
        from codexlens_search.mcp_server import _index_status
        codexlens_dir = tmp_path / ".codexlens"
        codexlens_dir.mkdir()
        (codexlens_dir / "metadata.db").touch()
        (codexlens_dir / "fts.db").touch()

        mock_meta = MagicMock()
        mock_meta.get_all_files.return_value = {"a.py": "hash1", "b.py": "hash2"}
        mock_meta.get_deleted_ids.return_value = []
        mock_meta.max_chunk_id.return_value = 5

        mock_fts = MagicMock()
        mock_fts._conn.execute.side_effect = [
            MagicMock(fetchone=MagicMock(return_value=(10,))),  # sym_count
            MagicMock(fetchone=MagicMock(return_value=(25,))),  # ref_count
        ]

        with patch("codexlens_search.mcp_server._db_path_for_project", return_value=codexlens_dir):
            with patch("codexlens_search.indexing.metadata.MetadataStore", return_value=mock_meta):
                with patch("codexlens_search.search.fts.FTSEngine", return_value=mock_fts):
                    result = _index_status(str(tmp_path))
                    assert "Symbols: 10" in result
                    assert "References: 25" in result
                    assert "Graph search: enabled" in result
