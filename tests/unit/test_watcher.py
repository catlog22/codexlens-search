"""Unit tests for watcher module — events, FileWatcher debounce/dedup, IncrementalIndexer."""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codexlens_search.watcher.events import ChangeType, FileEvent, WatcherConfig
from codexlens_search.watcher.incremental_indexer import BatchResult, IncrementalIndexer


# ---------------------------------------------------------------------------
# ChangeType enum
# ---------------------------------------------------------------------------

class TestChangeType:
    def test_values(self):
        assert ChangeType.CREATED.value == "created"
        assert ChangeType.MODIFIED.value == "modified"
        assert ChangeType.DELETED.value == "deleted"

    def test_all_members(self):
        assert len(ChangeType) == 3


# ---------------------------------------------------------------------------
# FileEvent
# ---------------------------------------------------------------------------

class TestFileEvent:
    def test_creation(self):
        e = FileEvent(path=Path("a.py"), change_type=ChangeType.CREATED)
        assert e.path == Path("a.py")
        assert e.change_type == ChangeType.CREATED
        assert isinstance(e.timestamp, float)

    def test_custom_timestamp(self):
        e = FileEvent(path=Path("b.py"), change_type=ChangeType.DELETED, timestamp=42.0)
        assert e.timestamp == 42.0


# ---------------------------------------------------------------------------
# WatcherConfig
# ---------------------------------------------------------------------------

class TestWatcherConfig:
    def test_defaults(self):
        cfg = WatcherConfig()
        assert cfg.debounce_ms == 500
        assert ".git" in cfg.ignored_patterns
        assert "__pycache__" in cfg.ignored_patterns
        assert "node_modules" in cfg.ignored_patterns
        assert ".codexlens" in cfg.ignored_patterns

    def test_custom(self):
        cfg = WatcherConfig(debounce_ms=1000, ignored_patterns={".custom"})
        assert cfg.debounce_ms == 1000
        assert cfg.ignored_patterns == {".custom"}


# ---------------------------------------------------------------------------
# BatchResult
# ---------------------------------------------------------------------------

class TestBatchResult:
    def test_defaults(self):
        r = BatchResult()
        assert r.files_indexed == 0
        assert r.files_removed == 0
        assert r.chunks_created == 0
        assert r.errors == []

    def test_total_processed(self):
        r = BatchResult(files_indexed=3, files_removed=2)
        assert r.total_processed == 5

    def test_has_errors(self):
        r = BatchResult()
        assert r.has_errors is False
        r.errors.append("oops")
        assert r.has_errors is True


# ---------------------------------------------------------------------------
# IncrementalIndexer — event routing
# ---------------------------------------------------------------------------

class TestIncrementalIndexer:
    @pytest.fixture
    def mock_pipeline(self):
        pipeline = MagicMock()
        pipeline.index_file.return_value = MagicMock(
            files_processed=1, chunks_created=3
        )
        return pipeline

    def test_routes_created_to_index_file(self, mock_pipeline):
        indexer = IncrementalIndexer(mock_pipeline, root=Path("/project"))
        events = [
            FileEvent(Path("/project/src/new.py"), ChangeType.CREATED),
        ]
        result = indexer.process_events(events)
        assert result.files_indexed == 1
        mock_pipeline.index_file.assert_called_once()
        # CREATED should NOT use force=True
        call_kwargs = mock_pipeline.index_file.call_args
        assert call_kwargs.kwargs.get("force", call_kwargs[1].get("force")) is False

    def test_routes_modified_to_index_file_with_force(self, mock_pipeline):
        indexer = IncrementalIndexer(mock_pipeline, root=Path("/project"))
        events = [
            FileEvent(Path("/project/src/changed.py"), ChangeType.MODIFIED),
        ]
        result = indexer.process_events(events)
        assert result.files_indexed == 1
        call_kwargs = mock_pipeline.index_file.call_args
        assert call_kwargs.kwargs.get("force", call_kwargs[1].get("force")) is True

    def test_routes_deleted_to_remove_file(self, mock_pipeline, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        indexer = IncrementalIndexer(mock_pipeline, root=root)
        events = [
            FileEvent(root / "src" / "old.py", ChangeType.DELETED),
        ]
        result = indexer.process_events(events)
        assert result.files_removed == 1
        # On Windows relative_to produces backslashes, normalize
        actual_arg = mock_pipeline.remove_file.call_args[0][0]
        assert actual_arg.replace("\\", "/") == "src/old.py"

    def test_batch_with_mixed_events(self, mock_pipeline):
        indexer = IncrementalIndexer(mock_pipeline, root=Path("/project"))
        events = [
            FileEvent(Path("/project/a.py"), ChangeType.CREATED),
            FileEvent(Path("/project/b.py"), ChangeType.MODIFIED),
            FileEvent(Path("/project/c.py"), ChangeType.DELETED),
        ]
        result = indexer.process_events(events)
        assert result.files_indexed == 2
        assert result.files_removed == 1
        assert result.total_processed == 3

    def test_error_isolation(self, mock_pipeline):
        """One file failure should not stop processing of others."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("disk error")
            return MagicMock(files_processed=1, chunks_created=1)

        mock_pipeline.index_file.side_effect = side_effect

        indexer = IncrementalIndexer(mock_pipeline, root=Path("/project"))
        events = [
            FileEvent(Path("/project/fail.py"), ChangeType.CREATED),
            FileEvent(Path("/project/ok.py"), ChangeType.CREATED),
        ]
        result = indexer.process_events(events)

        assert result.files_indexed == 1  # second succeeded
        assert len(result.errors) == 1  # first failed
        assert "disk error" in result.errors[0]

    def test_empty_events(self, mock_pipeline):
        indexer = IncrementalIndexer(mock_pipeline)
        result = indexer.process_events([])
        assert result.total_processed == 0
        mock_pipeline.index_file.assert_not_called()
        mock_pipeline.remove_file.assert_not_called()


# ---------------------------------------------------------------------------
# FileWatcher — debounce and dedup logic (unit-level, no actual FS)
# ---------------------------------------------------------------------------

class TestFileWatcherLogic:
    """Test FileWatcher internals without starting a real watchdog Observer."""

    @pytest.fixture
    def watcher_parts(self):
        """Create a FileWatcher with mocked observer, capture callbacks."""
        # Import here since watchdog is optional
        from codexlens_search.watcher.file_watcher import FileWatcher, _EVENT_PRIORITY

        collected = []

        def on_changes(events):
            collected.extend(events)

        cfg = WatcherConfig(debounce_ms=100, ignored_patterns=set())
        watcher = FileWatcher(Path("."), cfg, on_changes)
        return watcher, collected, _EVENT_PRIORITY

    def test_event_priority_ordering(self, watcher_parts):
        _, _, priority = watcher_parts
        assert priority[ChangeType.DELETED] > priority[ChangeType.MODIFIED]
        assert priority[ChangeType.MODIFIED] > priority[ChangeType.CREATED]

    def test_dedup_keeps_higher_priority(self, watcher_parts, tmp_path):
        watcher, collected, _ = watcher_parts
        f = str(tmp_path / "a.py")
        watcher._on_raw_event(f, ChangeType.CREATED)
        watcher._on_raw_event(f, ChangeType.DELETED)

        watcher.flush_now()

        assert len(collected) == 1
        assert collected[0].change_type == ChangeType.DELETED

    def test_dedup_does_not_downgrade(self, watcher_parts, tmp_path):
        watcher, collected, _ = watcher_parts
        f = str(tmp_path / "b.py")
        watcher._on_raw_event(f, ChangeType.DELETED)
        watcher._on_raw_event(f, ChangeType.CREATED)

        watcher.flush_now()
        assert len(collected) == 1
        # CREATED (priority 1) < DELETED (priority 3), so DELETED stays
        assert collected[0].change_type == ChangeType.DELETED

    def test_multiple_files_kept(self, watcher_parts, tmp_path):
        watcher, collected, _ = watcher_parts
        watcher._on_raw_event(str(tmp_path / "a.py"), ChangeType.CREATED)
        watcher._on_raw_event(str(tmp_path / "b.py"), ChangeType.MODIFIED)
        watcher._on_raw_event(str(tmp_path / "c.py"), ChangeType.DELETED)

        watcher.flush_now()
        assert len(collected) == 3
        paths = {str(e.path) for e in collected}
        assert len(paths) == 3

    def test_flush_clears_pending(self, watcher_parts, tmp_path):
        watcher, collected, _ = watcher_parts
        watcher._on_raw_event(str(tmp_path / "a.py"), ChangeType.CREATED)
        watcher.flush_now()
        assert len(collected) == 1

        collected.clear()
        watcher.flush_now()
        assert len(collected) == 0

    def test_should_watch_filters_ignored(self):
        from codexlens_search.watcher.file_watcher import FileWatcher
        cfg = WatcherConfig(debounce_ms=100)  # uses default ignored_patterns
        watcher = FileWatcher(Path("."), cfg, lambda e: None)
        assert watcher._should_watch(Path("/project/src/main.py")) is True
        assert watcher._should_watch(Path("/project/.git/config")) is False
        assert watcher._should_watch(Path("/project/node_modules/foo.js")) is False
        assert watcher._should_watch(Path("/project/__pycache__/mod.pyc")) is False

    def test_jsonl_serialization(self):
        from codexlens_search.watcher.file_watcher import FileWatcher
        import json

        events = [
            FileEvent(Path("/tmp/a.py"), ChangeType.CREATED, 1000.0),
            FileEvent(Path("/tmp/b.py"), ChangeType.DELETED, 2000.0),
        ]
        output = FileWatcher.events_to_jsonl(events)
        lines = output.strip().split("\n")
        assert len(lines) == 2

        obj1 = json.loads(lines[0])
        assert obj1["change_type"] == "created"
        assert obj1["timestamp"] == 1000.0

        obj2 = json.loads(lines[1])
        assert obj2["change_type"] == "deleted"


# ---------------------------------------------------------------------------
# FileWatcher — lifecycle and additional coverage
# ---------------------------------------------------------------------------

class TestFileWatcherLifecycle:
    """Test FileWatcher start/stop/is_running and edge cases."""

    def test_start_nonexistent_path_raises(self):
        from codexlens_search.watcher.file_watcher import FileWatcher
        cfg = WatcherConfig(debounce_ms=100)
        watcher = FileWatcher(Path("/nonexistent/path/xyz"), cfg, lambda e: None)
        with pytest.raises(ValueError, match="does not exist"):
            watcher.start()

    def test_start_and_stop(self, tmp_path):
        from codexlens_search.watcher.file_watcher import FileWatcher
        cfg = WatcherConfig(debounce_ms=100)
        collected = []
        watcher = FileWatcher(tmp_path, cfg, lambda e: collected.extend(e))

        assert watcher.is_running is False
        watcher.start()
        assert watcher.is_running is True

        # Double start should be a no-op (just logs warning)
        watcher.start()
        assert watcher.is_running is True

        watcher.stop()
        assert watcher.is_running is False

        # Double stop should be safe
        watcher.stop()
        assert watcher.is_running is False

    def test_create_with_indexer(self, tmp_path):
        from codexlens_search.watcher.file_watcher import FileWatcher
        mock_indexer = MagicMock()
        cfg = WatcherConfig(debounce_ms=100)
        watcher = FileWatcher.create_with_indexer(tmp_path, cfg, mock_indexer)
        assert watcher.on_changes is mock_indexer.process_events_async

    def test_jsonl_callback(self, capsys):
        from codexlens_search.watcher.file_watcher import FileWatcher
        events = [
            FileEvent(Path("/tmp/x.py"), ChangeType.MODIFIED, 500.0),
        ]
        FileWatcher.jsonl_callback(events)
        out = capsys.readouterr().out.strip()
        import json
        obj = json.loads(out)
        assert obj["change_type"] == "modified"
        assert obj["timestamp"] == 500.0

    def test_jsonl_callback_empty(self, capsys):
        from codexlens_search.watcher.file_watcher import FileWatcher
        FileWatcher.jsonl_callback([])
        out = capsys.readouterr().out
        assert out == ""  # No output for empty events

    def test_events_to_jsonl_empty(self):
        from codexlens_search.watcher.file_watcher import FileWatcher
        result = FileWatcher.events_to_jsonl([])
        assert result == ""

    def test_on_changes_error_caught(self, tmp_path):
        from codexlens_search.watcher.file_watcher import FileWatcher

        def bad_callback(events):
            raise RuntimeError("callback error")

        cfg = WatcherConfig(debounce_ms=100, ignored_patterns=set())
        watcher = FileWatcher(tmp_path, cfg, bad_callback)
        watcher._on_raw_event(str(tmp_path / "a.py"), ChangeType.CREATED)
        # Flush should not raise even though callback errors
        watcher.flush_now()  # Should not propagate exception


# ---------------------------------------------------------------------------
# FileWatcher _Handler — on_moved coverage
# ---------------------------------------------------------------------------

class TestFileWatcherHandler:
    def test_on_moved_creates_delete_and_create(self, tmp_path):
        from codexlens_search.watcher.file_watcher import FileWatcher, _Handler

        collected = []
        cfg = WatcherConfig(debounce_ms=100, ignored_patterns=set())
        watcher = FileWatcher(tmp_path, cfg, lambda e: collected.extend(e))

        handler = _Handler(watcher)

        # Simulate a move event
        mock_event = MagicMock()
        mock_event.is_directory = False
        mock_event.src_path = str(tmp_path / "old.py")
        mock_event.dest_path = str(tmp_path / "new.py")

        handler.on_moved(mock_event)
        watcher.flush_now()

        # Should produce at least one event (deduplicated by path)
        assert len(collected) >= 1

    def test_on_moved_directory_ignored(self, tmp_path):
        from codexlens_search.watcher.file_watcher import FileWatcher, _Handler

        collected = []
        cfg = WatcherConfig(debounce_ms=100)
        watcher = FileWatcher(tmp_path, cfg, lambda e: collected.extend(e))

        handler = _Handler(watcher)

        mock_event = MagicMock()
        mock_event.is_directory = True

        handler.on_moved(mock_event)
        watcher.flush_now()
        assert len(collected) == 0


# ---------------------------------------------------------------------------
# IncrementalIndexer — process_events_async and debounce
# ---------------------------------------------------------------------------

class TestIncrementalIndexerAsync:
    @pytest.fixture
    def mock_pipeline(self):
        pipeline = MagicMock()
        pipeline.index_file.return_value = MagicMock(
            files_processed=1, chunks_created=1
        )
        return pipeline

    def test_process_events_async_buffers(self, mock_pipeline):
        indexer = IncrementalIndexer(mock_pipeline, root=Path("/project"), debounce_window_ms=50)
        events = [
            FileEvent(Path("/project/a.py"), ChangeType.CREATED),
        ]
        indexer.process_events_async(events)

        # Events are buffered, not immediately processed
        assert len(indexer._event_buffer) == 1
        # Cancel timer to prevent background processing
        if indexer._flush_timer:
            indexer._flush_timer.cancel()

    def test_flush_buffer_deduplicates(self, mock_pipeline):
        indexer = IncrementalIndexer(mock_pipeline, root=Path("/project"))

        # Add duplicate events for same path
        p = Path("/project/a.py")
        indexer._event_buffer = [
            FileEvent(p, ChangeType.CREATED),
            FileEvent(p, ChangeType.MODIFIED),
        ]

        indexer._flush_buffer()

        # Only one index_file call (deduplicated, last event wins)
        assert mock_pipeline.index_file.call_count == 1

    def test_flush_buffer_empty_no_op(self, mock_pipeline):
        indexer = IncrementalIndexer(mock_pipeline)
        indexer._flush_buffer()
        mock_pipeline.index_file.assert_not_called()

    def test_no_root_uses_absolute(self, mock_pipeline, tmp_path):
        indexer = IncrementalIndexer(mock_pipeline, root=None)
        events = [
            FileEvent(tmp_path / "test.py", ChangeType.DELETED),
        ]
        result = indexer.process_events(events)
        assert result.files_removed == 1
        # Should use absolute path string
        actual_arg = mock_pipeline.remove_file.call_args[0][0]
        assert str(tmp_path) in actual_arg or "test.py" in actual_arg
