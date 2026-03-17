"""File system watcher using watchdog library.

Ported from codex-lens v1 with simplifications:
- Removed v1-specific Config dependency (uses WatcherConfig directly)
- Removed MAX_QUEUE_SIZE (v2 processes immediately via debounce)
- Removed flush.signal file mechanism
- Added optional JSONL output mode for bridge CLI integration
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .events import ChangeType, FileEvent, WatcherConfig

logger = logging.getLogger(__name__)


# Event priority for deduplication: higher wins when same file appears
# multiple times within one debounce window.
_EVENT_PRIORITY: Dict[ChangeType, int] = {
    ChangeType.CREATED: 1,
    ChangeType.MODIFIED: 2,
    ChangeType.DELETED: 3,
}


class _Handler(FileSystemEventHandler):
    """Internal watchdog handler that converts events to FileEvent."""

    def __init__(self, watcher: FileWatcher) -> None:
        super().__init__()
        self._watcher = watcher

    def on_created(self, event) -> None:
        if not event.is_directory:
            self._watcher._on_raw_event(event.src_path, ChangeType.CREATED)

    def on_modified(self, event) -> None:
        if not event.is_directory:
            self._watcher._on_raw_event(event.src_path, ChangeType.MODIFIED)

    def on_deleted(self, event) -> None:
        if not event.is_directory:
            self._watcher._on_raw_event(event.src_path, ChangeType.DELETED)

    def on_moved(self, event) -> None:
        if event.is_directory:
            return
        # Treat move as delete old + create new
        self._watcher._on_raw_event(event.src_path, ChangeType.DELETED)
        self._watcher._on_raw_event(event.dest_path, ChangeType.CREATED)


class FileWatcher:
    """File system watcher with debounce and event deduplication.

    Monitors a directory recursively using watchdog.  Raw events are
    collected into a queue.  After *debounce_ms* of silence the queue
    is flushed: events are deduplicated per-path (keeping the highest
    priority change type) and delivered via *on_changes*.

    Example::

        def handle(events: list[FileEvent]) -> None:
            for e in events:
                print(e.change_type.value, e.path)

        watcher = FileWatcher(Path("."), WatcherConfig(), handle)
        watcher.start()
        watcher.wait()
    """

    def __init__(
        self,
        root_path: Path,
        config: WatcherConfig,
        on_changes: Callable[[List[FileEvent]], None],
    ) -> None:
        self.root_path = Path(root_path).resolve()
        self.config = config
        self.on_changes = on_changes

        self._observer: Optional[Observer] = None
        self._running = False
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        # Pending events keyed by resolved path
        self._pending: Dict[Path, FileEvent] = {}
        self._pending_lock = threading.Lock()

        # True-debounce timer: resets on every new event
        self._flush_timer: Optional[threading.Timer] = None

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _should_watch(self, path: Path) -> bool:
        """Return True if *path* should not be ignored."""
        parts = path.parts
        for pattern in self.config.ignored_patterns:
            if pattern in parts:
                return False
        return True

    # ------------------------------------------------------------------
    # Event intake (called from watchdog thread)
    # ------------------------------------------------------------------

    def _on_raw_event(self, raw_path: str, change_type: ChangeType) -> None:
        """Accept a raw watchdog event, filter, and queue with debounce."""
        path = Path(raw_path).resolve()

        if not self._should_watch(path):
            return

        event = FileEvent(path=path, change_type=change_type)

        with self._pending_lock:
            existing = self._pending.get(path)
            if existing is None or _EVENT_PRIORITY[change_type] >= _EVENT_PRIORITY[existing.change_type]:
                self._pending[path] = event

            # Cancel previous timer and start a new one (true debounce)
            if self._flush_timer is not None:
                self._flush_timer.cancel()

            self._flush_timer = threading.Timer(
                self.config.debounce_ms / 1000.0,
                self._flush,
            )
            self._flush_timer.daemon = True
            self._flush_timer.start()

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        """Deduplicate and deliver pending events."""
        with self._pending_lock:
            if not self._pending:
                return
            events = list(self._pending.values())
            self._pending.clear()
            self._flush_timer = None

        try:
            self.on_changes(events)
        except Exception:
            logger.exception("Error in on_changes callback")

    def flush_now(self) -> None:
        """Immediately flush pending events (manual trigger)."""
        with self._pending_lock:
            if self._flush_timer is not None:
                self._flush_timer.cancel()
                self._flush_timer = None
        self._flush()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start watching the directory (non-blocking)."""
        with self._lock:
            if self._running:
                logger.warning("Watcher already running")
                return

            if not self.root_path.exists():
                raise ValueError(f"Root path does not exist: {self.root_path}")

            self._observer = Observer()
            handler = _Handler(self)
            self._observer.schedule(handler, str(self.root_path), recursive=True)

            self._running = True
            self._stop_event.clear()
            self._observer.start()
            logger.info("Started watching: %s", self.root_path)

    def stop(self) -> None:
        """Stop watching and flush remaining events."""
        with self._lock:
            if not self._running:
                return

            self._running = False
            self._stop_event.set()

            with self._pending_lock:
                if self._flush_timer is not None:
                    self._flush_timer.cancel()
                    self._flush_timer = None

            if self._observer is not None:
                self._observer.stop()
                self._observer.join(timeout=5.0)
                self._observer = None

            # Deliver any remaining events
            self._flush()
            logger.info("Stopped watching: %s", self.root_path)

    def wait(self) -> None:
        """Block until stopped (Ctrl+C or stop() from another thread)."""
        try:
            while self._running:
                self._stop_event.wait(timeout=1.0)
        except KeyboardInterrupt:
            logger.info("Received interrupt, stopping watcher...")
            self.stop()

    @property
    def is_running(self) -> bool:
        """True if the watcher is currently running."""
        return self._running

    # ------------------------------------------------------------------
    # JSONL output helper
    # ------------------------------------------------------------------

    @staticmethod
    def events_to_jsonl(events: List[FileEvent]) -> str:
        """Serialize a batch of events as newline-delimited JSON.

        Each line is a JSON object with keys: ``path``, ``change_type``,
        ``timestamp``.  Useful for bridge CLI integration.
        """
        lines: list[str] = []
        for evt in events:
            obj = {
                "path": str(evt.path),
                "change_type": evt.change_type.value,
                "timestamp": evt.timestamp,
            }
            lines.append(json.dumps(obj, ensure_ascii=False))
        return "\n".join(lines)

    @staticmethod
    def jsonl_callback(events: List[FileEvent]) -> None:
        """Callback that writes JSONL to stdout.

        Suitable as *on_changes* when running in bridge/CLI mode::

            watcher = FileWatcher(root, config, FileWatcher.jsonl_callback)
        """
        output = FileWatcher.events_to_jsonl(events)
        if output:
            sys.stdout.write(output + "\n")
            sys.stdout.flush()
