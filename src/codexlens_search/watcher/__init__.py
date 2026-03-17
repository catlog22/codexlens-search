"""File watcher and incremental indexer for codexlens-search.

Requires the ``watcher`` extra::

    pip install codexlens-search[watcher]
"""
from codexlens_search.watcher.events import ChangeType, FileEvent, WatcherConfig
from codexlens_search.watcher.file_watcher import FileWatcher
from codexlens_search.watcher.incremental_indexer import IncrementalIndexer

__all__ = [
    "ChangeType",
    "FileEvent",
    "FileWatcher",
    "IncrementalIndexer",
    "WatcherConfig",
]
