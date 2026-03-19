"""Event types for file watcher."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Set


class ChangeType(Enum):
    """Type of file system change."""

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class FileEvent:
    """A file system change event."""

    path: Path
    change_type: ChangeType
    timestamp: float = field(default_factory=time.time)


@dataclass
class WatcherConfig:
    """Configuration for file watcher.

    Attributes:
        debounce_ms: Milliseconds to wait after the last event before
            flushing the batch.  Default 500ms for low-latency indexing.
        ignored_patterns: Directory/file name patterns to skip.  Any
            path component matching one of these strings is ignored.
    """

    debounce_ms: int = 500
    ignored_patterns: Set[str] = field(default_factory=lambda: {
        # Version control
        ".git", ".svn", ".hg",
        # Python
        ".venv", "venv", "env", "__pycache__", ".pytest_cache",
        ".mypy_cache", ".ruff_cache",
        # Node.js
        "node_modules", "bower_components",
        # Build artifacts
        "dist", "build", "out", "target", "bin", "obj",
        "coverage", "htmlcov",
        # IDE / Editor
        ".idea", ".vscode", ".vs",
        # Package / cache
        ".cache", ".parcel-cache", ".turbo", ".next", ".nuxt", ".codexlens",
        # Logs / temp
        "logs", "tmp", "temp",
    })
