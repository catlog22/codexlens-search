"""Gitignore-aware file exclusion using pathspec library.

Discovers .gitignore files recursively from a root directory, caches parsed
PathSpec objects per directory, and provides is_excluded() for path matching.

Requires the optional ``pathspec`` dependency (``pip install codexlens-search[gitignore]``).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import pathspec  # type: ignore[import-untyped]

    _HAS_PATHSPEC = True
except ImportError:
    _HAS_PATHSPEC = False


class GitignoreAwareMatcher:
    """Recursive .gitignore discovery and matching with per-directory caching.

    Each .gitignore file is parsed into a ``pathspec.PathSpec`` and cached
    keyed by its parent directory.  Cache entries are invalidated when the
    file's mtime changes, so edits to .gitignore files are picked up
    automatically without restarting the process.

    Matching follows git semantics: a file is excluded if *any* ancestor
    .gitignore (from root to direct parent) contains a matching pattern.
    Child .gitignore files can *negate* patterns from parents by prefixing
    with ``!``, which pathspec handles natively.
    """

    def __init__(self, root: Path) -> None:
        if not _HAS_PATHSPEC:
            raise ImportError(
                "pathspec is required for gitignore filtering. "
                "Install it with: pip install codexlens-search[gitignore]"
            )
        self._root = root.resolve()
        # Cached specs: directory -> (mtime, PathSpec)
        self._cache: dict[Path, tuple[float, pathspec.PathSpec]] = {}
        # Directories known to have no .gitignore (avoid repeated stat calls)
        self._no_gitignore: set[Path] = set()
        # Initial discovery
        self._discover_gitignores()

    def _discover_gitignores(self) -> None:
        """Walk root and cache all .gitignore files found."""
        for dirpath, _dirnames, filenames in os.walk(self._root):
            if ".gitignore" in filenames:
                gi_path = Path(dirpath) / ".gitignore"
                self._get_spec(gi_path)

    def _get_spec(self, gitignore_path: Path) -> pathspec.PathSpec | None:
        """Return cached PathSpec for a .gitignore, re-parsing if mtime changed."""
        directory = gitignore_path.parent

        try:
            current_mtime = gitignore_path.stat().st_mtime
        except OSError:
            # File disappeared -- remove from cache
            self._cache.pop(directory, None)
            self._no_gitignore.discard(directory)
            return None

        cached = self._cache.get(directory)
        if cached is not None and cached[0] == current_mtime:
            return cached[1]

        # (Re-)parse
        try:
            lines = gitignore_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            self._cache.pop(directory, None)
            return None

        spec = pathspec.PathSpec.from_lines("gitwildmatch", lines)
        self._cache[directory] = (current_mtime, spec)
        self._no_gitignore.discard(directory)
        logger.debug("Parsed .gitignore: %s (%d patterns)", gitignore_path, len(spec.patterns))
        return spec

    def _invalidate_cache(self, directory: Path) -> None:
        """Remove cached spec for a directory, forcing re-parse on next check."""
        self._cache.pop(directory, None)
        self._no_gitignore.discard(directory)

    def is_excluded(self, path: Path) -> bool:
        """Check if *path* is excluded by any ancestor .gitignore.

        Walks from root to the file's parent directory, checking each
        directory's .gitignore.  The path is tested relative to the
        .gitignore's directory so patterns match correctly.

        Args:
            path: Absolute or relative file path to test.

        Returns:
            True if excluded by any .gitignore, False otherwise.
        """
        resolved = Path(path).resolve()

        # Must be under root
        try:
            rel_to_root = resolved.relative_to(self._root)
        except ValueError:
            return False

        # Collect ancestor directories from root down to parent
        parts = rel_to_root.parts
        current = self._root

        for i in range(len(parts)):
            gi_path = current / ".gitignore"

            if current not in self._no_gitignore:
                if gi_path.exists():
                    spec = self._get_spec(gi_path)
                    if spec is not None:
                        # Test path relative to this .gitignore's directory
                        rel = str(resolved.relative_to(current))
                        # Normalize to forward slashes for pathspec
                        rel = rel.replace("\\", "/")
                        if spec.match_file(rel):
                            return True
                else:
                    self._no_gitignore.add(current)

            if i < len(parts) - 1:
                current = current / parts[i]

        return False
