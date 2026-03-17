"""Incremental indexer that processes FileEvents via IndexingPipeline.

Ported from codex-lens v1 with simplifications:
- Uses IndexingPipeline.index_file() / remove_file() directly
- No v1-specific Config, ParserFactory, DirIndexStore dependencies
- Per-file error isolation: one failure does not stop batch processing
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from codexlens_search.indexing.pipeline import IndexingPipeline

from .events import ChangeType, FileEvent

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of processing a batch of file events."""

    files_indexed: int = 0
    files_removed: int = 0
    chunks_created: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def total_processed(self) -> int:
        return self.files_indexed + self.files_removed

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


class IncrementalIndexer:
    """Routes file change events to IndexingPipeline operations.

    CREATED / MODIFIED events call ``pipeline.index_file()``.
    DELETED events call ``pipeline.remove_file()``.

    Each file is processed in isolation so that a single failure
    does not prevent the rest of the batch from being indexed.

    Example::

        indexer = IncrementalIndexer(pipeline, root=Path("/project"))
        result = indexer.process_events([
            FileEvent(Path("src/main.py"), ChangeType.MODIFIED),
        ])
        print(f"Indexed {result.files_indexed}, removed {result.files_removed}")
    """

    def __init__(
        self,
        pipeline: IndexingPipeline,
        *,
        root: Optional[Path] = None,
    ) -> None:
        """Initialize the incremental indexer.

        Args:
            pipeline: The indexing pipeline with metadata store configured.
            root: Optional project root for computing relative paths.
                  If None, absolute paths are used as identifiers.
        """
        self._pipeline = pipeline
        self._root = root

    def process_events(self, events: List[FileEvent]) -> BatchResult:
        """Process a batch of file events with per-file error isolation.

        Args:
            events: List of file events to process.

        Returns:
            BatchResult with per-batch statistics.
        """
        result = BatchResult()

        for event in events:
            try:
                if event.change_type in (ChangeType.CREATED, ChangeType.MODIFIED):
                    self._handle_index(event, result)
                elif event.change_type == ChangeType.DELETED:
                    self._handle_remove(event, result)
            except Exception as exc:
                error_msg = (
                    f"Error processing {event.path} "
                    f"({event.change_type.value}): "
                    f"{type(exc).__name__}: {exc}"
                )
                logger.error(error_msg)
                result.errors.append(error_msg)

        if result.total_processed > 0:
            logger.info(
                "Batch complete: %d indexed, %d removed, %d errors",
                result.files_indexed,
                result.files_removed,
                len(result.errors),
            )

        return result

    def _handle_index(self, event: FileEvent, result: BatchResult) -> None:
        """Index a created or modified file."""
        stats = self._pipeline.index_file(
            event.path,
            root=self._root,
            force=(event.change_type == ChangeType.MODIFIED),
        )
        if stats.files_processed > 0:
            result.files_indexed += 1
            result.chunks_created += stats.chunks_created

    def _handle_remove(self, event: FileEvent, result: BatchResult) -> None:
        """Remove a deleted file from the index."""
        rel_path = (
            str(event.path.relative_to(self._root))
            if self._root
            else str(event.path)
        )
        self._pipeline.remove_file(rel_path)
        result.files_removed += 1
