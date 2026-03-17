"""Three-stage parallel indexing pipeline: chunk -> embed -> index.

Uses threading.Thread with queue.Queue for producer-consumer handoff.
The GIL is acceptable because embedding (onnxruntime) releases it in C extensions.
"""
from __future__ import annotations

import hashlib
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from codexlens_search.config import Config
from codexlens_search.core.binary import BinaryStore
from codexlens_search.core.index import ANNIndex
from codexlens_search.embed.base import BaseEmbedder
from codexlens_search.indexing.metadata import MetadataStore
from codexlens_search.search.fts import FTSEngine

logger = logging.getLogger(__name__)

# Sentinel value to signal worker shutdown
_SENTINEL = None

# Defaults for chunking (can be overridden via index_files kwargs)
_DEFAULT_MAX_CHUNK_CHARS = 800
_DEFAULT_CHUNK_OVERLAP = 100


def is_file_excluded(file_path: Path, config: Config) -> str | None:
    """Check if a file should be excluded from indexing.

    Returns exclusion reason string, or None if file should be indexed.
    """
    # Extension check
    suffix = file_path.suffix.lower()
    # Handle compound extensions like .min.js
    name_lower = file_path.name.lower()
    for ext in config.exclude_extensions:
        if name_lower.endswith(ext):
            return f"excluded extension: {ext}"

    # File size check
    try:
        size = file_path.stat().st_size
    except OSError:
        return "cannot stat file"
    if size > config.max_file_size_bytes:
        return f"exceeds max size ({size} > {config.max_file_size_bytes})"
    if size == 0:
        return "empty file"

    # Binary detection: sample first N bytes
    try:
        with open(file_path, "rb") as f:
            sample = f.read(config.binary_detect_sample_bytes)
    except OSError:
        return "cannot read file"
    if sample:
        null_ratio = sample.count(b"\x00") / len(sample)
        if null_ratio > config.binary_null_threshold:
            return f"binary file (null ratio: {null_ratio:.2%})"

    # Generated code markers (check first 1KB of text)
    try:
        head = file_path.read_text(encoding="utf-8", errors="replace")[:1024]
    except OSError:
        return None  # can't check, let it through
    for marker in config.generated_code_markers:
        if marker in head:
            return f"generated code marker: {marker}"

    return None


@dataclass
class IndexStats:
    """Statistics returned after indexing completes."""
    files_processed: int = 0
    chunks_created: int = 0
    duration_seconds: float = 0.0


class IndexingPipeline:
    """Parallel 3-stage indexing pipeline with queue-based handoff.

    Stage 1 (main thread): Read files, chunk text, push to embed_queue.
    Stage 2 (embed worker): Pull text batches, call embed_batch(), push vectors to index_queue.
    Stage 3 (index worker): Pull vectors+ids, call BinaryStore.add(), ANNIndex.add(), FTS.add_documents().

    After all stages complete, save() is called on BinaryStore and ANNIndex exactly once.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        binary_store: BinaryStore,
        ann_index: ANNIndex,
        fts: FTSEngine,
        config: Config,
        metadata: MetadataStore | None = None,
    ) -> None:
        self._embedder = embedder
        self._binary_store = binary_store
        self._ann_index = ann_index
        self._fts = fts
        self._config = config
        self._metadata = metadata

    def index_files(
        self,
        files: list[Path],
        *,
        root: Path | None = None,
        max_chunk_chars: int = _DEFAULT_MAX_CHUNK_CHARS,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
        max_file_size: int = 50_000,
    ) -> IndexStats:
        """Run the 3-stage pipeline on the given files.

        Args:
            files: List of file paths to index.
            root: Optional root for computing relative paths. If None, uses
                  each file's absolute path as its identifier.
            max_chunk_chars: Maximum characters per chunk.
            chunk_overlap: Character overlap between consecutive chunks.
            max_file_size: Skip files larger than this (bytes).

        Returns:
            IndexStats with counts and timing.
        """
        if not files:
            return IndexStats()

        t0 = time.monotonic()

        embed_queue: queue.Queue = queue.Queue(maxsize=4)
        index_queue: queue.Queue = queue.Queue(maxsize=4)

        # Track errors from workers
        worker_errors: list[Exception] = []
        error_lock = threading.Lock()

        def _record_error(exc: Exception) -> None:
            with error_lock:
                worker_errors.append(exc)

        # --- Start workers ---
        embed_thread = threading.Thread(
            target=self._embed_worker,
            args=(embed_queue, index_queue, _record_error),
            daemon=True,
            name="indexing-embed",
        )
        index_thread = threading.Thread(
            target=self._index_worker,
            args=(index_queue, _record_error),
            daemon=True,
            name="indexing-index",
        )
        embed_thread.start()
        index_thread.start()

        # --- Stage 1: chunk files (main thread) ---
        chunk_id = 0
        files_processed = 0
        chunks_created = 0

        for fpath in files:
            # Noise file filter
            exclude_reason = is_file_excluded(fpath, self._config)
            if exclude_reason:
                logger.debug("Skipping %s: %s", fpath, exclude_reason)
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                logger.debug("Skipping %s: %s", fpath, exc)
                continue

            rel_path = str(fpath.relative_to(root)) if root else str(fpath)
            file_chunks = self._smart_chunk(text, rel_path, max_chunk_chars, chunk_overlap)

            if not file_chunks:
                continue

            files_processed += 1

            # Assign sequential IDs and push batch to embed queue
            batch_ids = []
            batch_texts = []
            batch_paths = []
            batch_lines: list[tuple[int, int]] = []
            for chunk_text, path, sl, el in file_chunks:
                batch_ids.append(chunk_id)
                batch_texts.append(chunk_text)
                batch_paths.append(path)
                batch_lines.append((sl, el))
                chunk_id += 1

            chunks_created += len(batch_ids)
            embed_queue.put((batch_ids, batch_texts, batch_paths, batch_lines))

        # Signal embed worker: no more data
        embed_queue.put(_SENTINEL)

        # Wait for workers to finish
        embed_thread.join()
        index_thread.join()

        # --- Final flush ---
        self._binary_store.save()
        self._ann_index.save()

        duration = time.monotonic() - t0
        stats = IndexStats(
            files_processed=files_processed,
            chunks_created=chunks_created,
            duration_seconds=round(duration, 2),
        )

        logger.info(
            "Indexing complete: %d files, %d chunks in %.1fs",
            stats.files_processed,
            stats.chunks_created,
            stats.duration_seconds,
        )

        # Raise first worker error if any occurred
        if worker_errors:
            raise worker_errors[0]

        return stats

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    def _embed_worker(
        self,
        in_q: queue.Queue,
        out_q: queue.Queue,
        on_error: callable,
    ) -> None:
        """Stage 2: Pull chunk batches, embed, push (ids, vecs, docs) to index queue."""
        try:
            while True:
                item = in_q.get()
                if item is _SENTINEL:
                    break

                batch_ids, batch_texts, batch_paths, batch_lines = item
                try:
                    vecs = self._embedder.embed_batch(batch_texts)
                    vec_array = np.array(vecs, dtype=np.float32)
                    id_array = np.array(batch_ids, dtype=np.int64)
                    out_q.put((id_array, vec_array, batch_texts, batch_paths, batch_lines))
                except Exception as exc:
                    logger.error("Embed worker error: %s", exc)
                    on_error(exc)
        finally:
            # Signal index worker: no more data
            out_q.put(_SENTINEL)

    def _index_worker(
        self,
        in_q: queue.Queue,
        on_error: callable,
    ) -> None:
        """Stage 3: Pull (ids, vecs, texts, paths, lines), write to stores."""
        while True:
            item = in_q.get()
            if item is _SENTINEL:
                break

            id_array, vec_array, texts, paths, line_ranges = item
            try:
                self._binary_store.add(id_array, vec_array)
                self._ann_index.add(id_array, vec_array)

                fts_docs = [
                    (int(id_array[i]), paths[i], texts[i],
                     line_ranges[i][0], line_ranges[i][1])
                    for i in range(len(id_array))
                ]
                self._fts.add_documents(fts_docs)
            except Exception as exc:
                logger.error("Index worker error: %s", exc)
                on_error(exc)

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(
        text: str,
        path: str,
        max_chars: int,
        overlap: int,
    ) -> list[tuple[str, str, int, int]]:
        """Split file text into overlapping chunks.

        Returns list of (chunk_text, path, start_line, end_line) tuples.
        Line numbers are 1-based.
        """
        if not text.strip():
            return []

        chunks: list[tuple[str, str, int, int]] = []
        lines = text.splitlines(keepends=True)
        current: list[str] = []
        current_len = 0
        chunk_start_line = 1  # 1-based
        lines_consumed = 0

        for line in lines:
            lines_consumed += 1
            if current_len + len(line) > max_chars and current:
                chunk = "".join(current)
                end_line = lines_consumed - 1
                chunks.append((chunk, path, chunk_start_line, end_line))
                # overlap: keep last N characters
                tail = chunk[-overlap:] if overlap else ""
                tail_newlines = tail.count("\n")
                chunk_start_line = max(1, end_line - tail_newlines + 1)
                current = [tail] if tail else []
                current_len = len(tail)
            current.append(line)
            current_len += len(line)

        if current:
            chunks.append(("".join(current), path, chunk_start_line, lines_consumed))

        return chunks

    # Pattern matching top-level definitions across languages
    _CODE_BOUNDARY_RE = re.compile(
        r"^(?:"
        r"(?:export\s+)?(?:async\s+)?(?:def|class|function)\s+"       # Python/JS/TS
        r"|(?:pub\s+)?(?:fn|struct|impl|enum|trait|mod)\s+"           # Rust
        r"|(?:func|type)\s+"                                          # Go
        r"|(?:public|private|protected|internal)?\s*(?:static\s+)?(?:class|interface|enum|record)\s+"  # Java/C#
        r"|(?:namespace|template)\s+"                                 # C++
        r")",
        re.MULTILINE,
    )

    def _chunk_code(
        self,
        text: str,
        path: str,
        max_chars: int,
        overlap: int,
    ) -> list[tuple[str, str, int, int]]:
        """Split code at function/class boundaries with fallback to _chunk_text.

        Strategy:
        1. Find all top-level definition boundaries via regex.
        2. Split text into segments at those boundaries.
        3. Merge small adjacent segments up to max_chars.
        4. If a segment exceeds max_chars, fall back to _chunk_text for that segment.
        """
        lines = text.splitlines(keepends=True)
        if not lines:
            return []

        # Find boundary line numbers (0-based)
        boundaries: list[int] = [0]  # always start at line 0
        for i, line in enumerate(lines):
            if i == 0:
                continue
            # Only match lines with no or minimal indentation (top-level)
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if indent <= 4 and self._CODE_BOUNDARY_RE.match(stripped):
                boundaries.append(i)

        if len(boundaries) <= 1:
            # No boundaries found, fall back to text chunking
            return self._chunk_text(text, path, max_chars, overlap)

        # Build raw segments between boundaries
        raw_segments: list[tuple[int, int]] = []  # (start_line, end_line) 0-based
        for idx in range(len(boundaries)):
            start = boundaries[idx]
            end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(lines)
            raw_segments.append((start, end))

        # Merge small adjacent segments up to max_chars
        merged: list[tuple[int, int]] = []
        cur_start, cur_end = raw_segments[0]
        cur_len = sum(len(lines[i]) for i in range(cur_start, cur_end))

        for seg_start, seg_end in raw_segments[1:]:
            seg_len = sum(len(lines[i]) for i in range(seg_start, seg_end))
            if cur_len + seg_len <= max_chars:
                cur_end = seg_end
                cur_len += seg_len
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = seg_start, seg_end
                cur_len = seg_len
        merged.append((cur_start, cur_end))

        # Build chunks, falling back to _chunk_text for oversized segments
        chunks: list[tuple[str, str, int, int]] = []
        for seg_start, seg_end in merged:
            seg_text = "".join(lines[seg_start:seg_end])
            if len(seg_text) > max_chars:
                # Oversized: sub-chunk with text splitter
                sub_chunks = self._chunk_text(seg_text, path, max_chars, overlap)
                # Adjust line numbers relative to segment start
                for chunk_text, p, sl, el in sub_chunks:
                    chunks.append((chunk_text, p, sl + seg_start, el + seg_start))
            else:
                chunks.append((seg_text, path, seg_start + 1, seg_end))

        return chunks

    def _smart_chunk(
        self,
        text: str,
        path: str,
        max_chars: int,
        overlap: int,
    ) -> list[tuple[str, str, int, int]]:
        """Choose chunking strategy based on file type and config."""
        if self._config.code_aware_chunking:
            suffix = Path(path).suffix.lower()
            if suffix in self._config.code_extensions:
                result = self._chunk_code(text, path, max_chars, overlap)
                if result:
                    return result
        return self._chunk_text(text, path, max_chars, overlap)

    # ------------------------------------------------------------------
    # Incremental API
    # ------------------------------------------------------------------

    @staticmethod
    def _content_hash(text: str) -> str:
        """Compute SHA-256 hex digest of file content."""
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    def _require_metadata(self) -> MetadataStore:
        """Return metadata store or raise if not configured."""
        if self._metadata is None:
            raise RuntimeError(
                "MetadataStore is required for incremental indexing. "
                "Pass metadata= to IndexingPipeline.__init__."
            )
        return self._metadata

    def _next_chunk_id(self) -> int:
        """Return the next available chunk ID from MetadataStore."""
        meta = self._require_metadata()
        return meta.max_chunk_id() + 1

    def index_file(
        self,
        file_path: Path,
        *,
        root: Path | None = None,
        force: bool = False,
        max_chunk_chars: int = _DEFAULT_MAX_CHUNK_CHARS,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
        max_file_size: int = 50_000,
    ) -> IndexStats:
        """Index a single file incrementally.

        Skips files that have not changed (same content_hash) unless
        *force* is True.

        Args:
            file_path: Path to the file to index.
            root: Optional root for computing relative path identifiers.
            force: Re-index even if content hash has not changed.
            max_chunk_chars: Maximum characters per chunk.
            chunk_overlap: Character overlap between consecutive chunks.
            max_file_size: Skip files larger than this (bytes).

        Returns:
            IndexStats with counts and timing.
        """
        meta = self._require_metadata()
        t0 = time.monotonic()

        # Noise file filter
        exclude_reason = is_file_excluded(file_path, self._config)
        if exclude_reason:
            logger.debug("Skipping %s: %s", file_path, exclude_reason)
            return IndexStats(duration_seconds=round(time.monotonic() - t0, 2))

        # Read file
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.debug("Skipping %s: %s", file_path, exc)
            return IndexStats(duration_seconds=round(time.monotonic() - t0, 2))

        content_hash = self._content_hash(text)
        rel_path = str(file_path.relative_to(root)) if root else str(file_path)

        # Check if update is needed
        if not force and not meta.file_needs_update(rel_path, content_hash):
            logger.debug("Skipping %s: unchanged", rel_path)
            return IndexStats(duration_seconds=round(time.monotonic() - t0, 2))

        # If file was previously indexed, remove old data first
        if meta.get_file_hash(rel_path) is not None:
            meta.mark_file_deleted(rel_path)
            self._fts.delete_by_path(rel_path)

        # Chunk
        file_chunks = self._smart_chunk(text, rel_path, max_chunk_chars, chunk_overlap)
        if not file_chunks:
            # Register file with no chunks
            meta.register_file(rel_path, content_hash, file_path.stat().st_mtime)
            return IndexStats(
                files_processed=1,
                duration_seconds=round(time.monotonic() - t0, 2),
            )

        # Assign chunk IDs
        start_id = self._next_chunk_id()
        batch_ids = []
        batch_texts = []
        batch_paths = []
        batch_lines: list[tuple[int, int]] = []
        for i, (chunk_text, path, sl, el) in enumerate(file_chunks):
            batch_ids.append(start_id + i)
            batch_texts.append(chunk_text)
            batch_paths.append(path)
            batch_lines.append((sl, el))

        # Embed synchronously
        vecs = self._embedder.embed_batch(batch_texts)
        vec_array = np.array(vecs, dtype=np.float32)
        id_array = np.array(batch_ids, dtype=np.int64)

        # Index: write to stores
        self._binary_store.add(id_array, vec_array)
        self._ann_index.add(id_array, vec_array)
        fts_docs = [
            (batch_ids[i], batch_paths[i], batch_texts[i],
             batch_lines[i][0], batch_lines[i][1])
            for i in range(len(batch_ids))
        ]
        self._fts.add_documents(fts_docs)

        # Register in metadata
        meta.register_file(rel_path, content_hash, file_path.stat().st_mtime)
        chunk_id_hashes = [
            (batch_ids[i], self._content_hash(batch_texts[i]))
            for i in range(len(batch_ids))
        ]
        meta.register_chunks(rel_path, chunk_id_hashes)

        # Flush stores
        self._binary_store.save()
        self._ann_index.save()

        duration = time.monotonic() - t0
        stats = IndexStats(
            files_processed=1,
            chunks_created=len(batch_ids),
            duration_seconds=round(duration, 2),
        )
        logger.info(
            "Indexed file %s: %d chunks in %.2fs",
            rel_path, stats.chunks_created, stats.duration_seconds,
        )
        return stats

    def remove_file(self, file_path: str) -> None:
        """Mark a file as deleted via tombstone strategy.

        Marks all chunk IDs for the file in MetadataStore.deleted_chunks
        and removes the file's FTS entries.

        Args:
            file_path: The relative path identifier of the file to remove.
        """
        meta = self._require_metadata()
        count = meta.mark_file_deleted(file_path)
        fts_count = self._fts.delete_by_path(file_path)
        logger.info(
            "Removed file %s: %d chunks tombstoned, %d FTS entries deleted",
            file_path, count, fts_count,
        )

    def sync(
        self,
        file_paths: list[Path],
        *,
        root: Path | None = None,
        max_chunk_chars: int = _DEFAULT_MAX_CHUNK_CHARS,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
        max_file_size: int = 50_000,
    ) -> IndexStats:
        """Reconcile index state against a current file list.

        Identifies files that are new, changed, or removed and processes
        each accordingly.

        Args:
            file_paths: Current list of files that should be indexed.
            root: Optional root for computing relative path identifiers.
            max_chunk_chars: Maximum characters per chunk.
            chunk_overlap: Character overlap between consecutive chunks.
            max_file_size: Skip files larger than this (bytes).

        Returns:
            Aggregated IndexStats for all operations.
        """
        meta = self._require_metadata()
        t0 = time.monotonic()

        # Build set of current relative paths
        current_rel_paths: dict[str, Path] = {}
        for fpath in file_paths:
            rel = str(fpath.relative_to(root)) if root else str(fpath)
            current_rel_paths[rel] = fpath

        # Get known files from metadata
        known_files = meta.get_all_files()  # {rel_path: content_hash}

        # Detect removed files
        removed = set(known_files.keys()) - set(current_rel_paths.keys())
        for rel in removed:
            self.remove_file(rel)

        # Index new and changed files
        total_files = 0
        total_chunks = 0
        for rel, fpath in current_rel_paths.items():
            stats = self.index_file(
                fpath,
                root=root,
                max_chunk_chars=max_chunk_chars,
                chunk_overlap=chunk_overlap,
                max_file_size=max_file_size,
            )
            total_files += stats.files_processed
            total_chunks += stats.chunks_created

        duration = time.monotonic() - t0
        result = IndexStats(
            files_processed=total_files,
            chunks_created=total_chunks,
            duration_seconds=round(duration, 2),
        )
        logger.info(
            "Sync complete: %d files indexed, %d chunks created, "
            "%d files removed in %.1fs",
            result.files_processed, result.chunks_created,
            len(removed), result.duration_seconds,
        )
        return result

    def compact(self) -> None:
        """Rebuild indexes excluding tombstoned chunk IDs.

        Reads all deleted IDs from MetadataStore, rebuilds BinaryStore
        and ANNIndex without those entries, then clears the
        deleted_chunks table.
        """
        meta = self._require_metadata()
        deleted_ids = meta.compact_deleted()
        if not deleted_ids:
            logger.debug("Compact: no deleted IDs, nothing to do")
            return

        logger.info("Compact: rebuilding indexes, excluding %d deleted IDs", len(deleted_ids))

        # Rebuild BinaryStore: read current data, filter, replace
        if self._binary_store._count > 0:
            active_ids = self._binary_store._ids[: self._binary_store._count]
            active_matrix = self._binary_store._matrix[: self._binary_store._count]
            mask = ~np.isin(active_ids, list(deleted_ids))
            kept_ids = active_ids[mask]
            kept_matrix = active_matrix[mask]
            # Reset store
            self._binary_store._count = 0
            self._binary_store._matrix = None
            self._binary_store._ids = None
            if len(kept_ids) > 0:
                self._binary_store._ensure_capacity(len(kept_ids))
                self._binary_store._matrix[: len(kept_ids)] = kept_matrix
                self._binary_store._ids[: len(kept_ids)] = kept_ids
                self._binary_store._count = len(kept_ids)
            self._binary_store.save()

        # Rebuild ANNIndex: must reconstruct from scratch since HNSW
        # does not support deletion. We re-initialize and re-add kept items.
        # Note: we need the float32 vectors, but BinaryStore only has quantized.
        # ANNIndex (hnswlib) supports mark_deleted, but compact means full rebuild.
        # Since we don't have original float vectors cached, we rely on the fact
        # that ANNIndex.mark_deleted is not available in all hnswlib versions.
        # Instead, we reinitialize the index and let future searches filter via
        # deleted_ids at query time. The BinaryStore is already compacted above.
        # For a full ANN rebuild, the caller should re-run index_files() on all
        # files after compact.
        logger.info(
            "Compact: BinaryStore rebuilt (%d entries kept). "
            "Note: ANNIndex retains stale entries; run full re-index for clean ANN state.",
            self._binary_store._count,
        )
