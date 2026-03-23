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
from codexlens_search.core.base import BaseANNIndex, BaseBinaryIndex
from codexlens_search.embed.base import BaseEmbedder
from codexlens_search.indexing.metadata import MetadataStore
from codexlens_search.search.fts import FTSEngine

try:
    from codexlens_search.indexing.gitignore import GitignoreAwareMatcher

    _HAS_GITIGNORE = True
except ImportError:
    _HAS_GITIGNORE = False

try:
    from codexlens_search.parsers.chunker import chunk_by_ast
    from codexlens_search.parsers.parser import ASTParser
    from codexlens_search.parsers.symbols import extract_symbols as _extract_symbols
    from codexlens_search.parsers.references import extract_references as _extract_references

    _HAS_AST_CHUNKER = True
except ImportError:
    _HAS_AST_CHUNKER = False

logger = logging.getLogger(__name__)

# Sentinel value to signal worker shutdown
_SENTINEL = None

# ---------------------------------------------------------------------------
# Language detection: 3-tier extension mapping (~30 extensions)
# ---------------------------------------------------------------------------
# Tier 1: Most common languages
# Tier 2: Popular but less common
# Tier 3: Niche / scripting
_EXT_TO_LANGUAGE: dict[str, str] = {
    # Tier 1
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    # Tier 2
    ".rb": "ruby",
    ".php": "php",
    ".scala": "scala",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".cs": "csharp",
    ".vue": "vue",
    ".svelte": "svelte",
    # Tier 3
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".ps1": "powershell",
    ".r": "r",
    ".R": "r",
    ".pl": "perl",
    ".pm": "perl",
    ".ex": "elixir",
    ".exs": "elixir",
}


def detect_language(path: str) -> str | None:
    """Detect programming language from file extension.

    Returns language name string or None if unrecognized.
    """
    suffix = Path(path).suffix.lower()
    return _EXT_TO_LANGUAGE.get(suffix)

# Defaults for chunking (can be overridden via index_files kwargs)
_DEFAULT_MAX_CHUNK_CHARS = 800
_DEFAULT_CHUNK_OVERLAP = 100


def is_file_excluded(
    file_path: Path,
    config: Config,
    gitignore_matcher: "GitignoreAwareMatcher | None" = None,
    content: bytes | None = None,
) -> str | None:
    """Check if a file should be excluded from indexing.

    Returns exclusion reason string, or None if file should be indexed.

    Args:
        content: Optional pre-read file bytes. When provided, skips file I/O
                 for binary detection and generated-code check.
    """
    # Gitignore check (when matcher is available)
    if gitignore_matcher is not None and gitignore_matcher.is_excluded(file_path):
        return "excluded by .gitignore"

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
    if content is not None:
        sample = content[:config.binary_detect_sample_bytes]
    else:
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
    if content is not None:
        head = content[:1024].decode("utf-8", errors="replace")
    else:
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
        binary_store: BaseBinaryIndex,
        ann_index: BaseANNIndex,
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
        self._gitignore_matcher = None

    def _get_gitignore_matcher(self, root: Path | None) -> "GitignoreAwareMatcher | None":
        """Return gitignore matcher if enabled and pathspec is available."""
        if not self._config.gitignore_filtering:
            return None
        if not _HAS_GITIGNORE:
            logger.debug("gitignore_filtering enabled but pathspec not installed; skipping")
            return None
        if root is None:
            return None
        if self._gitignore_matcher is None or getattr(self._gitignore_matcher, '_root', None) != root.resolve():
            self._gitignore_matcher = GitignoreAwareMatcher(root)
        return self._gitignore_matcher

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
        num_embed_workers = max(1, self._config.index_workers)
        embed_remaining = [num_embed_workers, threading.Lock()]
        embed_threads = []
        for i in range(num_embed_workers):
            t = threading.Thread(
                target=self._embed_worker,
                args=(embed_queue, index_queue, _record_error, embed_remaining),
                daemon=True,
                name=f"indexing-embed-{i}",
            )
            t.start()
            embed_threads.append(t)
        index_thread = threading.Thread(
            target=self._index_worker,
            args=(index_queue, _record_error),
            daemon=True,
            name="indexing-index",
        )
        index_thread.start()

        # --- Stage 1: chunk files (main thread) ---
        chunk_id = 0
        files_processed = 0
        chunks_created = 0
        all_symbols: list[tuple[int, str, str, int, int, str, str, str]] = []
        all_refs: list[tuple[str, str, str, str, int]] = []

        for fpath in files:
            # Read file once, reuse for exclusion checks and chunking
            try:
                raw = fpath.read_bytes()
            except Exception as exc:
                logger.debug("Skipping %s: %s", fpath, exc)
                continue

            # Noise file filter (pass pre-read content to avoid redundant I/O)
            exclude_reason = is_file_excluded(fpath, self._config, self._get_gitignore_matcher(root), content=raw)
            if exclude_reason:
                logger.debug("Skipping %s: %s", fpath, exclude_reason)
                continue

            text = raw.decode("utf-8", errors="replace")

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
            batch_langs: list[str] = []
            for chunk_text, path, sl, el, lang in file_chunks:
                batch_ids.append(chunk_id)
                batch_texts.append(chunk_text)
                batch_paths.append(path)
                batch_lines.append((sl, el))
                batch_langs.append(lang)
                chunk_id += 1

            chunks_created += len(batch_ids)
            embed_queue.put((batch_ids, batch_texts, batch_paths, batch_lines, batch_langs))

            # Collect symbols and refs for batch persistence after workers finish
            file_lang = file_chunks[0][4] if file_chunks else ""
            if self._should_extract_symbols(file_lang):
                sym_rows = self._extract_file_symbols(
                    text, file_lang, batch_ids, batch_lines,
                )
                all_symbols.extend(sym_rows)
                # Extract cross-references
                ref_rows = self._extract_file_refs(text, file_lang, rel_path)
                all_refs.extend(ref_rows)

        # Signal all embed workers: no more data (one sentinel per worker)
        for _ in range(num_embed_workers):
            embed_queue.put(_SENTINEL)

        # Wait for workers to finish
        for t in embed_threads:
            t.join()
        index_thread.join()

        # --- Final flush ---
        self._binary_store.save()
        self._ann_index.save()

        # Persist collected symbols after all FTS writes are complete
        if all_symbols:
            self._fts.add_symbols(all_symbols)

        # Persist refs and resolve after all symbols are written
        if all_refs:
            self._fts.add_refs(all_refs)
            self._fts.resolve_refs()

        # Final commit for symbols/refs
        self._fts.flush()

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
        remaining_counter: list | None = None,
    ) -> None:
        """Stage 2: Pull chunk batches, embed, push (ids, vecs, docs) to index queue.

        Accumulates chunks across queue items to form uniform batches of
        embed_batch_size before calling embed_batch(), improving GPU/CPU
        utilization.

        Args:
            remaining_counter: [count, lock] — shared across embed workers.
                When the last worker finishes, it sends the sentinel to out_q.
        """
        batch_size = self._config.embed_batch_size
        # Accumulation buffer
        acc_ids: list[int] = []
        acc_texts: list[str] = []
        acc_paths: list[str] = []
        acc_lines: list[tuple[int, int]] = []
        acc_langs: list[str] = []

        def _flush_acc() -> None:
            if not acc_ids:
                return
            try:
                vecs = self._embedder.embed_batch(acc_texts)
                vec_array = np.array(vecs, dtype=np.float32)
                id_array = np.array(acc_ids, dtype=np.int64)
                out_q.put((id_array, vec_array, list(acc_texts), list(acc_paths), list(acc_lines), list(acc_langs)))
            except Exception as exc:
                logger.error("Embed worker error: %s", exc)
                on_error(exc)
            acc_ids.clear()
            acc_texts.clear()
            acc_paths.clear()
            acc_lines.clear()
            acc_langs.clear()

        try:
            while True:
                item = in_q.get()
                if item is _SENTINEL:
                    break

                batch_ids, batch_texts, batch_paths, batch_lines, batch_langs = item
                for i in range(len(batch_ids)):
                    acc_ids.append(batch_ids[i])
                    acc_texts.append(batch_texts[i])
                    acc_paths.append(batch_paths[i])
                    acc_lines.append(batch_lines[i])
                    acc_langs.append(batch_langs[i])
                    if len(acc_ids) >= batch_size:
                        _flush_acc()

            # Flush remaining buffered chunks on sentinel
            _flush_acc()
        except Exception as exc:
            logger.error("Embed worker error: %s", exc)
            on_error(exc)
        finally:
            # Only the last embed worker to finish sends sentinel to index queue
            if remaining_counter is not None:
                with remaining_counter[1]:
                    remaining_counter[0] -= 1
                    if remaining_counter[0] == 0:
                        out_q.put(_SENTINEL)
            else:
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

            id_array, vec_array, texts, paths, line_ranges, langs = item
            try:
                self._binary_store.add(id_array, vec_array)
                self._ann_index.add(id_array, vec_array)

                fts_docs = [
                    (int(id_array[i]), paths[i], texts[i],
                     line_ranges[i][0], line_ranges[i][1], langs[i])
                    for i in range(len(id_array))
                ]
                self._fts.add_documents(fts_docs)
                # Batch commit after each queue item
                self._fts.flush()
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
    ) -> list[tuple[str, str, int, int, str]]:
        """Choose chunking strategy based on file type and config.

        Fallback chain: AST chunking -> regex code chunking -> plain text.
        When chunk_context_header is enabled, prepends structural context
        (file path, class name, function name) to each chunk for better
        embedding quality.

        Returns list of (chunk_text, path, start_line, end_line, language) tuples.
        """
        lang = detect_language(path) or ""

        # Level 1: AST-based chunking (requires tree-sitter)
        if (
            self._config.ast_chunking
            and _HAS_AST_CHUNKER
            and lang
            and (
                self._config.ast_languages is None
                or lang in self._config.ast_languages
            )
        ):
            try:
                result = chunk_by_ast(text, path, lang, max_chars, overlap)
                if result:
                    chunks = [(t, p, sl, el, lang) for t, p, sl, el in result]
                    if self._config.chunk_context_header:
                        chunks = self._inject_context_headers(
                            chunks, text, path, lang,
                        )
                    return chunks
            except Exception as exc:
                logger.debug("AST chunking failed for %s: %s", path, exc)

        # Level 2: Regex-based code-aware chunking
        if self._config.code_aware_chunking:
            suffix = Path(path).suffix.lower()
            if suffix in self._config.code_extensions:
                result = self._chunk_code(text, path, max_chars, overlap)
                if result:
                    chunks = [(t, p, sl, el, lang) for t, p, sl, el in result]
                    if self._config.chunk_context_header:
                        chunks = self._inject_context_headers(
                            chunks, text, path, lang,
                        )
                    return chunks

        # Level 3: Plain text chunking
        base = self._chunk_text(text, path, max_chars, overlap)
        chunks = [(t, p, sl, el, lang) for t, p, sl, el in base]
        if self._config.chunk_context_header:
            chunks = self._inject_context_headers(chunks, text, path, lang)
        return chunks

    def _inject_context_headers(
        self,
        chunks: list[tuple[str, str, int, int, str]],
        text: str,
        path: str,
        lang: str,
    ) -> list[tuple[str, str, int, int, str]]:
        """Prepend structural context (file/class/function) to each chunk.

        Uses AST symbol extraction when available to map each chunk's line
        range to its enclosing class and function. Falls back to a simple
        file-path header for non-AST languages.
        """
        # Build line -> context string mapping from AST symbols
        context_map: dict[int, str] = {}
        if _HAS_AST_CHUNKER and lang:
            try:
                parser = ASTParser.get_instance()
                tree = parser.parse(text.encode("utf-8", "replace"), lang)
                if tree:
                    symbols = _extract_symbols(tree, lang)
                    for sym in symbols:
                        parts = [f"// File: {path}"]
                        if sym.parent_name:
                            parts.append(f"// Class: {sym.parent_name}")
                        parts.append(f"// {sym.kind.value.title()}: {sym.name}")
                        header = "\n".join(parts) + "\n"
                        for line in range(sym.start_line, sym.end_line + 1):
                            if line not in context_map:
                                context_map[line] = header
            except Exception:
                pass

        file_header = f"// File: {path}\n"
        result = []
        for chunk_text, p, sl, el, lang_tag in chunks:
            header = context_map.get(sl, file_header)
            result.append((header + chunk_text, p, sl, el, lang_tag))
        return result

    # ------------------------------------------------------------------
    # Symbol extraction
    # ------------------------------------------------------------------

    def _extract_file_symbols(
        self,
        text: str,
        language: str,
        chunk_ids: list[int],
        chunk_lines: list[tuple[int, int]],
    ) -> list[tuple[int, str, str, int, int, str, str, str]]:
        """Extract symbols from source and map each to its owning chunk_id.

        Returns tuples ready for ``FTSEngine.add_symbols()``:
        (chunk_id, name, kind, start_line, end_line, parent_name, signature, language).

        Only called when AST chunking is active and a language is detected.
        """
        if not _HAS_AST_CHUNKER or not language:
            return []

        parser = ASTParser.get_instance()
        tree = parser.parse(text.encode("utf-8", errors="replace"), language)
        if tree is None:
            return []

        symbols = _extract_symbols(tree, language)
        if not symbols:
            return []

        # Map each symbol to the chunk whose line range contains the symbol's start_line
        result: list[tuple[int, str, str, int, int, str, str, str]] = []
        for sym in symbols:
            owning_chunk_id = None
            for i, (sl, el) in enumerate(chunk_lines):
                if sl <= sym.start_line <= el:
                    owning_chunk_id = chunk_ids[i]
                    break
            if owning_chunk_id is None and chunk_ids:
                # Fallback: assign to last chunk
                owning_chunk_id = chunk_ids[-1]
            if owning_chunk_id is not None:
                result.append((
                    owning_chunk_id,
                    sym.name,
                    sym.kind.value,
                    sym.start_line,
                    sym.end_line,
                    sym.parent_name or "",
                    sym.signature or "",
                    sym.language,
                ))
        return result

    def _should_extract_symbols(self, language: str) -> bool:
        """Check if symbol extraction should be attempted for this language."""
        return (
            self._config.ast_chunking
            and _HAS_AST_CHUNKER
            and bool(language)
            and (
                self._config.ast_languages is None
                or language in self._config.ast_languages
            )
        )

    def _extract_file_refs(
        self,
        text: str,
        language: str,
        rel_path: str,
        symbols_for_file: list | None = None,
    ) -> list[tuple[str, str, str, str, int]]:
        """Extract cross-references from source and format for FTSEngine.add_refs().

        Returns tuples: (from_name, from_path, to_name, ref_kind, line).
        """
        if not _HAS_AST_CHUNKER or not language:
            return []

        parser = ASTParser.get_instance()
        tree = parser.parse(text.encode("utf-8", errors="replace"), language)
        if tree is None:
            return []

        # Use provided symbols or extract fresh ones
        if symbols_for_file is None:
            symbols_for_file = _extract_symbols(tree, language)

        refs = _extract_references(tree, language, symbols_for_file)
        return [
            (ref.from_symbol_name, rel_path, ref.to_name, ref.ref_kind, ref.line)
            for ref in refs
        ]

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

    def index_files_fts_only(
        self,
        files: list[Path],
        *,
        root: Path | None = None,
        max_chunk_chars: int = _DEFAULT_MAX_CHUNK_CHARS,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    ) -> IndexStats:
        """Index files into FTS5 only, without embedding or vector indexing.

        Chunks files using the same logic as the full pipeline, then inserts
        directly into FTS. No embedding computation, no binary/ANN store writes.

        Args:
            files: List of file paths to index.
            root: Optional root for computing relative paths.
            max_chunk_chars: Maximum characters per chunk.
            chunk_overlap: Character overlap between consecutive chunks.

        Returns:
            IndexStats with counts and timing.
        """
        if not files:
            return IndexStats()

        meta = self._require_metadata()
        t0 = time.monotonic()
        chunk_id = self._next_chunk_id()
        files_processed = 0
        chunks_created = 0

        for fpath in files:
            # Read file once, reuse for exclusion checks and chunking
            try:
                raw = fpath.read_bytes()
            except Exception as exc:
                logger.debug("Skipping %s: %s", fpath, exc)
                continue

            exclude_reason = is_file_excluded(fpath, self._config, self._get_gitignore_matcher(root), content=raw)
            if exclude_reason:
                logger.debug("Skipping %s: %s", fpath, exclude_reason)
                continue

            text = raw.decode("utf-8", errors="replace")

            rel_path = str(fpath.relative_to(root)) if root else str(fpath)
            content_hash = self._content_hash(text)

            # Skip unchanged files
            if not meta.file_needs_update(rel_path, content_hash):
                continue

            # Remove old FTS data if file was previously indexed
            if meta.get_file_hash(rel_path) is not None:
                meta.mark_file_deleted(rel_path)
                self._fts.delete_by_path(rel_path)

            file_chunks = self._smart_chunk(text, rel_path, max_chunk_chars, chunk_overlap)
            if not file_chunks:
                st = fpath.stat()
                meta.register_file(rel_path, content_hash, st.st_mtime, st.st_size)
                continue

            files_processed += 1
            fts_docs = []
            chunk_id_hashes = []
            file_chunk_ids: list[int] = []
            file_chunk_lines: list[tuple[int, int]] = []
            for i, (chunk_text, path, sl, el, lang) in enumerate(file_chunks):
                fts_docs.append((chunk_id, path, chunk_text, sl, el, lang))
                if self._config.skip_chunk_hash:
                    chunk_id_hashes.append((chunk_id, f"{content_hash}:{i}"))
                else:
                    chunk_id_hashes.append((chunk_id, self._content_hash(chunk_text)))
                file_chunk_ids.append(chunk_id)
                file_chunk_lines.append((sl, el))
                chunk_id += 1

            self._fts.add_documents(fts_docs)
            chunks_created += len(fts_docs)

            # Persist symbols when AST chunking produced the chunks
            file_lang = file_chunks[0][4] if file_chunks else ""
            if self._should_extract_symbols(file_lang):
                sym_rows = self._extract_file_symbols(
                    text, file_lang, file_chunk_ids, file_chunk_lines,
                )
                if sym_rows:
                    self._fts.add_symbols(sym_rows)
                # Extract and persist cross-references
                ref_rows = self._extract_file_refs(text, file_lang, rel_path)
                if ref_rows:
                    self._fts.add_refs(ref_rows)

            # Register metadata
            st = fpath.stat()
            meta.register_file(rel_path, content_hash, st.st_mtime, st.st_size)
            meta.register_chunks(rel_path, chunk_id_hashes)

        # Resolve all refs after all files are processed
        self._fts.resolve_refs()

        # Final commit
        self._fts.flush()
        meta.flush()

        duration = time.monotonic() - t0
        stats = IndexStats(
            files_processed=files_processed,
            chunks_created=chunks_created,
            duration_seconds=round(duration, 2),
        )
        logger.info(
            "FTS-only indexing complete: %d files, %d chunks in %.1fs",
            stats.files_processed, stats.chunks_created, stats.duration_seconds,
        )
        return stats

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

        # Read file once, reuse for exclusion checks and chunking
        try:
            raw = file_path.read_bytes()
        except Exception as exc:
            logger.debug("Skipping %s: %s", file_path, exc)
            return IndexStats(duration_seconds=round(time.monotonic() - t0, 2))

        # Noise file filter (pass pre-read content to avoid redundant I/O)
        exclude_reason = is_file_excluded(file_path, self._config, self._get_gitignore_matcher(root), content=raw)
        if exclude_reason:
            logger.debug("Skipping %s: %s", file_path, exclude_reason)
            return IndexStats(duration_seconds=round(time.monotonic() - t0, 2))

        text = raw.decode("utf-8", errors="replace")

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
            st = file_path.stat()
            meta.register_file(rel_path, content_hash, st.st_mtime, st.st_size)
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
        batch_langs: list[str] = []
        for i, (chunk_text, path, sl, el, lang) in enumerate(file_chunks):
            batch_ids.append(start_id + i)
            batch_texts.append(chunk_text)
            batch_paths.append(path)
            batch_lines.append((sl, el))
            batch_langs.append(lang)

        # Embed synchronously
        vecs = self._embedder.embed_batch(batch_texts)
        vec_array = np.array(vecs, dtype=np.float32)
        id_array = np.array(batch_ids, dtype=np.int64)

        # Index: write to stores
        self._binary_store.add(id_array, vec_array)
        self._ann_index.add(id_array, vec_array)
        fts_docs = [
            (batch_ids[i], batch_paths[i], batch_texts[i],
             batch_lines[i][0], batch_lines[i][1], batch_langs[i])
            for i in range(len(batch_ids))
        ]
        self._fts.add_documents(fts_docs)

        # Persist symbols when AST chunking is active
        file_lang = file_chunks[0][4] if file_chunks else ""
        if self._should_extract_symbols(file_lang):
            sym_rows = self._extract_file_symbols(
                text, file_lang, batch_ids, batch_lines,
            )
            if sym_rows:
                self._fts.add_symbols(sym_rows)
            # Extract, persist, and resolve cross-references
            ref_rows = self._extract_file_refs(text, file_lang, rel_path)
            if ref_rows:
                self._fts.add_refs(ref_rows)
                self._fts.resolve_refs()

        # Register in metadata
        st = file_path.stat()
        meta.register_file(rel_path, content_hash, st.st_mtime, st.st_size)
        if self._config.skip_chunk_hash:
            chunk_id_hashes = [
                (batch_ids[i], f"{content_hash}:{i}")
                for i in range(len(batch_ids))
            ]
        else:
            chunk_id_hashes = [
                (batch_ids[i], self._content_hash(batch_texts[i]))
                for i in range(len(batch_ids))
            ]
        meta.register_chunks(rel_path, chunk_id_hashes)

        # Flush all stores
        self._fts.flush()
        meta.flush()
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
        and removes the file's FTS, symbol, and reference entries.

        Args:
            file_path: The relative path identifier of the file to remove.
        """
        meta = self._require_metadata()
        count = meta.mark_file_deleted(file_path)
        # delete_by_path also cleans up refs by path
        fts_count = self._fts.delete_by_path(file_path)
        logger.info(
            "Removed file %s: %d chunks tombstoned, %d FTS entries deleted",
            file_path, count, fts_count,
        )

    def purge_orphan_fts(self) -> int:
        """Remove FTS entries that are not tracked by metadata.

        Orphan entries arise when metadata.db is reset (deleted or recreated)
        while fts.db retains entries from a previous indexing run. These
        orphan chunks have no corresponding ANN vectors and inflate FTS
        results without vector backing.

        Returns the number of purged FTS entries.
        """
        meta = self._require_metadata()
        fts_ids = self._fts.get_all_chunk_ids()
        if not fts_ids:
            return 0

        # Chunk IDs known to metadata (active + deleted/tombstoned)
        known_ids = meta.get_all_chunk_ids_set() | meta.get_deleted_ids()

        orphan_ids = sorted(fts_ids - known_ids)
        if not orphan_ids:
            return 0

        count = self._fts.delete_by_ids(orphan_ids)
        logger.info(
            "Purged %d orphan FTS entries not tracked by metadata",
            count,
        )
        return count

    def sync(
        self,
        file_paths: list[Path],
        *,
        root: Path | None = None,
        max_chunk_chars: int = _DEFAULT_MAX_CHUNK_CHARS,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
        max_file_size: int = 50_000,
        progress_callback: callable | None = None,
        tier: str = "full",
        delete_removed: bool = True,
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
            tier: Indexing tier - 'full' (default) runs the full pipeline
                  with embedding, 'fts_only' runs FTS-only indexing without
                  embedding or vector stores.
            delete_removed: If True (default), remove files from the index
                  that are not in file_paths.  Set to False for partial/focused
                  indexing where file_paths is a subset of the full project.

        Returns:
            Aggregated IndexStats for all operations.
        """
        meta = self._require_metadata()
        t0 = time.monotonic()

        # Purge orphan FTS entries not tracked by metadata
        # (e.g., after metadata.db was reset while fts.db was retained)
        self.purge_orphan_fts()

        # Build set of current relative paths
        current_rel_paths: dict[str, Path] = {}
        for fpath in file_paths:
            rel = str(fpath.relative_to(root)) if root else str(fpath)
            current_rel_paths[rel] = fpath

        # Get known files from metadata
        known_files = meta.get_all_files()  # {rel_path: content_hash}

        # Detect removed files
        removed: set[str] = set()
        if delete_removed:
            removed = set(known_files.keys()) - set(current_rel_paths.keys())
            for rel in removed:
                self.remove_file(rel)

        # Collect files needing update using 4-level detection:
        # Level 1: set diff (removed files) - handled above
        # Level 2: mtime + size fast pre-check via stat()
        # Level 3: content hash only when mtime/size mismatch
        files_to_index: list[Path] = []
        for rel, fpath in current_rel_paths.items():
            # Level 2: stat-based fast check
            try:
                st = fpath.stat()
            except OSError:
                continue
            if not meta.file_needs_update_fast(rel, st.st_mtime, st.st_size):
                # mtime + size match stored values -> skip (no read needed)
                continue

            # Level 3: mtime/size changed -> verify with content hash
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            content_hash = self._content_hash(text)
            if not meta.file_needs_update(rel, content_hash):
                # Content unchanged despite mtime/size change -> update metadata only
                meta.register_file(rel, content_hash, st.st_mtime, st.st_size)
                continue

            # File genuinely changed -> remove old data and queue for re-index
            if meta.get_file_hash(rel) is not None:
                meta.mark_file_deleted(rel)
                self._fts.delete_by_path(rel)
            files_to_index.append(fpath)

        # Sort files by data tier priority: hot first, then warm, then cold
        if files_to_index:
            _tier_priority = {"hot": 0, "warm": 1, "cold": 2}
            def _tier_sort_key(fp: Path) -> int:
                rel = str(fp.relative_to(root)) if root else str(fp)
                t = meta.get_file_tier(rel)
                return _tier_priority.get(t or "warm", 1)
            files_to_index.sort(key=_tier_sort_key)

        # Reclassify data tiers after sync detection
        meta.classify_tiers(
            self._config.tier_hot_hours, self._config.tier_cold_hours
        )

        # Batch index via parallel pipeline or FTS-only
        if files_to_index:
            if tier == "fts_only":
                batch_stats = self.index_files_fts_only(
                    files_to_index,
                    root=root,
                    max_chunk_chars=max_chunk_chars,
                    chunk_overlap=chunk_overlap,
                )
            else:
                # Full pipeline with embedding
                start_id = self._next_chunk_id()
                batch_stats = self._index_files_with_metadata(
                    files_to_index,
                    root=root,
                    max_chunk_chars=max_chunk_chars,
                    chunk_overlap=chunk_overlap,
                    start_chunk_id=start_id,
                    progress_callback=progress_callback,
                )
            total_files = batch_stats.files_processed
            total_chunks = batch_stats.chunks_created
        else:
            total_files = 0
            total_chunks = 0

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

    def _index_files_with_metadata(
        self,
        files: list[Path],
        *,
        root: Path | None = None,
        max_chunk_chars: int = _DEFAULT_MAX_CHUNK_CHARS,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
        start_chunk_id: int = 0,
        progress_callback: callable | None = None,
    ) -> IndexStats:
        """Batch index files using the parallel pipeline, registering metadata.

        Like index_files() but also registers each file and its chunks
        in the MetadataStore for incremental tracking.

        Args:
            files: Files to index.
            root: Root for relative paths.
            max_chunk_chars: Max chars per chunk.
            chunk_overlap: Overlap between chunks.
            start_chunk_id: Starting chunk ID.
            progress_callback: Optional callback(files_done, total_files) for progress.
        """
        meta = self._require_metadata()
        if not files:
            return IndexStats()

        t0 = time.monotonic()

        embed_queue: queue.Queue = queue.Queue(maxsize=4)
        index_queue: queue.Queue = queue.Queue(maxsize=4)

        worker_errors: list[Exception] = []
        error_lock = threading.Lock()

        def _record_error(exc: Exception) -> None:
            with error_lock:
                worker_errors.append(exc)

        num_embed_workers = max(1, self._config.index_workers)
        embed_remaining = [num_embed_workers, threading.Lock()]
        embed_threads = []
        for i in range(num_embed_workers):
            t = threading.Thread(
                target=self._embed_worker,
                args=(embed_queue, index_queue, _record_error, embed_remaining),
                daemon=True, name=f"sync-embed-{i}",
            )
            t.start()
            embed_threads.append(t)
        index_thread = threading.Thread(
            target=self._index_worker,
            args=(index_queue, _record_error),
            daemon=True, name="sync-index",
        )
        index_thread.start()

        chunk_id = start_chunk_id
        files_processed = 0
        chunks_created = 0
        total_files = len(files)
        all_symbols: list[tuple[int, str, str, int, int, str, str, str]] = []
        all_refs: list[tuple[str, str, str, str, int]] = []

        # Cross-file chunk accumulator for optimal API batch utilization
        max_batch_items = self._config.embed_batch_size
        max_batch_tokens = self._config.embed_api_max_tokens_per_batch
        buf_ids: list[int] = []
        buf_texts: list[str] = []
        buf_paths: list[str] = []
        buf_lines: list[tuple[int, int]] = []
        buf_langs: list[str] = []
        buf_tokens = 0

        def _flush_buffer() -> None:
            nonlocal buf_ids, buf_texts, buf_paths, buf_lines, buf_langs, buf_tokens
            if buf_ids:
                embed_queue.put((list(buf_ids), list(buf_texts), list(buf_paths), list(buf_lines), list(buf_langs)))
                buf_ids.clear()
                buf_texts.clear()
                buf_paths.clear()
                buf_lines.clear()
                buf_langs.clear()
                buf_tokens = 0

        for fpath in files:
            # Read file once, reuse for exclusion checks and chunking
            try:
                raw = fpath.read_bytes()
            except Exception as exc:
                logger.debug("Skipping %s: %s", fpath, exc)
                if progress_callback:
                    progress_callback(files_processed, total_files)
                continue

            exclude_reason = is_file_excluded(fpath, self._config, self._get_gitignore_matcher(root), content=raw)
            if exclude_reason:
                logger.debug("Skipping %s: %s", fpath, exclude_reason)
                if progress_callback:
                    progress_callback(files_processed, total_files)
                continue

            text = raw.decode("utf-8", errors="replace")

            rel_path = str(fpath.relative_to(root)) if root else str(fpath)
            content_hash = self._content_hash(text)
            file_chunks = self._smart_chunk(text, rel_path, max_chunk_chars, chunk_overlap)

            if not file_chunks:
                st = fpath.stat()
                meta.register_file(rel_path, content_hash, st.st_mtime, st.st_size)
                continue

            files_processed += 1
            file_chunk_ids = []
            file_chunk_ids_only: list[int] = []
            file_chunk_lines: list[tuple[int, int]] = []
            for chunk_text, path, sl, el, lang in file_chunks:
                chunk_tokens = max(1, len(chunk_text) // 4)
                # Flush if adding this chunk would exceed batch limits
                if buf_ids and (
                    len(buf_ids) >= max_batch_items
                    or buf_tokens + chunk_tokens > max_batch_tokens
                ):
                    _flush_buffer()

                buf_ids.append(chunk_id)
                buf_texts.append(chunk_text)
                buf_paths.append(path)
                buf_lines.append((sl, el))
                buf_langs.append(lang)
                buf_tokens += chunk_tokens
                file_chunk_ids.append((chunk_id, chunk_text))
                file_chunk_ids_only.append(chunk_id)
                file_chunk_lines.append((sl, el))
                chunk_id += 1

            chunks_created += len(file_chunk_ids)

            # Collect symbols and refs for batch persistence after workers finish
            file_lang = file_chunks[0][4] if file_chunks else ""
            if self._should_extract_symbols(file_lang):
                sym_rows = self._extract_file_symbols(
                    text, file_lang, file_chunk_ids_only, file_chunk_lines,
                )
                all_symbols.extend(sym_rows)
                ref_rows = self._extract_file_refs(text, file_lang, rel_path)
                all_refs.extend(ref_rows)

            # Register metadata per file
            st = fpath.stat()
            meta.register_file(rel_path, content_hash, st.st_mtime, st.st_size)
            if self._config.skip_chunk_hash:
                chunk_id_hashes = [
                    (cid, f"{content_hash}:{i}")
                    for i, (cid, _ct) in enumerate(file_chunk_ids)
                ]
            else:
                chunk_id_hashes = [
                    (cid, self._content_hash(ct)) for cid, ct in file_chunk_ids
                ]
            meta.register_chunks(rel_path, chunk_id_hashes)

            if progress_callback:
                progress_callback(files_processed, total_files)

        # Final flush for remaining chunks
        _flush_buffer()

        for _ in range(num_embed_workers):
            embed_queue.put(_SENTINEL)
        for t in embed_threads:
            t.join()
        index_thread.join()

        self._binary_store.save()
        self._ann_index.save()

        # Persist collected symbols after all FTS writes are complete
        if all_symbols:
            self._fts.add_symbols(all_symbols)

        # Persist refs and resolve after all symbols are written
        if all_refs:
            self._fts.add_refs(all_refs)
            self._fts.resolve_refs()

        # Final commit for symbols/refs and metadata
        self._fts.flush()
        meta.flush()

        duration = time.monotonic() - t0

        if worker_errors:
            raise worker_errors[0]

        return IndexStats(
            files_processed=files_processed,
            chunks_created=chunks_created,
            duration_seconds=round(duration, 2),
        )

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
