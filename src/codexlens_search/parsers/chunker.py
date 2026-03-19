"""AST-aware chunking: split source code at symbol boundaries."""
from __future__ import annotations

import logging

from codexlens_search.parsers.parser import ASTParser
from codexlens_search.parsers.symbols import extract_symbols

logger = logging.getLogger(__name__)


def chunk_by_ast(
    source: str,
    path: str,
    language: str,
    max_chars: int = 800,
    overlap: int = 100,
) -> list[tuple[str, str, int, int]]:
    """Chunk *source* at AST symbol boundaries.

    Returns a list of ``(chunk_text, path, start_line, end_line)`` tuples
    compatible with ``_chunk_code()`` output format. Line numbers are
    1-based.

    Strategy:
    1. Parse source with tree-sitter and extract top-level symbols.
    2. Build segments from symbol ranges, merging inter-symbol gaps
       (imports, blank lines) into the following symbol.
    3. Merge small adjacent segments up to *max_chars*.
    4. Sub-chunk oversized segments via simple line-based splitting.

    Returns an empty list if parsing fails or no symbols are found,
    allowing the caller to fall through to the next chunking strategy.
    """
    parser = ASTParser.get_instance()
    tree = parser.parse(source.encode("utf-8", errors="replace"), language)
    if tree is None:
        return []

    symbols = extract_symbols(tree, language)
    if not symbols:
        return []

    lines = source.splitlines(keepends=True)
    total_lines = len(lines)
    if total_lines == 0:
        return []

    # Sort symbols by start_line (1-based)
    symbols = sorted(symbols, key=lambda s: s.start_line)

    # Build segments: each segment is (start_line, end_line) 1-based inclusive
    # Merge inter-symbol gaps into the following symbol
    segments: list[tuple[int, int]] = []
    prev_end = 1  # track end of previous segment (1-based)

    for sym in symbols:
        seg_start = min(prev_end, sym.start_line)
        seg_end = sym.end_line
        segments.append((seg_start, seg_end))
        prev_end = seg_end + 1

    # Append trailing lines (after last symbol) to last segment
    if prev_end <= total_lines and segments:
        last_start, last_end = segments[-1]
        segments[-1] = (last_start, total_lines)
    elif prev_end <= total_lines:
        segments.append((prev_end, total_lines))

    # Merge small adjacent segments up to max_chars
    merged: list[tuple[int, int]] = []
    if not segments:
        return []

    cur_start, cur_end = segments[0]
    cur_len = sum(len(lines[i]) for i in range(cur_start - 1, min(cur_end, total_lines)))

    for seg_start, seg_end in segments[1:]:
        seg_len = sum(len(lines[i]) for i in range(seg_start - 1, min(seg_end, total_lines)))
        if cur_len + seg_len <= max_chars:
            cur_end = seg_end
            cur_len += seg_len
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = seg_start, seg_end
            cur_len = seg_len
    merged.append((cur_start, cur_end))

    # Build final chunks
    chunks: list[tuple[str, str, int, int]] = []
    for seg_start, seg_end in merged:
        # Convert to 0-based indices for slicing
        text = "".join(lines[seg_start - 1 : seg_end])
        if len(text) > max_chars:
            # Sub-chunk oversized segments with simple line splitting
            sub_chunks = _sub_chunk_lines(lines, seg_start, seg_end, path, max_chars, overlap)
            chunks.extend(sub_chunks)
        else:
            if text.strip():
                chunks.append((text, path, seg_start, seg_end))

    return chunks


def _sub_chunk_lines(
    lines: list[str],
    seg_start: int,
    seg_end: int,
    path: str,
    max_chars: int,
    overlap: int,
) -> list[tuple[str, str, int, int]]:
    """Split an oversized segment into sub-chunks by lines.

    Line numbers are 1-based. Applies character-based overlap.
    """
    chunks: list[tuple[str, str, int, int]] = []
    current: list[str] = []
    current_len = 0
    chunk_start = seg_start

    for line_idx in range(seg_start - 1, seg_end):
        line = lines[line_idx]
        if current_len + len(line) > max_chars and current:
            text = "".join(current)
            end_line = seg_start + len(current) - 1 + (chunk_start - seg_start)
            chunks.append((text, path, chunk_start, end_line))
            # Overlap: keep tail
            tail = text[-overlap:] if overlap else ""
            tail_newlines = tail.count("\n")
            chunk_start = max(seg_start, end_line - tail_newlines + 1)
            current = [tail] if tail else []
            current_len = len(tail)
        current.append(line)
        current_len += len(line)

    if current:
        text = "".join(current)
        if text.strip():
            end_line = seg_end
            chunks.append((text, path, chunk_start, end_line))

    return chunks
