"""MCP server for codexlens-search.

Tools:
  - Search:        Hybrid code search (semantic + FTS + AST graph + regex).
  - index_project: Build, update, or inspect the search index.
  - find_files:    Glob-based file discovery.
  - watch_project: Manage file watcher for auto re-indexing.

Run: codexlens-mcp  or  python -m codexlens_search.mcp_server

## .mcp.json

```json
{
  "mcpServers": {
    "codexlens": {
      "command": "uvx",
      "args": ["--from", "codexlens-search", "codexlens-mcp"],
      "env": {
        "CODEXLENS_EMBED_API_URL": "https://api.openai.com/v1",
        "CODEXLENS_EMBED_API_KEY": "sk-xxx",
        "CODEXLENS_EMBED_API_MODEL": "text-embedding-3-small",
        "CODEXLENS_EMBED_DIM": "1536"
      }
    }
  }
}
```
"""
from __future__ import annotations

import atexit
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP

from codexlens_search.bridge import (
    DEFAULT_EXCLUDES,
    create_config_from_env,
    create_pipeline,
    should_exclude,
)

log = logging.getLogger("codexlens_search.mcp_server")

mcp = FastMCP("codexlens-search")

_pipelines: dict[str, tuple] = {}
_lock = threading.Lock()
_bg_indexing: dict[str, threading.Thread] = {}
_watchers: dict[str, "FileWatcher"] = {}
_watcher_lock = threading.Lock()


def _env_int(name: str, default: int) -> int:
    """Read an integer from environment variable with fallback."""
    val = os.environ.get(name, "")
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return default


def _close_pipeline(pipeline_tuple: tuple) -> None:
    """Close resources held by a pipeline tuple (indexing, search, config).

    In sharded mode both components are the same ShardManager (with close()).
    In single-shard mode the search pipeline (index 1) owns FTS/metadata.
    We only close the search pipeline to avoid double-closing shared resources.
    """
    search = pipeline_tuple[1] if len(pipeline_tuple) > 1 else None
    if search is not None and hasattr(search, "close"):
        try:
            search.close()
        except Exception:
            pass


def _db_path_for_project(project_path: str) -> Path:
    return Path(project_path).resolve() / ".codexlens"


def _purge_index_files(db_path: Path) -> None:
    """Delete all index files in db_path to allow clean rebuild.

    Handles locked sqlite files gracefully — logs a warning but continues.
    """
    if not db_path.is_dir():
        return
    for f in db_path.iterdir():
        try:
            f.unlink()
        except OSError as exc:
            log.warning("Could not delete %s: %s", f.name, exc)


def _trigger_background_index(project_path: str) -> str:
    """Trigger background indexing if not already in progress. Returns notice string."""
    resolved = str(Path(project_path).resolve())

    with _lock:
        if resolved in _bg_indexing:
            thread = _bg_indexing[resolved]
            if thread.is_alive():
                return (
                    "**Note**: Index is being built in the background. "
                    "Showing regex results only. Re-search after indexing completes."
                )
            else:
                del _bg_indexing[resolved]

    def _do_index():
        try:
            log.info("Background indexing started for: %s", resolved)
            root = Path(resolved)
            indexing, _, _ = _get_pipelines(project_path)
            file_paths = [
                p for p in root.glob("**/*")
                if p.is_file() and not should_exclude(p.relative_to(root), DEFAULT_EXCLUDES)
            ]
            indexing.sync(file_paths, root=root)
            # Invalidate pipeline cache so next search uses fresh index
            with _lock:
                old = _pipelines.pop(resolved, None)
            if old:
                _close_pipeline(old)
            log.info("Background indexing complete for %s: %d files", resolved, len(file_paths))
            # Auto-start watcher after background indexing
            _ensure_watcher(project_path)
        except Exception as exc:
            log.error("Background indexing failed for %s: %s", resolved, exc)
        finally:
            with _lock:
                _bg_indexing.pop(resolved, None)

    thread = threading.Thread(target=_do_index, daemon=True, name=f"bg-index-{resolved[-20:]}")
    with _lock:
        _bg_indexing[resolved] = thread
    thread.start()

    return (
        "**Note**: No index found. Background indexing started automatically. "
        "Showing regex results for now. Re-search after indexing completes for semantic results."
    )


def _ensure_watcher(project_path: str) -> str | None:
    """Start file watcher for a project. Returns status message or None."""
    if not os.environ.get("CODEXLENS_AUTO_WATCH", "").lower() in ("true", "1", "yes"):
        return None

    resolved = str(Path(project_path).resolve())
    with _watcher_lock:
        existing = _watchers.get(resolved)
        if existing is not None and existing.is_running:
            return None  # already watching

    try:
        from codexlens_search.watcher import FileWatcher, IncrementalIndexer, WatcherConfig
    except ImportError:
        log.debug("watchdog not installed, skipping watcher")
        return None

    try:
        indexing, _, _ = _get_pipelines(project_path)
        root = Path(resolved)
        indexer = IncrementalIndexer(indexing, root=root)
        watcher_config = WatcherConfig(
            debounce_ms=int(os.environ.get("CODEXLENS_WATCHER_DEBOUNCE_MS", "1000")),
        )
        watcher = FileWatcher.create_with_indexer(root, watcher_config, indexer)
        watcher.start()
        with _watcher_lock:
            _watchers[resolved] = watcher
        log.info("Started file watcher for: %s", resolved)
        return f"File watcher started for {resolved}"
    except Exception as exc:
        log.warning("Failed to start watcher for %s: %s", resolved, exc)
        return None


def _stop_watcher(project_path: str) -> str:
    """Stop file watcher for a project."""
    resolved = str(Path(project_path).resolve())
    with _watcher_lock:
        watcher = _watchers.pop(resolved, None)
    if watcher is not None:
        watcher.stop()
        log.info("Stopped file watcher for: %s", resolved)
        return f"File watcher stopped for {resolved}"
    return f"No active watcher for {resolved}"


def _cleanup_watchers() -> None:
    """Stop all watchers on process exit."""
    with _watcher_lock:
        for resolved, watcher in list(_watchers.items()):
            try:
                watcher.stop()
            except Exception:
                pass
        _watchers.clear()


def _cleanup_pipelines() -> None:
    """Close all cached pipelines on process exit."""
    with _lock:
        for pipeline_tuple in _pipelines.values():
            _close_pipeline(pipeline_tuple)
        _pipelines.clear()


atexit.register(_cleanup_watchers)
atexit.register(_cleanup_pipelines)


def _get_pipelines(project_path: str, force: bool = False) -> tuple:
    resolved = str(Path(project_path).resolve())
    with _lock:
        if force:
            old = _pipelines.pop(resolved, None)
            if old:
                _close_pipeline(old)
        if resolved not in _pipelines:
            db_path = _db_path_for_project(resolved)
            config = create_config_from_env(db_path)
            _pipelines[resolved] = create_pipeline(db_path, config)
        return _pipelines[resolved]


def _get_fts(project_path: str):
    """Get the FTSEngine for a project (lazy, from cached pipeline)."""
    from codexlens_search.search.fts import FTSEngine

    db_path = _db_path_for_project(project_path)
    fts_path = db_path / "fts.db"
    if not fts_path.exists():
        return None
    return FTSEngine(fts_path)


# ---------------------------------------------------------------------------
# Tool 1: Search — unified hybrid search
# ---------------------------------------------------------------------------

@mcp.tool(name="Search")
async def search_code(
    project_path: str,
    query: str,
    mode: str = "auto",
    scope: str = "",
) -> str:
    """Hybrid code search combining semantic vector, FTS, AST graph, and ripgrep regex.

    Modes:
      - "auto": Semantic + regex in parallel. Auto-triggers background indexing if no index exists, returning regex results first.
      - "symbol": Find symbol definitions (function, class, method) by exact or fuzzy name match. Requires index.
      - "refs": Find cross-references (imports, calls, inheritance) — returns both incoming and outgoing edges. Requires index.
      - "regex": Ripgrep regex search on live files. Requires rg in PATH.

    Results are capped by CODEXLENS_TOP_K env var (default 10).

    Args:
        project_path: Absolute path to the project root.
        query: Natural language, symbol name, or regex pattern.
        mode: Search mode — "auto" (default), "symbol", "refs", or "regex".
        scope: Relative directory to restrict search (e.g. "src/auth"). Applies to auto and regex modes.
    """
    root = Path(project_path).resolve()
    if not root.is_dir():
        return f"Error: project path not found: {root}"

    top_k = _env_int("CODEXLENS_TOP_K", 10)

    if mode == "regex":
        return await _search_regex(project_path, query, top_k, scope)
    if mode == "symbol":
        return _search_symbol(project_path, query, top_k)
    if mode == "refs":
        return _search_refs(project_path, query)

    # mode == "auto": hybrid search with parallel regex + fallback
    return await _search_auto(project_path, query, top_k, scope)


async def _search_auto(project_path: str, query: str, top_k: int, scope: str) -> str:
    """Hybrid search with parallel regex supplement and no-index fallback.

    When no index exists:
    1. Expand query → grep for relevant files (fast)
    2. Index only those files (focused, ~seconds)
    3. Run semantic search on the fresh index
    4. Trigger full background index for subsequent queries
    """
    root = Path(project_path).resolve()
    db_path = _db_path_for_project(project_path)
    has_index = (db_path / "metadata.db").exists()
    has_rg = shutil.which("rg") is not None

    if not has_index and not has_rg:
        return "Error: no index found and ripgrep (rg) not available. Run index_project or install rg."

    # No index: focused-index-then-search
    if not has_index and has_rg:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, _focused_index_and_search, project_path, query, top_k, scope,
        )
        return result

    # Has index: normal semantic + regex in parallel
    loop = asyncio.get_event_loop()
    semantic_task = loop.run_in_executor(None, _semantic_search, project_path, query, top_k, scope)
    regex_task = _search_regex(project_path, query, top_k, scope) if has_rg else None

    semantic_results: list[tuple[str, int, int, float, str, str]] = []
    regex_results: list[tuple[str, int, str]] = []

    if regex_task:
        semantic_results, regex_raw = await asyncio.gather(semantic_task, regex_task, return_exceptions=True)
        if isinstance(semantic_results, BaseException):
            log.warning("Semantic search failed, using regex only: %s", semantic_results)
            semantic_results = []
        if isinstance(regex_raw, BaseException):
            regex_raw = ""
        regex_results = _parse_regex_output(regex_raw) if isinstance(regex_raw, str) else []
    else:
        result = await semantic_task
        semantic_results = result if not isinstance(result, BaseException) else []

    return _merge_results(semantic_results, regex_results, top_k)


def _expand_query_terms(query: str) -> list[str]:
    """Expand a natural language query into grep-friendly search terms.

    Splits on whitespace, expands camelCase/snake_case,
    and filters out short/common stop words.
    """
    _STOP_WORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "and", "or", "not", "no", "do", "does", "did", "has", "have",
        "how", "what", "where", "when", "why", "which", "who",
        "it", "its", "this", "that", "these", "those",
        "i", "we", "you", "he", "she", "they", "my", "your",
    })

    tokens = query.strip().split()
    terms: list[str] = []
    seen: set[str] = set()

    for token in tokens:
        lower = token.lower().strip(".,;:!?\"'`()[]{}#")
        if not lower or len(lower) < 2 or lower in _STOP_WORDS:
            continue
        if lower not in seen:
            seen.add(lower)
            terms.append(lower)
        # Expand camelCase: "searchQuery" → "search", "query"
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", token)
        for p in parts:
            pl = p.lower()
            if pl not in seen and len(pl) >= 3 and pl not in _STOP_WORDS:
                seen.add(pl)
                terms.append(pl)
        # Expand snake_case: "search_query" → "search", "query"
        if "_" in token:
            for p in token.split("_"):
                pl = p.lower().strip()
                if pl and pl not in seen and len(pl) >= 3 and pl not in _STOP_WORDS:
                    seen.add(pl)
                    terms.append(pl)

    return terms


def _grep_relevant_files(
    root: Path, terms: list[str], scope: str, max_files: int = 50,
) -> list[Path]:
    """Use ripgrep to find files matching any of the expanded query terms.

    Uses ``rg --count`` to rank files by match count, then returns the top
    *max_files* sorted deterministically (highest match count first, then
    by path for ties).  This ensures repeated calls with the same query
    always return the same file set.
    """
    rg = shutil.which("rg")
    if not rg or not terms:
        return []

    # Build alternation pattern: term1|term2|term3
    pattern = "|".join(re.escape(t) for t in terms[:8])  # cap at 8 terms
    search_path = str(root / scope) if scope else str(root)

    try:
        result = subprocess.run(
            [rg, "--count", "--ignore-case",
             "--type-add", "code:*.{py,js,ts,jsx,tsx,go,java,rs,cpp,c,h,hpp,rb,php,cs,kt,swift,scala,lua,sh,vue,svelte}",
             "--type", "code",
             pattern, search_path],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []

    if result.returncode not in (0, 1):
        return []

    # Parse "filepath:count" lines and sort by count descending, path ascending
    scored: list[tuple[int, str]] = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # rg --count output: "path:count"  (last colon separates count)
        sep = line.rfind(":")
        if sep < 0:
            continue
        path_str = line[:sep]
        try:
            count = int(line[sep + 1:])
        except ValueError:
            continue
        scored.append((count, path_str))

    # Sort: highest match count first, then alphabetical path for stability
    scored.sort(key=lambda x: (-x[0], x[1]))

    files: list[Path] = []
    for _, path_str in scored[:max_files]:
        p = Path(path_str)
        if p.is_file():
            files.append(p)
    return files


def _focused_index_and_search(
    project_path: str, query: str, top_k: int, scope: str,
) -> str:
    """Grep-guided focused indexing followed by semantic search.

    1. Expand query → grep terms
    2. ripgrep --files-with-matches → relevant files
    3. Index only those files (fast)
    4. Semantic search
    5. Trigger full background index for next queries
    """
    import time

    root = Path(project_path).resolve()
    t0 = time.monotonic()

    # Step 1-2: find relevant files via grep
    terms = _expand_query_terms(query)
    log.info("Focused index: query=%r, terms=%s", query, terms)

    relevant_files = _grep_relevant_files(root, terms, scope)
    if not relevant_files:
        # Fallback: no grep matches, do a small sample of all files
        all_files = [
            p for p in (root / scope if scope else root).rglob("*")
            if p.is_file() and not should_exclude(p.relative_to(root), DEFAULT_EXCLUDES)
        ]
        relevant_files = all_files[:50]

    if not relevant_files:
        _trigger_background_index(project_path)
        return "No matching files found. Background indexing started for full search next time."

    log.info("Focused index: %d files found in %.1fs", len(relevant_files), time.monotonic() - t0)

    # Step 3: index only relevant files
    try:
        indexing, search, _ = _get_pipelines(project_path)
        indexing.sync(relevant_files, root=root, delete_removed=False)
    except Exception as exc:
        log.error("Focused indexing failed: %s", exc)
        _trigger_background_index(project_path)
        return f"Error during focused indexing: {exc}"

    index_time = time.monotonic() - t0
    log.info("Focused index: indexed %d files in %.1fs", len(relevant_files), index_time)

    # Step 4: semantic search
    results = search.search(query, top_k=top_k, quality="auto")
    if scope:
        norm_scope = scope.replace("\\", "/").strip("/")
        results = [r for r in results if r.path.replace("\\", "/").startswith(norm_scope + "/")]

    # Step 5: trigger full background index for complete coverage next time
    _trigger_background_index(project_path)

    total_time = time.monotonic() - t0

    semantic_results = [
        (r.path, r.line, r.end_line, r.score, r.content, getattr(r, "language", ""))
        for r in results
    ]

    notice = (
        f"**Focused search**: indexed {len(relevant_files)} relevant files in {index_time:.1f}s, "
        f"searched in {total_time:.1f}s. Full index building in background."
    )

    output = _merge_results(semantic_results, [], top_k)
    if output == "No results found.":
        return notice + "\n\nNo semantic results found."
    return notice + "\n\n" + output


def _merge_results(
    semantic_results: list[tuple[str, int, int, float, str, str]],
    regex_results: list[tuple[str, int, str]],
    top_k: int,
) -> str:
    """Merge semantic and regex results into formatted output."""
    seen: set[tuple[str, int]] = set()
    lines: list[str] = []
    count = 0

    for path, line, end_line, score, content, lang in semantic_results:
        if count >= top_k:
            break
        seen.add((path, line))
        count += 1
        lang_tag = f" [{lang}]" if lang else ""
        lines.append(f"## {count}. {path} L{line}-{end_line} (score: {score:.4f}){lang_tag}")
        lines.append(f"```\n{content}\n```\n")

    for path, line_no, text in regex_results:
        if count >= top_k:
            break
        if (path, line_no) in seen:
            continue
        seen.add((path, line_no))
        count += 1
        lines.append(f"## {count}. {path} L{line_no} [rg]")
        lines.append(f"```\n{text}\n```\n")

    if not lines:
        return "No results found."
    return "\n".join(lines)


def _semantic_search(
    project_path: str, query: str, top_k: int, scope: str,
) -> list[tuple[str, int, int, float, str, str]]:
    """Run semantic pipeline search (called in executor thread)."""
    _, search, _ = _get_pipelines(project_path)
    results = search.search(query, top_k=top_k * (3 if scope else 1), quality="auto")

    if scope:
        norm_scope = scope.replace("\\", "/").strip("/")
        results = [r for r in results if r.path.replace("\\", "/").startswith(norm_scope + "/")]

    return [
        (r.path, r.line, r.end_line, r.score, r.content, getattr(r, "language", ""))
        for r in results
    ]


def _parse_regex_output(raw: str) -> list[tuple[str, int, str]]:
    """Parse _search_regex formatted output into (path, line, text) tuples."""
    results: list[tuple[str, int, str]] = []
    if not raw or raw.startswith("Error:") or raw == "No results found.":
        return results
    current_path = ""
    current_line = 0
    for line in raw.split("\n"):
        if line.startswith("## "):
            # "## 1. path/file.py L42"
            parts = line.split(". ", 1)
            if len(parts) < 2:
                continue
            rest = parts[1]
            tokens = rest.rsplit(" L", 1)
            if len(tokens) == 2:
                current_path = tokens[0]
                try:
                    current_line = int(tokens[1])
                except ValueError:
                    current_line = 0
        elif line.startswith("```"):
            continue
        elif current_path and line:
            results.append((current_path, current_line, line))
            current_path = ""
    return results


def _search_symbol(project_path: str, name: str, top_k: int) -> str:
    """Find symbol definitions by exact name, with fuzzy (substring) fallback."""
    fts = _get_fts(project_path)
    if fts is None:
        return "Error: no index found. Run index_project first."

    try:
        symbols = fts.get_symbols_by_name(name)
        if not symbols:
            # Fuzzy fallback: substring match
            try:
                rows = fts._conn.execute(
                    "SELECT id, chunk_id, name, kind, start_line, end_line, "
                    "parent_name, signature, language FROM symbols "
                    "WHERE name LIKE ? LIMIT ?",
                    (f"%{name}%", top_k),
                ).fetchall()
                symbols = [
                    {"id": r[0], "chunk_id": r[1], "name": r[2], "kind": r[3],
                     "start_line": r[4], "end_line": r[5], "parent_name": r[6],
                     "signature": r[7], "language": r[8]}
                    for r in rows
                ]
            except Exception:
                symbols = []

        if not symbols:
            return f"No symbols found matching '{name}'."

        lines = []
        for s in symbols[:top_k]:
            # Get file path from chunk
            meta = fts.get_doc_meta(s["chunk_id"]) if s["chunk_id"] else None
            path = Path(meta[0]).as_posix() if meta else "?"
            parent = f" in {s['parent_name']}" if s.get("parent_name") else ""
            sig = f"\n  `{s['signature']}`" if s.get("signature") else ""
            lines.append(
                f"- **{s['kind']}** `{s['name']}`{parent} — "
                f"{path}:L{s['start_line']}-{s['end_line']} [{s.get('language', '')}]{sig}"
            )
        return f"Found {len(symbols)} symbol(s) matching '{name}':\n\n" + "\n".join(lines)
    finally:
        fts.close()


def _search_refs(project_path: str, name: str) -> str:
    """Find all incoming and outgoing references for a symbol name."""
    fts = _get_fts(project_path)
    if fts is None:
        return "Error: no index found. Run index_project first."

    try:
        refs_to = fts.get_refs_to(name)
        refs_from = fts.get_refs_from(name)

        if not refs_to and not refs_from:
            return f"No references found for '{name}'."

        lines = []
        if refs_to:
            lines.append(f"### Referenced by ({len(refs_to)}):")
            for r in refs_to:
                lines.append(
                    f"- {r['ref_kind']:10s} `{r['from_name']}` → `{name}` "
                    f"— {Path(r['from_path']).as_posix()}:L{r['line']}"
                )

        if refs_from:
            lines.append(f"\n### References from `{name}` ({len(refs_from)}):")
            for r in refs_from:
                lines.append(
                    f"- {r['ref_kind']:10s} `{name}` → `{r['to_name']}` L{r['line']}"
                )

        return "\n".join(lines)
    finally:
        fts.close()


async def _search_regex(project_path: str, pattern: str, top_k: int, scope: str) -> str:
    """Search files with ripgrep regex pattern."""
    rg = shutil.which("rg")
    if rg is None:
        return "Error: regex search requires 'ripgrep' (rg) installed and in PATH."

    root = Path(project_path).resolve()
    search_path = str(root / scope) if scope else str(root)

    cmd = [rg, "--json", "--max-count", "50", pattern, search_path]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
    except OSError as e:
        return f"Error: failed to run rg: {e}"

    if proc.returncode not in (0, 1):  # 1 = no matches
        err_msg = stderr.decode(errors="replace").strip()
        return f"Error: rg failed: {err_msg}"

    # Parse rg --json output
    matches: list[tuple[str, int, str]] = []  # (rel_path, line_no, text)
    for raw_line in stdout.split(b"\n"):
        if not raw_line.strip():
            continue
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "match":
            continue
        data = obj["data"]
        abs_path = data["path"].get("text", "")
        line_no = data.get("line_number", 0)
        text = data["lines"].get("text", "").rstrip("\n")
        # Make path relative to project root
        try:
            rel = Path(abs_path).relative_to(root).as_posix()
        except ValueError:
            rel = abs_path
        matches.append((rel, line_no, text))
        if len(matches) >= top_k:
            break

    if not matches:
        return "No results found."

    lines = []
    for i, (path, line_no, text) in enumerate(matches, 1):
        lines.append(f"## {i}. {path} L{line_no}")
        lines.append(f"```\n{text}\n```\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 2: index_project — index / update / status
# ---------------------------------------------------------------------------

@mcp.tool()
async def index_project(
    project_path: str,
    action: str = "sync",
    scope: str = "",
    ctx: Context | None = None,
) -> str:
    """Build or update the search index for a project.

    Actions:
      - "sync" (default): Incremental update — only re-indexes changed files.
      - "rebuild": Full re-index from scratch, discards old index.
      - "status": Show index statistics (files, chunks, symbols, refs) without indexing.

    Args:
        project_path: Absolute path to the project root.
        action: "sync" (default), "rebuild", or "status".
        scope: Relative directory to limit indexing (e.g. "src/auth").
    """
    root = Path(project_path).resolve()
    if not root.is_dir():
        return f"Error: project path not found: {root}"

    if action == "status":
        return _index_status(project_path)

    scan_root = root / scope if scope else root
    if scope and not scan_root.is_dir():
        return f"Error: scope directory not found: {scan_root}"

    if action == "rebuild":
        with _lock:
            old = _pipelines.pop(str(root), None)
        if old:
            _close_pipeline(old)
        # Delete old index files to avoid dimension mismatch on model switch
        _purge_index_files(_db_path_for_project(project_path))

    indexing, _, _ = _get_pipelines(project_path, force=(action == "rebuild"))

    file_paths = [
        p for p in scan_root.glob("**/*")
        if p.is_file() and not should_exclude(p.relative_to(root), DEFAULT_EXCLUDES)
    ]

    if ctx:
        await ctx.report_progress(0, len(file_paths), f"Scanning {len(file_paths)} files...")

    loop = asyncio.get_event_loop()

    def _progress(done: int, total: int) -> None:
        if ctx:
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(done, total, f"Indexed {done}/{total}"),
                loop,
            )

    stats = indexing.sync(file_paths, root=root, progress_callback=_progress)

    if ctx:
        await ctx.report_progress(
            stats.files_processed, stats.files_processed,
            f"Done: {stats.files_processed} files"
        )

    # Invalidate pipeline cache so GraphSearcher picks up new symbols
    with _lock:
        old = _pipelines.pop(str(root), None)
    if old:
        _close_pipeline(old)

    # Auto-start watcher after successful indexing
    _ensure_watcher(project_path)

    scope_label = f" (scope: {scope})" if scope else ""
    return (
        f"Indexed {stats.files_processed} files, "
        f"{stats.chunks_created} chunks in {stats.duration_seconds:.1f}s{scope_label}. "
        f"DB: {_db_path_for_project(project_path)}"
    )


def _index_status(project_path: str) -> str:
    from codexlens_search.indexing.metadata import MetadataStore

    db_path = _db_path_for_project(project_path)
    meta_path = db_path / "metadata.db"

    if not meta_path.exists():
        return f"No index found at {db_path}. Run index_project first."

    metadata = MetadataStore(meta_path)
    try:
        all_files = metadata.get_all_files()
        deleted_ids = metadata.get_deleted_ids()
        max_chunk = metadata.max_chunk_id()
    finally:
        metadata.close()
    total = max_chunk + 1 if max_chunk >= 0 else 0

    # Symbol/ref counts
    sym_count = ref_count = 0
    fts_path = db_path / "fts.db"
    if fts_path.exists():
        from codexlens_search.search.fts import FTSEngine
        fts = FTSEngine(fts_path)
        try:
            sym_count = fts._conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
            ref_count = fts._conn.execute("SELECT COUNT(*) FROM symbol_refs").fetchone()[0]
        except Exception:
            pass
        finally:
            fts.close()

    lines = [
        f"Index: {db_path}",
        f"Files: {len(all_files)}",
        f"Chunks: {total} ({len(deleted_ids)} deleted)",
    ]
    if sym_count > 0:
        lines.append(f"Symbols: {sym_count}")
        lines.append(f"References: {ref_count}")
        lines.append("Graph search: enabled")
    else:
        lines.append("Graph search: disabled (no symbols — rebuild index with index_project action=rebuild)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 3: find_files
# ---------------------------------------------------------------------------

@mcp.tool()
def find_files(
    project_path: str,
    pattern: str = "**/*",
) -> str:
    """Find files in a project by glob pattern. Returns relative paths.

    Max results controlled by CODEXLENS_FIND_MAX_RESULTS env var (default 100).

    Args:
        project_path: Absolute path to the project root.
        pattern: Glob pattern to match (default "**/*").
    """
    root = Path(project_path).resolve()
    if not root.is_dir():
        return f"Error: project path not found: {root}"

    max_results = _env_int("CODEXLENS_FIND_MAX_RESULTS", 100)

    matches = []
    for p in root.glob(pattern):
        if p.is_file() and not should_exclude(p.relative_to(root), DEFAULT_EXCLUDES):
            matches.append(str(p.relative_to(root)))
            if len(matches) >= max_results:
                break

    if not matches:
        return "No files found matching the pattern."

    header = f"Found {len(matches)} files"
    if len(matches) >= max_results:
        header += f" (limited to {max_results})"
    return header + ":\n" + "\n".join(matches)


# ---------------------------------------------------------------------------
# Tool 4: watch_project — manage file watcher
# ---------------------------------------------------------------------------

@mcp.tool()
def watch_project(
    project_path: str,
    action: str = "status",
) -> str:
    """Manage file watcher for automatic re-indexing when files change.

    Args:
        project_path: Absolute path to the project root.
        action: "start", "stop", or "status" (default).
    """
    resolved = str(Path(project_path).resolve())

    if action == "start":
        # Force enable watcher regardless of env
        os.environ["CODEXLENS_AUTO_WATCH"] = "true"
        result = _ensure_watcher(project_path)
        return result or "Watcher already running"

    if action == "stop":
        return _stop_watcher(project_path)

    # status
    with _watcher_lock:
        watcher = _watchers.get(resolved)
    if watcher is not None and watcher.is_running:
        return f"Watcher: RUNNING for {resolved}"
    return f"Watcher: STOPPED for {resolved}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    mcp.run()


if __name__ == "__main__":
    main()
