"""MCP server for codexlens-search.

Tools:
  - Search:        Hybrid code search (semantic + regex) with symbol/reference lookup.
  - index_project: Build, update, or inspect the search index.
  - find_files:    Glob-based file discovery.

Run: codexlens-mcp  or  python -m codexlens_search.mcp_server

## .mcp.json

```json
{
  "mcpServers": {
    "codexlens": {
      "command": "codexlens-mcp",
      "env": {
        "CODEXLENS_EMBED_API_URL": "https://api.openai.com/v1",
        "CODEXLENS_EMBED_API_KEY": "sk-xxx",
        "CODEXLENS_EMBED_API_MODEL": "text-embedding-3-small",
        "CODEXLENS_EMBED_DIM": "1536",
        "CODEXLENS_AST_CHUNKING": "true"
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
import shutil
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


def _db_path_for_project(project_path: str) -> Path:
    return Path(project_path).resolve() / ".codexlens"


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
                _pipelines.pop(resolved, None)
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


atexit.register(_cleanup_watchers)


def _get_pipelines(project_path: str, force: bool = False) -> tuple:
    resolved = str(Path(project_path).resolve())
    with _lock:
        if force:
            _pipelines.pop(resolved, None)
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
# Tool 1: search_code — unified search
# ---------------------------------------------------------------------------

@mcp.tool(name="Search")
async def search_code(
    project_path: str,
    query: str,
    mode: str = "auto",
    top_k: int = 10,
    scope: str = "",
) -> str:
    """Search code in a project.

    Args:
        project_path: Absolute path to the project root.
        query: Search query — natural language, code symbol, or regex pattern.
        mode: "auto" (default) — semantic + regex parallel, falls back to regex if no index.
              "symbol" — find definitions by name (class, function, method).
              "refs" — find cross-references (imports, calls, inheritance).
              "regex" — ripgrep regex on live files (requires rg in PATH).
        top_k: Max results (default 10).
        scope: Relative path to restrict search (e.g. "src/auth").
    """
    root = Path(project_path).resolve()
    if not root.is_dir():
        return f"Error: project path not found: {root}"

    if mode == "regex":
        return await _search_regex(project_path, query, top_k, scope)
    if mode == "symbol":
        return _search_symbol(project_path, query, top_k)
    if mode == "refs":
        return _search_refs(project_path, query)

    # mode == "auto": hybrid search with parallel regex + fallback
    return await _search_auto(project_path, query, top_k, scope)


async def _search_auto(project_path: str, query: str, top_k: int, scope: str) -> str:
    """Hybrid search with parallel regex supplement and no-index fallback."""
    root = Path(project_path).resolve()
    db_path = _db_path_for_project(project_path)
    has_index = (db_path / "metadata.db").exists()
    has_rg = shutil.which("rg") is not None

    if not has_index and not has_rg:
        return "Error: no index found and ripgrep (rg) not available. Run index_project or install rg."

    # Trigger background indexing if no index exists
    indexing_notice = ""
    if not has_index:
        indexing_notice = _trigger_background_index(project_path)

    # Run semantic + regex in parallel when both available; fallback otherwise
    semantic_task = None
    regex_task = None

    loop = asyncio.get_event_loop()

    if has_index:
        semantic_task = loop.run_in_executor(None, _semantic_search, project_path, query, top_k, scope)
    if has_rg:
        regex_task = _search_regex(project_path, query, top_k, scope)

    semantic_results: list[tuple[str, int, int, float, str, str]] = []  # (path, line, end_line, score, content, lang)
    regex_results: list[tuple[str, int, str]] = []  # (path, line, text)

    if semantic_task and regex_task:
        semantic_results, regex_raw = await asyncio.gather(semantic_task, regex_task, return_exceptions=True)
        if isinstance(semantic_results, BaseException):
            log.warning("Semantic search failed, using regex only: %s", semantic_results)
            semantic_results = []
        if isinstance(regex_raw, BaseException):
            log.warning("Regex search failed: %s", regex_raw)
            regex_raw = ""
        regex_results = _parse_regex_output(regex_raw) if isinstance(regex_raw, str) else []
    elif semantic_task:
        result = await semantic_task
        semantic_results = result if not isinstance(result, BaseException) else []
    elif regex_task:
        regex_raw = await regex_task
        regex_results = _parse_regex_output(regex_raw) if isinstance(regex_raw, str) else []

    # Merge: semantic results first, then regex results deduped by (path, line)
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
        if indexing_notice:
            return indexing_notice + "\n\nNo results found."
        return "No results found."

    result = "\n".join(lines)
    if indexing_notice:
        result = indexing_notice + "\n\n" + result
    return result


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
    """Find symbol definitions by name."""
    fts = _get_fts(project_path)
    if fts is None:
        return "Error: no index found. Run index_project first."

    symbols = fts.get_symbols_by_name(name)
    if not symbols:
        # Try fuzzy: search symbols whose name contains the query
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


def _search_refs(project_path: str, name: str) -> str:
    """Find all references to/from a symbol."""
    fts = _get_fts(project_path)
    if fts is None:
        return "Error: no index found. Run index_project first."

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
    force: bool = False,
    ctx: Context | None = None,
) -> str:
    """Build, update, or check the search index for a project.

    Args:
        project_path: Absolute path to the project root.
        action: What to do:
            - "sync": Incremental update — only re-indexes changed files (default).
            - "rebuild": Full re-index from scratch.
            - "status": Show index statistics without indexing.
        scope: Optional relative directory to limit indexing (e.g. "src/auth").
        force: Alias for action="rebuild" (backward compat).
    """
    root = Path(project_path).resolve()
    if not root.is_dir():
        return f"Error: project path not found: {root}"

    if action == "status":
        return _index_status(project_path)

    if force:
        action = "rebuild"

    scan_root = root / scope if scope else root
    if scope and not scan_root.is_dir():
        return f"Error: scope directory not found: {scan_root}"

    if action == "rebuild":
        with _lock:
            _pipelines.pop(str(root), None)

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
        _pipelines.pop(str(root), None)

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
    all_files = metadata.get_all_files()
    deleted_ids = metadata.get_deleted_ids()
    max_chunk = metadata.max_chunk_id()
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
        lines.append("Graph search: disabled (no symbols — enable CODEXLENS_AST_CHUNKING=true)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 3: find_files
# ---------------------------------------------------------------------------

@mcp.tool()
def find_files(
    project_path: str, pattern: str = "**/*", max_results: int = 100
) -> str:
    """Find files in a project by glob pattern.

    Args:
        project_path: Absolute path to the project root.
        pattern: Glob pattern to match (default "**/*").
        max_results: Max file paths to return (default 100).
    """
    root = Path(project_path).resolve()
    if not root.is_dir():
        return f"Error: project path not found: {root}"

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
    """Manage file watcher for automatic re-indexing on file changes.

    Args:
        project_path: Absolute path to the project root.
        action: "start" -- start watching, "stop" -- stop watching,
                "status" -- check watcher status (default).
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
