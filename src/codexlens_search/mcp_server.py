"""MCP server for codexlens-search.

Exposes 3 tools via FastMCP: search_code, index_project, find_files.

Run as: codexlens-mcp (entry point) or python -m codexlens_search.mcp_server

## .mcp.json Configuration

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

Env vars: CODEXLENS_EMBED_API_URL/KEY/MODEL/DIM, CODEXLENS_AST_CHUNKING,
CODEXLENS_GITIGNORE_FILTERING, CODEXLENS_RERANKER_API_URL/KEY/MODEL
"""
from __future__ import annotations

import asyncio
import logging
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


def _db_path_for_project(project_path: str) -> Path:
    return Path(project_path).resolve() / ".codexlens"


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

@mcp.tool()
def search_code(
    project_path: str,
    query: str,
    mode: str = "auto",
    top_k: int = 10,
    scope: str = "",
) -> str:
    """Search code with hybrid fusion (FTS + vector + graph) or lookup symbols/references.

    Args:
        project_path: Absolute path to the project root.
        query: Search query — code symbol name, natural language, or pattern.
        mode: Search mode:
            - "auto": Hybrid search (FTS + vector + graph). Best for general queries.
            - "symbol": Find symbol definitions by name (class, function, method).
            - "refs": Find all references to a symbol (imports, calls, inheritance).
        top_k: Max results (default 10).
        scope: Optional relative path to restrict results (e.g. "src/auth").
    """
    root = Path(project_path).resolve()
    if not root.is_dir():
        return f"Error: project path not found: {root}"

    if mode == "symbol":
        return _search_symbol(project_path, query, top_k)
    if mode == "refs":
        return _search_refs(project_path, query)

    # mode == "auto": hybrid search
    db_path = _db_path_for_project(project_path)
    if not (db_path / "metadata.db").exists():
        return f"Error: no index found. Run index_project first."

    _, search, _ = _get_pipelines(project_path)
    results = search.search(query, top_k=top_k * (3 if scope else 1), quality="auto")

    if scope:
        norm_scope = scope.replace("\\", "/").strip("/")
        results = [r for r in results if r.path.replace("\\", "/").startswith(norm_scope + "/")]

    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results[:top_k], 1):
        lang_tag = f" [{r.language}]" if getattr(r, "language", "") else ""
        lines.append(f"## {i}. {r.path} L{r.line}-{r.end_line} (score: {r.score:.4f}){lang_tag}")
        lines.append(f"```\n{r.content}\n```\n")
    return "\n".join(lines)


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
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    mcp.run()


if __name__ == "__main__":
    main()
