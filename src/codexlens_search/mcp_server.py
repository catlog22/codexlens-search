"""MCP server for codexlens-search.

Exposes semantic code search tools via FastMCP for Claude Code integration.
Run as: codexlens-mcp (entry point) or python -m codexlens_search.mcp_server

## .mcp.json Configuration Examples

### API embedding + API reranker (single endpoint):
{
  "mcpServers": {
    "codexlens": {
      "command": "codexlens-mcp",
      "env": {
        "CODEXLENS_EMBED_API_URL": "https://api.openai.com/v1",
        "CODEXLENS_EMBED_API_KEY": "sk-xxx",
        "CODEXLENS_EMBED_API_MODEL": "text-embedding-3-small",
        "CODEXLENS_EMBED_DIM": "1536",
        "CODEXLENS_RERANKER_API_URL": "https://api.jina.ai/v1",
        "CODEXLENS_RERANKER_API_KEY": "jina-xxx",
        "CODEXLENS_RERANKER_API_MODEL": "jina-reranker-v2-base-multilingual"
      }
    }
  }
}

### API embedding (multi-endpoint load balancing):
{
  "mcpServers": {
    "codexlens": {
      "command": "codexlens-mcp",
      "env": {
        "CODEXLENS_EMBED_API_ENDPOINTS": "url1|key1|model1,url2|key2|model2",
        "CODEXLENS_EMBED_DIM": "1536",
        "CODEXLENS_RERANKER_API_URL": "https://api.jina.ai/v1",
        "CODEXLENS_RERANKER_API_KEY": "jina-xxx",
        "CODEXLENS_RERANKER_API_MODEL": "jina-reranker-v2-base-multilingual"
      }
    }
  }
}

### Local fastembed model (no API, requires codexlens-search[semantic]):
{
  "mcpServers": {
    "codexlens": {
      "command": "codexlens-mcp",
      "env": {}
    }
  }
}
Pre-download models via CLI: codexlens-search download-models

### Env vars reference:
Embedding:  CODEXLENS_EMBED_API_URL, _KEY, _MODEL, _ENDPOINTS (multi), _DIM
Reranker:   CODEXLENS_RERANKER_API_URL, _KEY, _MODEL
Tuning:     CODEXLENS_BINARY_TOP_K, _ANN_TOP_K, _FTS_TOP_K, _FUSION_K,
            CODEXLENS_RERANKER_TOP_K, _RERANKER_BATCH_SIZE
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

# Pipeline cache: keyed by resolved project_path -> (indexing, search, config)
_pipelines: dict[str, tuple] = {}
_lock = threading.Lock()


def _db_path_for_project(project_path: str) -> Path:
    """Return the index database path for a project."""
    return Path(project_path).resolve() / ".codexlens"


def _get_pipelines(project_path: str) -> tuple:
    """Get or create cached (indexing_pipeline, search_pipeline, config) for a project."""
    resolved = str(Path(project_path).resolve())
    with _lock:
        if resolved not in _pipelines:
            db_path = _db_path_for_project(resolved)
            config = create_config_from_env(db_path)
            _pipelines[resolved] = create_pipeline(db_path, config)
        return _pipelines[resolved]


# ---------------------------------------------------------------------------
# Search tools
# ---------------------------------------------------------------------------

@mcp.tool()
def search_code(
    project_path: str, query: str, top_k: int = 10, quality: str = "auto"
) -> str:
    """Semantic code search with hybrid fusion (vector + FTS + reranking).

    Args:
        project_path: Absolute path to the project root directory.
        query: Natural language or code search query.
        top_k: Maximum number of results to return (default 10).
        quality: Search quality tier (default "auto"):
            - "fast": FTS-only + rerank (no embedding needed, fastest)
            - "balanced": FTS + binary coarse search + rerank
            - "thorough": Full 2-stage vector + FTS + reranking (best quality)
            - "auto": Uses "thorough" if vector index exists, else "fast"

    Returns:
        Search results as formatted text with file paths, line numbers, scores, and code snippets.
    """
    root = Path(project_path).resolve()
    if not root.is_dir():
        return f"Error: project path not found: {root}"

    db_path = _db_path_for_project(project_path)
    if not (db_path / "metadata.db").exists():
        return f"Error: no index found at {db_path}. Run index_project first."

    valid_qualities = ("fast", "balanced", "thorough", "auto")
    if quality not in valid_qualities:
        return f"Error: invalid quality '{quality}'. Must be one of: {', '.join(valid_qualities)}"

    _, search, _ = _get_pipelines(project_path)
    results = search.search(query, top_k=top_k, quality=quality)

    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"## Result {i} -- {r.path} (L{r.line}-{r.end_line}, score: {r.score:.4f})")
        lines.append(f"```\n{r.content}\n```")
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def search_scope(
    project_path: str,
    query: str,
    scope_path: str,
    top_k: int = 10,
    quality: str = "auto",
) -> str:
    """Search within a specific directory scope of a project.

    Runs a normal search then filters results to only include files
    under the specified scope path.

    Args:
        project_path: Absolute path to the project root directory.
        query: Natural language or code search query.
        scope_path: Relative directory path to limit search scope (e.g. "src/auth").
        top_k: Maximum number of scoped results to return (default 10).
        quality: Search quality tier ("fast", "balanced", "thorough", "auto").

    Returns:
        Search results filtered to the scope path.
    """
    root = Path(project_path).resolve()
    if not root.is_dir():
        return f"Error: project path not found: {root}"

    db_path = _db_path_for_project(project_path)
    if not (db_path / "metadata.db").exists():
        return f"Error: no index found at {db_path}. Run index_project first."

    # Normalize scope path for prefix matching
    scope = scope_path.replace("\\", "/").strip("/")

    _, search, _ = _get_pipelines(project_path)
    # Fetch more results than top_k to account for filtering
    all_results = search.search(query, top_k=top_k * 5, quality=quality)

    # Filter by scope path prefix
    scoped = [
        r for r in all_results
        if r.path.replace("\\", "/").startswith(scope + "/")
        or r.path.replace("\\", "/") == scope
    ]

    if not scoped:
        return f"No results found in scope '{scope_path}'."

    lines = []
    for i, r in enumerate(scoped[:top_k], 1):
        lines.append(f"## Result {i} -- {r.path} (L{r.line}-{r.end_line}, score: {r.score:.4f})")
        lines.append(f"```\n{r.content}\n```")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Indexing tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def index_project(
    project_path: str, glob_pattern: str = "**/*", force: bool = False,
    ctx: Context | None = None,
) -> str:
    """Build or rebuild the search index for a project.

    Args:
        project_path: Absolute path to the project root directory.
        glob_pattern: Glob pattern for files to index (default "**/*").
        force: If True, rebuild index from scratch even if it exists.

    Returns:
        Indexing summary with file count, chunk count, and duration.
    """
    root = Path(project_path).resolve()
    if not root.is_dir():
        return f"Error: project path not found: {root}"

    if force:
        with _lock:
            _pipelines.pop(str(root), None)

    indexing, _, _ = _get_pipelines(project_path)

    file_paths = [
        p for p in root.glob(glob_pattern)
        if p.is_file() and not should_exclude(p.relative_to(root), DEFAULT_EXCLUDES)
    ]

    if ctx:
        await ctx.report_progress(0, len(file_paths), f"Scanning {len(file_paths)} files...")

    # Progress callback bridging sync pipeline → async MCP context
    loop = asyncio.get_event_loop()

    def _progress(done: int, total: int) -> None:
        if ctx:
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(done, total, f"Indexed {done}/{total} files"),
                loop,
            )

    stats = indexing.sync(file_paths, root=root, progress_callback=_progress)

    if ctx:
        await ctx.report_progress(
            stats.files_processed, stats.files_processed,
            f"Done: {stats.files_processed} files, {stats.chunks_created} chunks"
        )

    return (
        f"Indexed {stats.files_processed} files, "
        f"{stats.chunks_created} chunks in {stats.duration_seconds:.1f}s. "
        f"DB: {_db_path_for_project(project_path)}"
    )


@mcp.tool()
def index_status(project_path: str) -> str:
    """Show index statistics for a project.

    Args:
        project_path: Absolute path to the project root directory.

    Returns:
        Index statistics including file count, chunk count, and deleted chunks.
    """
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
    return (
        f"Index: {db_path}\n"
        f"Files tracked: {len(all_files)}\n"
        f"Total chunks: {total}\n"
        f"Deleted chunks: {len(deleted_ids)}"
    )


@mcp.tool()
async def index_update(
    project_path: str, glob_pattern: str = "**/*",
    ctx: Context | None = None,
) -> str:
    """Incrementally sync the index with current project files.

    Only re-indexes files that changed since last indexing.

    Args:
        project_path: Absolute path to the project root directory.
        glob_pattern: Glob pattern for files to sync (default "**/*").

    Returns:
        Sync summary with processed file count and duration.
    """
    root = Path(project_path).resolve()
    if not root.is_dir():
        return f"Error: project path not found: {root}"

    indexing, _, _ = _get_pipelines(project_path)

    file_paths = [
        p for p in root.glob(glob_pattern)
        if p.is_file() and not should_exclude(p.relative_to(root), DEFAULT_EXCLUDES)
    ]

    if ctx:
        await ctx.report_progress(0, len(file_paths), f"Scanning {len(file_paths)} files...")

    loop = asyncio.get_event_loop()

    def _progress(done: int, total: int) -> None:
        if ctx:
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(done, total, f"Synced {done}/{total} files"),
                loop,
            )

    stats = indexing.sync(file_paths, root=root, progress_callback=_progress)
    return (
        f"Synced {stats.files_processed} files, "
        f"{stats.chunks_created} chunks in {stats.duration_seconds:.1f}s."
    )


@mcp.tool()
def index_scope(
    project_path: str,
    scope_path: str,
    glob_pattern: str = "**/*",
    tier: str = "full",
) -> str:
    """Index a specific directory scope within a project.

    Useful for quickly indexing a subdirectory (e.g. after editing files
    in a specific module) without re-indexing the entire project.

    Args:
        project_path: Absolute path to the project root directory.
        scope_path: Relative directory path to index (e.g. "src/auth").
        glob_pattern: Glob pattern for files within scope (default "**/*").
        tier: Indexing tier - "full" (default) runs full pipeline with
              embedding, "fts_only" indexes text only (faster, no vectors).

    Returns:
        Indexing summary for the scoped directory.
    """
    root = Path(project_path).resolve()
    if not root.is_dir():
        return f"Error: project path not found: {root}"

    scope_dir = root / scope_path
    if not scope_dir.is_dir():
        return f"Error: scope directory not found: {scope_dir}"

    valid_tiers = ("full", "fts_only")
    if tier not in valid_tiers:
        return f"Error: invalid tier '{tier}'. Must be one of: {', '.join(valid_tiers)}"

    indexing, _, _ = _get_pipelines(project_path)

    file_paths = [
        p for p in scope_dir.glob(glob_pattern)
        if p.is_file() and not should_exclude(p.relative_to(root), DEFAULT_EXCLUDES)
    ]

    if not file_paths:
        return f"No files found in {scope_path} matching '{glob_pattern}'."

    stats = indexing.sync(file_paths, root=root, tier=tier)
    tier_label = "FTS-only" if tier == "fts_only" else "full"
    return (
        f"Indexed {stats.files_processed} files ({tier_label}), "
        f"{stats.chunks_created} chunks in {stats.duration_seconds:.1f}s. "
        f"Scope: {scope_path}"
    )


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

@mcp.tool()
def find_files(
    project_path: str, pattern: str = "**/*", max_results: int = 100
) -> str:
    """Find files in a project by glob pattern.

    Args:
        project_path: Absolute path to the project root directory.
        pattern: Glob pattern to match files (default "**/*").
        max_results: Maximum number of file paths to return (default 100).

    Returns:
        List of matching file paths (relative to project root), one per line.
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
# Model management tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_models() -> str:
    """List available embedding and reranker models with cache status.

    Shows which models are downloaded locally and ready for use.
    Models are needed when using local fastembed mode (no API URL configured).

    Returns:
        Table of models with name, type, and installed status.
    """
    from codexlens_search import model_manager
    from codexlens_search.config import Config

    config = create_config_from_env(".")
    models = model_manager.list_known_models(config)

    if not models:
        return "No known models found."

    lines = ["| Model | Type | Installed |", "| --- | --- | --- |"]
    for m in models:
        status = "Yes" if m["installed"] else "No"
        lines.append(f"| {m['name']} | {m['type']} | {status} |")

    # Show current config
    lines.append("")
    if config.embed_api_url:
        lines.append(f"Mode: API embedding ({config.embed_api_url})")
    else:
        lines.append(f"Mode: Local fastembed (model: {config.embed_model})")
    return "\n".join(lines)


@mcp.tool()
def download_models(embed_model: str = "", reranker_model: str = "") -> str:
    """Download embedding and reranker models for local (fastembed) mode.

    Not needed when using API embedding (CODEXLENS_EMBED_API_URL is set).
    Downloads are cached — subsequent calls are no-ops if already downloaded.

    Args:
        embed_model: Embedding model name (default: BAAI/bge-small-en-v1.5).
        reranker_model: Reranker model name (default: Xenova/ms-marco-MiniLM-L-6-v2).

    Returns:
        Download status for each model.
    """
    from codexlens_search import model_manager
    from codexlens_search.config import Config

    config = create_config_from_env(".")
    if embed_model:
        config.embed_model = embed_model
    if reranker_model:
        config.reranker_model = reranker_model

    results = []
    for name, kind in [
        (config.embed_model, "embedding"),
        (config.reranker_model, "reranker"),
    ]:
        try:
            model_manager.ensure_model(name, config)
            results.append(f"{kind}: {name} — ready")
        except Exception as e:
            results.append(f"{kind}: {name} — failed: {e}")

    return "\n".join(results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for codexlens-mcp command."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    mcp.run()


if __name__ == "__main__":
    main()
