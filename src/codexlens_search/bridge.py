"""CLI bridge for ccw integration.

Argparse-based CLI with JSON output protocol.
Each subcommand outputs a single JSON object to stdout.
Watch command outputs JSONL (one JSON per line).
All errors are JSON {"error": string} to stdout with non-zero exit code.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

log = logging.getLogger("codexlens_search.bridge")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_utf8_stdio() -> None:
    """Force UTF-8 encoding on stdout/stderr (Windows defaults to GBK/cp936)."""
    if sys.platform == "win32":
        for stream_name in ("stdout", "stderr"):
            stream = getattr(sys, stream_name)
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")


def _json_output(data: dict | list) -> None:
    """Print JSON to stdout with flush."""
    print(json.dumps(data, ensure_ascii=True), flush=True)


def _error_exit(message: str, code: int = 1) -> None:
    """Print JSON error to stdout and exit."""
    _json_output({"error": message})
    sys.exit(code)


def _resolve_db_path(args: argparse.Namespace) -> Path:
    """Return the --db-path as a resolved Path, creating parent dirs."""
    db_path = Path(args.db_path).resolve()
    db_path.mkdir(parents=True, exist_ok=True)
    return db_path


def create_config_from_env(db_path: str | Path, **overrides: object) -> "Config":
    """Build Config from environment variables and optional overrides.

    Used by both CLI bridge and MCP server.
    """
    from codexlens_search.config import Config

    kwargs: dict = {}
    # Apply explicit overrides first
    for key in ("embed_model", "embed_api_url", "embed_api_key", "embed_api_model"):
        if overrides.get(key):
            kwargs[key] = overrides[key]
    # Local model env vars
    if "embed_model" not in kwargs and os.environ.get("CODEXLENS_EMBED_MODEL"):
        kwargs["embed_model"] = os.environ["CODEXLENS_EMBED_MODEL"]
    if os.environ.get("CODEXLENS_MODEL_CACHE_DIR"):
        kwargs["model_cache_dir"] = os.environ["CODEXLENS_MODEL_CACHE_DIR"]
    if os.environ.get("CODEXLENS_HF_MIRROR"):
        kwargs["hf_mirror"] = os.environ["CODEXLENS_HF_MIRROR"]
    # Env vars as fallback
    if "embed_api_url" not in kwargs and os.environ.get("CODEXLENS_EMBED_API_URL"):
        kwargs["embed_api_url"] = os.environ["CODEXLENS_EMBED_API_URL"]
    if "embed_api_key" not in kwargs and os.environ.get("CODEXLENS_EMBED_API_KEY"):
        kwargs["embed_api_key"] = os.environ["CODEXLENS_EMBED_API_KEY"]
    if "embed_api_model" not in kwargs and os.environ.get("CODEXLENS_EMBED_API_MODEL"):
        kwargs["embed_api_model"] = os.environ["CODEXLENS_EMBED_API_MODEL"]
    # Multi-endpoint: CODEXLENS_EMBED_API_ENDPOINTS=url1|key1|model1,url2|key2|model2
    endpoints_env = os.environ.get("CODEXLENS_EMBED_API_ENDPOINTS", "")
    if endpoints_env:
        endpoints = []
        for entry in endpoints_env.split(","):
            parts = entry.strip().split("|")
            if len(parts) >= 2:
                ep = {"url": parts[0], "key": parts[1]}
                if len(parts) >= 3:
                    ep["model"] = parts[2]
                endpoints.append(ep)
        if endpoints:
            kwargs["embed_api_endpoints"] = endpoints
    # Embed dimension and concurrency from env
    if os.environ.get("CODEXLENS_EMBED_DIM"):
        kwargs["embed_dim"] = int(os.environ["CODEXLENS_EMBED_DIM"])
    if os.environ.get("CODEXLENS_EMBED_BATCH_SIZE"):
        kwargs["embed_batch_size"] = int(os.environ["CODEXLENS_EMBED_BATCH_SIZE"])
    if os.environ.get("CODEXLENS_EMBED_API_CONCURRENCY"):
        kwargs["embed_api_concurrency"] = int(os.environ["CODEXLENS_EMBED_API_CONCURRENCY"])
    if os.environ.get("CODEXLENS_EMBED_API_MAX_TOKENS"):
        kwargs["embed_api_max_tokens_per_batch"] = int(os.environ["CODEXLENS_EMBED_API_MAX_TOKENS"])
    if os.environ.get("CODEXLENS_EMBED_MAX_TOKENS"):
        kwargs["embed_max_tokens"] = int(os.environ["CODEXLENS_EMBED_MAX_TOKENS"])
    # Reranker env vars
    if os.environ.get("CODEXLENS_RERANKER_MODEL"):
        kwargs["reranker_model"] = os.environ["CODEXLENS_RERANKER_MODEL"]
    if os.environ.get("CODEXLENS_RERANKER_API_URL"):
        kwargs["reranker_api_url"] = os.environ["CODEXLENS_RERANKER_API_URL"]
    if os.environ.get("CODEXLENS_RERANKER_API_KEY"):
        kwargs["reranker_api_key"] = os.environ["CODEXLENS_RERANKER_API_KEY"]
    if os.environ.get("CODEXLENS_RERANKER_API_MODEL"):
        kwargs["reranker_api_model"] = os.environ["CODEXLENS_RERANKER_API_MODEL"]
    # Search pipeline params from env
    if os.environ.get("CODEXLENS_RERANKER_TOP_K"):
        kwargs["reranker_top_k"] = int(os.environ["CODEXLENS_RERANKER_TOP_K"])
    if os.environ.get("CODEXLENS_RERANKER_BATCH_SIZE"):
        kwargs["reranker_batch_size"] = int(os.environ["CODEXLENS_RERANKER_BATCH_SIZE"])
    if os.environ.get("CODEXLENS_BINARY_TOP_K"):
        kwargs["binary_top_k"] = int(os.environ["CODEXLENS_BINARY_TOP_K"])
    if os.environ.get("CODEXLENS_ANN_TOP_K"):
        kwargs["ann_top_k"] = int(os.environ["CODEXLENS_ANN_TOP_K"])
    if os.environ.get("CODEXLENS_FTS_TOP_K"):
        kwargs["fts_top_k"] = int(os.environ["CODEXLENS_FTS_TOP_K"])
    if os.environ.get("CODEXLENS_FUSION_K"):
        kwargs["fusion_k"] = int(os.environ["CODEXLENS_FUSION_K"])
    # AST chunking from env
    if os.environ.get("CODEXLENS_AST_CHUNKING"):
        kwargs["ast_chunking"] = os.environ["CODEXLENS_AST_CHUNKING"].lower() in ("true", "1", "yes")
    if os.environ.get("CODEXLENS_GITIGNORE_FILTERING"):
        kwargs["gitignore_filtering"] = os.environ["CODEXLENS_GITIGNORE_FILTERING"].lower() in ("true", "1", "yes")
    # Indexing params from env
    if os.environ.get("CODEXLENS_CODE_AWARE_CHUNKING"):
        kwargs["code_aware_chunking"] = os.environ["CODEXLENS_CODE_AWARE_CHUNKING"].lower() == "true"
    if os.environ.get("CODEXLENS_INDEX_WORKERS"):
        kwargs["index_workers"] = int(os.environ["CODEXLENS_INDEX_WORKERS"])
    if os.environ.get("CODEXLENS_MAX_FILE_SIZE"):
        kwargs["max_file_size_bytes"] = int(os.environ["CODEXLENS_MAX_FILE_SIZE"])
    if os.environ.get("CODEXLENS_HNSW_EF"):
        kwargs["hnsw_ef"] = int(os.environ["CODEXLENS_HNSW_EF"])
    if os.environ.get("CODEXLENS_HNSW_M"):
        kwargs["hnsw_M"] = int(os.environ["CODEXLENS_HNSW_M"])
    if os.environ.get("CODEXLENS_DEVICE"):
        kwargs["device"] = os.environ["CODEXLENS_DEVICE"]
    if os.environ.get("CODEXLENS_ANN_BACKEND"):
        kwargs["ann_backend"] = os.environ["CODEXLENS_ANN_BACKEND"]
    # Tier config from env
    if os.environ.get("CODEXLENS_TIER_HOT_HOURS"):
        kwargs["tier_hot_hours"] = int(os.environ["CODEXLENS_TIER_HOT_HOURS"])
    if os.environ.get("CODEXLENS_TIER_COLD_HOURS"):
        kwargs["tier_cold_hours"] = int(os.environ["CODEXLENS_TIER_COLD_HOURS"])
    # Search quality tier from env
    if os.environ.get("CODEXLENS_SEARCH_QUALITY"):
        kwargs["default_search_quality"] = os.environ["CODEXLENS_SEARCH_QUALITY"]
    # Shard config from env
    if os.environ.get("CODEXLENS_NUM_SHARDS"):
        kwargs["num_shards"] = int(os.environ["CODEXLENS_NUM_SHARDS"])
    if os.environ.get("CODEXLENS_MAX_LOADED_SHARDS"):
        kwargs["max_loaded_shards"] = int(os.environ["CODEXLENS_MAX_LOADED_SHARDS"])
    resolved = Path(db_path).resolve()
    kwargs["metadata_db_path"] = str(resolved / "metadata.db")
    return Config(**kwargs)


def _create_config(args: argparse.Namespace) -> "Config":
    """Build Config from CLI args (delegates to create_config_from_env)."""
    overrides: dict = {}
    if hasattr(args, "embed_model") and args.embed_model:
        overrides["embed_model"] = args.embed_model
    if hasattr(args, "embed_api_url") and args.embed_api_url:
        overrides["embed_api_url"] = args.embed_api_url
    if hasattr(args, "embed_api_key") and args.embed_api_key:
        overrides["embed_api_key"] = args.embed_api_key
    if hasattr(args, "embed_api_model") and args.embed_api_model:
        overrides["embed_api_model"] = args.embed_api_model
    return create_config_from_env(args.db_path, **overrides)


def _create_embedder(config: "Config"):
    """Create embedder based on config, auto-detecting embed_dim from API."""
    if config.embed_api_url:
        from codexlens_search.embed.api import APIEmbedder
        embedder = APIEmbedder(config)
        log.info("Using API embedder: %s", config.embed_api_url)
        # Auto-detect embed_dim from API
        probe_vec = embedder.embed_single("dimension probe")
        detected_dim = probe_vec.shape[0]
        if detected_dim != config.embed_dim:
            log.info("Auto-detected embed_dim=%d from API (was %d)", detected_dim, config.embed_dim)
            config.embed_dim = detected_dim
    else:
        from codexlens_search.embed.local import FastEmbedEmbedder
        embedder = FastEmbedEmbedder(config)
        # Auto-detect embed_dim from local model
        probe_vec = embedder.embed_single("dimension probe")
        detected_dim = probe_vec.shape[0]
        if detected_dim != config.embed_dim:
            log.warning(
                "Local model actual dim=%d differs from config embed_dim=%d — "
                "auto-correcting to %d",
                detected_dim, config.embed_dim, detected_dim,
            )
            config.embed_dim = detected_dim
    return embedder


def _create_reranker(config: "Config"):
    """Create reranker based on config."""
    if config.reranker_api_url:
        from codexlens_search.rerank.api import APIReranker
        reranker = APIReranker(config)
        log.info("Using API reranker: %s", config.reranker_api_url)
    else:
        from codexlens_search.rerank.local import FastEmbedReranker
        reranker = FastEmbedReranker(config)
    return reranker


def create_pipeline(
    db_path: str | Path,
    config: "Config | None" = None,
) -> tuple:
    """Construct pipeline components from db_path and config.

    Returns (indexing_pipeline, search_pipeline, config).
    Used by both CLI bridge and MCP server.

    When config.num_shards > 1, returns a ShardManager-backed pipeline
    where indexing and search are delegated to the ShardManager.
    The returned tuple is (shard_manager, shard_manager, config) so that
    callers can use shard_manager.sync() and shard_manager.search().
    """
    from codexlens_search.config import Config

    if config is None:
        config = create_config_from_env(db_path)
    resolved = Path(db_path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)

    embedder = _create_embedder(config)
    reranker = _create_reranker(config)

    # Sharded mode: delegate to ShardManager
    if config.num_shards > 1:
        from codexlens_search.core.shard_manager import ShardManager
        manager = ShardManager(
            num_shards=config.num_shards,
            db_path=resolved,
            config=config,
            embedder=embedder,
            reranker=reranker,
        )
        log.info(
            "Using ShardManager with %d shards (max_loaded=%d)",
            config.num_shards, config.max_loaded_shards,
        )
        return manager, manager, config

    # Single-shard mode: original behavior, no ShardManager overhead
    from codexlens_search.core.factory import create_ann_index, create_binary_index
    from codexlens_search.indexing.metadata import MetadataStore
    from codexlens_search.indexing.pipeline import IndexingPipeline
    from codexlens_search.search.fts import FTSEngine
    from codexlens_search.search.pipeline import SearchPipeline

    binary_store = create_binary_index(resolved, config.embed_dim, config)
    ann_index = create_ann_index(resolved, config.embed_dim, config)
    fts = FTSEngine(resolved / "fts.db")
    metadata = MetadataStore(resolved / "metadata.db")

    # GraphSearcher: always create — symbols may be populated after indexing
    from codexlens_search.search.graph import GraphSearcher
    graph_searcher: GraphSearcher | None = None
    try:
        graph_searcher = GraphSearcher(fts, expand_hops=1)
    except Exception:
        pass

    indexing = IndexingPipeline(
        embedder=embedder,
        binary_store=binary_store,
        ann_index=ann_index,
        fts=fts,
        config=config,
        metadata=metadata,
    )

    search = SearchPipeline(
        embedder=embedder,
        binary_store=binary_store,
        ann_index=ann_index,
        reranker=reranker,
        fts=fts,
        config=config,
        metadata_store=metadata,
        graph_searcher=graph_searcher,
    )

    return indexing, search, config


def _create_pipeline(
    args: argparse.Namespace,
) -> tuple:
    """CLI wrapper: construct pipeline from argparse args."""
    config = _create_config(args)
    db_path = _resolve_db_path(args)
    return create_pipeline(db_path, config)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> None:
    """Initialize an empty index at --db-path."""
    from codexlens_search.indexing.metadata import MetadataStore
    from codexlens_search.search.fts import FTSEngine

    db_path = _resolve_db_path(args)

    # Create empty stores - just touch the metadata and FTS databases
    meta = MetadataStore(db_path / "metadata.db")
    fts = FTSEngine(db_path / "fts.db")
    meta.close()
    fts.close()

    _json_output({
        "status": "initialized",
        "db_path": str(db_path),
    })


def cmd_search(args: argparse.Namespace) -> None:
    """Run search query, output JSON array of results."""
    _, search, _ = _create_pipeline(args)

    results = search.search(args.query, top_k=args.top_k)
    _json_output([
        {
            "path": r.path,
            "score": r.score,
            "line": r.line,
            "end_line": r.end_line,
            "snippet": r.snippet,
            "content": r.content,
        }
        for r in results
    ])


def cmd_index_file(args: argparse.Namespace) -> None:
    """Index a single file."""
    indexing, _, _ = _create_pipeline(args)

    file_path = Path(args.file).resolve()
    if not file_path.is_file():
        _error_exit(f"File not found: {file_path}")

    root = Path(args.root).resolve() if args.root else None

    stats = indexing.index_file(file_path, root=root)
    _json_output({
        "status": "indexed",
        "file": str(file_path),
        "files_processed": stats.files_processed,
        "chunks_created": stats.chunks_created,
        "duration_seconds": stats.duration_seconds,
    })


def cmd_remove_file(args: argparse.Namespace) -> None:
    """Remove a file from the index."""
    indexing, _, _ = _create_pipeline(args)

    indexing.remove_file(args.file)
    _json_output({
        "status": "removed",
        "file": args.file,
    })


DEFAULT_EXCLUDES = frozenset({
    "node_modules", ".git", "__pycache__", "dist", "build",
    ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache",
    ".next", ".nuxt", "coverage", ".eggs", "*.egg-info", ".codexlens",
})


def should_exclude(path: Path, exclude_dirs: frozenset[str]) -> bool:
    """Check if any path component matches an exclude pattern."""
    parts = path.parts
    return any(part in exclude_dirs for part in parts)


def cmd_sync(args: argparse.Namespace) -> None:
    """Sync index with files under --root matching --glob pattern."""
    indexing, _, _ = _create_pipeline(args)

    root = Path(args.root).resolve()
    if not root.is_dir():
        _error_exit(f"Root directory not found: {root}")

    exclude_dirs = frozenset(args.exclude) if args.exclude else DEFAULT_EXCLUDES
    pattern = args.glob or "**/*"
    file_paths = [
        p for p in root.glob(pattern)
        if p.is_file() and not should_exclude(p.relative_to(root), exclude_dirs)
    ]

    log.debug("Sync: %d files after exclusion (root=%s, pattern=%s)", len(file_paths), root, pattern)

    stats = indexing.sync(file_paths, root=root)
    _json_output({
        "status": "synced",
        "root": str(root),
        "files_processed": stats.files_processed,
        "chunks_created": stats.chunks_created,
        "duration_seconds": stats.duration_seconds,
    })


def cmd_watch(args: argparse.Namespace) -> None:
    """Watch --root for changes, output JSONL events."""
    root = Path(args.root).resolve()
    if not root.is_dir():
        _error_exit(f"Root directory not found: {root}")

    debounce_ms = args.debounce_ms

    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler, FileSystemEvent
    except ImportError:
        _error_exit(
            "watchdog is required for watch mode. "
            "Install with: pip install watchdog"
        )

    class _JsonEventHandler(FileSystemEventHandler):
        """Emit JSONL for file events."""

        def _emit(self, event_type: str, path: str) -> None:
            _json_output({
                "event": event_type,
                "path": path,
                "timestamp": time.time(),
            })

        def on_created(self, event: FileSystemEvent) -> None:
            if not event.is_directory:
                self._emit("created", event.src_path)

        def on_modified(self, event: FileSystemEvent) -> None:
            if not event.is_directory:
                self._emit("modified", event.src_path)

        def on_deleted(self, event: FileSystemEvent) -> None:
            if not event.is_directory:
                self._emit("deleted", event.src_path)

        def on_moved(self, event: FileSystemEvent) -> None:
            if not event.is_directory:
                self._emit("moved", event.dest_path)

    observer = Observer()
    observer.schedule(_JsonEventHandler(), str(root), recursive=True)
    observer.start()

    _json_output({
        "status": "watching",
        "root": str(root),
        "debounce_ms": debounce_ms,
    })

    try:
        while True:
            time.sleep(debounce_ms / 1000.0)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def cmd_download_models(args: argparse.Namespace) -> None:
    """Download embed + reranker models."""
    from codexlens_search import model_manager

    config = _create_config(args)

    model_manager.ensure_model(config.embed_model, config)
    model_manager.ensure_model(config.reranker_model, config)

    _json_output({
        "status": "downloaded",
        "embed_model": config.embed_model,
        "reranker_model": config.reranker_model,
    })


def cmd_list_models(args: argparse.Namespace) -> None:
    """List known embed/reranker models with cache status."""
    from codexlens_search import model_manager

    config = _create_config(args)
    models = model_manager.list_known_models(config)
    _json_output(models)


def cmd_download_model(args: argparse.Namespace) -> None:
    """Download a single model by name."""
    from codexlens_search import model_manager

    config = _create_config(args)
    model_name = args.model_name

    model_manager.ensure_model(model_name, config)

    cached = model_manager._model_is_cached(
        model_name, model_manager._resolve_cache_dir(config)
    )
    _json_output({
        "status": "downloaded" if cached else "failed",
        "model": model_name,
    })


def cmd_delete_model(args: argparse.Namespace) -> None:
    """Delete a model from cache."""
    from codexlens_search import model_manager

    config = _create_config(args)
    model_name = args.model_name

    deleted = model_manager.delete_model(model_name, config)
    _json_output({
        "status": "deleted" if deleted else "not_found",
        "model": model_name,
    })


def cmd_status(args: argparse.Namespace) -> None:
    """Report index statistics."""
    from codexlens_search.indexing.metadata import MetadataStore

    db_path = _resolve_db_path(args)
    meta_path = db_path / "metadata.db"

    if not meta_path.exists():
        _json_output({
            "status": "not_initialized",
            "db_path": str(db_path),
        })
        return

    metadata = MetadataStore(meta_path)
    all_files = metadata.get_all_files()
    deleted_ids = metadata.get_deleted_ids()
    max_chunk = metadata.max_chunk_id()

    _json_output({
        "status": "ok",
        "db_path": str(db_path),
        "files_tracked": len(all_files),
        "max_chunk_id": max_chunk,
        "total_chunks_approx": max_chunk + 1 if max_chunk >= 0 else 0,
        "deleted_chunks": len(deleted_ids),
    })


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codexlens-search",
        description="Lightweight semantic code search - CLI bridge",
    )
    parser.add_argument(
        "--db-path",
        default=os.environ.get("CODEXLENS_DB_PATH", ".codexlens"),
        help="Path to index database directory (default: .codexlens or $CODEXLENS_DB_PATH)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging to stderr",
    )

    # API embedding overrides (also read from CODEXLENS_EMBED_API_* env vars)
    parser.add_argument(
        "--embed-api-url",
        default="",
        help="Remote embedding API URL (OpenAI-compatible, e.g. https://api.openai.com/v1)",
    )
    parser.add_argument(
        "--embed-api-key",
        default="",
        help="API key for remote embedding",
    )
    parser.add_argument(
        "--embed-api-model",
        default="",
        help="Model name for remote embedding (e.g. text-embedding-3-small)",
    )

    sub = parser.add_subparsers(dest="command")

    # init
    sub.add_parser("init", help="Initialize empty index")

    # search
    p_search = sub.add_parser("search", help="Search the index")
    p_search.add_argument("--query", "-q", required=True, help="Search query")
    p_search.add_argument("--top-k", "-k", type=int, default=10, help="Number of results")

    # index-file
    p_index = sub.add_parser("index-file", help="Index a single file")
    p_index.add_argument("--file", "-f", required=True, help="File path to index")
    p_index.add_argument("--root", "-r", help="Root directory for relative paths")

    # remove-file
    p_remove = sub.add_parser("remove-file", help="Remove a file from index")
    p_remove.add_argument("--file", "-f", required=True, help="Relative file path to remove")

    # sync
    p_sync = sub.add_parser("sync", help="Sync index with directory")
    p_sync.add_argument("--root", "-r", required=True, help="Root directory to sync")
    p_sync.add_argument("--glob", "-g", default="**/*", help="Glob pattern (default: **/*)")
    p_sync.add_argument(
        "--exclude", "-e", action="append", default=None,
        help="Directory names to exclude (repeatable). "
             "Defaults: node_modules, .git, __pycache__, dist, build, .venv, venv, .tox, .mypy_cache",
    )

    # watch
    p_watch = sub.add_parser("watch", help="Watch directory for changes (JSONL output)")
    p_watch.add_argument("--root", "-r", required=True, help="Root directory to watch")
    p_watch.add_argument("--debounce-ms", type=int, default=500, help="Debounce interval in ms")

    # download-models
    p_dl = sub.add_parser("download-models", help="Download embed + reranker models")
    p_dl.add_argument("--embed-model", help="Override embed model name")

    # list-models
    sub.add_parser("list-models", help="List known models with cache status")

    # download-model (single model by name)
    p_dl_single = sub.add_parser("download-model", help="Download a single model by name")
    p_dl_single.add_argument("model_name", help="HuggingFace model name (e.g. BAAI/bge-small-en-v1.5)")

    # delete-model
    p_del = sub.add_parser("delete-model", help="Delete a model from cache")
    p_del.add_argument("model_name", help="HuggingFace model name to delete")

    # status
    sub.add_parser("status", help="Report index statistics")

    return parser


def main() -> None:
    """CLI entry point."""
    _ensure_utf8_stdio()
    parser = _build_parser()
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(levelname)s %(name)s: %(message)s",
            stream=sys.stderr,
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s: %(message)s",
            stream=sys.stderr,
        )

    if not args.command:
        parser.print_help(sys.stderr)
        sys.exit(1)

    dispatch = {
        "init": cmd_init,
        "search": cmd_search,
        "index-file": cmd_index_file,
        "remove-file": cmd_remove_file,
        "sync": cmd_sync,
        "watch": cmd_watch,
        "download-models": cmd_download_models,
        "list-models": cmd_list_models,
        "download-model": cmd_download_model,
        "delete-model": cmd_delete_model,
        "status": cmd_status,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        _error_exit(f"Unknown command: {args.command}")

    try:
        handler(args)
    except KeyboardInterrupt:
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as exc:
        log.debug("Command failed", exc_info=True)
        _error_exit(str(exc))


if __name__ == "__main__":
    main()
