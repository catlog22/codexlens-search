"""Benchmark: local model vs API embedding + indexing speed.

Usage:
    python bench_index.py

Sources API config from .mcp.json (SiliconFlow).

Tests:
    1. ANN backend comparison (usearch vs faiss vs hnswlib)
    2. Embedding speed: local (bge-small-en, 384d) vs API (bge-large-zh, 1024d)
    3. Full pipeline index: local+usearch vs API+usearch
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Config — load from .mcp.json
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent / "src"
BENCH_DIR = Path(__file__).parent / ".bench_tmp"

LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
LOCAL_DIM = 384


def _load_mcp_config() -> dict:
    """Load API config from .mcp.json codexlens server env."""
    for mcp_path in [
        Path(__file__).parent / ".mcp.json",
        Path(__file__).parent.parent / ".mcp.json",
    ]:
        if mcp_path.exists():
            data = json.loads(mcp_path.read_text(encoding="utf-8"))
            env = data.get("mcpServers", {}).get("codexlens", {}).get("env", {})
            if env.get("CODEXLENS_EMBED_API_URL"):
                return env
    return {}


MCP_ENV = _load_mcp_config()
API_URL = MCP_ENV.get("CODEXLENS_EMBED_API_URL", "")
API_KEY = MCP_ENV.get("CODEXLENS_EMBED_API_KEY", "")
API_MODEL = MCP_ENV.get("CODEXLENS_EMBED_API_MODEL", "")
API_DIM = int(MCP_ENV.get("CODEXLENS_EMBED_DIM", "1024"))


def _collect_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def _read_chunks(files: list[Path], max_chars: int = 2000) -> list[str]:
    chunks = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for i in range(0, len(text), max_chars):
            chunk = text[i:i + max_chars].strip()
            if chunk:
                chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Benchmark: Embedding only
# ---------------------------------------------------------------------------
def bench_embed_local(chunks: list[str]) -> dict:
    from codexlens_search.config import Config
    from codexlens_search.embed.local import FastEmbedEmbedder

    config = Config(embed_model=LOCAL_MODEL, embed_dim=LOCAL_DIM, embed_batch_size=32)
    embedder = FastEmbedEmbedder(config)

    print("  [local] Loading model...")
    t0 = time.perf_counter()
    embedder.embed_single("warmup")
    load_time = time.perf_counter() - t0
    print(f"  [local] Model loaded in {load_time:.2f}s")

    print(f"  [local] Embedding {len(chunks)} chunks...")
    t0 = time.perf_counter()
    vecs = embedder.embed_batch(chunks)
    elapsed = time.perf_counter() - t0

    return {
        "backend": f"local ({LOCAL_MODEL.split('/')[-1]})",
        "dim": LOCAL_DIM,
        "chunks": len(chunks),
        "model_load_s": round(load_time, 3),
        "embed_s": round(elapsed, 3),
        "chunks_per_s": round(len(chunks) / elapsed, 1),
    }


def bench_embed_api(chunks: list[str]) -> dict | None:
    if not API_KEY:
        print("  [api] SKIPPED — no API key in .mcp.json")
        return None

    from codexlens_search.config import Config
    from codexlens_search.embed.api import APIEmbedder

    config = Config(
        embed_api_url=API_URL,
        embed_api_key=API_KEY,
        embed_api_model=API_MODEL,
        embed_dim=API_DIM,
        embed_batch_size=4,
        embed_api_concurrency=2,
        embed_api_max_tokens_per_batch=2048,
        embed_max_tokens=256,
    )
    embedder = APIEmbedder(config)

    print(f"  [api] Warmup ({API_URL})...")
    t0 = time.perf_counter()
    embedder.embed_single("warmup text for dimension probe")
    warmup_s = time.perf_counter() - t0
    print(f"  [api] Warmup done in {warmup_s:.2f}s")

    # Truncate chunks to 1024 chars max for SiliconFlow compatibility
    api_chunks = [c[:1024] for c in chunks]
    print(f"  [api] Embedding {len(api_chunks)} chunks (truncated to 1024 chars)...")
    t0 = time.perf_counter()
    vecs = embedder.embed_batch(api_chunks)
    elapsed = time.perf_counter() - t0

    return {
        "backend": f"api ({API_MODEL.split('/')[-1]})",
        "dim": API_DIM,
        "chunks": len(chunks),
        "warmup_s": round(warmup_s, 3),
        "embed_s": round(elapsed, 3),
        "chunks_per_s": round(len(chunks) / elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Benchmark: Full pipeline index
# ---------------------------------------------------------------------------
def bench_pipeline_index(label: str, config, files: list[Path]) -> dict:
    from codexlens_search.bridge import create_pipeline

    db = BENCH_DIR / label
    if db.exists():
        shutil.rmtree(db)
    db.mkdir(parents=True)

    print(f"  [{label}] Creating pipeline...")
    t0 = time.perf_counter()
    indexing, search, cfg = create_pipeline(db, config)
    create_s = time.perf_counter() - t0

    print(f"  [{label}] Indexing {len(files)} files...")
    t0 = time.perf_counter()
    result = indexing.sync(files, root=PROJECT_ROOT)
    index_s = time.perf_counter() - t0

    print(f"  [{label}] Searching...")
    t0 = time.perf_counter()
    results = search.search("embedding pipeline config")
    search_s = time.perf_counter() - t0

    files_done = result.files_processed if hasattr(result, "files_processed") else len(files)
    chunks_done = result.chunks_created if hasattr(result, "chunks_created") else "?"
    ann_name = type(search._ann_index).__name__ if hasattr(search, "_ann_index") else "?"

    print(f"  [{label}] Done: {files_done} files, {chunks_done} chunks, index={index_s:.1f}s, search={search_s:.3f}s")

    return {
        "label": label,
        "files": files_done,
        "chunks": chunks_done,
        "create_s": round(create_s, 3),
        "index_s": round(index_s, 3),
        "search_s": round(search_s, 3),
        "search_results": len(results),
        "ann_backend": ann_name,
    }


# ---------------------------------------------------------------------------
# Benchmark: ANN backends
# ---------------------------------------------------------------------------
def bench_ann_backends(dim: int = 384, n_vecs: int = 5000, n_queries: int = 100) -> list[dict]:
    from codexlens_search.config import Config

    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n_vecs, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    ids = np.arange(n_vecs, dtype=np.int64)
    queries = vecs[rng.choice(n_vecs, n_queries, replace=False)]

    results = []
    for backend in ["usearch", "faiss", "hnswlib"]:
        cfg = Config(embed_dim=dim, ann_backend=backend, ann_top_k=20)
        db = BENCH_DIR / f"ann_{backend}"
        if db.exists():
            shutil.rmtree(db)
        db.mkdir(parents=True)

        try:
            from codexlens_search.core.factory import create_ann_index
            idx = create_ann_index(db, dim, cfg)
        except ImportError:
            print(f"  [{backend}] Not available, skipping")
            continue

        t0 = time.perf_counter()
        idx.add(ids, vecs)
        add_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        idx.save()
        save_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        for q in queries:
            idx.fine_search(q, top_k=20)
        search_s = time.perf_counter() - t0

        results.append({
            "backend": backend,
            "impl": type(idx).__name__,
            "add_s": round(add_s, 4),
            "save_s": round(save_s, 4),
            "search_s": round(search_s, 4),
            "qps": round(n_queries / search_s, 1),
        })
        print(f"  [{backend:<8}] add={add_s:.3f}s  save={save_s:.3f}s  search={search_s:.3f}s ({n_queries/search_s:.0f} qps)")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if BENCH_DIR.exists():
        shutil.rmtree(BENCH_DIR)
    BENCH_DIR.mkdir(parents=True)

    files = _collect_files(PROJECT_ROOT)
    chunks = _read_chunks(files)

    print(f"Project: {PROJECT_ROOT}")
    print(f"Files: {len(files)}, Chunks: {len(chunks)}")
    if API_URL:
        print(f"API: {API_URL} / {API_MODEL} (dim={API_DIM})")
    else:
        print("API: not configured (no .mcp.json)")
    print()

    # ---- 1. ANN backend comparison ----
    print("=" * 65)
    print("BENCHMARK 1: ANN Backend (5000 vecs, 100 queries, dim=384)")
    print("=" * 65)
    ann_results = bench_ann_backends()
    print()

    # ---- 2. Embedding speed ----
    print("=" * 65)
    print(f"BENCHMARK 2: Embedding Speed ({len(chunks)} chunks)")
    print("=" * 65)

    print("\n--- Local Model ---")
    local_result = bench_embed_local(chunks)
    print(f"  => {local_result['embed_s']}s, {local_result['chunks_per_s']} chunks/s")

    print("\n--- API Model ---")
    api_result = bench_embed_api(chunks)
    if api_result:
        print(f"  => {api_result['embed_s']}s, {api_result['chunks_per_s']} chunks/s")

    # ---- 3. Full pipeline index ----
    print()
    print("=" * 65)
    print(f"BENCHMARK 3: Full Pipeline Index ({len(files)} files)")
    print("=" * 65)

    from codexlens_search.config import Config

    print("\n--- Local + USearch ---")
    local_cfg = Config(
        embed_model=LOCAL_MODEL, embed_dim=LOCAL_DIM,
        ann_backend="usearch",
    )
    local_pipe = bench_pipeline_index("local+usearch", local_cfg, files)

    api_pipe = None
    if API_KEY:
        print("\n--- API + USearch ---")
        api_cfg = Config(
            embed_api_url=API_URL, embed_api_key=API_KEY, embed_api_model=API_MODEL,
            embed_dim=API_DIM, ann_backend="usearch",
            embed_batch_size=8, embed_api_concurrency=4,
            embed_api_max_tokens_per_batch=4096,
            embed_max_tokens=512,
        )
        api_pipe = bench_pipeline_index("api+usearch", api_cfg, files)

    # ---- Summary ----
    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)

    print("\n[ANN Backend] 5000 vecs, 100 queries, dim=384")
    print(f"  {'Backend':<10} {'Add':<10} {'Save':<10} {'Search':<10} {'QPS':<10}")
    print(f"  {'-'*50}")
    for r in ann_results:
        print(f"  {r['backend']:<10} {r['add_s']:<10.4f} {r['save_s']:<10.4f} {r['search_s']:<10.4f} {r['qps']:<10.1f}")

    print(f"\n[Embedding] {len(chunks)} chunks")
    print(f"  {'Backend':<28} {'Time(s)':<10} {'Chunks/s':<12} {'Dim':<6}")
    print(f"  {'-'*56}")
    print(f"  {local_result['backend']:<28} {local_result['embed_s']:<10} {local_result['chunks_per_s']:<12} {local_result['dim']:<6}")
    if api_result:
        print(f"  {api_result['backend']:<28} {api_result['embed_s']:<10} {api_result['chunks_per_s']:<12} {api_result['dim']:<6}")
        speedup = local_result['embed_s'] / api_result['embed_s'] if api_result['embed_s'] > 0 else 0
        print(f"  API is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than local")

    print(f"\n[Full Pipeline] {len(files)} files")
    print(f"  {'Pipeline':<16} {'Files':<7} {'Chunks':<8} {'Index(s)':<10} {'Search(s)':<10} {'ANN'}")
    print(f"  {'-'*65}")
    print(f"  {local_pipe['label']:<16} {local_pipe['files']:<7} {local_pipe['chunks']:<8} {local_pipe['index_s']:<10} {local_pipe['search_s']:<10} {local_pipe['ann_backend']}")
    if api_pipe:
        print(f"  {api_pipe['label']:<16} {api_pipe['files']:<7} {api_pipe['chunks']:<8} {api_pipe['index_s']:<10} {api_pipe['search_s']:<10} {api_pipe['ann_backend']}")
        speedup = local_pipe['index_s'] / api_pipe['index_s'] if api_pipe['index_s'] > 0 else 0
        print(f"  API pipeline is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} for indexing")

    # Cleanup
    if BENCH_DIR.exists():
        shutil.rmtree(BENCH_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
