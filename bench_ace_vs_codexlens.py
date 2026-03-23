"""Benchmark: codexlens-search results for 20 queries against ground truth."""
import json
import time
from pathlib import Path
from codexlens_search.bridge import create_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent

QUERIES = [
    {"id": "Q1",  "query": "how does the embedding model load and initialize",
     "expected": ["embed/local.py"]},
    {"id": "Q2",  "query": "binary quantization and hamming distance search",
     "expected": ["core/binary.py", "core/faiss_index.py"]},
    {"id": "Q3",  "query": "reciprocal rank fusion merging multiple search results",
     "expected": ["search/fusion.py"]},
    {"id": "Q4",  "query": "chunking source code files for indexing pipeline",
     "expected": ["indexing/pipeline.py"]},
    {"id": "Q5",  "query": "HNSW approximate nearest neighbor index implementation",
     "expected": ["core/index.py", "core/usearch_index.py", "core/faiss_index.py"]},
    {"id": "Q6",  "query": "full text search with SQLite FTS5 exact and fuzzy",
     "expected": ["search/fts.py"]},
    {"id": "Q7",  "query": "thread safety locking concurrent access with RLock",
     "expected": ["core/usearch_index.py", "core/faiss_index.py"]},
    {"id": "Q8",  "query": "MCP server tools for code search and indexing",
     "expected": ["mcp_server.py"]},
    {"id": "Q9",  "query": "incremental file watcher detecting changes for re-index",
     "expected": ["watcher/file_watcher.py", "watcher/incremental_indexer.py"]},
    {"id": "Q10", "query": "search quality routing fast balanced thorough auto",
     "expected": ["search/pipeline.py"]},
    {"id": "Q11", "query": "GPU acceleration CUDA DirectML embedding providers",
     "expected": ["config.py", "embed/local.py"]},
    {"id": "Q12", "query": "gitignore filtering excluding files from indexing",
     "expected": ["indexing/gitignore.py"]},
    {"id": "Q13", "query": "metadata store tracking file changes and deleted chunks",
     "expected": ["indexing/metadata.py"]},
    {"id": "Q14", "query": "AST tree-sitter parsing extracting symbols from source code",
     "expected": ["parsers/parser.py", "parsers/symbols.py"]},
    {"id": "Q15", "query": "cross-encoder reranker scoring query document pairs",
     "expected": ["rerank/local.py"]},
    {"id": "Q16", "query": "shard partitioning large codebase across multiple indexes",
     "expected": ["core/shard.py", "core/shard_manager.py"]},
    {"id": "Q17", "query": "bridge creating search and indexing pipeline from config",
     "expected": ["bridge.py"]},
    {"id": "Q18", "query": "factory pattern selecting ANN backend usearch faiss hnswlib",
     "expected": ["core/factory.py"]},
    {"id": "Q19", "query": "API embedding endpoint with httpx batching and rate limiting",
     "expected": ["embed/api.py"]},
    {"id": "Q20", "query": "code-aware chunking with AST chunk_by_ast function",
     "expected": ["parsers/chunker.py"]},
]


def normalize_path(p):
    p = p.replace("\\", "/")
    for pfx in ("src/codexlens_search/", "codexlens_search/"):
        if p.startswith(pfx):
            p = p[len(pfx):]
    return p


def evaluate(expected, found_paths, top_k=10):
    expected_norm = [normalize_path(e) for e in expected]
    found_norm = [normalize_path(p) for p in found_paths[:top_k]]
    # Deduplicate found paths preserving order
    seen = set()
    deduped = []
    for p in found_norm:
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    hits = sum(1 for e in expected_norm if any(e in f for f in deduped))
    recall = hits / len(expected_norm) if expected_norm else 0.0
    mrr = 0.0
    for rank, fp in enumerate(deduped, 1):
        if any(e in fp for e in expected_norm):
            mrr = 1.0 / rank
            break
    top3_hit = any(any(e in fp for e in expected_norm) for fp in deduped[:3])
    return {"recall": recall, "mrr": mrr, "top3_hit": top3_hit, "found": deduped[:5]}


def main():
    db_path = PROJECT_ROOT / ".codexlens"
    ip, sp, config = create_pipeline(str(db_path), None)

    results = []
    t0 = time.monotonic()
    for q in QUERIES:
        search_results = sp.search(q["query"], top_k=20)
        found_paths = [r.path for r in search_results]
        metrics = evaluate(q["expected"], found_paths)
        results.append({
            "id": q["id"],
            "query": q["query"],
            "expected": q["expected"],
            **metrics,
        })
    elapsed = time.monotonic() - t0

    # Print results
    print("=" * 80)
    print("CODEXLENS-SEARCH RESULTS")
    print("=" * 80)
    total_recall = sum(r["recall"] for r in results) / len(results)
    total_mrr = sum(r["mrr"] for r in results) / len(results)
    total_top3 = sum(r["top3_hit"] for r in results) / len(results)
    zero_count = sum(1 for r in results if r["recall"] == 0)
    print(f"Avg Recall: {total_recall:.3f}  Avg MRR: {total_mrr:.3f}  "
          f"Top3 Rate: {total_top3:.3f}  Zero-recall: {zero_count}  Time: {elapsed:.1f}s")
    print()
    for r in results:
        status = "OK" if r["recall"] > 0 else "MISS"
        print(f"  {r['id']:<5} [{status:>4}] recall={r['recall']:.2f} mrr={r['mrr']:.3f} "
              f"found={r['found'][:3]}")

    # Save JSON
    output = {
        "tool": "codexlens-search",
        "version": "0.7.1",
        "total_queries": len(results),
        "avg_recall": round(total_recall, 4),
        "avg_mrr": round(total_mrr, 4),
        "top3_rate": round(total_top3, 4),
        "zero_recall_count": zero_count,
        "elapsed_seconds": round(elapsed, 2),
        "queries": results,
    }
    out_path = PROJECT_ROOT / "bench_codexlens_results.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
