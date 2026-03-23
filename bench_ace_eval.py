"""Evaluate ACE search_context results against ground truth and generate comparison report."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Same ground truth as bench_ace_vs_codexlens.py
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

# ACE results: manually extracted ranked file paths from search_context output
# Each entry is a list of unique file paths in order of appearance (top = first returned)
ACE_RESULTS = {
    "Q1": ["config.py", "README.md", "embed/local.py", "bridge.py", "pyproject.toml",
            "core/faiss_index.py", "model_manager.py"],
    "Q2": ["core/binary.py", "core/faiss_index.py", "core/index.py", "search/pipeline.py",
            "core/base.py", "config.py"],
    "Q3": ["search/fusion.py", "search/pipeline.py", "core/shard_manager.py"],
    "Q4": ["indexing/pipeline.py", "config.py", "bridge.py", "parsers/chunker.py"],
    "Q5": ["core/index.py", "core/usearch_index.py", "core/faiss_index.py", "config.py",
            "core/factory.py", "README.md", "core/base.py"],
    "Q6": ["search/fts.py", "search/pipeline.py", "indexing/pipeline.py"],
    "Q7": ["core/usearch_index.py", "core/faiss_index.py", "core/index.py",
            "watcher/file_watcher.py", "embed/api.py"],
    "Q8": ["mcp_server.py", "bridge.py", "README.md"],
    "Q9": ["watcher/file_watcher.py", "watcher/incremental_indexer.py", "watcher/events.py",
            "mcp_server.py"],
    "Q10": ["search/pipeline.py", "config.py", "search/fusion.py", "README.md"],
    "Q11": ["config.py", "README.md", "embed/local.py", "bridge.py", "pyproject.toml",
             "embed/api.py", "core/faiss_index.py", "model_manager.py"],
    "Q12": ["indexing/pipeline.py", "indexing/gitignore.py", "watcher/events.py",
             "config.py", "bridge.py", "watcher/file_watcher.py"],
    "Q13": ["indexing/pipeline.py", "watcher/file_watcher.py", "watcher/incremental_indexer.py",
             "indexing/metadata.py", "mcp_server.py", "search/fts.py"],
    "Q14": ["parsers/symbols.py", "indexing/pipeline.py", "parsers/chunker.py",
             "parsers/parser.py", "parsers/__init__.py", "parsers/references.py"],
    "Q15": ["rerank/api.py", "rerank/base.py", "rerank/local.py", "search/pipeline.py",
             "model_manager.py", "bridge.py"],
    "Q16": ["core/shard_manager.py", "config.py", "core/shard.py", "indexing/pipeline.py",
             "bridge.py"],
    "Q17": ["bridge.py", "search/pipeline.py", "indexing/pipeline.py", "config.py",
             "core/factory.py"],
    "Q18": ["core/factory.py", "core/usearch_index.py", "core/faiss_index.py",
             "README.md", "config.py", "core/index.py"],
    "Q19": ["embed/api.py", "config.py", "bridge.py", "search/pipeline.py",
             "core/binary.py", "search/fusion.py", "indexing/pipeline.py", "embed/local.py"],
    "Q20": ["parsers/chunker.py", "indexing/pipeline.py", "parsers/symbols.py",
             "parsers/__init__.py", "parsers/parser.py"],
}


def normalize_path(p):
    p = p.replace("\\", "/")
    for pfx in ("src/codexlens_search/", "codexlens_search/", "src\\codexlens_search\\"):
        if p.startswith(pfx):
            p = p[len(pfx):]
    return p


def evaluate(expected, found_paths, top_k=10):
    expected_norm = [normalize_path(e) for e in expected]
    found_norm = [normalize_path(p) for p in found_paths[:top_k]]
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
    # Evaluate ACE
    ace_results = []
    for q in QUERIES:
        found = ACE_RESULTS.get(q["id"], [])
        metrics = evaluate(q["expected"], found)
        ace_results.append({
            "id": q["id"],
            "query": q["query"],
            "expected": q["expected"],
            **metrics,
        })

    # Load codexlens results
    cl_path = PROJECT_ROOT / "bench_codexlens_results.json"
    cl_data = json.loads(cl_path.read_text())
    cl_results = cl_data["queries"]

    # Compute ACE aggregates
    ace_recall = sum(r["recall"] for r in ace_results) / len(ace_results)
    ace_mrr = sum(r["mrr"] for r in ace_results) / len(ace_results)
    ace_top3 = sum(r["top3_hit"] for r in ace_results) / len(ace_results)
    ace_zero = sum(1 for r in ace_results if r["recall"] == 0)

    # Codexlens aggregates (from saved results)
    cl_recall = cl_data["avg_recall"]
    cl_mrr = cl_data["avg_mrr"]
    cl_top3 = cl_data["top3_rate"]
    cl_zero = cl_data["zero_recall_count"]

    # Print comparison
    print("=" * 80)
    print("ACE vs CODEXLENS-SEARCH COMPARISON REPORT")
    print("=" * 80)
    print(f"Project: codexlens-search v{cl_data['version']}")
    print(f"Test queries: {len(QUERIES)}")
    print()

    print("-" * 80)
    print(f"{'Metric':<25} {'ACE':>12} {'Codexlens':>12} {'Winner':>12}")
    print("-" * 80)

    def winner(a, b, higher_better=True):
        if abs(a - b) < 0.001:
            return "TIE"
        if higher_better:
            return "ACE" if a > b else "Codexlens"
        else:
            return "ACE" if a < b else "Codexlens"

    print(f"{'Avg Recall':<25} {ace_recall:>12.4f} {cl_recall:>12.4f} {winner(ace_recall, cl_recall):>12}")
    print(f"{'Avg MRR':<25} {ace_mrr:>12.4f} {cl_mrr:>12.4f} {winner(ace_mrr, cl_mrr):>12}")
    print(f"{'Top-3 Hit Rate':<25} {ace_top3:>12.4f} {cl_top3:>12.4f} {winner(ace_top3, cl_top3):>12}")
    print(f"{'Zero-recall Queries':<25} {ace_zero:>12d} {cl_zero:>12d} {winner(ace_zero, cl_zero, False):>12}")
    print("-" * 80)
    print()

    # Per-query comparison
    print("PER-QUERY DETAIL")
    print("-" * 80)
    print(f"{'ID':<5} {'ACE Recall':>10} {'CL Recall':>10} {'ACE MRR':>8} {'CL MRR':>8} {'Winner':>10}")
    print("-" * 80)

    ace_wins = 0
    cl_wins = 0
    ties = 0

    for ar, cr in zip(ace_results, cl_results):
        w = winner(ar["recall"] + ar["mrr"], cr["recall"] + cr["mrr"])
        if w == "ACE":
            ace_wins += 1
        elif w == "Codexlens":
            cl_wins += 1
        else:
            ties += 1
        print(f"{ar['id']:<5} {ar['recall']:>10.2f} {cr['recall']:>10.2f} "
              f"{ar['mrr']:>8.3f} {cr['mrr']:>8.3f} {w:>10}")

    print("-" * 80)
    print(f"Query wins: ACE={ace_wins}  Codexlens={cl_wins}  Tie={ties}")
    print()

    # Ranking quality analysis
    print("RANKING QUALITY (where expected file appears)")
    print("-" * 80)
    for ar, cr in zip(ace_results, cl_results):
        ace_rank = "MISS"
        cl_rank = "MISS"
        for i, fp in enumerate(ar["found"], 1):
            if any(e in fp for e in ar["expected"]):
                ace_rank = f"#{i}"
                break
        for i, fp in enumerate(cr["found"], 1):
            if any(e in fp for e in cr["expected"]):
                cl_rank = f"#{i}"
                break
        print(f"  {ar['id']:<5} ACE={ace_rank:<6} CL={cl_rank:<6} expected={ar['expected']}")

    # Save JSON
    output = {
        "comparison": "ACE vs codexlens-search",
        "project": "codexlens-search",
        "version": cl_data["version"],
        "total_queries": len(QUERIES),
        "summary": {
            "ace": {
                "avg_recall": round(ace_recall, 4),
                "avg_mrr": round(ace_mrr, 4),
                "top3_rate": round(ace_top3, 4),
                "zero_recall": ace_zero,
            },
            "codexlens": {
                "avg_recall": cl_recall,
                "avg_mrr": cl_mrr,
                "top3_rate": cl_top3,
                "zero_recall": cl_zero,
            },
            "query_wins": {
                "ace": ace_wins,
                "codexlens": cl_wins,
                "tie": ties,
            },
        },
        "ace_queries": ace_results,
        "codexlens_queries": cl_results,
    }
    out_path = PROJECT_ROOT / "bench_comparison_report.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
