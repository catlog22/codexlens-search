"""Build bench_complex_ace.json from collected ACE search_context results."""
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Import queries and evaluate from the main benchmark
from bench_complex_comparison import QUERIES, evaluate

# ACE results: unique file paths in order of appearance from search_context output
ACE_RESULTS = {
    "CQ1": ["search/pipeline.py", "search/fusion.py", "bridge.py", "core/binary.py",
             "core/faiss_index.py", "search/graph.py", "core/usearch_index.py", "core/index.py"],
    "CQ2": ["core/shard.py", "core/base.py", "rerank/base.py", "embed/base.py", "bridge.py",
             "core/factory.py", "embed/api.py", "rerank/local.py", "embed/local.py", "indexing/pipeline.py"],
    "CQ3": ["core/shard_manager.py", "core/shard.py", "config.py", "search/pipeline.py",
             "search/fusion.py", "indexing/pipeline.py"],
    "CQ4": ["config.py", "core/factory.py", "bridge.py", "embed/local.py", "embed/api.py",
             "rerank/local.py", "rerank/api.py", "search/pipeline.py"],
    "CQ5": ["watcher/incremental_indexer.py", "watcher/file_watcher.py", "watcher/events.py",
             "mcp_server.py", "indexing/pipeline.py", "core/index.py"],
    "CQ6": ["indexing/pipeline.py", "config.py", "core/binary.py", "core/factory.py"],
    "CQ7": ["parsers/chunker.py", "indexing/pipeline.py", "parsers/symbols.py",
             "parsers/parser.py", "config.py"],
    "CQ8": ["search/pipeline.py", "mcp_server.py", "core/factory.py", "bridge.py",
             "indexing/pipeline.py", "core/index.py", "search/fts.py"],
    "CQ9": ["embed/api.py", "rerank/api.py", "bridge.py", "config.py"],
    "CQ10": ["search/fusion.py", "search/pipeline.py", "config.py", "mcp_server.py",
              "core/shard_manager.py"],
    "CQ11": ["indexing/pipeline.py", "parsers/chunker.py", "parsers/symbols.py",
              "parsers/parser.py", "embed/api.py", "config.py"],
    "CQ12": ["parsers/references.py", "parsers/symbols.py", "indexing/pipeline.py",
              "search/graph.py", "search/fts.py"],
    "CQ13": ["model_manager.py", "config.py", "bridge.py", "README.md"],
    "CQ14": ["core/binary.py", "core/faiss_index.py", "core/base.py", "search/pipeline.py",
              "config.py", "core/index.py", "core/factory.py"],
    "CQ15": ["config.py", "bridge.py", "embed/local.py", "core/factory.py", "mcp_server.py"],
    "CQ16": ["mcp_server.py", "indexing/pipeline.py", "bridge.py", "search/pipeline.py"],
    "CQ17": ["indexing/pipeline.py", "embed/api.py", "search/pipeline.py", "parsers/symbols.py"],
    "CQ18": ["search/graph.py", "search/pipeline.py", "mcp_server.py", "search/fts.py",
              "search/fusion.py", "core/shard_manager.py", "indexing/pipeline.py", "config.py"],
    "CQ19": ["watcher/incremental_indexer.py", "indexing/pipeline.py", "indexing/metadata.py",
              "bridge.py", "mcp_server.py"],
    "CQ20": ["embed/api.py", "config.py", "rerank/api.py", "bridge.py",
              "core/shard_manager.py", "README.md"],
}


def main():
    results = []
    for q in QUERIES:
        found = ACE_RESULTS.get(q["id"], [])
        metrics = evaluate(q["expected"], found)
        results.append({
            "id": q["id"],
            "category": q["category"],
            "query": q["query"],
            "expected": q["expected"],
            "difficulty": q["difficulty"],
            **metrics,
        })

    n = len(results)
    output = {
        "tool": "ace-search-context",
        "total_queries": n,
        "avg_recall": round(sum(r["recall"] for r in results) / n, 4),
        "avg_mrr": round(sum(r["mrr"] for r in results) / n, 4),
        "avg_ndcg5": round(sum(r["ndcg5"] for r in results) / n, 4),
        "top3_rate": round(sum(r["top3_hit"] for r in results) / n, 4),
        "zero_recall": sum(1 for r in results if r["recall"] == 0),
        "queries": results,
    }

    out_path = PROJECT_ROOT / "bench_complex_ace.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Saved: {out_path}")

    # Print summary
    print(f"\nACE Complex Benchmark Summary:")
    print(f"  Avg Recall: {output['avg_recall']:.4f}")
    print(f"  Avg MRR:    {output['avg_mrr']:.4f}")
    print(f"  Avg NDCG@5: {output['avg_ndcg5']:.4f}")
    print(f"  Top-3 Rate: {output['top3_rate']:.4f}")
    print(f"  Zero-recall: {output['zero_recall']}")

    # Per-query detail
    print(f"\n{'ID':<5} {'Recall':>7} {'MRR':>7} {'NDCG5':>7} {'Top3':>5}  Found (top 3)")
    print("-" * 70)
    for r in results:
        status = "OK" if r["recall"] > 0 else "MISS"
        print(f"  {r['id']:<5} {r['recall']:>6.2f} {r['mrr']:>6.3f} {r['ndcg5']:>6.3f} "
              f"{'Y' if r['top3_hit'] else 'N':>4}  {r['found'][:3]}")


if __name__ == "__main__":
    main()
