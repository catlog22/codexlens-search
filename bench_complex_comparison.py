"""Complex Benchmark: 20 advanced search queries comparing ACE MCP vs codexlens-search.

Query design categories:
  - Cross-module architectural queries (span multiple modules)
  - Abstract behavioral queries (describe behavior, not file names)
  - Negative/inverse queries (what handles absence/failure)
  - Implementation detail queries (specific algorithms/patterns)
  - Integration flow queries (end-to-end data paths)

Usage:
  1. Run codexlens-search directly:  python bench_complex_comparison.py --codexlens
  2. Collect ACE results manually:    python bench_complex_comparison.py --ace-collect
  3. After filling ACE results:       python bench_complex_comparison.py --report
"""
import argparse
import json
import time
from pathlib import Path

from codexlens_search.bridge import create_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent

# ===========================================================================
# 20 Complex Queries — designed to stress-test semantic understanding
# ===========================================================================
QUERIES = [
    # --- Cross-module architectural queries ---
    {
        "id": "CQ1",
        "category": "cross-module",
        "query": "how does a natural language query flow from user input through embedding, binary coarse filter, ANN refinement, FTS matching, fusion, and reranking to produce final ranked results",
        "expected": ["search/pipeline.py", "search/fusion.py", "core/binary.py", "core/index.py"],
        "difficulty": "hard",
        "note": "End-to-end search data flow spanning 4+ modules",
    },
    {
        "id": "CQ2",
        "category": "cross-module",
        "query": "abstract base classes and interface contracts that define the plugin architecture for swappable backends",
        "expected": ["core/base.py", "embed/base.py", "rerank/base.py"],
        "difficulty": "hard",
        "note": "ABC pattern scattered across 3 separate packages",
    },
    {
        "id": "CQ3",
        "category": "cross-module",
        "query": "LRU eviction policy for shard lifecycle management when memory limit is exceeded during parallel search",
        "expected": ["core/shard_manager.py", "core/shard.py"],
        "difficulty": "medium",
        "note": "Combines memory management + concurrency + data structure concepts",
    },
    # --- Abstract behavioral queries (no file/class name hints) ---
    {
        "id": "CQ4",
        "category": "behavioral",
        "query": "how the system decides whether to use local ONNX inference or remote HTTP API for turning text into dense vectors",
        "expected": ["embed/local.py", "embed/api.py", "bridge.py"],
        "difficulty": "hard",
        "note": "Decision logic for local vs API embedding — no technical terms used",
    },
    {
        "id": "CQ5",
        "category": "behavioral",
        "query": "debouncing rapid filesystem notifications before triggering incremental re-processing of changed files",
        "expected": ["watcher/file_watcher.py", "watcher/events.py"],
        "difficulty": "medium",
        "note": "Describes debounce behavior without naming classes",
    },
    {
        "id": "CQ6",
        "category": "behavioral",
        "query": "detecting whether a file contains machine-generated code or binary content and skipping it during processing",
        "expected": ["indexing/pipeline.py", "config.py"],
        "difficulty": "medium",
        "note": "Binary detection + generated code markers — behavioral description",
    },
    {
        "id": "CQ7",
        "category": "behavioral",
        "query": "splitting source code at meaningful boundaries like function definitions and class declarations rather than fixed character counts",
        "expected": ["parsers/chunker.py", "indexing/pipeline.py"],
        "difficulty": "medium",
        "note": "Code-aware chunking described without technical terms",
    },
    # --- Negative/inverse queries ---
    {
        "id": "CQ8",
        "category": "negative",
        "query": "graceful fallback when the vector index is empty or missing and only keyword search is available",
        "expected": ["search/pipeline.py", "mcp_server.py"],
        "difficulty": "hard",
        "note": "Tests handling of absence/degraded mode",
    },
    {
        "id": "CQ9",
        "category": "negative",
        "query": "retry logic and error recovery when external API calls for embedding or reranking fail with network errors",
        "expected": ["embed/api.py", "rerank/api.py"],
        "difficulty": "medium",
        "note": "Error handling pattern across two API client modules",
    },
    # --- Implementation detail queries ---
    {
        "id": "CQ10",
        "category": "impl-detail",
        "query": "adaptive weight adjustment for fusion based on whether the query contains code identifiers versus natural language",
        "expected": ["search/fusion.py"],
        "difficulty": "hard",
        "note": "Query intent detection + adaptive weights — specific algorithm detail",
    },
    {
        "id": "CQ11",
        "category": "impl-detail",
        "query": "prepending file path and enclosing class or function name as context header to each chunk before embedding",
        "expected": ["indexing/pipeline.py"],
        "difficulty": "medium",
        "note": "chunk_context_header / _inject_context_headers detail",
    },
    {
        "id": "CQ12",
        "category": "impl-detail",
        "query": "mapping Python import statements and function calls to directed graph edges between symbols for structural code search",
        "expected": ["parsers/references.py", "search/graph.py"],
        "difficulty": "hard",
        "note": "Cross-reference extraction → graph traversal pipeline",
    },
    {
        "id": "CQ13",
        "category": "impl-detail",
        "query": "HuggingFace mirror configuration and ONNX model cache validation to avoid redundant downloads",
        "expected": ["model_manager.py", "config.py"],
        "difficulty": "medium",
        "note": "Model download manager details — niche topic",
    },
    {
        "id": "CQ14",
        "category": "impl-detail",
        "query": "converting high-dimensional float32 vectors to compact binary representations using sign-based quantization for fast hamming pre-filtering",
        "expected": ["core/binary.py"],
        "difficulty": "medium",
        "note": "Binary quantization internals described in detail",
    },
    # --- Integration flow queries ---
    {
        "id": "CQ15",
        "category": "integration",
        "query": "environment variable and dataclass configuration cascade that controls embedding model selection, GPU device, and index parameters",
        "expected": ["config.py", "bridge.py"],
        "difficulty": "hard",
        "note": "Config loading from env → dataclass → component init",
    },
    {
        "id": "CQ16",
        "category": "integration",
        "query": "how the MCP server lazily creates search pipelines per project and manages their lifecycle with background indexing triggers",
        "expected": ["mcp_server.py"],
        "difficulty": "medium",
        "note": "MCP pipeline management — lazy init + background indexing",
    },
    {
        "id": "CQ17",
        "category": "integration",
        "query": "parallel worker pool that simultaneously chunks files, extracts symbols, computes embeddings, and stores results in the database",
        "expected": ["indexing/pipeline.py"],
        "difficulty": "medium",
        "note": "Concurrent indexing pipeline with ThreadPoolExecutor",
    },
    {
        "id": "CQ18",
        "category": "integration",
        "query": "seed symbol discovery from initial search results followed by multi-hop graph traversal to find structurally related code",
        "expected": ["search/graph.py", "search/pipeline.py"],
        "difficulty": "hard",
        "note": "Graph expansion from vector/FTS seeds — 2-phase search",
    },
    {
        "id": "CQ19",
        "category": "integration",
        "query": "incremental sync that detects added modified and deleted files then updates only affected chunks and metadata without full re-index",
        "expected": ["indexing/pipeline.py", "indexing/metadata.py", "watcher/incremental_indexer.py"],
        "difficulty": "hard",
        "note": "Incremental update spanning metadata + indexing + watcher",
    },
    {
        "id": "CQ20",
        "category": "integration",
        "query": "multi-endpoint round-robin load balancing for distributing embedding API requests across multiple providers with token-aware batching",
        "expected": ["embed/api.py", "config.py"],
        "difficulty": "hard",
        "note": "Multi-endpoint rotation + token batching — advanced API embedding feature",
    },
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
    # NDCG@5 — considers position of ALL expected files, not just first
    dcg = 0.0
    for rank, fp in enumerate(deduped[:5], 1):
        if any(e in fp for e in expected_norm):
            dcg += 1.0 / _log2(rank + 1)
    idcg = sum(1.0 / _log2(i + 2) for i in range(min(len(expected_norm), 5)))
    ndcg5 = dcg / idcg if idcg > 0 else 0.0
    return {
        "recall": recall,
        "mrr": mrr,
        "top3_hit": top3_hit,
        "ndcg5": round(ndcg5, 4),
        "found": deduped[:7],
    }


def _log2(x):
    import math
    return math.log2(x)


# ===========================================================================
# Phase 1: Run codexlens-search
# ===========================================================================
def run_codexlens():
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
            "category": q["category"],
            "query": q["query"],
            "expected": q["expected"],
            "difficulty": q["difficulty"],
            **metrics,
        })
    elapsed = time.monotonic() - t0

    # Print
    print("=" * 90)
    print("CODEXLENS-SEARCH — COMPLEX BENCHMARK (20 queries)")
    print("=" * 90)
    _print_summary(results, elapsed)
    _print_details(results)

    # Save
    output = {
        "tool": "codexlens-search",
        "total_queries": len(results),
        "elapsed_seconds": round(elapsed, 2),
        **_aggregate(results),
        "queries": results,
    }
    out_path = PROJECT_ROOT / "bench_complex_codexlens.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {out_path}")


# ===========================================================================
# Phase 2: Scaffold for ACE results collection
# ===========================================================================
ACE_RESULTS: dict[str, list[str]] = {}  # Filled after manual ACE collection


def generate_ace_scaffold():
    """Print queries for manual ACE testing."""
    print("=" * 90)
    print("ACE COLLECTION SCAFFOLD")
    print("Copy each query into mcp__ace-tool__search_context and record results")
    print("=" * 90)
    for q in QUERIES:
        print(f"\n--- {q['id']} [{q['category']}] ---")
        print(f"Query: {q['query']}")
        print(f"Expected: {q['expected']}")
    print("\nAfter collection, fill ACE_RESULTS dict and run --report")


# ===========================================================================
# Phase 3: Generate comparison report
# ===========================================================================
def generate_report():
    # Load codexlens results
    cl_path = PROJECT_ROOT / "bench_complex_codexlens.json"
    if not cl_path.exists():
        print("ERROR: Run --codexlens first")
        return
    cl_data = json.loads(cl_path.read_text())

    # Load ACE results
    ace_path = PROJECT_ROOT / "bench_complex_ace.json"
    if not ace_path.exists():
        print("ERROR: bench_complex_ace.json not found. Run ACE collection first.")
        return
    ace_data = json.loads(ace_path.read_text())

    cl_queries = {r["id"]: r for r in cl_data["queries"]}
    ace_queries = {r["id"]: r for r in ace_data["queries"]}

    # Build comparison
    rows = []
    for q in QUERIES:
        qid = q["id"]
        cl = cl_queries.get(qid, {})
        ace = ace_queries.get(qid, {})
        rows.append({
            "id": qid,
            "category": q["category"],
            "difficulty": q["difficulty"],
            "query": q["query"][:60],
            "expected": q["expected"],
            "ace_recall": ace.get("recall", 0),
            "cl_recall": cl.get("recall", 0),
            "ace_mrr": ace.get("mrr", 0),
            "cl_mrr": cl.get("mrr", 0),
            "ace_ndcg5": ace.get("ndcg5", 0),
            "cl_ndcg5": cl.get("ndcg5", 0),
            "ace_top3": ace.get("top3_hit", False),
            "cl_top3": cl.get("top3_hit", False),
            "ace_found": ace.get("found", [])[:5],
            "cl_found": cl.get("found", [])[:5],
        })

    # Print report
    print("=" * 100)
    print("COMPLEX BENCHMARK: ACE vs CODEXLENS-SEARCH (20 Advanced Queries)")
    print("=" * 100)

    # Overall
    ace_avg = lambda key: sum(r[f"ace_{key}"] for r in rows) / len(rows)
    cl_avg = lambda key: sum(r[f"cl_{key}"] for r in rows) / len(rows)

    print(f"\n{'Metric':<22} {'ACE':>10} {'Codexlens':>10} {'Delta':>10} {'Winner':>12}")
    print("-" * 64)
    for metric in ["recall", "mrr", "ndcg5"]:
        a, c = ace_avg(metric), cl_avg(metric)
        delta = a - c
        w = "ACE" if delta > 0.005 else ("Codexlens" if delta < -0.005 else "TIE")
        print(f"  Avg {metric.upper():<16} {a:>10.4f} {c:>10.4f} {delta:>+10.4f} {w:>12}")
    a3 = sum(r["ace_top3"] for r in rows) / len(rows)
    c3 = sum(r["cl_top3"] for r in rows) / len(rows)
    d3 = a3 - c3
    w3 = "ACE" if d3 > 0.005 else ("Codexlens" if d3 < -0.005 else "TIE")
    print(f"  {'Top-3 Rate':<22} {a3:>10.4f} {c3:>10.4f} {d3:>+10.4f} {w3:>12}")

    ace_zero = sum(1 for r in rows if r["ace_recall"] == 0)
    cl_zero = sum(1 for r in rows if r["cl_recall"] == 0)
    print(f"  {'Zero-recall':<22} {ace_zero:>10d} {cl_zero:>10d}")

    # By category
    print(f"\n{'Category':<18} {'ACE Recall':>10} {'CL Recall':>10} {'ACE MRR':>8} {'CL MRR':>8}")
    print("-" * 54)
    cats = sorted(set(r["category"] for r in rows))
    for cat in cats:
        cat_rows = [r for r in rows if r["category"] == cat]
        ar = sum(r["ace_recall"] for r in cat_rows) / len(cat_rows)
        cr = sum(r["cl_recall"] for r in cat_rows) / len(cat_rows)
        am = sum(r["ace_mrr"] for r in cat_rows) / len(cat_rows)
        cm = sum(r["cl_mrr"] for r in cat_rows) / len(cat_rows)
        print(f"  {cat:<16} {ar:>10.3f} {cr:>10.3f} {am:>8.3f} {cm:>8.3f}")

    # By difficulty
    print(f"\n{'Difficulty':<18} {'ACE Recall':>10} {'CL Recall':>10} {'ACE MRR':>8} {'CL MRR':>8}")
    print("-" * 54)
    for diff in ["medium", "hard"]:
        diff_rows = [r for r in rows if r["difficulty"] == diff]
        if not diff_rows:
            continue
        ar = sum(r["ace_recall"] for r in diff_rows) / len(diff_rows)
        cr = sum(r["cl_recall"] for r in diff_rows) / len(diff_rows)
        am = sum(r["ace_mrr"] for r in diff_rows) / len(diff_rows)
        cm = sum(r["cl_mrr"] for r in diff_rows) / len(diff_rows)
        print(f"  {diff:<16} {ar:>10.3f} {cr:>10.3f} {am:>8.3f} {cm:>8.3f}")

    # Per-query
    print(f"\n{'ID':<5} {'Cat':<14} {'Diff':<7} {'ACE_R':>6} {'CL_R':>6} {'ACE_M':>6} {'CL_M':>6} {'Winner':>10}")
    print("-" * 70)
    ace_wins = cl_wins = ties = 0
    for r in rows:
        score_a = r["ace_recall"] * 0.5 + r["ace_mrr"] * 0.3 + r["ace_ndcg5"] * 0.2
        score_c = r["cl_recall"] * 0.5 + r["cl_mrr"] * 0.3 + r["cl_ndcg5"] * 0.2
        if abs(score_a - score_c) < 0.01:
            w = "TIE"
            ties += 1
        elif score_a > score_c:
            w = "ACE"
            ace_wins += 1
        else:
            w = "CL"
            cl_wins += 1
        print(f"  {r['id']:<5} {r['category']:<14} {r['difficulty']:<7} "
              f"{r['ace_recall']:>5.2f} {r['cl_recall']:>5.2f} "
              f"{r['ace_mrr']:>5.3f} {r['cl_mrr']:>5.3f} {w:>10}")
    print("-" * 70)
    print(f"  Wins: ACE={ace_wins}  Codexlens={cl_wins}  Tie={ties}")

    # Save JSON report
    report = {
        "benchmark": "complex-20-queries",
        "summary": {
            "ace": {"avg_recall": round(ace_avg("recall"), 4), "avg_mrr": round(ace_avg("mrr"), 4),
                    "avg_ndcg5": round(ace_avg("ndcg5"), 4), "top3_rate": round(a3, 4), "zero_recall": ace_zero},
            "codexlens": {"avg_recall": round(cl_avg("recall"), 4), "avg_mrr": round(cl_avg("mrr"), 4),
                          "avg_ndcg5": round(cl_avg("ndcg5"), 4), "top3_rate": round(c3, 4), "zero_recall": cl_zero},
            "query_wins": {"ace": ace_wins, "codexlens": cl_wins, "tie": ties},
        },
        "queries": rows,
    }
    out_path = PROJECT_ROOT / "bench_complex_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved report: {out_path}")


# ===========================================================================
# Helpers
# ===========================================================================
def _print_summary(results, elapsed):
    agg = _aggregate(results)
    print(f"Avg Recall: {agg['avg_recall']:.3f}  Avg MRR: {agg['avg_mrr']:.3f}  "
          f"Avg NDCG@5: {agg['avg_ndcg5']:.3f}  Top3: {agg['top3_rate']:.3f}  "
          f"Zero-recall: {agg['zero_recall']}  Time: {elapsed:.1f}s")


def _print_details(results):
    print()
    for r in results:
        status = "OK" if r["recall"] > 0 else "MISS"
        print(f"  {r['id']:<5} [{status:>4}] recall={r['recall']:.2f} mrr={r['mrr']:.3f} "
              f"ndcg5={r['ndcg5']:.3f} [{r['category']}] found={r['found'][:3]}")


def _aggregate(results):
    n = len(results)
    return {
        "avg_recall": round(sum(r["recall"] for r in results) / n, 4),
        "avg_mrr": round(sum(r["mrr"] for r in results) / n, 4),
        "avg_ndcg5": round(sum(r["ndcg5"] for r in results) / n, 4),
        "top3_rate": round(sum(r["top3_hit"] for r in results) / n, 4),
        "zero_recall": sum(1 for r in results if r["recall"] == 0),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--codexlens", action="store_true", help="Run codexlens-search benchmark")
    group.add_argument("--ace-collect", action="store_true", help="Print ACE collection scaffold")
    group.add_argument("--report", action="store_true", help="Generate comparison report")
    args = parser.parse_args()

    if args.codexlens:
        run_codexlens()
    elif args.ace_collect:
        generate_ace_scaffold()
    elif args.report:
        generate_report()
