"""Ablation test for recall optimization strategies S2/S3/S4.

Variants:
  baseline  — current setup (bge-small-en-v1.5 + standard chunking)
  S2        — context-aware chunking (prepend file/class/func metadata)
  S3        — symbol-based query expansion at search time
  S4        — graph expansion with increased hop depth
  S2+S3     — context chunking + query expansion
  S2+S3+S4  — all three strategies combined

Usage:
  python bench_ablation.py              # run all variants
  python bench_ablation.py --only S2    # run single variant
  python bench_ablation.py --skip-index # skip S2 re-indexing (reuse cached)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Benchmark ground truth (20 queries)
# ---------------------------------------------------------------------------

QUERIES: list[dict] = [
    {"id": "Q1",  "query": "how does the embedding model load and initialize",
     "expected": ["embed/fastembed.py"]},
    {"id": "Q2",  "query": "binary quantization and hamming distance search",
     "expected": ["core/binary.py", "core/faiss_index.py"]},
    {"id": "Q3",  "query": "merge FTS and vector results with reciprocal rank fusion",
     "expected": ["search/fusion.py"]},
    {"id": "Q4",  "query": "how are files chunked for indexing",
     "expected": ["indexing/pipeline.py"]},
    {"id": "Q5",  "query": "configuration options for search quality and performance",
     "expected": ["config.py"]},
    {"id": "Q6",  "query": "full text search with SQLite FTS5",
     "expected": ["search/fts.py"]},
    {"id": "Q7",  "query": "detect programming language from file extension",
     "expected": ["indexing/pipeline.py"]},
    {"id": "Q8",  "query": "HNSW approximate nearest neighbor index",
     "expected": ["core/index.py", "core/usearch_index.py", "core/faiss_index.py"]},
    {"id": "Q9",  "query": "MCP server tools for code search",
     "expected": ["mcp_server.py"]},
    {"id": "Q10", "query": "incremental file watcher for index updates",
     "expected": ["watcher/file_watcher.py", "watcher/incremental_indexer.py"]},
    {"id": "Q11", "query": "search pipeline stages and quality routing",
     "expected": ["search/pipeline.py"]},
    {"id": "Q12", "query": "batch embedding with GPU acceleration",
     "expected": ["embed/fastembed.py"]},
    {"id": "Q13", "query": "gitignore filtering for excluded files",
     "expected": ["indexing/gitignore.py"]},
    {"id": "Q14", "query": "thread safety locking concurrent access",
     "expected": ["core/usearch_index.py", "core/faiss_index.py"]},
    {"id": "Q15", "query": "metadata store for tracking file changes",
     "expected": ["indexing/metadata.py"]},
    {"id": "Q16", "query": "AST tree-sitter parsing and symbol extraction",
     "expected": ["parsers/parser.py", "parsers/symbols.py", "parsers/chunker.py"]},
    {"id": "Q17", "query": "reranker quality cross-encoder scoring",
     "expected": ["rerank/local.py"]},
    {"id": "Q18", "query": "shard manager for large codebase indexing",
     "expected": ["core/shard.py", "core/shard_manager.py"]},
    {"id": "Q19", "query": "bridge CLI command line interface",
     "expected": ["bridge.py"]},
    {"id": "Q20", "query": "factory pattern for creating ANN and binary indexes",
     "expected": ["core/factory.py"]},
]

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src" / "codexlens_search"


def normalize_path(p: str) -> str:
    """Normalize path separators and strip src/codexlens_search prefix."""
    p = p.replace("\\", "/")
    for prefix in ("src/codexlens_search/", "codexlens_search/"):
        if p.startswith(prefix):
            p = p[len(prefix):]
    return p


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    qid: str
    query: str
    expected: list[str]
    found: list[str]
    recall: float
    mrr: float
    top3_hit: bool


@dataclass
class VariantResult:
    name: str
    query_results: list[QueryResult] = field(default_factory=list)
    index_time: float = 0.0
    search_time: float = 0.0

    @property
    def avg_recall(self) -> float:
        return np.mean([q.recall for q in self.query_results]) if self.query_results else 0.0

    @property
    def avg_mrr(self) -> float:
        return np.mean([q.mrr for q in self.query_results]) if self.query_results else 0.0

    @property
    def top3_rate(self) -> float:
        return np.mean([q.top3_hit for q in self.query_results]) if self.query_results else 0.0

    @property
    def zero_recall_count(self) -> int:
        return sum(1 for q in self.query_results if q.recall == 0.0)


def evaluate_query(qid: str, query: str, expected: list[str],
                   results: list) -> QueryResult:
    """Evaluate a single query against expected files."""
    found_paths = [normalize_path(r.path) for r in results]
    expected_norm = [normalize_path(e) for e in expected]

    hits = sum(1 for e in expected_norm if any(e in f for f in found_paths))
    recall = hits / len(expected_norm) if expected_norm else 0.0

    mrr = 0.0
    for rank, fp in enumerate(found_paths, 1):
        if any(e in fp for e in expected_norm):
            mrr = 1.0 / rank
            break

    top3_hit = any(
        any(e in fp for e in expected_norm)
        for fp in found_paths[:3]
    )

    return QueryResult(qid=qid, query=query, expected=expected,
                       found=found_paths[:5], recall=recall,
                       mrr=mrr, top3_hit=top3_hit)


# ---------------------------------------------------------------------------
# Strategy S2: Context-aware chunking
# ---------------------------------------------------------------------------

def build_s2_index(db_path: Path) -> tuple:
    """Build index with context-aware chunking.

    Monkey-patches _smart_chunk to prepend AST structural metadata
    (file path, class name, function name) before each chunk text.
    """
    from codexlens_search.bridge import create_pipeline
    from codexlens_search.indexing import pipeline as pipeline_mod

    ip, sp, config = create_pipeline(str(db_path), None)

    # Save original method
    orig_smart_chunk = pipeline_mod.IndexingPipeline._smart_chunk

    def _context_aware_chunk(self, text, path, max_chars, overlap):
        """Inject file/class/function context header into each chunk."""
        chunks = orig_smart_chunk(self, text, path, max_chars, overlap)
        if not chunks:
            return chunks

        # Build symbol context from the file text
        lang = pipeline_mod.detect_language(path) or ""
        context_map: dict[int, str] = {}  # line -> context string

        if pipeline_mod._HAS_AST_CHUNKER and lang:
            try:
                parser = pipeline_mod.ASTParser.get_instance()
                tree = parser.parse(text.encode("utf-8", "replace"), lang)
                if tree:
                    symbols = pipeline_mod._extract_symbols(tree, lang)
                    # Build line-to-symbol mapping (nearest enclosing symbol)
                    for sym in symbols:
                        ctx_parts = [f"// File: {path}"]
                        if sym.parent_name:
                            ctx_parts.append(f"// Class: {sym.parent_name}")
                        ctx_parts.append(f"// {sym.kind.title()}: {sym.name}")
                        ctx_str = "\n".join(ctx_parts) + "\n"
                        for line in range(sym.start_line, sym.end_line + 1):
                            if line not in context_map:
                                context_map[line] = ctx_str
            except Exception:
                pass

        enriched = []
        for chunk_text, p, sl, el, lang_tag in chunks:
            # Find best context for this chunk's start line
            header = context_map.get(sl, f"// File: {path}\n")
            enriched_text = header + chunk_text
            enriched.append((enriched_text, p, sl, el, lang_tag))
        return enriched

    # Monkey-patch
    pipeline_mod.IndexingPipeline._smart_chunk = _context_aware_chunk

    # Collect files and index
    files = _collect_source_files()
    t0 = time.monotonic()
    ip.sync(files, root=SRC_ROOT.parent.parent)
    index_time = time.monotonic() - t0

    # Restore original
    pipeline_mod.IndexingPipeline._smart_chunk = orig_smart_chunk

    return ip, sp, config, index_time


# ---------------------------------------------------------------------------
# Strategy S3: Query expansion
# ---------------------------------------------------------------------------

def expand_query_with_symbols(query: str, fts) -> str:
    """Expand query with related symbol names from FTS.

    Strategy: do a fast FTS search, extract symbol names from top results,
    append them to the original query for better embedding coverage.
    """
    # Quick FTS search to find relevant symbols
    try:
        fts_results = fts.exact_search(query, top_k=10)
        if not fts_results:
            fts_results = fts.fuzzy_search(query, top_k=10)
    except Exception:
        return query

    if not fts_results:
        return query

    # Collect symbols from top FTS chunks
    symbol_names: list[str] = []
    seen: set[str] = set()
    for doc_id, _ in fts_results[:5]:
        try:
            syms = fts.get_symbols_by_chunk(doc_id)
            for s in syms:
                name = s["name"]
                if name not in seen and len(name) > 2:
                    seen.add(name)
                    symbol_names.append(name)
        except Exception:
            continue

    if not symbol_names:
        return query

    # Append top 5 symbol names to query
    expansion = " ".join(symbol_names[:5])
    expanded = f"{query} {expansion}"
    return expanded


# ---------------------------------------------------------------------------
# Strategy S4: Enhanced graph expansion
# ---------------------------------------------------------------------------

def search_with_enhanced_graph(sp, query: str, top_k: int = 20,
                                expand_hops: int = 2) -> list:
    """Search with enhanced graph expansion (more hops, higher graph weight)."""
    from codexlens_search.search.fusion import (
        detect_query_intent, get_adaptive_weights, reciprocal_rank_fusion,
    )
    from concurrent.futures import ThreadPoolExecutor

    cfg = sp._config
    intent = detect_query_intent(query)
    weights = get_adaptive_weights(intent, cfg.fusion_weights)
    # Boost graph weight for S4
    weights["graph"] = max(weights.get("graph", 0.15), 0.25)
    # Renormalize
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    query_vec = sp._embedder.embed_single(query)

    vector_results = []
    exact_results = []
    fuzzy_results = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        vec_future = pool.submit(sp._vector_search, query_vec)
        fts_future = pool.submit(sp._fts_search, query)
        try:
            vector_results = vec_future.result()
        except Exception:
            pass
        try:
            exact_results, fuzzy_results = fts_future.result()
        except Exception:
            pass

    fusion_input: dict[str, list[tuple[int, float]]] = {}
    if vector_results:
        fusion_input["vector"] = vector_results
    if exact_results:
        fusion_input["exact"] = exact_results
    if fuzzy_results:
        fusion_input["fuzzy"] = fuzzy_results

    # Enhanced graph search with more hops
    if sp._graph_searcher is not None:
        orig_hops = sp._graph_searcher._expand_hops
        sp._graph_searcher._expand_hops = expand_hops
        try:
            seed_ids = sp._collect_top_chunk_ids(fusion_input)
            if seed_ids:
                graph_results = sp._graph_searcher.search_from_chunks(seed_ids)
                if graph_results:
                    fusion_input["graph"] = graph_results
        except Exception:
            pass
        sp._graph_searcher._expand_hops = orig_hops

    if not fusion_input:
        return []

    fused = reciprocal_rank_fusion(fusion_input, weights=weights, k=cfg.fusion_k)
    fused = sp._filter_deleted(fused)
    return sp._rerank_and_build(query, fused, top_k, use_reranker=True)


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def _collect_source_files() -> list[Path]:
    """Collect all Python source files for indexing."""
    return sorted(SRC_ROOT.rglob("*.py"))


# ---------------------------------------------------------------------------
# Run variants
# ---------------------------------------------------------------------------

def run_baseline(db_path: Path | None = None) -> VariantResult:
    """Baseline: current setup, no modifications."""
    from codexlens_search.bridge import create_pipeline

    if db_path is None:
        db_path = PROJECT_ROOT / ".codexlens"
    ip, sp, config = create_pipeline(str(db_path), None)

    result = VariantResult(name="baseline")
    t0 = time.monotonic()
    for q in QUERIES:
        results = sp.search(q["query"], top_k=20)
        qr = evaluate_query(q["id"], q["query"], q["expected"], results)
        result.query_results.append(qr)
    result.search_time = time.monotonic() - t0
    return result


def run_s2(skip_index: bool = False) -> VariantResult:
    """S2: Context-aware chunking."""
    from codexlens_search.bridge import create_pipeline

    db_path = PROJECT_ROOT / ".codexlens_s2"
    result = VariantResult(name="S2: context-chunk")

    if skip_index and db_path.exists():
        ip, sp, config = create_pipeline(str(db_path), None)
        print("  [S2] Reusing cached index")
    else:
        print("  [S2] Building context-aware index...")
        ip, sp, config, idx_time = build_s2_index(db_path)
        result.index_time = idx_time
        print(f"  [S2] Index built in {idx_time:.1f}s")

    t0 = time.monotonic()
    for q in QUERIES:
        results = sp.search(q["query"], top_k=20)
        qr = evaluate_query(q["id"], q["query"], q["expected"], results)
        result.query_results.append(qr)
    result.search_time = time.monotonic() - t0
    return result


def run_s3(db_path: Path | None = None) -> VariantResult:
    """S3: Symbol-based query expansion (search-time only)."""
    from codexlens_search.bridge import create_pipeline

    if db_path is None:
        db_path = PROJECT_ROOT / ".codexlens"
    ip, sp, config = create_pipeline(str(db_path), None)

    result = VariantResult(name="S3: query-expand")
    t0 = time.monotonic()
    for q in QUERIES:
        expanded = expand_query_with_symbols(q["query"], sp._fts)
        results = sp.search(expanded, top_k=20)
        qr = evaluate_query(q["id"], q["query"], q["expected"], results)
        result.query_results.append(qr)
    result.search_time = time.monotonic() - t0
    return result


def run_s4(db_path: Path | None = None) -> VariantResult:
    """S4: Enhanced graph expansion (search-time only)."""
    from codexlens_search.bridge import create_pipeline

    if db_path is None:
        db_path = PROJECT_ROOT / ".codexlens"
    ip, sp, config = create_pipeline(str(db_path), None)

    result = VariantResult(name="S4: graph-expand")
    t0 = time.monotonic()
    for q in QUERIES:
        results = search_with_enhanced_graph(sp, q["query"], top_k=20, expand_hops=2)
        qr = evaluate_query(q["id"], q["query"], q["expected"], results)
        result.query_results.append(qr)
    result.search_time = time.monotonic() - t0
    return result


def run_s2_s3(skip_index: bool = False) -> VariantResult:
    """S2+S3: Context chunking + query expansion."""
    from codexlens_search.bridge import create_pipeline

    db_path = PROJECT_ROOT / ".codexlens_s2"
    result = VariantResult(name="S2+S3")

    if skip_index and db_path.exists():
        ip, sp, config = create_pipeline(str(db_path), None)
    else:
        ip, sp, config, idx_time = build_s2_index(db_path)
        result.index_time = idx_time

    t0 = time.monotonic()
    for q in QUERIES:
        expanded = expand_query_with_symbols(q["query"], sp._fts)
        results = sp.search(expanded, top_k=20)
        qr = evaluate_query(q["id"], q["query"], q["expected"], results)
        result.query_results.append(qr)
    result.search_time = time.monotonic() - t0
    return result


def run_s2_s3_s4(skip_index: bool = False) -> VariantResult:
    """S2+S3+S4: All three strategies combined."""
    from codexlens_search.bridge import create_pipeline

    db_path = PROJECT_ROOT / ".codexlens_s2"
    result = VariantResult(name="S2+S3+S4")

    if skip_index and db_path.exists():
        ip, sp, config = create_pipeline(str(db_path), None)
    else:
        ip, sp, config, idx_time = build_s2_index(db_path)
        result.index_time = idx_time

    t0 = time.monotonic()
    for q in QUERIES:
        expanded = expand_query_with_symbols(q["query"], sp._fts)
        results = search_with_enhanced_graph(sp, expanded, top_k=20, expand_hops=2)
        qr = evaluate_query(q["id"], q["query"], q["expected"], results)
        result.query_results.append(qr)
    result.search_time = time.monotonic() - t0
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(variants: list[VariantResult]) -> None:
    """Print comparison table."""
    print("\n" + "=" * 90)
    print("ABLATION TEST RESULTS")
    print("=" * 90)

    # Summary table
    header = f"{'Variant':<18} {'Recall':>8} {'MRR':>8} {'Top3':>8} {'Zero':>6} {'Search(s)':>10} {'Index(s)':>10}"
    print(header)
    print("-" * 90)
    for v in variants:
        print(
            f"{v.name:<18} {v.avg_recall:>8.3f} {v.avg_mrr:>8.3f} "
            f"{v.top3_rate:>8.3f} {v.zero_recall_count:>6d} "
            f"{v.search_time:>10.1f} {v.index_time:>10.1f}"
        )

    # Delta from baseline
    if len(variants) > 1:
        base = variants[0]
        print("\n--- Delta from baseline ---")
        for v in variants[1:]:
            dr = v.avg_recall - base.avg_recall
            dm = v.avg_mrr - base.avg_mrr
            dt = v.top3_rate - base.top3_rate
            dz = v.zero_recall_count - base.zero_recall_count
            print(
                f"{v.name:<18} {dr:>+8.3f} {dm:>+8.3f} "
                f"{dt:>+8.3f} {dz:>+6d}"
            )

    # Per-query breakdown for zero-recall queries
    print("\n--- Focus: Previously zero-recall queries (Q3, Q14, Q17) ---")
    focus_ids = {"Q3", "Q14", "Q17"}
    header2 = f"{'Query':<6} {'Variant':<18} {'Recall':>8} {'MRR':>8} {'Top Result':<40}"
    print(header2)
    print("-" * 90)
    for qid in sorted(focus_ids):
        for v in variants:
            qr = next((q for q in v.query_results if q.qid == qid), None)
            if qr:
                top = qr.found[0] if qr.found else "(none)"
                print(f"{qr.qid:<6} {v.name:<18} {qr.recall:>8.3f} {qr.mrr:>8.3f} {top:<40}")
        print()

    # Full per-query breakdown
    print("\n--- Full per-query comparison ---")
    for q in QUERIES:
        qid = q["id"]
        row = f"{qid:<6}"
        for v in variants:
            qr = next((qr for qr in v.query_results if qr.qid == qid), None)
            if qr:
                row += f"  {qr.recall:.2f}"
        print(row + f"  | {q['query'][:50]}")

    # Legend
    col_labels = "       " + "  ".join(f"{v.name[:6]:>4}" for v in variants)
    print(col_labels)


def save_results(variants: list[VariantResult], path: Path) -> None:
    """Save detailed results as JSON."""
    data = []
    for v in variants:
        vdata = {
            "name": v.name,
            "avg_recall": round(v.avg_recall, 4),
            "avg_mrr": round(v.avg_mrr, 4),
            "top3_rate": round(v.top3_rate, 4),
            "zero_recall_count": v.zero_recall_count,
            "search_time": round(v.search_time, 2),
            "index_time": round(v.index_time, 2),
            "queries": [
                {
                    "qid": qr.qid,
                    "recall": round(qr.recall, 4),
                    "mrr": round(qr.mrr, 4),
                    "top3_hit": qr.top3_hit,
                    "found": qr.found,
                }
                for qr in v.query_results
            ],
        }
        data.append(vdata)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\nDetailed results saved to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

VARIANT_MAP = {
    "baseline": run_baseline,
    "S2": run_s2,
    "S3": run_s3,
    "S4": run_s4,
    "S2+S3": run_s2_s3,
    "S2+S3+S4": run_s2_s3_s4,
}


def main():
    parser = argparse.ArgumentParser(description="Ablation test for S2/S3/S4 strategies")
    parser.add_argument("--only", type=str, help="Run only this variant (e.g., S2)")
    parser.add_argument("--skip-index", action="store_true",
                        help="Reuse cached S2 index if available")
    args = parser.parse_args()

    print("=" * 60)
    print("Codexlens Search — Ablation Test (S2/S3/S4)")
    print("=" * 60)

    variants_to_run: list[str]
    if args.only:
        if args.only not in VARIANT_MAP:
            print(f"Unknown variant: {args.only}")
            print(f"Available: {', '.join(VARIANT_MAP.keys())}")
            sys.exit(1)
        variants_to_run = ["baseline", args.only]
    else:
        variants_to_run = list(VARIANT_MAP.keys())

    results: list[VariantResult] = []

    for name in variants_to_run:
        print(f"\n>>> Running: {name}")
        fn = VARIANT_MAP[name]

        # Pass skip_index for S2-related variants
        if name in ("S2", "S2+S3", "S2+S3+S4"):
            result = fn(skip_index=args.skip_index)
        else:
            result = fn()

        results.append(result)
        print(f"    Recall={result.avg_recall:.3f}  MRR={result.avg_mrr:.3f}  "
              f"Zero={result.zero_recall_count}")

    print_summary(results)
    save_results(results, PROJECT_ROOT / "bench_ablation_results.json")


if __name__ == "__main__":
    main()
