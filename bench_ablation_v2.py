"""Ablation test for recall-enhancement strategies (v2).

Strategies:
  baseline — current production (context-aware chunking already integrated)
  C1       — index-time concept tagging (append // Concepts: ... to chunks)
  C2       — search-time query rewriting (FTS synonym expansion)
  C3       — symbol-aware FTS tags field (simulated via content enrichment)
  C1+C2    — concept tagging + query rewriting
  C1+C2+C3 — all three combined

Usage:
  python bench_ablation_v2.py              # run all
  python bench_ablation_v2.py --only C1    # single variant
  python bench_ablation_v2.py --skip-index # reuse cached indexes
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Ground truth (20 queries, corrected file names)
# ---------------------------------------------------------------------------

QUERIES: list[dict] = [
    {"id": "Q1",  "query": "how does the embedding model load and initialize",
     "expected": ["embed/local.py"]},
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
     "expected": ["embed/local.py"]},
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

# ---------------------------------------------------------------------------
# C1: Code-pattern-to-concept mapping
# ---------------------------------------------------------------------------

CODE_CONCEPT_MAP: list[tuple[str, list[str]]] = [
    # Thread safety / concurrency
    (r'\bthreading\.(R?Lock|Semaphore|Event|Condition|Barrier)\b',
     ['thread-safety', 'locking', 'concurrency', 'synchronization']),
    (r'with\s+self\._lock', ['thread-safety', 'locking', 'mutex', 'concurrent-access']),
    (r'\basyncio\.(Lock|Semaphore|Event)\b', ['async-locking', 'concurrency']),
    (r'\bmultiprocessing\.(Lock|Queue|Pool)\b', ['multiprocessing', 'parallelism', 'concurrency']),
    # Error handling patterns
    (r'try:.*?except\s+\w+.*?:', ['error-handling', 'exception']),
    (r'raise\s+\w+Error', ['error-raising', 'exception']),
    # Caching
    (r'\b(lru_cache|cache|memoize|functools\.cache)\b', ['caching', 'memoization']),
    # Singleton / factory
    (r'_instance\s*=\s*None|cls\._instance', ['singleton-pattern']),
    (r'def\s+create_\w+.*factory|Factory', ['factory-pattern']),
    # IO / file operations
    (r'\b(open|Path|pathlib)\b.*\b(read|write|exists)\b', ['file-io', 'filesystem']),
    # Lazy loading
    (r'_loaded\s*=\s*False|_ensure_loaded|lazy.?load', ['lazy-loading', 'deferred-init']),
    # Batch processing
    (r'batch|chunk.*process|embed_batch', ['batch-processing']),
    # GPU / acceleration
    (r'(CUDA|DirectML|GPU|cuda|gpu|DmlExecutionProvider|CUDAExecutionProvider)',
     ['gpu-acceleration', 'hardware-acceleration']),
    # Configuration / settings
    (r'@dataclass.*Config|class\s+Config', ['configuration', 'settings']),
    # Search / retrieval
    (r'(bm25|tf.?idf|cosine.?sim|hamming)', ['search-scoring', 'similarity']),
    (r'(HNSW|hnsw|ann_index|approximate.?nearest)', ['approximate-nearest-neighbor', 'vector-search']),
]

_compiled_concepts = [(re.compile(p, re.DOTALL), tags) for p, tags in CODE_CONCEPT_MAP]


def add_concept_tags(text: str) -> str:
    """Append concept tags to chunk text based on code pattern matching."""
    found_tags: set[str] = set()
    for pattern, tags in _compiled_concepts:
        if pattern.search(text):
            found_tags.update(tags)
    if found_tags:
        tag_str = ", ".join(sorted(found_tags))
        return f"{text}\n// Concepts: {tag_str}"
    return text


# ---------------------------------------------------------------------------
# C2: Search-time query rewriting (synonym expansion for FTS)
# ---------------------------------------------------------------------------

QUERY_SYNONYMS: dict[str, list[str]] = {
    "thread safety": ["threading", "RLock", "Lock", "mutex", "synchronization", "concurrent"],
    "locking": ["Lock", "RLock", "mutex", "acquire", "release", "synchronized"],
    "concurrent": ["threading", "thread-safe", "parallel", "Lock", "RLock"],
    "gpu acceleration": ["CUDA", "DirectML", "CUDAExecutionProvider", "DmlExecutionProvider", "gpu"],
    "batch": ["batch_size", "embed_batch", "batch_processing"],
    "lazy load": ["_ensure_loaded", "lazy", "deferred"],
    "cache": ["lru_cache", "cache_get", "cache_set", "memoize"],
    "factory": ["create_", "Factory", "factory_pattern"],
    "singleton": ["_instance", "get_instance"],
    "error handling": ["try", "except", "raise", "Exception"],
    "configuration": ["Config", "dataclass", "settings"],
    "initialize": ["__init__", "setup", "init", "load"],
}


def rewrite_query_for_fts(query: str) -> str:
    """Expand query with synonyms for FTS search only."""
    query_lower = query.lower()
    expansions: set[str] = set()
    for trigger, synonyms in QUERY_SYNONYMS.items():
        if trigger in query_lower:
            expansions.update(synonyms)
    if not expansions:
        return query
    # Build OR-expanded FTS query: original terms + expansion terms
    orig_tokens = query.strip().split()
    all_tokens = orig_tokens + [s for s in expansions if s.lower() not in query_lower]
    # Use OR to combine (FTS5 default is AND)
    return " OR ".join(f'"{t}"' if " " in t else t for t in all_tokens)


# ---------------------------------------------------------------------------
# C3: Symbol-aware concept enrichment (simulated via content append)
# ---------------------------------------------------------------------------
# Instead of adding a separate FTS field (which requires schema migration),
# we simulate by appending symbol-derived concept keywords to chunk content
# at index time — using AST-extracted symbols + concept mapping.

SYMBOL_CONCEPT_MAP: dict[str, list[str]] = {
    # Symbol names -> concept keywords
    "_lock": ["thread-safety", "locking", "mutex", "concurrent-access"],
    "RLock": ["thread-safety", "reentrant-lock", "mutex"],
    "Lock": ["thread-safety", "locking", "synchronization"],
    "ThreadPoolExecutor": ["concurrency", "thread-pool", "parallel-execution"],
    "Semaphore": ["concurrency", "rate-limiting", "synchronization"],
    "_ensure_loaded": ["lazy-loading", "deferred-initialization"],
    "_instance": ["singleton-pattern"],
    "embed_batch": ["batch-processing", "embedding"],
    "coarse_search": ["binary-search", "candidate-filtering"],
    "fine_search": ["ann-search", "approximate-nearest-neighbor"],
    "reciprocal_rank_fusion": ["result-fusion", "hybrid-search"],
}


def add_symbol_concepts(text: str, symbols: list[dict] | None = None) -> str:
    """Append concept keywords derived from symbol names in the chunk."""
    found_tags: set[str] = set()
    # Check symbol names against concept map
    if symbols:
        for sym in symbols:
            name = sym.get("name", "")
            if name in SYMBOL_CONCEPT_MAP:
                found_tags.update(SYMBOL_CONCEPT_MAP[name])
    # Also scan text for symbol names (fallback when symbols not available)
    for sym_name, concepts in SYMBOL_CONCEPT_MAP.items():
        if sym_name in text:
            found_tags.update(concepts)
    if found_tags:
        tag_str = " ".join(sorted(found_tags))
        return f"{text}\n// SymbolConcepts: {tag_str}"
    return text


# ---------------------------------------------------------------------------
# Metrics (same as bench_ablation.py)
# ---------------------------------------------------------------------------

def normalize_path(p: str) -> str:
    p = p.replace("\\", "/")
    for prefix in ("src/codexlens_search/", "codexlens_search/"):
        if p.startswith(prefix):
            p = p[len(prefix):]
    return p


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
        return float(np.mean([q.recall for q in self.query_results])) if self.query_results else 0.0

    @property
    def avg_mrr(self) -> float:
        return float(np.mean([q.mrr for q in self.query_results])) if self.query_results else 0.0

    @property
    def top3_rate(self) -> float:
        return float(np.mean([q.top3_hit for q in self.query_results])) if self.query_results else 0.0

    @property
    def zero_recall_count(self) -> int:
        return sum(1 for q in self.query_results if q.recall == 0.0)


def evaluate_query(qid: str, query: str, expected: list[str],
                   results: list) -> QueryResult:
    found_paths = [normalize_path(r.path) for r in results]
    expected_norm = [normalize_path(e) for e in expected]
    hits = sum(1 for e in expected_norm if any(e in f for f in found_paths))
    recall = hits / len(expected_norm) if expected_norm else 0.0
    mrr = 0.0
    for rank, fp in enumerate(found_paths, 1):
        if any(e in fp for e in expected_norm):
            mrr = 1.0 / rank
            break
    top3_hit = any(any(e in fp for e in expected_norm) for fp in found_paths[:3])
    return QueryResult(qid=qid, query=query, expected=expected,
                       found=found_paths[:5], recall=recall, mrr=mrr, top3_hit=top3_hit)


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def _collect_source_files() -> list[Path]:
    return sorted(SRC_ROOT.rglob("*.py"))


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------

def _build_index_with_enrichment(db_path: Path, enrichment_fn) -> tuple:
    """Build index with a chunk text enrichment function applied at index time."""
    from codexlens_search.bridge import create_pipeline
    from codexlens_search.indexing import pipeline as pipeline_mod

    # Clean slate
    if db_path.exists():
        shutil.rmtree(db_path)

    ip, sp, config = create_pipeline(str(db_path), None)

    # Save original _smart_chunk
    orig_smart_chunk = pipeline_mod.IndexingPipeline._smart_chunk

    def _enriched_chunk(self, text, path, max_chars, overlap):
        chunks = orig_smart_chunk(self, text, path, max_chars, overlap)
        if not chunks:
            return chunks
        return [
            (enrichment_fn(chunk_text), p, sl, el, lang_tag)
            for chunk_text, p, sl, el, lang_tag in chunks
        ]

    # Monkey-patch
    pipeline_mod.IndexingPipeline._smart_chunk = _enriched_chunk

    files = _collect_source_files()
    t0 = time.monotonic()
    ip.sync(files, root=SRC_ROOT.parent.parent)
    index_time = time.monotonic() - t0

    # Restore
    pipeline_mod.IndexingPipeline._smart_chunk = orig_smart_chunk

    return ip, sp, config, index_time


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def _search_with_query_rewrite(sp, query: str, top_k: int = 20) -> list:
    """Search using rewritten FTS query but original vector query."""
    from codexlens_search.search.fusion import (
        detect_query_intent, get_adaptive_weights, reciprocal_rank_fusion,
    )
    from concurrent.futures import ThreadPoolExecutor

    cfg = sp._config
    intent = detect_query_intent(query)
    weights = get_adaptive_weights(intent, cfg.fusion_weights)

    query_vec = sp._embedder.embed_single(query)
    rewritten = rewrite_query_for_fts(query)

    vector_results = []
    exact_results = []
    fuzzy_results = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        vec_future = pool.submit(sp._vector_search, query_vec)
        fts_future = pool.submit(sp._fts_search, rewritten)  # rewritten for FTS
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

    # Graph search
    if sp._graph_searcher is not None:
        try:
            seed_ids = sp._collect_top_chunk_ids(fusion_input)
            if seed_ids:
                graph_results = sp._graph_searcher.search_from_chunks(seed_ids)
                if graph_results:
                    fusion_input["graph"] = graph_results
        except Exception:
            pass

    if not fusion_input:
        return []

    fused = reciprocal_rank_fusion(fusion_input, weights=weights, k=cfg.fusion_k)
    fused = sp._filter_deleted(fused)
    # Rerank with ORIGINAL query (not rewritten) for precision
    return sp._rerank_and_build(query, fused, top_k, use_reranker=True)


# ---------------------------------------------------------------------------
# Variant runners
# ---------------------------------------------------------------------------

def run_baseline() -> VariantResult:
    """Baseline: current production index (context-aware chunking)."""
    from codexlens_search.bridge import create_pipeline
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


def run_c1(skip_index: bool = False) -> VariantResult:
    """C1: Index-time concept tagging."""
    from codexlens_search.bridge import create_pipeline
    db_path = PROJECT_ROOT / ".codexlens_c1"
    result = VariantResult(name="C1:concept-tag")

    if skip_index and db_path.exists():
        ip, sp, config = create_pipeline(str(db_path), None)
        print("  [C1] Reusing cached index")
    else:
        print("  [C1] Building concept-tagged index...")
        ip, sp, config, idx_time = _build_index_with_enrichment(db_path, add_concept_tags)
        result.index_time = idx_time
        print(f"  [C1] Index built in {idx_time:.1f}s")

    t0 = time.monotonic()
    for q in QUERIES:
        results = sp.search(q["query"], top_k=20)
        qr = evaluate_query(q["id"], q["query"], q["expected"], results)
        result.query_results.append(qr)
    result.search_time = time.monotonic() - t0
    return result


def run_c2() -> VariantResult:
    """C2: Search-time query rewriting (uses production index)."""
    from codexlens_search.bridge import create_pipeline
    db_path = PROJECT_ROOT / ".codexlens"
    ip, sp, config = create_pipeline(str(db_path), None)

    result = VariantResult(name="C2:query-rewrite")
    t0 = time.monotonic()
    for q in QUERIES:
        results = _search_with_query_rewrite(sp, q["query"], top_k=20)
        qr = evaluate_query(q["id"], q["query"], q["expected"], results)
        result.query_results.append(qr)
    result.search_time = time.monotonic() - t0
    return result


def run_c3(skip_index: bool = False) -> VariantResult:
    """C3: Symbol-aware concept enrichment at index time."""
    from codexlens_search.bridge import create_pipeline
    db_path = PROJECT_ROOT / ".codexlens_c3"
    result = VariantResult(name="C3:sym-concepts")

    if skip_index and db_path.exists():
        ip, sp, config = create_pipeline(str(db_path), None)
        print("  [C3] Reusing cached index")
    else:
        print("  [C3] Building symbol-concept index...")
        ip, sp, config, idx_time = _build_index_with_enrichment(db_path, add_symbol_concepts)
        result.index_time = idx_time
        print(f"  [C3] Index built in {idx_time:.1f}s")

    t0 = time.monotonic()
    for q in QUERIES:
        results = sp.search(q["query"], top_k=20)
        qr = evaluate_query(q["id"], q["query"], q["expected"], results)
        result.query_results.append(qr)
    result.search_time = time.monotonic() - t0
    return result


def run_c1_c2(skip_index: bool = False) -> VariantResult:
    """C1+C2: Concept tagging + query rewriting."""
    from codexlens_search.bridge import create_pipeline
    db_path = PROJECT_ROOT / ".codexlens_c1"
    result = VariantResult(name="C1+C2")

    if skip_index and db_path.exists():
        ip, sp, config = create_pipeline(str(db_path), None)
    else:
        print("  [C1+C2] Building concept-tagged index...")
        ip, sp, config, idx_time = _build_index_with_enrichment(db_path, add_concept_tags)
        result.index_time = idx_time

    t0 = time.monotonic()
    for q in QUERIES:
        results = _search_with_query_rewrite(sp, q["query"], top_k=20)
        qr = evaluate_query(q["id"], q["query"], q["expected"], results)
        result.query_results.append(qr)
    result.search_time = time.monotonic() - t0
    return result


def run_all_combined(skip_index: bool = False) -> VariantResult:
    """C1+C2+C3: All three strategies combined."""
    from codexlens_search.bridge import create_pipeline
    db_path = PROJECT_ROOT / ".codexlens_all"
    result = VariantResult(name="C1+C2+C3")

    def combined_enrichment(text: str) -> str:
        text = add_concept_tags(text)
        text = add_symbol_concepts(text)
        return text

    if skip_index and db_path.exists():
        ip, sp, config = create_pipeline(str(db_path), None)
        print("  [C1+C2+C3] Reusing cached index")
    else:
        print("  [C1+C2+C3] Building combined index...")
        ip, sp, config, idx_time = _build_index_with_enrichment(db_path, combined_enrichment)
        result.index_time = idx_time
        print(f"  [C1+C2+C3] Index built in {idx_time:.1f}s")

    t0 = time.monotonic()
    for q in QUERIES:
        results = _search_with_query_rewrite(sp, q["query"], top_k=20)
        qr = evaluate_query(q["id"], q["query"], q["expected"], results)
        result.query_results.append(qr)
    result.search_time = time.monotonic() - t0
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(variants: list[VariantResult]) -> None:
    print("\n" + "=" * 95)
    print("ABLATION TEST RESULTS (v2 — Concept/Rewrite/Symbol strategies)")
    print("=" * 95)

    header = f"{'Variant':<18} {'Recall':>8} {'MRR':>8} {'Top3':>8} {'Zero':>6} {'Search(s)':>10} {'Index(s)':>10}"
    print(header)
    print("-" * 95)
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

    # Q14 focus
    print("\n--- Focus: Q14 (thread safety locking concurrent access) ---")
    header2 = f"{'Variant':<18} {'Recall':>8} {'MRR':>8} {'Top-5 Results'}"
    print(header2)
    print("-" * 95)
    for v in variants:
        qr = next((q for q in v.query_results if q.qid == "Q14"), None)
        if qr:
            top5 = ", ".join(qr.found[:5]) if qr.found else "(none)"
            print(f"{v.name:<18} {qr.recall:>8.3f} {qr.mrr:>8.3f} {top5}")

    # Per-query zero-recall analysis
    print("\n--- Zero-recall queries per variant ---")
    for v in variants:
        zeros = [q for q in v.query_results if q.recall == 0.0]
        if zeros:
            qids = ", ".join(q.qid for q in zeros)
            print(f"  {v.name:<18} ({len(zeros)}): {qids}")
        else:
            print(f"  {v.name:<18} (0): none!")

    # Full per-query comparison
    print("\n--- Full per-query recall ---")
    for q in QUERIES:
        qid = q["id"]
        row = f"{qid:<6}"
        for v in variants:
            qr = next((qr for qr in v.query_results if qr.qid == qid), None)
            if qr:
                row += f"  {qr.recall:.2f}"
        row += f"  | {q['query'][:50]}"
        print(row)
    col_labels = "       " + "  ".join(f"{v.name[:6]:>4}" for v in variants)
    print(col_labels)


def save_results(variants: list[VariantResult], path: Path) -> None:
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
    "C1": run_c1,
    "C2": run_c2,
    "C3": run_c3,
    "C1+C2": run_c1_c2,
    "C1+C2+C3": run_all_combined,
}


def main():
    parser = argparse.ArgumentParser(description="Ablation test v2: concept/rewrite/symbol strategies")
    parser.add_argument("--only", type=str, help="Run only this variant (e.g., C1)")
    parser.add_argument("--skip-index", action="store_true",
                        help="Reuse cached indexes if available")
    args = parser.parse_args()

    print("=" * 60)
    print("Codexlens Search — Ablation Test v2 (C1/C2/C3)")
    print("=" * 60)

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

        if name in ("C1", "C3", "C1+C2", "C1+C2+C3"):
            result = fn(skip_index=args.skip_index)
        else:
            result = fn()

        results.append(result)
        print(f"    Recall={result.avg_recall:.3f}  MRR={result.avg_mrr:.3f}  "
              f"Zero={result.zero_recall_count}")

    print_summary(results)
    save_results(results, PROJECT_ROOT / "bench_ablation_v2_results.json")


if __name__ == "__main__":
    main()
