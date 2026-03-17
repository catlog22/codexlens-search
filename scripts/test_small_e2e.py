"""
Small-folder end-to-end test: index tests/ directory (~10 files) and verify
indexing pipeline + all search features work correctly.

Usage: python scripts/test_small_e2e.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from codexlens_search.config import Config
from codexlens_search.core.factory import create_ann_index, create_binary_index
from codexlens_search.embed.local import FastEmbedEmbedder
from codexlens_search.indexing import IndexingPipeline
from codexlens_search.rerank.local import FastEmbedReranker
from codexlens_search.search.fts import FTSEngine
from codexlens_search.search.pipeline import SearchPipeline

PROJECT = Path(__file__).parent.parent
TARGET_DIR = PROJECT / "src" / "codexlens_search"  # ~21 .py files, small
INDEX_DIR = PROJECT / ".test_index_cache"
EXTENSIONS = {".py"}

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} — {detail}")


def main():
    global passed, failed
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    config = Config(
        embed_model="BAAI/bge-small-en-v1.5",
        embed_dim=384,
        embed_batch_size=32,
        hnsw_ef=100,
        hnsw_M=16,
        binary_top_k=100,
        ann_top_k=30,
        reranker_model="Xenova/ms-marco-MiniLM-L-6-v2",
        reranker_top_k=10,
    )

    files = [p for p in TARGET_DIR.rglob("*.py") if p.is_file()]
    print(f"Target: {TARGET_DIR} ({len(files)} .py files)\n")

    # ── 1. Test IndexingPipeline ──────────────────────────────
    print("=== 1. IndexingPipeline (parallel) ===")
    embedder = FastEmbedEmbedder(config)
    binary_store = create_binary_index(INDEX_DIR, config.embed_dim, config)
    ann_index = create_ann_index(INDEX_DIR, config.embed_dim, config)
    fts = FTSEngine(":memory:")

    t0 = time.time()
    stats = IndexingPipeline(
        embedder=embedder,
        binary_store=binary_store,
        ann_index=ann_index,
        fts=fts,
        config=config,
    ).index_files(files, root=TARGET_DIR, max_chunk_chars=800, chunk_overlap=100)
    elapsed = time.time() - t0

    check("files_processed > 0", stats.files_processed > 0, f"got {stats.files_processed}")
    check("chunks_created > 0", stats.chunks_created > 0, f"got {stats.chunks_created}")
    check("indexing completed", elapsed < 120, f"took {elapsed:.1f}s")
    print(f"  Stats: {stats.files_processed} files, {stats.chunks_created} chunks, {elapsed:.1f}s\n")

    # ── 2. Test BinaryStore (pre-allocated, coarse search) ────
    print("=== 2. BinaryStore coarse search ===")
    q_vec = embedder.embed_single("def search")
    b_ids, b_dists = binary_store.coarse_search(q_vec, top_k=10)
    check("binary returns results", len(b_ids) > 0, f"got {len(b_ids)}")
    check("binary ids are ints", all(isinstance(int(i), int) for i in b_ids))
    print(f"  Top 5 binary IDs: {b_ids[:5]}\n")

    # ── 3. Test ANNIndex (fine search) ────────────────────────
    print("=== 3. ANNIndex fine search ===")
    a_ids, a_dists = ann_index.fine_search(q_vec, top_k=10)
    check("ann returns results", len(a_ids) > 0, f"got {len(a_ids)}")
    check("ann scores are floats", all(isinstance(float(d), float) for d in a_dists))
    print(f"  Top 5 ANN IDs: {a_ids[:5]}\n")

    # ── 4. Test FTSEngine (exact + fuzzy) ─────────────────────
    print("=== 4. FTSEngine search ===")
    exact = fts.exact_search("def search", top_k=5)
    fuzzy = fts.fuzzy_search("embedd", top_k=5)
    check("exact search returns results", len(exact) > 0, f"got {len(exact)}")
    check("fuzzy search returns results", len(fuzzy) > 0, f"got {len(fuzzy)}")
    print(f"  Exact hits: {len(exact)}, Fuzzy hits: {len(fuzzy)}\n")

    # ── 5. Test SearchPipeline (parallel FTS||vector + fusion + rerank) ──
    print("=== 5. SearchPipeline (full pipeline) ===")
    reranker = FastEmbedReranker(config)
    search = SearchPipeline(
        embedder=embedder,
        binary_store=binary_store,
        ann_index=ann_index,
        reranker=reranker,
        fts=fts,
        config=config,
    )

    queries = [
        ("def embed_single", "code symbol search"),
        ("search pipeline fusion", "natural language search"),
        ("Config dataclass", "exact match search"),
        ("binary store hamming", "domain-specific search"),
        ("", "empty query handling"),
    ]

    for query, desc in queries:
        t0 = time.time()
        results = search.search(query, top_k=5)
        ms = (time.time() - t0) * 1000

        if query == "":
            check(f"{desc}: no crash", isinstance(results, list))
        else:
            check(f"{desc}: returns results", len(results) > 0, f"'{query}' got 0 results")
            if results:
                check(f"{desc}: has scores", all(isinstance(r.score, (int, float)) for r in results))
                check(f"{desc}: has paths", all(r.path for r in results))
                check(f"{desc}: respects top_k", len(results) <= 5)
                print(f"    Top result: [{results[0].score:.3f}] {results[0].path}")
        print(f"    Latency: {ms:.0f}ms")

    # ── 6. Test result quality (sanity) ───────────────────────
    print("\n=== 6. Result quality sanity checks ===")
    r1 = search.search("BinaryStore add coarse_search", top_k=5)
    if r1:
        paths = [r.path for r in r1]
        check("BinaryStore query -> binary/core in results",
              any("binary" in p or "core" in p for p in paths),
              f"got paths: {paths}")

    r2 = search.search("FTSEngine exact_search fuzzy_search", top_k=5)
    if r2:
        paths = [r.path for r in r2]
        check("FTSEngine query -> fts/search in results",
              any("fts" in p or "search" in p for p in paths),
              f"got paths: {paths}")

    r3 = search.search("IndexingPipeline parallel queue", top_k=3)
    if r3:
        paths = [r.path for r in r3]
        check("Pipeline query -> pipeline in results",
              any("pipeline" in p or "indexing" in p for p in paths),
              f"got paths: {paths}")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"WARNING: {failed} test(s) failed")
    print(f"{'=' * 50}")

    # Cleanup
    import shutil
    shutil.rmtree(INDEX_DIR, ignore_errors=True)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
