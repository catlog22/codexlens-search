"""Pipeline diagnostic: test each search stage independently on LocBench instances.

Indexes a repo, then runs each pipeline component separately to measure
which stages find (or miss) the ground truth files.

Usage:
    python bench/pipeline_diagnostic.py --repo-base G:/locbench_repos --index-base G:/locbench_index --limit 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from datasets import load_dataset

log = logging.getLogger("pipeline_diagnostic")


def setup_repo(repo: str, base_commit: str, repo_base: Path) -> Path | None:
    import subprocess
    repo_slug = repo.replace("/", "__")
    repo_dir = repo_base / repo_slug / base_commit[:8]
    if repo_dir.exists() and (repo_dir / ".git").exists():
        try:
            subprocess.run(["git", "checkout", base_commit, "--force"],
                           cwd=str(repo_dir), capture_output=True, timeout=120)
            subprocess.run(["git", "clean", "-fdx"],
                           cwd=str(repo_dir), capture_output=True, timeout=60)
            return repo_dir
        except Exception as e:
            log.warning("Checkout failed for %s: %s", repo_slug, e)
            return None
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    clone_url = f"https://github.com/{repo}.git"
    try:
        subprocess.run(["git", "clone", "--quiet", clone_url, str(repo_dir)],
                       capture_output=True, timeout=600, check=True)
        subprocess.run(["git", "checkout", base_commit, "--force"],
                       cwd=str(repo_dir), capture_output=True, timeout=120, check=True)
        return repo_dir
    except Exception as e:
        log.error("Clone/checkout failed for %s: %s", repo, e)
        return None


def get_gt_files(instance) -> list[str]:
    """Extract ground truth file paths from edit_functions."""
    files = []
    seen = set()
    for func in instance["edit_functions"]:
        fn = func.split(":")[0]
        if fn not in seen:
            seen.add(fn)
            files.append(fn)
    return files


def diagnose_instance(instance, repo_base: Path, index_base: Path, top_k: int = 20):
    """Run full diagnostic on a single instance."""
    from codexlens_search.bridge import create_pipeline, create_config_from_env
    from codexlens_search.search.fusion import detect_query_intent, get_adaptive_weights

    instance_id = instance["instance_id"]
    query = instance["problem_statement"]
    gt_files = get_gt_files(instance)

    log.info("=" * 70)
    log.info("Instance: %s", instance_id)
    log.info("GT files: %s", gt_files)
    log.info("Query length: %d chars, %d words", len(query), len(query.split()))

    # Setup repo
    repo_dir = setup_repo(instance["repo"], instance["base_commit"], repo_base)
    if repo_dir is None:
        log.error("Failed to setup repo")
        return None

    # Index
    index_dir = index_base / instance_id.replace("/", "__")
    if index_dir.exists():
        shutil.rmtree(index_dir, ignore_errors=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    config = create_config_from_env(db_path=str(index_dir))
    config.ast_chunking = True
    config.gitignore_filtering = True
    config.expansion_enabled = True
    config.default_search_quality = "thorough"

    indexing, search, config = create_pipeline(index_dir, config)

    py_files = [
        p for p in repo_dir.rglob("*.py")
        if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts
    ]
    stats = indexing.sync(py_files, root=repo_dir)
    log.info("Indexed %d files, %d chunks", stats.files_processed, stats.chunks_created)

    # Collect diagnostics
    diag = {
        "instance_id": instance_id,
        "gt_files": gt_files,
        "query_words": len(query.split()),
        "query_chars": len(query),
        "files_indexed": stats.files_processed,
        "chunks_created": stats.chunks_created,
        "stages": {},
    }

    # --- Stage 0: Query intent detection ---
    intent = detect_query_intent(query)
    weights = get_adaptive_weights(intent, config.fusion_weights)
    diag["query_intent"] = intent.value
    diag["fusion_weights"] = weights
    log.info("Query intent: %s", intent.value)
    log.info("Fusion weights: %s", json.dumps(weights, indent=2))

    # --- Stage 1: Query expansion ---
    if search._query_expander is not None:
        try:
            expanded = search._query_expander.expand(query)
            added_terms = expanded[len(query):].strip() if len(expanded) > len(query) else ""
            diag["stages"]["expansion"] = {
                "original_len": len(query),
                "expanded_len": len(expanded),
                "added_terms": added_terms[:500],
            }
            log.info("Expansion: +%d chars, terms: %s", len(expanded) - len(query), added_terms[:200])
        except Exception as e:
            diag["stages"]["expansion"] = {"error": str(e)}
            log.warning("Expansion failed: %s", e)
    else:
        diag["stages"]["expansion"] = {"status": "no_expander"}

    # --- Stage 2: FTS exact + fuzzy ---
    try:
        exact_results, fuzzy_results = search._fts_search(query)
        exact_files = _results_to_files(exact_results, search)
        fuzzy_files = _results_to_files(fuzzy_results, search)
        diag["stages"]["fts_exact"] = _eval_stage(exact_files, gt_files, "fts_exact")
        diag["stages"]["fts_fuzzy"] = _eval_stage(fuzzy_files, gt_files, "fts_fuzzy")
        log.info("FTS exact: %d results, %d unique files, recall=%s",
                 len(exact_results), len(exact_files),
                 diag["stages"]["fts_exact"]["recall"])
        log.info("FTS fuzzy: %d results, %d unique files, recall=%s",
                 len(fuzzy_results), len(fuzzy_files),
                 diag["stages"]["fts_fuzzy"]["recall"])
    except Exception as e:
        diag["stages"]["fts_exact"] = {"error": str(e)}
        diag["stages"]["fts_fuzzy"] = {"error": str(e)}
        log.warning("FTS failed: %s", e)

    # --- Stage 3: Vector search (binary coarse + ANN fine) ---
    try:
        import numpy as np
        query_vec = search._embedder.embed_single(query)
        vector_results = search._vector_search(query_vec)
        vector_files = _results_to_files(vector_results, search)
        diag["stages"]["vector"] = _eval_stage(vector_files, gt_files, "vector")
        log.info("Vector: %d results, %d unique files, recall=%s",
                 len(vector_results), len(vector_files),
                 diag["stages"]["vector"]["recall"])

        # Also test binary coarse only
        coarse_results = search._binary_coarse_search(query_vec)
        coarse_files = _results_to_files(coarse_results, search)
        diag["stages"]["binary_coarse"] = _eval_stage(coarse_files, gt_files, "binary_coarse")
        log.info("Binary coarse: %d results, %d unique files, recall=%s",
                 len(coarse_results), len(coarse_files),
                 diag["stages"]["binary_coarse"]["recall"])
    except Exception as e:
        diag["stages"]["vector"] = {"error": str(e)}
        log.warning("Vector search failed: %s", e)

    # --- Stage 4: Symbol search ---
    try:
        symbol_results = search._symbol_search(query)
        symbol_files = _results_to_files(symbol_results, search)
        diag["stages"]["symbol"] = _eval_stage(symbol_files, gt_files, "symbol")
        candidates = search._extract_symbol_candidates(query)
        diag["stages"]["symbol"]["candidates"] = candidates[:20]
        log.info("Symbol: %d results, %d unique files, candidates=%s, recall=%s",
                 len(symbol_results), len(symbol_files), candidates[:10],
                 diag["stages"]["symbol"]["recall"])
    except Exception as e:
        diag["stages"]["symbol"] = {"error": str(e)}
        log.warning("Symbol search failed: %s", e)

    # --- Stage 5: Graph search ---
    if search._graph_searcher is not None:
        try:
            # Collect seed chunks from vector + FTS
            fusion_input = {}
            if vector_results:
                fusion_input["vector"] = vector_results
            if exact_results:
                fusion_input["exact"] = exact_results
            seed_ids = search._collect_top_chunk_ids(fusion_input)
            graph_results = search._graph_searcher.search_from_chunks(seed_ids)
            graph_files = _results_to_files(graph_results, search)
            diag["stages"]["graph"] = _eval_stage(graph_files, gt_files, "graph")
            diag["stages"]["graph"]["seed_count"] = len(seed_ids)
            log.info("Graph: %d seeds -> %d results, %d unique files, recall=%s",
                     len(seed_ids), len(graph_results), len(graph_files),
                     diag["stages"]["graph"]["recall"])
        except Exception as e:
            diag["stages"]["graph"] = {"error": str(e)}
            log.warning("Graph search failed: %s", e)
    else:
        diag["stages"]["graph"] = {"status": "no_graph_searcher"}

    # --- Stage 6: Entity graph expansion ---
    if search._entity_graph is not None and config.entity_graph_enabled:
        try:
            fusion_input = {}
            if vector_results:
                fusion_input["vector"] = vector_results
            if exact_results:
                fusion_input["exact"] = exact_results
            seed_ids = search._collect_top_chunk_ids(fusion_input)
            entity_results = search._entity_graph.expand_from_chunks(
                seed_ids, depth=config.entity_graph_depth, top_k=config.fts_top_k)
            entity_files = _results_to_files(entity_results, search)
            diag["stages"]["entity"] = _eval_stage(entity_files, gt_files, "entity")
            diag["stages"]["entity"]["seed_count"] = len(seed_ids)
            log.info("Entity graph: %d seeds -> %d results, %d unique files, recall=%s",
                     len(seed_ids), len(entity_results), len(entity_files),
                     diag["stages"]["entity"]["recall"])
        except Exception as e:
            diag["stages"]["entity"] = {"error": str(e)}
            log.warning("Entity graph failed: %s", e)
    else:
        diag["stages"]["entity"] = {"status": "disabled_or_missing"}

    # --- Stage 7: Full fusion (pre-rerank) ---
    try:
        from codexlens_search.search.fusion import reciprocal_rank_fusion
        fusion_input = {}
        if vector_results:
            fusion_input["vector"] = vector_results
        if exact_results:
            fusion_input["exact"] = exact_results
        if fuzzy_results:
            fusion_input["fuzzy"] = fuzzy_results
        if symbol_results:
            fusion_input["symbol"] = symbol_results
        if "graph" in diag["stages"] and "error" not in diag["stages"]["graph"]:
            if graph_results:
                fusion_input["graph"] = graph_results
        if "entity" in diag["stages"] and "error" not in diag["stages"]["entity"]:
            if entity_results:
                fusion_input["entity"] = entity_results

        fused = reciprocal_rank_fusion(fusion_input, weights=weights, k=config.fusion_k)
        fused_files = _results_to_files(fused[:100], search)
        diag["stages"]["fusion"] = _eval_stage(fused_files, gt_files, "fusion")
        diag["stages"]["fusion"]["sources"] = list(fusion_input.keys())
        diag["stages"]["fusion"]["source_sizes"] = {k: len(v) for k, v in fusion_input.items()}
        log.info("Fusion: %d total, sources=%s, recall=%s",
                 len(fused), list(fusion_input.keys()),
                 diag["stages"]["fusion"]["recall"])
    except Exception as e:
        diag["stages"]["fusion"] = {"error": str(e)}
        log.warning("Fusion failed: %s", e)

    # --- Stage 8: Reranked results ---
    try:
        reranked = search._rerank_and_build(query, fused[:50], top_k)
        reranked_files = [_normalize_path(r.path) for r in reranked]
        unique_reranked = list(dict.fromkeys(reranked_files))
        diag["stages"]["reranked"] = _eval_stage(unique_reranked, gt_files, "reranked")
        log.info("Reranked: %d results, %d unique files, recall=%s",
                 len(reranked), len(unique_reranked),
                 diag["stages"]["reranked"]["recall"])
    except Exception as e:
        diag["stages"]["reranked"] = {"error": str(e)}
        log.warning("Rerank failed: %s", e)

    # --- Stage 9: search_files() end-to-end ---
    try:
        file_results = search.search_files(query, top_k=top_k)
        file_paths = [_normalize_path(r.path) for r in file_results]
        diag["stages"]["search_files"] = _eval_stage(file_paths, gt_files, "search_files")
        diag["stages"]["search_files"]["top5_files"] = file_paths[:5]
        log.info("search_files: %d files, recall@5=%s, recall@all=%s",
                 len(file_paths),
                 _recall_at_k(file_paths, gt_files, 5),
                 diag["stages"]["search_files"]["recall"])
        # Show which GT files were found/missed
        found = set(file_paths) & set(gt_files)
        missed = set(gt_files) - set(file_paths)
        diag["stages"]["search_files"]["found_gt"] = sorted(found)
        diag["stages"]["search_files"]["missed_gt"] = sorted(missed)
        log.info("  Found GT: %s", sorted(found))
        log.info("  Missed GT: %s", sorted(missed))
    except Exception as e:
        diag["stages"]["search_files"] = {"error": str(e)}
        log.warning("search_files failed: %s", e)

    # --- Stage 10: Check if GT files exist in index ---
    try:
        gt_in_index = {}
        for gt in gt_files:
            # Try both forward and backslash versions
            chunks_fwd = list(search._fts.get_chunk_ids_by_path(gt))
            chunks_bk = list(search._fts.get_chunk_ids_by_path(gt.replace("/", "\\")))
            gt_in_index[gt] = {"fwd": len(chunks_fwd), "bk": len(chunks_bk)}
        diag["gt_chunks_in_index"] = gt_in_index
        log.info("GT files in index: %s", gt_in_index)

        # Show all indexed paths (sample) for debugging
        try:
            all_paths = search._fts._conn.execute(
                "SELECT DISTINCT path FROM docs_meta LIMIT 10"
            ).fetchall()
            sample_paths = [r[0] for r in all_paths]
            diag["sample_indexed_paths"] = sample_paths
            log.info("Sample indexed paths: %s", sample_paths[:5])
        except Exception:
            pass
    except Exception as e:
        log.warning("GT index check failed: %s", e)

    return diag


def _normalize_path(p: str) -> str:
    """Normalize path to forward slashes for consistent comparison."""
    return p.replace("\\", "/")


def _results_to_files(results: list[tuple[int, float]], search) -> list[str]:
    """Convert (chunk_id, score) results to unique file paths (normalized)."""
    seen = set()
    files = []
    for doc_id, _ in results:
        try:
            path = _normalize_path(search._fts.get_doc_meta(doc_id)[0])
        except Exception:
            continue
        if path not in seen:
            seen.add(path)
            files.append(path)
    return files


def _recall_at_k(pred_files: list[str], gt_files: list[str], k: int) -> float:
    pred_set = set(pred_files[:k])
    gt_set = set(gt_files)
    if not gt_set:
        return 0.0
    return len(pred_set & gt_set) / len(gt_set)


def _eval_stage(found_files: list[str], gt_files: list[str], stage_name: str) -> dict:
    """Compute recall metrics for a stage's file results."""
    gt_set = set(gt_files)
    found_set = set(found_files)
    hits = found_set & gt_set
    return {
        "total_results": len(found_files),
        "recall": round(len(hits) / len(gt_set), 4) if gt_set else 0.0,
        "recall_at_5": round(_recall_at_k(found_files, gt_files, 5), 4),
        "recall_at_10": round(_recall_at_k(found_files, gt_files, 10), 4),
        "hits": sorted(hits),
        "misses": sorted(gt_set - found_set),
    }


def main():
    parser = argparse.ArgumentParser(description="Pipeline diagnostic for LocBench")
    parser.add_argument("--repo-base", type=str, default="G:/locbench_repos")
    parser.add_argument("--index-base", type=str, default="G:/locbench_index")
    parser.add_argument("--dataset", type=str, default="czlll/Loc-Bench_V1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", type=str, default="bench/pipeline_diagnostic.jsonl")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Loading dataset %s ...", args.dataset)
    bench_data = load_dataset(args.dataset, split=args.split)
    if args.limit:
        bench_data = bench_data.select(range(min(args.limit, len(bench_data))))
    log.info("Loaded %d instances", len(bench_data))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_diags = []
    for i, instance in enumerate(bench_data):
        log.info("\n[%d/%d] Processing %s", i + 1, len(bench_data), instance["instance_id"])
        diag = diagnose_instance(
            instance,
            Path(args.repo_base),
            Path(args.index_base),
            top_k=args.top_k,
        )
        if diag is not None:
            all_diags.append(diag)
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(diag, ensure_ascii=False) + "\n")

    # Summary
    if all_diags:
        print("\n" + "=" * 70)
        print("PIPELINE DIAGNOSTIC SUMMARY")
        print("=" * 70)
        print(f"Instances: {len(all_diags)}")

        stage_names = [
            "fts_exact", "fts_fuzzy", "binary_coarse", "vector",
            "symbol", "graph", "entity", "fusion", "reranked", "search_files"
        ]
        for stage in stage_names:
            recalls = []
            recalls_at5 = []
            for d in all_diags:
                s = d.get("stages", {}).get(stage, {})
                if "recall" in s:
                    recalls.append(s["recall"])
                if "recall_at_5" in s:
                    recalls_at5.append(s["recall_at_5"])
            if recalls:
                avg_recall = sum(recalls) / len(recalls)
                avg_r5 = sum(recalls_at5) / len(recalls_at5) if recalls_at5 else 0
                print(f"  {stage:<20} recall={avg_recall:.4f}  recall@5={avg_r5:.4f}  (n={len(recalls)})")
            else:
                print(f"  {stage:<20} no data")

        # Per-instance GT analysis
        print("\nPer-instance GT analysis:")
        for d in all_diags:
            iid = d["instance_id"]
            gt = d["gt_files"]
            sf = d.get("stages", {}).get("search_files", {})
            print(f"\n  {iid}:")
            print(f"    GT: {gt}")
            print(f"    Found: {sf.get('found_gt', [])}")
            print(f"    Missed: {sf.get('missed_gt', [])}")
            print(f"    Top5: {sf.get('top5_files', [])}")
            # Show which stages found each GT file
            for gt_file in gt:
                found_in = []
                for sn in stage_names:
                    s = d.get("stages", {}).get(sn, {})
                    if gt_file in s.get("hits", []):
                        found_in.append(sn)
                status = "FOUND in: " + ", ".join(found_in) if found_in else "MISSED everywhere"
                print(f"    {gt_file}: {status}")


if __name__ == "__main__":
    main()
