"""LocBench evaluation for codexlens-search.

Usage:
    # Full run (all instances)
    python bench/locbench_eval.py --repo-base G:/locbench_repos --index-base G:/locbench_index

    # Quick smoke test (first 5 instances)
    python bench/locbench_eval.py --repo-base G:/locbench_repos --index-base G:/locbench_index --limit 5

    # Resume interrupted run (skips already-processed instances)
    python bench/locbench_eval.py --repo-base G:/locbench_repos --index-base G:/locbench_index --resume

    # Evaluate only (skip indexing+search, just compute metrics from existing output)
    python bench/locbench_eval.py --eval-only --output bench/locbench_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Add codex-lens-v2 src to path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from datasets import load_dataset

# Load .env file if present
_env_file = _PROJECT_ROOT / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

log = logging.getLogger("locbench_eval")


# ---------------------------------------------------------------------------
# Repo setup: clone + checkout specific commit
# ---------------------------------------------------------------------------

def setup_repo(repo: str, base_commit: str, repo_base: Path) -> Path | None:
    """Clone repo and checkout base_commit. Returns repo dir or None on failure."""
    # repo format: "django/django" -> clone dir: repo_base/django__django/<commit[:8]>
    repo_slug = repo.replace("/", "__")
    repo_dir = repo_base / repo_slug / base_commit[:8]

    if repo_dir.exists() and (repo_dir / ".git").exists():
        # Already cloned, just checkout
        try:
            subprocess.run(
                ["git", "checkout", base_commit, "--force"],
                cwd=str(repo_dir), capture_output=True, timeout=120,
            )
            subprocess.run(
                ["git", "clean", "-fdx"],
                cwd=str(repo_dir), capture_output=True, timeout=60,
            )
            return repo_dir
        except Exception as e:
            log.warning("Checkout failed for %s: %s", repo_slug, e)
            return None

    # Clone fresh
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    clone_url = f"https://github.com/{repo}.git"
    try:
        subprocess.run(
            ["git", "clone", "--quiet", clone_url, str(repo_dir)],
            capture_output=True, timeout=600, check=True,
        )
        subprocess.run(
            ["git", "checkout", base_commit, "--force"],
            cwd=str(repo_dir), capture_output=True, timeout=120, check=True,
        )
        return repo_dir
    except Exception as e:
        log.error("Clone/checkout failed for %s: %s", repo, e)
        return None


# ---------------------------------------------------------------------------
# Index + Search with codexlens-search
# ---------------------------------------------------------------------------

def index_and_search(
    repo_dir: Path,
    index_dir: Path,
    query: str,
    top_k: int = 20,
    agent: bool = False,
    file_level: bool = False,
) -> list[dict]:
    """Index a repo with codexlens-search and run a search query.

    Returns list of {path, score, line, end_line, content}.
    """
    from codexlens_search.bridge import create_agent, create_pipeline, create_config_from_env

    # Clean previous index for this repo
    if index_dir.exists():
        shutil.rmtree(index_dir, ignore_errors=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    config = create_config_from_env(
        db_path=str(index_dir),
        # Local model, no API
    )
    # Force settings suitable for benchmark
    config.ast_chunking = True
    config.gitignore_filtering = True
    config.expansion_enabled = True
    config.default_search_quality = "thorough"

    indexing, search, config = create_pipeline(index_dir, config)

    # Collect Python files (LocBench is Python-only repos)
    py_files = [
        p for p in repo_dir.rglob("*.py")
        if p.is_file()
        and ".git" not in p.parts
        and "__pycache__" not in p.parts
        and "node_modules" not in p.parts
    ]
    if not py_files:
        log.warning("No .py files found in %s", repo_dir)
        return []

    # Index
    stats = indexing.sync(py_files, root=repo_dir)
    log.info(
        "Indexed %d files, %d chunks in %.1fs",
        stats.files_processed, stats.chunks_created, stats.duration_seconds,
    )

    # Search / Agent locate
    if agent:
        config.agent_enabled = True
        entity_graph = getattr(search, "_entity_graph", None)
        loc_agent = create_agent(search, entity_graph, config)

        old_cwd = os.getcwd()
        try:
            os.chdir(repo_dir)
            file_results = loc_agent.run_sync(
                query,
                max_iterations=config.agent_max_iterations,
                top_k=top_k,
            )
        finally:
            os.chdir(old_cwd)

        return [
            {
                "path": r.path,
                "score": r.score,
                "line": r.line,
                "end_line": r.end_line,
            }
            for r in file_results
        ]

    # Search
    if file_level:
        results = search.search_files(query, top_k=top_k)
        return [
            {
                "path": r.path,
                "score": r.score,
                "line": r.line,
                "end_line": r.end_line,
            }
            for r in results
        ]

    results = search.search(query, top_k=top_k)

    return [
        {
            "path": r.path,
            "score": r.score,
            "line": r.line,
            "end_line": r.end_line,
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# Map search results -> LocBench output format
# ---------------------------------------------------------------------------

def results_to_loc_output(
    instance_id: str,
    search_results: list[dict],
    repo_dir: Path,
) -> dict:
    """Convert search results to LocBench evaluation format.

    Returns dict with found_files, found_modules, found_entities.
    Module/entity level mapping uses tree-sitter AST symbol extraction.
    """
    found_files = []
    found_modules = []
    found_entities = []
    seen_files = set()

    # Build symbol table for files that appear in results
    symbol_cache: dict[str, list] = {}

    for r in search_results:
        # Normalize to forward slashes (GT uses Unix-style paths)
        rel_path = r["path"].replace("\\", "/")

        # File level: deduplicated, ordered by search rank
        if rel_path not in seen_files:
            found_files.append(rel_path)
            seen_files.add(rel_path)

        # Module/entity level: map line range to symbol
        if rel_path not in symbol_cache:
            symbol_cache[rel_path] = _extract_file_symbols(repo_dir, rel_path)

        symbols = symbol_cache[rel_path]
        line = r["line"]
        end_line = r["end_line"]

        # Find symbols that overlap with the search result line range
        for sym in symbols:
            if sym["end_line"] < line or sym["start_line"] > end_line:
                continue
            # Build qualified name: file.py:ClassName or file.py:ClassName.method
            if sym["parent_name"]:
                qualified = f"{rel_path}:{sym['parent_name']}.{sym['name']}"
                module_name = f"{rel_path}:{sym['parent_name']}"
            else:
                qualified = f"{rel_path}:{sym['name']}"
                module_name = f"{rel_path}:{sym['name']}"

            if module_name not in found_modules:
                found_modules.append(module_name)
            if qualified not in found_entities:
                found_entities.append(qualified)

    return {
        "instance_id": instance_id,
        "found_files": found_files,
        "found_modules": found_modules,
        "found_entities": found_entities,
    }


def _extract_file_symbols(repo_dir: Path, rel_path: str) -> list[dict]:
    """Extract class/function symbols from a Python file using tree-sitter."""
    try:
        from codexlens_search.parsers.parser import ASTParser
        from codexlens_search.parsers.symbols import extract_symbols
    except ImportError:
        return []

    file_path = repo_dir / rel_path
    if not file_path.exists():
        return []

    try:
        source = file_path.read_bytes()
        parser = ASTParser()
        tree = parser.parse(source, "python")
        if tree is None:
            return []
        symbols = extract_symbols(tree, "python")
        return [
            {
                "name": s.name,
                "kind": s.kind.value,
                "start_line": s.start_line,
                "end_line": s.end_line,
                "parent_name": s.parent_name,
            }
            for s in symbols
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Evaluation metrics (standalone, no LocAgent dependency)
# ---------------------------------------------------------------------------

def compute_metrics(
    output_file: str,
    dataset_name: str = "czlll/Loc-Bench_V1",
    split: str = "test",
):
    """Compute Acc/Recall/NDCG/Precision/MAP at various k values."""
    import collections
    import math

    # Load ground truth from dataset
    bench_data = load_dataset(dataset_name, split=split)
    gt_files: dict[str, list[str]] = collections.defaultdict(list)
    gt_modules: dict[str, list[str]] = collections.defaultdict(list)
    gt_entities: dict[str, list[str]] = collections.defaultdict(list)

    for instance in bench_data:
        iid = instance["instance_id"]
        for func in instance["edit_functions"]:
            # format: "path/to/file.py:ClassName.method_name"
            fn = func.split(":")[0]
            if fn not in gt_files[iid]:
                gt_files[iid].append(fn)

            parts = func.split(":")
            if len(parts) == 2:
                module_name = parts[1].split(".")[0]
                mid = f"{parts[0]}:{module_name}"
                if mid not in gt_modules[iid]:
                    gt_modules[iid].append(mid)

                entity_name = parts[1]
                if entity_name.endswith(".__init__"):
                    entity_name = entity_name[: -len(".__init__")]
                eid = f"{parts[0]}:{entity_name}"
                if eid not in gt_entities[iid]:
                    gt_entities[iid].append(eid)

    # Load predictions
    predictions: dict[str, dict] = {}
    with open(output_file, "r") as f:
        for line in f:
            data = json.loads(line)
            predictions[data["instance_id"]] = data

    # Metric functions
    def recall_at_k(pred_list, gt_list, k):
        pred_set = set(pred_list[:k])
        gt_set = set(gt_list)
        if not gt_set:
            return 0.0
        return len(pred_set & gt_set) / len(gt_set)

    def acc_at_k(pred_list, gt_list, k):
        """All ground truth items found in top-k predictions."""
        pred_set = set(pred_list[:k])
        gt_set = set(gt_list)
        if not gt_set:
            return 0.0
        return 1.0 if gt_set.issubset(pred_set) else 0.0

    def ndcg_at_k(pred_list, gt_list, k):
        gt_set = set(gt_list)
        # DCG
        dcg = 0.0
        for i, item in enumerate(pred_list[:k]):
            if item in gt_set:
                dcg += 1.0 / math.log2(i + 2)
        # Ideal DCG
        idcg = 0.0
        for i in range(min(len(gt_list), k)):
            idcg += 1.0 / math.log2(i + 2)
        return dcg / idcg if idcg > 0 else 0.0

    def precision_at_k(pred_list, gt_list, k):
        gt_set = set(gt_list)
        hits = sum(1 for item in pred_list[:k] if item in gt_set)
        return hits / k

    def map_at_k(pred_list, gt_list, k):
        gt_set = set(gt_list)
        ap = 0.0
        relevant_count = 0
        for i, item in enumerate(pred_list[:k]):
            if item in gt_set:
                relevant_count += 1
                ap += relevant_count / (i + 1)
        return ap / k

    # Compute for each level
    levels = {
        "file": (gt_files, "found_files", [1, 3, 5]),
        "module": (gt_modules, "found_modules", [5, 10]),
        "function": (gt_entities, "found_entities", [5, 10]),
    }

    # Only evaluate instances that have predictions
    evaluated_ids = set(predictions.keys())

    results = {}
    for level_name, (gt_dict, pred_key, k_values) in levels.items():
        level_results = {}
        for k in k_values:
            accs, recalls, ndcgs, precs, maps = [], [], [], [], []
            for iid, gt_list in gt_dict.items():
                if not gt_list or iid not in evaluated_ids:
                    continue
                pred_list = predictions.get(iid, {}).get(pred_key, [])
                accs.append(acc_at_k(pred_list, gt_list, k))
                recalls.append(recall_at_k(pred_list, gt_list, k))
                ndcgs.append(ndcg_at_k(pred_list, gt_list, k))
                precs.append(precision_at_k(pred_list, gt_list, k))
                maps.append(map_at_k(pred_list, gt_list, k))

            n = len(accs)
            if n > 0:
                level_results[f"Acc@{k}"] = round(sum(accs) / n, 4)
                level_results[f"Recall@{k}"] = round(sum(recalls) / n, 4)
                level_results[f"NDCG@{k}"] = round(sum(ndcgs) / n, 4)
                level_results[f"P@{k}"] = round(sum(precs) / n, 4)
                level_results[f"MAP@{k}"] = round(sum(maps) / n, 4)

        results[level_name] = level_results

    results["_meta"] = {"evaluated": len(evaluated_ids), "total_gt": len(gt_files)}
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LocBench evaluation for codexlens-search")
    parser.add_argument("--repo-base", type=str, default="G:/locbench_repos",
                        help="Directory to clone repos into")
    parser.add_argument("--index-base", type=str, default="G:/locbench_index",
                        help="Directory for codexlens indexes")
    parser.add_argument("--output", type=str, default="bench/locbench_results.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--dataset", type=str, default="czlll/Loc-Bench_V1",
                        help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of instances (0 = all)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of search results per query")
    parser.add_argument("--file-level", action="store_true",
                        help="Use SearchPipeline.search_files() for file-level aggregation")
    parser.add_argument("--agent", action="store_true",
                        help="Use LLM agent loop (requires optional deps + API key)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed instances")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip indexing, only compute metrics from output file")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Eval-only mode
    if args.eval_only:
        if not output_path.exists():
            log.error("Output file not found: %s", output_path)
            sys.exit(1)
        results = compute_metrics(str(output_path), args.dataset, args.split)
        _print_results(results)
        return

    # Load dataset
    log.info("Loading dataset %s ...", args.dataset)
    bench_data = load_dataset(args.dataset, split=args.split)
    if args.limit:
        bench_data = bench_data.select(range(min(args.limit, len(bench_data))))
    log.info("Loaded %d instances", len(bench_data))

    # Load already-processed instances for resume
    processed = set()
    if args.resume and output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                data = json.loads(line)
                processed.add(data["instance_id"])
        log.info("Resuming: %d instances already processed", len(processed))

    repo_base = Path(args.repo_base)
    index_base = Path(args.index_base)
    total = len(bench_data)
    success = 0
    failed = 0
    t0 = time.time()

    for i, instance in enumerate(bench_data):
        instance_id = instance["instance_id"]

        if instance_id in processed:
            log.info("[%d/%d] SKIP %s (already processed)", i + 1, total, instance_id)
            continue

        log.info("[%d/%d] Processing %s ...", i + 1, total, instance_id)

        # 1. Setup repo
        repo_dir = setup_repo(
            repo=instance["repo"],
            base_commit=instance["base_commit"],
            repo_base=repo_base,
        )
        if repo_dir is None:
            log.error("  Failed to setup repo for %s", instance_id)
            failed += 1
            continue

        # 2. Index + Search
        index_dir = index_base / instance_id.replace("/", "__")
        try:
            search_results = index_and_search(
                repo_dir=repo_dir,
                index_dir=index_dir,
                query=instance["problem_statement"],
                top_k=args.top_k,
                agent=args.agent,
                file_level=args.file_level,
            )
        except Exception as e:
            log.error("  Index/search failed for %s: %s", instance_id, e)
            failed += 1
            continue

        # 3. Convert to LocBench format
        loc_output = results_to_loc_output(instance_id, search_results, repo_dir)

        # 4. Append to output
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(loc_output, ensure_ascii=False) + "\n")

        success += 1
        elapsed = time.time() - t0
        avg = elapsed / (success + failed)
        remaining = avg * (total - i - 1)
        log.info(
            "  Found %d files, %d modules, %d entities (%.0fs elapsed, ~%.0fs remaining)",
            len(loc_output["found_files"]),
            len(loc_output["found_modules"]),
            len(loc_output["found_entities"]),
            elapsed,
            remaining,
        )

    elapsed = time.time() - t0
    log.info("Done: %d success, %d failed, %.1f min total", success, failed, elapsed / 60)

    # 5. Compute metrics
    if output_path.exists():
        results = compute_metrics(str(output_path), args.dataset, args.split)
        _print_results(results)


def _print_results(results: dict):
    meta = results.pop("_meta", {})
    print("\n" + "=" * 70)
    print("LocBench Evaluation Results (codexlens-search)")
    if meta:
        print(f"  Evaluated: {meta.get('evaluated', '?')} / {meta.get('total_gt', '?')} instances")
    print("=" * 70)
    for level, metrics in results.items():
        print(f"\n  [{level.upper()}]")
        for metric, value in metrics.items():
            print(f"    {metric:<12} {value:.4f}")
    print()


if __name__ == "__main__":
    main()
