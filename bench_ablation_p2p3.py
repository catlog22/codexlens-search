"""Ablation experiment for P2 (hybrid graph seeding) and P3 (post-fusion expansion).

Runs 4 configurations on the 20 complex queries:
  1. Baseline:  p2=off, p3=off  (current behavior)
  2. +P2:       p2=on,  p3=off  (hybrid graph seeding only)
  3. +P3:       p2=off, p3=on   (post-fusion expansion only)
  4. +P2+P3:    p2=on,  p3=on   (both enabled)

Usage: python bench_ablation_p2p3.py
"""
import json
import math
import time
from pathlib import Path

from codexlens_search.bridge import create_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent

# Import shared queries and evaluate logic
from bench_complex_comparison import QUERIES, evaluate

CONFIGS = [
    {"name": "Baseline",  "p2": False, "p3": False},
    {"name": "+P2",       "p2": True,  "p3": False},
    {"name": "+P3",       "p2": False, "p3": True},
    {"name": "+P2+P3",    "p2": True,  "p3": True},
]


def run_config(sp, config_def):
    """Run all 20 queries with a given P2/P3 flag combination."""
    results = []
    t0 = time.monotonic()
    for q in QUERIES:
        search_results = sp.search(
            q["query"], top_k=20,
            p2_hybrid_seed=config_def["p2"],
            p3_post_expand=config_def["p3"],
        )
        found_paths = [r.path for r in search_results]
        metrics = evaluate(q["expected"], found_paths)
        results.append({
            "id": q["id"],
            "category": q["category"],
            "difficulty": q["difficulty"],
            "expected": q["expected"],
            **metrics,
        })
    elapsed = time.monotonic() - t0
    return results, elapsed


def aggregate(results):
    n = len(results)
    return {
        "avg_recall": round(sum(r["recall"] for r in results) / n, 4),
        "avg_mrr": round(sum(r["mrr"] for r in results) / n, 4),
        "avg_ndcg5": round(sum(r["ndcg5"] for r in results) / n, 4),
        "top3_rate": round(sum(r["top3_hit"] for r in results) / n, 4),
        "zero_recall": sum(1 for r in results if r["recall"] == 0),
    }


def main():
    db_path = PROJECT_ROOT / ".codexlens"
    ip, sp, config = create_pipeline(str(db_path), None)

    all_results = {}

    for cfg in CONFIGS:
        print(f"\n{'='*70}")
        print(f"Running: {cfg['name']}  (P2={cfg['p2']}, P3={cfg['p3']})")
        print(f"{'='*70}")

        results, elapsed = run_config(sp, cfg)
        agg = aggregate(results)
        all_results[cfg["name"]] = {"config": cfg, "elapsed": elapsed, "agg": agg, "queries": results}

        print(f"  Recall={agg['avg_recall']:.4f}  MRR={agg['avg_mrr']:.4f}  "
              f"NDCG@5={agg['avg_ndcg5']:.4f}  Top3={agg['top3_rate']:.4f}  "
              f"Zero={agg['zero_recall']}  Time={elapsed:.1f}s")

        # Show per-query changes from baseline
        for r in results:
            status = "OK" if r["recall"] > 0 else "MISS"
            print(f"    {r['id']:<5} [{status:>4}] recall={r['recall']:.2f} "
                  f"mrr={r['mrr']:.3f} ndcg5={r['ndcg5']:.3f} "
                  f"found={r['found'][:4]}")

    # --- Comparison table ---
    print(f"\n{'='*90}")
    print("ABLATION RESULTS COMPARISON")
    print(f"{'='*90}")

    baseline = all_results["Baseline"]["agg"]
    print(f"\n{'Config':<12} {'Recall':>8} {'dRecall':>8} {'MRR':>8} {'dMRR':>8} "
          f"{'NDCG@5':>8} {'dNDCG5':>8} {'Top3':>8} {'Zero':>5} {'Time':>7}")
    print("-" * 90)

    for name in ["Baseline", "+P2", "+P3", "+P2+P3"]:
        r = all_results[name]
        a = r["agg"]
        dr = a["avg_recall"] - baseline["avg_recall"]
        dm = a["avg_mrr"] - baseline["avg_mrr"]
        dn = a["avg_ndcg5"] - baseline["avg_ndcg5"]
        print(f"  {name:<10} {a['avg_recall']:>8.4f} {dr:>+8.4f} "
              f"{a['avg_mrr']:>8.4f} {dm:>+8.4f} "
              f"{a['avg_ndcg5']:>8.4f} {dn:>+8.4f} "
              f"{a['top3_rate']:>8.4f} {a['zero_recall']:>5d} "
              f"{r['elapsed']:>6.1f}s")

    # --- Per-query delta table (Baseline vs +P2+P3) ---
    combined = all_results["+P2+P3"]["queries"]
    base_q = all_results["Baseline"]["queries"]

    print(f"\n{'='*90}")
    print("PER-QUERY DELTA: Baseline -> +P2+P3")
    print(f"{'='*90}")
    print(f"{'ID':<5} {'Cat':<14} {'Base_R':>7} {'New_R':>7} {'dR':>6} "
          f"{'Base_M':>7} {'New_M':>7} {'dM':>6}")
    print("-" * 70)

    improved = 0
    degraded = 0
    for bq, cq in zip(base_q, combined):
        dr = cq["recall"] - bq["recall"]
        dm = cq["mrr"] - bq["mrr"]
        marker = ""
        if dr > 0.001 or dm > 0.001:
            marker = " +"
            improved += 1
        elif dr < -0.001 or dm < -0.001:
            marker = " -"
            degraded += 1
        print(f"  {bq['id']:<5} {bq['category']:<14} "
              f"{bq['recall']:>6.2f} {cq['recall']:>6.2f} {dr:>+5.2f} "
              f"{bq['mrr']:>6.3f} {cq['mrr']:>6.3f} {dm:>+5.3f}{marker}")

    print(f"\n  Improved: {improved}  Degraded: {degraded}  Unchanged: {20 - improved - degraded}")

    # --- By category ---
    print(f"\n{'='*90}")
    print("BY CATEGORY: Baseline vs +P2+P3")
    print(f"{'='*90}")
    cats = sorted(set(r["category"] for r in base_q))
    print(f"{'Category':<16} {'Base_R':>8} {'New_R':>8} {'dR':>7} {'Base_M':>8} {'New_M':>8} {'dM':>7}")
    print("-" * 62)
    for cat in cats:
        b_rows = [r for r in base_q if r["category"] == cat]
        c_rows = [r for r in combined if r["category"] == cat]
        br = sum(r["recall"] for r in b_rows) / len(b_rows)
        cr = sum(r["recall"] for r in c_rows) / len(c_rows)
        bm = sum(r["mrr"] for r in b_rows) / len(b_rows)
        cm = sum(r["mrr"] for r in c_rows) / len(c_rows)
        print(f"  {cat:<14} {br:>8.3f} {cr:>8.3f} {cr-br:>+7.3f} "
              f"{bm:>8.3f} {cm:>8.3f} {cm-bm:>+7.3f}")

    # Save JSON
    output = {
        "experiment": "P2+P3 ablation",
        "configs": {},
    }
    for name, data in all_results.items():
        output["configs"][name] = {
            "p2": data["config"]["p2"],
            "p3": data["config"]["p3"],
            "elapsed": round(data["elapsed"], 2),
            **data["agg"],
            "queries": data["queries"],
        }
    out_path = PROJECT_ROOT / "bench_ablation_p2p3.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
