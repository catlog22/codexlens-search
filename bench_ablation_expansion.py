"""Ablation experiment for query expansion: VecExpand vs FTSExpand.

Runs 4 configurations on the 20 complex queries:
  1. Baseline:     no expansion
  2. +VecExpand:   sentence-vector nearest-neighbor expansion from symbol vocabulary
  3. +FTSExpand:   index-aware expansion via FTS top-chunk symbol extraction
  4. +Both:        VecExpand + FTSExpand combined (union of expanded terms)

Usage: python bench_ablation_expansion.py
"""
import json
import time
from pathlib import Path

import numpy as np

from codexlens_search.bridge import create_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent

from bench_complex_comparison import QUERIES, evaluate


# ---------------------------------------------------------------------------
# Expansion Method A: Sentence-Vector Nearest-Neighbor
# ---------------------------------------------------------------------------
class VecExpander:
    """Expand query by finding nearest symbol names via embedding similarity."""

    def __init__(self, fts, embedder, top_k=8):
        self._top_k = top_k
        self._vocab: list[str] = []
        self._vocab_vecs: np.ndarray | None = None

        # Build vocabulary from symbols table
        rows = fts._conn.execute(
            "SELECT DISTINCT name FROM symbols WHERE length(name) > 2"
        ).fetchall()
        raw_names = [r[0] for r in rows]

        # Also add file-level path tokens (e.g., "base", "pipeline", "fusion")
        path_rows = fts._conn.execute(
            "SELECT DISTINCT path FROM docs_meta"
        ).fetchall()
        for r in path_rows:
            parts = r[0].replace("\\", "/").split("/")
            for part in parts:
                stem = part.rsplit(".", 1)[0] if "." in part else part
                if len(stem) > 2 and stem not in raw_names:
                    raw_names.append(stem)

        # Deduplicate
        seen = set()
        for name in raw_names:
            low = name.lower()
            if low not in seen:
                seen.add(low)
                self._vocab.append(name)

        print(f"  VecExpander: {len(self._vocab)} vocabulary terms")

        # Batch embed all vocabulary terms
        t0 = time.monotonic()
        vecs = embedder.embed_batch(self._vocab)
        self._vocab_vecs = np.array(vecs, dtype=np.float32)
        # Normalize for cosine similarity
        norms = np.linalg.norm(self._vocab_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vocab_vecs /= norms
        elapsed = time.monotonic() - t0
        print(f"  VecExpander: vocabulary embedded in {elapsed:.1f}s")

        self._embedder = embedder

    def expand(self, query: str) -> list[str]:
        """Return top-K symbol names most similar to the query embedding."""
        qvec = self._embedder.embed_single(query).astype(np.float32)
        qvec /= (np.linalg.norm(qvec) or 1.0)

        # Cosine similarity via dot product (both normalized)
        scores = self._vocab_vecs @ qvec
        top_idx = np.argsort(scores)[::-1][: self._top_k]

        terms = []
        for i in top_idx:
            term = self._vocab[i]
            # Skip terms that are already substrings of the query
            if term.lower() not in query.lower():
                terms.append(term)
        return terms


# ---------------------------------------------------------------------------
# Expansion Method B: Index-Aware FTS Expansion
# ---------------------------------------------------------------------------
class FTSExpander:
    """Expand query by extracting symbol names from top FTS-matched chunks."""

    def __init__(self, fts, top_chunks=10, max_symbols=10):
        self._fts = fts
        self._top_chunks = top_chunks
        self._max_symbols = max_symbols

    def expand(self, query: str) -> list[str]:
        """Run FTS search, extract symbols from top chunks, return names."""
        # Use both exact and fuzzy search for broader coverage
        exact = self._fts.exact_search(query, top_k=self._top_chunks)
        fuzzy = self._fts.fuzzy_search(query, top_k=self._top_chunks)

        # Merge chunk IDs (exact results first, then fuzzy)
        seen_ids = set()
        chunk_ids = []
        for cid, _ in exact + fuzzy:
            if cid not in seen_ids:
                seen_ids.add(cid)
                chunk_ids.append(cid)

        if not chunk_ids:
            return []

        # Get symbols from these chunks
        placeholders = ",".join("?" for _ in chunk_ids[:20])
        rows = self._fts._conn.execute(
            f"SELECT DISTINCT name, kind FROM symbols "
            f"WHERE chunk_id IN ({placeholders}) AND length(name) > 2 "
            f"ORDER BY CASE kind "
            f"  WHEN 'class' THEN 1 "
            f"  WHEN 'function' THEN 2 "
            f"  WHEN 'method' THEN 3 "
            f"  ELSE 4 END",
            chunk_ids[:20],
        ).fetchall()

        terms = []
        query_lower = query.lower()
        for name, kind in rows:
            if name.lower() not in query_lower and name not in terms:
                terms.append(name)
                if len(terms) >= self._max_symbols:
                    break
        return terms


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
CONFIGS = [
    {"name": "Baseline",    "vec": False, "fts": False},
    {"name": "+VecExpand",  "vec": True,  "fts": False},
    {"name": "+FTSExpand",  "vec": False, "fts": True},
    {"name": "+Both",       "vec": True,  "fts": True},
]


def expand_query(query, config, vec_expander, fts_expander):
    """Apply configured expansion methods and return expanded query + terms."""
    terms = []
    if config["vec"] and vec_expander:
        terms.extend(vec_expander.expand(query))
    if config["fts"] and fts_expander:
        fts_terms = fts_expander.expand(query)
        for t in fts_terms:
            if t not in terms:
                terms.append(t)

    if terms:
        return f"{query} {' '.join(terms)}", terms
    return query, []


def run_config(sp, config, vec_expander, fts_expander):
    """Run all 20 queries with a given expansion configuration."""
    results = []
    t0 = time.monotonic()
    for q in QUERIES:
        expanded_query, added_terms = expand_query(
            q["query"], config, vec_expander, fts_expander
        )
        search_results = sp.search(expanded_query, top_k=20)
        found_paths = [r.path for r in search_results]
        metrics = evaluate(q["expected"], found_paths)
        results.append({
            "id": q["id"],
            "category": q["category"],
            "difficulty": q["difficulty"],
            "expected": q["expected"],
            "expanded_terms": added_terms,
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

    # Build expanders
    print("Building expanders...")
    vec_expander = VecExpander(sp._fts, sp._embedder, top_k=8)
    fts_expander = FTSExpander(sp._fts, top_chunks=10, max_symbols=10)

    all_results = {}

    for cfg in CONFIGS:
        print(f"\n{'='*70}")
        print(f"Running: {cfg['name']}  (Vec={cfg['vec']}, FTS={cfg['fts']})")
        print(f"{'='*70}")

        results, elapsed = run_config(sp, cfg, vec_expander, fts_expander)
        agg = aggregate(results)
        all_results[cfg["name"]] = {
            "config": cfg, "elapsed": elapsed, "agg": agg, "queries": results
        }

        print(f"  Recall={agg['avg_recall']:.4f}  MRR={agg['avg_mrr']:.4f}  "
              f"NDCG@5={agg['avg_ndcg5']:.4f}  Top3={agg['top3_rate']:.4f}  "
              f"Zero={agg['zero_recall']}  Time={elapsed:.1f}s")

        for r in results:
            status = "OK" if r["recall"] > 0 else "MISS"
            terms_str = ", ".join(r["expanded_terms"][:5]) if r["expanded_terms"] else "-"
            print(f"    {r['id']:<5} [{status:>4}] recall={r['recall']:.2f} "
                  f"mrr={r['mrr']:.3f} ndcg5={r['ndcg5']:.3f} "
                  f"found={r['found'][:3]}  +[{terms_str}]")

    # --- Comparison table ---
    print(f"\n{'='*90}")
    print("EXPANSION ABLATION RESULTS")
    print(f"{'='*90}")

    baseline = all_results["Baseline"]["agg"]
    print(f"\n{'Config':<14} {'Recall':>8} {'dRecall':>8} {'MRR':>8} {'dMRR':>8} "
          f"{'NDCG@5':>8} {'dNDCG5':>8} {'Top3':>8} {'Zero':>5} {'Time':>7}")
    print("-" * 92)

    for name in ["Baseline", "+VecExpand", "+FTSExpand", "+Both"]:
        r = all_results[name]
        a = r["agg"]
        dr = a["avg_recall"] - baseline["avg_recall"]
        dm = a["avg_mrr"] - baseline["avg_mrr"]
        dn = a["avg_ndcg5"] - baseline["avg_ndcg5"]
        print(f"  {name:<12} {a['avg_recall']:>8.4f} {dr:>+8.4f} "
              f"{a['avg_mrr']:>8.4f} {dm:>+8.4f} "
              f"{a['avg_ndcg5']:>8.4f} {dn:>+8.4f} "
              f"{a['top3_rate']:>8.4f} {a['zero_recall']:>5d} "
              f"{r['elapsed']:>6.1f}s")

    # --- Per-query delta: Baseline vs best ---
    for compare_name in ["+VecExpand", "+FTSExpand", "+Both"]:
        combined = all_results[compare_name]["queries"]
        base_q = all_results["Baseline"]["queries"]

        print(f"\n{'='*90}")
        print(f"PER-QUERY DELTA: Baseline -> {compare_name}")
        print(f"{'='*90}")
        print(f"{'ID':<5} {'Cat':<14} {'Base_R':>7} {'New_R':>7} {'dR':>6} "
              f"{'Base_M':>7} {'New_M':>7} {'dM':>6}  Expanded Terms")
        print("-" * 90)

        improved = degraded = 0
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
            terms_str = ", ".join(cq["expanded_terms"][:4]) if cq["expanded_terms"] else "-"
            print(f"  {bq['id']:<5} {bq['category']:<14} "
                  f"{bq['recall']:>6.2f} {cq['recall']:>6.2f} {dr:>+5.2f} "
                  f"{bq['mrr']:>6.3f} {cq['mrr']:>6.3f} {dm:>+5.3f}{marker}  [{terms_str}]")

        print(f"\n  Improved: {improved}  Degraded: {degraded}  "
              f"Unchanged: {20 - improved - degraded}")

    # --- By category ---
    print(f"\n{'='*90}")
    print("BY CATEGORY: Baseline vs +Both")
    print(f"{'='*90}")
    base_q = all_results["Baseline"]["queries"]
    combined = all_results["+Both"]["queries"]
    cats = sorted(set(r["category"] for r in base_q))
    print(f"{'Category':<16} {'Base_R':>8} {'New_R':>8} {'dR':>7} "
          f"{'Base_M':>8} {'New_M':>8} {'dM':>7}")
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

    # --- Focus: CQ2 detail (the zero-recall query) ---
    print(f"\n{'='*90}")
    print("FOCUS: CQ2 (abstract base classes) — The zero-recall query")
    print(f"{'='*90}")
    for name in ["Baseline", "+VecExpand", "+FTSExpand", "+Both"]:
        q = next(r for r in all_results[name]["queries"] if r["id"] == "CQ2")
        terms = ", ".join(q["expanded_terms"][:8]) if q["expanded_terms"] else "(none)"
        print(f"  {name:<14} recall={q['recall']:.2f} mrr={q['mrr']:.3f} "
              f"found={q['found'][:5]}")
        print(f"                expanded: [{terms}]")

    # Save JSON
    output = {"experiment": "Query Expansion Ablation", "configs": {}}
    for name, data in all_results.items():
        output["configs"][name] = {
            "vec": data["config"]["vec"],
            "fts": data["config"]["fts"],
            "elapsed": round(data["elapsed"], 2),
            **data["agg"],
            "queries": data["queries"],
        }
    out_path = PROJECT_ROOT / "bench_ablation_expansion.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
