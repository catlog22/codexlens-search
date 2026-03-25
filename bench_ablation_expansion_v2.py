"""Ablation experiment v2: Optimized query expansion strategies.

Improvements over v1:
  - Intent gating: only expand NATURAL_LANGUAGE queries
  - Cosine threshold: filter low-similarity terms
  - Symbol quality filter: prefer public classes/functions, skip _private
  - Two-hop expansion: VecExpand → FTS symbol lookup → discover neighbors
  - Concept vocabulary: hardcoded patterns for known abstract→code mappings

Runs 6 configurations:
  1. Baseline:         no expansion
  2. +VecV1:           v1 VecExpand (for comparison)
  3. +VecV2:           optimized VecExpand (threshold + filter + intent gate)
  4. +TwoHop:          VecV2 terms → FTS symbol lookup → neighbor discovery
  5. +ConceptVocab:    hardcoded concept→code term mappings
  6. +VecV2+Concept:   VecV2 + ConceptVocab combined

Usage: python bench_ablation_expansion_v2.py
"""
import json
import time
from pathlib import Path

import numpy as np

from codexlens_search.bridge import create_pipeline
from codexlens_search.search.fusion import QueryIntent, detect_query_intent

PROJECT_ROOT = Path(__file__).resolve().parent

from bench_complex_comparison import QUERIES, evaluate


# ---------------------------------------------------------------------------
# Concept Vocabulary: abstract terms → concrete code tokens
# ---------------------------------------------------------------------------
CONCEPT_VOCAB: list[tuple[list[str], list[str]]] = [
    # (trigger phrases, expansion tokens)
    (["abstract base", "interface contract", "plugin architecture", "swappable backend"],
     ["ABC", "abstractmethod", "Base", "BaseEmbedder", "BaseReranker", "BaseANNIndex", "BaseBinaryIndex"]),
    (["shard", "eviction", "lru", "memory limit"],
     ["ShardManager", "Shard", "max_loaded_shards", "_evict_lru"]),
    (["debounce", "filesystem notification", "file change"],
     ["FileWatcher", "FileEvent", "debounce", "events"]),
    (["binary quantization", "hamming", "sign-based"],
     ["_quantize", "binary", "coarse_search", "BinaryStore"]),
    (["graph traversal", "symbol reference", "multi-hop"],
     ["GraphSearcher", "_traverse", "search_from_chunks", "_find_seed_symbols"]),
    (["incremental", "sync", "re-index", "changed files"],
     ["IncrementalIndexer", "file_needs_update", "metadata", "sync"]),
    (["embedding", "dense vector", "onnx inference", "local inference"],
     ["FastEmbedEmbedder", "embed_single", "embed_batch", "TextEmbedding"]),
    (["rerank", "cross-encoder"],
     ["FastEmbedReranker", "TextCrossEncoder", "score_pairs"]),
    (["fusion", "reciprocal rank", "weight"],
     ["reciprocal_rank_fusion", "get_adaptive_weights", "detect_query_intent"]),
]


def concept_expand(query: str) -> list[str]:
    """Match query against concept vocabulary, return expansion tokens."""
    lower = query.lower()
    terms = []
    for triggers, tokens in CONCEPT_VOCAB:
        if any(t in lower for t in triggers):
            for tok in tokens:
                if tok.lower() not in lower and tok not in terms:
                    terms.append(tok)
    return terms


# ---------------------------------------------------------------------------
# VecExpander V1 (same as v1 ablation, for comparison)
# ---------------------------------------------------------------------------
class VecExpanderV1:
    def __init__(self, fts, embedder, top_k=8):
        self._top_k = top_k
        self._vocab, self._vocab_vecs = _build_vocab(fts, embedder)
        self._embedder = embedder

    def expand(self, query: str) -> list[str]:
        qvec = _embed_and_normalize(self._embedder, query)
        scores = self._vocab_vecs @ qvec
        top_idx = np.argsort(scores)[::-1][: self._top_k]
        return [self._vocab[i] for i in top_idx
                if self._vocab[i].lower() not in query.lower()]


# ---------------------------------------------------------------------------
# VecExpander V2 (optimized)
# ---------------------------------------------------------------------------
class VecExpanderV2:
    """Optimized VecExpand with intent gating, threshold, and symbol quality."""

    def __init__(self, fts, embedder, top_k=5, threshold=0.35):
        self._top_k = top_k
        self._threshold = threshold
        self._embedder = embedder

        # Build vocabulary with quality metadata
        self._names: list[str] = []
        self._kinds: list[str] = []
        self._is_public: list[bool] = []

        # Symbols from DB
        rows = fts._conn.execute(
            "SELECT DISTINCT name, kind FROM symbols WHERE length(name) > 2"
        ).fetchall()
        for name, kind in rows:
            self._names.append(name)
            self._kinds.append(kind or "")
            self._is_public.append(not name.startswith("_"))

        # File stem tokens
        path_rows = fts._conn.execute(
            "SELECT DISTINCT path FROM docs_meta"
        ).fetchall()
        seen = {n.lower() for n in self._names}
        for r in path_rows:
            parts = r[0].replace("\\", "/").split("/")
            for part in parts:
                stem = part.rsplit(".", 1)[0] if "." in part else part
                if len(stem) > 2 and stem.lower() not in seen:
                    seen.add(stem.lower())
                    self._names.append(stem)
                    self._kinds.append("file")
                    self._is_public.append(True)

        print(f"  VecExpanderV2: {len(self._names)} terms (threshold={threshold})")

        # Embed vocabulary
        t0 = time.monotonic()
        vecs = embedder.embed_batch(self._names)
        self._vocab_vecs = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(self._vocab_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vocab_vecs /= norms
        print(f"  VecExpanderV2: embedded in {time.monotonic() - t0:.1f}s")

    def expand(self, query: str, intent: QueryIntent | None = None) -> list[str]:
        """Expand only NL/MIXED queries with threshold + quality filtering."""
        # Intent gate: skip CODE_SYMBOL queries
        if intent is None:
            intent = detect_query_intent(query)
        if intent == QueryIntent.CODE_SYMBOL:
            return []

        qvec = _embed_and_normalize(self._embedder, query)
        scores = self._vocab_vecs @ qvec

        # Sort by score descending
        order = np.argsort(scores)[::-1]

        query_lower = query.lower()
        terms = []
        for i in order:
            if scores[i] < self._threshold:
                break
            name = self._names[i]
            if name.lower() in query_lower:
                continue
            # Quality: prefer public symbols, penalize private
            if not self._is_public[i]:
                # Only include private symbols if very high similarity
                if scores[i] < self._threshold + 0.1:
                    continue
            terms.append(name)
            if len(terms) >= self._top_k:
                break
        return terms


# ---------------------------------------------------------------------------
# Two-Hop Expander: VecV2 → FTS symbol lookup
# ---------------------------------------------------------------------------
class TwoHopExpander:
    """First hop: VecV2 finds relevant symbols.
    Second hop: FTS finds chunks containing those symbols → extract neighbors.
    """

    def __init__(self, fts, vec_v2: VecExpanderV2):
        self._fts = fts
        self._vec_v2 = vec_v2

    def expand(self, query: str, intent: QueryIntent | None = None) -> list[str]:
        # First hop: VecV2
        first_hop = self._vec_v2.expand(query, intent)
        if not first_hop:
            return []

        # Second hop: find chunks containing first-hop symbols via FTS
        all_chunk_ids = set()
        for term in first_hop[:5]:
            # Search by symbol name in symbols table
            rows = self._fts._conn.execute(
                "SELECT DISTINCT chunk_id FROM symbols WHERE name = ?",
                (term,),
            ).fetchall()
            for r in rows:
                all_chunk_ids.add(r[0])

        if not all_chunk_ids:
            return first_hop

        # Extract symbols from those chunks (second-hop neighbors)
        chunk_list = list(all_chunk_ids)[:30]
        placeholders = ",".join("?" for _ in chunk_list)
        rows = self._fts._conn.execute(
            f"SELECT DISTINCT name, kind FROM symbols "
            f"WHERE chunk_id IN ({placeholders}) AND length(name) > 2",
            chunk_list,
        ).fetchall()

        # Combine first-hop + second-hop terms
        query_lower = query.lower()
        seen = set(t.lower() for t in first_hop)
        second_hop = []
        for name, kind in rows:
            if name.lower() not in seen and name.lower() not in query_lower:
                # Prefer classes and functions
                if kind in ("class", "function") or not name.startswith("_"):
                    second_hop.append(name)
                    seen.add(name.lower())
                    if len(second_hop) >= 5:
                        break

        return first_hop + second_hop


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _build_vocab(fts, embedder):
    """Build and embed symbol vocabulary (shared by V1)."""
    rows = fts._conn.execute(
        "SELECT DISTINCT name FROM symbols WHERE length(name) > 2"
    ).fetchall()
    names = [r[0] for r in rows]

    path_rows = fts._conn.execute(
        "SELECT DISTINCT path FROM docs_meta"
    ).fetchall()
    seen = {n.lower() for n in names}
    for r in path_rows:
        parts = r[0].replace("\\", "/").split("/")
        for part in parts:
            stem = part.rsplit(".", 1)[0] if "." in part else part
            if len(stem) > 2 and stem.lower() not in seen:
                seen.add(stem.lower())
                names.append(stem)

    vecs = embedder.embed_batch(names)
    arr = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr /= norms
    return names, arr


def _embed_and_normalize(embedder, text):
    vec = embedder.embed_single(text).astype(np.float32)
    vec /= (np.linalg.norm(vec) or 1.0)
    return vec


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
CONFIGS = [
    {"name": "Baseline",        "method": "none"},
    {"name": "+VecV1",          "method": "vec_v1"},
    {"name": "+VecV2",          "method": "vec_v2"},
    {"name": "+TwoHop",         "method": "two_hop"},
    {"name": "+Concept",        "method": "concept"},
    {"name": "+VecV2+Concept",  "method": "vec_v2_concept"},
]


def expand_query(query, method, vec_v1, vec_v2, two_hop):
    """Apply expansion method and return (expanded_query, terms)."""
    intent = detect_query_intent(query)
    terms = []

    if method == "none":
        pass
    elif method == "vec_v1":
        terms = vec_v1.expand(query)
    elif method == "vec_v2":
        terms = vec_v2.expand(query, intent)
    elif method == "two_hop":
        terms = two_hop.expand(query, intent)
    elif method == "concept":
        terms = concept_expand(query)
    elif method == "vec_v2_concept":
        # Concept vocab first (high precision), then VecV2 to fill gaps
        terms = concept_expand(query)
        vec_terms = vec_v2.expand(query, intent)
        for t in vec_terms:
            if t not in terms:
                terms.append(t)

    if terms:
        return f"{query} {' '.join(terms)}", terms
    return query, []


def run_config(sp, method, vec_v1, vec_v2, two_hop):
    results = []
    t0 = time.monotonic()
    for q in QUERIES:
        expanded_query, added_terms = expand_query(
            q["query"], method, vec_v1, vec_v2, two_hop
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

    print("Building expanders...")
    vec_v1 = VecExpanderV1(sp._fts, sp._embedder, top_k=8)
    vec_v2 = VecExpanderV2(sp._fts, sp._embedder, top_k=5, threshold=0.35)
    two_hop = TwoHopExpander(sp._fts, vec_v2)

    all_results = {}

    for cfg in CONFIGS:
        print(f"\n{'='*70}")
        print(f"Running: {cfg['name']}  (method={cfg['method']})")
        print(f"{'='*70}")

        results, elapsed = run_config(sp, cfg["method"], vec_v1, vec_v2, two_hop)
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
                  f"found={r['found'][:4]}  +[{terms_str}]")

    # --- Comparison table ---
    print(f"\n{'='*95}")
    print("EXPANSION ABLATION V2 — COMPARISON TABLE")
    print(f"{'='*95}")

    baseline = all_results["Baseline"]["agg"]
    print(f"\n{'Config':<18} {'Recall':>8} {'dRecall':>8} {'MRR':>8} {'dMRR':>8} "
          f"{'NDCG@5':>8} {'dNDCG5':>8} {'Top3':>8} {'Zero':>5} {'Time':>7}")
    print("-" * 95)

    for name in [c["name"] for c in CONFIGS]:
        r = all_results[name]
        a = r["agg"]
        dr = a["avg_recall"] - baseline["avg_recall"]
        dm = a["avg_mrr"] - baseline["avg_mrr"]
        dn = a["avg_ndcg5"] - baseline["avg_ndcg5"]
        print(f"  {name:<16} {a['avg_recall']:>8.4f} {dr:>+8.4f} "
              f"{a['avg_mrr']:>8.4f} {dm:>+8.4f} "
              f"{a['avg_ndcg5']:>8.4f} {dn:>+8.4f} "
              f"{a['top3_rate']:>8.4f} {a['zero_recall']:>5d} "
              f"{r['elapsed']:>6.1f}s")

    # --- Per-query delta for each method vs baseline ---
    base_q = all_results["Baseline"]["queries"]
    for compare_name in ["+VecV2", "+TwoHop", "+Concept", "+VecV2+Concept"]:
        cmp_q = all_results[compare_name]["queries"]

        print(f"\n{'='*95}")
        print(f"PER-QUERY DELTA: Baseline -> {compare_name}")
        print(f"{'='*95}")
        print(f"{'ID':<5} {'Cat':<14} {'Base_R':>7} {'New_R':>7} {'dR':>6} "
              f"{'Base_M':>7} {'New_M':>7} {'dM':>6}  Terms")
        print("-" * 95)

        improved = degraded = 0
        for bq, cq in zip(base_q, cmp_q):
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

    # --- Focus: weak queries ---
    weak_ids = ["CQ1", "CQ2", "CQ3", "CQ5", "CQ18"]
    print(f"\n{'='*95}")
    print(f"FOCUS: Weak queries ({', '.join(weak_ids)})")
    print(f"{'='*95}")
    for qid in weak_ids:
        q_def = next(q for q in QUERIES if q["id"] == qid)
        print(f"\n  {qid}: {q_def['query'][:70]}...")
        print(f"  Expected: {q_def['expected']}")
        for name in [c["name"] for c in CONFIGS]:
            q = next(r for r in all_results[name]["queries"] if r["id"] == qid)
            terms = ", ".join(q["expanded_terms"][:6]) if q["expanded_terms"] else "(none)"
            print(f"    {name:<18} recall={q['recall']:.2f} mrr={q['mrr']:.3f} "
                  f"found={q['found'][:4]}  +[{terms}]")

    # --- By category ---
    print(f"\n{'='*95}")
    print("BY CATEGORY — ALL METHODS")
    print(f"{'='*95}")
    cats = sorted(set(r["category"] for r in base_q))
    header = f"{'Category':<16}"
    for name in [c["name"] for c in CONFIGS]:
        header += f" {name:>14}"
    print(header)
    print("-" * (16 + 15 * len(CONFIGS)))
    for cat in cats:
        row = f"  {cat:<14}"
        for name in [c["name"] for c in CONFIGS]:
            cat_rows = [r for r in all_results[name]["queries"] if r["category"] == cat]
            avg_r = sum(r["recall"] for r in cat_rows) / len(cat_rows)
            row += f" {avg_r:>14.3f}"
        print(row)

    # Save JSON
    output = {"experiment": "Query Expansion Ablation V2", "configs": {}}
    for name, data in all_results.items():
        output["configs"][name] = {
            "method": data["config"]["method"],
            "elapsed": round(data["elapsed"], 2),
            **data["agg"],
            "queries": data["queries"],
        }
    out_path = PROJECT_ROOT / "bench_ablation_expansion_v2.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
