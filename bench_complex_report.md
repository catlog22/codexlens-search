# Complex Benchmark: ACE vs Codexlens-Search (20 Advanced Queries)

**Project**: codexlens-search
**Date**: 2026-03-23
**Query Design**: Cross-module architectural, Abstract behavioral, Negative/inverse, Implementation detail, Integration flow

---

## Overall Results

| Metric | ACE | Codexlens | Delta | Winner |
|--------|-----|-----------|-------|--------|
| **Avg Recall@10** | **1.0000** | 0.8333 | +0.1667 | ACE |
| **Avg MRR** | **0.9167** | 0.9000 | +0.0167 | ACE |
| **Avg NDCG@5** | **0.9363** | 0.8236 | +0.1127 | ACE |
| **Top-3 Hit Rate** | **1.0000** | 0.9500 | +0.0500 | ACE |
| **Zero-recall** | **0** | 1 | - | ACE |

**Query Wins**: ACE=7, Codexlens=1, Tie=12

---

## Results by Category

| Category | ACE Recall | CL Recall | ACE MRR | CL MRR | Notes |
|----------|-----------|-----------|---------|--------|-------|
| cross-module | **1.000** | 0.333 | **0.833** | 0.667 | ACE dominates — multi-file queries are its strength |
| behavioral | **1.000** | 0.792 | 0.708 | **0.875** | CL ranks better when it finds files, but ACE has better coverage |
| negative | 1.000 | 1.000 | 1.000 | 1.000 | Perfect tie |
| impl-detail | 1.000 | 1.000 | **1.000** | 0.900 | Near-tie, ACE slightly better ranking |
| integration | **1.000** | 0.917 | 1.000 | 1.000 | ACE finds more files in multi-module flows |

---

## Results by Difficulty

| Difficulty | ACE Recall | CL Recall | ACE MRR | CL MRR |
|------------|-----------|-----------|---------|--------|
| medium (10) | **1.000** | 0.900 | **0.950** | 0.900 |
| hard (10) | **1.000** | 0.767 | 0.883 | **0.900** |

---

## Per-Query Detail

| ID | Category | Diff | ACE Recall | CL Recall | ACE MRR | CL MRR | Winner |
|----|----------|------|-----------|-----------|---------|--------|--------|
| CQ1 | cross-module | hard | **1.00** | 0.50 | 1.000 | 1.000 | **ACE** |
| CQ2 | cross-module | hard | **1.00** | 0.00 | **0.500** | 0.000 | **ACE** |
| CQ3 | cross-module | medium | **1.00** | 0.50 | 1.000 | 1.000 | **ACE** |
| CQ4 | behavioral | hard | **1.00** | 0.67 | 0.333 | **1.000** | **CL** |
| CQ5 | behavioral | medium | **1.00** | 0.50 | 0.500 | 0.500 | **ACE** |
| CQ6 | behavioral | medium | 1.00 | 1.00 | 1.000 | 1.000 | TIE |
| CQ7 | behavioral | medium | 1.00 | 1.00 | 1.000 | 1.000 | TIE |
| CQ8 | negative | hard | 1.00 | 1.00 | 1.000 | 1.000 | TIE |
| CQ9 | negative | medium | 1.00 | 1.00 | 1.000 | 1.000 | TIE |
| CQ10 | impl-detail | hard | 1.00 | 1.00 | 1.000 | 1.000 | TIE |
| CQ11 | impl-detail | medium | 1.00 | 1.00 | 1.000 | 1.000 | TIE |
| CQ12 | impl-detail | hard | 1.00 | 1.00 | 1.000 | 1.000 | TIE |
| CQ13 | impl-detail | medium | 1.00 | 1.00 | 1.000 | 1.000 | TIE |
| CQ14 | impl-detail | medium | 1.00 | 1.00 | **1.000** | 0.500 | **ACE** |
| CQ15 | integration | hard | 1.00 | 1.00 | 1.000 | 1.000 | TIE |
| CQ16 | integration | medium | 1.00 | 1.00 | 1.000 | 1.000 | TIE |
| CQ17 | integration | medium | 1.00 | 1.00 | 1.000 | 1.000 | TIE |
| CQ18 | integration | hard | **1.00** | 0.50 | 1.000 | 1.000 | **ACE** |
| CQ19 | integration | hard | 1.00 | 1.00 | 1.000 | 1.000 | **ACE** |
| CQ20 | integration | hard | 1.00 | 1.00 | 1.000 | 1.000 | TIE |

---

## Key Findings

### ACE Strengths
1. **Perfect recall (1.0)** across all 20 queries — ACE never misses an expected file
2. **Cross-module queries**: ACE excels at finding files scattered across multiple packages (CQ1, CQ2, CQ3). Where codexlens found only 33% of expected files in cross-module queries, ACE found 100%
3. **ABC/interface pattern** (CQ2): Codexlens completely missed all 3 expected base class files (`core/base.py`, `embed/base.py`, `rerank/base.py`), while ACE found all 3. This is a significant weakness for pattern-oriented queries
4. **Multi-hop integration flows** (CQ18, CQ19): ACE finds all components in complex data flows spanning 3+ modules

### Codexlens Strengths
1. **Better ranking in some cases** (CQ4): When codexlens finds files, it sometimes ranks the most relevant one higher (MRR 1.0 vs ACE's 0.333)
2. **Implementation detail queries**: Near-perfect on focused, single-module queries (impl-detail category: recall 1.0, MRR 0.9)
3. **Speed**: ~24 seconds for 20 queries (local inference), while ACE requires cloud API calls

### Codexlens Weaknesses to Address
1. **Cross-module coverage gap**: 33% recall on cross-module queries. The 2-stage search (binary coarse → ANN fine) may miss files when relevant content uses different vocabulary
2. **Abstract concept queries**: Queries describing behavior without technical keywords (CQ2, CQ5) have lower recall
3. **Graph search dependency**: CQ18 (graph traversal) shows codexlens only found 50% of expected files despite having a graph search module — the graph may not be seeded effectively for all queries

### Comparison with Simple Benchmark (Previous 20 Queries)

| Metric | Simple (ACE) | Simple (CL) | Complex (ACE) | Complex (CL) |
|--------|-------------|-------------|--------------|--------------|
| Avg Recall | 1.000 | 0.975 | **1.000** | 0.833 |
| Avg MRR | 0.871 | 0.867 | **0.917** | 0.900 |
| Top-3 Rate | 1.000 | 0.950 | **1.000** | 0.950 |

Complex queries expose a bigger gap: codexlens recall drops from 0.975 to 0.833, while ACE maintains perfect 1.0 recall. The gap widens from ~2.5% to ~16.7% on recall, confirming that complex, multi-module queries are significantly harder for local semantic search.

---

## Methodology

- **Ground truth**: Manually verified expected files for each query based on actual source code analysis
- **Evaluation**: Recall@10, MRR (first relevant result rank), NDCG@5 (position-aware), Top-3 hit rate
- **Codexlens**: Direct Python API call via `SearchPipeline.search(query, top_k=20)` with BGE-small-en-v1.5 embeddings + FAISS binary + USearch ANN + FTS5 + RRF fusion + cross-encoder reranking
- **ACE**: MCP tool `search_context` with cloud-based proprietary retrieval model
- **Winner scoring**: Weighted composite = 0.5×Recall + 0.3×MRR + 0.2×NDCG@5, with ±0.01 tolerance for ties
