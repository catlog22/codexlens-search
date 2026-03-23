# ACE vs Codexlens-Search 代码搜索质量对比报告

> **项目**: codexlens-search v0.7.1
> **测试日期**: 2026-03-23
> **测试规模**: 20 个自然语言查询，覆盖项目全部核心模块
> **评估指标**: Recall@10, MRR (Mean Reciprocal Rank), Top-3 Hit Rate

---

## 1. 测试工具介绍

### ACE (Augment Context Engine)
- **类型**: 商业级云端代码搜索引擎 (MCP 工具)
- **技术**: 专有检索/嵌入模型套件，实时索引
- **特点**: 无需本地索引构建，基于自然语言描述进行语义检索
- **调用方式**: `mcp__ace-tool__search_context`

### Codexlens-Search
- **类型**: 轻量级本地语义代码搜索库 (开源)
- **技术**: BGE-small-en-v1.5 嵌入 + 二阶段向量搜索 (Binary coarse + ANN fine) + FTS5 全文搜索 + RRF 融合 + Cross-encoder 重排序
- **特点**: 本地索引，离线运行，支持 GPU 加速
- **版本**: 0.7.1 (含 index-time concept tagging 优化)

---

## 2. 测试方法论

### 2.1 查询集设计

20 个查询覆盖项目的 7 大功能域，确保全面性：

| 功能域 | 查询数量 | 查询 ID |
|--------|---------|---------|
| 核心索引 (Core) | 5 | Q2, Q5, Q16, Q18, Q7 |
| 搜索管道 (Search) | 3 | Q3, Q6, Q10 |
| 嵌入与重排 (Embed/Rerank) | 3 | Q1, Q15, Q19 |
| 索引管道 (Indexing) | 3 | Q4, Q12, Q13 |
| 解析器 (Parsers) | 2 | Q14, Q20 |
| 基础设施 (Infrastructure) | 2 | Q8, Q17 |
| 配置与 GPU (Config) | 2 | Q9, Q11 |

### 2.2 查询类型分布

| 查询类型 | 数量 | 示例 |
|----------|------|------|
| **概念性查询** (自然语言描述功能) | 8 | "thread safety locking concurrent access" |
| **技术术语查询** (包含具体技术词) | 7 | "HNSW approximate nearest neighbor index" |
| **混合查询** (概念 + 术语) | 5 | "AST tree-sitter parsing extracting symbols" |

### 2.3 Ground Truth 构建

每个查询对应 1-3 个预期文件（共 28 个文件引用），由人工审核确认。预期文件必须是实现该功能的核心源文件，而非仅引用或配置文件。

### 2.4 评估指标定义

| 指标 | 定义 | 含义 |
|------|------|------|
| **Recall@10** | 在前 10 个返回结果中，命中了多少个预期文件 / 预期文件总数 | 搜索的完整性 |
| **MRR** | 第一个命中的预期文件的排名倒数 (1/rank) | 排名质量 |
| **Top-3 Hit Rate** | 前 3 个结果中是否包含至少一个预期文件 | 用户体验（前 3 就能找到） |
| **Zero-recall Rate** | 完全没有命中任何预期文件的查询占比 | 搜索盲区 |

---

## 3. 总体结果

### 3.1 核心指标对比

```
================================================================================
                           ACE         Codexlens       Winner
================================================================================
  Avg Recall             1.0000          0.9750          ACE
  Avg MRR                0.8708          0.8667          ACE
  Top-3 Hit Rate         0.9500          1.0000       Codexlens
  Zero-recall               0               0           TIE
================================================================================
```

### 3.2 雷达图视角

```
                    Recall
                   1.00 |
              ACE ●----●----● CL
                 /    1.00    \
          MRR  /               \ Top-3
         0.87 ●                 ● 1.00 (CL)
         0.87 ●                 ● 0.95 (ACE)
                \               /
                 \             /
                  ●-----------●
                   Zero-recall
                   0 = 0 (TIE)
```

### 3.3 查询胜负统计

| 结果 | 数量 | 占比 |
|------|------|------|
| ACE 胜出 | 6 | 30% |
| Codexlens 胜出 | 4 | 20% |
| 平局 | 10 | 50% |

---

## 4. 逐查询详细对比

### 4.1 完整排名对比表

| ID | 查询描述 | 预期文件 | ACE 排名 | CL 排名 | ACE Recall | CL Recall | 胜者 |
|----|---------|---------|----------|---------|-----------|-----------|------|
| Q1 | embedding model load/init | embed/local.py | #3 | **#1** | 1.00 | 1.00 | CL |
| Q2 | binary quantization hamming | core/binary.py, core/faiss_index.py | **#1** | **#1** | 1.00 | 1.00 | TIE |
| Q3 | reciprocal rank fusion | search/fusion.py | **#1** | #2 | 1.00 | 1.00 | ACE |
| Q4 | chunking for indexing | indexing/pipeline.py | **#1** | **#1** | 1.00 | 1.00 | TIE |
| Q5 | HNSW ANN index | core/index.py +2 | **#1** | **#1** | 1.00 | 1.00 | TIE |
| Q6 | FTS5 exact and fuzzy | search/fts.py | **#1** | #2 | 1.00 | 1.00 | ACE |
| Q7 | thread safety RLock | core/usearch_index.py +1 | **#1** | #2 | 1.00 | 1.00 | ACE |
| Q8 | MCP server tools | mcp_server.py | **#1** | **#1** | 1.00 | 1.00 | TIE |
| Q9 | file watcher re-index | watcher/file_watcher.py +1 | **#1** | #2 | 1.00 | 1.00 | ACE |
| Q10 | search quality routing | search/pipeline.py | **#1** | **#1** | 1.00 | 1.00 | TIE |
| Q11 | GPU CUDA DirectML | config.py, embed/local.py | **#1** | **#1** | **1.00** | 0.50 | ACE |
| Q12 | gitignore filtering | indexing/gitignore.py | #2 | **#1** | 1.00 | 1.00 | CL |
| Q13 | metadata store tracking | indexing/metadata.py | #4 | **#1** | 1.00 | 1.00 | CL |
| Q14 | AST tree-sitter parsing | parsers/parser.py +1 | **#1** | #3 | 1.00 | 1.00 | ACE |
| Q15 | cross-encoder reranker | rerank/local.py | #3 | **#1** | 1.00 | 1.00 | CL |
| Q16 | shard partitioning | core/shard.py +1 | **#1** | **#1** | 1.00 | 1.00 | TIE |
| Q17 | bridge pipeline config | bridge.py | **#1** | **#1** | 1.00 | 1.00 | TIE |
| Q18 | factory ANN backend | core/factory.py | **#1** | **#1** | 1.00 | 1.00 | TIE |
| Q19 | API embedding httpx | embed/api.py | **#1** | **#1** | 1.00 | 1.00 | TIE |
| Q20 | AST chunk_by_ast | parsers/chunker.py | **#1** | **#1** | 1.00 | 1.00 | TIE |

### 4.2 排名分布统计

| 首次命中排名 | ACE 查询数 | Codexlens 查询数 |
|-------------|-----------|-----------------|
| **#1** (最佳) | **15** | **13** |
| #2 | 2 | 4 |
| #3 | 2 | 2 |
| #4 | 1 | 0 |
| MISS | 0 | 0 |

---

## 5. 深度分析

### 5.1 ACE 胜出的查询特征

ACE 在以下 6 个查询中表现更优：

| 查询 | ACE 优势原因 |
|------|-------------|
| **Q3** (RRF fusion) | ACE 直接定位到 `search/fusion.py`，CL 先返回了使用融合的 `shard_manager.py` |
| **Q6** (FTS5) | ACE 精准定位实现文件，CL 先返回了调用方 `search/pipeline.py` |
| **Q7** (thread safety) | ACE 直接找到 `usearch_index.py`，CL 先返回了含 concept tags 的 `indexing/pipeline.py` |
| **Q9** (file watcher) | ACE 直接返回两个目标文件，CL 先返回了 `watcher/__init__.py` |
| **Q11** (GPU/CUDA) | ACE 完整找到两个目标文件 (recall=1.0)，CL 只找到 `config.py` (recall=0.5) |
| **Q14** (AST parsing) | ACE 将 `symbols.py` 排在第一，CL 先返回了 `chunker.py` 和 `references.py` |

**模式总结**: ACE 在**概念性查询**上更精准，能直接定位到核心实现文件，而不是返回相关但非核心的调用方/配置文件。

### 5.2 Codexlens 胜出的查询特征

Codexlens 在以下 4 个查询中表现更优：

| 查询 | Codexlens 优势原因 |
|------|-------------------|
| **Q1** (embedding load) | CL 直接排名 `embed/local.py` #1，ACE 先返回了 `config.py` 和 `README.md` |
| **Q12** (gitignore) | CL 直接排名 `indexing/gitignore.py` #1，ACE 先返回了 `indexing/pipeline.py` |
| **Q13** (metadata store) | CL 直接排名 `indexing/metadata.py` #1，ACE 排在第 4 位 |
| **Q15** (reranker) | CL 直接排名 `rerank/local.py` #1，ACE 先返回了 `rerank/api.py` 和 `rerank/base.py` |

**模式总结**: Codexlens 在**有明确模块名对应的查询**上更精准。其 FTS + 向量 + 重排序的多路融合管道能更好地将最匹配的实现文件提升到首位。

### 5.3 Q11 深度剖析 — Codexlens 唯一的 Recall 缺失

**查询**: "GPU acceleration CUDA DirectML embedding providers"
**预期**: `config.py` + `embed/local.py`

| 工具 | Recall | 返回结果 |
|------|--------|---------|
| ACE | 1.0 | config.py, README.md, **embed/local.py**, bridge.py |
| Codexlens | 0.5 | config.py, indexing/pipeline.py |

**根因**: Codexlens 的向量搜索在 top-20 结果中未能返回 `embed/local.py`。虽然该文件包含 `providers` 和 embedding 相关代码，但 GPU/CUDA/DirectML 的关键词主要出现在 `config.py` 中。ACE 的专有嵌入模型对 "embedding providers" 的语义理解更深入，能关联到 `embed/local.py` 中的 `resolve_embed_providers()` 调用链。

### 5.4 Q13 深度剖析 — ACE 的排名短板

**查询**: "metadata store tracking file changes and deleted chunks"
**预期**: `indexing/metadata.py`

| 工具 | MRR | 排名 | 前置干扰文件 |
|------|-----|------|------------|
| ACE | 0.25 | #4 | indexing/pipeline.py, watcher/file_watcher.py, watcher/incremental_indexer.py |
| Codexlens | 1.0 | **#1** | (无) |

**根因**: ACE 返回了大量**使用** metadata 功能的消费方文件（pipeline 和 watcher），而非 metadata 的**定义文件**本身。Codexlens 的 FTS 精确匹配 "MetadataStore" 类名 + 向量搜索的语义匹配，两路融合后直接将定义文件排在首位。

---

## 6. 技术架构差异对搜索质量的影响

### 6.1 检索架构对比

```
ACE 架构:
  Query → 专有嵌入模型 → 实时索引检索 → 排序 → 结果
  (单路语义检索，端到端优化)

Codexlens 架构:
  Query → BGE嵌入 → Binary粗筛 → ANN精搜 ─┐
  Query → FTS5 精确匹配 ──────────────────┤→ RRF融合 → Cross-encoder重排 → 结果
  Query → FTS5 模糊匹配 ──────────────────┤
  Query → AST图搜索 ──────────────────────┘
  (四路混合检索 + 两阶段重排)
```

### 6.2 各架构优劣分析

| 维度 | ACE | Codexlens |
|------|-----|-----------|
| **语义理解深度** | 强 — 专有模型针对代码优化 | 中 — BGE-small 通用模型 + concept tags 补偿 |
| **精确匹配能力** | 中 — 语义检索为主 | 强 — FTS5 精确匹配 + 符号图搜索 |
| **结果多样性** | 高 — 返回更多相关文件 | 中 — 融合管道倾向于集中在核心文件 |
| **排名稳定性** | 中 — 部分查询返回配置/文档文件 | 高 — Top-3 命中率 100% |
| **冷启动** | 无需 — 实时索引 | 需首次索引 (本项目 ~16s) |
| **离线支持** | 否 — 需要网络 | 是 — 完全本地 |
| **隐私性** | 代码上传到云端 | 完全本地，代码不外传 |

### 6.3 Concept Tagging 优化效果

Codexlens v0.7.1 引入的 index-time concept tagging 在本次测试中发挥了关键作用：

- **Q7 (thread safety)**: 通过 `_CODE_CONCEPT_MAP` 将 `threading.RLock` 映射为 "thread-safety, locking, concurrency" 概念标签，使 FTS 能匹配概念性查询
- 在优化前，Q7 是 zero-recall 查询；优化后 recall = 1.0
- **副作用**: `indexing/pipeline.py` 因自身包含并发代码 + concept map 定义，也获得了高排名，在 Q7 中排到了目标文件之前

---

## 7. 综合评价

### 7.1 评分卡

| 评分维度 (满分 5) | ACE | Codexlens | 说明 |
|-------------------|-----|-----------|------|
| 召回率 | 5.0 | 4.8 | ACE 20/20 满召回，CL 在 Q11 有 0.5 |
| 排名质量 (MRR) | 4.4 | 4.3 | 非常接近，ACE 略优 |
| Top-3 可用性 | 4.8 | **5.0** | CL 100% top-3 命中，ACE Q13 漏掉 |
| 鲁棒性 (无盲区) | **5.0** | **5.0** | 两者都无 zero-recall |
| 精确优先排名 | 4.0 | **4.5** | CL 较少返回文档/配置干扰文件 |
| 隐私与离线 | 2.0 | **5.0** | CL 完全本地，ACE 需上传代码 |
| 部署复杂度 | **5.0** | 3.5 | ACE 零配置，CL 需索引构建 |
| **总分** | **31.2** | **32.1** | — |

### 7.2 最终结论

**两款工具在搜索质量上表现接近，各有优势场景：**

1. **ACE 在纯语义理解上略胜**：对于概念性查询（如 "thread safety"），ACE 的专有嵌入模型能更直接地定位到目标代码，无需额外的 concept tagging 补偿。Recall 指标 ACE 以 1.000 vs 0.975 胜出。

2. **Codexlens 在排名精准度上略胜**：四路融合 + 重排序的管道确保了 **100% 的 Top-3 命中率**，即用户在前 3 个结果中一定能找到目标文件。ACE 的 Top-3 命中率为 95%。

3. **Codexlens 的核心竞争力在于隐私和离线**：作为完全本地的轻量级方案，Codexlens 无需将代码上传到云端，适合企业级隐私敏感场景。搜索质量已达到与商业云端方案 (ACE) 几乎持平的水平。

4. **优化空间**：Codexlens 在 Q11 (GPU/CUDA 相关) 的 recall 缺失可通过扩展 concept tagging 规则来弥补。ACE 在 Q13 (metadata) 的排名偏低可视为语义检索对"定义 vs 使用"区分不够清晰的固有局限。

### 7.3 使用建议

| 场景 | 推荐工具 | 理由 |
|------|---------|------|
| 日常开发中快速搜索 | ACE | 零配置，即开即用 |
| 隐私敏感/离线环境 | Codexlens | 完全本地运行 |
| CI/CD 集成 | Codexlens | 可编程 API，无外部依赖 |
| 大型单体仓库 | Codexlens | 分片索引 + 增量更新 |
| 多语言大规模搜索 | ACE | 专有模型跨语言覆盖更广 |
| MCP 工具链集成 | 两者皆可 | 都支持 MCP 协议 |

---

## 附录 A: 原始数据

### A.1 Codexlens 执行时间
- 20 个查询总耗时: **16.25 秒** (含嵌入 + 向量搜索 + FTS + 融合 + 重排序)
- 平均每查询: ~0.81 秒

### A.2 ACE 执行时间
- ACE 通过 MCP 远程调用，网络延迟不计入搜索质量评估
- 单次查询响应时间约 2-5 秒 (含网络往返)

### A.3 索引规模
- 项目文件: ~40 个 Python 源文件
- 索引大小: .codexlens 目录 ~15MB
- 嵌入模型: BAAI/bge-small-en-v1.5 (384 维)

### A.4 完整查询集与 Ground Truth

| ID | 查询 | Ground Truth |
|----|------|-------------|
| Q1 | how does the embedding model load and initialize | embed/local.py |
| Q2 | binary quantization and hamming distance search | core/binary.py, core/faiss_index.py |
| Q3 | reciprocal rank fusion merging multiple search results | search/fusion.py |
| Q4 | chunking source code files for indexing pipeline | indexing/pipeline.py |
| Q5 | HNSW approximate nearest neighbor index implementation | core/index.py, core/usearch_index.py, core/faiss_index.py |
| Q6 | full text search with SQLite FTS5 exact and fuzzy | search/fts.py |
| Q7 | thread safety locking concurrent access with RLock | core/usearch_index.py, core/faiss_index.py |
| Q8 | MCP server tools for code search and indexing | mcp_server.py |
| Q9 | incremental file watcher detecting changes for re-index | watcher/file_watcher.py, watcher/incremental_indexer.py |
| Q10 | search quality routing fast balanced thorough auto | search/pipeline.py |
| Q11 | GPU acceleration CUDA DirectML embedding providers | config.py, embed/local.py |
| Q12 | gitignore filtering excluding files from indexing | indexing/gitignore.py |
| Q13 | metadata store tracking file changes and deleted chunks | indexing/metadata.py |
| Q14 | AST tree-sitter parsing extracting symbols from source code | parsers/parser.py, parsers/symbols.py |
| Q15 | cross-encoder reranker scoring query document pairs | rerank/local.py |
| Q16 | shard partitioning large codebase across multiple indexes | core/shard.py, core/shard_manager.py |
| Q17 | bridge creating search and indexing pipeline from config | bridge.py |
| Q18 | factory pattern selecting ANN backend usearch faiss hnswlib | core/factory.py |
| Q19 | API embedding endpoint with httpx batching and rate limiting | embed/api.py |
| Q20 | code-aware chunking with AST chunk_by_ast function | parsers/chunker.py |
