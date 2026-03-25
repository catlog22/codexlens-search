# codexlens-search 搜索策略

多通道并行检索 + 融合 + 重排序。

```
Query → 意图检测 → 查询扩展(仅NL)
  ├─ Binary粗筛 (Hamming) → ANN精排 (cosine) → 交集 → vector结果
  ├─ FTS精确 (BM25) + FTS模糊 (prefix*)
  ├─ 符号图搜索 (seeded from vector/FTS top results)
  └─ Ripgrep正则 (auto模式)
       └─ RRF加权融合 → Cross-Encoder重排序 → top-k
```

---

## 意图检测与查询扩展

检测 camelCase/snake_case/代码关键词/疑问词等信号，分为 `CODE_SYMBOL`、`NATURAL_LANGUAGE`、`MIXED` 三类。

对 NL 查询做 Two-Hop 扩展：查询向量与符号词汇表余弦匹配(>0.35)得到 first-hop 符号，再从 FTS 中找同块共现的邻居符号，追加到查询中。

---

## 检索通道

| 通道 | 原理 | 作用 |
|------|------|------|
| **Binary粗筛** | float32→符号位量化→XOR+popcount Hamming距离 | O(N)位运算快速缩小候选到 top-200 |
| **ANN精排** | HNSW图 cosine近邻(USearch/FAISS/hnswlib) | 从 top-200 候选中精排 top-50 |
| **FTS** | SQLite FTS5, Porter stemming, BM25 + prefix* | 关键词精确/模糊匹配 |
| **符号图** | 沿 import/call/inherit/type_ref 边双向遍历 | 发现结构相关但语义不相似的代码 |
| **Ripgrep** | 实时正则匹配 | 精确 pattern 搜索，无需索引 |

---

## RRF 融合

```
score(doc) = Σ weight[source] × 1/(60 + rank)
```

| 意图 | vector | exact | fuzzy | graph |
|------|--------|-------|-------|-------|
| CODE_SYMBOL | 0.25 | **0.35** | 0.05 | **0.35** |
| NATURAL_LANGUAGE | **0.70** | 0.10 | 0.10 | 0.10 |
| MIXED | 0.45 | 0.25 | 0.10 | 0.20 |

融合后 top-50 送入 cross-encoder reranker 输出最终结果。

---

## 质量路由

| 模式 | Binary | ANN | FTS | Graph | 选择逻辑 |
|------|:------:|:---:|:---:|:-----:|---------|
| `fast` | - | - | yes | - | 无向量索引时 |
| `balanced` | yes | - | yes | yes | 折中 |
| `thorough` | yes | yes | yes | yes | 完整管线 |
| `auto` | 自动 | 自动 | 自动 | 自动 | 有索引→thorough，无→fast |

---

## 冷启动

无索引时：拆词 → `rg --count` 找 top-50 相关文件 → 仅索引这些文件(~10s) → 语义搜索 → 后台异步构建全量索引。
