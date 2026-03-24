# codexlens-search

面向 Claude Code 的语义代码搜索引擎 MCP 服务。

混合搜索：向量 + 全文 + AST 符号图 + ripgrep 正则 — RRF 融合 + 重排序。

[English](README.md)

## 快速开始

```bash
pip install codexlens-search[all]
```

在项目 `.mcp.json` 中添加：

```json
{
  "mcpServers": {
    "codexlens": {
      "command": "uvx",
      "args": ["--from", "codexlens-search[all]", "codexlens-mcp"],
      "env": {
        "CODEXLENS_EMBED_API_URL": "https://api.openai.com/v1",
        "CODEXLENS_EMBED_API_KEY": "${OPENAI_API_KEY}",
        "CODEXLENS_EMBED_API_MODEL": "text-embedding-3-small",
        "CODEXLENS_EMBED_DIM": "1536"
      }
    }
  }
}
```

完成。Claude Code 会自动发现工具：`index_project` -> `Search`。

## 安装

选择匹配平台的安装方式：

```bash
# 基础版 — CPU 推理（fastembed 内置 onnxruntime CPU）
pip install codexlens-search

# Windows GPU — DirectML，支持任何 DirectX 12 GPU（NVIDIA/AMD/Intel）
pip install codexlens-search[directml]

# Linux/Windows NVIDIA GPU — CUDA（需要 CUDA + cuDNN）
pip install codexlens-search[cuda]

# 自动选择 — Windows 用 DirectML，其他平台用 CPU
pip install codexlens-search[all]
```

### 平台推荐

| 平台 | 推荐 | 命令 |
|------|------|------|
| **Windows + 任意 GPU** | `[directml]` | `pip install codexlens-search[directml]` |
| **Windows 仅 CPU** | 基础版 | `pip install codexlens-search` |
| **Linux + NVIDIA GPU** | `[cuda]` | `pip install codexlens-search[cuda]` |
| **Linux CPU / AMD GPU** | 基础版 | `pip install codexlens-search` |
| **macOS (Apple Silicon)** | 基础版 | `pip install codexlens-search` |
| **不确定 / CI** | `[all]` | `pip install codexlens-search[all]` |

> **注意**：Windows 下安装基础版（不带 `[directml]`），MCP 服务器会在首次启动时自动检测并安装 `onnxruntime-directml`，第二次启动后 GPU 生效。

### 包含功能

所有安装方式均包含：

- **MCP 服务器** — `codexlens-mcp` 命令
- **AST 解析** — tree-sitter 符号提取 + 图搜索
- **USearch** — 高性能 HNSW ANN 后端（默认）
- **FAISS** — ANN + 二值索引后端（Hamming 粗筛）
- **文件监控** — watchdog 自动重新索引
- **Gitignore 过滤** — 递归 `.gitignore` 支持
- **聚焦搜索** — 无索引时，grep 相关文件 → 仅索引这些文件（~10s）→ 语义搜索，无需等待全量索引

### ANN 后端选择

三种近似最近邻搜索后端，按优先级自动选择：

| 后端 | 安装方式 | 适用场景 |
|------|---------|---------|
| `usearch`（默认） | 内置 | 跨平台，CPU HNSW 最快 |
| `faiss` | 内置 | GPU 加速，二值 Hamming 搜索 |
| `hnswlib` | 内置 | 轻量级备选 |

通过 `CODEXLENS_ANN_BACKEND` 覆盖：

```bash
CODEXLENS_ANN_BACKEND=faiss    # 使用 FAISS（有 GPU 时自动启用）
CODEXLENS_ANN_BACKEND=usearch  # 使用 USearch（默认）
CODEXLENS_ANN_BACKEND=hnswlib  # 使用 hnswlib
CODEXLENS_ANN_BACKEND=auto     # 自动选择（usearch > faiss > hnswlib）
```

## MCP 工具

### Search

混合代码搜索，结合语义向量、全文检索、AST 符号图和 ripgrep 正则。

| 模式 | 说明 | 依赖 |
|------|------|------|
| `auto`（默认） | 语义 + 正则并行。无索引？聚焦搜索 ~10s 出结果。 | |
| `symbol` | 按名称查找符号定义（精确/模糊） | 索引 |
| `refs` | 查找交叉引用 — 入引用和出引用 | 索引 |
| `regex` | ripgrep 正则匹配 | rg |

参数：`project_path`、`query`、`mode`、`scope`（限制搜索子目录）

结果数量由 `CODEXLENS_TOP_K` 环境变量控制（默认 10）。

#### 冷启动搜索

无索引时，`auto` 模式使用聚焦搜索管线，无需等待全量索引构建：

1. **展开查询** — 拆分 camelCase/snake_case 为搜索词
2. **Grep 文件** — `rg --count` 找到 top 50 相关文件，按匹配数排序
3. **索引** — 仅嵌入这 50 个文件（GPU ~8-10s）
4. **搜索** — 在新索引上进行语义向量搜索
5. **后台** — 异步构建全量索引供后续查询使用

首次查询 ~10s 出结果，而非等待全量索引 ~100s。

### index_project

构建、更新或检查搜索索引。

| 操作 | 说明 |
|------|------|
| `sync`（默认） | 增量索引 — 仅处理变更文件 |
| `rebuild` | 全量重建索引 |
| `status` | 索引统计（文件数、块数、符号数、引用数） |

参数：`project_path`、`action`、`scope`

### find_files

基于 glob 的文件发现。参数：`project_path`、`pattern`（默认 `**/*`）

结果上限由 `CODEXLENS_FIND_MAX_RESULTS` 控制（默认 100）。

### watch_project

管理文件监控器，文件变更时自动重新索引。

参数：`project_path`、`action`（`start` / `stop` / `status`）

## AST 功能

默认启用。通过 `CODEXLENS_AST_CHUNKING=false` 禁用。

- **智能分块** — 在符号边界（函数/类定义）处切分，而非固定字符数
- **符号提取** — 12 种类型：function、class、method、module、variable、constant、interface、type_alias、enum、struct、trait、property
- **交叉引用** — import、call、inherit、type_ref 边
- **图搜索** — 以向量/FTS 结果为种子，BFS 扩展，自适应边权重
- **查询扩展** — 两跳符号词汇表扩展，提升自然语言查询召回率

支持语言：Python、JavaScript、TypeScript、Go、Java、Rust、C、C++、Ruby、PHP、Scala、Kotlin、Swift、C#、Bash、Lua、Haskell、Elixir、Erlang。

## 配置示例

### 重排序器（最佳质量）

在快速开始配置基础上添加重排序 API：

```json
"CODEXLENS_RERANKER_API_URL": "https://api.jina.ai/v1",
"CODEXLENS_RERANKER_API_KEY": "${JINA_API_KEY}",
"CODEXLENS_RERANKER_API_MODEL": "jina-reranker-v2-base-multilingual"
```

### 多端点负载均衡

```json
"CODEXLENS_EMBED_API_ENDPOINTS": "https://api1.example.com/v1|sk-key1|model,https://api2.example.com/v1|sk-key2|model",
"CODEXLENS_EMBED_DIM": "1536"
```

格式：`url|key|model,url|key|model,...` — 替代单端点 `EMBED_API_URL/KEY/MODEL`。

### 本地模型（离线）

无需 API — `fastembed` 通过 ONNX runtime 在本地运行模型。

```bash
# 查看可用模型
codexlens-search list-models

# 预下载模型（可选 — 首次使用时自动下载）
codexlens-search download-models

# 下载特定模型
codexlens-search download-model nomic-ai/nomic-embed-text-v1.5-Q
```

```json
{
  "mcpServers": {
    "codexlens": {
      "command": "codexlens-mcp",
      "env": {
        "CODEXLENS_DEVICE": "directml"
      }
    }
  }
}
```

默认本地模型：`BAAI/bge-small-en-v1.5`（384d，512 tokens）。使用其他模型：

```json
{
  "mcpServers": {
    "codexlens": {
      "command": "codexlens-mcp",
      "env": {
        "CODEXLENS_EMBED_MODEL": "nomic-ai/nomic-embed-text-v1.5-Q",
        "CODEXLENS_EMBED_DIM": "768",
        "CODEXLENS_DEVICE": "directml"
      }
    }
  }
}
```

#### 可用本地模型

**通用**

| 模型 | 维度 | 上下文 | 大小 | 说明 |
|------|------|--------|------|------|
| `BAAI/bge-small-en-v1.5` | 384 | 512 | 68MB | 默认，最快 |
| `BAAI/bge-base-en-v1.5` | 768 | 512 | 215MB | 质量更好 |
| `BAAI/bge-large-en-v1.5` | 1024 | 512 | 1.2GB | 英文最佳质量 |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 256 | 92MB | 轻量通用 |
| `snowflake/snowflake-arctic-embed-xs` | 384 | 512 | 92MB | 紧凑，质量好 |
| `snowflake/snowflake-arctic-embed-s` | 384 | 512 | 133MB | 轻量，优于 xs |

**代码 / 长上下文**

| 模型 | 维度 | 上下文 | 大小 | 说明 |
|------|------|--------|------|------|
| `jinaai/jina-embeddings-v2-base-code` | 768 | 8192 | 655MB | 代码专精，30+ 编程语言 |
| `nomic-ai/nomic-embed-text-v1.5-Q` | 768 | 8192 | 133MB | 量化版，代码搜索性价比最高 |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | 8192 | 532MB | 长上下文，代码+文本 |
| `jinaai/jina-embeddings-v2-small-en` | 512 | 8192 | 122MB | 长上下文，轻量 |

**中文 / 多语言**

| 模型 | 维度 | 上下文 | 大小 | 说明 |
|------|------|--------|------|------|
| `BAAI/bge-small-zh-v1.5` | 512 | 512 | 92MB | 中文，快速 |
| `BAAI/bge-large-zh-v1.5` | 1024 | 512 | 1.2GB | 中文，最佳质量 |
| `jinaai/jina-embeddings-v2-base-zh` | 768 | 8192 | 655MB | 中英双语 |
| `intfloat/multilingual-e5-large` | 1024 | 512 | 2.2GB | 100+ 语言 |

> `CODEXLENS_EMBED_DIM` 必须与模型输出维度匹配，不匹配会导致索引错误。

#### 代码搜索推荐

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| **性价比最高** | `nomic-ai/nomic-embed-text-v1.5-Q` | 768d，8192 tokens，仅 133MB |
| **代码专精** | `jinaai/jina-embeddings-v2-base-code` | 针对 30+ 编程语言训练 |
| **轻量+长上下文** | `jinaai/jina-embeddings-v2-small-en` | 512d，8192 tokens，122MB |
| **最快** | `BAAI/bge-small-en-v1.5` | 384d，68MB，默认 |

#### 手动下载模型

如果 `codexlens-search download-model` 失败（例如网络受限），可以手动下载：

1. **找到 ONNX 仓库** — fastembed 会重映射模型名。实际 HuggingFace 仓库：

   | 模型名 | 实际 HF 仓库 |
   |--------|-------------|
   | `BAAI/bge-small-en-v1.5` | `qdrant/bge-small-en-v1.5-onnx-q` |
   | `nomic-ai/nomic-embed-text-v1.5-Q` | `nomic-ai/nomic-embed-text-v1.5` |
   | `jinaai/jina-embeddings-v2-base-code` | `jinaai/jina-embeddings-v2-base-code` |

   运行 `codexlens-search list-models --json` 查看缓存路径。

2. **从 HuggingFace 下载**：

   ```bash
   # 使用 git（需要 git-lfs）
   git lfs install
   git clone https://huggingface.co/qdrant/bge-small-en-v1.5-onnx-q

   # 或使用 huggingface-cli
   pip install huggingface-hub
   huggingface-cli download qdrant/bge-small-en-v1.5-onnx-q --local-dir ./model-files

   # 国内可用 hf-mirror.com
   HF_ENDPOINT=https://hf-mirror.com huggingface-cli download qdrant/bge-small-en-v1.5-onnx-q --local-dir ./model-files
   ```

3. **放入缓存目录** — 缓存遵循 HuggingFace Hub 布局：

   ```
   <cache_dir>/
     models--<org>--<model>/
       snapshots/
         <commit_hash>/
           model_optimized.onnx   （或 model.onnx）
           tokenizer.json
           config.json
           special_tokens_map.json
           tokenizer_config.json
   ```

   默认缓存位置：
   - **Windows**：`%LOCALAPPDATA%\fastembed_cache` 或 `%TEMP%\fastembed_cache`
   - **Linux/macOS**：`/tmp/fastembed_cache`
   - **自定义**：设置 `CODEXLENS_MODEL_CACHE_DIR`

   以 `BAAI/bge-small-en-v1.5` 为例：
   ```bash
   mkdir -p /tmp/fastembed_cache/models--qdrant--bge-small-en-v1.5-onnx-q/snapshots/main/
   cp model-files/*.onnx model-files/*.json \
      /tmp/fastembed_cache/models--qdrant--bge-small-en-v1.5-onnx-q/snapshots/main/
   ```

4. **验证** — 运行 `codexlens-search list-models`，确认状态显示为 `●`。

#### 国内镜像

```json
"CODEXLENS_HF_MIRROR": "https://hf-mirror.com"
```

#### 自定义模型缓存

```json
"CODEXLENS_MODEL_CACHE_DIR": "/path/to/cache"
```

## GPU

**Windows**：`pip install codexlens-search[directml]` — 支持任何 DirectX 12 GPU（NVIDIA/AMD/Intel），无需 CUDA。即使不安装 `[directml]`，服务器也会在首次启动时自动安装。

**Linux**：`pip install codexlens-search[cuda]` 添加 CUDA 支持（需要 CUDA + cuDNN）。

自动检测优先级：CUDA > DirectML > CPU
- **嵌入** — ONNX runtime 选择最佳可用 GPU provider，比 CPU 快 ~12 倍
- **FAISS** — 索引自动传输到 GPU 0（仅 CUDA）

强制指定设备：`CODEXLENS_DEVICE=directml` / `cuda` / `cpu`

## CLI

```bash
codexlens-search --db-path .codexlens sync --root ./src
codexlens-search --db-path .codexlens search -q "auth handler" -k 10
codexlens-search --db-path .codexlens status
codexlens-search list-models
codexlens-search download-models
codexlens-search download-model BAAI/bge-base-en-v1.5
codexlens-search delete-model BAAI/bge-small-en-v1.5
```

## 环境变量

### 本地模型

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CODEXLENS_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | 本地 fastembed 模型名 |
| `CODEXLENS_EMBED_DIM` | `384` | 向量维度（必须与模型匹配） |
| `CODEXLENS_MODEL_CACHE_DIR` | fastembed 默认 | 模型下载缓存目录 |
| `CODEXLENS_HF_MIRROR` | | HuggingFace 镜像（如 `https://hf-mirror.com`） |

### 嵌入 API（覆盖本地模型）

| 变量 | 说明 |
|------|------|
| `CODEXLENS_EMBED_API_URL` | API 基础 URL（如 `https://api.openai.com/v1`） |
| `CODEXLENS_EMBED_API_KEY` | API 密钥 |
| `CODEXLENS_EMBED_API_MODEL` | 模型名（如 `text-embedding-3-small`） |
| `CODEXLENS_EMBED_API_ENDPOINTS` | 多端点：`url\|key\|model,...` |

### 重排序器

| 变量 | 说明 |
|------|------|
| `CODEXLENS_RERANKER_API_URL` | 重排序 API 基础 URL |
| `CODEXLENS_RERANKER_API_KEY` | API 密钥 |
| `CODEXLENS_RERANKER_API_MODEL` | 模型名 |

### 功能开关

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CODEXLENS_AST_CHUNKING` | `true` | AST 分块 + 符号提取 |
| `CODEXLENS_GITIGNORE_FILTERING` | `true` | 递归 `.gitignore` 过滤 |
| `CODEXLENS_EXPANSION_ENABLED` | `true` | 两跳查询扩展（自然语言查询） |
| `CODEXLENS_DEVICE` | `auto` | `auto` / `cuda` / `directml` / `cpu` |
| `CODEXLENS_AUTO_WATCH` | `false` | 索引后自动启动文件监控 |

### MCP 工具默认值

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CODEXLENS_TOP_K` | `10` | 搜索结果数量上限 |
| `CODEXLENS_FIND_MAX_RESULTS` | `100` | find_files 结果上限 |

### 调优参数

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CODEXLENS_BINARY_TOP_K` | `200` | 二值粗筛候选数 |
| `CODEXLENS_ANN_TOP_K` | `50` | ANN 精排候选数 |
| `CODEXLENS_FTS_TOP_K` | `50` | 每种 FTS 方法结果数 |
| `CODEXLENS_FUSION_K` | `60` | RRF 融合 k 参数 |
| `CODEXLENS_RERANKER_TOP_K` | `20` | 送入重排序的结果数 |
| `CODEXLENS_EMBED_BATCH_SIZE` | `32` | 每批 API 嵌入文本数 |
| `CODEXLENS_EMBED_MAX_TOKENS` | `8192` | 每段文本最大 token 数（0=不限） |
| `CODEXLENS_INDEX_WORKERS` | `2` | 并行索引工作线程数 |
| `CODEXLENS_MAX_FILE_SIZE` | `1000000` | 最大文件大小（字节） |

## 架构

```
查询 -> [QueryExpander] -> 扩展后查询（仅自然语言查询）
          |-> [Embedder] -> 查询向量
          |     |-> [FAISS Binary] -> 候选集（Hamming 距离）
          |     |     +-> [USearch/FAISS HNSW] -> 精排结果（cosine）
          |     +-> [FTS 精确 + 模糊] -> 文本匹配
          |-> [GraphSearcher] -> 符号邻居（以向量/FTS 结果为种子）
          +-> [ripgrep] -> 正则匹配
               +-> [RRF 融合] -> 合并排序
                     +-> [Reranker] -> 最终 top-k
```

## 开发

```bash
git clone https://github.com/catlog22/codexlens-search.git
cd codexlens-search
pip install -e ".[dev,all]"
pytest
```

## 许可证

MIT
