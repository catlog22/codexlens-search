# codexlens-search

Semantic code search engine with MCP server for Claude Code.

Hybrid search: vector + FTS + AST graph + ripgrep regex — with RRF fusion and reranking.

[中文文档](README_zh.md)

## Quick Start

```bash
pip install codexlens-search[all]
```

Add to your project `.mcp.json`:

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

That's it. Claude Code will auto-discover the tools: `index_project` -> `Search` -> `locate`.

## Install

Choose the install that matches your platform:

```bash
# Minimal — CPU inference (fastembed bundles onnxruntime CPU)
pip install codexlens-search

# Windows GPU — DirectML, any DirectX 12 GPU (NVIDIA/AMD/Intel)
pip install codexlens-search[directml]

# Linux/Windows NVIDIA GPU — CUDA (requires CUDA + cuDNN)
pip install codexlens-search[cuda]

# Auto-select — DirectML on Windows, CPU elsewhere
pip install codexlens-search[all]
```

### Platform Recommendations

| Platform | Recommended | Command |
|----------|-------------|---------|
| **Windows + any GPU** | `[directml]` | `pip install codexlens-search[directml]` |
| **Windows CPU only** | base | `pip install codexlens-search` |
| **Linux + NVIDIA GPU** | `[cuda]` | `pip install codexlens-search[cuda]` |
| **Linux CPU / AMD GPU** | base | `pip install codexlens-search` |
| **macOS (Apple Silicon)** | base | `pip install codexlens-search` |
| **Don't know / CI** | `[all]` | `pip install codexlens-search[all]` |

> **Note**: On Windows, if you install the base package without `[directml]`, the MCP server will auto-detect the missing GPU runtime and install `onnxruntime-directml` on first launch. GPU takes effect from the second start.

### What's Included

All install variants include:

- **MCP server** — `codexlens-mcp` command
- **AST parsing** — tree-sitter symbol extraction + graph search
- **USearch** — high-performance HNSW ANN backend (default)
- **FAISS** — ANN + binary index backend (Hamming coarse search)
- **File watcher** — watchdog auto-indexing
- **Gitignore filtering** — recursive `.gitignore` support
- **Focused search** — when no index exists, greps relevant files, indexes only those (~10s), then runs semantic search — no waiting for full index build

### ANN Backend Selection

Three backends for approximate nearest neighbor search, auto-selected in order:

| Backend | Install | Best for |
|---------|---------|----------|
| `usearch` (default) | Included | Cross-platform, fastest CPU HNSW |
| `faiss` | Included | GPU acceleration, binary Hamming search |
| `hnswlib` | Included | Lightweight fallback |

Override with `CODEXLENS_ANN_BACKEND`:

```bash
CODEXLENS_ANN_BACKEND=faiss    # use FAISS (GPU when available)
CODEXLENS_ANN_BACKEND=usearch  # use USearch (default)
CODEXLENS_ANN_BACKEND=hnswlib  # use hnswlib
CODEXLENS_ANN_BACKEND=auto     # auto-select (usearch > faiss > hnswlib)
```

## MCP Tools

### Search

Hybrid code search combining semantic vector, FTS, AST graph, and ripgrep regex.

| Mode | Description | Requires |
|------|-------------|----------|
| `auto` (default) | Semantic + regex parallel. No index? Focused grep-index-search in ~10s. | |
| `symbol` | Find definitions by exact/fuzzy name match | Index |
| `refs` | Find cross-references — incoming and outgoing edges | Index |
| `regex` | Ripgrep regex on live files | rg |

Parameters: `project_path`, `query`, `mode`, `scope` (restricts auto/regex to subdirectory)

Results capped by `CODEXLENS_TOP_K` env var (default 10).

#### Cold Start Search

When no index exists, `auto` mode uses a focused search pipeline instead of waiting for a full index build:

1. **Expand query** — split camelCase/snake_case into search terms
2. **Grep files** — `rg --count` finds top 50 relevant files, ranked by match count
3. **Index** — embed only those 50 files (~8-10s with GPU)
4. **Search** — semantic vector search on the fresh index
5. **Background** — full index builds asynchronously for next queries

This gives semantic results in ~10s vs ~100s for a full index build.

### locate

LLM-driven code localization — finds ALL files that need changes for a bug fix or feature request.

Uses an iterative agent loop: search → read files → extract symbols → follow imports → discover dependencies. Much more thorough than keyword search for multi-file changes.

Parameters: `project_path`, `query`, `top_k` (default 10), `max_iterations` (default 5)

Requires LLM API configuration (any OpenAI-compatible API — GLM, DeepSeek, Qwen, OpenAI, etc.):

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
        "CODEXLENS_EMBED_DIM": "1536",
        "CODEXLENS_AGENT_LLM_API_KEY": "${GLM_API_KEY}",
        "CODEXLENS_AGENT_LLM_MODEL": "glm-5-turbo",
        "CODEXLENS_AGENT_LLM_API_BASE": "https://open.bigmodel.cn/api/paas/v4/"
      }
    }
  }
}
```

Falls back to plain semantic search if no LLM API key is configured.

#### How it works

1. **Initial Search** — searches codebase with keywords from the query
2. **Read Top Files** — reads top 3-5 files to understand code structure
3. **Symbol Extraction** — identifies function/class names, searches for exact symbol names
4. **Dependency Discovery** — follows imports and entity graph edges to find related files
5. **Import-Aware Expansion** — automatically resolves project-local imports from top-ranked files, interleaving structurally-related files even if they share no keywords with the query

### index_project

Build, update, or inspect the search index.

| Action | Description |
|--------|-------------|
| `sync` (default) | Incremental — only changed files |
| `rebuild` | Full re-index from scratch |
| `status` | Index statistics (files, chunks, symbols, refs) |

Parameters: `project_path`, `action`, `scope`

### find_files

Glob-based file discovery. Parameters: `project_path`, `pattern` (default `**/*`)

Max results controlled by `CODEXLENS_FIND_MAX_RESULTS` env var (default 100).

### watch_project

Manage file watcher for automatic re-indexing on file changes.

Parameters: `project_path`, `action` (`start` / `stop` / `status`)

## AST Features

Enabled by default. Disable with `CODEXLENS_AST_CHUNKING=false`.

- **Smart chunking** — splits at symbol boundaries instead of fixed-size windows
- **Symbol extraction** — 12 kinds: function, class, method, module, variable, constant, interface, type_alias, enum, struct, trait, property
- **Cross-references** — import, call, inherit, type_ref edges
- **Graph search** — seeded from vector/FTS results, BFS expansion with adaptive weights
- **Query expansion** — two-hop symbol vocabulary expansion for natural language queries

Languages: Python, JavaScript, TypeScript, Go, Java, Rust, C, C++, Ruby, PHP, Scala, Kotlin, Swift, C#, Bash, Lua, Haskell, Elixir, Erlang.

## Configuration Examples

### Reranker (best quality)

Add reranker API on top of the Quick Start config:

```json
"CODEXLENS_RERANKER_API_URL": "https://api.jina.ai/v1",
"CODEXLENS_RERANKER_API_KEY": "${JINA_API_KEY}",
"CODEXLENS_RERANKER_API_MODEL": "jina-reranker-v2-base-multilingual"
```

### Multi-Endpoint Load Balancing

```json
"CODEXLENS_EMBED_API_ENDPOINTS": "https://api1.example.com/v1|sk-key1|model,https://api2.example.com/v1|sk-key2|model",
"CODEXLENS_EMBED_DIM": "1536"
```

Format: `url|key|model,url|key|model,...` — replaces single-endpoint `EMBED_API_URL/KEY/MODEL`.

### Local Models (Offline)

No API needed — `fastembed` runs the model locally via ONNX runtime.

```bash
# List available models
codexlens-search list-models

# Pre-download models (optional — auto-downloads on first use)
codexlens-search download-models

# Download a specific model
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

Default local model: `BAAI/bge-small-en-v1.5` (384d, 512 tokens). To use a different model:

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

#### Available Local Models

**General**

| Model | Dim | Tokens | Size | Notes |
|-------|-----|--------|------|-------|
| `BAAI/bge-small-en-v1.5` | 384 | 512 | 68MB | Default, fastest |
| `BAAI/bge-base-en-v1.5` | 768 | 512 | 215MB | Better quality |
| `BAAI/bge-large-en-v1.5` | 1024 | 512 | 1.2GB | Best English quality |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 256 | 92MB | Lightweight general |
| `snowflake/snowflake-arctic-embed-xs` | 384 | 512 | 92MB | Compact, good quality |
| `snowflake/snowflake-arctic-embed-s` | 384 | 512 | 133MB | Light, better than xs |

**Code / Long Context**

| Model | Dim | Tokens | Size | Notes |
|-------|-----|--------|------|-------|
| `jinaai/jina-embeddings-v2-base-code` | 768 | 8192 | 655MB | Code-specialized, 30+ programming languages |
| `nomic-ai/nomic-embed-text-v1.5-Q` | 768 | 8192 | 133MB | Quantized, best value for code |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | 8192 | 532MB | Long context, code + text |
| `jinaai/jina-embeddings-v2-small-en` | 512 | 8192 | 122MB | Long context, lightweight |

**Chinese / Multilingual**

| Model | Dim | Tokens | Size | Notes |
|-------|-----|--------|------|-------|
| `BAAI/bge-small-zh-v1.5` | 512 | 512 | 92MB | Chinese, fast |
| `BAAI/bge-large-zh-v1.5` | 1024 | 512 | 1.2GB | Chinese, best quality |
| `jinaai/jina-embeddings-v2-base-zh` | 768 | 8192 | 655MB | Chinese-English bilingual |
| `intfloat/multilingual-e5-large` | 1024 | 512 | 2.2GB | 100+ languages |

> `CODEXLENS_EMBED_DIM` must match the model's output dimension. Mismatched dim will cause indexing errors.

#### Recommended for Code Search

| Use case | Model | Why |
|----------|-------|-----|
| **Best value** | `nomic-ai/nomic-embed-text-v1.5-Q` | 768d, 8192 tokens, only 133MB |
| **Code-specialized** | `jinaai/jina-embeddings-v2-base-code` | Trained on 30+ programming languages |
| **Lightweight + long context** | `jinaai/jina-embeddings-v2-small-en` | 512d, 8192 tokens, 122MB |
| **Fastest** | `BAAI/bge-small-en-v1.5` | 384d, 68MB, default |

#### Manual Model Download

If `codexlens-search download-model` fails (e.g. behind a firewall), you can download models manually:

1. **Find the ONNX repo** — fastembed remaps model names. Check the actual HuggingFace repo:

   | Model name | Actual HF repo |
   |------------|----------------|
   | `BAAI/bge-small-en-v1.5` | `qdrant/bge-small-en-v1.5-onnx-q` |
   | `nomic-ai/nomic-embed-text-v1.5-Q` | `nomic-ai/nomic-embed-text-v1.5` |
   | `jinaai/jina-embeddings-v2-base-code` | `jinaai/jina-embeddings-v2-base-code` |

   Run `codexlens-search list-models --json` to check cache paths.

2. **Download from HuggingFace** — clone or download the repo:

   ```bash
   # Using git (requires git-lfs)
   git lfs install
   git clone https://huggingface.co/qdrant/bge-small-en-v1.5-onnx-q

   # Or download via huggingface-cli
   pip install huggingface-hub
   huggingface-cli download qdrant/bge-small-en-v1.5-onnx-q --local-dir ./model-files
   ```

3. **Place in cache directory** — the cache follows HuggingFace Hub layout:

   ```
   <cache_dir>/
     models--<org>--<model>/
       snapshots/
         <commit_hash>/
           model_optimized.onnx   (or model.onnx)
           tokenizer.json
           config.json
           special_tokens_map.json
           tokenizer_config.json
   ```

   Default cache locations:
   - **Windows**: `%LOCALAPPDATA%\fastembed_cache` or `%TEMP%\fastembed_cache`
   - **Linux/macOS**: `/tmp/fastembed_cache`
   - **Custom**: set `CODEXLENS_MODEL_CACHE_DIR`

   Example for `BAAI/bge-small-en-v1.5`:
   ```bash
   mkdir -p /tmp/fastembed_cache/models--qdrant--bge-small-en-v1.5-onnx-q/snapshots/main/
   cp model-files/*.onnx model-files/*.json \
      /tmp/fastembed_cache/models--qdrant--bge-small-en-v1.5-onnx-q/snapshots/main/
   ```

4. **Verify** — run `codexlens-search list-models` and check the `●` status.

#### China Mirror

```json
"CODEXLENS_HF_MIRROR": "https://hf-mirror.com"
```

#### Custom Model Cache

```json
"CODEXLENS_MODEL_CACHE_DIR": "/path/to/cache"
```

## GPU

**Windows**: `pip install codexlens-search[directml]` — works with any DirectX 12 GPU (NVIDIA/AMD/Intel). No CUDA needed. Even without `[directml]`, the server auto-installs it on first launch.

**Linux**: `pip install codexlens-search[cuda]` adds CUDA support (requires CUDA + cuDNN).

Auto-detection priority: CUDA > DirectML > CPU
- **Embedding** — ONNX runtime selects best available GPU provider, ~12x faster than CPU
- **FAISS** — index auto-transfers to GPU 0 (CUDA only)

Force specific device: `CODEXLENS_DEVICE=directml` / `cuda` / `cpu`

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

## Environment Variables

### Local Model

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEXLENS_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Local fastembed model name |
| `CODEXLENS_EMBED_DIM` | `384` | Vector dimension (must match model) |
| `CODEXLENS_MODEL_CACHE_DIR` | fastembed default | Model download cache directory |
| `CODEXLENS_HF_MIRROR` | | HuggingFace mirror (e.g. `https://hf-mirror.com`) |

### Embedding API (overrides local model)

| Variable | Description |
|----------|-------------|
| `CODEXLENS_EMBED_API_URL` | API base URL (e.g. `https://api.openai.com/v1`) |
| `CODEXLENS_EMBED_API_KEY` | API key |
| `CODEXLENS_EMBED_API_MODEL` | Model name (e.g. `text-embedding-3-small`) |
| `CODEXLENS_EMBED_API_ENDPOINTS` | Multi-endpoint: `url\|key\|model,...` |

### Reranker

| Variable | Description |
|----------|-------------|
| `CODEXLENS_RERANKER_API_URL` | Reranker API base URL |
| `CODEXLENS_RERANKER_API_KEY` | API key |
| `CODEXLENS_RERANKER_API_MODEL` | Model name |

### LLM Agent (locate tool)

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEXLENS_AGENT_LLM_API_KEY` | | API key for LLM (required for locate) |
| `CODEXLENS_AGENT_LLM_MODEL` | `glm-5-turbo` | Model name (any OpenAI-compatible) |
| `CODEXLENS_AGENT_LLM_API_BASE` | `https://open.bigmodel.cn/api/paas/v4/` | API base URL |
| `CODEXLENS_AGENT_MAX_ITERATIONS` | `5` | Max agent tool-call iterations |

### Features

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEXLENS_AST_CHUNKING` | `true` | AST chunking + symbol extraction |
| `CODEXLENS_GITIGNORE_FILTERING` | `true` | Recursive `.gitignore` filtering |
| `CODEXLENS_EXPANSION_ENABLED` | `true` | Two-hop query expansion for NL queries |
| `CODEXLENS_DEVICE` | `auto` | `auto` / `cuda` / `directml` / `cpu` |
| `CODEXLENS_AUTO_WATCH` | `false` | Auto-start file watcher after indexing |

### MCP Tool Defaults

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEXLENS_TOP_K` | `10` | Search result limit |
| `CODEXLENS_FIND_MAX_RESULTS` | `100` | find_files result limit |

### Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEXLENS_BINARY_TOP_K` | `200` | Binary coarse search candidates |
| `CODEXLENS_ANN_TOP_K` | `50` | ANN fine search candidates |
| `CODEXLENS_FTS_TOP_K` | `50` | FTS results per method |
| `CODEXLENS_FUSION_K` | `60` | RRF fusion k parameter |
| `CODEXLENS_RERANKER_TOP_K` | `20` | Results to rerank |
| `CODEXLENS_EMBED_BATCH_SIZE` | `32` | Texts per API batch |
| `CODEXLENS_EMBED_MAX_TOKENS` | `8192` | Max tokens per text (0=no limit) |
| `CODEXLENS_INDEX_WORKERS` | `2` | Parallel indexing workers |
| `CODEXLENS_MAX_FILE_SIZE` | `1000000` | Max file size in bytes |

## Architecture

```
Query -> [QueryExpander] -> expanded query (NL queries only)
          |-> [Embedder] -> query vector
          |     |-> [FAISS Binary] -> candidates (Hamming)
          |     |     +-> [USearch/FAISS HNSW] -> ranked IDs (cosine)
          |     +-> [FTS exact + fuzzy] -> text matches
          |-> [GraphSearcher] -> symbol neighbors (seeded from vector/FTS)
          +-> [ripgrep] -> regex matches
               +-> [RRF Fusion] -> merged ranking
                     +-> [Reranker] -> final top-k
```

## Development

```bash
git clone https://github.com/catlog22/codexlens-search.git
cd codexlens-search
pip install -e ".[dev,all]"
pytest
```

## License

MIT
