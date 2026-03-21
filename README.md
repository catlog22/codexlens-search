# codexlens-search

Semantic code search engine with MCP server for Claude Code.

Hybrid search: vector + FTS + AST graph + ripgrep regex — with RRF fusion and reranking.

## Quick Start

```bash
uv pip install codexlens-search
```

Add to your project `.mcp.json`:

```json
{
  "mcpServers": {
    "codexlens": {
      "command": "uvx",
      "args": ["--from", "codexlens-search", "codexlens-mcp"],
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

That's it. Claude Code will auto-discover the tools: `index_project` → `Search`.

All features are included by default — MCP server, AST parsing, USearch + FAISS backends, file watcher, gitignore filtering. GPU acceleration is auto-detected when available.

## Install

```bash
# One command — GPU auto-detected on Windows, no extra steps
uv pip install codexlens-search

# Linux with NVIDIA GPU (requires CUDA + cuDNN)
uv pip install codexlens-search[gpu]
```

Default install includes:

- **MCP server** — `codexlens-mcp` command
- **AST parsing** — tree-sitter symbol extraction + graph search (on by default)
- **USearch** — high-performance HNSW ANN backend (default, cross-platform)
- **FAISS** — ANN + binary index backend (Hamming coarse search)
- **File watcher** — watchdog auto-indexing
- **Gitignore filtering** — recursive `.gitignore` support (on by default)
- **GPU acceleration** — Windows auto-installs `onnxruntime-directml`, works with any DirectX 12 GPU (NVIDIA/AMD/Intel), no CUDA needed. GPU is auto-detected at runtime — no config needed

`[gpu]` adds `onnxruntime-gpu` + `faiss-gpu` for Linux CUDA setups.

### ANN Backend Selection

Three backends for approximate nearest neighbor search, auto-selected in order:

| Backend | Install | Best for |
|---------|---------|----------|
| `usearch` (default) | Included | Cross-platform, fastest CPU HNSW |
| `faiss` | Included | GPU acceleration, binary Hamming search |
| `hnswlib` | Included | Lightweight fallback |

Override with `CODEXLENS_ANN_BACKEND`:

```bash
# Force specific backend
CODEXLENS_ANN_BACKEND=faiss    # use FAISS (GPU when available)
CODEXLENS_ANN_BACKEND=usearch  # use USearch (default)
CODEXLENS_ANN_BACKEND=hnswlib  # use hnswlib
CODEXLENS_ANN_BACKEND=auto     # auto-select (usearch → faiss → hnswlib)
```

## MCP Tools

### Search

Hybrid code search combining semantic vector, FTS, AST graph, and ripgrep regex.

| Mode | Description | Requires |
|------|-------------|----------|
| `auto` (default) | Semantic + regex parallel. Auto-triggers background indexing if none exists. | |
| `symbol` | Find definitions by exact/fuzzy name match | Index |
| `refs` | Find cross-references — incoming and outgoing edges | Index |
| `regex` | Ripgrep regex on live files | rg |

Parameters: `project_path`, `query`, `mode`, `scope` (restricts auto/regex to subdirectory)

Results capped by `CODEXLENS_TOP_K` env var (default 10).

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
- **Graph search** — BFS expansion from matches, fused with adaptive weights

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

```bash
codexlens-search download-models
```

```json
{
  "mcpServers": {
    "codexlens": {
      "command": "codexlens-mcp",
      "env": {}
    }
  }
}
```

## GPU

**Windows**: GPU acceleration is included by default — `onnxruntime-directml` is auto-installed and works with any DirectX 12 GPU (NVIDIA/AMD/Intel). No CUDA, no extra install, no config.

**Linux**: `uv pip install codexlens-search[gpu]` adds CUDA support (requires CUDA + cuDNN).

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
```

## Environment Variables

### Embedding

| Variable | Description |
|----------|-------------|
| `CODEXLENS_EMBED_API_URL` | API base URL (e.g. `https://api.openai.com/v1`) |
| `CODEXLENS_EMBED_API_KEY` | API key |
| `CODEXLENS_EMBED_API_MODEL` | Model name (e.g. `text-embedding-3-small`) |
| `CODEXLENS_EMBED_API_ENDPOINTS` | Multi-endpoint: `url\|key\|model,...` |
| `CODEXLENS_EMBED_DIM` | Vector dimension (e.g. `1536`) |

### Reranker

| Variable | Description |
|----------|-------------|
| `CODEXLENS_RERANKER_API_URL` | Reranker API base URL |
| `CODEXLENS_RERANKER_API_KEY` | API key |
| `CODEXLENS_RERANKER_API_MODEL` | Model name |

### Features

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEXLENS_AST_CHUNKING` | `true` | AST chunking + symbol extraction |
| `CODEXLENS_GITIGNORE_FILTERING` | `true` | Recursive `.gitignore` filtering |
| `CODEXLENS_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
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
Query → [Embedder] → query vector
         ├→ [FAISS Binary] → candidates (Hamming)
         │     └→ [USearch/FAISS HNSW] → ranked IDs (cosine)
         ├→ [FTS exact + fuzzy] → text matches
         ├→ [GraphSearcher] → symbol neighbors (BFS)
         └→ [ripgrep] → regex matches
              └→ [RRF Fusion] → merged ranking
                    └→ [Reranker] → final top-k
```

## Development

```bash
git clone https://github.com/catlog22/codexlens-search.git
cd codexlens-search
uv pip install -e ".[dev]"
pytest
```

## License

MIT
