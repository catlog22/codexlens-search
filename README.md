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

All features are included by default — MCP server, AST parsing, FAISS backend, file watcher, gitignore filtering. GPU acceleration is auto-detected when available.

## Install

```bash
# Standard (batteries included)
uv pip install codexlens-search

# GPU acceleration (CUDA)
uv pip install codexlens-search[gpu]
```

Default install includes:

- **MCP server** — `codexlens-mcp` command
- **AST parsing** — tree-sitter symbol extraction + graph search (on by default)
- **FAISS** — ANN + binary index backend
- **File watcher** — watchdog auto-indexing
- **Gitignore filtering** — recursive `.gitignore` support (on by default)

`[gpu]` adds `onnxruntime-gpu` + `faiss-gpu`. When GPU is detected, embedding and FAISS indexing automatically use CUDA — no config needed.

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

```bash
uv pip install codexlens-search[gpu]
```

Auto-detection handles everything:
- **Embedding** — ONNX runtime selects CUDA provider
- **FAISS** — index auto-transfers to GPU 0

Force CPU: `CODEXLENS_DEVICE=cpu`

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
         │     └→ [FAISS HNSW] → ranked IDs (cosine)
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
