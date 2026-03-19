# codexlens-search

Semantic code search engine with MCP server for Claude Code.

Hybrid search: vector + FTS + AST graph + ripgrep regex — with RRF fusion and reranking.

## Quick Start (Claude Code MCP)

Add to your project `.mcp.json`:

```json
{
  "mcpServers": {
    "codexlens": {
      "command": "uvx",
      "args": ["--from", "codexlens-search[mcp]", "codexlens-mcp"],
      "env": {
        "CODEXLENS_EMBED_API_URL": "https://api.openai.com/v1",
        "CODEXLENS_EMBED_API_KEY": "${OPENAI_API_KEY}",
        "CODEXLENS_EMBED_API_MODEL": "text-embedding-3-small",
        "CODEXLENS_EMBED_DIM": "1536",
        "CODEXLENS_AST_CHUNKING": "true"
      }
    }
  }
}
```

That's it. Claude Code will auto-discover the tools: `index_project` → `Search`.

## Install

```bash
# Standard install
uv pip install codexlens-search

# With MCP server
uv pip install codexlens-search[mcp]

# With AST parsing (symbol extraction, cross-references, graph search)
uv pip install codexlens-search[mcp,ast]
```

Optional extras:

| Extra | Description |
|-------|-------------|
| `mcp` | MCP server (`codexlens-mcp` command) |
| `ast` | tree-sitter AST parsing (symbol extraction, graph search) |
| `gpu` | GPU-accelerated embedding (onnxruntime-gpu) |
| `faiss-cpu` | FAISS ANN backend |
| `watcher` | File watcher for auto-indexing |
| `gitignore` | Recursive `.gitignore` filtering |

## MCP Tools

### Search

Unified code search with 4 modes:

| Mode | Description | Requires Index | Requires rg |
|------|-------------|:-:|:-:|
| `auto` (default) | Semantic + regex parallel, falls back to regex if no index | - | - |
| `symbol` | Find definitions by name (class, function, method) | ✓ | |
| `refs` | Find cross-references (imports, calls, inheritance) | ✓ | |
| `regex` | Ripgrep regex on live files | | ✓ |

Parameters:
- `project_path` — Absolute path to the project root
- `query` — Natural language, code symbol, or regex pattern
- `mode` — `auto` / `symbol` / `refs` / `regex`
- `top_k` — Max results (default 10)
- `scope` — Relative path to restrict search (e.g. `src/auth`)

**Auto mode behavior:**
- Has index + has rg → semantic and regex run in parallel, results merged with dedup
- Has index + no rg → semantic only
- No index + has rg → regex fallback
- No index + no rg → error with guidance

### index_project

Build, update, or inspect the search index.

| Action | Description |
|--------|-------------|
| `sync` (default) | Incremental update — only re-indexes changed files |
| `rebuild` | Full re-index from scratch |
| `status` | Show index statistics (files, chunks, symbols, refs) |

Parameters:
- `project_path` — Absolute path to the project root
- `action` — `sync` / `rebuild` / `status`
- `scope` — Relative directory to limit indexing
- `force` — Alias for `action="rebuild"`

### find_files

Glob-based file discovery.

- `project_path` — Absolute path to the project root
- `pattern` — Glob pattern (default `**/*`)
- `max_results` — Max file paths to return (default 100)

## AST Features

When `CODEXLENS_AST_CHUNKING=true` and `[ast]` extra is installed:

- **Smart chunking** — Splits code at symbol boundaries (functions, classes, methods) instead of fixed-size windows
- **Symbol extraction** — Indexes 12 symbol kinds: function, class, method, module, variable, constant, interface, type_alias, enum, struct, trait, property
- **Cross-references** — Extracts import, call, inherit, type_ref edges between symbols
- **Graph search** — BFS expansion from matched symbols, fused into hybrid results with adaptive weights

```bash
# Install AST support (tree-sitter 0.23+)
uv pip install codexlens-search[ast]

# Or individual grammar packages for Python 3.13+
pip install tree-sitter tree-sitter-python tree-sitter-javascript tree-sitter-typescript
```

Supported languages: Python, JavaScript, TypeScript, Go, Java, Rust, C, C++, Ruby, PHP, Scala, Kotlin, Swift, C#, Bash, Lua, Haskell, Elixir, Erlang.

## MCP Configuration Examples

### API Embedding + AST (recommended)

```json
{
  "mcpServers": {
    "codexlens": {
      "command": "uvx",
      "args": ["--from", "codexlens-search[mcp,ast]", "codexlens-mcp"],
      "env": {
        "CODEXLENS_EMBED_API_URL": "https://api.openai.com/v1",
        "CODEXLENS_EMBED_API_KEY": "${OPENAI_API_KEY}",
        "CODEXLENS_EMBED_API_MODEL": "text-embedding-3-small",
        "CODEXLENS_EMBED_DIM": "1536",
        "CODEXLENS_AST_CHUNKING": "true"
      }
    }
  }
}
```

### API Embedding + API Reranker (best quality)

```json
{
  "mcpServers": {
    "codexlens": {
      "command": "uvx",
      "args": ["--from", "codexlens-search[mcp,ast]", "codexlens-mcp"],
      "env": {
        "CODEXLENS_EMBED_API_URL": "https://api.openai.com/v1",
        "CODEXLENS_EMBED_API_KEY": "${OPENAI_API_KEY}",
        "CODEXLENS_EMBED_API_MODEL": "text-embedding-3-small",
        "CODEXLENS_EMBED_DIM": "1536",
        "CODEXLENS_RERANKER_API_URL": "https://api.jina.ai/v1",
        "CODEXLENS_RERANKER_API_KEY": "${JINA_API_KEY}",
        "CODEXLENS_RERANKER_API_MODEL": "jina-reranker-v2-base-multilingual",
        "CODEXLENS_AST_CHUNKING": "true"
      }
    }
  }
}
```

### Multi-Endpoint Load Balancing

```json
{
  "mcpServers": {
    "codexlens": {
      "command": "uvx",
      "args": ["--from", "codexlens-search[mcp]", "codexlens-mcp"],
      "env": {
        "CODEXLENS_EMBED_API_ENDPOINTS": "https://api1.example.com/v1|sk-key1|model,https://api2.example.com/v1|sk-key2|model",
        "CODEXLENS_EMBED_DIM": "1536"
      }
    }
  }
}
```

Format: `url|key|model,url|key|model,...`

### Local Models (Offline, No API)

```bash
uv pip install codexlens-search[mcp]
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

| Variable | Description | Example |
|----------|-------------|---------|
| `CODEXLENS_EMBED_API_URL` | Embedding API base URL | `https://api.openai.com/v1` |
| `CODEXLENS_EMBED_API_KEY` | API key | `sk-xxx` |
| `CODEXLENS_EMBED_API_MODEL` | Model name | `text-embedding-3-small` |
| `CODEXLENS_EMBED_API_ENDPOINTS` | Multi-endpoint: `url\|key\|model,...` | See above |
| `CODEXLENS_EMBED_DIM` | Vector dimension | `1536` |

### Reranker

| Variable | Description | Example |
|----------|-------------|---------|
| `CODEXLENS_RERANKER_API_URL` | Reranker API base URL | `https://api.jina.ai/v1` |
| `CODEXLENS_RERANKER_API_KEY` | API key | `jina-xxx` |
| `CODEXLENS_RERANKER_API_MODEL` | Model name | `jina-reranker-v2-base-multilingual` |

### AST & Filtering

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEXLENS_AST_CHUNKING` | `false` | Enable tree-sitter AST chunking + symbol extraction |
| `CODEXLENS_GITIGNORE_FILTERING` | `false` | Enable recursive `.gitignore` filtering |

### Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEXLENS_BINARY_TOP_K` | `200` | Binary coarse search candidates |
| `CODEXLENS_ANN_TOP_K` | `50` | ANN fine search candidates |
| `CODEXLENS_FTS_TOP_K` | `50` | FTS results per method |
| `CODEXLENS_FUSION_K` | `60` | RRF fusion k parameter |
| `CODEXLENS_RERANKER_TOP_K` | `20` | Results to rerank |
| `CODEXLENS_EMBED_BATCH_SIZE` | `32` | Max texts per API batch (auto-splits on 413) |
| `CODEXLENS_EMBED_MAX_TOKENS` | `8192` | Max tokens per text (truncate if exceeded, 0=no limit) |
| `CODEXLENS_INDEX_WORKERS` | `2` | Parallel indexing workers |
| `CODEXLENS_MAX_FILE_SIZE` | `1000000` | Max file size in bytes |

## Architecture

```
Query → [Embedder] → query vector
         ├→ [BinaryStore] → candidates (Hamming)
         │     └→ [ANNIndex] → ranked IDs (cosine)
         ├→ [FTS exact] → exact matches
         ├→ [FTS fuzzy] → fuzzy matches
         ├→ [GraphSearcher] → symbol neighbors (BFS)
         └→ [ripgrep] → regex matches (parallel)
              └→ [RRF Fusion] → merged ranking
                    └→ [Reranker] → final top-k
```

## Development

```bash
git clone https://github.com/catlog22/codexlens-search.git
cd codexlens-search
uv pip install -e ".[dev,ast]"
pytest
```

## License

MIT
