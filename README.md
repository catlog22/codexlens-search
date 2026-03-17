# codexlens-search

Semantic code search engine with MCP server for Claude Code.

2-stage vector search + FTS + RRF fusion + reranking — install once, configure API keys, ready to use.

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
        "CODEXLENS_EMBED_DIM": "1536"
      }
    }
  }
}
```

That's it. Claude Code will auto-discover the tools: `index_project` → `search_code`.

## Install

```bash
# Standard install (includes vector search + API clients)
pip install codexlens-search

# With MCP server for Claude Code
pip install codexlens-search[mcp]
```

Optional extras for advanced use:

| Extra | Description |
|-------|-------------|
| `mcp` | MCP server (`codexlens-mcp` command) |
| `gpu` | GPU-accelerated embedding (onnxruntime-gpu) |
| `faiss-cpu` | FAISS ANN backend |
| `watcher` | File watcher for auto-indexing |

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_code` | Semantic search with hybrid fusion + reranking |
| `index_project` | Build or rebuild the search index |
| `index_status` | Show index statistics |
| `index_update` | Incremental sync (only changed files) |
| `find_files` | Glob file discovery |
| `list_models` | List models with cache status |
| `download_models` | Download local fastembed models |

## MCP Configuration Examples

### API Embedding Only (simplest)

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
        "CODEXLENS_EMBED_DIM": "1536"
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
      "args": ["--from", "codexlens-search[mcp]", "codexlens-mcp"],
      "env": {
        "CODEXLENS_EMBED_API_URL": "https://api.openai.com/v1",
        "CODEXLENS_EMBED_API_KEY": "${OPENAI_API_KEY}",
        "CODEXLENS_EMBED_API_MODEL": "text-embedding-3-small",
        "CODEXLENS_EMBED_DIM": "1536",
        "CODEXLENS_RERANKER_API_URL": "https://api.jina.ai/v1",
        "CODEXLENS_RERANKER_API_KEY": "${JINA_API_KEY}",
        "CODEXLENS_RERANKER_API_MODEL": "jina-reranker-v2-base-multilingual"
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
pip install codexlens-search[mcp]
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

### Pre-installed (no uvx)

```json
{
  "mcpServers": {
    "codexlens": {
      "command": "codexlens-mcp",
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

### Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEXLENS_BINARY_TOP_K` | `200` | Binary coarse search candidates |
| `CODEXLENS_ANN_TOP_K` | `50` | ANN fine search candidates |
| `CODEXLENS_FTS_TOP_K` | `50` | FTS results per method |
| `CODEXLENS_FUSION_K` | `60` | RRF fusion k parameter |
| `CODEXLENS_RERANKER_TOP_K` | `20` | Results to rerank |
| `CODEXLENS_INDEX_WORKERS` | `2` | Parallel indexing workers |
| `CODEXLENS_MAX_FILE_SIZE` | `1000000` | Max file size in bytes |

## Architecture

```
Query → [Embedder] → query vector
         ├→ [BinaryStore] → candidates (Hamming)
         │     └→ [ANNIndex] → ranked IDs (cosine)
         ├→ [FTS exact] → exact matches
         └→ [FTS fuzzy] → fuzzy matches
              └→ [RRF Fusion] → merged ranking
                    └→ [Reranker] → final top-k
```

## Development

```bash
git clone https://github.com/catlog22/codexlens-search.git
cd codexlens-search
pip install -e ".[dev]"
pytest
```

## License

MIT
