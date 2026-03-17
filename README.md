# codexlens-search

Lightweight semantic code search engine with 2-stage vector search, full-text search, and Reciprocal Rank Fusion.

## Overview

codexlens-search provides fast, accurate code search through a multi-stage retrieval pipeline:

1. **Binary coarse search** - Hamming-distance filtering narrows candidates quickly
2. **ANN fine search** - HNSW or FAISS refines the candidate set with float vectors
3. **Full-text search** - SQLite FTS5 handles exact and fuzzy keyword matching
4. **RRF fusion** - Reciprocal Rank Fusion merges vector and text results
5. **Reranking** - Optional cross-encoder or API-based reranker for final ordering

The core library has **zero required dependencies**. Install optional extras to enable semantic search, GPU acceleration, or FAISS backends.

## Installation

```bash
# Core only (FTS search, no vector search)
pip install codexlens-search

# With semantic search (recommended)
pip install codexlens-search[semantic]

# Semantic search + GPU acceleration
pip install codexlens-search[semantic-gpu]

# With FAISS backend (CPU)
pip install codexlens-search[faiss-cpu]

# With API-based reranker
pip install codexlens-search[reranker-api]

# Everything (semantic + GPU + FAISS + reranker)
pip install codexlens-search[semantic-gpu,faiss-gpu,reranker-api]
```

## Quick Start

```python
from codexlens_search import Config, IndexingPipeline, SearchPipeline
from codexlens_search.core import create_ann_index, create_binary_index
from codexlens_search.embed.local import FastEmbedEmbedder
from codexlens_search.rerank.local import LocalReranker
from codexlens_search.search.fts import FTSEngine

# 1. Configure
config = Config(embed_model="BAAI/bge-small-en-v1.5", embed_dim=384)

# 2. Create components
embedder = FastEmbedEmbedder(config)
binary_store = create_binary_index(config, db_path="index/binary.db")
ann_index = create_ann_index(config, index_path="index/ann.bin")
fts = FTSEngine("index/fts.db")
reranker = LocalReranker()

# 3. Index files
indexer = IndexingPipeline(embedder, binary_store, ann_index, fts, config)
stats = indexer.index_directory("./src")
print(f"Indexed {stats.files_processed} files, {stats.chunks_created} chunks")

# 4. Search
pipeline = SearchPipeline(embedder, binary_store, ann_index, reranker, fts, config)
results = pipeline.search("authentication handler", top_k=10)
for r in results:
    print(f"  {r.path} (score={r.score:.3f})")
```

## Extras

| Extra | Dependencies | Description |
|-------|-------------|-------------|
| `semantic` | hnswlib, numpy, fastembed | Vector search with local embeddings |
| `gpu` | onnxruntime-gpu | GPU-accelerated embedding inference |
| `semantic-gpu` | semantic + gpu combined | Vector search with GPU acceleration |
| `faiss-cpu` | faiss-cpu | FAISS ANN backend (CPU) |
| `faiss-gpu` | faiss-gpu | FAISS ANN backend (GPU) |
| `embed-api` | httpx | Remote embedding API client (OpenAI-compatible) |
| `reranker-api` | httpx | Remote reranker API client |
| `mcp` | mcp[cli], semantic, embed-api, reranker-api | MCP server for Claude Code |
| `dev` | pytest, pytest-cov | Development and testing |

## MCP Server (Claude Code Integration)

codexlens-search ships a built-in MCP server that exposes semantic search tools to Claude Code.

### Install

```bash
pip install codexlens-search[mcp]
# or with uv
uv pip install codexlens-search[mcp]
```

### Tools Exposed

| Tool | Description |
|------|-------------|
| `search_code` | Semantic search with hybrid fusion + reranking |
| `index_project` | Build or rebuild the search index |
| `index_status` | Show index statistics |
| `index_update` | Incremental sync (only changed files) |
| `find_files` | Glob file discovery |
| `list_models` | List models with cache status |
| `download_models` | Download local fastembed models |

### .mcp.json Configuration

Add to your project's `.mcp.json` (or `~/.claude/.mcp.json` for global):

#### API Embedding + API Reranker (recommended)

```json
{
  "mcpServers": {
    "codexlens": {
      "command": "codexlens-mcp",
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

#### Multi-Endpoint Load Balancing

Round-robin across multiple API keys/providers for higher throughput:

```json
{
  "mcpServers": {
    "codexlens": {
      "command": "codexlens-mcp",
      "env": {
        "CODEXLENS_EMBED_API_ENDPOINTS": "https://api1.example.com/v1|sk-key1|text-embedding-3-small,https://api2.example.com/v1|sk-key2|text-embedding-3-small",
        "CODEXLENS_EMBED_DIM": "1536",
        "CODEXLENS_RERANKER_API_URL": "https://api.jina.ai/v1",
        "CODEXLENS_RERANKER_API_KEY": "${JINA_API_KEY}",
        "CODEXLENS_RERANKER_API_MODEL": "jina-reranker-v2-base-multilingual"
      }
    }
  }
}
```

Format: `url1|key1|model1,url2|key2|model2,...`

#### Local Models (No API Required)

Uses fastembed for embedding and reranking — runs entirely offline:

```bash
# Pre-download models first
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

#### With uvx (No Pre-install)

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

### Environment Variables Reference

#### Embedding

| Variable | Description | Example |
|----------|-------------|---------|
| `CODEXLENS_EMBED_API_URL` | OpenAI-compatible embedding API base URL | `https://api.openai.com/v1` |
| `CODEXLENS_EMBED_API_KEY` | API key for embedding endpoint | `sk-xxx` |
| `CODEXLENS_EMBED_API_MODEL` | Embedding model name | `text-embedding-3-small` |
| `CODEXLENS_EMBED_API_ENDPOINTS` | Multi-endpoint config: `url\|key\|model,...` | See above |
| `CODEXLENS_EMBED_DIM` | Embedding vector dimension | `1536` |
| `CODEXLENS_EMBED_BATCH_SIZE` | Texts per embedding batch | `64` |
| `CODEXLENS_EMBED_API_CONCURRENCY` | Parallel API calls | `4` |
| `CODEXLENS_EMBED_API_MAX_TOKENS` | Max tokens per batch | `8192` |

#### Reranker

| Variable | Description | Example |
|----------|-------------|---------|
| `CODEXLENS_RERANKER_API_URL` | Reranker API base URL (Jina, Cohere, etc.) | `https://api.jina.ai/v1` |
| `CODEXLENS_RERANKER_API_KEY` | API key for reranker | `jina-xxx` |
| `CODEXLENS_RERANKER_API_MODEL` | Reranker model name | `jina-reranker-v2-base-multilingual` |
| `CODEXLENS_RERANKER_TOP_K` | Results to rerank | `20` |
| `CODEXLENS_RERANKER_BATCH_SIZE` | Reranker batch size | `32` |

#### Search Tuning

| Variable | Description | Default |
|----------|-------------|---------|
| `CODEXLENS_BINARY_TOP_K` | Binary coarse search candidates | `200` |
| `CODEXLENS_ANN_TOP_K` | ANN fine search candidates | `50` |
| `CODEXLENS_FTS_TOP_K` | FTS results per method | `50` |
| `CODEXLENS_FUSION_K` | RRF fusion k parameter | `60` |

#### Indexing

| Variable | Description | Default |
|----------|-------------|---------|
| `CODEXLENS_CODE_AWARE_CHUNKING` | Enable code-aware chunking | `true` |
| `CODEXLENS_INDEX_WORKERS` | Parallel indexing workers | `2` |
| `CODEXLENS_MAX_FILE_SIZE` | Max file size in bytes | `1000000` |
| `CODEXLENS_HNSW_EF` | HNSW ef parameter | `150` |
| `CODEXLENS_HNSW_M` | HNSW M parameter | `32` |

## CLI

```bash
# Index a project
codexlens-search --db-path .codexlens sync --root ./src --glob "**/*.py"

# Search
codexlens-search --db-path .codexlens search -q "authentication handler" -k 10

# Index status
codexlens-search --db-path .codexlens status

# Model management
codexlens-search list-models
codexlens-search download-models
codexlens-search download-model BAAI/bge-base-en-v1.5
codexlens-search delete-model BAAI/bge-base-en-v1.5
```

## Architecture

```
Query
  |
  v
[Embedder] --> query vector
  |
  +---> [BinaryStore.coarse_search] --> candidate IDs (Hamming distance)
  |         |
  |         v
  +---> [ANNIndex.fine_search] ------> ranked IDs (cosine/L2)
  |         |
  |         v  (intersect)
  |     vector_results
  |
  +---> [FTSEngine.exact_search] ----> exact text matches
  +---> [FTSEngine.fuzzy_search] ----> fuzzy text matches
  |
  v
[RRF Fusion] --> merged ranking (adaptive weights by query intent)
  |
  v
[Reranker] --> final top-k results
```

### Key Design Decisions

- **2-stage vector search**: Binary coarse search (fast Hamming distance on binarized vectors) filters candidates before the more expensive ANN search. This keeps memory usage low and search fast even on large corpora.
- **Parallel retrieval**: Vector search and FTS run concurrently via ThreadPoolExecutor.
- **Adaptive fusion weights**: Query intent detection adjusts RRF weights between vector and text signals.
- **Backend abstraction**: ANN index supports both hnswlib and FAISS backends via a factory function.
- **Zero core dependencies**: The base package requires only Python 3.10+. All heavy dependencies are optional.

## Configuration

The `Config` dataclass controls all pipeline parameters:

```python
from codexlens_search import Config

config = Config(
    embed_model="BAAI/bge-small-en-v1.5",  # embedding model name
    embed_dim=384,                           # embedding dimension
    embed_batch_size=64,                     # batch size for embedding
    ann_backend="auto",                      # 'auto', 'faiss', 'hnswlib'
    binary_top_k=200,                        # binary coarse search candidates
    ann_top_k=50,                            # ANN fine search candidates
    fts_top_k=50,                            # FTS results per method
    device="auto",                           # 'auto', 'cuda', 'cpu'
)
```

## Development

```bash
git clone https://github.com/nicepkg/codexlens-search.git
cd codexlens-search
pip install -e ".[dev,semantic]"
pytest
```

## License

MIT
