"""codexlens-search: Lightweight semantic code search engine.

Public API for consumers (e.g. codex-lens):

    from codexlens_search import SearchPipeline, IndexingPipeline, Config
    from codexlens_search.core import create_ann_index, create_binary_index
    from codexlens_search.embed.local import FastEmbedEmbedder
    from codexlens_search.rerank.api import APIReranker
"""
from codexlens_search.config import Config
from codexlens_search.indexing import IndexingPipeline, IndexStats
from codexlens_search.search.pipeline import SearchPipeline, SearchResult

__all__ = [
    "Config",
    "IndexingPipeline",
    "IndexStats",
    "SearchPipeline",
    "SearchResult",
]
