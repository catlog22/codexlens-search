"""Unit tests for search layer: FTSEngine, fusion, and SearchPipeline."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from codexlens_search.search.fts import FTSEngine
from codexlens_search.search.fusion import (
    DEFAULT_WEIGHTS,
    QueryIntent,
    detect_query_intent,
    get_adaptive_weights,
    reciprocal_rank_fusion,
)
from codexlens_search.search.pipeline import SearchPipeline, SearchResult
from codexlens_search.config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fts(docs: list[tuple[int, str, str]] | None = None) -> FTSEngine:
    """Create an in-memory FTSEngine and optionally add documents."""
    engine = FTSEngine(":memory:")
    if docs:
        engine.add_documents(docs)
    return engine


# ---------------------------------------------------------------------------
# FTSEngine tests
# ---------------------------------------------------------------------------

def test_fts_add_and_exact_search():
    docs = [
        (1, "a.py", "def authenticate user password login"),
        (2, "b.py", "connect to database with credentials"),
        (3, "c.py", "render template html response"),
    ]
    engine = make_fts(docs)
    results = engine.exact_search("authenticate", top_k=10)
    ids = [r[0] for r in results]
    assert 1 in ids, "doc 1 should match 'authenticate'"
    assert 2 not in ids or results[0][0] == 1  # doc 1 must rank higher


def test_fts_fuzzy_search_prefix():
    docs = [
        (10, "auth.py", "authentication token refresh"),
        (11, "db.py", "database connection pool"),
        (12, "ui.py", "render button click handler"),
    ]
    engine = make_fts(docs)
    # Prefix 'auth' should match 'authentication' in doc 10
    results = engine.fuzzy_search("auth", top_k=10)
    ids = [r[0] for r in results]
    assert 10 in ids, "prefix 'auth' should match doc 10 with 'authentication'"


# ---------------------------------------------------------------------------
# RRF fusion tests
# ---------------------------------------------------------------------------

def test_rrf_fusion_ordering():
    """When two sources agree on top-1, it should rank first in fused result."""
    source_a = [(1, 0.9), (2, 0.5), (3, 0.2)]
    source_b = [(1, 0.8), (3, 0.6), (2, 0.1)]
    fused = reciprocal_rank_fusion({"a": source_a, "b": source_b})
    assert fused[0][0] == 1, "doc 1 agreed top by both sources must rank first"


def test_rrf_equal_weight_default():
    """Calling with None weights should use DEFAULT_WEIGHTS shape (not crash)."""
    source_exact = [(5, 1.0), (6, 0.8)]
    source_vector = [(6, 0.9), (5, 0.7)]
    # Should not raise and should return results
    fused = reciprocal_rank_fusion(
        {"exact": source_exact, "vector": source_vector},
        weights=None,
    )
    assert len(fused) == 2
    ids = [r[0] for r in fused]
    assert 5 in ids and 6 in ids


# ---------------------------------------------------------------------------
# detect_query_intent tests
# ---------------------------------------------------------------------------

def test_detect_intent_code_symbol():
    assert detect_query_intent("def authenticate()") == QueryIntent.CODE_SYMBOL


def test_detect_intent_natural():
    assert detect_query_intent("how do I authenticate users") == QueryIntent.NATURAL_LANGUAGE


# ---------------------------------------------------------------------------
# SearchPipeline tests
# ---------------------------------------------------------------------------

def _make_pipeline(fts: FTSEngine, top_k: int = 5) -> SearchPipeline:
    """Build a SearchPipeline with mocked heavy components."""
    cfg = Config.small()
    cfg.reranker_top_k = top_k

    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * cfg.embed_dim]

    binary_store = MagicMock()
    binary_store.coarse_search.return_value = ([1, 2, 3], None)

    ann_index = MagicMock()
    ann_index.fine_search.return_value = ([1, 2, 3], [0.9, 0.8, 0.7])

    reranker = MagicMock()
    # Return a score for each content string passed
    reranker.score_pairs.side_effect = lambda q, contents: [0.9 - i * 0.1 for i in range(len(contents))]

    return SearchPipeline(
        embedder=embedder,
        binary_store=binary_store,
        ann_index=ann_index,
        reranker=reranker,
        fts=fts,
        config=cfg,
    )


def test_pipeline_search_returns_results():
    docs = [
        (1, "a.py", "test content alpha"),
        (2, "b.py", "test content beta"),
        (3, "c.py", "test content gamma"),
    ]
    fts = make_fts(docs)
    pipeline = _make_pipeline(fts)
    results = pipeline.search("test")
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)


def test_pipeline_top_k_limit():
    docs = [
        (1, "a.py", "hello world one"),
        (2, "b.py", "hello world two"),
        (3, "c.py", "hello world three"),
        (4, "d.py", "hello world four"),
        (5, "e.py", "hello world five"),
    ]
    fts = make_fts(docs)
    pipeline = _make_pipeline(fts, top_k=2)
    results = pipeline.search("hello", top_k=2)
    assert len(results) <= 2, "pipeline must respect top_k limit"
