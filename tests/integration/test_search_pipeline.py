"""Integration tests for SearchPipeline using real components and mock embedder/reranker."""
from __future__ import annotations


def test_vector_search_returns_results(search_pipeline):
    results = search_pipeline.search("authentication middleware")
    assert len(results) > 0
    assert all(isinstance(r.score, float) for r in results)


def test_exact_keyword_search(search_pipeline):
    results = search_pipeline.search("authenticate")
    assert len(results) > 0
    result_ids = {r.id for r in results}
    # Doc 0 and 10 both contain "authenticate"
    assert result_ids & {0, 10}, f"Expected doc 0 or 10 in results, got {result_ids}"


def test_pipeline_top_k_limit(search_pipeline):
    results = search_pipeline.search("user", top_k=5)
    assert len(results) <= 5


def test_search_result_fields_populated(search_pipeline):
    results = search_pipeline.search("password")
    assert len(results) > 0
    for r in results:
        assert r.id >= 0
        assert r.score >= 0
        assert isinstance(r.path, str)


def test_empty_query_handled(search_pipeline):
    results = search_pipeline.search("")
    assert isinstance(results, list)  # no exception


def test_different_queries_give_different_results(search_pipeline):
    r1 = search_pipeline.search("authenticate user")
    r2 = search_pipeline.search("cache redis")
    # Results should differ (different top IDs or scores), unless both are empty
    ids1 = [r.id for r in r1]
    ids2 = [r.id for r in r2]
    assert ids1 != ids2 or len(r1) == 0
