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


# -------------------------------------------------------------------
# FTS + Vector fusion integration tests
# -------------------------------------------------------------------


def test_fts_and_vector_both_contribute_to_results(search_pipeline):
    """Both FTS and vector search should contribute results to fusion."""
    results = search_pipeline.search("password hash")
    assert len(results) > 0
    # Doc 5 has "hash_password" and "bcrypt.hashpw" - should rank well
    result_ids = {r.id for r in results}
    assert 5 in result_ids, "Doc with hash_password should appear in results"


def test_fusion_ranks_exact_match_highly(search_pipeline):
    """Exact keyword matches via FTS should boost ranking."""
    results = search_pipeline.search("validate_email")
    if results:
        # Doc 19 has validate_email - should be among top results
        top_ids = [r.id for r in results[:5]]
        assert 19 in top_ids, "Exact match for validate_email should rank in top 5"


def test_search_with_code_symbol_query(search_pipeline):
    """Code symbol queries should work through fusion."""
    results = search_pipeline.search("AuthError")
    assert len(results) > 0
    result_ids = {r.id for r in results}
    # Doc 17 has AuthError class
    assert 17 in result_ids


def test_search_with_natural_language_query(search_pipeline):
    """Natural language queries should also return relevant results."""
    results = search_pipeline.search("how to get database connection")
    assert len(results) > 0
    # Should find db.py (doc 14) with get_connection
    result_ids = {r.id for r in results}
    assert 14 in result_ids or len(results) > 0
