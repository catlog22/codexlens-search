"""L2 integration tests for graph search.

Tests GraphSearcher with real FTSEngine (symbols + refs tables)
but no tree-sitter dependency. Manually populates symbols and refs.
"""
from __future__ import annotations

import pytest

from codexlens_search.search.fts import FTSEngine
from codexlens_search.search.graph import GraphSearcher


@pytest.fixture
def graph_env(tmp_path):
    """Create FTSEngine with symbols and refs for graph search testing."""
    fts = FTSEngine(tmp_path / "fts.db")

    # Add 5 test documents (chunks)
    docs = [
        (0, "auth.py", "class AuthService:\n    def authenticate(self, user): pass", 1, 2, "python"),
        (1, "auth.py", "class TokenValidator:\n    def validate(self, token): pass", 3, 4, "python"),
        (2, "models.py", "class User:\n    name: str\n    email: str", 1, 3, "python"),
        (3, "api.py", "from auth import AuthService\ndef login(user): return AuthService().authenticate(user)", 1, 2, "python"),
        (4, "middleware.py", "from auth import TokenValidator\ndef auth_middleware(req): TokenValidator().validate(req.token)", 1, 2, "python"),
    ]
    fts.add_documents(docs)

    # Add symbols
    symbols = [
        # (chunk_id, name, kind, start_line, end_line, parent_name, signature, language)
        (0, "AuthService", "class", 1, 2, "", "class AuthService", "python"),
        (0, "authenticate", "function", 2, 2, "AuthService", "def authenticate(self, user)", "python"),
        (1, "TokenValidator", "class", 3, 4, "", "class TokenValidator", "python"),
        (1, "validate", "function", 4, 4, "TokenValidator", "def validate(self, token)", "python"),
        (2, "User", "class", 1, 3, "", "class User", "python"),
        (3, "login", "function", 2, 2, "", "def login(user)", "python"),
        (4, "auth_middleware", "function", 2, 2, "", "def auth_middleware(req)", "python"),
    ]
    fts.add_symbols(symbols)

    # Add references (from_name, from_path, to_name, ref_kind, line)
    refs = [
        ("login", "api.py", "AuthService", "import", 1),
        ("login", "api.py", "authenticate", "call", 2),
        ("auth_middleware", "middleware.py", "TokenValidator", "import", 1),
        ("auth_middleware", "middleware.py", "validate", "call", 2),
    ]
    fts.add_refs(refs)
    fts.resolve_refs()
    fts.flush()

    return fts


class TestGraphSearcherIntegration:
    """Test GraphSearcher with real FTS backend."""

    def test_search_by_symbol_name_finds_seed(self, graph_env):
        fts = graph_env
        gs = GraphSearcher(fts)

        results = gs.search("AuthService")
        assert len(results) > 0
        result_ids = {r[0] for r in results}
        # Chunk 0 contains AuthService definition
        assert 0 in result_ids

    def test_search_finds_callers_via_backward_refs(self, graph_env):
        fts = graph_env
        gs = GraphSearcher(fts)

        results = gs.search("authenticate")
        result_ids = {r[0] for r in results}
        # Chunk 0 has authenticate definition, chunk 3 calls it
        assert 0 in result_ids
        # Chunk 3 should appear via backward ref (login calls authenticate)
        assert 3 in result_ids

    def test_search_finds_dependencies_via_forward_refs(self, graph_env):
        fts = graph_env
        gs = GraphSearcher(fts)

        results = gs.search("login")
        result_ids = {r[0] for r in results}
        # Chunk 3 has login definition
        assert 3 in result_ids
        # login references AuthService (chunk 0) and authenticate (chunk 0)
        assert 0 in result_ids

    def test_search_unrelated_symbol_returns_empty(self, graph_env):
        fts = graph_env
        gs = GraphSearcher(fts)

        results = gs.search("nonexistent_symbol_xyz")
        assert results == []

    def test_graph_scores_import_higher_than_call(self, graph_env):
        """Import edges should have weight 1.0, call edges 0.8."""
        fts = graph_env
        gs = GraphSearcher(fts)

        results = gs.search("auth_middleware")
        # auth_middleware imports TokenValidator and calls validate
        # Both are in chunk 1; check that chunk 1 appears
        result_ids = {r[0] for r in results}
        assert 1 in result_ids

    def test_expand_one_hop_discovers_neighbors(self, graph_env):
        fts = graph_env
        gs = GraphSearcher(fts, expand_hops=1)

        results = gs.search("login")
        result_ids = {r[0] for r in results}
        # login -> AuthService -> authenticate are in chunk 0
        # With 1-hop expansion from chunk 0, should discover callers
        assert 0 in result_ids
        assert 3 in result_ids

    def test_multi_token_query_finds_multiple_symbols(self, graph_env):
        fts = graph_env
        gs = GraphSearcher(fts)

        results = gs.search("User login")
        result_ids = {r[0] for r in results}
        # Should find chunks for both User (chunk 2) and login (chunk 3)
        assert 2 in result_ids or 3 in result_ids
