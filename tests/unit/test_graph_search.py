"""Unit tests for search/graph.py — GraphSearcher."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from codexlens_search.search.graph import GraphSearcher, _KIND_WEIGHT, _DIR_WEIGHT


def _make_fts_mock(
    symbols: dict[str, list[dict]] | None = None,
    refs_from: dict[str, list[dict]] | None = None,
    refs_to: dict[str, list[dict]] | None = None,
    symbols_by_chunk: dict[int, list[dict]] | None = None,
) -> MagicMock:
    """Create a mock FTSEngine with symbol/ref lookup methods."""
    fts = MagicMock()
    symbols = symbols or {}
    refs_from = refs_from or {}
    refs_to = refs_to or {}
    symbols_by_chunk = symbols_by_chunk or {}

    fts.get_symbols_by_name.side_effect = lambda name: symbols.get(name, [])
    fts.get_refs_from.side_effect = lambda name: refs_from.get(name, [])
    fts.get_refs_to.side_effect = lambda name: refs_to.get(name, [])
    fts.get_symbols_by_chunk.side_effect = lambda cid: symbols_by_chunk.get(cid, [])
    return fts


class TestScoreEdge:
    """Test _score_edge computation."""

    def test_import_backward_distance_1(self) -> None:
        fts = _make_fts_mock()
        gs = GraphSearcher(fts)
        score = gs._score_edge("import", "backward", 1)
        assert score == _KIND_WEIGHT["import"] * _DIR_WEIGHT["backward"] * 1.0
        assert score == 1.3  # 1.0 * 1.3 * 1.0

    def test_call_forward_distance_1(self) -> None:
        fts = _make_fts_mock()
        gs = GraphSearcher(fts)
        score = gs._score_edge("call", "forward", 1)
        assert score == pytest.approx(1.5 * 0.6 * 1.0)

    def test_unknown_kind_uses_default(self) -> None:
        fts = _make_fts_mock()
        gs = GraphSearcher(fts)
        score = gs._score_edge("unknown_kind", "backward", 1)
        assert score == pytest.approx(0.3 * 1.3 * 1.0)

    def test_distance_decay(self) -> None:
        fts = _make_fts_mock()
        gs = GraphSearcher(fts)
        score_d1 = gs._score_edge("import", "backward", 1)
        score_d2 = gs._score_edge("import", "backward", 2)
        assert score_d1 == 2 * score_d2


class TestFindSeedSymbols:
    """Test _find_seed_symbols."""

    def test_exact_match(self) -> None:
        fts = _make_fts_mock(symbols={
            "authenticate": [{"name": "authenticate", "id": 1, "chunk_id": 10}]
        })
        gs = GraphSearcher(fts)
        seeds = gs._find_seed_symbols("authenticate")
        assert len(seeds) == 1
        assert seeds[0]["name"] == "authenticate"

    def test_multi_word_tokenizes(self) -> None:
        fts = _make_fts_mock(symbols={
            "user": [{"name": "user", "id": 2, "chunk_id": 20}],
            "auth": [{"name": "auth", "id": 3, "chunk_id": 30}],
        })
        gs = GraphSearcher(fts)
        seeds = gs._find_seed_symbols("user auth")
        assert len(seeds) == 2

    def test_single_char_token_skipped(self) -> None:
        fts = _make_fts_mock(symbols={
            "x": [{"name": "x", "id": 1, "chunk_id": 10}],
        })
        gs = GraphSearcher(fts)
        # "a x" -> token "a" is 1 char, skipped; "x" is also 1 char, skipped
        seeds = gs._find_seed_symbols("a x")
        assert len(seeds) == 0

    def test_no_match_returns_empty(self) -> None:
        fts = _make_fts_mock()
        gs = GraphSearcher(fts)
        seeds = gs._find_seed_symbols("nonexistent")
        assert seeds == []


class TestResolveRefChunk:
    """Test _resolve_ref_chunk."""

    def test_forward_with_to_symbol_id(self) -> None:
        fts = _make_fts_mock(symbols={
            "target_fn": [
                {"name": "target_fn", "id": 5, "chunk_id": 50},
                {"name": "target_fn", "id": 6, "chunk_id": 60},
            ]
        })
        gs = GraphSearcher(fts)
        ref = {"to_name": "target_fn", "to_symbol_id": 6, "ref_kind": "call"}
        chunk = gs._resolve_ref_chunk(ref, direction="forward")
        assert chunk == 60

    def test_forward_fallback_to_first_symbol(self) -> None:
        fts = _make_fts_mock(symbols={
            "target_fn": [{"name": "target_fn", "id": 5, "chunk_id": 50}]
        })
        gs = GraphSearcher(fts)
        ref = {"to_name": "target_fn", "to_symbol_id": 999, "ref_kind": "call"}
        chunk = gs._resolve_ref_chunk(ref, direction="forward")
        assert chunk == 50

    def test_backward_with_from_symbol_id(self) -> None:
        fts = _make_fts_mock(symbols={
            "caller": [{"name": "caller", "id": 10, "chunk_id": 100}]
        })
        gs = GraphSearcher(fts)
        ref = {"from_name": "caller", "from_symbol_id": 10, "ref_kind": "import"}
        chunk = gs._resolve_ref_chunk(ref, direction="backward")
        assert chunk == 100

    def test_unresolvable_returns_none(self) -> None:
        fts = _make_fts_mock()
        gs = GraphSearcher(fts)
        ref = {"to_name": "missing", "ref_kind": "call"}
        chunk = gs._resolve_ref_chunk(ref, direction="forward")
        assert chunk is None


class TestSearch:
    """Test full GraphSearcher.search flow."""

    def test_no_seeds_returns_empty(self) -> None:
        fts = _make_fts_mock()
        gs = GraphSearcher(fts)
        results = gs.search("nonexistent")
        assert results == []

    def test_seed_chunk_gets_baseline_score(self) -> None:
        fts = _make_fts_mock(
            symbols={"fn": [{"name": "fn", "id": 1, "chunk_id": 10}]},
            refs_from={},
            refs_to={},
        )
        gs = GraphSearcher(fts)
        results = gs.search("fn")
        assert len(results) == 1
        assert results[0][0] == 10
        assert results[0][1] == pytest.approx(1.0)

    def test_forward_refs_add_score(self) -> None:
        fts = _make_fts_mock(
            symbols={
                "fn": [{"name": "fn", "id": 1, "chunk_id": 10}],
                "helper": [{"name": "helper", "id": 2, "chunk_id": 20}],
            },
            refs_from={"fn": [
                {"to_name": "helper", "to_symbol_id": 2, "ref_kind": "call"},
            ]},
        )
        gs = GraphSearcher(fts)
        results = gs.search("fn")
        result_dict = dict(results)
        assert 10 in result_dict  # seed
        assert 20 in result_dict  # forward ref target

    def test_backward_refs_add_score(self) -> None:
        fts = _make_fts_mock(
            symbols={
                "fn": [{"name": "fn", "id": 1, "chunk_id": 10}],
                "caller": [{"name": "caller", "id": 3, "chunk_id": 30}],
            },
            refs_to={"fn": [
                {"from_name": "caller", "from_symbol_id": 3, "ref_kind": "import"},
            ]},
        )
        gs = GraphSearcher(fts)
        results = gs.search("fn")
        result_dict = dict(results)
        assert 30 in result_dict  # backward ref source

    def test_results_sorted_descending(self) -> None:
        fts = _make_fts_mock(
            symbols={"fn": [{"name": "fn", "id": 1, "chunk_id": 10}]},
            refs_from={},
            refs_to={},
        )
        gs = GraphSearcher(fts)
        results = gs.search("fn")
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_output(self) -> None:
        syms = {"fn": [{"name": "fn", "id": 1, "chunk_id": 10}]}
        refs = {"fn": [
            {"to_name": f"t{i}", "to_symbol_id": i + 10, "ref_kind": "call"}
            for i in range(10)
        ]}
        target_syms = {
            f"t{i}": [{"name": f"t{i}", "id": i + 10, "chunk_id": 100 + i}]
            for i in range(10)
        }
        all_syms = {**syms, **target_syms}
        fts = _make_fts_mock(symbols=all_syms, refs_from=refs)
        gs = GraphSearcher(fts)
        results = gs.search("fn", top_k=3)
        assert len(results) <= 3

    def test_self_ref_not_duplicated(self) -> None:
        """A symbol referencing itself should not create duplicate scores."""
        fts = _make_fts_mock(
            symbols={"fn": [{"name": "fn", "id": 1, "chunk_id": 10}]},
            refs_from={"fn": [
                {"to_name": "fn", "to_symbol_id": 1, "ref_kind": "call"},
            ]},
        )
        gs = GraphSearcher(fts)
        results = gs.search("fn")
        # chunk_id 10 should appear once (self-ref filtered by != sym_chunk_id)
        chunk_ids = [cid for cid, _ in results]
        assert chunk_ids.count(10) == 1


class TestExpandOneHop:
    """Test BFS expansion."""

    def test_expand_adds_neighbors(self) -> None:
        fts = _make_fts_mock(
            symbols={
                "fn": [{"name": "fn", "id": 1, "chunk_id": 10}],
                "helper": [{"name": "helper", "id": 2, "chunk_id": 20}],
                "deep": [{"name": "deep", "id": 3, "chunk_id": 30}],
            },
            refs_from={
                "fn": [{"to_name": "helper", "to_symbol_id": 2, "ref_kind": "call"}],
                "helper": [{"to_name": "deep", "to_symbol_id": 3, "ref_kind": "call"}],
            },
            symbols_by_chunk={
                10: [{"name": "fn", "id": 1, "chunk_id": 10}],
                20: [{"name": "helper", "id": 2, "chunk_id": 20}],
            },
        )
        gs = GraphSearcher(fts, expand_hops=1)
        results = gs.search("fn")
        result_dict = dict(results)
        # Should reach chunk 30 via 1-hop expansion from chunk 20
        assert 30 in result_dict

    def test_no_expansion_when_hops_zero(self) -> None:
        fts = _make_fts_mock(
            symbols={
                "fn": [{"name": "fn", "id": 1, "chunk_id": 10}],
                "helper": [{"name": "helper", "id": 2, "chunk_id": 20}],
                "deep": [{"name": "deep", "id": 3, "chunk_id": 30}],
            },
            refs_from={
                "fn": [{"to_name": "helper", "to_symbol_id": 2, "ref_kind": "call"}],
                "helper": [{"to_name": "deep", "to_symbol_id": 3, "ref_kind": "call"}],
            },
            symbols_by_chunk={
                10: [{"name": "fn"}],
                20: [{"name": "helper"}],
            },
        )
        gs = GraphSearcher(fts, expand_hops=0)
        results = gs.search("fn")
        result_dict = dict(results)
        # Without expansion, chunk 30 should NOT be reachable
        assert 30 not in result_dict
