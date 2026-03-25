"""Phase 1 unit tests: symbol boost + file aggregation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from codexlens_search.config import Config
from codexlens_search.search.fts import FTSEngine
from codexlens_search.search.pipeline import SearchPipeline, SearchResult


def _make_pipeline(fts: FTSEngine, cfg: Config | None = None) -> SearchPipeline:
    cfg = cfg or Config.small()

    embedder = MagicMock()
    embedder.embed_single.return_value = np.random.randn(cfg.embed_dim).astype(np.float32)

    binary_store = MagicMock()
    ann_index = MagicMock()

    reranker = MagicMock()
    reranker.score_pairs.side_effect = lambda q, contents: [1.0 for _ in contents]

    return SearchPipeline(
        embedder=embedder,
        binary_store=binary_store,
        ann_index=ann_index,
        reranker=reranker,
        fts=fts,
        config=cfg,
    )


class TestSymbolBoost:
    def test_thorough_includes_symbol_source_for_code_symbol_query(self) -> None:
        fts = FTSEngine(":memory:")
        fts.add_documents([
            (1, "a.py", "alpha"),
            (2, "b.py", "beta"),
        ])
        fts.add_symbols([
            (2, "MyClass", "class", 1, 10, "", "", "python"),
        ])

        cfg = Config.small()
        cfg.symbol_search_enabled = True
        pipeline = _make_pipeline(fts, cfg)

        # Avoid unrelated sources; keep symbol-only fusion.
        pipeline._vector_search = MagicMock(return_value=[])
        pipeline._fts_search = MagicMock(return_value=([], []))

        def _fake_rrf(results, weights=None, k=60):
            assert "symbol" in results
            assert results["symbol"][0][0] == 2
            assert weights is not None and weights.get("symbol", 0.0) > 0.0
            return [(2, 1.0)]

        with patch("codexlens_search.search.pipeline.reciprocal_rank_fusion", side_effect=_fake_rrf):
            out = pipeline.search("MyClass", quality="thorough", top_k=5)

        assert out and out[0].id == 2

    def test_symbol_boost_gated_on_code_symbol_intent(self) -> None:
        fts = FTSEngine(":memory:")
        fts.add_documents([
            (1, "a.py", "alpha"),
            (2, "b.py", "beta"),
        ])
        fts.add_symbols([
            (2, "MyClass", "class", 1, 10, "", "", "python"),
        ])

        cfg = Config.small()
        cfg.symbol_search_enabled = True
        pipeline = _make_pipeline(fts, cfg)

        pipeline._vector_search = MagicMock(return_value=[(1, 0.1)])
        pipeline._fts_search = MagicMock(return_value=([], []))
        pipeline._symbol_search = MagicMock(return_value=[(2, 1.0)])

        def _fake_rrf(results, weights=None, k=60):
            assert "symbol" not in results
            return [(1, 1.0)]

        with patch("codexlens_search.search.pipeline.reciprocal_rank_fusion", side_effect=_fake_rrf):
            out = pipeline.search("how do I use MyClass", quality="thorough", top_k=5)

        pipeline._symbol_search.assert_not_called()
        assert out and out[0].id == 1


class TestFileAggregation:
    def test_search_files_groups_by_path_and_uses_max_score(self) -> None:
        fts = FTSEngine(":memory:")
        fts.add_documents([
            (1, "a.py", "alpha"),
            (2, "a.py", "alpha2"),
            (3, "b.py", "beta"),
            (4, "c.py", "gamma"),
        ])
        pipeline = _make_pipeline(fts)

        chunk_results = [
            SearchResult(id=1, path="a.py", score=0.90, snippet="a1", line=1, end_line=2),
            SearchResult(id=2, path="a.py", score=0.95, snippet="a2", line=5, end_line=6),
            SearchResult(id=3, path="b.py", score=0.80, snippet="b1", line=1, end_line=1),
            SearchResult(id=4, path="c.py", score=0.70, snippet="c1", line=1, end_line=1),
        ]
        pipeline.search = MagicMock(return_value=chunk_results)  # type: ignore[method-assign]

        files = pipeline.search_files("q", top_k=2)
        pipeline.search.assert_called_once()
        assert pipeline.search.call_args.kwargs["top_k"] == 10

        assert [f.path for f in files] == ["a.py", "b.py"]
        assert files[0].best_chunk_id == 2
        assert files[0].chunk_ids == (1, 2)
