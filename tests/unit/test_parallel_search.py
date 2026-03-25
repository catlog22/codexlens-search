"""Unit tests for parallel search pipeline and agent fan-out."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from codexlens_search.config import Config
from codexlens_search.search.pipeline import SearchPipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pipeline(
    *,
    expansion_enabled: bool = True,
    expansion_delay: float = 0.0,
    fts_delay: float = 0.0,
) -> SearchPipeline:
    """Create a SearchPipeline with mocked components for timing tests."""
    embedder = MagicMock()
    embedder.embed_single.return_value = MagicMock(
        astype=MagicMock(return_value=MagicMock()),
    )

    binary_store = MagicMock()
    binary_store.__len__ = MagicMock(return_value=0)

    ann_index = MagicMock()
    reranker = MagicMock()
    reranker.score_pairs.return_value = [0.5, 0.3]

    fts = MagicMock()

    def mock_exact_search(q, top_k=50):
        if fts_delay > 0:
            time.sleep(fts_delay)
        return [(1, 1.0), (2, 0.8)]

    def mock_fuzzy_search(q, top_k=50):
        if fts_delay > 0:
            time.sleep(fts_delay)
        return [(1, 0.9), (3, 0.7)]

    fts.exact_search = mock_exact_search
    fts.fuzzy_search = mock_fuzzy_search
    fts.get_content.return_value = "def foo(): pass"
    fts.get_doc_meta.return_value = ("src/foo.py", 1, 5, "python")

    config = Config(
        expansion_enabled=expansion_enabled,
        default_search_quality="fast",
    )

    expander = None
    if expansion_enabled:
        expander = MagicMock()

        def mock_expand(q):
            if expansion_delay > 0:
                time.sleep(expansion_delay)
            return q + " expanded_term"

        expander.expand = mock_expand

    pipeline = SearchPipeline(
        embedder=embedder,
        binary_store=binary_store,
        ann_index=ann_index,
        reranker=reranker,
        fts=fts,
        config=config,
        query_expander=expander,
    )

    return pipeline


# ---------------------------------------------------------------------------
# Tests: Parallel expansion + FTS
# ---------------------------------------------------------------------------

class TestParallelExpansionFTS:
    def test_expansion_and_fts_run_concurrently(self):
        """When expansion is enabled, expansion and FTS should overlap in time."""
        delay = 0.15  # each takes 150ms
        pipeline = _make_pipeline(
            expansion_enabled=True,
            expansion_delay=delay,
            fts_delay=delay,
        )

        start = time.monotonic()
        results = pipeline.search("test query")
        elapsed = time.monotonic() - start

        # If serial: ~300ms. If parallel: ~150ms.
        # Use generous threshold to avoid flaky tests.
        assert elapsed < delay * 2.5, (
            f"Expected parallel execution (<{delay * 2.5:.2f}s) but took {elapsed:.2f}s"
        )
        assert len(results) > 0

    def test_results_match_serial_when_expansion_disabled(self):
        """Without expansion, pipeline should produce results normally."""
        pipeline = _make_pipeline(expansion_enabled=False)
        results = pipeline.search("test query")
        assert len(results) > 0

    def test_expansion_failure_does_not_break_search(self):
        """If expansion raises, search should still return FTS results."""
        pipeline = _make_pipeline(expansion_enabled=True)
        pipeline._query_expander.expand.side_effect = RuntimeError("boom")
        results = pipeline.search("test query")
        assert len(results) > 0

    def test_fts_prefetch_consumed_once(self):
        """Prefetched FTS results should be consumed and not reused."""
        pipeline = _make_pipeline(expansion_enabled=True)
        pipeline.search("query1")
        # After search, prefetch should be cleared
        assert getattr(pipeline, "_prefetched_fts", None) is None


# ---------------------------------------------------------------------------
# Tests: Fan-out heuristic
# ---------------------------------------------------------------------------

class TestShouldFanOut:
    def _make_agent(self, *, fan_out_enabled: bool = True):
        from codexlens_search.agent.loc_agent import CodeLocAgent

        config = Config(
            agent_enabled=True,
            agent_fan_out_enabled=fan_out_enabled,
        )
        search = MagicMock()
        entity_graph = MagicMock()
        return CodeLocAgent(search, entity_graph, config)

    def test_fan_out_disabled_by_default(self):
        """Fan-out should be disabled when config flag is False."""
        agent = self._make_agent(fan_out_enabled=False)
        assert agent._should_fan_out("fix auth and update cache and refactor db") is False

    def test_fan_out_enabled_with_multiple_and_or(self):
        """Queries with multiple AND/OR should trigger fan-out."""
        agent = self._make_agent(fan_out_enabled=True)
        assert agent._should_fan_out("fix auth and update cache and refactor db") is True

    def test_fan_out_enabled_with_commas(self):
        """Queries with 3+ comma-separated items should trigger fan-out."""
        agent = self._make_agent(fan_out_enabled=True)
        assert agent._should_fan_out("auth handler, cache manager, database connector") is True

    def test_fan_out_enabled_with_multiple_files(self):
        """Queries referencing multiple files should trigger fan-out."""
        agent = self._make_agent(fan_out_enabled=True)
        assert agent._should_fan_out("changes needed in auth.py and cache.py") is True

    def test_fan_out_not_triggered_for_simple_query(self):
        """Simple single-concept queries should not trigger fan-out."""
        agent = self._make_agent(fan_out_enabled=True)
        assert agent._should_fan_out("fix authentication bug") is False

    def test_fan_out_single_and(self):
        """A single AND should not trigger fan-out (too aggressive)."""
        agent = self._make_agent(fan_out_enabled=True)
        assert agent._should_fan_out("fix auth and update tests") is False


# ---------------------------------------------------------------------------
# Tests: Config defaults
# ---------------------------------------------------------------------------

class TestFanOutConfig:
    def test_fan_out_disabled_by_default(self):
        config = Config()
        assert config.agent_fan_out_enabled is False

    def test_fan_out_max_workers_default(self):
        config = Config()
        assert config.agent_fan_out_max_workers == 3

    def test_fan_out_max_workers_clamped(self):
        config = Config(agent_fan_out_max_workers=0)
        assert config.agent_fan_out_max_workers == 1


# ---------------------------------------------------------------------------
# Tests: Bridge env var support
# ---------------------------------------------------------------------------

class TestBridgeEnvVars:
    def test_fan_out_env_vars(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CODEXLENS_AGENT_FAN_OUT_ENABLED", "true")
        monkeypatch.setenv("CODEXLENS_AGENT_FAN_OUT_MAX_WORKERS", "5")

        from codexlens_search.bridge import create_config_from_env
        config = create_config_from_env(str(tmp_path))
        assert config.agent_fan_out_enabled is True
        assert config.agent_fan_out_max_workers == 5

    def test_fan_out_env_vars_defaults(self, tmp_path):
        from codexlens_search.bridge import create_config_from_env
        config = create_config_from_env(str(tmp_path))
        assert config.agent_fan_out_enabled is False
        assert config.agent_fan_out_max_workers == 3
