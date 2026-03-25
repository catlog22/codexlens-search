"""Tests for parallel reranker batch scoring (P1: Parallel Rerank Batches)."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from codexlens_search.config import Config
from codexlens_search.rerank.api import APIReranker


def _make_config(concurrency: int = 1, max_tokens: int = 100) -> Config:
    return Config(
        reranker_api_url="https://api.example.com",
        reranker_api_key="test-key",
        reranker_api_model="test-model",
        reranker_api_max_tokens_per_batch=max_tokens,
        reranker_api_concurrency=concurrency,
    )


def _make_api_response(docs: list[tuple[int, str]], scores: list[float]) -> MagicMock:
    """Build a mock HTTP response for a batch of docs."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "results": [
            {"index": i, "relevance_score": s}
            for i, s in enumerate(scores)
        ]
    }
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_concurrency_1_matches_serial() -> None:
    """Verify concurrency=1 produces identical scores to serial behavior."""
    config = _make_config(concurrency=1, max_tokens=100)

    # 4 docs, 50 tokens each -> 2 per batch -> 2 batches
    docs = ["x" * 200] * 4

    batch0_resp = _make_api_response([], [0.9, 0.8])
    batch1_resp = _make_api_response([], [0.7, 0.6])

    with patch("httpx.Client") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.post.side_effect = [batch0_resp, batch1_resp]

        reranker = APIReranker(config)
        scores = reranker.score_pairs("test query", docs)

    assert len(scores) == 4
    assert scores[0] == pytest.approx(0.9)
    assert scores[1] == pytest.approx(0.8)
    assert scores[2] == pytest.approx(0.7)
    assert scores[3] == pytest.approx(0.6)


def test_concurrency_4_same_scores() -> None:
    """Verify concurrency=4 produces same scores as serial."""
    config = _make_config(concurrency=4, max_tokens=100)

    docs = ["x" * 200] * 4

    batch0_resp = _make_api_response([], [0.9, 0.8])
    batch1_resp = _make_api_response([], [0.7, 0.6])

    with patch("httpx.Client") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.post.side_effect = [batch0_resp, batch1_resp]

        reranker = APIReranker(config)
        scores = reranker.score_pairs("test query", docs)

    assert len(scores) == 4
    assert scores[0] == pytest.approx(0.9)
    assert scores[1] == pytest.approx(0.8)
    assert scores[2] == pytest.approx(0.7)
    assert scores[3] == pytest.approx(0.6)


def test_concurrency_4_parallel_wall_time() -> None:
    """Verify concurrency=4 with 4+ batches shows measurable wall-time reduction."""
    config = _make_config(concurrency=4, max_tokens=50)

    # 8 docs, 50 tokens each, max 50 tokens/batch -> 1 per batch -> 8 batches
    docs = ["y" * 200] * 8
    sleep_duration = 0.1

    def slow_post(url, json=None, **kwargs):
        time.sleep(sleep_duration)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.5}]
        }
        resp.raise_for_status = MagicMock()
        return resp

    with patch("httpx.Client") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.post.side_effect = slow_post

        reranker = APIReranker(config)

        start = time.monotonic()
        scores = reranker.score_pairs("test", docs)
        elapsed = time.monotonic() - start

    assert len(scores) == 8
    serial_time = sleep_duration * 8
    # Parallel with 4 workers: ~2 rounds -> ~0.2s, not ~0.8s
    # Use generous threshold (>20% reduction)
    assert elapsed < serial_time * 0.8, (
        f"Parallel took {elapsed:.3f}s, serial would be {serial_time:.3f}s "
        f"(expected >20% reduction)"
    )


def test_score_mapping_correctness_across_batches() -> None:
    """Verify scores are mapped to correct original document indices across batches."""
    config = _make_config(concurrency=4, max_tokens=100)

    # 6 docs, 50 tokens each -> 2 per batch -> 3 batches
    docs = ["doc" * 67] * 6  # 201 chars -> 50 tokens each

    # Each batch gets 2 docs; scores should map to original indices
    resp0 = MagicMock()
    resp0.status_code = 200
    resp0.json.return_value = {
        "results": [
            {"index": 0, "relevance_score": 0.95},
            {"index": 1, "relevance_score": 0.85},
        ]
    }
    resp0.raise_for_status = MagicMock()

    resp1 = MagicMock()
    resp1.status_code = 200
    resp1.json.return_value = {
        "results": [
            {"index": 0, "relevance_score": 0.75},
            {"index": 1, "relevance_score": 0.65},
        ]
    }
    resp1.raise_for_status = MagicMock()

    resp2 = MagicMock()
    resp2.status_code = 200
    resp2.json.return_value = {
        "results": [
            {"index": 0, "relevance_score": 0.55},
            {"index": 1, "relevance_score": 0.45},
        ]
    }
    resp2.raise_for_status = MagicMock()

    with patch("httpx.Client") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.post.side_effect = [resp0, resp1, resp2]

        reranker = APIReranker(config)
        scores = reranker.score_pairs("query", docs)

    assert len(scores) == 6
    # Original indices 0-5 should map correctly
    assert scores[0] == pytest.approx(0.95)
    assert scores[1] == pytest.approx(0.85)
    assert scores[2] == pytest.approx(0.75)
    assert scores[3] == pytest.approx(0.65)
    assert scores[4] == pytest.approx(0.55)
    assert scores[5] == pytest.approx(0.45)


def test_single_batch_no_thread_pool() -> None:
    """Verify single batch does not use ThreadPoolExecutor even with concurrency > 1."""
    config = _make_config(concurrency=4, max_tokens=10000)

    # All docs fit in one batch
    docs = ["short doc"] * 3

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "results": [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.8},
            {"index": 2, "relevance_score": 0.7},
        ]
    }
    resp.raise_for_status = MagicMock()

    with patch("httpx.Client") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.post.return_value = resp

        reranker = APIReranker(config)

        with patch(
            "codexlens_search.rerank.api.ThreadPoolExecutor"
        ) as mock_tpe:
            scores = reranker.score_pairs("query", docs)
            # Should not create ThreadPoolExecutor for single batch
            mock_tpe.assert_not_called()

    assert len(scores) == 3


def test_empty_documents_returns_empty() -> None:
    """Verify empty documents returns empty scores."""
    config = _make_config(concurrency=4)

    with patch("httpx.Client"):
        reranker = APIReranker(config)
        scores = reranker.score_pairs("query", [])

    assert scores == []


def test_config_reranker_concurrency_clamp() -> None:
    """Verify reranker_api_concurrency is clamped to >= 1."""
    cfg = Config(reranker_api_concurrency=0)
    assert cfg.reranker_api_concurrency == 1

    cfg = Config(reranker_api_concurrency=-3)
    assert cfg.reranker_api_concurrency == 1

    cfg = Config(reranker_api_concurrency=8)
    assert cfg.reranker_api_concurrency == 8
