from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from codexlens_search.config import Config
from codexlens_search.rerank.base import BaseReranker
from codexlens_search.rerank.local import FastEmbedReranker
from codexlens_search.rerank.api import APIReranker


# ---------------------------------------------------------------------------
# BaseReranker
# ---------------------------------------------------------------------------

def test_base_reranker_is_abstract():
    with pytest.raises(TypeError):
        BaseReranker()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# FastEmbedReranker
# ---------------------------------------------------------------------------

def _make_rerank_result(index: int, score: float) -> object:
    obj = types.SimpleNamespace(index=index, score=score)
    return obj


def test_local_reranker_score_pairs_length():
    config = Config()
    reranker = FastEmbedReranker(config)

    mock_results = [
        _make_rerank_result(0, 0.9),
        _make_rerank_result(1, 0.5),
        _make_rerank_result(2, 0.1),
    ]

    mock_model = MagicMock()
    mock_model.rerank.return_value = iter(mock_results)
    reranker._model = mock_model

    docs = ["doc0", "doc1", "doc2"]
    scores = reranker.score_pairs("query", docs)

    assert len(scores) == 3


def test_local_reranker_preserves_order():
    config = Config()
    reranker = FastEmbedReranker(config)

    # rerank returns results in reverse order (index 2, 1, 0)
    mock_results = [
        _make_rerank_result(2, 0.1),
        _make_rerank_result(1, 0.5),
        _make_rerank_result(0, 0.9),
    ]

    mock_model = MagicMock()
    mock_model.rerank.return_value = iter(mock_results)
    reranker._model = mock_model

    docs = ["doc0", "doc1", "doc2"]
    scores = reranker.score_pairs("query", docs)

    assert scores[0] == pytest.approx(0.9)
    assert scores[1] == pytest.approx(0.5)
    assert scores[2] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# APIReranker
# ---------------------------------------------------------------------------

def _make_config(max_tokens_per_batch: int = 512) -> Config:
    return Config(
        reranker_api_url="https://api.example.com",
        reranker_api_key="test-key",
        reranker_api_model="test-model",
        reranker_api_max_tokens_per_batch=max_tokens_per_batch,
    )


def test_api_reranker_batch_splitting():
    config = _make_config(max_tokens_per_batch=512)

    with patch("httpx.Client"):
        reranker = APIReranker(config)

    # 10 docs, each ~200 tokens (800 chars)
    docs = ["x" * 800] * 10
    batches = reranker._split_batches(docs, max_tokens=512)

    # Each doc is 200 tokens; batches should have at most 2 docs (200+200=400 <= 512, 400+200=600 > 512)
    assert len(batches) > 1
    for batch in batches:
        total = sum(len(text) // 4 for _, text in batch)
        assert total <= 512 or len(batch) == 1


def test_api_reranker_retry_on_429():
    config = _make_config()

    mock_429 = MagicMock()
    mock_429.status_code = 429

    mock_200 = MagicMock()
    mock_200.status_code = 200
    mock_200.json.return_value = {
        "results": [
            {"index": 0, "relevance_score": 0.8},
            {"index": 1, "relevance_score": 0.3},
        ]
    }
    mock_200.raise_for_status = MagicMock()

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.post.side_effect = [mock_429, mock_429, mock_200]

        reranker = APIReranker(config)

        with patch("time.sleep"):
            result = reranker._call_api_with_retry(
                "query",
                [(0, "doc0"), (1, "doc1")],
                max_retries=3,
            )

    assert mock_client.post.call_count == 3
    assert 0 in result
    assert 1 in result


def test_api_reranker_merge_batches():
    config = _make_config(max_tokens_per_batch=100)

    # 4 docs of 25 tokens each (100 chars); each batch holds at most 4 docs
    # Use smaller docs to force 2 batches: 2 docs per batch (50 tokens each = 200 chars)
    docs = ["x" * 200] * 4  # 50 tokens each; 50+50=100 <= 100, 100+50=150 > 100 -> 2 per batch

    batch0_response = MagicMock()
    batch0_response.status_code = 200
    batch0_response.json.return_value = {
        "results": [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.8},
        ]
    }
    batch0_response.raise_for_status = MagicMock()

    batch1_response = MagicMock()
    batch1_response.status_code = 200
    batch1_response.json.return_value = {
        "results": [
            {"index": 0, "relevance_score": 0.7},
            {"index": 1, "relevance_score": 0.6},
        ]
    }
    batch1_response.raise_for_status = MagicMock()

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.post.side_effect = [batch0_response, batch1_response]

        reranker = APIReranker(config)

        with patch("time.sleep"):
            scores = reranker.score_pairs("query", docs)

    assert len(scores) == 4
    # All original indices should have scores
    assert all(s > 0 for s in scores)
