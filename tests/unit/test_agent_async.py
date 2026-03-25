"""Tests for async agent behavior (P1: Async LLM API Calls)."""
from __future__ import annotations

import asyncio
import inspect
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codexlens_search.agent.loc_agent import (
    CodeLocAgent,
    _call_openai_async,
    _create_async_openai_client,
)
from codexlens_search.config import Config
from codexlens_search.search.pipeline import FileSearchResult


def _make_config(**overrides) -> Config:
    defaults = {
        "agent_enabled": True,
        "agent_llm_api_key": "test-key",
        "agent_llm_api_base": "https://api.example.com/v1",
        "agent_llm_model": "test-model",
    }
    defaults.update(overrides)
    return Config(**defaults)


def _make_mock_search() -> MagicMock:
    search = MagicMock()
    search.search.return_value = []
    search.search_files.return_value = [
        FileSearchResult(path="fallback.py", score=0.5, best_chunk_id=1),
    ]
    return search


def _make_async_openai_response(
    tool_calls: list[dict] | None = None, content: str = "",
):
    """Build a mock async OpenAI ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    if tool_calls:
        tc_mocks = []
        for tc in tool_calls:
            tc_mock = MagicMock()
            tc_mock.id = tc["id"]
            tc_mock.function.name = tc["function"]["name"]
            tc_mock.function.arguments = tc["function"]["arguments"]
            tc_mocks.append(tc_mock)
        msg.tool_calls = tc_mocks
    else:
        msg.tool_calls = None

    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_is_coroutine_function() -> None:
    """Verify that agent.run() is an async coroutine function."""
    assert inspect.iscoroutinefunction(CodeLocAgent.run)


def test_run_sync_is_not_coroutine_function() -> None:
    """Verify that agent.run_sync() is a regular sync function."""
    assert not inspect.iscoroutinefunction(CodeLocAgent.run_sync)


def test_run_sync_returns_results() -> None:
    """Verify run_sync() works from synchronous context and returns valid results."""
    cfg = _make_config(agent_enabled=False)
    search = _make_mock_search()
    agent = CodeLocAgent(search_pipeline=search, entity_graph=None, config=cfg)

    results = agent.run_sync("find auth", max_iterations=3, top_k=5)

    assert len(results) > 0
    assert results[0].path == "fallback.py"
    search.search_files.assert_called_once_with("find auth", top_k=5)


def test_run_sync_with_llm_calls() -> None:
    """Verify run_sync() drives full async iteration loop with mocked LLM."""
    cfg = _make_config()
    search = _make_mock_search()

    mock_chunk = MagicMock()
    mock_chunk.id = 1
    mock_chunk.path = "auth.py"
    mock_chunk.score = 0.9
    mock_chunk.line = 1
    mock_chunk.end_line = 50
    mock_chunk.snippet = "def login():"
    search.search.return_value = [mock_chunk]
    search.search_files.return_value = [
        FileSearchResult(path="auth.py", score=0.9, best_chunk_id=1, line=1, end_line=50),
    ]

    agent = CodeLocAgent(search_pipeline=search, entity_graph=None, config=cfg)

    # Iteration 1: search tool call
    resp1 = _make_async_openai_response(tool_calls=[
        {
            "id": "call_1",
            "function": {
                "name": "search_code",
                "arguments": json.dumps({"query": "login"}),
            },
        }
    ])
    # Iteration 2: natural termination
    resp2 = _make_async_openai_response(
        content="The relevant file is auth.py"
    )

    mock_async_client = AsyncMock()
    mock_async_client.chat.completions.create = AsyncMock(side_effect=[resp1, resp2])

    with patch(
        "codexlens_search.agent.loc_agent._create_async_openai_client",
        return_value=mock_async_client,
    ):
        results = agent.run_sync("find login bug", max_iterations=5, top_k=5)

    assert len(results) > 0
    assert results[0].path == "auth.py"
    # Async client should have been called twice
    assert mock_async_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_run_async_uses_async_client() -> None:
    """Verify run() uses AsyncOpenAI client when running in async context."""
    cfg = _make_config()
    search = _make_mock_search()
    agent = CodeLocAgent(search_pipeline=search, entity_graph=None, config=cfg)

    # Natural termination response
    resp = _make_async_openai_response(content="No files found")

    mock_async_client = AsyncMock()
    mock_async_client.chat.completions.create = AsyncMock(return_value=resp)

    with patch(
        "codexlens_search.agent.loc_agent._create_async_openai_client",
        return_value=mock_async_client,
    ):
        results = await agent.run("test query", max_iterations=1, top_k=5)

    # Should have used async client
    mock_async_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_run_async_falls_back_when_disabled() -> None:
    """Verify async run() still falls back to search_files when agent is disabled."""
    cfg = _make_config(agent_enabled=False)
    search = _make_mock_search()
    agent = CodeLocAgent(search_pipeline=search, entity_graph=None, config=cfg)

    results = await agent.run("test", max_iterations=3, top_k=5)

    assert len(results) > 0
    assert results[0].path == "fallback.py"


@pytest.mark.asyncio
async def test_call_openai_async_returns_none_on_failure() -> None:
    """Verify _call_openai_async returns None when API call fails."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))

    result = await _call_openai_async(
        mock_client, "test-model", [{"role": "user", "content": "test"}], []
    )

    assert result is None


@pytest.mark.asyncio
async def test_call_openai_async_parses_tool_calls() -> None:
    """Verify _call_openai_async correctly parses tool calls from response."""
    resp = _make_async_openai_response(
        tool_calls=[
            {
                "id": "tc_1",
                "function": {
                    "name": "search_code",
                    "arguments": '{"query": "test"}',
                },
            }
        ],
        content="searching...",
    )

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=resp)

    result = await _call_openai_async(
        mock_client, "model", [{"role": "user", "content": "q"}], []
    )

    assert result is not None
    assert result["role"] == "assistant"
    assert result["content"] == "searching..."
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "search_code"


def test_create_async_openai_client_no_key() -> None:
    """Verify _create_async_openai_client returns None when no key is available."""
    with patch.dict("os.environ", {}, clear=True):
        client = _create_async_openai_client("", "")
    assert client is None


def test_get_async_client_lazy_creates() -> None:
    """Verify _get_async_client() lazily creates the async client."""
    cfg = _make_config()
    search = _make_mock_search()
    agent = CodeLocAgent(search_pipeline=search, entity_graph=None, config=cfg)

    assert agent._async_client is None

    with patch(
        "codexlens_search.agent.loc_agent._create_async_openai_client",
        return_value=MagicMock(),
    ) as mock_create:
        client = agent._get_async_client()
        assert client is not None
        mock_create.assert_called_once()

        # Second call should reuse
        client2 = agent._get_async_client()
        assert client2 is client
        assert mock_create.call_count == 1
