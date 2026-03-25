from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from codexlens_search.agent.loc_agent import (
    CodeLocAgent,
    _extract_file_paths_from_messages,
    _extract_paths_from_search_results,
)
from codexlens_search.config import Config
from codexlens_search.search.pipeline import FileSearchResult


def test_loc_agent_falls_back_when_disabled() -> None:
    cfg = Config()
    cfg.agent_enabled = False

    search = MagicMock()
    search.search_files.return_value = [
        FileSearchResult(path="a.py", score=1.0, best_chunk_id=1),
    ]

    agent = CodeLocAgent(search_pipeline=search, entity_graph=None, config=cfg)
    out = agent.run("find auth", max_iterations=3, top_k=5)

    assert out and out[0].path == "a.py"
    search.search_files.assert_called_once_with("find auth", top_k=5)


def _make_openai_response(tool_calls: list[dict] | None = None, content: str = ""):
    """Build a mock OpenAI ChatCompletion response."""
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


def test_loc_agent_natural_termination() -> None:
    """Agent stops when LLM responds with text only (no tool calls)."""
    cfg = Config()
    cfg.agent_enabled = True
    cfg.agent_llm_model = "glm-4-flash"
    cfg.agent_llm_api_key = "test-key"
    cfg.agent_llm_api_base = "https://open.bigmodel.cn/api/paas/v4/"

    mock_chunk = MagicMock()
    mock_chunk.id = 1
    mock_chunk.path = "a.py"
    mock_chunk.score = 0.5
    mock_chunk.line = 10
    mock_chunk.end_line = 20
    mock_chunk.snippet = "def foo():"

    search = MagicMock()
    search.search.return_value = [mock_chunk]
    search.search_files.return_value = [
        FileSearchResult(path="a.py", score=0.9, best_chunk_id=1, line=10, end_line=20),
        FileSearchResult(path="b.py", score=0.8, best_chunk_id=2, line=30, end_line=40),
    ]

    agent = CodeLocAgent(search_pipeline=search, entity_graph=None, config=cfg)

    # Iteration 1: search_code tool call
    resp1 = _make_openai_response(tool_calls=[
        {
            "id": "call_1",
            "function": {
                "name": "search_code",
                "arguments": json.dumps({"query": "auth", "mode": "thorough", "top_k": 3}),
            },
        }
    ])
    # Iteration 2: text-only response (natural termination)
    resp2 = _make_openai_response(
        content="Based on my analysis, the relevant files are:\n1. a.py\n2. b.py"
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [resp1, resp2]

    with patch("codexlens_search.agent.loc_agent._create_openai_client", return_value=mock_client):
        out = agent.run("find auth", max_iterations=5, top_k=2)

    # Should extract a.py from search results
    assert len(out) > 0
    assert out[0].path == "a.py"
    search.search.assert_called_once()
    # search_files called for baseline scoring during result building
    search.search_files.assert_called_once()


def test_loc_agent_max_iterations_force_extract() -> None:
    """Agent extracts results from history when max iterations reached."""
    cfg = Config()
    cfg.agent_enabled = True
    cfg.agent_llm_model = "glm-5-turbo"
    cfg.agent_llm_api_key = "test-key"

    mock_chunk = MagicMock()
    mock_chunk.id = 1
    mock_chunk.path = "auth/login.py"
    mock_chunk.score = 0.9
    mock_chunk.line = 1
    mock_chunk.end_line = 50
    mock_chunk.snippet = "def login():"

    search = MagicMock()
    search.search.return_value = [mock_chunk]
    search.search_files.return_value = [
        FileSearchResult(path="auth/login.py", score=0.9, best_chunk_id=1),
    ]

    agent = CodeLocAgent(search_pipeline=search, entity_graph=None, config=cfg)

    # Both iterations: tool calls (never stops naturally)
    resp_search = _make_openai_response(tool_calls=[
        {
            "id": "call_1",
            "function": {
                "name": "search_code",
                "arguments": json.dumps({"query": "login"}),
            },
        }
    ])
    resp_search2 = _make_openai_response(tool_calls=[
        {
            "id": "call_2",
            "function": {
                "name": "search_code",
                "arguments": json.dumps({"query": "authentication"}),
            },
        }
    ])

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [resp_search, resp_search2]

    with patch("codexlens_search.agent.loc_agent._create_openai_client", return_value=mock_client):
        out = agent.run("find login bug", max_iterations=2, top_k=5)

    # Should extract auth/login.py from search results even without natural stop
    assert len(out) > 0
    assert out[0].path == "auth/login.py"


def test_loc_agent_read_files_batch_tool() -> None:
    """Agent can use read_files_batch tool."""
    cfg = Config()
    cfg.agent_enabled = True
    cfg.agent_llm_api_key = "test-key"

    search = MagicMock()
    search.search.return_value = []
    search.search_files.return_value = [
        FileSearchResult(path="x.py", score=0.5, best_chunk_id=1),
    ]

    agent = CodeLocAgent(search_pipeline=search, entity_graph=None, config=cfg)

    resp1 = _make_openai_response(tool_calls=[
        {
            "id": "call_1",
            "function": {
                "name": "read_files_batch",
                "arguments": json.dumps({"file_paths": ["x.py", "y.py"]}),
            },
        }
    ])
    resp2 = _make_openai_response(content="The relevant file is x.py")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [resp1, resp2]

    with patch("codexlens_search.agent.loc_agent._create_openai_client", return_value=mock_client):
        out = agent.run("find bug", max_iterations=3, top_k=2)

    assert len(out) > 0


def test_extract_file_paths_from_messages() -> None:
    """Test file path extraction from conversation messages."""
    messages = [
        {"role": "assistant", "content": "I found the issue in src/auth/login.py and utils/helpers.py"},
        {"role": "tool", "content": "1: def foo():\n2:   pass"},
        {"role": "assistant", "content": "Also check config.yaml and main.ts"},
    ]
    paths = _extract_file_paths_from_messages(messages)
    assert "src/auth/login.py" in paths
    assert "utils/helpers.py" in paths
    assert "config.yaml" in paths
    assert "main.ts" in paths


def test_extract_paths_from_search_results() -> None:
    """Test file path extraction from search_code JSON results."""
    search_result = json.dumps([
        {"path": "a.py", "score": 0.9, "id": 1, "line": 1, "end_line": 10, "snippet": "..."},
        {"path": "b.py", "score": 0.8, "id": 2, "line": 5, "end_line": 15, "snippet": "..."},
    ])
    messages = [
        {"role": "tool", "content": search_result},
        {"role": "assistant", "content": "Found some files"},
    ]
    paths = _extract_paths_from_search_results(messages)
    assert paths == ["a.py", "b.py"]
