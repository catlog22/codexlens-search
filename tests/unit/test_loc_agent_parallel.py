"""Tests for parallel tool execution in CodeLocAgent."""
from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch

from codexlens_search.agent.loc_agent import CodeLocAgent
from codexlens_search.config import Config
from codexlens_search.search.pipeline import FileSearchResult


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


def _make_agent(concurrency: int = 1) -> tuple[CodeLocAgent, MagicMock]:
    """Create a CodeLocAgent with mocked search pipeline."""
    cfg = Config()
    cfg.agent_enabled = True
    cfg.agent_llm_api_key = "test-key"
    cfg.agent_tool_concurrency = concurrency

    search = MagicMock()
    search.search.return_value = []
    search.search_files.return_value = [
        FileSearchResult(path="a.py", score=0.9, best_chunk_id=1),
    ]

    agent = CodeLocAgent(search_pipeline=search, entity_graph=None, config=cfg)
    return agent, search


def test_serial_execution_order_preserved() -> None:
    """With concurrency=1, verify serial execution order is preserved."""
    agent, search = _make_agent(concurrency=1)

    execution_order: list[str] = []
    original_execute = agent._execute_tool

    def tracking_execute(name, args, *, default_top_k):
        execution_order.append(name)
        return original_execute(name, args, default_top_k=default_top_k)

    agent._execute_tool = tracking_execute

    tool_calls = [
        {
            "id": "call_1",
            "function": {
                "name": "read_files_batch",
                "arguments": json.dumps({"file_paths": ["a.py"]}),
            },
        },
        {
            "id": "call_2",
            "function": {
                "name": "get_entity_content",
                "arguments": json.dumps({"file_path": "b.py", "start_line": 1, "end_line": 10}),
            },
        },
        {
            "id": "call_3",
            "function": {
                "name": "search_code",
                "arguments": json.dumps({"query": "test"}),
            },
        },
    ]

    msgs, symbols = agent._dispatch_tool_calls(tool_calls, default_top_k=10)

    # Order preserved
    assert len(msgs) == 3
    assert msgs[0]["tool_call_id"] == "call_1"
    assert msgs[1]["tool_call_id"] == "call_2"
    assert msgs[2]["tool_call_id"] == "call_3"
    # All tools executed serially (concurrency=1)
    assert execution_order == ["read_files_batch", "get_entity_content", "search_code"]


def test_parallel_execution_same_results() -> None:
    """With concurrency=4, verify parallel execution of allowlisted tools produces same results."""
    agent, search = _make_agent(concurrency=4)

    tool_calls = [
        {
            "id": "call_1",
            "function": {
                "name": "read_files_batch",
                "arguments": json.dumps({"file_paths": ["a.py"]}),
            },
        },
        {
            "id": "call_2",
            "function": {
                "name": "get_entity_content",
                "arguments": json.dumps({"file_path": "b.py", "start_line": 1, "end_line": 10}),
            },
        },
    ]

    # Run with concurrency=1 for baseline
    agent_serial, _ = _make_agent(concurrency=1)
    serial_msgs, serial_syms = agent_serial._dispatch_tool_calls(tool_calls, default_top_k=10)

    # Run with concurrency=4
    parallel_msgs, parallel_syms = agent._dispatch_tool_calls(tool_calls, default_top_k=10)

    # Same number of results
    assert len(parallel_msgs) == len(serial_msgs)
    # Same tool_call_ids in same order
    for s, p in zip(serial_msgs, parallel_msgs):
        assert s["tool_call_id"] == p["tool_call_id"]
        assert s["role"] == p["role"]
        assert s["content"] == p["content"]


def test_non_allowlisted_tools_always_serial() -> None:
    """Verify search_code and traverse_graph always run serially regardless of concurrency."""
    agent, search = _make_agent(concurrency=4)

    # Track which thread each tool runs on
    thread_ids: dict[str, int] = {}
    original_execute = agent._execute_tool

    def tracking_execute(name, args, *, default_top_k):
        thread_ids[name] = threading.current_thread().ident
        return original_execute(name, args, default_top_k=default_top_k)

    agent._execute_tool = tracking_execute

    tool_calls = [
        {
            "id": "call_1",
            "function": {
                "name": "search_code",
                "arguments": json.dumps({"query": "test"}),
            },
        },
        {
            "id": "call_2",
            "function": {
                "name": "traverse_graph",
                "arguments": json.dumps({"entity_name": "Foo"}),
            },
        },
    ]

    msgs, _ = agent._dispatch_tool_calls(tool_calls, default_top_k=10)

    assert len(msgs) == 2
    assert msgs[0]["tool_call_id"] == "call_1"
    assert msgs[1]["tool_call_id"] == "call_2"
    # Both serial tools should run on the main thread
    main_thread = threading.current_thread().ident
    assert thread_ids["search_code"] == main_thread
    assert thread_ids["traverse_graph"] == main_thread


def test_result_order_matches_tool_calls_order() -> None:
    """Verify result order matches original tool_calls order regardless of completion order."""
    agent, search = _make_agent(concurrency=4)

    # Make the first tool call slower than the second to test ordering
    original_execute = agent._execute_tool
    call_count = {"n": 0}

    def delayed_execute(name, args, *, default_top_k):
        call_count["n"] += 1
        if call_count["n"] == 1:
            time.sleep(0.05)  # First call takes longer
        return original_execute(name, args, default_top_k=default_top_k)

    agent._execute_tool = delayed_execute

    tool_calls = [
        {
            "id": "call_slow",
            "function": {
                "name": "read_files_batch",
                "arguments": json.dumps({"file_paths": ["a.py"]}),
            },
        },
        {
            "id": "call_fast",
            "function": {
                "name": "read_files_batch",
                "arguments": json.dumps({"file_paths": ["b.py"]}),
            },
        },
        {
            "id": "call_serial",
            "function": {
                "name": "search_code",
                "arguments": json.dumps({"query": "test"}),
            },
        },
    ]

    msgs, _ = agent._dispatch_tool_calls(tool_calls, default_top_k=10)

    # Order must match original tool_calls order
    assert [m["tool_call_id"] for m in msgs] == ["call_slow", "call_fast", "call_serial"]


def test_parallel_wall_time_less_than_serial() -> None:
    """With concurrency=4, 3 parallel-safe tool_calls run concurrently (wall-time check)."""
    agent, search = _make_agent(concurrency=4)

    sleep_duration = 0.1
    original_execute = agent._execute_tool

    def slow_execute(name, args, *, default_top_k):
        time.sleep(sleep_duration)
        return original_execute(name, args, default_top_k=default_top_k)

    agent._execute_tool = slow_execute

    tool_calls = [
        {
            "id": f"call_{i}",
            "function": {
                "name": "read_files_batch",
                "arguments": json.dumps({"file_paths": [f"file{i}.py"]}),
            },
        }
        for i in range(3)
    ]

    start = time.monotonic()
    msgs, _ = agent._dispatch_tool_calls(tool_calls, default_top_k=10)
    elapsed = time.monotonic() - start

    assert len(msgs) == 3
    # Parallel: should take ~0.1s, not ~0.3s
    # Use generous threshold to avoid flaky tests
    serial_time = sleep_duration * 3
    assert elapsed < serial_time * 0.8, (
        f"Parallel execution took {elapsed:.3f}s, expected < {serial_time * 0.8:.3f}s"
    )


def test_config_concurrency_clamp() -> None:
    """Verify agent_tool_concurrency is clamped to >= 1."""
    cfg = Config(agent_tool_concurrency=0)
    assert cfg.agent_tool_concurrency == 1

    cfg = Config(agent_tool_concurrency=-5)
    assert cfg.agent_tool_concurrency == 1

    cfg = Config(agent_tool_concurrency=8)
    assert cfg.agent_tool_concurrency == 8


def test_system_prompt_contains_parallel_hint() -> None:
    """Verify system prompt includes parallel tool call encouragement."""
    agent, _ = _make_agent(concurrency=1)
    prompt = agent._build_system_prompt(top_k=10)
    assert "multiple tools in a single response" in prompt
    assert "parallel" in prompt


def test_mixed_parallel_and_serial_calls() -> None:
    """Verify mixed allowlisted and non-allowlisted tools work correctly together."""
    agent, search = _make_agent(concurrency=4)

    thread_ids: dict[str, int] = {}
    original_execute = agent._execute_tool

    def tracking_execute(name, args, *, default_top_k):
        thread_ids[f"{name}_{args}"] = threading.current_thread().ident
        return original_execute(name, args, default_top_k=default_top_k)

    agent._execute_tool = tracking_execute

    tool_calls = [
        {
            "id": "call_1",
            "function": {
                "name": "read_files_batch",
                "arguments": json.dumps({"file_paths": ["a.py"]}),
            },
        },
        {
            "id": "call_2",
            "function": {
                "name": "search_code",
                "arguments": json.dumps({"query": "test"}),
            },
        },
        {
            "id": "call_3",
            "function": {
                "name": "get_entity_content",
                "arguments": json.dumps({"file_path": "c.py", "start_line": 1, "end_line": 10}),
            },
        },
    ]

    msgs, _ = agent._dispatch_tool_calls(tool_calls, default_top_k=10)

    # All results present in correct order
    assert len(msgs) == 3
    assert msgs[0]["tool_call_id"] == "call_1"
    assert msgs[1]["tool_call_id"] == "call_2"
    assert msgs[2]["tool_call_id"] == "call_3"

    # search_code must run on main thread (serial)
    main_thread = threading.current_thread().ident
    search_key = [k for k in thread_ids if k.startswith("search_code")][0]
    assert thread_ids[search_key] == main_thread
