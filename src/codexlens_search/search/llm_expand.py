"""LLM-based query expansion for enhanced pipeline search.

Uses a single LLM call to extract symbols, concepts, and alternative
search queries from a natural language issue description, then runs
multiple pipeline searches and merges results via reciprocal rank fusion.

Adds ~5-8s latency over plain pipeline search but significantly improves
recall — R@10 matches full iterative agent mode.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..config import Config

_log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
# Role
Code search query optimizer.

# Task
Extract search terms from a bug report / feature request to find relevant source files.

# Output Schema
```json
{
  "symbols": ["exact_function_name", "ClassName", "variable_name"],
  "concepts": ["2-4 word technical phrase", "another concept"],
  "error_terms": ["ErrorType", "error message substring"],
  "sub_queries": ["alternative search angle 1", "alternative search angle 2"]
}
```

# Rules
1. `symbols`: Extract EXACT names from the text. Include class names, function names, variable names, module names.
2. `concepts`: Technical phrases describing the problem domain (2-4 words each, max 4).
3. `error_terms`: Error messages, exception names, log strings mentioned in the report.
4. `sub_queries`: 2-3 queries approaching the problem from DIFFERENT angles — think about what code would need to change:
   - One query targeting the implementation (function/class that needs fixing)
   - One query targeting the caller/consumer of the broken code
   - One query targeting related tests or configuration

# Example
Input: "The `validate_token` function in auth module raises TypeError when token is None"
Output:
```json
{
  "symbols": ["validate_token", "auth"],
  "concepts": ["token validation", "null token handling"],
  "error_terms": ["TypeError"],
  "sub_queries": ["def validate_token None check", "token authentication middleware", "test validate_token edge cases"]
}
```

# Output
Reply with ONLY the JSON object. No markdown fences, no explanation."""

_USER_TEMPLATE = """\
# Issue Description
{query}"""

_USER_TEMPLATE_WITH_GRAPH = """\
# Issue Description
{query}

# Related File Dependencies
```
{graph_info}
```
Target files connected by import/call/inherit edges in your sub_queries."""


def _create_client(config: Config) -> Any:
    """Create an async OpenAI-compatible client for LLM expansion."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        _log.warning("openai package not installed — LLM expansion disabled")
        return None

    api_key = (
        config.llm_expand_api_key
        or config.agent_llm_api_key
        or ""
    )
    if not api_key:
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        _log.warning("No API key for LLM expansion")
        return None

    api_base = config.llm_expand_api_base or config.agent_llm_api_base or ""
    kwargs: dict[str, Any] = {"api_key": api_key}
    if api_base:
        kwargs["base_url"] = api_base
    return AsyncOpenAI(**kwargs)


async def llm_expand_query(
    query: str, config: Config, *, graph_context: str = "",
) -> dict[str, list[str]]:
    """Call LLM to extract search terms from a natural language query.

    Args:
        graph_context: Optional string describing entity graph relationships
            (e.g. import/call edges between files). When provided, the LLM
            is instructed to use these relationships for better expansion.

    Returns dict with keys: symbols, concepts, error_terms, sub_queries.
    Returns empty dict on failure (caller falls back to plain search).
    """
    client = _create_client(config)
    if client is None:
        return {}

    model = config.llm_expand_model or config.agent_llm_model or "glm-5-turbo"

    if graph_context:
        user_content = _USER_TEMPLATE_WITH_GRAPH.format(
            query=query, graph_info=graph_context,
        )
    else:
        user_content = _USER_TEMPLATE.format(query=query)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        # Try structured JSON mode first (OpenAI, GLM, DeepSeek support this)
        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "timeout": 30,
        }
        try:
            create_kwargs["response_format"] = {"type": "json_object"}
            resp = await client.chat.completions.create(**create_kwargs)
        except Exception:
            # Fallback: some providers don't support response_format
            _log.debug("response_format not supported, retrying without")
            del create_kwargs["response_format"]
            resp = await client.chat.completions.create(**create_kwargs)

        raw = (resp.choices[0].message.content or "").strip()
        return _parse_json_response(raw)
    except Exception:
        _log.warning("LLM expansion call failed", exc_info=True)

    return {}


def _parse_json_response(raw: str) -> dict[str, list[str]]:
    """Parse LLM response into structured extraction dict.

    Tries direct JSON parse first, then regex extraction as fallback.
    Validates and normalizes field types.
    """
    # Try direct parse (works when response_format=json_object is used)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return _normalize_extraction(data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: extract JSON object from mixed text
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return _normalize_extraction(data)
        except (json.JSONDecodeError, ValueError):
            pass

    _log.warning("Failed to parse JSON from LLM response: %s", raw[:200])
    return {}


def _normalize_extraction(data: dict) -> dict[str, list[str]]:
    """Ensure all expected fields exist and have list[str] type."""
    result: dict[str, list[str]] = {}
    for key in ("symbols", "concepts", "error_terms", "sub_queries"):
        val = data.get(key, [])
        if isinstance(val, list):
            result[key] = [str(v) for v in val if v]
        elif isinstance(val, str):
            result[key] = [val] if val else []
        else:
            result[key] = []
    return result


def build_expanded_queries(
    query: str, extracted: dict[str, list[str]],
) -> list[str]:
    """Build a list of search queries from LLM extraction results.

    Always includes the original query as first element.
    """
    queries: list[str] = [query]

    symbols = extracted.get("symbols", [])
    error_terms = extracted.get("error_terms", [])
    concepts = extracted.get("concepts", [])
    sub_queries = extracted.get("sub_queries", [])

    # Symbol-augmented query
    extra = symbols + error_terms
    if extra:
        queries.append(f"{query} {' '.join(extra[:10])}")

    # Concept-focused queries
    for concept in concepts[:3]:
        queries.append(concept)

    # Alternative search queries
    for sq in sub_queries[:3]:
        if sq not in queries:
            queries.append(sq)

    return queries


def extract_graph_context(
    file_paths: list[str],
    entity_graph: Any,
    *,
    max_neighbors: int = 5,
    max_lines: int = 30,
) -> str:
    """Extract entity graph relationships for a set of file paths.

    Queries the entity graph for edges involving the given files and formats
    them as human-readable lines for inclusion in the LLM prompt.

    Returns empty string if no graph info is available.
    """
    if entity_graph is None or not file_paths:
        return ""

    from ..core.entity_graph import _entity_for_file

    try:
        entity_graph._ensure_loaded()
    except Exception:
        return ""

    lines: list[str] = []
    seen_edges: set[tuple[str, str, str]] = set()

    for path in file_paths[:10]:
        file_ent = _entity_for_file(path)
        neighbors = entity_graph._neighbors(file_ent)
        for neighbor_ent, weight in neighbors[:max_neighbors]:
            edge_key = (path, neighbor_ent.file_path, neighbor_ent.symbol_name)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            if neighbor_ent.symbol_name:
                lines.append(
                    f"- {path} -> {neighbor_ent.file_path}::{neighbor_ent.symbol_name} "
                    f"({neighbor_ent.kind.value}, weight={weight:.1f})"
                )
            else:
                lines.append(
                    f"- {path} -> {neighbor_ent.file_path} (weight={weight:.1f})"
                )
            if len(lines) >= max_lines:
                break
        if len(lines) >= max_lines:
            break

    return "\n".join(lines)


def merge_file_results_rrf(
    ranked_lists: list[list[Any]],
    *,
    k: int = 60,
    top_k: int = 10,
) -> list[Any]:
    """Merge multiple ranked file result lists via reciprocal rank fusion.

    Each element must have a `.path` and `.score` attribute.
    Returns top_k results with highest RRF scores.
    """
    scores: dict[str, float] = {}
    best_obj: dict[str, Any] = {}

    for results in ranked_lists:
        for rank, r in enumerate(results):
            rrf_score = 1.0 / (k + rank + 1)
            scores[r.path] = scores.get(r.path, 0.0) + rrf_score
            if r.path not in best_obj or r.score > best_obj[r.path].score:
                best_obj[r.path] = r

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [best_obj[path] for path, _ in ranked[:top_k]]
