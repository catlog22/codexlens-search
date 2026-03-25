"""LLM-driven code localization agent for codexlens-search.

This package is optional and designed to degrade gracefully when LLM deps
are not installed.
"""

from codexlens_search.agent.loc_agent import CodeLocAgent
from codexlens_search.agent.tools import (
    GET_ENTITY_CONTENT_TOOL,
    LIST_RELATED_FILES_TOOL,
    READ_FILES_BATCH_TOOL,
    SEARCH_CODE_TOOL,
    TOOL_SCHEMAS,
    TRAVERSE_GRAPH_TOOL,
    get_tool_schemas,
)

__all__ = [
    "CodeLocAgent",
    "GET_ENTITY_CONTENT_TOOL",
    "LIST_RELATED_FILES_TOOL",
    "READ_FILES_BATCH_TOOL",
    "SEARCH_CODE_TOOL",
    "TOOL_SCHEMAS",
    "TRAVERSE_GRAPH_TOOL",
    "get_tool_schemas",
]
