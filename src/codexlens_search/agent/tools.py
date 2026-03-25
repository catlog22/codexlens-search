from __future__ import annotations

from typing import Any

ToolSchema = dict[str, Any]


SEARCH_CODE_TOOL: ToolSchema = {
    "type": "function",
    "function": {
        "name": "search_code",
        "description": (
            "Search the indexed codebase for relevant chunks. "
            "Use this to discover candidate files and code regions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string."},
                "mode": {
                    "type": "string",
                    "description": "Search quality mode.",
                    "enum": ["fast", "balanced", "thorough", "auto"],
                    "default": "thorough",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return.",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
}


TRAVERSE_GRAPH_TOOL: ToolSchema = {
    "type": "function",
    "function": {
        "name": "traverse_graph",
        "description": (
            "Traverse the entity dependency graph starting from an entity name "
            "(symbol) or a file path to discover related entities."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entity_name": {
                    "type": "string",
                    "description": "Seed entity name (symbol) or file path.",
                },
                "depth": {
                    "type": "integer",
                    "description": "Traversal depth.",
                    "minimum": 0,
                    "maximum": 5,
                    "default": 2,
                },
            },
            "required": ["entity_name"],
        },
    },
}


GET_ENTITY_CONTENT_TOOL: ToolSchema = {
    "type": "function",
    "function": {
        "name": "get_entity_content",
        "description": (
            "Read source content for a file and optional line range. "
            "Use this to confirm candidate implementations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "File path to read."},
                "start_line": {
                    "type": "integer",
                    "description": "1-based start line (inclusive).",
                    "minimum": 1,
                    "default": 1,
                },
                "end_line": {
                    "type": "integer",
                    "description": "1-based end line (inclusive).",
                    "minimum": 1,
                    "default": 200,
                },
            },
            "required": ["file_path"],
        },
    },
}


READ_FILES_BATCH_TOOL: ToolSchema = {
    "type": "function",
    "function": {
        "name": "read_files_batch",
        "description": (
            "Read multiple files at once. Returns the first N lines of each file. "
            "Use this to quickly scan several candidate files in a single call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths to read.",
                    "minItems": 1,
                    "maxItems": 20,
                },
                "max_lines_per_file": {
                    "type": "integer",
                    "description": "Maximum lines to read per file.",
                    "minimum": 10,
                    "maximum": 200,
                    "default": 50,
                },
            },
            "required": ["file_paths"],
        },
    },
}


LIST_RELATED_FILES_TOOL: ToolSchema = {
    "type": "function",
    "function": {
        "name": "list_related_files",
        "description": (
            "List files related to a given entity via the dependency graph. "
            "Returns file paths with relationship info (import, call, inherit). "
            "Use this to discover files connected to a known entity without "
            "reading their content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entity_name": {
                    "type": "string",
                    "description": "Seed entity name (symbol) or file path.",
                },
                "depth": {
                    "type": "integer",
                    "description": "Traversal depth.",
                    "minimum": 1,
                    "maximum": 3,
                    "default": 2,
                },
            },
            "required": ["entity_name"],
        },
    },
}


TOOL_SCHEMAS: list[ToolSchema] = [
    SEARCH_CODE_TOOL,
    TRAVERSE_GRAPH_TOOL,
    GET_ENTITY_CONTENT_TOOL,
    READ_FILES_BATCH_TOOL,
    LIST_RELATED_FILES_TOOL,
]


def get_tool_schemas() -> list[ToolSchema]:
    return list(TOOL_SCHEMAS)
