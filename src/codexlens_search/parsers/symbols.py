"""Symbol kinds, dataclass, and AST-based symbol extraction."""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class SymbolKind(enum.Enum):
    """Kinds of code symbols that can be extracted from an AST."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    INTERFACE = "interface"
    ENUM = "enum"
    STRUCT = "struct"
    MODULE = "module"
    VARIABLE = "variable"
    CONSTANT = "constant"
    TYPE_ALIAS = "type_alias"
    TRAIT = "trait"
    IMPL = "impl"


@dataclass(frozen=True)
class Symbol:
    """A code symbol extracted from a tree-sitter AST."""

    name: str
    kind: SymbolKind
    start_line: int  # 1-based
    end_line: int    # 1-based
    parent_name: str | None
    signature: str | None
    language: str


# ---------------------------------------------------------------------------
# Language -> {node_type: SymbolKind} mapping
# ---------------------------------------------------------------------------
# Tier 1 languages with full coverage

_SYMBOL_NODE_TYPES: dict[str, dict[str, SymbolKind]] = {
    "python": {
        "function_definition": SymbolKind.FUNCTION,
        "class_definition": SymbolKind.CLASS,
    },
    "javascript": {
        "function_declaration": SymbolKind.FUNCTION,
        "class_declaration": SymbolKind.CLASS,
        "method_definition": SymbolKind.METHOD,
        "arrow_function": SymbolKind.FUNCTION,
    },
    "typescript": {
        "function_declaration": SymbolKind.FUNCTION,
        "class_declaration": SymbolKind.CLASS,
        "method_definition": SymbolKind.METHOD,
        "arrow_function": SymbolKind.FUNCTION,
        "interface_declaration": SymbolKind.INTERFACE,
        "enum_declaration": SymbolKind.ENUM,
        "type_alias_declaration": SymbolKind.TYPE_ALIAS,
    },
    "go": {
        "function_declaration": SymbolKind.FUNCTION,
        "method_declaration": SymbolKind.METHOD,
        "type_declaration": SymbolKind.TYPE_ALIAS,
    },
    "java": {
        "method_declaration": SymbolKind.METHOD,
        "class_declaration": SymbolKind.CLASS,
        "interface_declaration": SymbolKind.INTERFACE,
        "enum_declaration": SymbolKind.ENUM,
    },
    "rust": {
        "function_item": SymbolKind.FUNCTION,
        "struct_item": SymbolKind.STRUCT,
        "enum_item": SymbolKind.ENUM,
        "trait_item": SymbolKind.TRAIT,
        "impl_item": SymbolKind.IMPL,
        "mod_item": SymbolKind.MODULE,
        "type_item": SymbolKind.TYPE_ALIAS,
    },
    "c": {
        "function_definition": SymbolKind.FUNCTION,
        "struct_specifier": SymbolKind.STRUCT,
        "enum_specifier": SymbolKind.ENUM,
        "type_definition": SymbolKind.TYPE_ALIAS,
    },
    "cpp": {
        "function_definition": SymbolKind.FUNCTION,
        "class_specifier": SymbolKind.CLASS,
        "struct_specifier": SymbolKind.STRUCT,
        "enum_specifier": SymbolKind.ENUM,
        "namespace_definition": SymbolKind.MODULE,
        "type_definition": SymbolKind.TYPE_ALIAS,
    },
}


def _find_name_node(node) -> str | None:
    """Extract the name identifier from a tree-sitter node.

    Walks immediate children looking for common name fields.
    """
    # Try the standard 'name' field first (most grammars use this)
    name_child = node.child_by_field_name("name")
    if name_child is not None:
        return name_child.text.decode("utf-8", errors="replace")

    # Fallback: look for first identifier child
    for child in node.children:
        if child.type in ("identifier", "type_identifier", "field_identifier"):
            return child.text.decode("utf-8", errors="replace")
    return None


def _find_parent_name(node) -> str | None:
    """Walk up the tree to find the nearest named parent symbol."""
    current = node.parent
    while current is not None:
        name = _find_name_node(current)
        if name is not None:
            return name
        current = current.parent
    return None


def _extract_signature(node, source_lines: list[str]) -> str | None:
    """Extract the first line of a symbol as its signature."""
    start = node.start_point[0]
    if start < len(source_lines):
        return source_lines[start].rstrip()
    return None


def extract_symbols(tree, language: str) -> list[Symbol]:
    """Extract symbols from a tree-sitter parse tree.

    Args:
        tree: A ``tree_sitter.Tree`` from ``ASTParser.parse()``.
        language: Language name matching ``_SYMBOL_NODE_TYPES`` keys.

    Returns:
        List of ``Symbol`` instances found in the tree. Empty list if the
        language has no node-type mapping.
    """
    type_map = _SYMBOL_NODE_TYPES.get(language)
    if type_map is None:
        return []

    source_text = tree.root_node.text.decode("utf-8", errors="replace")
    source_lines = source_text.splitlines()
    symbols: list[Symbol] = []

    def _walk(node) -> None:
        kind = type_map.get(node.type)
        if kind is not None:
            name = _find_name_node(node)
            if name:
                parent_name = _find_parent_name(node)
                signature = _extract_signature(node, source_lines)
                symbols.append(
                    Symbol(
                        name=name,
                        kind=kind,
                        start_line=node.start_point[0] + 1,  # 1-based
                        end_line=node.end_point[0] + 1,       # 1-based
                        parent_name=parent_name,
                        signature=signature,
                        language=language,
                    )
                )
        for child in node.children:
            _walk(child)

    _walk(tree.root_node)
    return symbols
