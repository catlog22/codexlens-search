"""AST-based chunking and symbol extraction using tree-sitter.

Requires optional ``ast`` extra: ``pip install codexlens-search[ast]``
"""
from __future__ import annotations

from codexlens_search.parsers.symbols import Symbol, SymbolKind, extract_symbols
from codexlens_search.parsers.parser import ASTParser
from codexlens_search.parsers.chunker import chunk_by_ast
from codexlens_search.parsers.references import SymbolRef, extract_references

__all__ = [
    "ASTParser",
    "Symbol",
    "SymbolKind",
    "SymbolRef",
    "chunk_by_ast",
    "extract_references",
    "extract_symbols",
]
