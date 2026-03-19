"""Cross-reference extraction from tree-sitter ASTs.

Extracts import, call, inherit, and type_ref references using tree-sitter
node traversal. Each reference links a source symbol (or file scope) to a
target name.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from codexlens_search.parsers.symbols import Symbol

logger = logging.getLogger(__name__)

# Primitive types to exclude from type_ref extraction
_PRIMITIVE_TYPES: frozenset[str] = frozenset({
    # Python
    "int", "str", "bool", "float", "None", "bytes", "complex",
    # JS/TS
    "void", "string", "number", "boolean", "any", "undefined", "null",
    "never", "unknown", "object", "symbol", "bigint",
    # Java
    "byte", "short", "long", "double", "char",
    # Go
    "int8", "int16", "int32", "int64",
    "uint", "uint8", "uint16", "uint32", "uint64",
    "float32", "float64", "complex64", "complex128",
    "uintptr", "rune", "error",
})


@dataclass(frozen=True)
class SymbolRef:
    """A cross-reference between two symbols or a file scope and a symbol."""

    from_symbol_name: str  # Name of the enclosing symbol, or "" for file scope
    to_name: str           # Name of the referenced symbol/module
    ref_kind: str          # "import" | "call" | "inherit" | "type_ref"
    line: int              # 1-based line number


# ---------------------------------------------------------------------------
# Per-language node type patterns for reference extraction
# ---------------------------------------------------------------------------
# Each language maps ref_kind -> list of (parent_node_type, child_field_or_type)
# patterns to look for during tree traversal.

# Python import patterns
_PYTHON_IMPORT_TYPES = frozenset({
    "import_statement",
    "import_from_statement",
})

# JS/TS import patterns
_JS_IMPORT_TYPES = frozenset({
    "import_statement",
    "import_declaration",       # TS uses this sometimes
})

# Python call expression
_CALL_TYPES = frozenset({
    "call",                     # Python
    "call_expression",          # JS/TS/Go/Java
})

# Python inheritance
_PYTHON_CLASS_TYPES = frozenset({
    "class_definition",
})

# JS/TS inheritance
_JS_CLASS_TYPES = frozenset({
    "class_declaration",
})

# Java inheritance
_JAVA_CLASS_TYPES = frozenset({
    "class_declaration",
})

# Go call expression
_GO_CALL_TYPES = frozenset({
    "call_expression",
})

# Go import
_GO_IMPORT_TYPES = frozenset({
    "import_declaration",
    "import_spec",
})

# Java import
_JAVA_IMPORT_TYPES = frozenset({
    "import_declaration",
})


def _get_node_text(node) -> str:
    """Decode node text to string."""
    return node.text.decode("utf-8", errors="replace")


def _find_enclosing_symbol(node, symbols: list[Symbol]) -> str:
    """Find the innermost symbol that contains the given node's line."""
    line = node.start_point[0] + 1  # 1-based
    best: Symbol | None = None
    for sym in symbols:
        if sym.start_line <= line <= sym.end_line:
            if best is None or (sym.end_line - sym.start_line) < (best.end_line - best.start_line):
                best = sym
    return best.name if best else ""


def _extract_identifier_name(node) -> str | None:
    """Extract a simple identifier name from a node, handling dotted access.

    For dotted names like ``os.path``, returns the last component (``path``).
    For simple identifiers, returns the text directly.
    """
    if node.type in ("identifier", "type_identifier", "shorthand_property_identifier"):
        return _get_node_text(node)
    # Dotted name: a.b.c -> return "c" or full text for imports
    if node.type in ("dotted_name", "attribute", "member_expression",
                      "scoped_identifier", "field_expression",
                      "selector_expression"):
        text = _get_node_text(node)
        return text.rsplit(".", 1)[-1] if "." in text else text
    return None


# ---------------------------------------------------------------------------
# Language-specific extractors
# ---------------------------------------------------------------------------

def _extract_python_refs(root_node, symbols: list[Symbol]) -> list[SymbolRef]:
    """Extract references from a Python AST."""
    refs: list[SymbolRef] = []

    def _walk(node) -> None:
        ntype = node.type
        line = node.start_point[0] + 1

        # Imports
        if ntype in _PYTHON_IMPORT_TYPES:
            from_sym = _find_enclosing_symbol(node, symbols)
            if ntype == "import_statement":
                # import foo, bar
                for child in node.children:
                    if child.type in ("dotted_name", "aliased_import"):
                        target = child.child_by_field_name("name") if child.type == "aliased_import" else child
                        if target is not None:
                            name = _get_node_text(target)
                            # Use the base module name
                            base = name.split(".")[0] if "." in name else name
                            refs.append(SymbolRef(from_sym, base, "import", line))
            elif ntype == "import_from_statement":
                # from foo import bar, baz
                module_node = node.child_by_field_name("module_name")
                if module_node is None:
                    # Try to find the dotted_name child
                    for child in node.children:
                        if child.type == "dotted_name":
                            module_node = child
                            break
                # Extract imported names
                for child in node.children:
                    if child.type in ("dotted_name",) and child != module_node:
                        refs.append(SymbolRef(from_sym, _get_node_text(child), "import", line))
                    elif child.type == "aliased_import":
                        target = child.child_by_field_name("name")
                        if target is not None:
                            refs.append(SymbolRef(from_sym, _get_node_text(target), "import", line))
                    elif child.type == "identifier" and child != module_node:
                        # Plain identifier in import list
                        text = _get_node_text(child)
                        if text not in ("from", "import", "as"):
                            refs.append(SymbolRef(from_sym, text, "import", line))
                # Also reference the module itself
                if module_node is not None:
                    mod_text = _get_node_text(module_node)
                    base = mod_text.split(".")[0] if "." in mod_text else mod_text
                    refs.append(SymbolRef(from_sym, base, "import", line))
            return  # Don't recurse into import children

        # Calls
        if ntype == "call":
            from_sym = _find_enclosing_symbol(node, symbols)
            func_node = node.child_by_field_name("function")
            if func_node is not None:
                name = _extract_identifier_name(func_node)
                if name and name not in _PRIMITIVE_TYPES:
                    refs.append(SymbolRef(from_sym, name, "call", line))

        # Class inheritance
        if ntype == "class_definition":
            from_sym = ""
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                from_sym = _get_node_text(name_node)
            # Look for superclasses in argument_list
            superclasses = node.child_by_field_name("superclasses")
            if superclasses is not None:
                for child in superclasses.children:
                    base_name = _extract_identifier_name(child)
                    if base_name and base_name not in _PRIMITIVE_TYPES:
                        refs.append(SymbolRef(from_sym, base_name, "inherit", line))

        # Type annotations
        if ntype == "type":
            from_sym = _find_enclosing_symbol(node, symbols)
            type_text = _get_node_text(node)
            # Split on common type separators and extract identifiers
            for part in type_text.replace("[", " ").replace("]", " ").replace(",", " ").replace("|", " ").split():
                part = part.strip()
                if part and part not in _PRIMITIVE_TYPES and part.isidentifier():
                    refs.append(SymbolRef(from_sym, part, "type_ref", line))
            return  # Don't recurse into type children

        for child in node.children:
            _walk(child)

    _walk(root_node)
    return refs


def _extract_js_ts_refs(root_node, symbols: list[Symbol]) -> list[SymbolRef]:
    """Extract references from a JavaScript/TypeScript AST."""
    refs: list[SymbolRef] = []

    def _walk(node) -> None:
        ntype = node.type
        line = node.start_point[0] + 1

        # Imports
        if ntype in _JS_IMPORT_TYPES or ntype == "import_statement":
            from_sym = _find_enclosing_symbol(node, symbols)
            # Extract imported names from import specifiers
            for child in node.children:
                if child.type == "import_clause":
                    for spec in child.children:
                        if spec.type == "identifier":
                            refs.append(SymbolRef(from_sym, _get_node_text(spec), "import", line))
                        elif spec.type == "named_imports":
                            for imp in spec.children:
                                if imp.type == "import_specifier":
                                    name_node = imp.child_by_field_name("name")
                                    if name_node is not None:
                                        refs.append(SymbolRef(from_sym, _get_node_text(name_node), "import", line))
                                elif imp.type == "identifier":
                                    refs.append(SymbolRef(from_sym, _get_node_text(imp), "import", line))
                        elif spec.type == "namespace_import":
                            # import * as X
                            for sub in spec.children:
                                if sub.type == "identifier":
                                    refs.append(SymbolRef(from_sym, _get_node_text(sub), "import", line))
                elif child.type == "string" or child.type == "string_fragment":
                    # Module path - extract last component
                    mod_text = _get_node_text(child).strip("'\"")
                    parts = mod_text.replace("\\", "/").split("/")
                    base = parts[-1] if parts else mod_text
                    if base and base.isidentifier():
                        refs.append(SymbolRef(from_sym, base, "import", line))
            return

        # Calls
        if ntype == "call_expression":
            from_sym = _find_enclosing_symbol(node, symbols)
            func_node = node.child_by_field_name("function")
            if func_node is not None:
                name = _extract_identifier_name(func_node)
                if name and name not in _PRIMITIVE_TYPES:
                    refs.append(SymbolRef(from_sym, name, "call", line))

        # Class heritage (extends/implements)
        if ntype in ("class_declaration", "class"):
            class_name = ""
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                class_name = _get_node_text(name_node)
            # Look for class_heritage node
            for child in node.children:
                if child.type == "class_heritage":
                    for heritage_child in child.children:
                        if heritage_child.type in ("extends_clause", "implements_clause"):
                            # Wrapped in extends_clause/implements_clause (some grammars)
                            for sub in heritage_child.children:
                                name = _extract_identifier_name(sub)
                                if name and name not in _PRIMITIVE_TYPES and name not in ("extends", "implements"):
                                    refs.append(SymbolRef(class_name, name, "inherit", line))
                        else:
                            # Direct identifier under class_heritage (tree-sitter 0.23+)
                            name = _extract_identifier_name(heritage_child)
                            if name and name not in _PRIMITIVE_TYPES and name not in ("extends", "implements"):
                                refs.append(SymbolRef(class_name, name, "inherit", line))

        # Type annotations (TypeScript)
        if ntype in ("type_annotation", "type_identifier", "generic_type"):
            from_sym = _find_enclosing_symbol(node, symbols)
            if ntype == "type_identifier":
                text = _get_node_text(node)
                if text not in _PRIMITIVE_TYPES:
                    refs.append(SymbolRef(from_sym, text, "type_ref", line))
                return
            elif ntype == "generic_type":
                # Extract the base type name
                name_node = node.child_by_field_name("name") or (
                    node.children[0] if node.children else None
                )
                if name_node is not None:
                    name = _extract_identifier_name(name_node)
                    if name and name not in _PRIMITIVE_TYPES:
                        refs.append(SymbolRef(from_sym, name, "type_ref", line))

        for child in node.children:
            _walk(child)

    _walk(root_node)
    return refs


def _extract_go_refs(root_node, symbols: list[Symbol]) -> list[SymbolRef]:
    """Extract references from a Go AST."""
    refs: list[SymbolRef] = []

    def _walk(node) -> None:
        ntype = node.type
        line = node.start_point[0] + 1

        # Imports
        if ntype == "import_spec":
            from_sym = _find_enclosing_symbol(node, symbols)
            path_node = node.child_by_field_name("path")
            if path_node is not None:
                path_text = _get_node_text(path_node).strip('"')
                base = path_text.rsplit("/", 1)[-1] if "/" in path_text else path_text
                if base:
                    refs.append(SymbolRef(from_sym, base, "import", line))
            return

        if ntype == "import_declaration":
            # Recurse into children to find import_spec nodes
            for child in node.children:
                _walk(child)
            return

        # Calls
        if ntype == "call_expression":
            from_sym = _find_enclosing_symbol(node, symbols)
            func_node = node.child_by_field_name("function")
            if func_node is not None:
                name = _extract_identifier_name(func_node)
                if name and name not in _PRIMITIVE_TYPES:
                    refs.append(SymbolRef(from_sym, name, "call", line))

        # Type references in type declarations
        if ntype == "type_spec":
            name_node = node.child_by_field_name("name")
            type_node = node.child_by_field_name("type")
            type_name = _get_node_text(name_node) if name_node else ""
            if type_node is not None:
                # Check for struct embedding (inherit-like)
                if type_node.type == "struct_type":
                    for field in type_node.children:
                        if field.type == "field_declaration_list":
                            for fdecl in field.children:
                                if fdecl.type == "field_declaration":
                                    # Anonymous field = embedding
                                    name_f = fdecl.child_by_field_name("name")
                                    type_f = fdecl.child_by_field_name("type")
                                    if name_f is None and type_f is not None:
                                        embed_name = _extract_identifier_name(type_f)
                                        if embed_name and embed_name not in _PRIMITIVE_TYPES:
                                            refs.append(SymbolRef(type_name, embed_name, "inherit", line))
                                    elif type_f is not None:
                                        t_name = _extract_identifier_name(type_f)
                                        if t_name and t_name not in _PRIMITIVE_TYPES:
                                            refs.append(SymbolRef(type_name, t_name, "type_ref", fdecl.start_point[0] + 1))
                elif type_node.type == "interface_type":
                    # Interface embedding
                    for child in type_node.children:
                        if child.type == "type_identifier":
                            embed_name = _get_node_text(child)
                            if embed_name not in _PRIMITIVE_TYPES:
                                refs.append(SymbolRef(type_name, embed_name, "inherit", child.start_point[0] + 1))

        # Type identifiers in general context
        if ntype == "type_identifier":
            from_sym = _find_enclosing_symbol(node, symbols)
            text = _get_node_text(node)
            if text not in _PRIMITIVE_TYPES:
                # Only add if parent is not a type_spec name (avoid self-ref)
                parent = node.parent
                if parent and parent.type == "type_spec":
                    name_field = parent.child_by_field_name("name")
                    if name_field is not None and name_field == node:
                        # This is the definition name, skip
                        pass
                    else:
                        refs.append(SymbolRef(from_sym, text, "type_ref", line))
                else:
                    refs.append(SymbolRef(from_sym, text, "type_ref", line))

        for child in node.children:
            _walk(child)

    _walk(root_node)
    return refs


def _extract_java_refs(root_node, symbols: list[Symbol]) -> list[SymbolRef]:
    """Extract references from a Java AST."""
    refs: list[SymbolRef] = []

    def _walk(node) -> None:
        ntype = node.type
        line = node.start_point[0] + 1

        # Imports
        if ntype == "import_declaration":
            from_sym = _find_enclosing_symbol(node, symbols)
            for child in node.children:
                if child.type == "scoped_identifier":
                    text = _get_node_text(child)
                    # Use the last component
                    name = text.rsplit(".", 1)[-1] if "." in text else text
                    if name != "*":
                        refs.append(SymbolRef(from_sym, name, "import", line))
            return

        # Calls
        if ntype == "method_invocation":
            from_sym = _find_enclosing_symbol(node, symbols)
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                name = _get_node_text(name_node)
                if name not in _PRIMITIVE_TYPES:
                    refs.append(SymbolRef(from_sym, name, "call", line))

        # Object creation (constructor call)
        if ntype == "object_creation_expression":
            from_sym = _find_enclosing_symbol(node, symbols)
            type_node = node.child_by_field_name("type")
            if type_node is not None:
                name = _extract_identifier_name(type_node)
                if name and name not in _PRIMITIVE_TYPES:
                    refs.append(SymbolRef(from_sym, name, "call", line))

        # Class inheritance (extends/implements)
        if ntype == "class_declaration":
            class_name = ""
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                class_name = _get_node_text(name_node)
            superclass = node.child_by_field_name("superclass")
            if superclass is not None:
                name = _extract_identifier_name(superclass)
                if name is None:
                    # superclass node wraps extends keyword + type_identifier
                    for sc_child in superclass.children:
                        name = _extract_identifier_name(sc_child)
                        if name and name not in _PRIMITIVE_TYPES:
                            refs.append(SymbolRef(class_name, name, "inherit", line))
                            break
                elif name not in _PRIMITIVE_TYPES:
                    refs.append(SymbolRef(class_name, name, "inherit", line))
            interfaces = node.child_by_field_name("interfaces")
            if interfaces is not None:
                for child in interfaces.children:
                    name = _extract_identifier_name(child)
                    if name and name not in _PRIMITIVE_TYPES and name != "implements":
                        refs.append(SymbolRef(class_name, name, "inherit", line))

        # Interface inheritance
        if ntype == "interface_declaration":
            iface_name = ""
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                iface_name = _get_node_text(name_node)
            extends = node.child_by_field_name("extends_interfaces")
            if extends is None:
                # Try type_list child
                for child in node.children:
                    if child.type == "extends_interfaces":
                        extends = child
                        break
            if extends is not None:
                for child in extends.children:
                    name = _extract_identifier_name(child)
                    if name and name not in _PRIMITIVE_TYPES and name != "extends":
                        refs.append(SymbolRef(iface_name, name, "inherit", line))

        # Type references
        if ntype == "type_identifier":
            from_sym = _find_enclosing_symbol(node, symbols)
            text = _get_node_text(node)
            if text not in _PRIMITIVE_TYPES:
                refs.append(SymbolRef(from_sym, text, "type_ref", line))

        for child in node.children:
            _walk(child)

    _walk(root_node)
    return refs


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------
_LANGUAGE_EXTRACTORS = {
    "python": _extract_python_refs,
    "javascript": _extract_js_ts_refs,
    "typescript": _extract_js_ts_refs,
    "go": _extract_go_refs,
    "java": _extract_java_refs,
}


def extract_references(
    tree,
    language: str,
    symbols: list[Symbol],
) -> list[SymbolRef]:
    """Extract cross-references from a tree-sitter parse tree.

    Uses tree-sitter node traversal to find imports, calls, inheritance,
    and type references.

    Args:
        tree: A ``tree_sitter.Tree`` from ``ASTParser.parse()``.
        language: Language name (python, javascript, typescript, go, java).
        symbols: List of symbols previously extracted from the same tree,
                 used to determine the enclosing symbol for each reference.

    Returns:
        List of ``SymbolRef`` instances. Empty list if language is unsupported.
    """
    extractor = _LANGUAGE_EXTRACTORS.get(language)
    if extractor is None:
        return []
    try:
        return extractor(tree.root_node, symbols)
    except Exception:
        logger.debug("Reference extraction failed for language %s", language, exc_info=True)
        return []
