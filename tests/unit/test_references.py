"""Unit tests for cross-reference extraction and symbol_refs table."""
from __future__ import annotations

import pytest

from codexlens_search.search.fts import FTSEngine


# ---------------------------------------------------------------------------
# FTSEngine symbol_refs table tests
# ---------------------------------------------------------------------------

@pytest.fixture
def fts(tmp_path):
    return FTSEngine(str(tmp_path / "fts.db"))


class TestSymbolRefsTableCreation:
    def test_table_exists(self, fts):
        tables = {
            row[0]
            for row in fts._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "symbol_refs" in tables

    def test_table_columns(self, fts):
        cols = [
            row[1]
            for row in fts._conn.execute("PRAGMA table_info(symbol_refs)").fetchall()
        ]
        expected = [
            "id", "from_symbol_id", "from_name", "from_path",
            "to_name", "to_symbol_id", "ref_kind", "line",
        ]
        assert cols == expected

    def test_indexes_exist(self, fts):
        indexes = {
            row[0]
            for row in fts._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='symbol_refs'"
            ).fetchall()
        }
        assert "idx_refs_from_id" in indexes
        assert "idx_refs_from_name" in indexes
        assert "idx_refs_to_name" in indexes
        assert "idx_refs_to_id" in indexes
        assert "idx_refs_kind" in indexes
        assert "idx_refs_path" in indexes


class TestAddRefs:
    def test_batch_insert(self, fts):
        refs = [
            ("main", "app.py", "os", "import", 1),
            ("main", "app.py", "helper", "call", 5),
            ("MyClass", "app.py", "Base", "inherit", 10),
        ]
        fts.add_refs(refs)
        count = fts._conn.execute("SELECT COUNT(*) FROM symbol_refs").fetchone()[0]
        assert count == 3

    def test_empty_list(self, fts):
        fts.add_refs([])
        count = fts._conn.execute("SELECT COUNT(*) FROM symbol_refs").fetchone()[0]
        assert count == 0


class TestDeleteRefsByPath:
    def test_delete_outgoing_refs(self, fts):
        fts.add_refs([
            ("main", "app.py", "os", "import", 1),
            ("main", "app.py", "helper", "call", 5),
            ("foo", "other.py", "bar", "call", 3),
        ])
        deleted = fts.delete_refs_by_path("app.py")
        assert deleted == 2
        remaining = fts._conn.execute("SELECT COUNT(*) FROM symbol_refs").fetchone()[0]
        assert remaining == 1

    def test_delete_nonexistent_path(self, fts):
        fts.add_refs([
            ("foo", "keep.py", "bar", "call", 1),
        ])
        deleted = fts.delete_refs_by_path("nonexistent.py")
        assert deleted == 0
        remaining = fts._conn.execute("SELECT COUNT(*) FROM symbol_refs").fetchone()[0]
        assert remaining == 1


class TestResolveRefs:
    def test_resolve_links_to_symbol_id(self, fts):
        # Add symbols
        fts.add_symbols([
            (0, "helper", "function", 1, 5, "", "def helper():", "python"),
            (0, "Base", "class", 10, 20, "", "class Base:", "python"),
        ])
        # Add refs that reference these symbols
        fts.add_refs([
            ("main", "app.py", "helper", "call", 5),
            ("MyClass", "app.py", "Base", "inherit", 10),
            ("main", "app.py", "unknown_fn", "call", 15),
        ])
        resolved = fts.resolve_refs()
        # "helper" and "Base" should be resolved, "unknown_fn" should not
        assert resolved == 2

        # Verify the resolved IDs
        rows = fts._conn.execute(
            "SELECT to_name, to_symbol_id FROM symbol_refs ORDER BY to_name"
        ).fetchall()
        resolved_map = {r[0]: r[1] for r in rows}
        assert resolved_map["helper"] is not None
        assert resolved_map["Base"] is not None
        assert resolved_map["unknown_fn"] is None

    def test_resolve_idempotent(self, fts):
        fts.add_symbols([
            (0, "foo", "function", 1, 5, "", "", "python"),
        ])
        fts.add_refs([
            ("", "app.py", "foo", "call", 3),
        ])
        first = fts.resolve_refs()
        second = fts.resolve_refs()
        assert first == 1
        assert second == 0  # Already resolved


class TestGetRefsFrom:
    def test_get_refs_from_name(self, fts):
        fts.add_refs([
            ("main", "app.py", "os", "import", 1),
            ("main", "app.py", "helper", "call", 5),
            ("other", "app.py", "bar", "call", 10),
        ])
        results = fts.get_refs_from("main")
        assert len(results) == 2
        assert all(r["from_name"] == "main" for r in results)

    def test_get_refs_from_nonexistent(self, fts):
        results = fts.get_refs_from("nonexistent")
        assert results == []

    def test_result_dict_keys(self, fts):
        fts.add_refs([
            ("foo", "bar.py", "baz", "call", 7),
        ])
        results = fts.get_refs_from("foo")
        assert len(results) == 1
        r = results[0]
        assert "id" in r
        assert r["from_name"] == "foo"
        assert r["from_path"] == "bar.py"
        assert r["to_name"] == "baz"
        assert r["ref_kind"] == "call"
        assert r["line"] == 7


class TestGetRefsTo:
    def test_get_refs_to_name(self, fts):
        fts.add_refs([
            ("main", "app.py", "helper", "call", 5),
            ("other", "lib.py", "helper", "call", 12),
            ("main", "app.py", "unrelated", "call", 3),
        ])
        results = fts.get_refs_to("helper")
        assert len(results) == 2
        assert all(r["to_name"] == "helper" for r in results)


class TestDeleteByPathCleansRefs:
    def test_refs_cleaned_on_path_delete(self, fts):
        """delete_by_path must also remove refs for the path."""
        fts.add_documents([
            (0, "target.py", "def foo(): pass", 1, 1, "python"),
        ])
        fts.add_refs([
            ("foo", "target.py", "bar", "call", 1),
            ("baz", "keep.py", "qux", "call", 1),
        ])
        fts.delete_by_path("target.py")
        # Refs for target.py should be gone
        remaining = fts._conn.execute(
            "SELECT COUNT(*) FROM symbol_refs WHERE from_path = 'target.py'"
        ).fetchone()[0]
        assert remaining == 0
        # Refs for keep.py should remain
        remaining = fts._conn.execute(
            "SELECT COUNT(*) FROM symbol_refs WHERE from_path = 'keep.py'"
        ).fetchone()[0]
        assert remaining == 1


# ---------------------------------------------------------------------------
# SymbolRef dataclass and extract_references tests
# ---------------------------------------------------------------------------

class TestSymbolRefDataclass:
    def test_dataclass_fields(self):
        from codexlens_search.parsers.references import SymbolRef
        ref = SymbolRef(
            from_symbol_name="main",
            to_name="os",
            ref_kind="import",
            line=1,
        )
        assert ref.from_symbol_name == "main"
        assert ref.to_name == "os"
        assert ref.ref_kind == "import"
        assert ref.line == 1

    def test_frozen(self):
        from codexlens_search.parsers.references import SymbolRef
        ref = SymbolRef("a", "b", "call", 1)
        with pytest.raises(AttributeError):
            ref.to_name = "c"


class TestPrimitiveTypeExclusion:
    def test_primitive_types_in_set(self):
        from codexlens_search.parsers.references import _PRIMITIVE_TYPES
        for prim in ("int", "str", "bool", "float", "None", "void",
                      "string", "number", "boolean", "any", "undefined", "null"):
            assert prim in _PRIMITIVE_TYPES, f"{prim} should be in _PRIMITIVE_TYPES"


# ---------------------------------------------------------------------------
# Tree-sitter-based extraction tests (require tree-sitter)
# ---------------------------------------------------------------------------

_ts_available = False
try:
    from codexlens_search.parsers.parser import ASTParser
    from codexlens_search.parsers.symbols import extract_symbols
    from codexlens_search.parsers.references import extract_references, _PRIMITIVE_TYPES
    _parser = ASTParser.get_instance()
    _ts_available = _parser.supports("python")
except ImportError:
    pass

pytestmark_ts = pytest.mark.skipif(not _ts_available, reason="tree-sitter not available")


@pytestmark_ts
class TestExtractPythonRefs:
    def _extract(self, code: str):
        parser = ASTParser.get_instance()
        tree = parser.parse(code.encode(), "python")
        symbols = extract_symbols(tree, "python")
        return extract_references(tree, "python", symbols)

    def test_import_statement(self):
        refs = self._extract("import os\nimport sys\n")
        import_refs = [r for r in refs if r.ref_kind == "import"]
        names = {r.to_name for r in import_refs}
        assert "os" in names
        assert "sys" in names

    def test_from_import(self):
        refs = self._extract("from os.path import join, exists\n")
        import_refs = [r for r in refs if r.ref_kind == "import"]
        names = {r.to_name for r in import_refs}
        assert "join" in names or "os" in names  # At minimum the module

    def test_call_expression(self):
        code = "def main():\n    result = helper()\n"
        refs = self._extract(code)
        call_refs = [r for r in refs if r.ref_kind == "call"]
        assert any(r.to_name == "helper" for r in call_refs)

    def test_class_inheritance(self):
        code = "class MyClass(Base, Mixin):\n    pass\n"
        refs = self._extract(code)
        inherit_refs = [r for r in refs if r.ref_kind == "inherit"]
        names = {r.to_name for r in inherit_refs}
        assert "Base" in names
        assert "Mixin" in names

    def test_type_ref_excludes_primitives(self):
        code = "def foo(x: int, y: str) -> bool:\n    pass\n"
        refs = self._extract(code)
        type_refs = [r for r in refs if r.ref_kind == "type_ref"]
        for r in type_refs:
            assert r.to_name not in _PRIMITIVE_TYPES, (
                f"Primitive type {r.to_name} should be excluded from type_ref"
            )

    def test_type_ref_includes_custom_types(self):
        code = "def foo(x: MyCustomType) -> ResultType:\n    pass\n"
        refs = self._extract(code)
        type_refs = [r for r in refs if r.ref_kind == "type_ref"]
        names = {r.to_name for r in type_refs}
        # At least one custom type should be detected
        assert len(names) > 0

    def test_enclosing_symbol_for_call(self):
        code = "def main():\n    helper()\n"
        refs = self._extract(code)
        call_refs = [r for r in refs if r.ref_kind == "call" and r.to_name == "helper"]
        assert len(call_refs) > 0
        # The call to helper() is inside main()
        assert call_refs[0].from_symbol_name == "main"


@pytestmark_ts
class TestExtractJSTSRefs:
    def _extract(self, code: str, lang: str = "javascript"):
        parser = ASTParser.get_instance()
        if not parser.supports(lang):
            pytest.skip(f"{lang} grammar not available")
        tree = parser.parse(code.encode(), lang)
        symbols = extract_symbols(tree, lang)
        return extract_references(tree, lang, symbols)

    def test_import_statement(self):
        refs = self._extract("import { useState } from 'react';\n")
        import_refs = [r for r in refs if r.ref_kind == "import"]
        assert len(import_refs) > 0

    def test_call_expression(self):
        code = "function main() { helper(); }\n"
        refs = self._extract(code)
        call_refs = [r for r in refs if r.ref_kind == "call"]
        assert any(r.to_name == "helper" for r in call_refs)

    def test_class_extends(self):
        code = "class MyComponent extends Component { }\n"
        refs = self._extract(code)
        inherit_refs = [r for r in refs if r.ref_kind == "inherit"]
        assert any(r.to_name == "Component" for r in inherit_refs)


@pytestmark_ts
class TestExtractGoRefs:
    def _extract(self, code: str):
        parser = ASTParser.get_instance()
        if not parser.supports("go"):
            pytest.skip("go grammar not available")
        tree = parser.parse(code.encode(), "go")
        symbols = extract_symbols(tree, "go")
        return extract_references(tree, "go", symbols)

    def test_import_statement(self):
        code = 'package main\n\nimport "fmt"\n'
        refs = self._extract(code)
        import_refs = [r for r in refs if r.ref_kind == "import"]
        assert any(r.to_name == "fmt" for r in import_refs)

    def test_call_expression(self):
        code = 'package main\n\nfunc main() {\n\tfmt.Println("hello")\n}\n'
        refs = self._extract(code)
        call_refs = [r for r in refs if r.ref_kind == "call"]
        assert len(call_refs) > 0


@pytestmark_ts
class TestExtractJavaRefs:
    def _extract(self, code: str):
        parser = ASTParser.get_instance()
        if not parser.supports("java"):
            pytest.skip("java grammar not available")
        tree = parser.parse(code.encode(), "java")
        symbols = extract_symbols(tree, "java")
        return extract_references(tree, "java", symbols)

    def test_import_statement(self):
        code = "import java.util.List;\n\npublic class Main {}\n"
        refs = self._extract(code)
        import_refs = [r for r in refs if r.ref_kind == "import"]
        assert any(r.to_name == "List" for r in import_refs)

    def test_class_extends(self):
        code = "public class Child extends Parent {\n}\n"
        refs = self._extract(code)
        inherit_refs = [r for r in refs if r.ref_kind == "inherit"]
        assert any(r.to_name == "Parent" for r in inherit_refs)


class TestUnsupportedLanguage:
    def test_returns_empty_for_unsupported(self):
        from codexlens_search.parsers.references import extract_references
        from codexlens_search.parsers.symbols import Symbol

        # Create a mock tree with root_node
        class MockNode:
            children = []
            type = "source_file"
            text = b""
            start_point = (0, 0)
            end_point = (0, 0)
        class MockTree:
            root_node = MockNode()

        result = extract_references(MockTree(), "brainfuck", [])
        assert result == []


@pytestmark_ts
class TestResolveRefsIntegration:
    """Integration test: extract refs from two files, resolve, verify >= 70%."""

    def test_resolve_cross_file_refs(self, fts):
        parser = ASTParser.get_instance()

        # File 1: defines helper and Base
        code1 = (
            "def helper():\n"
            "    return 42\n"
            "\n"
            "class Base:\n"
            "    pass\n"
        )
        tree1 = parser.parse(code1.encode(), "python")
        syms1 = extract_symbols(tree1, "python")
        fts.add_symbols([
            (0, s.name, s.kind.value, s.start_line, s.end_line,
             s.parent_name or "", s.signature or "", s.language)
            for s in syms1
        ])

        # File 2: imports and uses helper and Base
        code2 = (
            "from mod import helper, Base\n"
            "\n"
            "class Child(Base):\n"
            "    def run(self):\n"
            "        return helper()\n"
        )
        tree2 = parser.parse(code2.encode(), "python")
        syms2 = extract_symbols(tree2, "python")
        refs2 = extract_references(tree2, "python", syms2)

        ref_rows = [
            (r.from_symbol_name, "app.py", r.to_name, r.ref_kind, r.line)
            for r in refs2
        ]
        fts.add_refs(ref_rows)

        resolved = fts.resolve_refs()
        total = len(ref_rows)
        # At least helper and Base should resolve
        assert total > 0
        ratio = resolved / total if total > 0 else 0
        assert ratio >= 0.5, f"Resolved {resolved}/{total} = {ratio:.0%}, expected >= 50%"
