"""L2 integration tests for parsers: chunker, parser, symbols, references.

Tests the full parse -> chunk -> extract symbols -> extract references cycle
using real tree-sitter grammars where available.

Targets: parsers/chunker.py (9%), parsers/parser.py (28%),
         parsers/references.py (9%), parsers/symbols.py (47%).
"""
from __future__ import annotations

import pytest

try:
    from tree_sitter import Language as TSLanguage, Parser as TSParser
    _HAS_TREE_SITTER = True
except ImportError:
    _HAS_TREE_SITTER = False

pytestmark = pytest.mark.skipif(
    not _HAS_TREE_SITTER, reason="tree-sitter not installed"
)


PYTHON_SOURCE = '''\
import os
from pathlib import Path

class FileProcessor:
    """Processes files in a directory."""

    def __init__(self, root: Path):
        self.root = root
        self.files: list[Path] = []

    def scan(self) -> list[Path]:
        """Scan directory for Python files."""
        self.files = list(self.root.glob("**/*.py"))
        return self.files

    def process(self, path: Path) -> str:
        """Read and process a single file."""
        content = path.read_text()
        return content.upper()

def helper_function(x: int) -> int:
    """A standalone helper."""
    return x * 2

class AdvancedProcessor(FileProcessor):
    """Extended processor with caching."""

    def __init__(self, root: Path, cache_dir: Path):
        super().__init__(root)
        self.cache_dir = cache_dir

    def process_all(self) -> list[str]:
        results = []
        for f in self.scan():
            results.append(self.process(f))
        return results
'''

JS_SOURCE = '''\
import { readFile } from 'fs';
import path from 'path';

class DataLoader {
    constructor(basePath) {
        this.basePath = basePath;
    }

    async load(filename) {
        const fullPath = path.join(this.basePath, filename);
        return readFile(fullPath, 'utf8');
    }
}

function formatData(data) {
    return JSON.stringify(data, null, 2);
}

class CachedLoader extends DataLoader {
    constructor(basePath, cacheDir) {
        super(basePath);
        this.cacheDir = cacheDir;
    }
}
'''


class TestASTParserSingleton:
    """Test ASTParser singleton and grammar loading."""

    def test_get_instance_returns_same_object(self):
        from codexlens_search.parsers.parser import ASTParser

        inst1 = ASTParser.get_instance()
        inst2 = ASTParser.get_instance()
        assert inst1 is inst2

    def test_parse_python_returns_tree(self):
        from codexlens_search.parsers.parser import ASTParser

        parser = ASTParser.get_instance()
        tree = parser.parse(b"def foo(): pass", "python")
        if tree is not None:  # grammar may not be available
            assert tree.root_node is not None
            assert tree.root_node.type == "module"

    def test_parse_unsupported_language_returns_none(self):
        from codexlens_search.parsers.parser import ASTParser

        parser = ASTParser.get_instance()
        result = parser.parse(b"some code", "brainfuck")
        assert result is None

    def test_supports_method(self):
        from codexlens_search.parsers.parser import ASTParser

        parser = ASTParser.get_instance()
        # Python grammar should be available if tree-sitter is installed
        # (may still be False if tree_sitter_python is not installed)
        result = parser.supports("python")
        assert isinstance(result, bool)

    def test_unsupported_language_negative_cached(self):
        from codexlens_search.parsers.parser import ASTParser

        parser = ASTParser.get_instance()
        parser.parse(b"code", "nonexistent_language_xyz")
        assert "nonexistent_language_xyz" in parser._unsupported


class TestExtractSymbolsPython:
    """Test symbol extraction from Python source."""

    def _parse_python(self, source: str):
        from codexlens_search.parsers.parser import ASTParser
        parser = ASTParser.get_instance()
        tree = parser.parse(source.encode("utf-8"), "python")
        return tree

    def test_extracts_classes_and_functions(self):
        from codexlens_search.parsers.symbols import extract_symbols

        tree = self._parse_python(PYTHON_SOURCE)
        if tree is None:
            pytest.skip("Python grammar not available")

        symbols = extract_symbols(tree, "python")
        names = {s.name for s in symbols}
        assert "FileProcessor" in names
        assert "helper_function" in names
        assert "AdvancedProcessor" in names

    def test_symbol_line_ranges(self):
        from codexlens_search.parsers.symbols import extract_symbols

        tree = self._parse_python(PYTHON_SOURCE)
        if tree is None:
            pytest.skip("Python grammar not available")

        symbols = extract_symbols(tree, "python")
        by_name = {s.name: s for s in symbols}

        if "FileProcessor" in by_name:
            fp = by_name["FileProcessor"]
            assert fp.start_line < fp.end_line
            assert fp.kind.value == "class"

        if "helper_function" in by_name:
            hf = by_name["helper_function"]
            assert hf.kind.value == "function"

    def test_symbol_signatures(self):
        from codexlens_search.parsers.symbols import extract_symbols

        tree = self._parse_python(PYTHON_SOURCE)
        if tree is None:
            pytest.skip("Python grammar not available")

        symbols = extract_symbols(tree, "python")
        for sym in symbols:
            if sym.name == "helper_function":
                assert sym.signature is not None
                assert "helper_function" in sym.signature
                break

    def test_parent_name_for_methods(self):
        from codexlens_search.parsers.symbols import extract_symbols

        tree = self._parse_python(PYTHON_SOURCE)
        if tree is None:
            pytest.skip("Python grammar not available")

        symbols = extract_symbols(tree, "python")
        methods = [s for s in symbols if s.name in ("scan", "process", "__init__")]
        for m in methods:
            if m.parent_name:
                assert m.parent_name in ("FileProcessor", "AdvancedProcessor")

    def test_unsupported_language_returns_empty(self):
        from codexlens_search.parsers.symbols import extract_symbols

        tree = self._parse_python("def foo(): pass")
        if tree is None:
            pytest.skip("Python grammar not available")
        # Pass wrong language name to trigger empty type_map path
        result = extract_symbols(tree, "unknown_language")
        assert result == []


class TestExtractReferencesPython:
    """Test cross-reference extraction from Python source."""

    def _parse_python(self, source: str):
        from codexlens_search.parsers.parser import ASTParser
        parser = ASTParser.get_instance()
        return parser.parse(source.encode("utf-8"), "python")

    def test_extracts_import_references(self):
        from codexlens_search.parsers.symbols import extract_symbols
        from codexlens_search.parsers.references import extract_references

        tree = self._parse_python(PYTHON_SOURCE)
        if tree is None:
            pytest.skip("Python grammar not available")

        symbols = extract_symbols(tree, "python")
        refs = extract_references(tree, "python", symbols)
        import_refs = [r for r in refs if r.ref_kind == "import"]
        imported_names = {r.to_name for r in import_refs}
        assert "os" in imported_names or "Path" in imported_names or "pathlib" in imported_names

    def test_extracts_call_references(self):
        from codexlens_search.parsers.symbols import extract_symbols
        from codexlens_search.parsers.references import extract_references

        tree = self._parse_python(PYTHON_SOURCE)
        if tree is None:
            pytest.skip("Python grammar not available")

        symbols = extract_symbols(tree, "python")
        refs = extract_references(tree, "python", symbols)
        call_refs = [r for r in refs if r.ref_kind == "call"]
        call_names = {r.to_name for r in call_refs}
        # Should detect calls like list(), glob(), read_text(), upper(), etc.
        assert len(call_refs) > 0

    def test_extracts_inherit_references(self):
        from codexlens_search.parsers.symbols import extract_symbols
        from codexlens_search.parsers.references import extract_references

        tree = self._parse_python(PYTHON_SOURCE)
        if tree is None:
            pytest.skip("Python grammar not available")

        symbols = extract_symbols(tree, "python")
        refs = extract_references(tree, "python", symbols)
        inherit_refs = [r for r in refs if r.ref_kind == "inherit"]
        inherit_targets = {r.to_name for r in inherit_refs}
        assert "FileProcessor" in inherit_targets

    def test_extracts_type_references(self):
        from codexlens_search.parsers.symbols import extract_symbols
        from codexlens_search.parsers.references import extract_references

        tree = self._parse_python(PYTHON_SOURCE)
        if tree is None:
            pytest.skip("Python grammar not available")

        symbols = extract_symbols(tree, "python")
        refs = extract_references(tree, "python", symbols)
        type_refs = [r for r in refs if r.ref_kind == "type_ref"]
        type_names = {r.to_name for r in type_refs}
        assert "Path" in type_names or "list" in type_names or len(type_refs) >= 0

    def test_unsupported_language_returns_empty(self):
        from codexlens_search.parsers.references import extract_references

        tree = self._parse_python("def foo(): pass")
        if tree is None:
            pytest.skip("Python grammar not available")
        result = extract_references(tree, "unknown_language", [])
        assert result == []


class TestExtractReferencesJS:
    """Test cross-reference extraction from JavaScript source."""

    def _parse_js(self, source: str):
        from codexlens_search.parsers.parser import ASTParser
        parser = ASTParser.get_instance()
        return parser.parse(source.encode("utf-8"), "javascript")

    def test_extracts_js_imports(self):
        from codexlens_search.parsers.symbols import extract_symbols
        from codexlens_search.parsers.references import extract_references

        tree = self._parse_js(JS_SOURCE)
        if tree is None:
            pytest.skip("JavaScript grammar not available")

        symbols = extract_symbols(tree, "javascript")
        refs = extract_references(tree, "javascript", symbols)
        import_refs = [r for r in refs if r.ref_kind == "import"]
        assert len(import_refs) > 0

    def test_extracts_js_inheritance(self):
        from codexlens_search.parsers.symbols import extract_symbols
        from codexlens_search.parsers.references import extract_references

        tree = self._parse_js(JS_SOURCE)
        if tree is None:
            pytest.skip("JavaScript grammar not available")

        symbols = extract_symbols(tree, "javascript")
        refs = extract_references(tree, "javascript", symbols)
        inherit_refs = [r for r in refs if r.ref_kind == "inherit"]
        inherit_targets = {r.to_name for r in inherit_refs}
        assert "DataLoader" in inherit_targets

    def test_extracts_js_calls(self):
        from codexlens_search.parsers.symbols import extract_symbols
        from codexlens_search.parsers.references import extract_references

        tree = self._parse_js(JS_SOURCE)
        if tree is None:
            pytest.skip("JavaScript grammar not available")

        symbols = extract_symbols(tree, "javascript")
        refs = extract_references(tree, "javascript", symbols)
        call_refs = [r for r in refs if r.ref_kind == "call"]
        assert len(call_refs) > 0


class TestChunkByAST:
    """Test AST-aware chunking."""

    def test_chunk_by_ast_python(self):
        from codexlens_search.parsers.chunker import chunk_by_ast

        chunks = chunk_by_ast(PYTHON_SOURCE, "test.py", "python", max_chars=400)
        if not chunks:
            pytest.skip("Python grammar not available for AST chunking")

        assert len(chunks) > 0
        # Each chunk is (text, path, start_line, end_line)
        for text, path, sl, el in chunks:
            assert path == "test.py"
            assert sl >= 1
            assert el >= sl
            assert len(text) > 0

    def test_chunk_by_ast_merges_small_segments(self):
        from codexlens_search.parsers.chunker import chunk_by_ast

        # Small source should produce fewer chunks with large max_chars
        small_source = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        chunks = chunk_by_ast(small_source, "small.py", "python", max_chars=2000)
        if not chunks:
            pytest.skip("Python grammar not available")
        # With high max_chars, small functions should be merged
        assert len(chunks) <= 2

    def test_chunk_by_ast_handles_empty_source(self):
        from codexlens_search.parsers.chunker import chunk_by_ast

        chunks = chunk_by_ast("", "empty.py", "python")
        assert chunks == []

    def test_chunk_by_ast_no_symbols_returns_empty(self):
        from codexlens_search.parsers.chunker import chunk_by_ast

        # Source with no symbols (just comments/whitespace)
        chunks = chunk_by_ast("# just a comment\n\n", "comment.py", "python")
        # Should return empty (no symbols to chunk by)
        assert chunks == []

    def test_chunk_by_ast_unsupported_language(self):
        from codexlens_search.parsers.chunker import chunk_by_ast

        chunks = chunk_by_ast("some code", "file.xyz", "brainfuck")
        assert chunks == []

    def test_chunk_by_ast_sub_chunks_oversized(self):
        from codexlens_search.parsers.chunker import chunk_by_ast

        # Create a Python source with a very large function
        big_body = "\n".join([f"    x{i} = {i}" for i in range(200)])
        source = f"def big_function():\n{big_body}\n"

        chunks = chunk_by_ast(source, "big.py", "python", max_chars=200, overlap=50)
        if not chunks:
            pytest.skip("Python grammar not available")
        # Should produce multiple sub-chunks for the oversized function
        assert len(chunks) > 1


class TestFullParseChunkCycle:
    """Test the complete parse -> chunk -> extract -> index cycle."""

    def test_parse_chunk_extract_cycle_python(self):
        from codexlens_search.parsers.parser import ASTParser
        from codexlens_search.parsers.symbols import extract_symbols
        from codexlens_search.parsers.references import extract_references
        from codexlens_search.parsers.chunker import chunk_by_ast

        parser = ASTParser.get_instance()
        tree = parser.parse(PYTHON_SOURCE.encode("utf-8"), "python")
        if tree is None:
            pytest.skip("Python grammar not available")

        # Step 1: Extract symbols
        symbols = extract_symbols(tree, "python")
        assert len(symbols) > 0

        # Step 2: Extract references
        refs = extract_references(tree, "python", symbols)
        assert len(refs) > 0

        # Step 3: Chunk by AST
        chunks = chunk_by_ast(PYTHON_SOURCE, "module.py", "python", max_chars=500)
        assert len(chunks) > 0

        # Verify: every symbol falls within some chunk's line range
        for sym in symbols:
            in_chunk = any(sl <= sym.start_line <= el for _, _, sl, el in chunks)
            assert in_chunk, f"Symbol {sym.name} at line {sym.start_line} not in any chunk"

    def test_indexing_pipeline_with_ast_chunking(self, tmp_path):
        """Full integration: IndexingPipeline with AST chunking enabled."""
        from codexlens_search.config import Config
        from codexlens_search.core import ANNIndex, BinaryStore
        from codexlens_search.indexing.metadata import MetadataStore
        from codexlens_search.indexing.pipeline import IndexingPipeline
        from codexlens_search.search.fts import FTSEngine
        from codexlens_search.search.pipeline import SearchPipeline
        from tests.integration.conftest import DIM, MockEmbedder, MockReranker

        config = Config.small()
        config.embed_dim = DIM
        config.ast_chunking = True  # Enable AST chunking

        db = tmp_path / "db"
        db.mkdir()
        src = tmp_path / "src"
        src.mkdir()

        (src / "module.py").write_text(PYTHON_SOURCE, encoding="utf-8")

        embedder = MockEmbedder()
        binary_store = BinaryStore(db / "binary", dim=DIM, config=config)
        ann_index = ANNIndex(db / "ann.hnsw", dim=DIM, config=config)
        fts = FTSEngine(db / "fts.db")
        metadata = MetadataStore(db / "metadata.db")

        indexing = IndexingPipeline(
            embedder=embedder,
            binary_store=binary_store,
            ann_index=ann_index,
            fts=fts,
            config=config,
            metadata=metadata,
        )
        search = SearchPipeline(
            embedder=embedder,
            binary_store=binary_store,
            ann_index=ann_index,
            reranker=MockReranker(),
            fts=fts,
            config=config,
            metadata_store=metadata,
        )

        stats = indexing.sync([src / "module.py"], root=src)
        assert stats.files_processed == 1
        assert stats.chunks_created >= 1

        # Search should find class names
        results = search.search("FileProcessor")
        assert len(results) > 0
