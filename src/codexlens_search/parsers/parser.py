"""Thread-safe lazy singleton AST parser backed by tree-sitter."""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

try:
    from tree_sitter import Language as TSLanguage, Parser as TSParser

    _HAS_TREE_SITTER = True
except ImportError:
    _HAS_TREE_SITTER = False

# Individual grammar packages (tree-sitter 0.23+ modern approach)
# Mapping: language name -> module that exposes a language() function
_GRAMMAR_MODULES: dict[str, str] = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "go": "tree_sitter_go",
    "java": "tree_sitter_java",
    "rust": "tree_sitter_rust",
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
    "ruby": "tree_sitter_ruby",
    "php": "tree_sitter_php",
    "scala": "tree_sitter_scala",
    "kotlin": "tree_sitter_kotlin",
    "swift": "tree_sitter_swift",
    "csharp": "tree_sitter_c_sharp",
    "bash": "tree_sitter_bash",
    "lua": "tree_sitter_lua",
    "haskell": "tree_sitter_haskell",
    "elixir": "tree_sitter_elixir",
    "erlang": "tree_sitter_erlang",
}


class ASTParser:
    """Thread-safe lazy singleton for tree-sitter parsing.

    Grammars are loaded on first use and cached. Languages that fail to
    load are negative-cached so we never retry them.
    """

    _instance: ASTParser | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._grammars: dict[str, object] = {}
        self._unsupported: set[str] = set()
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> ASTParser:
        """Return the singleton instance, creating it if needed."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load_grammar(self, language: str) -> object | None:
        """Load and cache a tree-sitter grammar for *language*.

        Returns the language object or None if unsupported.
        """
        with self._lock:
            if language in self._unsupported:
                return None
            if language in self._grammars:
                return self._grammars[language]

            if not _HAS_TREE_SITTER:
                self._unsupported.add(language)
                return None

            lang_obj = self._try_load_grammar(language)
            if lang_obj is not None:
                self._grammars[language] = lang_obj
                logger.debug("Loaded tree-sitter grammar for %s", language)
                return lang_obj

            logger.debug("tree-sitter grammar not available for %s", language)
            self._unsupported.add(language)
            return None

    @staticmethod
    def _try_load_grammar(language: str):
        """Try loading grammar via individual package, then tree-sitter-languages."""
        # Strategy 1: Individual grammar package (modern, Python 3.13+ compatible)
        mod_name = _GRAMMAR_MODULES.get(language)
        if mod_name:
            try:
                import importlib

                mod = importlib.import_module(mod_name)
                lang_fn = getattr(mod, "language", None)
                if lang_fn is not None:
                    # tree-sitter-typescript exposes typescript/tsx as sub-attrs
                    if language == "typescript" and hasattr(mod, "language_typescript"):
                        return TSLanguage(mod.language_typescript())
                    return TSLanguage(lang_fn())
            except Exception:
                pass

        # Strategy 2: tree-sitter-languages bundle (legacy, broad coverage)
        try:
            import tree_sitter_languages  # type: ignore[import-untyped]

            return tree_sitter_languages.get_language(language)
        except Exception:
            return None

    def parse(self, source: bytes, language: str):
        """Parse *source* bytes using the grammar for *language*.

        Returns a ``tree_sitter.Tree`` or None if the language is
        unsupported or tree-sitter is not installed.
        """
        lang_obj = self._load_grammar(language)
        if lang_obj is None:
            return None

        parser = TSParser()
        parser.language = lang_obj
        return parser.parse(source)

    def supports(self, language: str) -> bool:
        """Return True if *language* grammar can be loaded."""
        return self._load_grammar(language) is not None
