"""Thread-safe lazy singleton AST parser backed by tree-sitter."""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

try:
    from tree_sitter import Parser as TSParser
    import tree_sitter_languages

    _HAS_TREE_SITTER = True
except ImportError:
    _HAS_TREE_SITTER = False


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

            try:
                lang_obj = tree_sitter_languages.get_language(language)
                self._grammars[language] = lang_obj
                logger.debug("Loaded tree-sitter grammar for %s", language)
                return lang_obj
            except Exception:
                logger.debug("tree-sitter grammar not available for %s", language)
                self._unsupported.add(language)
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
