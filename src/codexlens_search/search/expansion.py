"""Query expansion via symbol vocabulary embedding + two-hop neighbor discovery.

Bridges the semantic gap between abstract natural language queries and concrete
code tokens by:
  1. Building a vocabulary of symbol names from the indexed codebase
  2. Embedding the vocabulary using the same model as search
  3. At query time: finding nearest symbols via cosine similarity (first hop)
  4. Looking up chunk neighbors of those symbols in the FTS DB (second hop)
  5. Appending discovered terms to the query before search

Only activates for natural language queries (intent gating) to avoid
polluting code-symbol queries that already work well.
"""
from __future__ import annotations

import logging
import re
import threading

import numpy as np

from ..config import Config
from ..embed import BaseEmbedder
from .fts import FTSEngine
from .fusion import QueryIntent, detect_query_intent

_log = logging.getLogger(__name__)

_IDENT_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Common abbreviations seen in code and issue descriptions.
_ABBREV_MAP: dict[str, str] = {
    "auth": "authentication",
    "cfg": "config",
    "conf": "config",
    "db": "database",
    "svc": "service",
    "repo": "repository",
    "ctx": "context",
    "env": "environment",
    "req": "request",
    "resp": "response",
    "msg": "message",
    "init": "initialize",
    "impl": "implementation",
    "deps": "dependencies",
    "perms": "permissions",
}


def _split_identifier(token: str) -> list[str]:
    """Split CamelCase / snake_case identifier into lowercased parts."""
    t = token.strip("_")
    if not t:
        return []

    # snake_case → spaces
    t = t.replace("_", " ")
    # HTTPServer → HTTP Server, getUserName → get User Name
    t = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", t)
    t = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", t)

    parts = [p.lower() for p in t.split() if p]
    # Filter trivial parts ("i", "x")
    return [p for p in parts if len(p) > 1]


def _split_identifiers(text: str, max_terms: int = 20) -> list[str]:
    """Extract and split identifiers from free-form text into search terms."""
    tokens = _IDENT_TOKEN_RE.findall(text)
    terms: list[str] = []
    seen: set[str] = set()

    existing = {t.lower() for t in tokens}
    for tok in tokens:
        parts = _split_identifier(tok)
        if len(parts) <= 1 and "_" not in tok and tok.lower() not in _ABBREV_MAP:
            continue

        for p in parts or [tok.lower()]:
            expanded = _ABBREV_MAP.get(p)
            if expanded and expanded not in seen and expanded not in existing:
                seen.add(expanded)
                terms.append(expanded)
                if len(terms) >= max_terms:
                    return terms

            if p in seen or p in existing:
                continue
            seen.add(p)
            terms.append(p)
            if len(terms) >= max_terms:
                return terms

    return terms


def _term_matches_query(term: str, query_words: set[str]) -> bool:
    """Heuristic filter: keep expansion terms that lexically match query words."""
    parts = _split_identifier(term)
    if not parts:
        parts = [term.lower()]

    for p in parts:
        if len(p) < 3:
            continue
        for qw in query_words:
            if qw == p:
                return True
            if qw.startswith(p) or p.startswith(qw):
                return True
    return False


class QueryExpander:
    """Two-hop query expansion: VecExpand → FTS symbol neighbor discovery."""

    def __init__(
        self,
        fts: FTSEngine,
        embedder: BaseEmbedder,
        config: Config,
    ) -> None:
        self._fts = fts
        self._embedder = embedder
        self._config = config
        self._names: list[str] | None = None
        self._kinds: list[str] | None = None
        self._is_public: list[bool] | None = None
        self._vocab_vecs: np.ndarray | None = None
        self._init_lock = threading.Lock()

    def _ensure_vocab(self) -> None:
        """Lazy-build and embed the symbol vocabulary (thread-safe)."""
        if self._vocab_vecs is not None:
            return
        with self._init_lock:
            if self._vocab_vecs is not None:
                return
            self._build_vocab()

    def _build_vocab(self) -> None:
        """Extract symbols + file stems, embed them as vocabulary vectors."""
        names: list[str] = []
        embed_texts: list[str] = []
        kinds: list[str] = []
        is_public: list[bool] = []

        # Symbols from DB
        rows = self._fts._conn.execute(
            "SELECT DISTINCT name, kind FROM symbols WHERE length(name) > 2"
        ).fetchall()
        for name, kind in rows:
            names.append(name)
            extra = _split_identifier(name)
            extras = extra + [_ABBREV_MAP[t] for t in extra if t in _ABBREV_MAP]
            embed_texts.append(f"{name} {' '.join(extras)}".strip() if extras else name)
            kinds.append(kind or "")
            is_public.append(not name.startswith("_"))

        # File stem tokens
        seen = {n.lower() for n in names}
        path_rows = self._fts._conn.execute(
            "SELECT DISTINCT path FROM docs_meta"
        ).fetchall()
        for r in path_rows:
            parts = r[0].replace("\\", "/").split("/")
            for part in parts:
                stem = part.rsplit(".", 1)[0] if "." in part else part
                if len(stem) > 2 and stem.lower() not in seen:
                    seen.add(stem.lower())
                    names.append(stem)
                    extra = _split_identifier(stem)
                    extras = extra + [_ABBREV_MAP[t] for t in extra if t in _ABBREV_MAP]
                    embed_texts.append(f"{stem} {' '.join(extras)}".strip() if extras else stem)
                    kinds.append("file")
                    is_public.append(True)

        if not names:
            self._names = []
            self._kinds = []
            self._is_public = []
            self._vocab_vecs = np.empty((0, 1), dtype=np.float32)
            return

        # Batch embed
        vecs = self._embedder.embed_batch(embed_texts)
        arr = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr /= norms

        self._names = names
        self._kinds = kinds
        self._is_public = is_public
        self._vocab_vecs = arr
        _log.info("QueryExpander: vocabulary built with %d terms", len(names))

    def expand(self, query: str) -> str:
        """Expand a query if it's natural language. Returns expanded query string.

        For CODE_SYMBOL queries, returns the original query unchanged.
        """
        word_count = len(query.strip().split())
        intent = detect_query_intent(query)
        # Long natural-language issues may contain code blocks and still benefit from expansion.
        if intent == QueryIntent.CODE_SYMBOL and word_count < 20:
            return query

        # Pre-expand: split identifiers and apply abbreviation mapping.
        pre_terms = _split_identifiers(query)
        embed_query = f"{query} {' '.join(pre_terms)}".strip() if pre_terms else query

        self._ensure_vocab()
        if self._vocab_vecs is None or len(self._vocab_vecs) == 0:
            return query

        cfg = self._config
        top_k = cfg.expansion_top_k
        threshold = cfg.expansion_threshold

        # First hop: find nearest symbols by cosine similarity
        first_hop = self._vec_expand(embed_query, top_k, threshold)
        # Short keyword-style queries are prone to noisy expansion; filter + skip 2nd hop.
        if first_hop and word_count < 8:
            query_words = {w.lower() for w in re.findall(r"[A-Za-z]+", query)}
            first_hop = [t for t in first_hop if _term_matches_query(t, query_words)]
        if not first_hop:
            if pre_terms:
                return f"{query} {' '.join(pre_terms)}"
            return query

        # Second hop: find chunk neighbors of first-hop symbols
        second_hop: list[str] = []
        if word_count >= 8:
            second_hop = self._neighbor_expand(embed_query, first_hop)

        all_terms = pre_terms + first_hop + second_hop
        if all_terms:
            expanded = f"{query} {' '.join(all_terms)}"
            _log.debug("Query expanded: '%s' -> '%s'", query, expanded)
            return expanded
        return query

    def _vec_expand(
        self, query: str, top_k: int, threshold: float
    ) -> list[str]:
        """First hop: cosine similarity between query and symbol vocabulary."""
        qvec = self._embedder.embed_single(query).astype(np.float32)
        qnorm = np.linalg.norm(qvec)
        if qnorm > 0:
            qvec /= qnorm

        scores = self._vocab_vecs @ qvec
        order = np.argsort(scores)[::-1]

        query_lower = query.lower()
        terms: list[str] = []
        for i in order:
            if scores[i] < threshold:
                break
            name = self._names[i]
            if name.lower() in query_lower:
                continue
            # Prefer public symbols; only include private if high similarity
            if not self._is_public[i] and scores[i] < threshold + 0.1:
                continue
            terms.append(name)
            if len(terms) >= top_k:
                break
        return terms

    def _neighbor_expand(
        self, query: str, first_hop: list[str], max_neighbors: int = 5
    ) -> list[str]:
        """Second hop: find symbols co-located in the same chunks."""
        # Find chunk IDs containing first-hop symbols
        all_chunk_ids: set[int] = set()
        for term in first_hop[:5]:
            rows = self._fts._conn.execute(
                "SELECT DISTINCT chunk_id FROM symbols WHERE name = ?",
                (term,),
            ).fetchall()
            for r in rows:
                all_chunk_ids.add(r[0])

        if not all_chunk_ids:
            return []

        # Get symbols from those chunks
        chunk_list = list(all_chunk_ids)[:30]
        placeholders = ",".join("?" for _ in chunk_list)
        rows = self._fts._conn.execute(
            f"SELECT DISTINCT name, kind FROM symbols "
            f"WHERE chunk_id IN ({placeholders}) AND length(name) > 2",
            chunk_list,
        ).fetchall()

        query_lower = query.lower()
        seen = {t.lower() for t in first_hop}
        neighbors: list[str] = []
        for name, kind in rows:
            nl = name.lower()
            if nl in seen or nl in query_lower:
                continue
            if kind in ("class", "function") or not name.startswith("_"):
                neighbors.append(name)
                seen.add(nl)
                if len(neighbors) >= max_neighbors:
                    break
        return neighbors
