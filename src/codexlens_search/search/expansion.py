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
import threading

import numpy as np

from ..config import Config
from ..embed import BaseEmbedder
from .fts import FTSEngine
from .fusion import QueryIntent, detect_query_intent

_log = logging.getLogger(__name__)


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
        kinds: list[str] = []
        is_public: list[bool] = []

        # Symbols from DB
        rows = self._fts._conn.execute(
            "SELECT DISTINCT name, kind FROM symbols WHERE length(name) > 2"
        ).fetchall()
        for name, kind in rows:
            names.append(name)
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
                    kinds.append("file")
                    is_public.append(True)

        if not names:
            self._names = []
            self._kinds = []
            self._is_public = []
            self._vocab_vecs = np.empty((0, 1), dtype=np.float32)
            return

        # Batch embed
        vecs = self._embedder.embed_batch(names)
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
        intent = detect_query_intent(query)
        if intent == QueryIntent.CODE_SYMBOL:
            return query

        self._ensure_vocab()
        if self._vocab_vecs is None or len(self._vocab_vecs) == 0:
            return query

        cfg = self._config
        top_k = cfg.expansion_top_k
        threshold = cfg.expansion_threshold

        # First hop: find nearest symbols by cosine similarity
        first_hop = self._vec_expand(query, top_k, threshold)
        if not first_hop:
            return query

        # Second hop: find chunk neighbors of first-hop symbols
        second_hop = self._neighbor_expand(query, first_hop)

        all_terms = first_hop + second_hop
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
