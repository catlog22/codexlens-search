from __future__ import annotations

import logging
from collections import defaultdict

from .fts import FTSEngine

_log = logging.getLogger(__name__)

_KIND_WEIGHT: dict[str, float] = {
    "import": 1.0,
    "call": 1.5,
    "inherit": 0.9,
    "type_ref": 0.3,
}

_DIR_WEIGHT: dict[str, float] = {
    "backward": 1.3,
    "forward": 0.6,
}


class GraphSearcher:
    """Search code graph using symbol_refs edges from FTSEngine."""

    def __init__(
        self,
        fts: FTSEngine,
        expand_hops: int = 0,
        *,
        kind_weights: dict[str, float] | None = None,
        dir_weights: dict[str, float] | None = None,
    ) -> None:
        self._fts = fts
        self._expand_hops = expand_hops
        self._kind_weight = dict(_KIND_WEIGHT)
        if kind_weights:
            self._kind_weight.update(kind_weights)
        self._dir_weight = dict(_DIR_WEIGHT)
        if dir_weights:
            self._dir_weight.update(dir_weights)

    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        """Find chunks related to query via symbol reference graph.

        Uses exact symbol name matching as seed discovery.
        Returns list of (chunk_id, score) sorted by score descending.
        """
        seed_symbols = self._find_seed_symbols(query)
        return self._traverse(seed_symbols, top_k)

    def search_from_chunks(
        self, chunk_ids: list[int], top_k: int = 50,
    ) -> list[tuple[int, float]]:
        """Find related chunks by extracting symbols from given chunks and traversing the graph.

        This is the primary entry point for natural language queries:
        vector/FTS results provide relevant chunk_ids, and the graph
        discovers structurally related code (callers, callees, imports).

        Returns list of (chunk_id, score) sorted by score descending.
        Excludes the seed chunk_ids themselves to avoid duplicating
        results already found by vector/FTS.
        """
        seed_symbols: list[dict] = []
        seen_names: set[str] = set()
        for cid in chunk_ids:
            for sym in self._fts.get_symbols_by_chunk(cid):
                if sym["name"] not in seen_names:
                    seen_names.add(sym["name"])
                    seed_symbols.append(sym)
        exclude = set(chunk_ids)
        return self._traverse(seed_symbols, top_k, exclude_chunk_ids=exclude)

    def _traverse(
        self,
        seed_symbols: list[dict],
        top_k: int = 50,
        exclude_chunk_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Core graph traversal from seed symbols."""
        if not seed_symbols:
            return []

        chunk_scores: dict[int, float] = defaultdict(float)

        for sym in seed_symbols:
            sym_name = sym["name"]
            sym_chunk_id = sym["chunk_id"]

            # Forward refs: this symbol references others
            for ref in self._fts.get_refs_from(sym_name):
                score = self._score_edge(ref["ref_kind"], "forward", 1)
                target_chunk = self._resolve_ref_chunk(ref, direction="forward")
                if target_chunk is not None and target_chunk != sym_chunk_id:
                    chunk_scores[target_chunk] += score

            # Backward refs: others reference this symbol
            for ref in self._fts.get_refs_to(sym_name):
                score = self._score_edge(ref["ref_kind"], "backward", 1)
                source_chunk = self._resolve_ref_chunk(ref, direction="backward")
                if source_chunk is not None and source_chunk != sym_chunk_id:
                    chunk_scores[source_chunk] += score

            # Include seed chunk itself with a baseline score
            if sym_chunk_id is not None:
                chunk_scores[sym_chunk_id] += 1.0

        # Optional 1-hop BFS expansion
        if self._expand_hops > 0 and chunk_scores:
            seed_ids = set(chunk_scores.keys())
            expanded = self._expand_one_hop(seed_ids)
            for cid, exp_score in expanded.items():
                if cid not in seed_ids:
                    chunk_scores[cid] += exp_score

        # Exclude seed chunks if requested (avoid duplicating vector/FTS results)
        if exclude_chunk_ids:
            for cid in exclude_chunk_ids:
                chunk_scores.pop(cid, None)

        # Sort by score descending, take top_k
        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def _score_edge(self, ref_kind: str, direction: str, distance: int) -> float:
        """Score a single graph edge based on kind, direction, and distance."""
        kw = self._kind_weight.get(ref_kind, 0.3)
        dw = self._dir_weight.get(direction, 0.6)
        return kw * dw * (1.0 / distance)

    def _find_seed_symbols(self, query: str) -> list[dict]:
        """Find symbols matching query name as seed nodes."""
        # Try exact match first
        symbols = self._fts.get_symbols_by_name(query)
        if symbols:
            return symbols

        # Try individual tokens for multi-word queries
        tokens = query.strip().split()
        if len(tokens) > 1:
            all_syms: list[dict] = []
            for token in tokens:
                if len(token) >= 2:
                    syms = self._fts.get_symbols_by_name(token)
                    all_syms.extend(syms)
            return all_syms

        return []

    def _resolve_ref_chunk(self, ref: dict, direction: str) -> int | None:
        """Resolve a reference to a chunk_id.

        For forward refs, find the chunk of the target symbol (to_name).
        For backward refs, find the chunk of the source symbol (from_name).
        """
        if direction == "forward":
            # Target: look up to_symbol_id -> get chunk_id from symbols
            to_sym_id = ref.get("to_symbol_id")
            if to_sym_id is not None:
                syms = self._fts.get_symbols_by_name(ref["to_name"])
                for s in syms:
                    if s["id"] == to_sym_id:
                        return s["chunk_id"]
            # Fallback: just look up by name
            syms = self._fts.get_symbols_by_name(ref["to_name"])
            if syms:
                return syms[0]["chunk_id"]
        else:
            # Source: look up from_symbol_id -> get chunk_id
            from_sym_id = ref.get("from_symbol_id")
            if from_sym_id is not None:
                syms = self._fts.get_symbols_by_name(ref["from_name"])
                for s in syms:
                    if s["id"] == from_sym_id:
                        return s["chunk_id"]
            syms = self._fts.get_symbols_by_name(ref["from_name"])
            if syms:
                return syms[0]["chunk_id"]
        return None

    def _expand_one_hop(self, seed_ids: set[int]) -> dict[int, float]:
        """BFS expand one hop from seed chunk IDs with distance decay.

        Finds symbols in seed chunks, then follows their references
        to discover neighbor chunks at distance=2.
        """
        expanded: dict[int, float] = defaultdict(float)
        decay = 0.5  # distance 2 decay factor

        for chunk_id in seed_ids:
            syms = self._fts.get_symbols_by_chunk(chunk_id)
            for sym in syms:
                sym_name = sym["name"]

                for ref in self._fts.get_refs_from(sym_name):
                    score = self._score_edge(ref["ref_kind"], "forward", 2) * decay
                    target = self._resolve_ref_chunk(ref, direction="forward")
                    if target is not None:
                        expanded[target] += score

                for ref in self._fts.get_refs_to(sym_name):
                    score = self._score_edge(ref["ref_kind"], "backward", 2) * decay
                    source = self._resolve_ref_chunk(ref, direction="backward")
                    if source is not None:
                        expanded[source] += score

        return dict(expanded)
