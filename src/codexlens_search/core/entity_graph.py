from __future__ import annotations

import logging
import threading
from collections import defaultdict, deque

from codexlens_search.core.entity import EntityId, EntityKind

try:
    import networkx as _nx  # type: ignore

    _HAS_NETWORKX = True
except ImportError:  # pragma: no cover
    _nx = None
    _HAS_NETWORKX = False

_log = logging.getLogger(__name__)


def _safe_entity_from_key(key: str) -> EntityId | None:
    try:
        return EntityId.from_key(key)
    except Exception:
        return None


def _entity_for_file(path: str) -> EntityId:
    return EntityId(
        file_path=path,
        symbol_name="",
        kind=EntityKind.FILE,
        start_line=0,
        end_line=0,
    )


class EntityGraph:
    """Entity dependency graph backed by an optional NetworkX DiGraph.

    Stores edges between EntityId nodes. When instantiated with an FTSEngine
    that has an entity_edges table, the graph can lazy-load edges from disk.
    """

    def __init__(
        self,
        fts,
        *,
        depth: int = 2,
        backend: str = "auto",
        enabled: bool = True,
    ) -> None:
        self._fts = fts
        self._depth = max(0, int(depth))
        self._enabled = bool(enabled)

        backend_norm = (backend or "auto").strip().lower()
        use_nx = backend_norm in ("auto", "networkx", "nx") and _HAS_NETWORKX
        if backend_norm in ("networkx", "nx") and not _HAS_NETWORKX:
            _log.warning("EntityGraph backend=networkx requested but not available; falling back to pure python")
        self._use_networkx = use_nx

        self._nx = _nx.DiGraph() if self._use_networkx else None
        self._adj: dict[EntityId, dict[EntityId, float]] = defaultdict(dict)

        self._loaded = False
        self._init_lock = threading.Lock()

        self._chunk_path_cache: dict[int, str] = {}
        self._chunk_symbols_cache: dict[int, list[dict]] = {}
        self._chunks_by_path_cache: dict[str, list[int]] = {}

    def add_entity(self, entity_id: EntityId) -> None:
        if self._use_networkx and self._nx is not None:
            self._nx.add_node(entity_id)
        else:
            self._adj.setdefault(entity_id, {})
        self._loaded = True

    def add_edge(
        self,
        from_id: EntityId,
        to_id: EntityId,
        kind: str,
        *,
        weight: float = 1.0,
        bidirectional: bool = True,
    ) -> None:
        if not self._enabled:
            return
        w = float(weight)
        self.add_entity(from_id)
        self.add_entity(to_id)

        self._add_directed_edge(from_id, to_id, kind, w)
        if bidirectional and from_id != to_id:
            self._add_directed_edge(to_id, from_id, kind, w)

    def _add_directed_edge(self, from_id: EntityId, to_id: EntityId, kind: str, weight: float) -> None:
        if self._use_networkx and self._nx is not None:
            if self._nx.has_edge(from_id, to_id):
                self._nx[from_id][to_id]["weight"] = float(self._nx[from_id][to_id].get("weight", 0.0)) + weight
                self._nx[from_id][to_id]["kind"] = kind
            else:
                self._nx.add_edge(from_id, to_id, kind=kind, weight=weight)
        else:
            prev = self._adj[from_id].get(to_id, 0.0)
            self._adj[from_id][to_id] = prev + weight

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._init_lock:
            if self._loaded:
                return
            self._load_from_db()
            self._loaded = True

    def _load_from_db(self) -> None:
        if self._fts is None:
            return
        conn = getattr(self._fts, "_conn", None)
        if conn is None:
            return
        try:
            rows = conn.execute(
                "SELECT from_entity, to_entity, edge_kind, weight FROM entity_edges"
            ).fetchall()
        except Exception:
            return

        for from_key, to_key, edge_kind, weight in rows:
            from_id = _safe_entity_from_key(from_key)
            to_id = _safe_entity_from_key(to_key)
            if from_id is None or to_id is None:
                continue
            self._add_directed_edge(from_id, to_id, str(edge_kind), float(weight or 1.0))

    def traverse(self, entity_id: EntityId, depth: int | None = None) -> list[EntityId]:
        if not self._enabled:
            return []
        self._ensure_loaded()
        max_depth = self._depth if depth is None else max(0, int(depth))
        if max_depth <= 0:
            return []

        out: list[EntityId] = []
        seen: set[EntityId] = {entity_id}
        q: deque[tuple[EntityId, int]] = deque([(entity_id, 0)])

        while q:
            current, d = q.popleft()
            if d >= max_depth:
                continue
            for neighbor, _w in self._neighbors(current):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                out.append(neighbor)
                q.append((neighbor, d + 1))

        return out

    def expand_from_chunks(
        self,
        chunk_ids: list[int],
        *,
        depth: int | None = None,
        top_k: int = 50,
    ) -> list[tuple[int, float]]:
        if not self._enabled or not chunk_ids:
            return []
        seeds = self._seed_entities_from_chunks(chunk_ids)
        exclude = set(int(cid) for cid in chunk_ids)
        return self.search(seeds, top_k=top_k, depth=depth, exclude_chunk_ids=exclude)

    def search(
        self,
        seed_entities: list[EntityId],
        top_k: int = 50,
        *,
        depth: int | None = None,
        exclude_chunk_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        if not self._enabled or not seed_entities:
            return []
        self._ensure_loaded()

        max_depth = self._depth if depth is None else max(0, int(depth))
        if max_depth <= 0:
            return []

        seed_set: list[EntityId] = []
        seen_seed: set[EntityId] = set()
        for e in seed_entities:
            if e in seen_seed:
                continue
            seen_seed.add(e)
            seed_set.append(e)

        entity_scores: dict[EntityId, float] = defaultdict(float)
        q: deque[tuple[EntityId, int, float]] = deque()
        best_depth: dict[EntityId, int] = {}
        for seed in seed_set:
            q.append((seed, 0, 1.0))
            best_depth[seed] = 0

        max_visits = 2000
        visits = 0

        while q and visits < max_visits:
            current, d, score = q.popleft()
            visits += 1
            if d >= max_depth:
                continue

            next_d = d + 1
            decay = 1.0 / next_d
            for neighbor, w in self._neighbors(current):
                if neighbor in seed_set:
                    continue
                step_score = score * float(w) * decay
                if step_score <= 0.0:
                    continue
                entity_scores[neighbor] += step_score

                prev_d = best_depth.get(neighbor)
                if prev_d is None or next_d < prev_d:
                    best_depth[neighbor] = next_d
                    q.append((neighbor, next_d, step_score))

        chunk_scores: dict[int, float] = defaultdict(float)
        exclude = exclude_chunk_ids or set()
        for ent, score in entity_scores.items():
            for cid in self._chunks_for_entity(ent):
                if cid in exclude:
                    continue
                chunk_scores[cid] += float(score)

        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[: max(0, int(top_k))]

    def _neighbors(self, entity_id: EntityId) -> list[tuple[EntityId, float]]:
        if self._use_networkx and self._nx is not None and self._nx.has_node(entity_id):
            out: list[tuple[EntityId, float]] = []
            for nbr in self._nx.successors(entity_id):
                data = self._nx.get_edge_data(entity_id, nbr) or {}
                out.append((nbr, float(data.get("weight", 1.0))))
            return out
        return list(self._adj.get(entity_id, {}).items())

    def _seed_entities_from_chunks(self, chunk_ids: list[int], max_seeds: int = 25) -> list[EntityId]:
        seeds: list[EntityId] = []
        seen: set[EntityId] = set()

        for cid in chunk_ids:
            path = self._chunk_path(int(cid))
            if path:
                file_ent = _entity_for_file(path)
                if file_ent not in seen:
                    seen.add(file_ent)
                    seeds.append(file_ent)

            for sym in self._chunk_symbols(int(cid)):
                ent = self._entity_from_symbol_row(path, sym)
                if ent is None or ent in seen:
                    continue
                seen.add(ent)
                seeds.append(ent)
                if len(seeds) >= max_seeds:
                    return seeds

        return seeds

    def _entity_from_symbol_row(self, path: str, sym: dict) -> EntityId | None:
        kind = str(sym.get("kind", "")).lower()
        if kind == "class":
            ent_kind = EntityKind.CLASS
        elif kind == "function":
            ent_kind = EntityKind.FUNCTION
        elif kind == "method":
            ent_kind = EntityKind.METHOD
        elif kind == "module":
            ent_kind = EntityKind.MODULE
        else:
            return None
        return EntityId(
            file_path=path,
            symbol_name=str(sym.get("name", "")),
            kind=ent_kind,
            start_line=int(sym.get("start_line", 0) or 0),
            end_line=int(sym.get("end_line", 0) or 0),
        )

    def _chunk_path(self, chunk_id: int) -> str:
        cached = self._chunk_path_cache.get(chunk_id)
        if cached is not None:
            return cached
        try:
            path = self._fts.get_doc_meta(chunk_id)[0]
        except Exception:
            path = ""
        self._chunk_path_cache[chunk_id] = path
        return path

    def _chunk_symbols(self, chunk_id: int) -> list[dict]:
        cached = self._chunk_symbols_cache.get(chunk_id)
        if cached is not None:
            return cached
        try:
            syms = list(self._fts.get_symbols_by_chunk(chunk_id))
        except Exception:
            syms = []
        self._chunk_symbols_cache[chunk_id] = syms
        return syms

    def _chunks_for_entity(self, ent: EntityId) -> list[int]:
        if ent.kind == EntityKind.FILE:
            return self._chunks_for_path(ent.file_path)
        return self._chunks_for_symbol_entity(ent)

    def _chunks_for_path(self, path: str) -> list[int]:
        cached = self._chunks_by_path_cache.get(path)
        if cached is not None:
            return cached
        try:
            ids = [int(x) for x in self._fts.get_chunk_ids_by_path(path)]
        except Exception:
            ids = []
        self._chunks_by_path_cache[path] = ids
        return ids

    def _chunks_for_symbol_entity(self, ent: EntityId) -> list[int]:
        conn = getattr(self._fts, "_conn", None)
        if conn is not None:
            try:
                rows = conn.execute(
                    "SELECT chunk_id FROM symbols WHERE name = ? AND kind = ? AND start_line = ? AND end_line = ?",
                    (ent.symbol_name, ent.kind.value, ent.start_line, ent.end_line),
                ).fetchall()
                chunk_ids = [int(r[0]) for r in rows if r and r[0] is not None]
                matched = [cid for cid in chunk_ids if self._chunk_path(cid) == ent.file_path]
                if matched:
                    return matched[:3]
            except Exception:
                pass

        for cid in self._chunks_for_path(ent.file_path):
            for sym in self._chunk_symbols(cid):
                if (
                    str(sym.get("name", "")) == ent.symbol_name
                    and str(sym.get("kind", "")).lower() == ent.kind.value
                    and int(sym.get("start_line", 0) or 0) == ent.start_line
                    and int(sym.get("end_line", 0) or 0) == ent.end_line
                ):
                    return [int(cid)]
        return []

    def dumps_entity(self, ent: EntityId) -> str:
        return ent.to_key()

    def loads_entity(self, key: str) -> EntityId | None:
        return _safe_entity_from_key(key)

    def dumps_edge(self, from_id: EntityId, to_id: EntityId, kind: str, weight: float) -> tuple[str, str, str, float]:
        return from_id.to_key(), to_id.to_key(), str(kind), float(weight)
