"""ShardManager: manages multiple Shard instances with LRU eviction."""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from codexlens_search.config import Config
from codexlens_search.core.shard import Shard
from codexlens_search.embed.base import BaseEmbedder
from codexlens_search.indexing.pipeline import IndexStats
from codexlens_search.rerank import BaseReranker
from codexlens_search.search.fusion import reciprocal_rank_fusion
from codexlens_search.search.pipeline import SearchResult

logger = logging.getLogger(__name__)


class ShardManager:
    """Manages multiple Shard instances with hash-based file routing and LRU eviction.

    Files are deterministically routed to shards via hash(path) % num_shards.
    Search queries all shards in parallel and merges results via RRF fusion.
    At most max_loaded_shards are kept in memory; least-recently-used shards
    are unloaded when the limit is exceeded.
    """

    def __init__(
        self,
        num_shards: int,
        db_path: str | Path,
        config: Config,
        embedder: BaseEmbedder,
        reranker: BaseReranker,
    ) -> None:
        if num_shards < 1:
            raise ValueError("num_shards must be >= 1")

        self._num_shards = num_shards
        self._db_path = Path(db_path).resolve()
        self._config = config
        self._embedder = embedder
        self._reranker = reranker
        self._max_loaded = config.max_loaded_shards

        # Create all Shard objects (lazy-loaded, no I/O yet)
        self._shards: dict[int, Shard] = {
            i: Shard(i, self._db_path, config)
            for i in range(num_shards)
        }

        # LRU tracking: keys are shard_ids, most-recently-used at end
        self._loaded_order: OrderedDict[int, None] = OrderedDict()
        self._lru_lock = threading.Lock()

    @property
    def num_shards(self) -> int:
        return self._num_shards

    def close(self) -> None:
        """Unload all shards and release resources."""
        with self._lru_lock:
            for shard in self._shards.values():
                if shard.is_loaded:
                    shard.unload()
            self._loaded_order.clear()

    def route_file(self, path: str) -> int:
        """Deterministically route a file path to a shard ID.

        Uses hash(path) % num_shards for uniform distribution.
        """
        return hash(path) % self._num_shards

    def get_shard(self, shard_id: int) -> Shard:
        """Return the Shard instance for a given shard_id."""
        if shard_id not in self._shards:
            raise ValueError(
                f"Invalid shard_id {shard_id}, valid range: 0-{self._num_shards - 1}"
            )
        return self._shards[shard_id]

    def _ensure_loaded(self, shard_id: int) -> Shard:
        """Load a shard if needed, applying LRU eviction policy.

        Thread-safe: protects OrderedDict mutations with a lock.
        Returns the loaded Shard.
        """
        shard = self._shards[shard_id]

        with self._lru_lock:
            # Mark as most-recently-used
            if shard_id in self._loaded_order:
                self._loaded_order.move_to_end(shard_id)
            else:
                self._loaded_order[shard_id] = None

            # Load if not already loaded
            if not shard.is_loaded:
                shard.load(self._embedder, self._reranker)

            # Evict LRU shards if over limit
            while len(self._loaded_order) > self._max_loaded:
                evict_id, _ = self._loaded_order.popitem(last=False)
                evict_shard = self._shards[evict_id]
                if evict_shard.is_loaded:
                    logger.info("LRU evicting shard %d", evict_id)
                    evict_shard.unload()

        return shard

    def sync(
        self,
        files: list[Path],
        root: Path | None = None,
        **kwargs: object,
    ) -> IndexStats:
        """Sync index with files, routing each file to its shard.

        Groups files by shard via route_file(), then syncs each shard
        with its subset of files.

        Args:
            files: Current list of files to index.
            root: Root directory for relative paths.
            **kwargs: Forwarded to Shard.sync().

        Returns:
            Aggregated IndexStats across all shards.
        """
        # Group files by shard
        shard_files: dict[int, list[Path]] = {i: [] for i in range(self._num_shards)}
        for fpath in files:
            rel = str(fpath.relative_to(root)) if root else str(fpath)
            shard_id = self.route_file(rel)
            shard_files[shard_id].append(fpath)

        total_files = 0
        total_chunks = 0
        total_duration = 0.0

        for shard_id, shard_file_list in shard_files.items():
            if not shard_file_list:
                continue
            self._ensure_loaded(shard_id)
            shard = self._shards[shard_id]
            stats = shard.sync(
                shard_file_list,
                root=root,
                embedder=self._embedder,
                reranker=self._reranker,
                **kwargs,
            )
            total_files += stats.files_processed
            total_chunks += stats.chunks_created
            total_duration += stats.duration_seconds

        return IndexStats(
            files_processed=total_files,
            chunks_created=total_chunks,
            duration_seconds=round(total_duration, 2),
        )

    def search(
        self,
        query: str,
        quality: str | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Search all shards in parallel, merge results via RRF fusion.

        Each shard returns its own ranked results. Cross-shard merging
        uses reciprocal_rank_fusion with equal weights across shards.
        Per-shard top_k is increased to compensate for cross-shard dilution.

        Args:
            query: Search query string.
            quality: Search quality tier.
            top_k: Maximum final results to return.

        Returns:
            Merged list of SearchResult ordered by relevance.
        """
        cfg = self._config
        final_top_k = top_k if top_k is not None else cfg.reranker_top_k

        # Increase per-shard top_k to get enough candidates for cross-shard RRF
        per_shard_top_k = max(final_top_k, final_top_k * 2)

        # Load all shards for search
        for shard_id in range(self._num_shards):
            self._ensure_loaded(shard_id)

        # Parallel search across shards
        shard_results: dict[int, list[SearchResult]] = {}

        def _search_shard(sid: int) -> tuple[int, list[SearchResult]]:
            shard = self._shards[sid]
            results = shard.search(
                query,
                embedder=self._embedder,
                reranker=self._reranker,
                quality=quality,
                top_k=per_shard_top_k,
            )
            return sid, results

        with ThreadPoolExecutor(max_workers=min(self._num_shards, 4)) as pool:
            futures = [pool.submit(_search_shard, sid) for sid in range(self._num_shards)]
            for future in futures:
                try:
                    sid, results = future.result()
                    shard_results[sid] = results
                except Exception:
                    logger.warning("Shard search failed", exc_info=True)

        # If only one shard returned results, no merging needed
        non_empty = {k: v for k, v in shard_results.items() if v}
        if not non_empty:
            return []
        if len(non_empty) == 1:
            results = list(non_empty.values())[0]
            return results[:final_top_k]

        # Cross-shard RRF merge
        # Build ranked lists keyed by shard name, with (doc_id, score) tuples
        # Use a global result map to look up SearchResult by a unique key
        # Since doc_ids are shard-local, we need a composite key
        rrf_input: dict[str, list[tuple[int, float]]] = {}
        global_results: dict[int, SearchResult] = {}
        global_id = 0

        for sid, results in non_empty.items():
            ranked: list[tuple[int, float]] = []
            for r in results:
                global_results[global_id] = r
                ranked.append((global_id, r.score))
                global_id += 1
            rrf_input[f"shard_{sid}"] = ranked

        fused = reciprocal_rank_fusion(rrf_input, k=cfg.fusion_k)

        merged: list[SearchResult] = []
        for gid, fused_score in fused[:final_top_k]:
            result = global_results[gid]
            merged.append(SearchResult(
                id=result.id,
                path=result.path,
                score=fused_score,
                snippet=result.snippet,
                line=result.line,
                end_line=result.end_line,
                content=result.content,
            ))

        return merged
