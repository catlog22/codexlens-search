"""Single index partition (shard) that owns FTS, binary, ANN, and metadata stores."""
from __future__ import annotations

import logging
from pathlib import Path

from codexlens_search.config import Config
from codexlens_search.core.base import BaseANNIndex, BaseBinaryIndex
from codexlens_search.embed.base import BaseEmbedder
from codexlens_search.indexing.metadata import MetadataStore
from codexlens_search.indexing.pipeline import IndexingPipeline, IndexStats
from codexlens_search.rerank import BaseReranker
from codexlens_search.search.fts import FTSEngine
from codexlens_search.search.pipeline import SearchPipeline, SearchResult

logger = logging.getLogger(__name__)


class Shard:
    """A complete index partition with its own FTS, binary, ANN, and metadata stores.

    Components are lazy-loaded on first access and can be explicitly unloaded
    to release memory. The embedder and reranker are shared across shards
    (passed in from ShardManager) since they are expensive to instantiate.
    """

    def __init__(
        self,
        shard_id: int,
        db_path: str | Path,
        config: Config,
    ) -> None:
        self._shard_id = shard_id
        self._shard_dir = Path(db_path).resolve() / f"shard_{shard_id}"
        self._config = config

        # Lazy-loaded components (created on _ensure_loaded)
        self._fts: FTSEngine | None = None
        self._binary_store: BaseBinaryIndex | None = None
        self._ann_index: BaseANNIndex | None = None
        self._metadata: MetadataStore | None = None
        self._indexing: IndexingPipeline | None = None
        self._search: SearchPipeline | None = None
        self._loaded = False

    @property
    def shard_id(self) -> int:
        return self._shard_id

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _ensure_loaded(
        self,
        embedder: BaseEmbedder,
        reranker: BaseReranker,
    ) -> None:
        """Lazy-create all per-shard components if not yet loaded."""
        if self._loaded:
            return

        from codexlens_search.core.factory import create_ann_index, create_binary_index

        self._shard_dir.mkdir(parents=True, exist_ok=True)

        self._fts = FTSEngine(self._shard_dir / "fts.db")
        self._binary_store = create_binary_index(
            self._shard_dir, self._config.embed_dim, self._config
        )
        self._ann_index = create_ann_index(
            self._shard_dir, self._config.embed_dim, self._config
        )
        self._metadata = MetadataStore(self._shard_dir / "metadata.db")

        self._indexing = IndexingPipeline(
            embedder=embedder,
            binary_store=self._binary_store,
            ann_index=self._ann_index,
            fts=self._fts,
            config=self._config,
            metadata=self._metadata,
        )

        self._search = SearchPipeline(
            embedder=embedder,
            binary_store=self._binary_store,
            ann_index=self._ann_index,
            reranker=reranker,
            fts=self._fts,
            config=self._config,
            metadata_store=self._metadata,
        )

        self._loaded = True
        logger.debug("Shard %d loaded from %s", self._shard_id, self._shard_dir)

    def unload(self) -> None:
        """Release memory by closing connections and dropping references."""
        if not self._loaded:
            return

        if self._metadata is not None:
            self._metadata.close()

        self._fts = None
        self._binary_store = None
        self._ann_index = None
        self._metadata = None
        self._indexing = None
        self._search = None
        self._loaded = False
        logger.debug("Shard %d unloaded", self._shard_id)

    def load(
        self,
        embedder: BaseEmbedder,
        reranker: BaseReranker,
    ) -> None:
        """Explicitly load shard components."""
        self._ensure_loaded(embedder, reranker)

    def save(self) -> None:
        """Persist binary and ANN indexes to disk."""
        if not self._loaded:
            return
        if self._binary_store is not None:
            self._binary_store.save()
        if self._ann_index is not None:
            self._ann_index.save()

    def search(
        self,
        query: str,
        embedder: BaseEmbedder,
        reranker: BaseReranker,
        quality: str | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Search this shard's index.

        Args:
            query: Search query string.
            embedder: Shared embedder instance.
            reranker: Shared reranker instance.
            quality: Search quality tier.
            top_k: Maximum results to return.

        Returns:
            List of SearchResult from this shard.
        """
        self._ensure_loaded(embedder, reranker)
        assert self._search is not None
        return self._search.search(query, top_k=top_k, quality=quality)

    def sync(
        self,
        files: list[Path],
        root: Path | None,
        embedder: BaseEmbedder,
        reranker: BaseReranker,
        **kwargs: object,
    ) -> IndexStats:
        """Sync this shard's index with the given files.

        Args:
            files: Files that belong to this shard.
            root: Root directory for relative paths.
            embedder: Shared embedder instance.
            reranker: Shared reranker instance.
            **kwargs: Forwarded to IndexingPipeline.sync().

        Returns:
            IndexStats for this shard's sync operation.
        """
        self._ensure_loaded(embedder, reranker)
        assert self._indexing is not None
        return self._indexing.sync(files, root=root, **kwargs)
