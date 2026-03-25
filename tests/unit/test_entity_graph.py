from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from codexlens_search.config import Config
from codexlens_search.core.entity import EntityId, EntityKind
from codexlens_search.core.entity_graph import EntityGraph
from codexlens_search.search.fts import FTSEngine
from codexlens_search.search.pipeline import SearchPipeline


def _file_entity(path: str) -> EntityId:
    return EntityId(
        file_path=path,
        symbol_name="",
        kind=EntityKind.FILE,
        start_line=0,
        end_line=0,
    )


def test_fts_entity_edges_upsert_accumulates_weight() -> None:
    fts = FTSEngine(":memory:")
    a = _file_entity("a.py")
    b = _file_entity("b.py")

    fts.add_entity_edges(
        [
            (a.to_key(), b.to_key(), "import", 1.0),
            (a.to_key(), b.to_key(), "import", 2.0),
        ]
    )
    fts.flush()

    row = fts._conn.execute(
        "SELECT weight FROM entity_edges WHERE from_entity = ? AND to_entity = ? AND edge_kind = ?",
        (a.to_key(), b.to_key(), "import"),
    ).fetchone()
    assert row is not None
    assert float(row[0]) == 3.0


def test_fts_delete_by_path_removes_outgoing_edges_only() -> None:
    fts = FTSEngine(":memory:")
    a = _file_entity("a.py")
    b = _file_entity("b.py")

    fts.add_entity_edges(
        [
            (a.to_key(), b.to_key(), "call", 1.0),
            (b.to_key(), a.to_key(), "call", 1.0),
        ]
    )
    fts.flush()

    fts.delete_by_path("a.py")

    rows = fts._conn.execute(
        "SELECT from_entity, to_entity FROM entity_edges ORDER BY from_entity, to_entity"
    ).fetchall()
    assert (a.to_key(), b.to_key()) not in rows
    assert (b.to_key(), a.to_key()) in rows


def test_entity_graph_traverse_lazy_load_fallback_backend() -> None:
    fts = FTSEngine(":memory:")
    a = _file_entity("a.py")
    b = _file_entity("b.py")

    fts.add_entity_edges([(a.to_key(), b.to_key(), "import", 1.0)])
    fts.flush()

    graph = EntityGraph(fts, depth=1, backend="networkx", enabled=True)
    out = graph.traverse(a, depth=1)
    assert b in out


def test_entity_graph_expand_from_chunks_returns_related_chunk_ids() -> None:
    fts = FTSEngine(":memory:")
    fts.add_documents(
        [
            (1, "a.py", "alpha"),
            (2, "b.py", "beta"),
        ]
    )
    a = _file_entity("a.py")
    b = _file_entity("b.py")
    fts.add_entity_edges([(a.to_key(), b.to_key(), "import", 1.0)])
    fts.flush()

    graph = EntityGraph(fts, depth=1, backend="auto", enabled=True)
    results = graph.expand_from_chunks([1], depth=1, top_k=10)
    assert results and results[0][0] == 2


def test_search_pipeline_thorough_includes_entity_fusion_source() -> None:
    fts = FTSEngine(":memory:")
    fts.add_documents([(1, "a.py", "alpha"), (2, "b.py", "beta")])
    a = _file_entity("a.py")
    b = _file_entity("b.py")
    fts.add_entity_edges([(a.to_key(), b.to_key(), "import", 1.0)])
    fts.flush()

    cfg = Config.small()
    cfg.entity_graph_enabled = True
    cfg.entity_graph_depth = 1
    cfg.symbol_search_enabled = False

    embedder = MagicMock()
    embedder.embed_single.return_value = np.random.randn(cfg.embed_dim).astype(np.float32)

    binary_store = MagicMock()
    ann_index = MagicMock()

    reranker = MagicMock()
    reranker.score_pairs.side_effect = lambda q, contents: [1.0 for _ in contents]

    entity_graph = EntityGraph(
        fts,
        depth=cfg.entity_graph_depth,
        backend=cfg.entity_graph_backend,
        enabled=True,
    )

    pipeline = SearchPipeline(
        embedder=embedder,
        binary_store=binary_store,
        ann_index=ann_index,
        reranker=reranker,
        fts=fts,
        config=cfg,
        entity_graph=entity_graph,
    )

    pipeline._vector_search = MagicMock(return_value=[(1, 0.1)])
    pipeline._fts_search = MagicMock(return_value=([], []))

    def _fake_rrf(results, weights=None, k=60):
        assert "entity" in results
        assert any(doc_id == 2 for doc_id, _ in results["entity"])
        return [(2, 1.0)]

    with patch("codexlens_search.search.pipeline.reciprocal_rank_fusion", side_effect=_fake_rrf):
        out = pipeline.search("find something", quality="thorough", top_k=5)

    assert out and out[0].id == 2

