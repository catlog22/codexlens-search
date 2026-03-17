"""Unit tests for BinaryStore and ANNIndex (no fastembed required)."""
from __future__ import annotations

import concurrent.futures
import tempfile
from pathlib import Path

import numpy as np
import pytest

from codexlens_search.config import Config
from codexlens_search.core import ANNIndex, BinaryStore


DIM = 32
RNG = np.random.default_rng(42)


def make_vectors(n: int, dim: int = DIM) -> np.ndarray:
    return RNG.standard_normal((n, dim)).astype(np.float32)


def make_ids(n: int, start: int = 0) -> np.ndarray:
    return np.arange(start, start + n, dtype=np.int64)


# ---------------------------------------------------------------------------
# BinaryStore tests
# ---------------------------------------------------------------------------


class TestBinaryStore:
    def test_binary_store_add_and_search(self, tmp_path: Path) -> None:
        cfg = Config.small()
        store = BinaryStore(tmp_path, DIM, cfg)
        vecs = make_vectors(10)
        ids = make_ids(10)
        store.add(ids, vecs)

        assert len(store) == 10

        top_k = 5
        ret_ids, ret_dists = store.coarse_search(vecs[0], top_k=top_k)
        assert ret_ids.shape == (top_k,)
        assert ret_dists.shape == (top_k,)
        # distances are non-negative integers
        assert (ret_dists >= 0).all()

    def test_binary_hamming_correctness(self, tmp_path: Path) -> None:
        cfg = Config.small()
        store = BinaryStore(tmp_path, DIM, cfg)
        vecs = make_vectors(20)
        ids = make_ids(20)
        store.add(ids, vecs)

        # Query with the exact stored vector; it must be the top-1 result
        query = vecs[7]
        ret_ids, ret_dists = store.coarse_search(query, top_k=1)
        assert ret_ids[0] == 7
        assert ret_dists[0] == 0  # Hamming distance to itself is 0

    def test_binary_store_persist(self, tmp_path: Path) -> None:
        cfg = Config.small()
        store = BinaryStore(tmp_path, DIM, cfg)
        vecs = make_vectors(15)
        ids = make_ids(15)
        store.add(ids, vecs)
        store.save()

        # Load into a fresh instance
        store2 = BinaryStore(tmp_path, DIM, cfg)
        assert len(store2) == 15

        query = vecs[3]
        ret_ids, ret_dists = store2.coarse_search(query, top_k=1)
        assert ret_ids[0] == 3
        assert ret_dists[0] == 0


# ---------------------------------------------------------------------------
# ANNIndex tests
# ---------------------------------------------------------------------------


class TestANNIndex:
    def test_ann_index_add_and_search(self, tmp_path: Path) -> None:
        cfg = Config.small()
        idx = ANNIndex(tmp_path, DIM, cfg)
        vecs = make_vectors(50)
        ids = make_ids(50)
        idx.add(ids, vecs)

        assert len(idx) == 50

        ret_ids, ret_dists = idx.fine_search(vecs[0], top_k=5)
        assert len(ret_ids) == 5
        assert len(ret_dists) == 5

    def test_ann_index_thread_safety(self, tmp_path: Path) -> None:
        cfg = Config.small()
        idx = ANNIndex(tmp_path, DIM, cfg)
        vecs = make_vectors(50)
        ids = make_ids(50)
        idx.add(ids, vecs)

        query = vecs[0]
        errors: list[Exception] = []

        def search() -> None:
            try:
                idx.fine_search(query, top_k=3)
            except Exception as exc:
                errors.append(exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(search) for _ in range(5)]
            concurrent.futures.wait(futures)

        assert errors == [], f"Thread safety errors: {errors}"

    def test_ann_index_save_load(self, tmp_path: Path) -> None:
        cfg = Config.small()
        idx = ANNIndex(tmp_path, DIM, cfg)
        vecs = make_vectors(30)
        ids = make_ids(30)
        idx.add(ids, vecs)
        idx.save()

        # Load into a fresh instance
        idx2 = ANNIndex(tmp_path, DIM, cfg)
        idx2.load()
        assert len(idx2) == 30

        ret_ids, ret_dists = idx2.fine_search(vecs[10], top_k=1)
        assert len(ret_ids) == 1
        assert ret_ids[0] == 10
