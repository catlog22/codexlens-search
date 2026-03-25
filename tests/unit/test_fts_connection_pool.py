"""Unit tests for FTSEngine thread-local connection pooling."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from codexlens_search.search.fts import FTSEngine


@pytest.fixture
def fts(tmp_path):
    engine = FTSEngine(str(tmp_path / "fts.db"))
    # Add sample documents for search tests
    engine.add_documents([
        (1, "src/main.py", "def main(): pass", 1, 5, "python"),
        (2, "src/utils.py", "def helper(): return True", 1, 3, "python"),
        (3, "src/config.py", "class Config: debug = False", 1, 4, "python"),
    ])
    engine.flush()
    return engine


class TestThreadLocalConnections:
    def test_main_thread_uses_primary_connection(self, fts):
        """The creating thread should use the primary connection."""
        conn = fts._get_conn()
        assert conn is fts._conn

    def test_worker_threads_get_separate_connections(self, fts):
        """Worker threads get their own connections to avoid concurrent access errors."""
        connections = {}
        barrier = threading.Barrier(3)

        def worker(idx):
            conn = fts._get_conn()
            connections[idx] = conn
            barrier.wait()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Worker threads should all get connections (distinct from each other)
        conn_ids = [id(c) for c in connections.values()]
        assert len(set(conn_ids)) == 3

    def test_same_thread_gets_same_connection(self, fts):
        """Repeated calls from the same thread return the same connection."""
        results = []

        def worker():
            c1 = fts._get_conn()
            c2 = fts._get_conn()
            results.append(c1 is c2)

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert results[0] is True


class TestConcurrentQueries:
    def test_concurrent_fts_queries_no_error(self, fts):
        """Multiple threads running FTS queries should not raise errors."""
        errors: list[Exception] = []

        def search_worker(query: str) -> list:
            try:
                return fts.exact_search(query, top_k=10)
            except Exception as e:
                errors.append(e)
                return []

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(search_worker, q)
                for q in ["main", "helper", "Config", "debug"] * 5
            ]
            results = [f.result() for f in futures]

        assert not errors, f"Concurrent queries raised errors: {errors}"
        non_empty = [r for r in results if r]
        assert len(non_empty) > 0

    def test_concurrent_read_operations(self, fts):
        """Multiple threads doing various read operations concurrently."""
        errors: list[Exception] = []

        def read_meta(doc_id: int):
            try:
                return fts.get_doc_meta(doc_id)
            except Exception as e:
                errors.append(e)
                return None

        def read_content(doc_id: int):
            try:
                return fts.get_content(doc_id)
            except Exception as e:
                errors.append(e)
                return None

        def fuzzy_search(q: str):
            try:
                return fts.fuzzy_search(q, top_k=5)
            except Exception as e:
                errors.append(e)
                return []

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = []
            for i in range(1, 4):
                futures.append(pool.submit(read_meta, i))
                futures.append(pool.submit(read_content, i))
            for q in ["main", "helper", "Config"]:
                futures.append(pool.submit(fuzzy_search, q))
            for f in futures:
                f.result()

        assert not errors, f"Concurrent reads raised errors: {errors}"

    def test_concurrent_mixed_exact_fuzzy(self, fts):
        """Interleaved exact and fuzzy searches from multiple threads."""
        errors: list[Exception] = []

        def worker(idx: int):
            try:
                if idx % 2 == 0:
                    fts.exact_search("main", top_k=5)
                else:
                    fts.fuzzy_search("help", top_k=5)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(worker, i) for i in range(20)]
            for f in futures:
                f.result()

        assert not errors, f"Concurrent mixed queries raised errors: {errors}"


class TestCloseCleanup:
    def test_close_cleans_up_all_connections(self, tmp_path):
        """close() should close all pooled connections."""
        fts = FTSEngine(str(tmp_path / "fts.db"))

        def create_conn():
            fts._get_conn()

        threads = [threading.Thread(target=create_conn) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with fts._read_conns_lock:
            assert len(fts._read_conns) == 3

        fts.close()

        with fts._read_conns_lock:
            assert len(fts._read_conns) == 0
