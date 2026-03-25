"""Performance benchmarks for P0-P2 async/parallel improvements.

Measures wall-time improvement for:
- P0: Parallel vs serial tool dispatch
- P1: Parallel vs serial reranker batch scoring
- P2: Parallel expansion + FTS pre-warm vs serial
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import numpy as np


def _time_it(fn, runs=5):
    """Run fn multiple times, return (median_ms, results)."""
    times = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2], result


# ─── P0: Parallel tool dispatch ──────────────────────────────────────

def bench_p0_tool_dispatch():
    """Simulate parallel vs serial tool call dispatch."""
    print("\n=== P0: Parallel Tool Dispatch ===")

    def fake_tool_call(sleep_ms=50):
        """Simulate I/O-bound tool call."""
        time.sleep(sleep_ms / 1000)
        return {"result": "ok"}

    num_calls = 4
    sleep_ms = 50

    # Serial
    def serial():
        return [fake_tool_call(sleep_ms) for _ in range(num_calls)]

    serial_ms, _ = _time_it(serial, runs=3)

    # Parallel
    def parallel():
        with ThreadPoolExecutor(max_workers=num_calls) as pool:
            futures = [pool.submit(fake_tool_call, sleep_ms) for _ in range(num_calls)]
            return [f.result() for f in futures]

    parallel_ms, _ = _time_it(parallel, runs=3)

    speedup = serial_ms / parallel_ms if parallel_ms > 0 else float("inf")
    print(f"  {num_calls} calls × {sleep_ms}ms each")
    print(f"  Serial:   {serial_ms:7.1f} ms")
    print(f"  Parallel: {parallel_ms:7.1f} ms")
    print(f"  Speedup:  {speedup:.2f}x")
    return speedup


# ─── P1: Parallel reranker batches ───────────────────────────────────

def bench_p1_reranker_batches():
    """Simulate parallel vs serial reranker API batch scoring."""
    print("\n=== P1: Parallel Reranker Batches ===")

    def fake_api_call(batch, sleep_ms=80):
        """Simulate reranker API call."""
        time.sleep(sleep_ms / 1000)
        return {idx: 0.5 + idx * 0.01 for idx, _ in batch}

    num_batches = 4
    batch_size = 10
    sleep_ms = 80
    batches = [[(i * batch_size + j, f"doc {j}") for j in range(batch_size)] for i in range(num_batches)]

    # Serial
    def serial():
        scores = {}
        for batch in batches:
            scores.update(fake_api_call(batch, sleep_ms))
        return scores

    serial_ms, _ = _time_it(serial, runs=3)

    # Parallel
    def parallel():
        scores = {}
        with ThreadPoolExecutor(max_workers=num_batches) as pool:
            results = list(pool.map(lambda b: fake_api_call(b, sleep_ms), batches))
        for batch_scores in results:
            scores.update(batch_scores)
        return scores

    parallel_ms, _ = _time_it(parallel, runs=3)

    speedup = serial_ms / parallel_ms if parallel_ms > 0 else float("inf")
    print(f"  {num_batches} batches × {batch_size} docs, {sleep_ms}ms/batch")
    print(f"  Serial:   {serial_ms:7.1f} ms")
    print(f"  Parallel: {parallel_ms:7.1f} ms")
    print(f"  Speedup:  {speedup:.2f}x")
    return speedup


# ─── P2: Parallel expansion + FTS pre-warm ───────────────────────────

def bench_p2_expansion_fts():
    """Simulate parallel query expansion + FTS pre-warm vs serial."""
    print("\n=== P2: Parallel Expansion + FTS Pre-Warm ===")

    expand_ms = 100
    fts_ms = 60

    def fake_expand(query):
        time.sleep(expand_ms / 1000)
        return query + " expanded"

    def fake_fts(query):
        time.sleep(fts_ms / 1000)
        return [(1, 0.9), (2, 0.8), (3, 0.7)]

    # Serial: expand first, then FTS
    def serial():
        expanded = fake_expand("test query")
        results = fake_fts(expanded)
        return expanded, results

    serial_ms, _ = _time_it(serial, runs=3)

    # Parallel: expand + FTS concurrent (FTS uses original query for pre-warm)
    def parallel():
        with ThreadPoolExecutor(max_workers=2) as pool:
            expand_future = pool.submit(fake_expand, "test query")
            fts_future = pool.submit(fake_fts, "test query")
            expanded = expand_future.result()
            results = fts_future.result()
        return expanded, results

    parallel_ms, _ = _time_it(parallel, runs=3)

    speedup = serial_ms / parallel_ms if parallel_ms > 0 else float("inf")
    print(f"  Expansion: {expand_ms}ms, FTS: {fts_ms}ms")
    print(f"  Serial:   {serial_ms:7.1f} ms")
    print(f"  Parallel: {parallel_ms:7.1f} ms")
    print(f"  Speedup:  {speedup:.2f}x  (FTS latency hidden behind expansion)")
    return speedup


# ─── P2: Concurrent FTS reads (thread-local connections) ─────────────

def bench_p2_concurrent_fts():
    """Benchmark concurrent FTS reads with thread-local connections."""
    print("\n=== P2: Concurrent FTS Reads ===")

    try:
        import tempfile
        from codexlens_search.search.fts import FTSEngine

        with tempfile.TemporaryDirectory() as tmp:
            fts = FTSEngine(f"{tmp}/bench.db")
            # Index 100 documents
            docs = [
                (i, f"src/file_{i}.py", f"class Foo{i}: def method_{i}(self): return {i}", 1, 3, "python")
                for i in range(100)
            ]
            fts.add_documents(docs)
            fts.flush()

            queries = ["class", "method", "return", "def", "Foo"] * 4

            # Serial
            def serial():
                return [fts.exact_search(q, top_k=10) for q in queries]

            serial_ms, _ = _time_it(serial, runs=5)

            # Parallel (4 workers)
            def parallel():
                with ThreadPoolExecutor(max_workers=4) as pool:
                    return list(pool.map(lambda q: fts.exact_search(q, top_k=10), queries))

            parallel_ms, _ = _time_it(parallel, runs=5)

            speedup = serial_ms / parallel_ms if parallel_ms > 0 else float("inf")
            print(f"  {len(queries)} queries on 100 docs, 4 workers")
            print(f"  Serial:   {serial_ms:7.1f} ms")
            print(f"  Parallel: {parallel_ms:7.1f} ms")
            print(f"  Speedup:  {speedup:.2f}x")
            fts.close()
            return speedup
    except Exception as e:
        print(f"  Skipped: {e}")
        return 1.0


# ─── Summary ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  P0-P2 Parallel Improvement Benchmarks")
    print("=" * 60)

    results = {}
    results["P0: Tool Dispatch"] = bench_p0_tool_dispatch()
    results["P1: Reranker Batches"] = bench_p1_reranker_batches()
    results["P2: Expansion+FTS"] = bench_p2_expansion_fts()
    results["P2: Concurrent FTS"] = bench_p2_concurrent_fts()

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for name, speedup in results.items():
        bar = "█" * int(speedup * 5)
        print(f"  {name:<25s} {speedup:5.2f}x  {bar}")
    print()


if __name__ == "__main__":
    main()
