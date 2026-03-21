"""Benchmark: DirectML GPU vs CPU vs API embedding + full pipeline.

Runs each embedding mode in a subprocess to avoid ONNX session conflicts.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PYTHON = sys.executable


def run_embed_bench(device: str, label: str) -> dict:
    """Run embedding benchmark in subprocess, return parsed results."""
    code = textwrap.dedent(f"""\
        import time, json
        from pathlib import Path
        from codexlens_search.config import Config
        from codexlens_search.embed.local import FastEmbedEmbedder

        chunks = []
        for f in sorted(Path('src').rglob('*.py')):
            text = f.read_text(encoding='utf-8', errors='replace')
            for i in range(0, len(text), 2000):
                c = text[i:i+2000].strip()
                if c: chunks.append(c)

        cfg = Config(embed_model='BAAI/bge-small-en-v1.5', embed_dim=384,
                     device='{device}', embed_batch_size=32)
        emb = FastEmbedEmbedder(cfg)
        providers = cfg.resolve_embed_providers()

        emb.embed_single('warmup')

        t0 = time.perf_counter()
        vecs = emb.embed_batch(chunks)
        elapsed = time.perf_counter() - t0

        print(json.dumps({{
            'device': '{device}',
            'provider': providers[0],
            'chunks': len(chunks),
            'time_s': round(elapsed, 3),
            'chunks_per_s': round(len(chunks) / elapsed, 1),
        }}))
    """)
    result = subprocess.run(
        [PYTHON, "-c", code],
        cwd=str(SCRIPT_DIR), capture_output=True, timeout=300,
    )
    stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
    stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
    if result.returncode != 0:
        print(f"  [{label}] ERROR: {stderr[-300:]}")
        return {"device": device, "error": True}

    for line in stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return {"device": device, "error": True}


def run_api_bench() -> dict | None:
    """Run API embedding benchmark in subprocess."""
    mcp_path = SCRIPT_DIR.parent / ".mcp.json"
    if not mcp_path.exists():
        mcp_path = SCRIPT_DIR / ".mcp.json"
    if not mcp_path.exists():
        return None

    data = json.loads(mcp_path.read_text(encoding="utf-8"))
    env = data.get("mcpServers", {}).get("codexlens", {}).get("env", {})
    api_url = env.get("CODEXLENS_EMBED_API_URL", "")
    api_key = env.get("CODEXLENS_EMBED_API_KEY", "")
    api_model = env.get("CODEXLENS_EMBED_API_MODEL", "")
    api_dim = env.get("CODEXLENS_EMBED_DIM", "1024")
    if not api_key:
        return None

    code = textwrap.dedent(f"""\
        import time, json
        from pathlib import Path
        from codexlens_search.config import Config
        from codexlens_search.embed.api import APIEmbedder

        chunks = []
        for f in sorted(Path('src').rglob('*.py')):
            text = f.read_text(encoding='utf-8', errors='replace')
            for i in range(0, len(text), 2000):
                c = text[i:i+2000].strip()
                if c: chunks.append(c)
        chunks = [c[:1024] for c in chunks]

        cfg = Config(
            embed_api_url='{api_url}', embed_api_key='{api_key}',
            embed_api_model='{api_model}', embed_dim={api_dim},
            embed_batch_size=4, embed_api_concurrency=2,
            embed_api_max_tokens_per_batch=2048, embed_max_tokens=256,
        )
        emb = APIEmbedder(cfg)
        emb.embed_single('warmup')

        t0 = time.perf_counter()
        vecs = emb.embed_batch(chunks)
        elapsed = time.perf_counter() - t0

        print(json.dumps({{
            'device': 'api',
            'provider': '{api_model}',
            'chunks': len(chunks),
            'time_s': round(elapsed, 3),
            'chunks_per_s': round(len(chunks) / elapsed, 1),
        }}))
    """)
    result = subprocess.run(
        [PYTHON, "-c", code],
        cwd=str(SCRIPT_DIR), capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print(f"  [api] ERROR: {result.stderr[-300:]}")
        return None

    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return None


def run_pipeline_bench(device: str, label: str) -> dict:
    """Run full pipeline index benchmark in subprocess."""
    code = textwrap.dedent(f"""\
        import sys, time, json, shutil, traceback
        from pathlib import Path

        try:
            from codexlens_search.config import Config
            from codexlens_search.bridge import create_pipeline

            src = Path('src')
            db = Path('.bench_tmp') / '{label}'
            if db.exists(): shutil.rmtree(db)
            db.mkdir(parents=True)

            files = sorted(p for p in src.rglob('*.py') if p.is_file())

            cfg = Config(embed_model='BAAI/bge-small-en-v1.5', embed_dim=384,
                         device='{device}', ann_backend='usearch')

            indexing, search, _ = create_pipeline(db, cfg)

            t0 = time.perf_counter()
            result = indexing.sync(files, root=src)
            index_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            sr = search.search('embedding pipeline config')
            search_s = time.perf_counter() - t0

            fc = result.files_processed if hasattr(result, 'files_processed') else len(files)
            cc = result.chunks_created if hasattr(result, 'chunks_created') else 0
            ann = type(search._ann_index).__name__ if hasattr(search, '_ann_index') else '?'

            # Close resources before cleanup
            if hasattr(search, 'close'): search.close()
            if hasattr(indexing, 'close'): indexing.close()
            try:
                shutil.rmtree(db)
            except Exception:
                pass

            print(json.dumps({{
                'device': '{device}',
                'files': fc,
                'chunks': cc,
                'index_s': round(index_s, 3),
                'search_s': round(search_s, 3),
                'search_results': len(sr),
                'ann': ann,
            }}))
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    """)
    result = subprocess.run(
        [PYTHON, "-c", code],
        cwd=str(SCRIPT_DIR), capture_output=True, timeout=600,
    )
    stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
    stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
    if result.returncode != 0:
        print(f"  [{label}] ERROR: {stderr[-500:]}")
        return {"device": device, "error": True}

    for line in stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return {"device": device, "error": True}


def main():
    print("=" * 65)
    print("BENCHMARK: Embedding Speed (bge-small-en-v1.5, 384d)")
    print("=" * 65)

    # Embedding benchmarks
    embed_results = []
    for device, label in [("cpu", "CPU"), ("directml", "DirectML GPU")]:
        print(f"  Running {label}...")
        r = run_embed_bench(device, label)
        if not r.get("error"):
            embed_results.append((label, r))
            print(f"  {label}: {r['time_s']}s, {r['chunks_per_s']} chunks/s")
        else:
            print(f"  {label}: FAILED")

    print(f"\n  Running API...")
    api_r = run_api_bench()
    if api_r:
        embed_results.append(("API (SiliconFlow)", api_r))
        print(f"  API: {api_r['time_s']}s, {api_r['chunks_per_s']} chunks/s")

    # Pipeline benchmarks
    print()
    print("=" * 65)
    print("BENCHMARK: Full Pipeline Index (USearch ANN)")
    print("=" * 65)

    pipe_results = []
    for device, label in [("cpu", "CPU+USearch"), ("directml", "DirectML+USearch")]:
        print(f"  Running {label}...")
        r = run_pipeline_bench(device, label)
        if not r.get("error"):
            pipe_results.append((label, r))
            print(f"  {label}: index={r['index_s']}s, search={r['search_s']}s, {r['chunks']} chunks")
        else:
            print(f"  {label}: FAILED")

    # Summary
    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)

    if embed_results:
        print(f"\n  Embedding ({embed_results[0][1]['chunks']} chunks):")
        print(f"  {'Mode':<22} {'Time(s)':<10} {'Chunks/s':<12} {'Provider'}")
        print(f"  {'-'*60}")
        for label, r in embed_results:
            print(f"  {label:<22} {r['time_s']:<10} {r['chunks_per_s']:<12} {r['provider']}")

        if len(embed_results) >= 2:
            cpu_t = embed_results[0][1]["time_s"]
            dml_t = embed_results[1][1]["time_s"]
            print(f"\n  DirectML is {cpu_t/dml_t:.1f}x faster than CPU")

    if pipe_results:
        print(f"\n  Full Pipeline:")
        print(f"  {'Mode':<22} {'Files':<7} {'Chunks':<8} {'Index(s)':<10} {'Search(s)':<10}")
        print(f"  {'-'*57}")
        for label, r in pipe_results:
            print(f"  {label:<22} {r['files']:<7} {r['chunks']:<8} {r['index_s']:<10} {r['search_s']:<10}")

        if len(pipe_results) >= 2:
            cpu_idx = pipe_results[0][1]["index_s"]
            dml_idx = pipe_results[1][1]["index_s"]
            print(f"\n  DirectML pipeline is {cpu_idx/dml_idx:.1f}x faster for indexing")

    print("\nDone.")


if __name__ == "__main__":
    main()
