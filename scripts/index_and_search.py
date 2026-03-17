"""
对 D:/Claude_dms3 仓库进行索引并测试搜索。
用法: python scripts/index_and_search.py
"""
import sys
import time
from pathlib import Path

# 确保 src 可被导入
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codexlens_search.config import Config
from codexlens_search.core.factory import create_ann_index, create_binary_index
from codexlens_search.embed.local import FastEmbedEmbedder
from codexlens_search.indexing import IndexingPipeline
from codexlens_search.rerank.local import FastEmbedReranker
from codexlens_search.search.fts import FTSEngine
from codexlens_search.search.pipeline import SearchPipeline

# ─── 配置 ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path("D:/Claude_dms3")
INDEX_DIR = Path("D:/Claude_dms3/codex-lens-v2/.index_cache")
EXTENSIONS = {".py", ".ts", ".js", ".md"}
MAX_FILE_SIZE = 50_000   # bytes
MAX_CHUNK_CHARS = 800    # 每个 chunk 的最大字符数
CHUNK_OVERLAP = 100

# ─── 文件收集 ───────────────────────────────────────────────────────────────
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".pytest_cache",
    "dist", "build", ".venv", "venv", ".cache", ".index_cache",
    "codex-lens-v2",  # 不索引自身
}

def collect_files(root: Path) -> list[Path]:
    files = []
    for p in root.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.is_file() and p.suffix in EXTENSIONS:
            if p.stat().st_size <= MAX_FILE_SIZE:
                files.append(p)
    return files

# ─── 主流程 ─────────────────────────────────────────────────────────────────
def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 使用小 profile 加快速度
    config = Config(
        embed_model="BAAI/bge-small-en-v1.5",
        embed_dim=384,
        embed_batch_size=32,
        hnsw_ef=100,
        hnsw_M=16,
        binary_top_k=100,
        ann_top_k=30,
        reranker_top_k=10,
    )

    print("=== codex-lens-v2 索引测试 ===\n")

    # 2. 收集文件
    print(f"[1/4] 扫描 {REPO_ROOT} ...")
    files = collect_files(REPO_ROOT)
    print(f"      找到 {len(files)} 个文件")

    # 3. 初始化组件
    print(f"\n[2/4] 加载嵌入模型 (bge-small-en-v1.5, dim=384) ...")
    embedder = FastEmbedEmbedder(config)
    binary_store = create_binary_index(INDEX_DIR, config.embed_dim, config)
    ann_index = create_ann_index(INDEX_DIR, config.embed_dim, config)
    fts = FTSEngine(":memory:")   # 内存 FTS，不持久化

    # 4. 使用 IndexingPipeline 并行索引 (chunk -> embed -> index)
    print(f"[3/4] 并行索引 {len(files)} 个文件 ...")
    pipeline = IndexingPipeline(
        embedder=embedder,
        binary_store=binary_store,
        ann_index=ann_index,
        fts=fts,
        config=config,
    )
    stats = pipeline.index_files(
        files,
        root=REPO_ROOT,
        max_chunk_chars=MAX_CHUNK_CHARS,
        chunk_overlap=CHUNK_OVERLAP,
        max_file_size=MAX_FILE_SIZE,
    )
    print(f"      索引完成: {stats.files_processed} 文件, {stats.chunks_created} chunks ({stats.duration_seconds:.1f}s)")

    # 5. 搜索测试
    print(f"\n[4/4] 构建 SearchPipeline ...")
    reranker = FastEmbedReranker(config)
    pipeline = SearchPipeline(
        embedder=embedder,
        binary_store=binary_store,
        ann_index=ann_index,
        reranker=reranker,
        fts=fts,
        config=config,
    )

    queries = [
        "authentication middleware function",
        "def embed_single",
        "RRF fusion weights",
        "fastembed TextCrossEncoder reranker",
        "how to search code semantic",
    ]

    print("\n" + "=" * 60)
    for query in queries:
        t0 = time.time()
        results = pipeline.search(query, top_k=5)
        elapsed = time.time() - t0
        print(f"\nQuery: {query!r}  ({elapsed*1000:.0f}ms)")
        if results:
            for r in results:
                print(f"  [{r.score:.3f}] {r.path}")
        else:
            print("  (无结果)")
    print("=" * 60)
    print("\n测试完成 ✓")

if __name__ == "__main__":
    main()
