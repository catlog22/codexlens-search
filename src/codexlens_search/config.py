from __future__ import annotations
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class Config:
    # Embedding
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    embed_batch_size: int = 32

    # API embedding (optional — overrides local fastembed when set)
    embed_api_url: str = ""  # e.g. "https://api.openai.com/v1"
    embed_api_key: str = ""
    embed_api_model: str = ""  # e.g. "text-embedding-3-small"
    # Multi-endpoint: list of {"url": "...", "key": "...", "model": "..."} dicts
    embed_api_endpoints: list[dict[str, str]] = None  # type: ignore[assignment]
    embed_api_concurrency: int = 4
    embed_api_max_tokens_per_batch: int = 32768
    embed_max_tokens: int = 8192  # max tokens per single text (0 = no limit)

    # Model download / cache
    model_cache_dir: str = ""  # empty = fastembed default cache
    hf_mirror: str = ""  # HuggingFace mirror URL, e.g. "https://hf-mirror.com"

    # GPU / execution providers
    device: str = "auto"  # 'auto', 'cuda', 'directml', 'cpu'
    embed_providers: list[str] | None = None  # explicit ONNX providers override

    # File filtering
    max_file_size_bytes: int = 1_000_000  # 1MB
    exclude_extensions: frozenset[str] = None  # type: ignore[assignment]  # set in __post_init__
    binary_detect_sample_bytes: int = 2048
    binary_null_threshold: float = 0.10  # >10% null bytes = binary
    generated_code_markers: tuple[str, ...] = ("@generated", "DO NOT EDIT", "auto-generated", "AUTO GENERATED")
    gitignore_filtering: bool = True  # use .gitignore rules to exclude files

    # Code-aware chunking
    code_aware_chunking: bool = True
    code_extensions: frozenset[str] = frozenset({
        ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".java", ".cpp", ".c",
        ".h", ".hpp", ".cs", ".rs", ".rb", ".php", ".scala", ".kt", ".swift",
        ".lua", ".sh", ".bash", ".zsh", ".ps1", ".vue", ".svelte",
    })

    # AST-based chunking (uses tree-sitter)
    ast_chunking: bool = True
    ast_languages: frozenset[str] | None = None  # per-language opt-in, None = all detected
    chunk_context_header: bool = True  # prepend file/class/func context to chunks for better embedding

    # Backend selection: 'auto', 'usearch', 'faiss', 'hnswlib'
    ann_backend: str = "auto"
    binary_backend: str = "faiss"

    # Indexing pipeline
    index_workers: int = 2  # number of parallel indexing workers
    skip_chunk_hash: bool = True  # use sequential chunk ID instead of SHA-256 per chunk

    # HNSW index (ANNIndex)
    hnsw_ef: int = 150
    hnsw_M: int = 32
    hnsw_ef_construction: int = 200

    # Binary coarse search (BinaryStore)
    binary_top_k: int = 200

    # ANN fine search
    ann_top_k: int = 50

    # Reranker
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 20
    reranker_batch_size: int = 32

    # API reranker (optional)
    reranker_api_url: str = ""
    reranker_api_key: str = ""
    reranker_api_model: str = ""
    reranker_api_max_tokens_per_batch: int = 2048
    reranker_api_concurrency: int = 1  # parallel batch scoring (1 = serial)

    # Metadata store
    metadata_db_path: str = ""  # empty = no metadata tracking

    # Data tiering (hot/warm/cold)
    tier_hot_hours: int = 24  # files accessed within this window are 'hot'
    tier_cold_hours: int = 168  # files not accessed for this long are 'cold'

    # Search quality tier: 'fast', 'balanced', 'thorough', 'auto'
    default_search_quality: str = "auto"

    # Shard partitioning
    num_shards: int = 1  # 1 = single partition (no sharding), >1 = hash-based sharding
    max_loaded_shards: int = 4  # LRU limit for loaded shards in ShardManager

    # FTS
    fts_top_k: int = 50

    # Query expansion (symbol vocabulary nearest-neighbor + two-hop)
    expansion_enabled: bool = True
    expansion_top_k: int = 5  # max first-hop terms from vector similarity
    expansion_threshold: float = 0.35  # cosine similarity cutoff

    # Fusion
    fusion_k: int = 60  # RRF k parameter
    fusion_weights: dict = field(default_factory=lambda: {
        "exact": 0.17,
        "fuzzy": 0.085,
        "vector": 0.34,
        "graph": 0.085,
        "symbol": 0.17,
        "entity": 0.15,
    })

    # Graph search weights (symbol reference graph scoring)
    graph_kind_weights: dict[str, float] | None = None
    graph_dir_weights: dict[str, float] | None = None

    # Symbol-level search boost (exact symbol name lookup)
    symbol_search_enabled: bool = True

    # Entity dependency graph expansion
    entity_graph_enabled: bool = True
    entity_graph_depth: int = 2
    entity_graph_backend: str = "auto"

    # LLM Agent loop (optional)
    agent_enabled: bool = False
    agent_llm_model: str = "glm-5-turbo"
    agent_llm_api_key: str = ""
    agent_llm_api_base: str = "https://open.bigmodel.cn/api/paas/v4/"
    agent_max_iterations: int = 5
    agent_tool_concurrency: int = 1
    agent_parallel_tools_allowlist: tuple[str, ...] = ("read_files_batch", "get_entity_content")
    agent_fan_out_enabled: bool = False
    agent_fan_out_max_workers: int = 3

    _DEFAULT_EXCLUDE_EXTENSIONS: frozenset[str] = frozenset({
        # binaries / images
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".svg",
        ".zip", ".gz", ".tar", ".rar", ".7z", ".bz2",
        ".bin", ".exe", ".dll", ".so", ".dylib", ".a", ".o", ".obj",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        # build / generated
        ".min.js", ".min.css", ".map", ".lock",
        ".pyc", ".pyo", ".class", ".wasm",
        # data
        ".sqlite", ".db", ".npy", ".npz", ".pkl", ".pickle",
        ".parquet", ".arrow", ".feather",
        # media
        ".mp3", ".mp4", ".wav", ".avi", ".mov", ".flv",
        ".ttf", ".otf", ".woff", ".woff2", ".eot",
    })

    def __post_init__(self) -> None:
        if self.exclude_extensions is None:
            object.__setattr__(self, "exclude_extensions", self._DEFAULT_EXCLUDE_EXTENSIONS)
        if self.embed_api_endpoints is None:
            object.__setattr__(self, "embed_api_endpoints", [])
        # Ensure fusion weights include new sources (backward compatible)
        if self.fusion_weights is None:
            object.__setattr__(self, "fusion_weights", {})
        default_fusion = {
            "exact": 0.17,
            "fuzzy": 0.085,
            "vector": 0.34,
            "graph": 0.085,
            "symbol": 0.17,
            "entity": 0.15,
        }
        merged_fusion = dict(default_fusion)
        merged_fusion.update(self.fusion_weights)
        object.__setattr__(self, "fusion_weights", merged_fusion)

        # Entity graph config normalization
        try:
            object.__setattr__(self, "entity_graph_depth", max(0, int(self.entity_graph_depth)))
        except Exception:
            object.__setattr__(self, "entity_graph_depth", 2)

        # Agent loop config normalization
        try:
            object.__setattr__(self, "agent_max_iterations", max(1, int(self.agent_max_iterations)))
        except Exception:
            object.__setattr__(self, "agent_max_iterations", 5)
        try:
            object.__setattr__(self, "agent_tool_concurrency", max(1, int(self.agent_tool_concurrency)))
        except Exception:
            object.__setattr__(self, "agent_tool_concurrency", 1)
        try:
            object.__setattr__(self, "reranker_api_concurrency", max(1, int(self.reranker_api_concurrency)))
        except Exception:
            object.__setattr__(self, "reranker_api_concurrency", 1)
        try:
            object.__setattr__(self, "agent_fan_out_max_workers", max(1, int(self.agent_fan_out_max_workers)))
        except Exception:
            object.__setattr__(self, "agent_fan_out_max_workers", 3)

        # Graph weights: allow partial overrides, fill missing keys
        default_kind = {
            "import": 1.0,
            "call": 1.5,
            "inherit": 0.9,
            "type_ref": 0.3,
        }
        merged_kind = dict(default_kind)
        if self.graph_kind_weights:
            merged_kind.update(self.graph_kind_weights)
        object.__setattr__(self, "graph_kind_weights", merged_kind)

        default_dir = {
            "backward": 1.3,
            "forward": 0.6,
        }
        merged_dir = dict(default_dir)
        if self.graph_dir_weights:
            merged_dir.update(self.graph_dir_weights)
        object.__setattr__(self, "graph_dir_weights", merged_dir)

        # GPU ONNX sessions are not thread-safe — clamp to 1 embed worker
        # and increase batch size to leverage GPU-internal parallelism
        if self._uses_gpu():
            if self.index_workers > 1:
                object.__setattr__(self, "index_workers", 1)
            if self.embed_batch_size <= 32:
                object.__setattr__(self, "embed_batch_size", 64)

    def _uses_gpu(self) -> bool:
        """Check if GPU execution provider will be used (explicit or auto-detected)."""
        if self.embed_providers is not None:
            return any(p in self.embed_providers for p in ("CUDAExecutionProvider", "DmlExecutionProvider"))
        if self.device in ("cuda", "directml"):
            return True
        if self.device == "auto":
            try:
                import onnxruntime
                available = onnxruntime.get_available_providers()
                return "CUDAExecutionProvider" in available or "DmlExecutionProvider" in available
            except ImportError:
                pass
        return False

    def resolve_embed_providers(self) -> list[str]:
        """Return ONNX execution providers based on device config.

        Priority: explicit embed_providers > device setting > auto-detect.
        """
        if self.embed_providers is not None:
            return list(self.embed_providers)

        if self.device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        if self.device == "directml":
            return ["DmlExecutionProvider", "CPUExecutionProvider"]

        if self.device == "cpu":
            return ["CPUExecutionProvider"]

        # auto-detect: CUDA > DirectML > CPU
        try:
            import onnxruntime
            available = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" in available:
                log.info("CUDA detected via onnxruntime, using GPU for embedding")
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "DmlExecutionProvider" in available:
                log.info("DirectML detected via onnxruntime, using GPU for embedding")
                return ["DmlExecutionProvider", "CPUExecutionProvider"]

            # Windows: fastembed's onnxruntime dep may have overwritten
            # onnxruntime-directml.  Auto-repair by reinstalling directml.
            # C extensions can't hot-reload, so after install we verify via
            # subprocess and tell the user to restart for GPU acceleration.
            import sys
            if sys.platform == "win32" and "DmlExecutionProvider" not in available:
                if self._try_install_directml():
                    log.warning(
                        "onnxruntime-directml installed successfully. "
                        "Restart the process to enable GPU acceleration. "
                        "Using CPU for this session."
                    )
        except ImportError:
            pass

        return ["CPUExecutionProvider"]

    @staticmethod
    def _try_install_directml() -> bool:
        """Attempt to pip-install onnxruntime-directml to restore GPU support."""
        import subprocess
        import sys

        log.info("DirectML not available — attempting auto-install of onnxruntime-directml")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install",
                 "onnxruntime-directml", "--force-reinstall", "--no-deps", "--quiet"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                log.info("onnxruntime-directml installed successfully")
                return True
            log.warning("onnxruntime-directml install failed: %s", result.stderr.strip())
        except Exception as exc:
            log.warning("onnxruntime-directml auto-install error: %s", exc)
        return False

    @classmethod
    def defaults(cls) -> "Config":
        return cls()

    @classmethod
    def small(cls) -> "Config":
        """Smaller config for testing or small corpora."""
        return cls(
            hnsw_ef=50,
            hnsw_M=16,
            binary_top_k=50,
            ann_top_k=20,
            reranker_top_k=10,
        )
