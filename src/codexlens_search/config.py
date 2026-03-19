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
    device: str = "auto"  # 'auto', 'cuda', 'cpu'
    embed_providers: list[str] | None = None  # explicit ONNX providers override

    # File filtering
    max_file_size_bytes: int = 1_000_000  # 1MB
    exclude_extensions: frozenset[str] = None  # type: ignore[assignment]  # set in __post_init__
    binary_detect_sample_bytes: int = 2048
    binary_null_threshold: float = 0.10  # >10% null bytes = binary
    generated_code_markers: tuple[str, ...] = ("@generated", "DO NOT EDIT", "auto-generated", "AUTO GENERATED")

    # Code-aware chunking
    code_aware_chunking: bool = True
    code_extensions: frozenset[str] = frozenset({
        ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".java", ".cpp", ".c",
        ".h", ".hpp", ".cs", ".rs", ".rb", ".php", ".scala", ".kt", ".swift",
        ".lua", ".sh", ".bash", ".zsh", ".ps1", ".vue", ".svelte",
    })

    # Backend selection: 'auto', 'faiss', 'hnswlib'
    ann_backend: str = "auto"
    binary_backend: str = "auto"

    # Indexing pipeline
    index_workers: int = 2  # number of parallel indexing workers

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

    # Metadata store
    metadata_db_path: str = ""  # empty = no metadata tracking

    # FTS
    fts_top_k: int = 50

    # Fusion
    fusion_k: int = 60  # RRF k parameter
    fusion_weights: dict = field(default_factory=lambda: {
        "exact": 0.25,
        "fuzzy": 0.10,
        "vector": 0.50,
        "graph": 0.15,
    })

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

    def resolve_embed_providers(self) -> list[str]:
        """Return ONNX execution providers based on device config.

        Priority: explicit embed_providers > device setting > auto-detect.
        """
        if self.embed_providers is not None:
            return list(self.embed_providers)

        if self.device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        if self.device == "cpu":
            return ["CPUExecutionProvider"]

        # auto-detect
        try:
            import onnxruntime
            available = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" in available:
                log.info("CUDA detected via onnxruntime, using GPU for embedding")
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except ImportError:
            pass

        return ["CPUExecutionProvider"]

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
