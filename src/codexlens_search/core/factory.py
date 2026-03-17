from __future__ import annotations

import logging
from pathlib import Path

from codexlens_search.config import Config
from codexlens_search.core.base import BaseANNIndex, BaseBinaryIndex

logger = logging.getLogger(__name__)

try:
    import faiss as _faiss  # noqa: F401
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

try:
    import hnswlib as _hnswlib  # noqa: F401
    _HNSWLIB_AVAILABLE = True
except ImportError:
    _HNSWLIB_AVAILABLE = False


def _has_faiss_gpu() -> bool:
    """Check whether faiss-gpu is available (has GPU resources)."""
    if not _FAISS_AVAILABLE:
        return False
    try:
        import faiss
        res = faiss.StandardGpuResources()  # noqa: F841
        return True
    except (AttributeError, RuntimeError):
        return False


def create_ann_index(path: str | Path, dim: int, config: Config) -> BaseANNIndex:
    """Create an ANN index based on config.ann_backend.

    Fallback chain for 'auto': faiss-gpu -> faiss-cpu -> hnswlib.

    Args:
        path: directory for index persistence
        dim: vector dimensionality
        config: project configuration

    Returns:
        A BaseANNIndex implementation

    Raises:
        ImportError: if no suitable backend is available
    """
    backend = config.ann_backend

    if backend == "faiss":
        from codexlens_search.core.faiss_index import FAISSANNIndex
        return FAISSANNIndex(path, dim, config)

    if backend == "hnswlib":
        from codexlens_search.core.index import ANNIndex
        return ANNIndex(path, dim, config)

    # auto: try faiss first, then hnswlib
    if _FAISS_AVAILABLE:
        from codexlens_search.core.faiss_index import FAISSANNIndex
        gpu_tag = " (GPU available)" if _has_faiss_gpu() else " (CPU)"
        logger.info("Auto-selected FAISS ANN backend%s", gpu_tag)
        return FAISSANNIndex(path, dim, config)

    if _HNSWLIB_AVAILABLE:
        from codexlens_search.core.index import ANNIndex
        logger.info("Auto-selected hnswlib ANN backend")
        return ANNIndex(path, dim, config)

    raise ImportError(
        "No ANN backend available. Install faiss-cpu, faiss-gpu, or hnswlib."
    )


def create_binary_index(
    path: str | Path, dim: int, config: Config
) -> BaseBinaryIndex:
    """Create a binary index based on config.binary_backend.

    Fallback chain for 'auto': faiss -> numpy BinaryStore.

    Args:
        path: directory for index persistence
        dim: vector dimensionality
        config: project configuration

    Returns:
        A BaseBinaryIndex implementation

    Raises:
        ImportError: if no suitable backend is available
    """
    backend = config.binary_backend

    if backend == "faiss":
        from codexlens_search.core.faiss_index import FAISSBinaryIndex
        return FAISSBinaryIndex(path, dim, config)

    if backend == "hnswlib":
        from codexlens_search.core.binary import BinaryStore
        return BinaryStore(path, dim, config)

    # auto: try faiss first, then numpy-based BinaryStore
    if _FAISS_AVAILABLE:
        from codexlens_search.core.faiss_index import FAISSBinaryIndex
        logger.info("Auto-selected FAISS binary backend")
        return FAISSBinaryIndex(path, dim, config)

    # numpy BinaryStore is always available (no extra deps)
    from codexlens_search.core.binary import BinaryStore
    logger.info("Auto-selected numpy BinaryStore backend")
    return BinaryStore(path, dim, config)
