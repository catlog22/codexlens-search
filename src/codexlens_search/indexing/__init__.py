from __future__ import annotations

from .metadata import MetadataStore
from .pipeline import IndexingPipeline, IndexStats

try:
    from .gitignore import GitignoreAwareMatcher
except ImportError:
    pass

__all__ = ["GitignoreAwareMatcher", "IndexingPipeline", "IndexStats", "MetadataStore"]
