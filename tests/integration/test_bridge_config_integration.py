"""L2 integration tests for bridge.py and config.py.

Tests create_config_from_env, should_exclude, create_pipeline,
and Config.resolve_embed_providers with real/mocked environments.

Targets: bridge.py, config.py coverage improvement.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from codexlens_search.bridge import (
    create_config_from_env,
    create_pipeline,
    should_exclude,
    DEFAULT_EXCLUDES,
)
from codexlens_search.config import Config


class TestShouldExclude:
    """Test should_exclude path filtering."""

    def test_exclude_node_modules(self):
        path = Path("node_modules/package/index.js")
        assert should_exclude(path, DEFAULT_EXCLUDES) is True

    def test_exclude_git(self):
        path = Path(".git/objects/abc123")
        assert should_exclude(path, DEFAULT_EXCLUDES) is True

    def test_exclude_pycache(self):
        path = Path("src/__pycache__/module.cpython-310.pyc")
        assert should_exclude(path, DEFAULT_EXCLUDES) is True

    def test_exclude_venv(self):
        path = Path(".venv/lib/python3.10/site-packages/pkg.py")
        assert should_exclude(path, DEFAULT_EXCLUDES) is True

    def test_normal_path_not_excluded(self):
        path = Path("src/auth/login.py")
        assert should_exclude(path, DEFAULT_EXCLUDES) is False

    def test_custom_exclude_set(self):
        path = Path("vendor/lib/thing.js")
        assert should_exclude(path, frozenset({"vendor"})) is True

    def test_empty_exclude_set(self):
        path = Path("node_modules/pkg.js")
        assert should_exclude(path, frozenset()) is False


class TestCreateConfigFromEnv:
    """Test create_config_from_env with various environment variables."""

    def test_default_config(self, tmp_path):
        """With no env vars, should create default config."""
        with mock.patch.dict(os.environ, {}, clear=True):
            config = create_config_from_env(tmp_path)

        assert config.embed_dim == 384
        assert config.embed_model == "BAAI/bge-small-en-v1.5"
        assert config.metadata_db_path.endswith("metadata.db")

    def test_embed_api_from_env(self, tmp_path):
        env = {
            "CODEXLENS_EMBED_API_URL": "https://api.example.com/v1",
            "CODEXLENS_EMBED_API_KEY": "test-key-123",
            "CODEXLENS_EMBED_API_MODEL": "text-embedding-3-small",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = create_config_from_env(tmp_path)

        assert config.embed_api_url == "https://api.example.com/v1"
        assert config.embed_api_key == "test-key-123"
        assert config.embed_api_model == "text-embedding-3-small"

    def test_embed_dim_from_env(self, tmp_path):
        with mock.patch.dict(os.environ, {"CODEXLENS_EMBED_DIM": "768"}, clear=True):
            config = create_config_from_env(tmp_path)
        assert config.embed_dim == 768

    def test_overrides_take_precedence(self, tmp_path):
        """Explicit overrides should take precedence over env vars."""
        env = {"CODEXLENS_EMBED_API_URL": "https://env.example.com/v1"}
        with mock.patch.dict(os.environ, env, clear=True):
            config = create_config_from_env(
                tmp_path, embed_api_url="https://override.example.com/v1"
            )
        assert config.embed_api_url == "https://override.example.com/v1"

    def test_multi_endpoint_from_env(self, tmp_path):
        env = {
            "CODEXLENS_EMBED_API_ENDPOINTS": "url1|key1|model1,url2|key2|model2",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = create_config_from_env(tmp_path)

        assert len(config.embed_api_endpoints) == 2
        assert config.embed_api_endpoints[0]["url"] == "url1"
        assert config.embed_api_endpoints[0]["key"] == "key1"
        assert config.embed_api_endpoints[0]["model"] == "model1"
        assert config.embed_api_endpoints[1]["url"] == "url2"

    def test_shard_config_from_env(self, tmp_path):
        env = {
            "CODEXLENS_NUM_SHARDS": "4",
            "CODEXLENS_MAX_LOADED_SHARDS": "2",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = create_config_from_env(tmp_path)

        assert config.num_shards == 4
        assert config.max_loaded_shards == 2

    def test_search_quality_from_env(self, tmp_path):
        env = {"CODEXLENS_SEARCH_QUALITY": "fast"}
        with mock.patch.dict(os.environ, env, clear=True):
            config = create_config_from_env(tmp_path)

        assert config.default_search_quality == "fast"

    def test_reranker_api_from_env(self, tmp_path):
        env = {
            "CODEXLENS_RERANKER_API_URL": "https://rerank.example.com/v1",
            "CODEXLENS_RERANKER_API_KEY": "rk-123",
            "CODEXLENS_RERANKER_API_MODEL": "rerank-model",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = create_config_from_env(tmp_path)

        assert config.reranker_api_url == "https://rerank.example.com/v1"
        assert config.reranker_api_key == "rk-123"
        assert config.reranker_api_model == "rerank-model"

    @mock.patch("codexlens_search.config.Config._uses_gpu", return_value=False)
    def test_indexing_params_from_env(self, _mock_gpu, tmp_path):
        env = {
            "CODEXLENS_CODE_AWARE_CHUNKING": "true",
            "CODEXLENS_INDEX_WORKERS": "4",
            "CODEXLENS_MAX_FILE_SIZE": "5000000",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = create_config_from_env(tmp_path)

        assert config.code_aware_chunking is True
        assert config.index_workers == 4
        assert config.max_file_size_bytes == 5_000_000

    def test_search_params_from_env(self, tmp_path):
        env = {
            "CODEXLENS_BINARY_TOP_K": "100",
            "CODEXLENS_ANN_TOP_K": "30",
            "CODEXLENS_FTS_TOP_K": "25",
            "CODEXLENS_FUSION_K": "40",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = create_config_from_env(tmp_path)

        assert config.binary_top_k == 100
        assert config.ann_top_k == 30
        assert config.fts_top_k == 25
        assert config.fusion_k == 40

    def test_ast_and_gitignore_from_env(self, tmp_path):
        env = {
            "CODEXLENS_AST_CHUNKING": "true",
            "CODEXLENS_GITIGNORE_FILTERING": "yes",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = create_config_from_env(tmp_path)

        assert config.ast_chunking is True
        assert config.gitignore_filtering is True

    def test_hnsw_params_from_env(self, tmp_path):
        env = {
            "CODEXLENS_HNSW_EF": "200",
            "CODEXLENS_HNSW_M": "48",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = create_config_from_env(tmp_path)

        assert config.hnsw_ef == 200
        assert config.hnsw_M == 48

    def test_tier_config_from_env(self, tmp_path):
        env = {
            "CODEXLENS_TIER_HOT_HOURS": "12",
            "CODEXLENS_TIER_COLD_HOURS": "336",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = create_config_from_env(tmp_path)

        assert config.tier_hot_hours == 12
        assert config.tier_cold_hours == 336

    def test_embed_batch_concurrency_from_env(self, tmp_path):
        env = {
            "CODEXLENS_EMBED_BATCH_SIZE": "64",
            "CODEXLENS_EMBED_API_CONCURRENCY": "8",
            "CODEXLENS_EMBED_API_MAX_TOKENS": "65536",
            "CODEXLENS_EMBED_MAX_TOKENS": "4096",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = create_config_from_env(tmp_path)

        assert config.embed_batch_size == 64
        assert config.embed_api_concurrency == 8
        assert config.embed_api_max_tokens_per_batch == 65536
        assert config.embed_max_tokens == 4096

    def test_reranker_params_from_env(self, tmp_path):
        env = {
            "CODEXLENS_RERANKER_TOP_K": "15",
            "CODEXLENS_RERANKER_BATCH_SIZE": "16",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = create_config_from_env(tmp_path)

        assert config.reranker_top_k == 15
        assert config.reranker_batch_size == 16


class TestConfigResolveEmbedProviders:
    """Test Config.resolve_embed_providers with various device settings."""

    def test_explicit_providers(self):
        config = Config(embed_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        providers = config.resolve_embed_providers()
        assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_device_cuda(self):
        config = Config(device="cuda")
        providers = config.resolve_embed_providers()
        assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_device_cpu(self):
        config = Config(device="cpu")
        providers = config.resolve_embed_providers()
        assert providers == ["CPUExecutionProvider"]

    def test_device_auto_no_onnxruntime(self):
        """When onnxruntime is not available, auto should fallback to CPU."""
        config = Config(device="auto")
        with mock.patch.dict("sys.modules", {"onnxruntime": None}):
            providers = config.resolve_embed_providers()
        assert providers == ["CPUExecutionProvider"]


class TestConfigPostInit:
    """Test Config.__post_init__ defaults."""

    def test_default_exclude_extensions(self):
        config = Config()
        assert ".png" in config.exclude_extensions
        assert ".pyc" in config.exclude_extensions
        assert ".exe" in config.exclude_extensions

    def test_custom_exclude_extensions(self):
        custom = frozenset({".custom"})
        config = Config(exclude_extensions=custom)
        assert config.exclude_extensions == custom
        assert ".png" not in config.exclude_extensions

    def test_default_embed_api_endpoints(self):
        config = Config()
        assert config.embed_api_endpoints == []

    def test_small_config(self):
        config = Config.small()
        assert config.hnsw_ef == 50
        assert config.hnsw_M == 16
        assert config.binary_top_k == 50
        assert config.ann_top_k == 20
        assert config.reranker_top_k == 10


class TestBinaryStorePersistence:
    """Test BinaryStore save/load cycle (integration with filesystem)."""

    def test_save_and_load(self, tmp_path):
        from codexlens_search.core.binary import BinaryStore

        config = Config(embed_dim=32, binary_top_k=10)
        store = BinaryStore(tmp_path / "bs", dim=32, config=config)

        rng = np.random.default_rng(42)
        ids = np.arange(20, dtype=np.int64)
        vecs = rng.standard_normal((20, 32)).astype(np.float32)

        store.add(ids, vecs)
        store.save()

        # Create new store from same path - should load from disk
        store2 = BinaryStore(tmp_path / "bs", dim=32, config=config)
        assert len(store2) == 20

        result_ids, dists = store2.coarse_search(vecs[0], top_k=5)
        assert len(result_ids) > 0
        assert 0 in result_ids

    def test_capacity_growth(self, tmp_path):
        """Test that BinaryStore grows capacity as needed."""
        from codexlens_search.core.binary import BinaryStore

        config = Config(embed_dim=32, binary_top_k=10)
        store = BinaryStore(tmp_path / "bs_grow", dim=32, config=config)

        rng = np.random.default_rng(42)

        # Add in multiple batches to trigger capacity growth
        for batch in range(5):
            ids = np.arange(batch * 500, (batch + 1) * 500, dtype=np.int64)
            vecs = rng.standard_normal((500, 32)).astype(np.float32)
            store.add(ids, vecs)

        assert len(store) == 2500

        # Search should work across all added vectors
        query = rng.standard_normal(32).astype(np.float32)
        result_ids, dists = store.coarse_search(query, top_k=10)
        assert len(result_ids) == 10


class TestANNIndexPersistence:
    """Test ANNIndex save/load cycle."""

    def test_save_and_load(self, tmp_path):
        from codexlens_search.core.index import ANNIndex

        config = Config(embed_dim=32, hnsw_ef=50, hnsw_M=16, ann_top_k=10)
        ann_dir = tmp_path / "ann_dir"
        idx = ANNIndex(ann_dir, dim=32, config=config)

        rng = np.random.default_rng(42)
        ids = np.arange(20, dtype=np.int64)
        vecs = rng.standard_normal((20, 32)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms

        idx.add(ids, vecs)
        idx.save()

        # Create new index from same path - load() is lazy, trigger it
        idx2 = ANNIndex(ann_dir, dim=32, config=config)
        idx2.load()
        assert len(idx2) == 20

        result_ids, scores = idx2.fine_search(vecs[0], top_k=5)
        assert len(result_ids) > 0


# Import numpy for BinaryStore/ANNIndex tests
import numpy as np
