"""Additional bridge.py coverage tests — env vars, config creation, pipeline helpers, main dispatch."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from codexlens_search.bridge import (
    create_config_from_env,
    should_exclude,
    DEFAULT_EXCLUDES,
    _build_parser,
)


# ---------------------------------------------------------------------------
# create_config_from_env — more env vars (covers lines 89-144)
# ---------------------------------------------------------------------------

class TestCreateConfigEnvVarsExtended:
    def test_embed_batch_size_env(self, tmp_path):
        with patch.dict(os.environ, {"CODEXLENS_EMBED_BATCH_SIZE": "64"}):
            config = create_config_from_env(tmp_path)
            assert config.embed_batch_size == 64

    def test_embed_api_concurrency_env(self, tmp_path):
        with patch.dict(os.environ, {"CODEXLENS_EMBED_API_CONCURRENCY": "8"}):
            config = create_config_from_env(tmp_path)
            assert config.embed_api_concurrency == 8

    def test_embed_api_max_tokens_env(self, tmp_path):
        with patch.dict(os.environ, {"CODEXLENS_EMBED_API_MAX_TOKENS": "16384"}):
            config = create_config_from_env(tmp_path)
            assert config.embed_api_max_tokens_per_batch == 16384

    def test_embed_max_tokens_env(self, tmp_path):
        with patch.dict(os.environ, {"CODEXLENS_EMBED_MAX_TOKENS": "4096"}):
            config = create_config_from_env(tmp_path)
            assert config.embed_max_tokens == 4096

    def test_reranker_env_vars(self, tmp_path):
        env = {
            "CODEXLENS_RERANKER_API_URL": "https://reranker.example.com",
            "CODEXLENS_RERANKER_API_KEY": "rr-key",
            "CODEXLENS_RERANKER_API_MODEL": "rr-model",
            "CODEXLENS_RERANKER_TOP_K": "20",
            "CODEXLENS_RERANKER_BATCH_SIZE": "16",
        }
        with patch.dict(os.environ, env):
            config = create_config_from_env(tmp_path)
            assert config.reranker_api_url == "https://reranker.example.com"
            assert config.reranker_api_key == "rr-key"
            assert config.reranker_api_model == "rr-model"
            assert config.reranker_top_k == 20
            assert config.reranker_batch_size == 16

    def test_gitignore_filtering_env(self, tmp_path):
        with patch.dict(os.environ, {"CODEXLENS_GITIGNORE_FILTERING": "true"}):
            config = create_config_from_env(tmp_path)
            assert config.gitignore_filtering is True

    def test_code_aware_chunking_env(self, tmp_path):
        with patch.dict(os.environ, {"CODEXLENS_CODE_AWARE_CHUNKING": "true"}):
            config = create_config_from_env(tmp_path)
            assert config.code_aware_chunking is True

    @patch("codexlens_search.config.Config._uses_gpu", return_value=False)
    def test_index_workers_env(self, _mock_gpu, tmp_path):
        with patch.dict(os.environ, {"CODEXLENS_INDEX_WORKERS": "4"}):
            config = create_config_from_env(tmp_path)
            assert config.index_workers == 4

    def test_max_file_size_env(self, tmp_path):
        with patch.dict(os.environ, {"CODEXLENS_MAX_FILE_SIZE": "2000000"}):
            config = create_config_from_env(tmp_path)
            assert config.max_file_size_bytes == 2000000

    def test_hnsw_params_env(self, tmp_path):
        env = {
            "CODEXLENS_HNSW_EF": "128",
            "CODEXLENS_HNSW_M": "32",
        }
        with patch.dict(os.environ, env):
            config = create_config_from_env(tmp_path)
            assert config.hnsw_ef == 128
            assert config.hnsw_M == 32

    def test_tier_config_env(self, tmp_path):
        env = {
            "CODEXLENS_TIER_HOT_HOURS": "48",
            "CODEXLENS_TIER_COLD_HOURS": "720",
        }
        with patch.dict(os.environ, env):
            config = create_config_from_env(tmp_path)
            assert config.tier_hot_hours == 48
            assert config.tier_cold_hours == 720

    def test_search_quality_env(self, tmp_path):
        with patch.dict(os.environ, {"CODEXLENS_SEARCH_QUALITY": "fast"}):
            config = create_config_from_env(tmp_path)
            assert config.default_search_quality == "fast"

    def test_shard_config_env(self, tmp_path):
        env = {
            "CODEXLENS_NUM_SHARDS": "4",
            "CODEXLENS_MAX_LOADED_SHARDS": "2",
        }
        with patch.dict(os.environ, env):
            config = create_config_from_env(tmp_path)
            assert config.num_shards == 4
            assert config.max_loaded_shards == 2

    def test_multi_endpoint_single_entry(self, tmp_path):
        env = {"CODEXLENS_EMBED_API_ENDPOINTS": "url1|key1"}
        with patch.dict(os.environ, env):
            config = create_config_from_env(tmp_path)
            assert len(config.embed_api_endpoints) == 1
            assert "model" not in config.embed_api_endpoints[0]


# ---------------------------------------------------------------------------
# _create_config (covers lines 152-161)
# ---------------------------------------------------------------------------

class TestCreateConfig:
    def test_create_config_with_all_overrides(self, tmp_path):
        from codexlens_search.bridge import _create_config
        args = argparse.Namespace(
            db_path=str(tmp_path),
            embed_model="custom-model",
            embed_api_url="https://custom.api.com",
            embed_api_key="custom-key",
            embed_api_model="custom-api-model",
        )
        config = _create_config(args)
        assert config.embed_api_url == "https://custom.api.com"
        assert config.embed_api_key == "custom-key"
        assert config.embed_api_model == "custom-api-model"

    def test_create_config_no_overrides(self, tmp_path):
        from codexlens_search.bridge import _create_config
        args = argparse.Namespace(db_path=str(tmp_path))
        config = _create_config(args)
        assert config.embed_api_url == ""


# ---------------------------------------------------------------------------
# _create_embedder / _create_reranker (covers lines 166-192)
# ---------------------------------------------------------------------------

class TestCreateEmbedderReranker:
    def test_create_embedder_api(self, tmp_path):
        from codexlens_search.bridge import _create_embedder
        from codexlens_search.config import Config
        config = Config(embed_api_url="https://api.example.com/v1", embed_api_key="key", embed_api_model="model")
        with patch("codexlens_search.embed.api.APIEmbedder") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            mock_instance.embed_single.return_value = MagicMock(shape=(384,))
            result = _create_embedder(config)
            assert result is mock_instance

    def test_create_embedder_local(self, tmp_path):
        from codexlens_search.bridge import _create_embedder
        from codexlens_search.config import Config
        config = Config()
        with patch("codexlens_search.embed.local.FastEmbedEmbedder") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            result = _create_embedder(config)
            assert result is mock_instance

    def test_create_reranker_api(self):
        from codexlens_search.bridge import _create_reranker
        from codexlens_search.config import Config
        config = Config(reranker_api_url="https://reranker.example.com", reranker_api_key="key")
        with patch("codexlens_search.rerank.api.APIReranker") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            result = _create_reranker(config)
            assert result is mock_instance

    def test_create_reranker_local(self):
        from codexlens_search.bridge import _create_reranker
        from codexlens_search.config import Config
        config = Config()
        with patch("codexlens_search.rerank.local.FastEmbedReranker") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            result = _create_reranker(config)
            assert result is mock_instance


# ---------------------------------------------------------------------------
# main() dispatch (covers lines 640-688)
# ---------------------------------------------------------------------------

class TestMainDispatch:
    def test_main_no_command(self):
        from codexlens_search.bridge import main
        with patch("sys.argv", ["codexlens-search"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_main_unknown_command(self):
        from codexlens_search.bridge import main
        with patch("sys.argv", ["codexlens-search", "nonexistent"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # argparse exits with code 2 for invalid choice
            assert exc_info.value.code == 2

    def test_main_dispatches_init(self, tmp_path, capsys):
        from codexlens_search.bridge import main
        db_path = str(tmp_path / "test_db")
        with patch("sys.argv", ["codexlens-search", "--db-path", db_path, "init"]):
            main()
        out = json.loads(capsys.readouterr().out.strip())
        assert out["status"] == "initialized"

    def test_main_verbose_flag(self, tmp_path, capsys):
        from codexlens_search.bridge import main
        db_path = str(tmp_path / "test_db")
        with patch("sys.argv", ["codexlens-search", "--db-path", db_path, "-v", "init"]):
            main()
        out = json.loads(capsys.readouterr().out.strip())
        assert out["status"] == "initialized"

    def test_main_handler_exception(self, tmp_path):
        from codexlens_search.bridge import main
        with patch("sys.argv", ["codexlens-search", "--db-path", str(tmp_path), "status"]):
            with patch("codexlens_search.bridge.cmd_status", side_effect=ValueError("boom")):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# cmd_download_models / cmd_list_models / cmd_download_model / cmd_delete_model
# (covers lines 462-511)
# ---------------------------------------------------------------------------

class TestModelCommands:
    def test_cmd_download_models(self, tmp_path, capsys):
        from codexlens_search.bridge import cmd_download_models
        with patch("codexlens_search.model_manager.ensure_model"):
            args = argparse.Namespace(
                db_path=str(tmp_path),
                embed_model=None,
                embed_api_url="",
                embed_api_key="",
                embed_api_model="",
            )
            cmd_download_models(args)
        out = json.loads(capsys.readouterr().out.strip())
        assert out["status"] == "downloaded"

    def test_cmd_list_models(self, tmp_path, capsys):
        from codexlens_search.bridge import cmd_list_models
        with patch("codexlens_search.model_manager.list_known_models", return_value=[]):
            args = argparse.Namespace(
                db_path=str(tmp_path),
                embed_model=None,
                embed_api_url="",
                embed_api_key="",
                embed_api_model="",
            )
            cmd_list_models(args)
        out = json.loads(capsys.readouterr().out.strip())
        assert isinstance(out, list)

    def test_cmd_download_model(self, tmp_path, capsys):
        from codexlens_search.bridge import cmd_download_model
        with patch("codexlens_search.model_manager.ensure_model"):
            with patch("codexlens_search.model_manager._model_is_cached", return_value=True):
                with patch("codexlens_search.model_manager._resolve_cache_dir", return_value=str(tmp_path)):
                    args = argparse.Namespace(
                        db_path=str(tmp_path),
                        model_name="BAAI/bge-small-en-v1.5",
                        embed_model=None,
                        embed_api_url="",
                        embed_api_key="",
                        embed_api_model="",
                    )
                    cmd_download_model(args)
        out = json.loads(capsys.readouterr().out.strip())
        assert out["status"] == "downloaded"

    def test_cmd_delete_model(self, tmp_path, capsys):
        from codexlens_search.bridge import cmd_delete_model
        with patch("codexlens_search.model_manager.delete_model", return_value=True):
            args = argparse.Namespace(
                db_path=str(tmp_path),
                model_name="BAAI/bge-small-en-v1.5",
                embed_model=None,
                embed_api_url="",
                embed_api_key="",
                embed_api_model="",
            )
            cmd_delete_model(args)
        out = json.loads(capsys.readouterr().out.strip())
        assert out["status"] == "deleted"


# ---------------------------------------------------------------------------
# should_exclude additional edge cases
# ---------------------------------------------------------------------------

class TestShouldExcludeAdditional:
    def test_deeply_nested_venv(self):
        assert should_exclude(Path("project/.venv/lib/python3.11/site-packages/mod.py"), DEFAULT_EXCLUDES) is True

    def test_pytest_cache(self):
        assert should_exclude(Path(".pytest_cache/v/cache/abc"), DEFAULT_EXCLUDES) is True

    def test_mypy_cache(self):
        assert should_exclude(Path(".mypy_cache/3.11/module.json"), DEFAULT_EXCLUDES) is True

    def test_custom_excludes(self):
        custom = frozenset({"vendor"})
        assert should_exclude(Path("vendor/lib/foo.py"), custom) is True
        assert should_exclude(Path("src/lib/foo.py"), custom) is False
