"""Unit tests for model_manager.py — cache detection, download, list, delete."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codexlens_search.config import Config
from codexlens_search.model_manager import (
    _default_fastembed_cache,
    _dir_has_onnx,
    _find_model_cache_path,
    _model_is_cached,
    _resolve_cache_dir,
    delete_model,
    ensure_model,
    get_cache_kwargs,
    list_known_models,
)


class TestResolveCacheDir:
    def test_returns_none_when_empty(self) -> None:
        cfg = Config()
        cfg.model_cache_dir = ""
        assert _resolve_cache_dir(cfg) is None

    def test_returns_configured_path(self) -> None:
        cfg = Config()
        cfg.model_cache_dir = "/custom/cache"
        assert _resolve_cache_dir(cfg) == "/custom/cache"


class TestDirHasOnnx:
    def test_no_snapshots_dir(self, tmp_path: Path) -> None:
        assert _dir_has_onnx(tmp_path) is False

    def test_empty_snapshots(self, tmp_path: Path) -> None:
        (tmp_path / "snapshots").mkdir()
        assert _dir_has_onnx(tmp_path) is False

    def test_snapshots_with_onnx(self, tmp_path: Path) -> None:
        snap = tmp_path / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / "model.onnx").touch()
        assert _dir_has_onnx(tmp_path) is True

    def test_snapshots_without_onnx(self, tmp_path: Path) -> None:
        snap = tmp_path / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / "config.json").touch()
        assert _dir_has_onnx(tmp_path) is False


class TestModelIsCached:
    def test_nonexistent_cache_dir(self) -> None:
        assert _model_is_cached("BAAI/bge-small-en-v1.5", "/nonexistent/path") is False

    def test_exact_match(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "models--BAAI--bge-small-en-v1.5"
        snap = model_dir / "snapshots" / "rev1"
        snap.mkdir(parents=True)
        (snap / "model.onnx").touch()
        assert _model_is_cached("BAAI/bge-small-en-v1.5", str(tmp_path)) is True

    def test_partial_match(self, tmp_path: Path) -> None:
        # fastembed remaps: BAAI/bge-small-en-v1.5 -> qdrant/bge-small-en-v1.5-onnx-q
        model_dir = tmp_path / "models--qdrant--bge-small-en-v1.5-onnx-q"
        snap = model_dir / "snapshots" / "rev1"
        snap.mkdir(parents=True)
        (snap / "model.onnx").touch()
        assert _model_is_cached("BAAI/bge-small-en-v1.5", str(tmp_path)) is True

    def test_no_match(self, tmp_path: Path) -> None:
        assert _model_is_cached("BAAI/bge-large-en-v1.5", str(tmp_path)) is False


class TestFindModelCachePath:
    def test_returns_none_for_nonexistent(self, tmp_path: Path) -> None:
        assert _find_model_cache_path("missing/model", str(tmp_path)) is None

    def test_returns_path_for_exact_match(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "models--BAAI--bge-small-en-v1.5"
        snap = model_dir / "snapshots" / "rev1"
        snap.mkdir(parents=True)
        (snap / "model.onnx").touch()
        result = _find_model_cache_path("BAAI/bge-small-en-v1.5", str(tmp_path))
        assert result == str(model_dir)


class TestEnsureModel:
    def test_cached_model_is_noop(self, tmp_path: Path) -> None:
        cfg = Config()
        cfg.model_cache_dir = str(tmp_path)
        # Create cached model
        model_dir = tmp_path / "models--BAAI--bge-small-en-v1.5"
        snap = model_dir / "snapshots" / "rev1"
        snap.mkdir(parents=True)
        (snap / "model.onnx").touch()
        # Should not attempt download
        ensure_model("BAAI/bge-small-en-v1.5", cfg)

    @patch("codexlens_search.model_manager._model_is_cached", return_value=False)
    def test_download_when_not_cached(self, mock_cached, tmp_path: Path) -> None:
        cfg = Config()
        cfg.model_cache_dir = str(tmp_path)
        cfg.hf_mirror = ""
        # huggingface_hub not installed -> should log warning, not crash
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            ensure_model("test/model", cfg)


class TestDeleteModel:
    def test_delete_existing_model(self, tmp_path: Path) -> None:
        cfg = Config()
        cfg.model_cache_dir = str(tmp_path)
        model_dir = tmp_path / "models--test--model"
        snap = model_dir / "snapshots" / "rev1"
        snap.mkdir(parents=True)
        (snap / "model.onnx").touch()
        result = delete_model("test/model", cfg)
        assert result is True
        assert not model_dir.exists()

    def test_delete_nonexistent_returns_false(self, tmp_path: Path) -> None:
        cfg = Config()
        cfg.model_cache_dir = str(tmp_path)
        result = delete_model("missing/model", cfg)
        assert result is False


class TestListKnownModels:
    def test_returns_list(self, tmp_path: Path) -> None:
        cfg = Config()
        cfg.model_cache_dir = str(tmp_path)
        models = list_known_models(cfg)
        assert isinstance(models, list)
        assert len(models) > 0
        # Should contain default embed and reranker models
        names = {m["name"] for m in models}
        assert cfg.embed_model in names
        assert cfg.reranker_model in names

    def test_installed_flag_correct(self, tmp_path: Path) -> None:
        cfg = Config()
        cfg.model_cache_dir = str(tmp_path)
        # Create cache for default embed model
        model_dir = tmp_path / f"models--{cfg.embed_model.replace('/', '--')}"
        snap = model_dir / "snapshots" / "rev1"
        snap.mkdir(parents=True)
        (snap / "model.onnx").touch()
        models = list_known_models(cfg)
        embed_entry = next(m for m in models if m["name"] == cfg.embed_model)
        assert embed_entry["installed"] is True
        assert embed_entry["type"] == "embedding"


class TestGetCacheKwargs:
    def test_empty_cache_dir(self) -> None:
        cfg = Config()
        cfg.model_cache_dir = ""
        assert get_cache_kwargs(cfg) == {}

    def test_custom_cache_dir(self) -> None:
        cfg = Config()
        cfg.model_cache_dir = "/custom"
        assert get_cache_kwargs(cfg) == {"cache_dir": "/custom"}
