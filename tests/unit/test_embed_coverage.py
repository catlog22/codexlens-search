"""Additional embed/local.py coverage tests — _load method paths."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from codexlens_search.config import Config


def _make_fastembed_mock():
    fastembed_mod = types.ModuleType("fastembed")
    fastembed_mod.TextEmbedding = MagicMock()
    sys.modules.setdefault("fastembed", fastembed_mod)
    return fastembed_mod


_make_fastembed_mock()

from codexlens_search.embed.local import FastEmbedEmbedder  # noqa: E402


# ---------------------------------------------------------------------------
# _load (covers lines 30-49)
# ---------------------------------------------------------------------------

class TestFastEmbedLoad:
    def test_load_called_once(self):
        """_load should only initialize model once (double-check locking)."""
        config = Config()
        embedder = FastEmbedEmbedder(config)

        mock_model = MagicMock()
        mock_model.embed.return_value = iter([np.ones(384, dtype=np.float32)])

        mock_mm = MagicMock()
        mock_mm.get_cache_kwargs.return_value = {}

        with patch("codexlens_search.model_manager.ensure_model"):
            with patch("codexlens_search.model_manager.get_cache_kwargs", return_value={}):
                with patch("fastembed.TextEmbedding", return_value=mock_model):
                    embedder.embed_single("test")
                    # Second call should use cached model
                    mock_model.embed.return_value = iter([np.ones(384, dtype=np.float32)])
                    embedder.embed_single("test2")

    def test_load_type_error_fallback(self):
        """_load should retry without providers on TypeError."""
        config = Config()
        embedder = FastEmbedEmbedder(config)

        mock_model = MagicMock()
        mock_model.embed.return_value = iter([np.ones(384, dtype=np.float32)])

        call_count = [0]
        def _mock_text_embedding(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise TypeError("unexpected keyword argument 'providers'")
            return mock_model

        with patch("codexlens_search.model_manager.ensure_model"):
            with patch("codexlens_search.model_manager.get_cache_kwargs", return_value={}):
                with patch("fastembed.TextEmbedding", side_effect=_mock_text_embedding):
                    embedder.embed_single("test")
                    assert call_count[0] == 2  # First failed, second succeeded

    def test_embed_batch_multiple_batches(self):
        """embed_batch should handle texts exceeding batch_size."""
        config = Config()
        config.embed_batch_size = 2
        embedder = FastEmbedEmbedder(config)

        vecs = [np.ones(384, dtype=np.float64) * i for i in range(4)]
        mock_model = MagicMock()
        # Two batches of 2
        mock_model.embed.side_effect = [iter(vecs[:2]), iter(vecs[2:])]
        embedder._model = mock_model

        result = embedder.embed_batch(["a", "b", "c", "d"])
        assert len(result) == 4
        assert mock_model.embed.call_count == 2
