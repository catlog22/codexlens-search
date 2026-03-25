from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


def _make_fastembed_mock():
    """Build a minimal fastembed stub so imports succeed without the real package."""
    fastembed_mod = types.ModuleType("fastembed")
    fastembed_mod.TextEmbedding = MagicMock()
    sys.modules.setdefault("fastembed", fastembed_mod)
    return fastembed_mod


_make_fastembed_mock()

from codexlens_search.config import Config  # noqa: E402
from codexlens_search.embed.base import BaseEmbedder  # noqa: E402
from codexlens_search.embed.local import EMBED_PROFILES, FastEmbedEmbedder  # noqa: E402
from codexlens_search.embed.api import APIEmbedder  # noqa: E402


class TestEmbedSingle(unittest.TestCase):
    def test_embed_single_returns_float32_ndarray(self):
        config = Config()
        embedder = FastEmbedEmbedder(config)

        mock_model = MagicMock()
        mock_model.embed.return_value = iter([np.ones(384, dtype=np.float64)])

        # Inject mock model directly to bypass lazy load (no real fastembed needed)
        embedder._model = mock_model
        result = embedder.embed_single("hello world")

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (384,))


class TestEmbedBatch(unittest.TestCase):
    def test_embed_batch_returns_list(self):
        config = Config()
        embedder = FastEmbedEmbedder(config)

        vecs = [np.ones(384, dtype=np.float64) * i for i in range(3)]
        mock_model = MagicMock()
        mock_model.embed.return_value = iter(vecs)

        embedder._model = mock_model
        result = embedder.embed_batch(["a", "b", "c"])

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        for arr in result:
            self.assertIsInstance(arr, np.ndarray)
            self.assertEqual(arr.dtype, np.float32)


class TestEmbedProfiles(unittest.TestCase):
    def test_embed_profiles_all_have_valid_keys(self):
        expected_keys = {"small", "base", "large", "code", "code-light", "long"}
        self.assertEqual(set(EMBED_PROFILES.keys()), expected_keys)

    def test_embed_profiles_model_ids_non_empty(self):
        for key, model_id in EMBED_PROFILES.items():
            self.assertIsInstance(model_id, str, msg=f"{key} model id should be str")
            self.assertTrue(len(model_id) > 0, msg=f"{key} model id should be non-empty")


class TestBaseEmbedderAbstract(unittest.TestCase):
    def test_base_embedder_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseEmbedder()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# APIEmbedder
# ---------------------------------------------------------------------------

def _make_api_config(**overrides) -> Config:
    defaults = dict(
        embed_api_url="https://api.example.com/v1",
        embed_api_key="test-key",
        embed_api_model="text-embedding-3-small",
        embed_dim=384,
        embed_batch_size=2,
        embed_api_max_tokens_per_batch=8192,
        embed_api_concurrency=2,
        device="cpu",
    )
    defaults.update(overrides)
    return Config(**defaults)


def _mock_200(count=1, dim=384):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "data": [{"index": j, "embedding": [0.1 * (j + 1)] * dim} for j in range(count)]
    }
    r.raise_for_status = MagicMock()
    return r


class TestAPIEmbedderSingle(unittest.TestCase):
    def test_embed_single_returns_float32(self):
        config = _make_api_config()
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = _mock_200(1, 384)

            embedder = APIEmbedder(config)
            result = embedder.embed_single("hello")

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (384,))


class TestAPIEmbedderBatch(unittest.TestCase):
    def test_embed_batch_splits_by_batch_size(self):
        config = _make_api_config(embed_batch_size=2)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.post.side_effect = [_mock_200(2, 384), _mock_200(1, 384)]

            embedder = APIEmbedder(config)
            result = embedder.embed_batch(["a", "b", "c"])

        self.assertEqual(len(result), 3)
        for arr in result:
            self.assertIsInstance(arr, np.ndarray)
            self.assertEqual(arr.dtype, np.float32)

    def test_embed_batch_empty_returns_empty(self):
        config = _make_api_config()
        with patch("httpx.Client"):
            embedder = APIEmbedder(config)
        result = embedder.embed_batch([])
        self.assertEqual(result, [])


class TestAPIEmbedderRetry(unittest.TestCase):
    def test_retry_on_429(self):
        config = _make_api_config()
        mock_429 = MagicMock()
        mock_429.status_code = 429

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.post.side_effect = [mock_429, _mock_200(1, 384)]

            embedder = APIEmbedder(config)
            ep = embedder._endpoints[0]
            with patch("time.sleep"):
                result = embedder._call_api(["test"], ep)

        self.assertEqual(len(result), 1)
        self.assertEqual(mock_client.post.call_count, 2)

    def test_raises_after_max_retries(self):
        config = _make_api_config()
        mock_429 = MagicMock()
        mock_429.status_code = 429

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = mock_429

            embedder = APIEmbedder(config)
            ep = embedder._endpoints[0]
            with patch("time.sleep"):
                with self.assertRaises(RuntimeError):
                    embedder._call_api(["test"], ep, max_retries=2)


class TestAPIEmbedderTokenPacking(unittest.TestCase):
    def test_packs_small_texts_together(self):
        config = _make_api_config(
            embed_batch_size=100,
            embed_api_max_tokens_per_batch=100,  # ~400 chars
        )
        with patch("httpx.Client"):
            embedder = APIEmbedder(config)

        # 5 texts of 80 chars each (~20 tokens) -> 100 tokens = 1 batch at limit
        texts = ["x" * 80] * 5
        batches = embedder._pack_batches(texts)
        # Should pack as many as fit under 100 tokens
        self.assertTrue(len(batches) >= 1)
        total_items = sum(len(b) for b in batches)
        self.assertEqual(total_items, 5)

    def test_large_text_gets_own_batch(self):
        config = _make_api_config(
            embed_batch_size=100,
            embed_api_max_tokens_per_batch=50,  # ~200 chars
        )
        with patch("httpx.Client"):
            embedder = APIEmbedder(config)

        # Mix of small and large texts
        texts = ["small" * 10, "x" * 800, "tiny"]
        batches = embedder._pack_batches(texts)
        # Large text (200 tokens) exceeds 50 limit, should be separate
        self.assertTrue(len(batches) >= 2)


class TestAPIEmbedderMultiEndpoint(unittest.TestCase):
    def test_multi_endpoint_config(self):
        config = _make_api_config(
            embed_api_endpoints=[
                {"url": "https://ep1.example.com/v1", "key": "k1", "model": "m1"},
                {"url": "https://ep2.example.com/v1", "key": "k2", "model": "m2"},
            ]
        )
        with patch("httpx.Client"):
            embedder = APIEmbedder(config)
        self.assertEqual(len(embedder._endpoints), 2)
        self.assertTrue(embedder._endpoints[0].url.endswith("/embeddings"))
        self.assertTrue(embedder._endpoints[1].url.endswith("/embeddings"))

    def test_single_endpoint_fallback(self):
        config = _make_api_config()  # no embed_api_endpoints
        with patch("httpx.Client"):
            embedder = APIEmbedder(config)
        self.assertEqual(len(embedder._endpoints), 1)


class TestAPIEmbedderUrlNormalization(unittest.TestCase):
    def test_appends_embeddings_path(self):
        config = _make_api_config(embed_api_url="https://api.example.com/v1")
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = _mock_200(1, 384)
            embedder = APIEmbedder(config)
            ep = embedder._endpoints[0]
        self.assertTrue(ep.url.endswith("/embeddings"))

    def test_does_not_double_append(self):
        config = _make_api_config(embed_api_url="https://api.example.com/v1/embeddings")
        with patch("httpx.Client"):
            embedder = APIEmbedder(config)
            ep = embedder._endpoints[0]
        self.assertFalse(ep.url.endswith("/embeddings/embeddings"))


if __name__ == "__main__":
    unittest.main()
