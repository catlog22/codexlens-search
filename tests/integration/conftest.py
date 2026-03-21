import pytest
import numpy as np
import tempfile
from pathlib import Path

from codexlens_search.config import Config
from codexlens_search.core import ANNIndex, BinaryStore
from codexlens_search.embed.base import BaseEmbedder
from codexlens_search.rerank.base import BaseReranker
from codexlens_search.indexing.metadata import MetadataStore
from codexlens_search.search.fts import FTSEngine
from codexlens_search.search.pipeline import SearchPipeline

# Test documents: 20 code snippets with id, path, content
TEST_DOCS = [
    (0, "auth.py", "def authenticate(user, password): return check_hash(password, user.hash)"),
    (1, "auth.py", "def authorize(user, permission): return permission in user.roles"),
    (2, "models.py", "class User: def __init__(self, name, email): self.name = name; self.email = email"),
    (3, "models.py", "class Session: token = None; expires_at = None"),
    (4, "middleware.py", "def auth_middleware(request): token = request.headers.get('Authorization')"),
    (5, "utils.py", "def hash_password(password): import bcrypt; return bcrypt.hashpw(password)"),
    (6, "config.py", "DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///db.sqlite3')"),
    (7, "search.py", "def search_users(query): return User.objects.filter(name__icontains=query)"),
    (8, "api.py", "def get_user(request, user_id): user = User.objects.get(id=user_id)"),
    (9, "api.py", "def create_user(request): data = request.json(); user = User(**data)"),
    (10, "tests.py", "def test_authenticate(): assert authenticate('admin', 'pass') is not None"),
    (11, "tests.py", "def test_search(): results = search_users('alice'); assert len(results) > 0"),
    (12, "router.py", "app.route('/users', methods=['GET'])(list_users)"),
    (13, "router.py", "app.route('/login', methods=['POST'])(login_handler)"),
    (14, "db.py", "def get_connection(): return sqlite3.connect(DATABASE_URL)"),
    (15, "cache.py", "def cache_get(key): return redis_client.get(key)"),
    (16, "cache.py", "def cache_set(key, value, ttl=3600): redis_client.setex(key, ttl, value)"),
    (17, "errors.py", "class AuthError(Exception): status_code = 401"),
    (18, "errors.py", "class NotFoundError(Exception): status_code = 404"),
    (19, "validators.py", "def validate_email(email): return '@' in email and '.' in email.split('@')[1]"),
]

DIM = 32  # Use small dim for fast tests


def make_stable_vec(doc_id: int, dim: int = DIM) -> np.ndarray:
    """Generate a deterministic float32 vector for a given doc_id."""
    rng = np.random.default_rng(seed=doc_id)
    vec = rng.standard_normal(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


class MockEmbedder(BaseEmbedder):
    """Returns stable deterministic vectors based on content hash."""

    def embed_single(self, text: str) -> np.ndarray:
        seed = hash(text) % (2**31)
        rng = np.random.default_rng(seed=seed)
        vec = rng.standard_normal(DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed_single(t) for t in texts]

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Called by SearchPipeline as self._embedder.embed([query])[0]."""
        return self.embed_batch(texts)


class MockReranker(BaseReranker):
    """Returns score based on simple keyword overlap."""

    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        query_words = set(query.lower().split())
        scores = []
        for doc in documents:
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            scores.append(float(overlap) / max(len(query_words), 1))
        return scores


@pytest.fixture
def config():
    return Config.small()  # hnsw_ef=50, hnsw_M=16, binary_top_k=50, ann_top_k=20, rerank_top_k=10


@pytest.fixture
def mock_embedder():
    return MockEmbedder()


@pytest.fixture
def mock_reranker():
    return MockReranker()


@pytest.fixture
def fts_engine(tmp_path):
    fts = FTSEngine(tmp_path / "fts.db")
    yield fts
    fts.close()


@pytest.fixture
def metadata_store(tmp_path):
    store = MetadataStore(tmp_path / "metadata.db")
    yield store
    store.close()


@pytest.fixture
def search_pipeline(tmp_path, config):
    """Build a full SearchPipeline with 20 test docs indexed."""
    embedder = MockEmbedder()
    binary_store = BinaryStore(tmp_path / "binary", dim=DIM, config=config)
    ann_index = ANNIndex(tmp_path / "ann.hnsw", dim=DIM, config=config)
    fts = FTSEngine(tmp_path / "fts.db")
    reranker = MockReranker()

    # Index all test docs
    ids = np.array([d[0] for d in TEST_DOCS], dtype=np.int64)
    vectors = np.array([embedder.embed_single(d[2]) for d in TEST_DOCS], dtype=np.float32)

    binary_store.add(ids, vectors)
    ann_index.add(ids, vectors)
    fts.add_documents(TEST_DOCS)

    pipeline = SearchPipeline(
        embedder=embedder,
        binary_store=binary_store,
        ann_index=ann_index,
        reranker=reranker,
        fts=fts,
        config=config,
    )
    yield pipeline
    pipeline.close()
