"""Shared fixtures for E2E tests."""
import pytest
import shutil
from pathlib import Path

# Fixtures directory containing sample Python files for indexing
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    """Return path to the test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def project_dir(tmp_path, fixtures_dir):
    """Create a temporary project directory with fixture files copied in.

    Returns the tmp_path containing a 'src/' subdirectory with all fixture .py files.
    """
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    for f in fixtures_dir.glob("*.py"):
        shutil.copy2(f, src_dir / f.name)
    return tmp_path


@pytest.fixture
def db_path(tmp_path):
    """Return a temporary database path for index storage."""
    p = tmp_path / ".codexlens"
    p.mkdir()
    return p


def _model_available() -> bool:
    """Check if fastembed model is downloadable/cached."""
    try:
        from fastembed import TextEmbedding
        TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return True
    except Exception:
        return False


requires_model = pytest.mark.skipif(
    not _model_available(),
    reason="fastembed model BAAI/bge-small-en-v1.5 not available",
)
