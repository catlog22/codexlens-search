"""Unit tests for MetadataStore tier management methods."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from codexlens_search.indexing.metadata import MetadataStore


class TestRecordAccess:
    def test_record_access_updates_timestamp(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        store.register_file("a.py", "hash1", 1000.0, 100)
        store.flush()
        store.record_access("a.py")
        row = store._conn.execute(
            "SELECT last_accessed FROM files WHERE file_path = ?", ("a.py",)
        ).fetchone()
        assert row[0] is not None
        assert row[0] > 0

    def test_record_access_batch(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        store.register_file("a.py", "h1", 1000.0, 100)
        store.register_file("b.py", "h2", 1000.0, 200)
        store.flush()
        store.record_access_batch(["a.py", "b.py"])
        for fp in ("a.py", "b.py"):
            row = store._conn.execute(
                "SELECT last_accessed FROM files WHERE file_path = ?", (fp,)
            ).fetchone()
            assert row[0] is not None

    def test_record_access_batch_empty(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        store.record_access_batch([])  # Should not raise


class TestClassifyTiers:
    def test_hot_tier(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        store.register_file("hot.py", "h1", 1000.0, 100)
        store.flush()
        # Set last_accessed to now
        now = time.time()
        store._conn.execute(
            "UPDATE files SET last_accessed = ? WHERE file_path = ?",
            (now, "hot.py"),
        )
        store._conn.commit()
        store.classify_tiers(hot_threshold_hours=24, cold_threshold_hours=168)
        assert store.get_file_tier("hot.py") == "hot"

    def test_cold_tier_never_accessed(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        store.register_file("cold.py", "h1", 1000.0, 100)
        store.flush()
        # last_accessed is NULL -> cold
        store.classify_tiers(hot_threshold_hours=24, cold_threshold_hours=168)
        assert store.get_file_tier("cold.py") == "cold"

    def test_warm_tier(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        store.register_file("warm.py", "h1", 1000.0, 100)
        store.flush()
        # Set last_accessed to 3 days ago (between hot 24h and cold 168h)
        three_days_ago = time.time() - 3 * 24 * 3600
        store._conn.execute(
            "UPDATE files SET last_accessed = ? WHERE file_path = ?",
            (three_days_ago, "warm.py"),
        )
        store._conn.commit()
        store.classify_tiers(hot_threshold_hours=24, cold_threshold_hours=168)
        assert store.get_file_tier("warm.py") == "warm"

    def test_cold_tier_old_access(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        store.register_file("old.py", "h1", 1000.0, 100)
        store.flush()
        # Set last_accessed to 30 days ago (well past cold 168h)
        old_time = time.time() - 30 * 24 * 3600
        store._conn.execute(
            "UPDATE files SET last_accessed = ? WHERE file_path = ?",
            (old_time, "old.py"),
        )
        store._conn.commit()
        store.classify_tiers(hot_threshold_hours=24, cold_threshold_hours=168)
        assert store.get_file_tier("old.py") == "cold"


class TestGetFilesByTier:
    def test_get_files_by_tier(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        store.register_file("hot.py", "h1", 1000.0, 100)
        store.register_file("cold.py", "h2", 1000.0, 200)
        store.flush()
        # Make hot.py hot
        store._conn.execute(
            "UPDATE files SET tier = 'hot' WHERE file_path = ?", ("hot.py",)
        )
        store._conn.execute(
            "UPDATE files SET tier = 'cold' WHERE file_path = ?", ("cold.py",)
        )
        store._conn.commit()
        hot_files = store.get_files_by_tier("hot")
        cold_files = store.get_files_by_tier("cold")
        assert "hot.py" in hot_files
        assert "cold.py" in cold_files

    def test_get_cold_files(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        store.register_file("cold1.py", "h1", 1000.0, 100)
        store.flush()
        store._conn.execute(
            "UPDATE files SET tier = 'cold' WHERE file_path = ?", ("cold1.py",)
        )
        store._conn.commit()
        cold = store.get_cold_files()
        assert "cold1.py" in cold


class TestGetFileTier:
    def test_returns_tier(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        store.register_file("a.py", "h1", 1000.0, 100)
        store.flush()
        tier = store.get_file_tier("a.py")
        assert tier == "warm"  # default tier

    def test_returns_none_for_unknown(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.db")
        assert store.get_file_tier("nonexistent.py") is None
