"""SQLite-backed metadata store for file-to-chunk mapping and tombstone tracking."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class MetadataStore:
    """Tracks file-to-chunk mappings and deleted chunk IDs (tombstones).

    Tables:
        files      - file_path (PK), content_hash, last_modified, file_size,
                     tier ('hot'/'warm'/'cold'), last_accessed (epoch float)
        chunks     - chunk_id (PK), file_path (FK CASCADE), chunk_hash
        deleted_chunks - chunk_id (PK) for tombstone tracking
    """

    def __init__(self, db_path: str | Path) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._create_tables()
        self._migrate_size_column()
        self._migrate_tier_columns()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS files (
                file_path   TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                last_modified REAL NOT NULL,
                file_size   INTEGER NOT NULL DEFAULT 0,
                tier        TEXT NOT NULL DEFAULT 'warm',
                last_accessed REAL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id    INTEGER PRIMARY KEY,
                file_path   TEXT NOT NULL,
                chunk_hash  TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (file_path) REFERENCES files(file_path) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS deleted_chunks (
                chunk_id    INTEGER PRIMARY KEY
            );
        """)
        self._conn.commit()

    def _migrate_size_column(self) -> None:
        """Add file_size column if missing (for pre-existing DBs)."""
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(files)").fetchall()
        }
        if "file_size" not in cols:
            self._conn.execute(
                "ALTER TABLE files ADD COLUMN file_size INTEGER NOT NULL DEFAULT 0"
            )
            self._conn.commit()

    def _migrate_tier_columns(self) -> None:
        """Add tier and last_accessed columns if missing (for pre-existing DBs)."""
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(files)").fetchall()
        }
        if "tier" not in cols:
            self._conn.execute(
                "ALTER TABLE files ADD COLUMN tier TEXT NOT NULL DEFAULT 'warm'"
            )
        if "last_accessed" not in cols:
            self._conn.execute(
                "ALTER TABLE files ADD COLUMN last_accessed REAL"
            )
        if "tier" not in cols or "last_accessed" not in cols:
            self._conn.commit()

    def register_file(
        self,
        file_path: str,
        content_hash: str,
        mtime: float,
        file_size: int = 0,
    ) -> None:
        """Insert or update a file record."""
        self._conn.execute(
            "INSERT OR REPLACE INTO files "
            "(file_path, content_hash, last_modified, file_size) "
            "VALUES (?, ?, ?, ?)",
            (file_path, content_hash, mtime, file_size),
        )

    def register_chunks(
        self, file_path: str, chunk_ids_and_hashes: list[tuple[int, str]]
    ) -> None:
        """Register chunk IDs belonging to a file.

        Args:
            file_path: The owning file path (must already exist in files table).
            chunk_ids_and_hashes: List of (chunk_id, chunk_hash) tuples.
        """
        if not chunk_ids_and_hashes:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO chunks (chunk_id, file_path, chunk_hash) "
            "VALUES (?, ?, ?)",
            [(cid, file_path, chash) for cid, chash in chunk_ids_and_hashes],
        )

    def flush(self) -> None:
        """Commit pending writes to disk."""
        self._conn.commit()

    def mark_file_deleted(self, file_path: str) -> int:
        """Move all chunk IDs for a file to deleted_chunks, then remove the file.

        Returns the number of chunks tombstoned.
        """
        # Collect chunk IDs before CASCADE deletes them
        rows = self._conn.execute(
            "SELECT chunk_id FROM chunks WHERE file_path = ?", (file_path,)
        ).fetchall()

        if not rows:
            # Still remove the file record if it exists
            self._conn.execute(
                "DELETE FROM files WHERE file_path = ?", (file_path,)
            )
            self._conn.commit()
            return 0

        chunk_ids = [(r[0],) for r in rows]
        self._conn.executemany(
            "INSERT OR IGNORE INTO deleted_chunks (chunk_id) VALUES (?)",
            chunk_ids,
        )
        # CASCADE deletes chunks rows automatically
        self._conn.execute(
            "DELETE FROM files WHERE file_path = ?", (file_path,)
        )
        self._conn.commit()
        return len(chunk_ids)

    def get_deleted_ids(self) -> set[int]:
        """Return all tombstoned chunk IDs for search-time filtering."""
        rows = self._conn.execute(
            "SELECT chunk_id FROM deleted_chunks"
        ).fetchall()
        return {r[0] for r in rows}

    def get_file_hash(self, file_path: str) -> str | None:
        """Return the stored content hash for a file, or None if not tracked."""
        row = self._conn.execute(
            "SELECT content_hash FROM files WHERE file_path = ?", (file_path,)
        ).fetchone()
        return row[0] if row else None

    def file_needs_update(self, file_path: str, content_hash: str) -> bool:
        """Check if a file needs re-indexing based on its content hash."""
        stored = self.get_file_hash(file_path)
        if stored is None:
            return True  # New file
        return stored != content_hash

    def file_needs_update_fast(
        self, file_path: str, mtime: float, size: int
    ) -> bool:
        """Fast pre-check using mtime and file size (no content read needed).

        Returns True if the file appears changed or is not yet tracked.
        When mtime and size both match stored values, the file is assumed
        unchanged (~1000x faster than content-hash comparison).
        """
        row = self._conn.execute(
            "SELECT last_modified, file_size FROM files WHERE file_path = ?",
            (file_path,),
        ).fetchone()
        if row is None:
            return True  # New file
        stored_mtime, stored_size = row
        return stored_mtime != mtime or stored_size != size

    def compact_deleted(self) -> set[int]:
        """Return deleted IDs and clear the deleted_chunks table.

        Call this after rebuilding the vector index to reclaim space.
        """
        deleted = self.get_deleted_ids()
        if deleted:
            self._conn.execute("DELETE FROM deleted_chunks")
            self._conn.commit()
        return deleted

    def get_chunk_ids_for_file(self, file_path: str) -> list[int]:
        """Return all chunk IDs belonging to a file."""
        rows = self._conn.execute(
            "SELECT chunk_id FROM chunks WHERE file_path = ?", (file_path,)
        ).fetchall()
        return [r[0] for r in rows]

    def get_all_files(self) -> dict[str, str]:
        """Return all tracked files as {file_path: content_hash}."""
        rows = self._conn.execute(
            "SELECT file_path, content_hash FROM files"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def max_chunk_id(self) -> int:
        """Return the maximum chunk_id across chunks and deleted_chunks.

        Returns -1 if no chunks exist, so that next_id = max_chunk_id() + 1
        starts at 0 for an empty store.
        """
        row = self._conn.execute(
            "SELECT MAX(m) FROM ("
            "  SELECT MAX(chunk_id) AS m FROM chunks"
            "  UNION ALL"
            "  SELECT MAX(chunk_id) AS m FROM deleted_chunks"
            ")"
        ).fetchone()
        return row[0] if row[0] is not None else -1

    # ------------------------------------------------------------------
    # Tier management
    # ------------------------------------------------------------------

    def record_access(self, file_path: str) -> None:
        """Update last_accessed timestamp for a file."""
        self._conn.execute(
            "UPDATE files SET last_accessed = ? WHERE file_path = ?",
            (time.time(), file_path),
        )
        self._conn.commit()

    def record_access_batch(self, file_paths: list[str]) -> None:
        """Batch-update last_accessed timestamps for multiple files."""
        if not file_paths:
            return
        now = time.time()
        self._conn.executemany(
            "UPDATE files SET last_accessed = ? WHERE file_path = ?",
            [(now, fp) for fp in file_paths],
        )
        self._conn.commit()

    def classify_tiers(
        self, hot_threshold_hours: int = 24, cold_threshold_hours: int = 168
    ) -> None:
        """Reclassify all files into hot/warm/cold tiers based on last_accessed.

        - hot:  last_accessed within hot_threshold_hours
        - cold: last_accessed older than cold_threshold_hours (or never accessed)
        - warm: everything in between
        """
        now = time.time()
        hot_cutoff = now - hot_threshold_hours * 3600
        cold_cutoff = now - cold_threshold_hours * 3600

        # Hot: recently accessed
        self._conn.execute(
            "UPDATE files SET tier = 'hot' "
            "WHERE last_accessed IS NOT NULL AND last_accessed >= ?",
            (hot_cutoff,),
        )
        # Cold: not accessed for a long time, or never accessed
        self._conn.execute(
            "UPDATE files SET tier = 'cold' "
            "WHERE last_accessed IS NULL "
            "OR (last_accessed < ? AND last_accessed < ?)",
            (cold_cutoff, hot_cutoff),
        )
        # Warm: between hot and cold cutoffs
        self._conn.execute(
            "UPDATE files SET tier = 'warm' "
            "WHERE last_accessed IS NOT NULL "
            "AND last_accessed >= ? AND last_accessed < ?",
            (cold_cutoff, hot_cutoff),
        )
        self._conn.commit()

    def get_files_by_tier(self, tier: str) -> list[str]:
        """Return file paths in the specified tier ('hot', 'warm', or 'cold')."""
        rows = self._conn.execute(
            "SELECT file_path FROM files WHERE tier = ?", (tier,)
        ).fetchall()
        return [r[0] for r in rows]

    def get_cold_files(self) -> list[str]:
        """Return file paths in the 'cold' tier."""
        return self.get_files_by_tier("cold")

    def get_file_tier(self, file_path: str) -> str | None:
        """Return the tier for a specific file, or None if not tracked."""
        row = self._conn.execute(
            "SELECT tier FROM files WHERE file_path = ?", (file_path,)
        ).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self._conn.close()
