from __future__ import annotations

import sqlite3
from pathlib import Path


class FTSEngine:
    def __init__(self, db_path: str | Path) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS docs "
            "USING fts5(content, tokenize='porter unicode61')"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS docs_meta "
            "(id INTEGER PRIMARY KEY, path TEXT, "
            "start_line INTEGER DEFAULT 0, end_line INTEGER DEFAULT 0)"
        )
        self._conn.commit()
        self._migrate_line_columns()

    def _migrate_line_columns(self) -> None:
        """Add start_line/end_line columns if missing (for pre-existing DBs)."""
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(docs_meta)").fetchall()
        }
        for col in ("start_line", "end_line"):
            if col not in cols:
                self._conn.execute(
                    f"ALTER TABLE docs_meta ADD COLUMN {col} INTEGER DEFAULT 0"
                )
        self._conn.commit()

    def add_documents(self, docs: list[tuple]) -> None:
        """Add documents in batch.

        docs: list of (id, path, content) or (id, path, content, start_line, end_line).
        """
        if not docs:
            return
        meta_rows = []
        fts_rows = []
        for doc in docs:
            if len(doc) >= 5:
                doc_id, path, content, sl, el = doc[0], doc[1], doc[2], doc[3], doc[4]
            else:
                doc_id, path, content = doc[0], doc[1], doc[2]
                sl, el = 0, 0
            meta_rows.append((doc_id, path, sl, el))
            fts_rows.append((doc_id, content))
        self._conn.executemany(
            "INSERT OR REPLACE INTO docs_meta (id, path, start_line, end_line) "
            "VALUES (?, ?, ?, ?)",
            meta_rows,
        )
        self._conn.executemany(
            "INSERT OR REPLACE INTO docs (rowid, content) VALUES (?, ?)",
            fts_rows,
        )
        self._conn.commit()

    def exact_search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        """FTS5 MATCH query, return (id, bm25_score) sorted by score descending."""
        try:
            rows = self._conn.execute(
                "SELECT rowid, bm25(docs) AS score FROM docs "
                "WHERE docs MATCH ? ORDER BY score LIMIT ?",
                (query, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        # bm25 in SQLite FTS5 returns negative values (lower = better match)
        # Negate so higher is better
        return [(int(row[0]), -float(row[1])) for row in rows]

    def fuzzy_search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        """Prefix search: each token + '*', return (id, score) sorted descending."""
        tokens = query.strip().split()
        if not tokens:
            return []
        prefix_query = " ".join(t + "*" for t in tokens)
        try:
            rows = self._conn.execute(
                "SELECT rowid, bm25(docs) AS score FROM docs "
                "WHERE docs MATCH ? ORDER BY score LIMIT ?",
                (prefix_query, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(int(row[0]), -float(row[1])) for row in rows]

    def get_content(self, doc_id: int) -> str:
        """Retrieve content for a doc_id."""
        row = self._conn.execute(
            "SELECT content FROM docs WHERE rowid = ?", (doc_id,)
        ).fetchone()
        return row[0] if row else ""

    def get_chunk_ids_by_path(self, path: str) -> list[int]:
        """Return all doc IDs associated with a given file path."""
        rows = self._conn.execute(
            "SELECT id FROM docs_meta WHERE path = ?", (path,)
        ).fetchall()
        return [r[0] for r in rows]

    def delete_by_path(self, path: str) -> int:
        """Delete all docs and docs_meta rows for a given file path.

        Returns the number of deleted documents.
        """
        ids = self.get_chunk_ids_by_path(path)
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        self._conn.execute(
            f"DELETE FROM docs WHERE rowid IN ({placeholders})", ids
        )
        self._conn.execute(
            f"DELETE FROM docs_meta WHERE id IN ({placeholders})", ids
        )
        self._conn.commit()
        return len(ids)

    def get_doc_meta(self, doc_id: int) -> tuple[str, int, int]:
        """Return (path, start_line, end_line) for a doc_id."""
        row = self._conn.execute(
            "SELECT path, start_line, end_line FROM docs_meta WHERE id = ?",
            (doc_id,),
        ).fetchone()
        if row:
            return row[0], row[1] or 0, row[2] or 0
        return "", 0, 0
