from __future__ import annotations

import sqlite3
import threading
from pathlib import Path


class FTSEngine:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._local = threading.local()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
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
        self._migrate_language_column()
        self._create_symbols_table()
        self._create_symbol_refs_table()
        self._create_entity_edges_table()
        # Mark the primary connection for the creating thread
        self._local.conn = self._conn
        self._read_conns: list[sqlite3.Connection] = []
        self._read_conns_lock = threading.Lock()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local SQLite connection for read operations.

        The creating thread reuses the primary connection. Worker threads
        get their own connection to avoid SQLite concurrent access errors.
        """
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode = WAL")
        self._local.conn = conn
        with self._read_conns_lock:
            self._read_conns.append(conn)
        return conn

    def _migrate_line_columns(self) -> None:
        """Add start_line/end_line columns if missing (for pre-existing DBs)."""
        conn = self._conn
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(docs_meta)").fetchall()
        }
        for col in ("start_line", "end_line"):
            if col not in cols:
                conn.execute(
                    f"ALTER TABLE docs_meta ADD COLUMN {col} INTEGER DEFAULT 0"
                )
        conn.commit()

    def _migrate_language_column(self) -> None:
        """Add language column if missing (for pre-existing DBs)."""
        conn = self._conn
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(docs_meta)").fetchall()
        }
        if "language" not in cols:
            conn.execute(
                "ALTER TABLE docs_meta ADD COLUMN language TEXT DEFAULT ''"
            )
            conn.commit()

    def _create_symbols_table(self) -> None:
        """Create symbols table and indexes if they do not exist."""
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS symbols ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "chunk_id INTEGER, "
            "name TEXT NOT NULL, "
            "kind TEXT NOT NULL, "
            "start_line INTEGER, "
            "end_line INTEGER, "
            "parent_name TEXT, "
            "signature TEXT, "
            "language TEXT)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbols_chunk_id "
            "ON symbols (chunk_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbols_name "
            "ON symbols (name)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbols_kind "
            "ON symbols (kind)"
        )
        self._conn.commit()

    def _create_symbol_refs_table(self) -> None:
        """Create symbol_refs table and indexes if they do not exist."""
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS symbol_refs ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "from_symbol_id INTEGER, "
            "from_name TEXT NOT NULL, "
            "from_path TEXT NOT NULL, "
            "to_name TEXT NOT NULL, "
            "to_symbol_id INTEGER, "
            "ref_kind TEXT NOT NULL, "
            "line INTEGER)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_refs_from_id "
            "ON symbol_refs (from_symbol_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_refs_from_name "
            "ON symbol_refs (from_name)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_refs_to_name "
            "ON symbol_refs (to_name)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_refs_to_id "
            "ON symbol_refs (to_symbol_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_refs_kind "
            "ON symbol_refs (ref_kind)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_refs_path "
            "ON symbol_refs (from_path)"
        )
        self._conn.commit()

    def _create_entity_edges_table(self) -> None:
        """Create entity_edges table and indexes if they do not exist."""
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS entity_edges ("
            "from_entity TEXT NOT NULL, "
            "to_entity TEXT NOT NULL, "
            "edge_kind TEXT NOT NULL, "
            "weight REAL DEFAULT 1.0, "
            "PRIMARY KEY (from_entity, to_entity, edge_kind))"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_edges_from ON entity_edges (from_entity)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_edges_to ON entity_edges (to_entity)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_edges_kind ON entity_edges (edge_kind)"
        )
        self._conn.commit()

    def add_entity_edges(self, edges: list[tuple[str, str, str, float]]) -> None:
        """Batch-insert entity graph edges in a single transaction.

        Each tuple: (from_entity, to_entity, edge_kind, weight).
        Uses UPSERT to accumulate weights across repeated edges.
        """
        if not edges:
            return
        self._conn.executemany(
            "INSERT INTO entity_edges (from_entity, to_entity, edge_kind, weight) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(from_entity, to_entity, edge_kind) "
            "DO UPDATE SET weight = weight + excluded.weight",
            edges,
        )

    # ------------------------------------------------------------------
    # Symbol reference operations
    # ------------------------------------------------------------------

    def add_refs(
        self,
        refs: list[tuple[str, str, str, str, int]],
    ) -> None:
        """Batch-insert symbol references in a single transaction.

        Each tuple: (from_name, from_path, to_name, ref_kind, line).
        from_symbol_id and to_symbol_id are left NULL for later resolution.
        """
        if not refs:
            return
        self._conn.executemany(
            "INSERT INTO symbol_refs "
            "(from_name, from_path, to_name, ref_kind, line) "
            "VALUES (?, ?, ?, ?, ?)",
            refs,
        )

    def delete_refs_by_path(self, path: str) -> int:
        """Delete all outgoing references from a given file path.

        Also invalidates incoming to_symbol_id for refs that point to
        symbols defined in the deleted file.

        Returns the number of deleted rows.
        """
        cursor = self._conn.execute(
            "DELETE FROM symbol_refs WHERE from_path = ?", (path,)
        )
        self._conn.commit()
        return cursor.rowcount

    def resolve_refs(self) -> int:
        """Batch resolve to_symbol_id for unresolved references.

        Links each ref's to_name to the id of a matching symbol in the
        symbols table. Only updates rows where to_symbol_id IS NULL.

        Returns the number of resolved references.
        """
        cursor = self._conn.execute(
            "UPDATE symbol_refs SET to_symbol_id = ("
            "  SELECT id FROM symbols WHERE symbols.name = symbol_refs.to_name LIMIT 1"
            ") WHERE to_symbol_id IS NULL "
            "AND EXISTS ("
            "  SELECT 1 FROM symbols WHERE symbols.name = symbol_refs.to_name"
            ")"
        )
        self._conn.commit()
        return cursor.rowcount

    def get_refs_from(self, name: str) -> list[dict]:
        """Get all references originating from a symbol name.

        Returns list of dicts with keys: id, from_symbol_id, from_name,
        from_path, to_name, to_symbol_id, ref_kind, line.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, from_symbol_id, from_name, from_path, "
            "to_name, to_symbol_id, ref_kind, line "
            "FROM symbol_refs WHERE from_name = ?",
            (name,),
        ).fetchall()
        cols = (
            "id", "from_symbol_id", "from_name", "from_path",
            "to_name", "to_symbol_id", "ref_kind", "line",
        )
        return [dict(zip(cols, row)) for row in rows]

    def get_refs_to(self, name: str) -> list[dict]:
        """Get all references pointing to a symbol name.

        Returns list of dicts with keys: id, from_symbol_id, from_name,
        from_path, to_name, to_symbol_id, ref_kind, line.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, from_symbol_id, from_name, from_path, "
            "to_name, to_symbol_id, ref_kind, line "
            "FROM symbol_refs WHERE to_name = ?",
            (name,),
        ).fetchall()
        cols = (
            "id", "from_symbol_id", "from_name", "from_path",
            "to_name", "to_symbol_id", "ref_kind", "line",
        )
        return [dict(zip(cols, row)) for row in rows]

    # ------------------------------------------------------------------
    # Symbol operations
    # ------------------------------------------------------------------

    def add_symbols(
        self,
        symbols: list[tuple[int, str, str, int, int, str, str, str]],
    ) -> None:
        """Batch-insert symbols in a single transaction.

        Each tuple: (chunk_id, name, kind, start_line, end_line,
                      parent_name, signature, language).
        """
        if not symbols:
            return
        self._conn.executemany(
            "INSERT INTO symbols "
            "(chunk_id, name, kind, start_line, end_line, parent_name, signature, language) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            symbols,
        )

    def get_symbols_by_name(
        self, name: str, kind: str | None = None
    ) -> list[dict]:
        """Look up symbols by name, optionally filtered by kind.

        Returns list of dicts with keys: id, chunk_id, name, kind,
        start_line, end_line, parent_name, signature, language.
        """
        conn = self._get_conn()
        if kind is not None:
            rows = conn.execute(
                "SELECT id, chunk_id, name, kind, start_line, end_line, "
                "parent_name, signature, language "
                "FROM symbols WHERE name = ? AND kind = ?",
                (name, kind),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, chunk_id, name, kind, start_line, end_line, "
                "parent_name, signature, language "
                "FROM symbols WHERE name = ?",
                (name,),
            ).fetchall()
        cols = (
            "id", "chunk_id", "name", "kind", "start_line", "end_line",
            "parent_name", "signature", "language",
        )
        return [dict(zip(cols, row)) for row in rows]

    def get_symbols_by_chunk(self, chunk_id: int) -> list[dict]:
        """Return all symbols associated with a given chunk_id."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, chunk_id, name, kind, start_line, end_line, "
            "parent_name, signature, language "
            "FROM symbols WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchall()
        cols = (
            "id", "chunk_id", "name", "kind", "start_line", "end_line",
            "parent_name", "signature", "language",
        )
        return [dict(zip(cols, row)) for row in rows]

    def delete_symbols_by_chunk_ids(self, chunk_ids: list[int]) -> int:
        """Delete all symbols associated with the given chunk IDs.

        Returns the number of deleted rows.
        """
        if not chunk_ids:
            return 0
        placeholders = ",".join("?" for _ in chunk_ids)
        cursor = self._conn.execute(
            f"DELETE FROM symbols WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        self._conn.commit()
        return cursor.rowcount

    def add_documents(self, docs: list[tuple]) -> None:
        """Add documents in batch.

        docs: list of (id, path, content)
              or (id, path, content, start_line, end_line)
              or (id, path, content, start_line, end_line, language).
        """
        if not docs:
            return
        meta_rows = []
        fts_rows = []
        for doc in docs:
            if len(doc) >= 6:
                doc_id, path, content = doc[0], doc[1], doc[2]
                sl, el, lang = doc[3], doc[4], doc[5]
            elif len(doc) >= 5:
                doc_id, path, content, sl, el = doc[0], doc[1], doc[2], doc[3], doc[4]
                lang = ""
            else:
                doc_id, path, content = doc[0], doc[1], doc[2]
                sl, el, lang = 0, 0, ""
            meta_rows.append((doc_id, path, sl, el, lang))
            fts_rows.append((doc_id, content))
        self._conn.executemany(
            "INSERT OR REPLACE INTO docs_meta (id, path, start_line, end_line, language) "
            "VALUES (?, ?, ?, ?, ?)",
            meta_rows,
        )
        self._conn.executemany(
            "INSERT OR REPLACE INTO docs (rowid, content) VALUES (?, ?)",
            fts_rows,
        )
        self._conn.commit()

    def exact_search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        """FTS5 MATCH query, return (id, bm25_score) sorted by score descending."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
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
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT rowid, bm25(docs) AS score FROM docs "
                "WHERE docs MATCH ? ORDER BY score LIMIT ?",
                (prefix_query, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(int(row[0]), -float(row[1])) for row in rows]

    def get_content(self, doc_id: int) -> str:
        """Retrieve content for a doc_id."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT content FROM docs WHERE rowid = ?", (doc_id,)
        ).fetchone()
        return row[0] if row else ""

    def get_all_chunk_ids(self) -> set[int]:
        """Return all doc IDs in the FTS index."""
        conn = self._get_conn()
        rows = conn.execute("SELECT id FROM docs_meta").fetchall()
        return {r[0] for r in rows}

    def get_chunk_ids_by_path(self, path: str) -> list[int]:
        """Return all doc IDs associated with a given file path."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id FROM docs_meta WHERE path = ?", (path,)
        ).fetchall()
        return [r[0] for r in rows]

    def delete_by_ids(self, ids: list[int]) -> int:
        """Delete docs, docs_meta, symbols, and refs for the given chunk IDs.

        Used for purging orphan entries that are not tracked by metadata.
        Processes in batches of 500 to avoid SQLite variable limits.

        Returns the number of deleted documents.
        """
        if not ids:
            return 0
        total = 0
        batch_size = 500
        for start in range(0, len(ids), batch_size):
            batch = ids[start : start + batch_size]
            placeholders = ",".join("?" for _ in batch)
            self._conn.execute(
                f"DELETE FROM symbols WHERE chunk_id IN ({placeholders})",
                batch,
            )
            self._conn.execute(
                f"DELETE FROM docs WHERE rowid IN ({placeholders})",
                batch,
            )
            self._conn.execute(
                f"DELETE FROM docs_meta WHERE id IN ({placeholders})",
                batch,
            )
            total += len(batch)
        self._conn.commit()
        return total

    def delete_by_path(self, path: str) -> int:
        """Delete all docs, docs_meta, symbols, and refs rows for a given file path.

        Symbols and refs are deleted BEFORE chunks because FTS5 does not
        support CASCADE constraints.

        Returns the number of deleted documents.
        """
        ids = self.get_chunk_ids_by_path(path)
        if not ids:
            # Still clean up refs by path even when no chunks found
            self._conn.execute(
                "DELETE FROM symbol_refs WHERE from_path = ?", (path,)
            )
            # Clean up outgoing entity edges for this path
            self._conn.execute(
                "DELETE FROM entity_edges WHERE from_entity LIKE ?",
                (f"{path}\t%",),
            )
            self._conn.commit()
            return 0
        placeholders = ",".join("?" for _ in ids)
        # Delete refs by path
        self._conn.execute(
            "DELETE FROM symbol_refs WHERE from_path = ?", (path,)
        )
        # Delete outgoing entity edges by path prefix
        self._conn.execute(
            "DELETE FROM entity_edges WHERE from_entity LIKE ?",
            (f"{path}\t%",),
        )
        # Delete symbols first (no CASCADE in FTS5)
        self._conn.execute(
            f"DELETE FROM symbols WHERE chunk_id IN ({placeholders})", ids
        )
        self._conn.execute(
            f"DELETE FROM docs WHERE rowid IN ({placeholders})", ids
        )
        self._conn.execute(
            f"DELETE FROM docs_meta WHERE id IN ({placeholders})", ids
        )
        self._conn.commit()
        return len(ids)

    def flush(self) -> None:
        """Commit pending writes to disk."""
        self._conn.commit()

    def close(self) -> None:
        """Close all SQLite connections (primary + worker thread connections)."""
        try:
            self._conn.close()
        except Exception:
            pass
        with self._read_conns_lock:
            for conn in self._read_conns:
                try:
                    conn.close()
                except Exception:
                    pass
            self._read_conns.clear()

    def __enter__(self) -> "FTSEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def get_doc_meta(self, doc_id: int) -> tuple[str, int, int, str]:
        """Return (path, start_line, end_line, language) for a doc_id."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT path, start_line, end_line, language FROM docs_meta WHERE id = ?",
            (doc_id,),
        ).fetchone()
        if row:
            return row[0], row[1] or 0, row[2] or 0, row[3] or ""
        return "", 0, 0, ""
