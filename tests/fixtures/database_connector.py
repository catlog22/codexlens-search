"""Database connection pool and query builder."""
import sqlite3
from contextlib import contextmanager
from typing import Any, Generator


class ConnectionPool:
    """Simple connection pool for SQLite databases."""

    def __init__(self, db_path: str, max_connections: int = 5):
        self._db_path = db_path
        self._max = max_connections
        self._pool: list[sqlite3.Connection] = []

    def acquire(self) -> sqlite3.Connection:
        if self._pool:
            return self._pool.pop()
        return sqlite3.connect(self._db_path)

    def release(self, conn: sqlite3.Connection) -> None:
        if len(self._pool) < self._max:
            self._pool.append(conn)
        else:
            conn.close()

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self.acquire()
        try:
            yield conn
        finally:
            self.release(conn)


class QueryBuilder:
    """Fluent query builder for SQL statements."""

    def __init__(self, table: str):
        self._table = table
        self._conditions: list[str] = []
        self._params: list[Any] = []
        self._columns = "*"
        self._order: str = ""
        self._limit: int | None = None

    def select(self, *columns: str) -> "QueryBuilder":
        self._columns = ", ".join(columns) if columns else "*"
        return self

    def where(self, condition: str, *params: Any) -> "QueryBuilder":
        self._conditions.append(condition)
        self._params.extend(params)
        return self

    def order_by(self, column: str, desc: bool = False) -> "QueryBuilder":
        self._order = f" ORDER BY {column}" + (" DESC" if desc else "")
        return self

    def limit(self, n: int) -> "QueryBuilder":
        self._limit = n
        return self

    def build(self) -> tuple[str, list[Any]]:
        sql = f"SELECT {self._columns} FROM {self._table}"
        if self._conditions:
            sql += " WHERE " + " AND ".join(self._conditions)
        sql += self._order
        if self._limit is not None:
            sql += f" LIMIT {self._limit}"
        return sql, self._params
