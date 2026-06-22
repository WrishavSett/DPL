"""
SQLite connection management.

Concurrency model: many short-lived read connections (FastAPI routers) plus
exactly one long-lived write connection, owned by app/db_writer.py. WAL mode
lets readers and that single writer proceed without blocking each other;
busy_timeout absorbs the brief lock window during a commit instead of
raising "database is locked".
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from app.config import settings

SCHEMA_PATH = settings.base_dir / "db" / "schema.sql"

# Applied to every connection opened by this module.
_BUSY_TIMEOUT_MS = 5000


def _configure(conn: sqlite3.Connection) -> sqlite3.Connection:
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS};")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def get_connection() -> sqlite3.Connection:
    """Open a new, fully configured connection. Caller owns closing it.

    Used directly by long-lived owners (e.g. the db_writer's single write
    connection) that don't want the context manager closing things early.
    """
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    return _configure(conn)


@contextmanager
def get_db() -> Iterator[sqlite3.Connection]:
    """Context-managed connection for one-off reads/writes (e.g. FastAPI routes)."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create tables from schema.sql if they don't already exist.

    Idempotent (schema.sql uses CREATE TABLE IF NOT EXISTS) — safe to call
    on every app startup.
    """
    schema_sql = SCHEMA_PATH.read_text()
    with get_db() as conn:
        conn.executescript(schema_sql)
        conn.commit()