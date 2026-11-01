"""Supabase PostgreSQL connection pool and application schema."""
from __future__ import annotations

from threading import RLock

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from config.settings import settings


_pool: ConnectionPool | None = None
_pool_lock = RLock()
_schema_initialized = False


class DatabaseConfigurationError(RuntimeError):
    """Raised when the remote PostgreSQL service is not configured."""


class _PooledCursor:
    def __init__(self, cursor):
        self._cursor = cursor

    @staticmethod
    def _sql(query: str) -> str:
        return query.replace("BEGIN IMMEDIATE", "BEGIN").replace("?", "%s")

    def execute(self, query, params=None):
        if params is None:
            return self._cursor.execute(self._sql(query))
        return self._cursor.execute(self._sql(query), params)

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class _PooledConnection:
    """Preserve the repository layer's close()/commit() connection contract."""
    def __init__(self, connection, pool):
        self._connection = connection
        self._pool = pool
        self._returned = False

    @staticmethod
    def _sql(query: str) -> str:
        return query.replace("BEGIN IMMEDIATE", "BEGIN").replace("?", "%s")

    def execute(self, query, params=None):
        if params is None:
            return self._connection.execute(self._sql(query))
        return self._connection.execute(self._sql(query), params)

    def cursor(self):
        return _PooledCursor(self._connection.cursor())

    def commit(self):
        return self._connection.commit()

    def rollback(self):
        return self._connection.rollback()

    def close(self):
        if not self._returned:
            self._returned = True
            self._pool.putconn(self._connection)

    def __getattr__(self, name):
        return getattr(self._connection, name)


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is None:
            if not settings.DATABASE_URL:
                raise DatabaseConfigurationError(
                    "DATABASE_URL is required. Configure the Supabase PostgreSQL connection string "
                    "in the environment or Streamlit secrets."
                )
            pool = ConnectionPool(
                conninfo=settings.DATABASE_URL,
                min_size=settings.DB_POOL_MIN_SIZE,
                max_size=settings.DB_POOL_MAX_SIZE,
                timeout=settings.DB_POOL_TIMEOUT_SECONDS,
                kwargs={
                    "row_factory": dict_row,
                    "connect_timeout": settings.DB_CONNECT_TIMEOUT_SECONDS,
                    "options": f"-c statement_timeout={settings.DB_STATEMENT_TIMEOUT_MS}",
                    "application_name": "memory-vault",
                },
                open=False,
            )
            try:
                pool.open(wait=True)
            except Exception:
                pool.close()
                raise RuntimeError("Unable to connect to Supabase PostgreSQL.") from None
            _pool = pool
    return _pool


def get_connection():
    """Borrow a pooled PostgreSQL connection; close() returns it to the pool."""
    pool = _get_pool()
    try:
        connection = pool.getconn(timeout=settings.DB_POOL_TIMEOUT_SECONDS)
    except Exception:
        raise RuntimeError("Unable to acquire a PostgreSQL connection from the pool.") from None
    return _PooledConnection(connection, pool)


def initialize_database():
    """Create the fresh-start PostgreSQL schema once per application process."""
    global _schema_initialized
    if _schema_initialized:
        return
    with _pool_lock:
        if _schema_initialized:
            return
        connection = get_connection()
        try:
            statements = (
                """CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    original_file_key TEXT,
                    extracted_text_key TEXT,
                    legacy_file_path TEXT,
                    legacy_extracted_text TEXT,
                    summary TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )""",
                """CREATE TABLE IF NOT EXISTS preparation_jobs (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued'
                        CHECK (status IN ('queued', 'running', 'retrying', 'succeeded', 'failed')),
                    current_stage TEXT NOT NULL DEFAULT 'discovery'
                        CHECK (current_stage IN ('discovery', 'extraction', 'chunking', 'embedding', 'indexing', 'complete')),
                    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
                    completed_files INTEGER NOT NULL DEFAULT 0 CHECK (completed_files >= 0),
                    total_files INTEGER NOT NULL DEFAULT 1 CHECK (total_files >= 0),
                    error_category TEXT,
                    error_message TEXT,
                    lease_expires_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMPTZ,
                    finished_at TIMESTAMPTZ,
                    CHECK (completed_files <= total_files)
                )""",
                """CREATE TABLE IF NOT EXISTS ingestion_sources (
                    source_id TEXT PRIMARY KEY,
                    memory_id TEXT,
                    user_id TEXT NOT NULL,
                    original_file_name TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
                    sha256 TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('discovered', 'validated', 'loaded', 'skipped', 'failed')),
                    stored_file_path TEXT,
                    duplicate_of_memory_id TEXT,
                    warnings_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    error_category TEXT,
                    error_message TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )""",
                """CREATE TABLE IF NOT EXISTS chat_threads (
                    thread_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )""",
                "CREATE INDEX IF NOT EXISTS idx_preparation_jobs_user_status ON preparation_jobs (user_id, status)",
                "CREATE INDEX IF NOT EXISTS idx_preparation_jobs_memory ON preparation_jobs (memory_id, created_at)",
                "CREATE INDEX IF NOT EXISTS idx_ingestion_sources_user_hash ON ingestion_sources (user_id, sha256, status)",
                "CREATE INDEX IF NOT EXISTS idx_ingestion_sources_memory ON ingestion_sources (user_id, memory_id)",
                "CREATE INDEX IF NOT EXISTS idx_chat_threads_user_updated ON chat_threads (user_id, updated_at ASC)",
            )
            for statement in statements:
                connection.execute(statement)
            connection.commit()
            _schema_initialized = True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def close_database_pool():
    """Close the process-wide pool during controlled shutdown or tests."""
    global _pool, _schema_initialized
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None
        _schema_initialized = False
