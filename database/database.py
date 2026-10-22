import sqlite3
from pathlib import Path

from config.settings import settings


DATABASE_PATH = Path(settings.DATABASE_PATH)


def get_connection():
    """Open a SQLite connection using the configured application database."""
    timeout_ms = max(0, int(getattr(settings, "SQLITE_BUSY_TIMEOUT_MS", 30000)))
    connection = sqlite3.connect(DATABASE_PATH, timeout=timeout_ms / 1000.0)
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout={timeout_ms}")
    return connection


def initialize_database():
    """Create missing tables only; existing memories and vectors are untouched."""
    connection = get_connection()
    try:
        # WAL improves reader/writer coexistence. Set it once during DB init;
        # do not issue journal_mode changes from every worker connection.
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                title TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                summary TEXT,
                extracted_text TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS preparation_jobs (
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
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                started_at TEXT,
                finished_at TEXT,
                CHECK (completed_files <= total_files)
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_preparation_jobs_user_status "
            "ON preparation_jobs (user_id, status)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_preparation_jobs_memory "
            "ON preparation_jobs (memory_id, created_at)"
        )
        # Phase 2: source metadata and fingerprints. No foreign keys/cascades are
        # added, so failed ingestion attempts can remain auditable after cleanup.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_sources (
                source_id TEXT PRIMARY KEY,
                memory_id TEXT,
                user_id TEXT NOT NULL,
                original_file_name TEXT NOT NULL,
                extension TEXT NOT NULL,
                media_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
                sha256 TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('discovered', 'validated', 'loaded', 'skipped', 'failed')),
                stored_file_path TEXT,
                duplicate_of_memory_id TEXT,
                warnings_json TEXT NOT NULL DEFAULT '[]',
                error_category TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_ingestion_sources_user_hash "
            "ON ingestion_sources (user_id, sha256, status)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_ingestion_sources_memory "
            "ON ingestion_sources (user_id, memory_id)"
        )
        # Phase 3 additive migration: keep structured extraction metadata separate
        # from extracted text and warnings. Existing rows receive an empty object.
        source_columns = {row["name"] for row in cursor.execute("PRAGMA table_info(ingestion_sources)").fetchall()}
        if "metadata_json" not in source_columns:
            cursor.execute("ALTER TABLE ingestion_sources ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}' ")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
