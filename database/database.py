import sqlite3
from pathlib import Path

from config.settings import settings


DATABASE_PATH = Path(settings.DATABASE_PATH)


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    connection = get_connection()
    cursor = connection.cursor()

    # Create the memories table if it does not exist.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            title TEXT NOT NULL,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            extracted_text TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    # Check existing columns.
    cursor.execute("PRAGMA table_info(memories)")
    columns = [row["name"] for row in cursor.fetchall()]

    # Migration from Phase 1.
    #
    # Phase 1 did not have extracted_text.
    # Add it if the existing database does not have it.
    if "extracted_text" not in columns:
        cursor.execute(
            """
            ALTER TABLE memories
            ADD COLUMN extracted_text TEXT
            """
        )

    # Normalize memory_type values created during Phase 1.
    #
    # Phase 1 stored:
    #   pdf
    #   txt
    #
    # Phase 2 uses:
    #   document
    #   image
    #   audio
    #
    # Therefore convert the old document extensions
    # into the new canonical category.
    cursor.execute(
        """
        UPDATE memories
        SET memory_type = 'document'
        WHERE memory_type IN ('pdf', 'txt')
        """
    )

    connection.commit()
    connection.close()