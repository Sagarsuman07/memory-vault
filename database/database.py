import sqlite3
from pathlib import Path

from config.settings import settings


DATABASE_PATH = Path(
    settings.DATABASE_PATH
)


def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()


    # =========================================
    # Create memories table if it doesn't exist
    # =========================================

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


    # =========================================
    # Check existing columns
    # =========================================

    cursor.execute(
        "PRAGMA table_info(memories)"
    )

    columns = [
        row["name"]
        for row in cursor.fetchall()
    ]


    # =========================================
    # Phase 2 migration
    # =========================================

    if "extracted_text" not in columns:

        cursor.execute(
            """
            ALTER TABLE memories
            ADD COLUMN extracted_text TEXT
            """
        )


    # =========================================
    # Phase 3 migration
    # =========================================

    if "summary" not in columns:

        cursor.execute(
            """
            ALTER TABLE memories
            ADD COLUMN summary TEXT
            """
        )


    connection.commit()

    connection.close()