from database.database import get_connection
from database.models import Memory


def create_memory(memory: Memory):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO memories (
            id,
            user_id,
            memory_type,
            title,
            file_name,
            file_path,
            extracted_text,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            memory.id,
            memory.user_id,
            memory.memory_type,
            memory.title,
            memory.file_name,
            memory.file_path,
            memory.extracted_text,
            memory.created_at,
            memory.updated_at,
        ),
    )

    connection.commit()
    connection.close()


def get_all_memories(user_id: str):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM memories
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (user_id,),
    )

    memories = cursor.fetchall()

    connection.close()

    return memories


def get_memory(memory_id: str, user_id: str):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM memories
        WHERE id = ?
        AND user_id = ?
        """,
        (
            memory_id,
            user_id,
        ),
    )

    memory = cursor.fetchone()

    connection.close()

    return memory


def delete_memory(memory_id: str, user_id: str):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM memories
        WHERE id = ?
        AND user_id = ?
        """,
        (
            memory_id,
            user_id,
        ),
    )

    connection.commit()
    connection.close()