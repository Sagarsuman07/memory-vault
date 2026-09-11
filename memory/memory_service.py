from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
import uuid

from langchain_groq import ChatGroq

from config.settings import settings

from database.models import Memory

from database.repositories import (
    create_memory,
    get_memory,
    get_all_memories,
    update_memory as update_memory_record,
    delete_memory as delete_memory_record,
)

from ingestion.pipeline import (
    extract_content,
    get_memory_type,
)

from prompts.qa import QA_PROMPT
from prompts.summary import SUMMARY_PROMPT
from prompts.safety import SAFETY_PROMPT

from rag.chunker import split_text

from rag.vector_store import (
    index_memory,
    get_vector_store,
)

from rag.retriever import retrieve

from rag.grounding import (
    get_grounded_results,
)


# ============================================================
# Configuration
# ============================================================

UPLOAD_DIR = Path(
    settings.UPLOAD_DIR
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ALLOWED_EXTENSIONS = {

    # Documents
    ".pdf",
    ".docx",
    ".txt",

    # Images
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",

    # Audio
    ".mp3",
    ".wav",
    ".m4a",
    ".mpeg",
    ".mpga",
    ".webm",
    ".ogg",
    ".flac",
}


MAX_FILE_SIZE = (
    10 * 1024 * 1024
)


# ============================================================
# Create Memory
# ============================================================

def create_new_memory(
    uploaded_file,
    title: str,
    user_id: str,
    progress_callback: Optional[
        Callable[[str], None]
    ] = None,
):
    """
    Create a new memory using the existing
    ingestion and RAG architecture.

    Phase 13 only adds progress reporting.

    Pipeline:

        Upload
          ↓
        Extract
          ↓
        SQLite
          ↓
        Chunk
          ↓
        Embedding
          ↓
        Chroma
          ↓
        Done
    """

    # ========================================================
    # Validate uploaded file
    # ========================================================

    if uploaded_file is None:
        raise ValueError(
            "No file was provided."
        )

    # ========================================================
    # Validate user
    # ========================================================

    if not user_id or not user_id.strip():
        raise ValueError(
            "User ID cannot be empty."
        )

    # ========================================================
    # Validate title
    # ========================================================

    if not title or not title.strip():
        raise ValueError(
            "Memory title cannot be empty."
        )

    # ========================================================
    # Get file information
    # ========================================================

    file_name = uploaded_file.name

    extension = Path(
        file_name
    ).suffix.lower()

    # ========================================================
    # Validate extension
    # ========================================================

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    # ========================================================
    # Read file bytes
    # ========================================================

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        raise ValueError(
            "The selected file is empty."
        )

    # ========================================================
    # Validate file size
    # ========================================================

    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError(
            "File size cannot exceed 10 MB."
        )

    # ========================================================
    # Generate memory ID
    # ========================================================

    memory_id = str(
        uuid.uuid4()
    )

    # ========================================================
    # Determine memory type
    # ========================================================

    memory_type = get_memory_type(
        file_name
    )

    # ========================================================
    # Save original file
    # ========================================================

    file_path = (
        UPLOAD_DIR
        / f"{memory_id}{extension}"
    )

    file_path.write_bytes(
        file_bytes
    )

    try:

        # ====================================================
        # Stage 1: Extracting
        # ====================================================

        if progress_callback:
            progress_callback(
                "extracting"
            )

        extracted_text = extract_content(
            str(file_path)
        )

        if (
            not extracted_text
            or not extracted_text.strip()
        ):
            raise ValueError(
                "No content could be extracted "
                "from the file."
            )

        # ====================================================
        # Create Memory object
        # ====================================================

        now = datetime.now(
            timezone.utc
        ).isoformat()

        memory = Memory(
            id=memory_id,
            user_id=user_id,
            memory_type=memory_type,
            title=title.strip(),
            file_name=file_name,
            file_path=str(file_path),
            summary=None,
            extracted_text=extracted_text,
            created_at=now,
            updated_at=now,
        )

        # ====================================================
        # Save memory to SQLite
        # ====================================================

        create_memory(
            memory
        )

        # ====================================================
        # Create chunks
        # ====================================================

        chunks = split_text(
            extracted_text
        )

        if not chunks:
            raise ValueError(
                "No chunks could be created "
                "for indexing."
            )

        # ====================================================
        # Stage 2 + 3:
        # Embedding + Indexing
        # ====================================================

        index_memory(
            memory_id=memory.id,
            user_id=memory.user_id,
            memory_type=memory.memory_type,
            chunks=chunks,
            progress_callback=progress_callback,
        )

        # ====================================================
        # Stage 4: Done
        # ====================================================

        if progress_callback:
            progress_callback(
                "done"
            )

        return memory

    except Exception:

        # ====================================================
        # Cleanup Chroma vectors
        # ====================================================

        try:

            vector_store = (
                get_vector_store()
            )

            vector_store.delete(
                where={
                    "memory_id": memory_id
                }
            )

        except Exception:
            pass

        # ====================================================
        # Cleanup original file
        # ====================================================

        if file_path.exists():

            try:
                file_path.unlink()

            except Exception:
                pass

        # ====================================================
        # Cleanup SQLite record
        # ====================================================

        try:

            delete_memory_record(
                memory_id,
                user_id,
            )

        except Exception:
            pass

        raise


# ============================================================
# Get Memory
# ============================================================

def get_memory_by_id(
    memory_id: str,
    user_id: str,
):
    if not user_id or not user_id.strip():
        raise ValueError(
            "User ID cannot be empty."
        )

    if not memory_id or not memory_id.strip():
        raise ValueError(
            "Memory ID cannot be empty."
        )

    memory = get_memory(
        memory_id,
        user_id,
    )

    if memory is None:
        raise ValueError(
            "Memory not found."
        )

    return memory


# ============================================================
# List Memories
# ============================================================

def list_memories(
    user_id: str,
):
    if not user_id or not user_id.strip():
        raise ValueError(
            "User ID cannot be empty."
        )

    return get_all_memories(
        user_id
    )


# ============================================================
# Update Memory
# ============================================================

def update_memory(
    memory_id: str,
    user_id: str,
    title: str,
    summary: str,
):
    if not user_id or not user_id.strip():
        raise ValueError(
            "User ID cannot be empty."
        )

    if not memory_id or not memory_id.strip():
        raise ValueError(
            "Memory ID cannot be empty."
        )

    if not title or not title.strip():
        raise ValueError(
            "Memory title cannot be empty."
        )

    updated = update_memory_record(
        memory_id=memory_id,
        user_id=user_id,
        title=title.strip(),
        summary=summary.strip(),
    )

    if not updated:
        raise ValueError(
            "Memory not found."
        )

    return get_memory(
        memory_id,
        user_id,
    )


# ============================================================
# Delete Memory
# ============================================================

def delete_memory(
    memory_id: str,
    user_id: str,
):
    if not user_id or not user_id.strip():
        raise ValueError(
            "User ID cannot be empty."
        )

    if not memory_id or not memory_id.strip():
        raise ValueError(
            "Memory ID cannot be empty."
        )

    memory = get_memory(
        memory_id,
        user_id,
    )

    if memory is None:
        raise ValueError(
            "Memory not found."
        )

    # ========================================================
    # Delete vectors
    # ========================================================

    try:

        vector_store = (
            get_vector_store()
        )

        vector_store.delete(
            where={
                "memory_id": memory_id
            }
        )

    except Exception:
        pass

    # ========================================================
    # Delete original file
    # ========================================================

    file_path = Path(
        memory["file_path"]
    )

    if file_path.exists():
        file_path.unlink()

    # ========================================================
    # Delete database record
    # ========================================================

    deleted = delete_memory_record(
        memory_id,
        user_id,
    )

    if not deleted:
        raise ValueError(
            "Memory could not be deleted."
        )


# ============================================================
# Manual Re-index Existing Memory
# ============================================================

def index_existing_memory(
    memory_id: str,
    user_id: str,
):
    if not user_id or not user_id.strip():
        raise ValueError(
            "User ID cannot be empty."
        )

    if not memory_id or not memory_id.strip():
        raise ValueError(
            "Memory ID cannot be empty."
        )

    memory = get_memory(
        memory_id,
        user_id,
    )

    if memory is None:
        raise ValueError(
            "Memory not found."
        )

    extracted_text = memory[
        "extracted_text"
    ]

    if (
        not extracted_text
        or not extracted_text.strip()
    ):
        raise ValueError(
            "Memory has no extracted content."
        )

    chunks = split_text(
        extracted_text
    )

    if not chunks:
        raise ValueError(
            "No chunks could be created."
        )

    return index_memory(
        memory_id=memory["id"],
        user_id=memory["user_id"],
        memory_type=memory["memory_type"],
        chunks=chunks,
    )


# ============================================================
# QA Model
# ============================================================

def get_qa_model():

    if not settings.GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not configured."
        )

    return ChatGroq(
        model=settings.GROQ_LLM_MODEL,
        temperature=0,
        api_key=settings.GROQ_API_KEY,
    )


# ============================================================
# Summary Model
# ============================================================

def get_summary_model():

    if not settings.GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not configured."
        )

    return ChatGroq(
        model=settings.GROQ_LLM_MODEL,
        temperature=0,
        api_key=settings.GROQ_API_KEY,
    )


# ============================================================
# Safety Model
# ============================================================

def get_safety_model():

    if not settings.GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not configured."
        )

    return ChatGroq(
        model=settings.GROQ_LLM_MODEL,
        temperature=0,
        api_key=settings.GROQ_API_KEY,
    )


# ============================================================
# Check Content Safety
# ============================================================

def check_content_safety(
    content: str,
) -> bool:

    if not content or not content.strip():
        return False

    if not settings.SAFETY_ENABLED:
        return True

    model = get_safety_model()

    prompt = SAFETY_PROMPT.invoke(
        {
            "content": content,
        }
    )

    response = model.invoke(
        prompt
    )

    result = response.content

    if not isinstance(
        result,
        str,
    ):
        return False

    result = result.strip().upper()

    return result == "ALLOW"


# ============================================================
# Generate Memory Summary
# ============================================================

def generate_memory_summary(
    memory_id: str,
    user_id: str,
):
    if not user_id or not user_id.strip():
        raise ValueError(
            "User ID cannot be empty."
        )

    if not memory_id or not memory_id.strip():
        raise ValueError(
            "Memory ID cannot be empty."
        )

    memory = get_memory(
        memory_id,
        user_id,
    )

    if memory is None:
        raise ValueError(
            "Memory not found."
        )

    extracted_text = memory[
        "extracted_text"
    ]

    if (
        not extracted_text
        or not extracted_text.strip()
    ):
        raise ValueError(
            "No extracted content is available "
            "for summarization."
        )

    if not check_content_safety(
        extracted_text
    ):
        raise ValueError(
            "This memory cannot be summarized "
            "because its content did not pass "
            "the safety check."
        )

    model = get_summary_model()

    prompt = SUMMARY_PROMPT.invoke(
        {
            "extracted_text": extracted_text,
        }
    )

    response = model.invoke(
        prompt
    )

    content = response.content

    if not isinstance(
        content,
        str,
    ):
        raise ValueError(
            "Summary model returned invalid content."
        )

    content = content.strip()

    title_marker = "TITLE:"
    summary_marker = "SUMMARY:"

    if title_marker not in content:
        raise ValueError(
            "Summary response does not contain "
            "a TITLE section."
        )

    if summary_marker not in content:
        raise ValueError(
            "Summary response does not contain "
            "a SUMMARY section."
        )

    title_part = content.split(
        title_marker,
        1,
    )[1]

    title_part = title_part.split(
        summary_marker,
        1,
    )[0]

    generated_title = (
        title_part.strip()
    )

    generated_summary = (
        content.split(
            summary_marker,
            1,
        )[1]
        .strip()
    )

    if not generated_title:
        raise ValueError(
            "Generated title is empty."
        )

    if not generated_summary:
        raise ValueError(
            "Generated summary is empty."
        )

    updated = update_memory_record(
        memory_id=memory_id,
        user_id=user_id,
        title=generated_title,
        summary=generated_summary,
    )

    if not updated:
        raise ValueError(
            "Memory could not be updated."
        )

    return get_memory(
        memory_id,
        user_id,
    )


# ============================================================
# Build RAG Context
# ============================================================

def build_context(
    grounded_results,
):
    context_parts = []

    for rank, (
        document,
        distance,
    ) in enumerate(
        grounded_results,
        start=1,
    ):

        context_parts.append(
            f"""
Source {rank}
Memory ID: {document.metadata['memory_id']}
Memory Type: {document.metadata['memory_type']}
Chunk: {document.metadata['chunk_index']}

Content:
{document.page_content}
"""
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# Answer Question
# ============================================================

def answer_question(
    question: str,
    user_id: str,
    memory_id: str | None = None,
):
    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    if not user_id or not user_id.strip():
        raise ValueError(
            "User ID cannot be empty."
        )

    if memory_id is not None:

        if not memory_id.strip():
            raise ValueError(
                "Memory ID cannot be empty."
            )

    # ========================================================
    # Retrieve
    # ========================================================

    results = retrieve(
        question=question,
        user_id=user_id,
        top_k=3,
        memory_id=memory_id,
    )

    # ========================================================
    # Grounding threshold
    # ========================================================

    grounded_results = (
        get_grounded_results(
            results
        )
    )

    # ========================================================
    # Not found
    # ========================================================

    if not grounded_results:

        return {
            "answer": (
                "This information wasn't found "
                "in your memory."
            ),
            "sources": [],
            "grounded": False,
        }

    # ========================================================
    # Build context
    # ========================================================

    context = build_context(
        grounded_results
    )

    # ========================================================
    # Ask LLM
    # ========================================================

    model = get_qa_model()

    messages = QA_PROMPT.format_messages(
        context=context,
        question=question,
    )

    response = model.invoke(
        messages
    )

    # ========================================================
    # Sources
    # ========================================================

    sources = []

    for document, distance in grounded_results:

        sources.append(
            {
                "memory_id":
                    document.metadata[
                        "memory_id"
                    ],

                "memory_type":
                    document.metadata[
                        "memory_type"
                    ],

                "chunk_index":
                    document.metadata[
                        "chunk_index"
                    ],

                "distance":
                    distance,
            }
        )

    return {
        "answer": response.content,
        "sources": sources,
        "grounded": True,
    }