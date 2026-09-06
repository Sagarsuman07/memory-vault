from datetime import datetime, timezone
from pathlib import Path
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

from rag.chunker import split_text

from rag.vector_store import index_memory

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
    exist_ok=True
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


MAX_FILE_SIZE = 10 * 1024 * 1024


# ============================================================
# Create Memory
# ============================================================

def create_new_memory(
    uploaded_file,
    title: str,
    user_id: str,
):

    if uploaded_file is None:

        raise ValueError(
            "No file was provided."
        )


    if not title or not title.strip():

        raise ValueError(
            "Memory title cannot be empty."
        )


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
    # Validate file size
    # ========================================================

    file_bytes = uploaded_file.getvalue()


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
        # Extract content
        # ====================================================

        extracted_text = extract_content(
            str(file_path)
        )


        if not extracted_text or not extracted_text.strip():

            raise ValueError(
                "No content could be extracted "
                "from the file."
            )


        # ====================================================
        # Create memory object
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
        # Save to database
        # ====================================================

        create_memory(
            memory
        )


        # ====================================================
        # Automatically index memory
        # ====================================================

        chunks = split_text(
            extracted_text
        )


        if not chunks:

            raise ValueError(
                "No chunks could be created "
                "for indexing."
            )


        index_memory(
            memory_id=memory.id,
            user_id=memory.user_id,
            memory_type=memory.memory_type,
            chunks=chunks,
        )


        return memory


    except Exception:

        # ====================================================
        # Cleanup original file
        # ====================================================

        if file_path.exists():

            file_path.unlink()


        # ====================================================
        # Cleanup database record
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
    user_id: str
):

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

    memory = get_memory(
        memory_id,
        user_id,
    )


    if memory is None:

        raise ValueError(
            "Memory not found."
        )


    file_path = Path(
        memory["file_path"]
    )


    if file_path.exists():

        file_path.unlink()


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


    if not extracted_text:

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


    indexed_count = index_memory(
        memory_id=memory["id"],
        user_id=memory["user_id"],
        memory_type=memory["memory_type"],
        chunks=chunks,
    )


    return indexed_count


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
# Generate Memory Summary
# ============================================================

def generate_memory_summary(
    memory_id: str,
    user_id: str,
):

    # ========================================================
    # 1. Get memory
    # ========================================================

    memory = get_memory(
        memory_id,
        user_id,
    )


    if memory is None:

        raise ValueError(
            "Memory not found."
        )


    # ========================================================
    # 2. Get extracted content
    # ========================================================

    extracted_text = memory[
        "extracted_text"
    ]


    if not extracted_text or not extracted_text.strip():

        raise ValueError(
            "Memory has no extracted content "
            "to summarize."
        )


    # ========================================================
    # 3. Create summary model
    # ========================================================

    model = get_summary_model()


    # ========================================================
    # 4. Build prompt
    # ========================================================

    messages = SUMMARY_PROMPT.format_messages(
        extracted_text=extracted_text,
    )


    # ========================================================
    # 5. Generate title + summary
    # ========================================================

    response = model.invoke(
        messages
    )


    generated_text = response.content.strip()


    if not generated_text:

        raise ValueError(
            "The LLM returned an empty summary."
        )


    # ========================================================
    # 6. Parse title and summary
    # ========================================================

    title_marker = "TITLE:"
    summary_marker = "SUMMARY:"


    if (
        title_marker not in generated_text
        or summary_marker not in generated_text
    ):

        raise ValueError(
            "The LLM returned an unexpected "
            "summary format."
        )


    title_start = (
        generated_text.index(
            title_marker
        )
        + len(title_marker)
    )


    summary_start = generated_text.index(
        summary_marker
    )


    generated_title = (
        generated_text[
            title_start:summary_start
        ]
        .strip()
    )


    generated_summary = (
        generated_text[
            summary_start
            + len(summary_marker):
        ]
        .strip()
    )


    if not generated_title:

        raise ValueError(
            "The generated title is empty."
        )


    if not generated_summary:

        raise ValueError(
            "The generated summary is empty."
        )


    # ========================================================
    # 7. Save title + summary
    # ========================================================

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


    # ========================================================
    # 8. Return updated memory
    # ========================================================

    return get_memory(
        memory_id,
        user_id,
    )


# ============================================================
# Build QA Context
# ============================================================

def build_context(
    grounded_results
):

    context_parts = []


    for rank, (
        document,
        distance
    ) in enumerate(
        grounded_results,
        start=1
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


    # ========================================================
    # 1. Retrieve
    # ========================================================

    results = retrieve(
        question=question,
        user_id=user_id,
        top_k=3,
        memory_id=memory_id,
    )


    # ========================================================
    # 2. Grounding Check
    # ========================================================

    grounded_results = get_grounded_results(
        results
    )


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
    # 3. Build Context
    # ========================================================

    context = build_context(
        grounded_results
    )


    # ========================================================
    # 4. Create LLM
    # ========================================================

    model = get_qa_model()


    # ========================================================
    # 5. Build Prompt
    # ========================================================

    messages = QA_PROMPT.format_messages(
        context=context,
        question=question,
    )


    # ========================================================
    # 6. Generate Answer
    # ========================================================

    response = model.invoke(
        messages
    )


    # ========================================================
    # 7. Build Sources
    # ========================================================

    sources = []


    for document, distance in grounded_results:

        sources.append(
            {
                "memory_id":
                    document.metadata["memory_id"],

                "memory_type":
                    document.metadata["memory_type"],

                "chunk_index":
                    document.metadata["chunk_index"],

                "distance":
                    distance,
            }
        )


    # ========================================================
    # 8. Return Result
    # ========================================================

    return {
        "answer": response.content,
        "sources": sources,
        "grounded": True,
    }