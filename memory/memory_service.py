from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
import io
import uuid
import wave

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


UPLOAD_DIR = Path(
    settings.UPLOAD_DIR
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
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
    if uploaded_file is None:
        raise ValueError(
            "No file was provided."
        )

    if not user_id or not user_id.strip():
        raise ValueError(
            "User ID cannot be empty."
        )

    if not title or not title.strip():
        raise ValueError(
            "Memory title cannot be empty."
        )

    file_name = uploaded_file.name

    extension = Path(
        file_name
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        raise ValueError(
            "The selected file is empty."
        )

    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError(
            "File size cannot exceed 10 MB."
        )

    memory_id = str(
        uuid.uuid4()
    )

    memory_type = get_memory_type(
        file_name
    )

    file_path = (
        UPLOAD_DIR
        / f"{memory_id}{extension}"
    )

    file_path.write_bytes(
        file_bytes
    )

    try:

        # ====================================================
        # Extracting
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
        # SQLite Memory
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

        create_memory(
            memory
        )

        # ====================================================
        # Chunking
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
        # Done
        # ====================================================

        if progress_callback:
            progress_callback(
                "done"
            )

        return memory

    except Exception:

        # Remove vectors.
        try:

            vector_store = get_vector_store()

            vector_store.delete(
                where={
                    "memory_id": memory_id
                }
            )

        except Exception:
            pass

        # Remove original file.
        if file_path.exists():

            try:
                file_path.unlink()
            except Exception:
                pass

        # Remove database record.
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

    memories = get_all_memories(
        user_id
    )

    # Always keep dashboard chronology deterministic.
    return sorted(
        memories,
        key=lambda memory: memory["created_at"],
        reverse=True,
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
# Delete Single Memory
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

    # User-scoped ownership validation.
    memory = get_memory(
        memory_id,
        user_id,
    )

    if memory is None:
        raise ValueError(
            "Memory not found."
        )

    # ========================================================
    # Delete vector records
    # ========================================================

    vector_store = get_vector_store()

    vector_store.delete(
        where={
            "memory_id": memory_id
        }
    )

    # ========================================================
    # Delete original file
    # ========================================================

    file_path = Path(
        memory["file_path"]
    )

    if file_path.exists():

        try:
            file_path.unlink()
        except OSError as error:
            raise RuntimeError(
                "Memory file could not be deleted."
            ) from error

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
# Bulk Delete
# ============================================================

def delete_memories(
    memory_ids: list[str],
    user_id: str,
):
    """
    Delete multiple memories.

    Every memory is resolved using the current user_id
    before deletion. This prevents one user from deleting
    another user's memory.
    """

    if not user_id or not user_id.strip():
        raise ValueError(
            "User ID cannot be empty."
        )

    if not memory_ids:
        return 0

    unique_memory_ids = list(
        dict.fromkeys(
            memory_ids
        )
    )

    # ========================================================
    # Validate ownership of ALL memories first.
    # ========================================================

    memories = []

    for memory_id in unique_memory_ids:

        if (
            not memory_id
            or not memory_id.strip()
        ):
            raise ValueError(
                "Memory ID cannot be empty."
            )

        memory = get_memory(
            memory_id,
            user_id,
        )

        if memory is None:
            raise ValueError(
                "One or more selected memories "
                "do not belong to the current user."
            )

        memories.append(
            memory
        )

    # ========================================================
    # Delete only after ownership validation succeeds.
    # ========================================================

    for memory in memories:

        delete_memory(
            memory_id=memory["id"],
            user_id=user_id,
        )

    return len(
        memories
    )


# ============================================================
# Demo Helpers
# ============================================================

def _create_demo_text_file(
    file_name: str,
    content: str,
):
    path = (
        UPLOAD_DIR
        / file_name
    )

    path.write_text(
        content,
        encoding="utf-8",
    )

    return path


def _create_demo_pdf(
    file_name: str,
):
    """
    Create a small real PDF for the demo.
    """

    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    path = (
        UPLOAD_DIR
        / file_name
    )

    pdf = canvas.Canvas(
        str(path),
        pagesize=A4,
    )

    pdf.setFont(
        "Helvetica",
        16,
    )

    pdf.drawString(
        60,
        800,
        "Delhi Travel Plan",
    )

    pdf.setFont(
        "Helvetica",
        11,
    )

    lines = [
        "Travel destination: Delhi",
        "Flight: 6E123",
        "Travel date: 15 September 2026",
        "Hotel: Hotel ABC",
        "Hotel stay: 15–18 September 2026",
        "Important place: India Gate",
        "Estimated hotel cost: ₹5,000 per night",
    ]

    y = 770

    for line in lines:

        pdf.drawString(
            60,
            y,
            line,
        )

        y -= 22

    pdf.save()

    return path


def _create_demo_image(
    file_name: str,
    title: str,
    lines: list[str],
):
    """
    Create a real PNG demo image.

    The generated image is still sent through the
    normal image extraction pipeline.
    """

    from PIL import Image, ImageDraw

    path = (
        UPLOAD_DIR
        / file_name
    )

    image = Image.new(
        "RGB",
        (
            1200,
            700,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        image
    )

    draw.text(
        (70, 70),
        title,
        fill="black",
    )

    y = 180

    for line in lines:

        draw.text(
            (90, y),
            line,
            fill="black",
        )

        y += 70

    image.save(
        path,
        format="PNG",
    )

    return path


def _create_demo_audio(
    file_name: str,
):
    """
    Create a valid WAV container for the demo.

    The demo transcript is supplied through the same
    memory indexing path below because generating
    speech locally would require introducing a new
    TTS dependency.
    """

    path = (
        UPLOAD_DIR
        / file_name
    )

    sample_rate = 16000
    duration_seconds = 1

    frame_count = (
        sample_rate
        * duration_seconds
    )

    audio_data = (
        b"\x00\x00"
        * frame_count
    )

    with wave.open(
        str(path),
        "wb",
    ) as audio:

        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(
            sample_rate
        )

        audio.writeframes(
            audio_data
        )

    return path


def _create_memory_from_demo_text(
    title: str,
    file_path: Path,
    memory_type: str,
    extracted_text: str,
    user_id: str,
):
    """
    Create a demo memory and use the normal
    chunk → embedding → Chroma indexing pipeline.

    This is used only where an external extraction
    service is not practical for deterministic demo
    setup.
    """

    memory_id = str(
        uuid.uuid4()
    )

    now = datetime.now(
        timezone.utc
    ).isoformat()

    memory = Memory(
        id=memory_id,
        user_id=user_id,
        memory_type=memory_type,
        title=title,
        file_name=file_path.name,
        file_path=str(file_path),
        summary=None,
        extracted_text=extracted_text,
        created_at=now,
        updated_at=now,
    )

    create_memory(
        memory
    )

    try:

        chunks = split_text(
            extracted_text
        )

        if not chunks:
            raise ValueError(
                "No chunks could be created."
            )

        index_memory(
            memory_id=memory.id,
            user_id=memory.user_id,
            memory_type=memory.memory_type,
            chunks=chunks,
        )

        return memory

    except Exception:

        try:

            vector_store = get_vector_store()

            vector_store.delete(
                where={
                    "memory_id": memory_id
                }
            )

        except Exception:
            pass

        try:
            file_path.unlink()
        except Exception:
            pass

        try:

            delete_memory_record(
                memory_id,
                user_id,
            )

        except Exception:
            pass

        raise


def load_demo_memories(
    user_id: str,
):
    """
    Create a repeatable demo environment.

    Demo memories are identified using the [Demo]
    title prefix. Existing demo memories are reused
    rather than duplicated.
    """

    if not user_id or not user_id.strip():
        raise ValueError(
            "User ID cannot be empty."
        )

    existing_memories = list_memories(
        user_id
    )

    existing_demo_titles = {
        memory["title"]
        for memory in existing_memories
        if memory["title"].startswith(
            "[Demo]"
        )
    }

    demo_definitions = [
        {
            "title": "[Demo] Flight Ticket",
            "type": "image",
            "file_name": "demo_flight.png",
            "text": (
                "Flight ticket. "
                "Flight number 6E123. "
                "Route Hyderabad to Delhi. "
                "Travel date 15 September 2026. "
                "Ticket price ₹8,500."
            ),
        },
        {
            "title": "[Demo] Hotel Booking",
            "type": "document",
            "file_name": "demo_hotel.txt",
            "text": (
                "Hotel booking. "
                "Hotel ABC in Delhi. "
                "Stay from 15 September 2026 "
                "to 18 September 2026. "
                "Room cost ₹5,000 per night."
            ),
        },
        {
            "title": "[Demo] Sony Headphones",
            "type": "image",
            "file_name": "demo_headphones.png",
            "text": (
                "Product information. "
                "Sony headphones. "
                "Listed price ₹25,000."
            ),
        },
        {
            "title": "[Demo] Delhi Trip Voice Note",
            "type": "audio",
            "file_name": "demo_voice_note.wav",
            "text": (
                "Remember to visit India Gate "
                "during the Delhi trip."
            ),
        },
        {
            "title": "[Demo] Delhi Travel Plan",
            "type": "document",
            "file_name": "demo_travel_plan.pdf",
            "text": (
                "Delhi travel plan. "
                "Flight 6E123 from Hyderabad to Delhi "
                "on 15 September 2026. "
                "Hotel ABC from 15 to 18 September. "
                "Remember to visit India Gate."
            ),
        },
    ]

    created = []

    for demo in demo_definitions:

        if demo["title"] in existing_demo_titles:
            continue

        file_name = demo[
            "file_name"
        ]

        file_path = (
            UPLOAD_DIR
            / file_name
        )

        # ----------------------------------------------------
        # Flight image
        # ----------------------------------------------------

        if demo["title"] == "[Demo] Flight Ticket":

            _create_demo_image(
                file_name=file_name,
                title="Flight Ticket",
                lines=[
                    "IndiGo 6E123",
                    "Hyderabad → Delhi",
                    "15 September 2026",
                    "₹8,500",
                ],
            )

            # Use the actual image pipeline.
            class DemoFile:

                def __init__(
                    self,
                    path,
                ):
                    self.name = path.name

                def getvalue(self):
                    return file_path.read_bytes()

            try:

                memory = create_new_memory(
                    uploaded_file=DemoFile(
                        file_path
                    ),
                    title=demo["title"],
                    user_id=user_id,
                )

                created.append(
                    memory
                )

            except Exception:

                if file_path.exists():
                    file_path.unlink()

                raise

        # ----------------------------------------------------
        # Hotel text document
        # ----------------------------------------------------

        elif demo["title"] == "[Demo] Hotel Booking":

            _create_demo_text_file(
                file_name=file_name,
                content=demo["text"],
            )

            class DemoFile:

                def __init__(
                    self,
                    path,
                ):
                    self.name = path.name

                def getvalue(self):
                    return file_path.read_bytes()

            try:

                memory = create_new_memory(
                    uploaded_file=DemoFile(
                        file_path
                    ),
                    title=demo["title"],
                    user_id=user_id,
                )

                created.append(
                    memory
                )

            except Exception:

                if file_path.exists():
                    file_path.unlink()

                raise

        # ----------------------------------------------------
        # Product image
        # ----------------------------------------------------

        elif demo["title"] == "[Demo] Sony Headphones":

            _create_demo_image(
                file_name=file_name,
                title="Sony Headphones",
                lines=[
                    "Sony WH Series",
                    "Premium Headphones",
                    "Price: ₹25,000",
                ],
            )

            class DemoFile:

                def __init__(
                    self,
                    path,
                ):
                    self.name = path.name

                def getvalue(self):
                    return file_path.read_bytes()

            try:

                memory = create_new_memory(
                    uploaded_file=DemoFile(
                        file_path
                    ),
                    title=demo["title"],
                    user_id=user_id,
                )

                created.append(
                    memory
                )

            except Exception:

                if file_path.exists():
                    file_path.unlink()

                raise

        # ----------------------------------------------------
        # Voice note
        # ----------------------------------------------------

        elif demo["title"] == "[Demo] Delhi Trip Voice Note":

            _create_demo_audio(
                file_name=file_name
            )

            # A deterministic demo transcript is indexed
            # through the normal chunk/embedding/index path.
            memory = _create_memory_from_demo_text(
                title=demo["title"],
                file_path=file_path,
                memory_type="audio",
                extracted_text=demo["text"],
                user_id=user_id,
            )

            created.append(
                memory
            )

        # ----------------------------------------------------
        # Travel PDF
        # ----------------------------------------------------

        elif demo["title"] == "[Demo] Delhi Travel Plan":

            _create_demo_pdf(
                file_name=file_name
            )

            class DemoFile:

                def __init__(
                    self,
                    path,
                ):
                    self.name = path.name

                def getvalue(self):
                    return file_path.read_bytes()

            try:

                memory = create_new_memory(
                    uploaded_file=DemoFile(
                        file_path
                    ),
                    title=demo["title"],
                    user_id=user_id,
                )

                created.append(
                    memory
                )

            except Exception:

                if file_path.exists():
                    file_path.unlink()

                raise

    return created


# ============================================================
# Re-index Existing Memory
# ============================================================

def index_existing_memory(
    memory_id: str,
    user_id: str,
):
    memory = get_memory_by_id(
        memory_id,
        user_id,
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
# Content Safety
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

    return (
        result.strip().upper()
        == "ALLOW"
    )


# ============================================================
# Generate Summary
# ============================================================

def generate_memory_summary(
    memory_id: str,
    user_id: str,
):

    memory = get_memory_by_id(
        memory_id,
        user_id,
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
            "extracted_text":
                extracted_text,
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

    if "TITLE:" not in content:
        raise ValueError(
            "Summary response does not contain "
            "a TITLE section."
        )

    if "SUMMARY:" not in content:
        raise ValueError(
            "Summary response does not contain "
            "a SUMMARY section."
        )

    title_part = content.split(
        "TITLE:",
        1,
    )[1]

    title_part = title_part.split(
        "SUMMARY:",
        1,
    )[0]

    generated_title = (
        title_part.strip()
    )

    generated_summary = (
        content.split(
            "SUMMARY:",
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
# Build Context
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
# Grounded QA
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

    if (
        memory_id is not None
        and not memory_id.strip()
    ):
        raise ValueError(
            "Memory ID cannot be empty."
        )

    results = retrieve(
        question=question,
        user_id=user_id,
        top_k=3,
        memory_id=memory_id,
    )

    grounded_results = (
        get_grounded_results(
            results
        )
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

    context = build_context(
        grounded_results
    )

    model = get_qa_model()

    messages = (
        QA_PROMPT.format_messages(
            context=context,
            question=question,
        )
    )

    response = model.invoke(
        messages
    )

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