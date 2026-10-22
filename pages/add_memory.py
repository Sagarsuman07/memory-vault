import io
import wave

import streamlit as st

from config.settings import settings

from database.database import initialize_database

from memory.async_ingestion import enqueue_new_memory
from database.database import get_connection


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Add Memory | Memory Vault",
    page_icon="➕",
    layout="wide",
)


# ============================================================
# Initialize Application
# ============================================================

initialize_database()

USER_ID = settings.DEMO_USER_ID


# ============================================================
# Simplified preparation status
# ============================================================

_ACTIVE_JOB_STATUSES = ("queued", "running", "retrying")


def get_active_preparation_count(user_id: str) -> int:
    """Return the number of preparation jobs that are queued or in progress."""
    connection = get_connection()
    try:
        placeholders = ", ".join("?" for _ in _ACTIVE_JOB_STATUSES)
        row = connection.execute(
            f"SELECT COUNT(*) AS total FROM preparation_jobs "
            f"WHERE user_id = ? AND status IN ({placeholders})",
            (user_id, *_ACTIVE_JOB_STATUSES),
        ).fetchone()
        return int(row["total"] if row else 0)
    finally:
        connection.close()


def render_preparation_status(user_id: str) -> None:
    """Show only the requested two-state knowledge-base status text."""
    try:
        count = get_active_preparation_count(user_id)
    except Exception:
        st.caption("Memory base status is temporarily unavailable.")
        return

    if count:
        st.info(f"Memory base preparing ({count})")
    else:
        st.success("Memory base prepared")


def _install_status_refresh(function):
    fragment = getattr(st, "fragment", None)
    if fragment is None:
        return function
    return fragment(run_every="2s")(function)


render_preparation_status = _install_status_refresh(render_preparation_status)
render_preparation_status(USER_ID)


# ============================================================
# Recorded Audio Adapter
# ============================================================

class RecordedAudioFile:
    """
    Adapter that makes browser-recorded audio behave
    like Streamlit's UploadedFile.

    The recorded audio is passed through the same
    existing audio ingestion pipeline.
    """

    def __init__(
        self,
        data: bytes,
        name: str = "recorded_audio.wav",
    ):

        self._data = data
        self.name = name

    def getvalue(self):

        return self._data


# ============================================================
# WAV Duration
# ============================================================

def get_wav_duration_seconds(
    audio_bytes: bytes,
) -> float:
    """
    Calculate WAV recording duration.
    """

    if not audio_bytes:

        return 0.0

    try:

        with wave.open(
            io.BytesIO(audio_bytes),
            "rb",
        ) as audio:

            frame_rate = (
                audio.getframerate()
            )

            frame_count = (
                audio.getnframes()
            )

            if frame_rate <= 0:

                return 0.0

            return (
                frame_count
                / float(frame_rate)
            )

    except (
        wave.Error,
        EOFError,
    ):

        return 0.0


# ============================================================
# Upload Progress
# ============================================================

def render_upload_progress(
    status,
    stage: str,
):

    labels = {

        "extracting":
            "Extracting content...",

        "embedding":
            "Creating embeddings...",

        "indexing":
            "Indexing memory...",

        "done":
            "Memory ready.",
    }

    stage_order = [
        "extracting",
        "embedding",
        "indexing",
        "done",
    ]

    stage_names = [
        "Extracting",
        "Embedding",
        "Indexing",
        "Done",
    ]

    if stage not in stage_order:

        return

    current_index = (
        stage_order.index(stage)
    )

    status.update(
        label=labels[stage],
        state="running",
    )

    progress_text = []

    for index, name in enumerate(
        stage_names
    ):

        if index <= current_index:

            prefix = "✓ "

        else:

            prefix = "○ "

        progress_text.append(
            prefix + name
        )

    status.write(
        "\n".join(
            progress_text
        )
    )


# ============================================================
# Process Upload
# ============================================================

def process_memory_upload(
    uploaded_file,
    title: str,
):
    """Accept the original upload and queue background preparation."""
    try:
        queued = enqueue_new_memory(
            uploaded_file=uploaded_file,
            title=title,
            user_id=USER_ID,
        )

        # Keep the confirmation accurate: enqueueing is not the same as
        # successful extraction or indexing. Phase 7 will add live progress.
        st.session_state.memory_queued = {
            "memory_id": queued["memory_id"],
            "job_id": queued["job_id"],
            "status": queued["status"],
            "title": title.strip(),
        }
        st.toast(
            f"'{title.strip()}' was added to the preparation queue.",
            icon="⏳",
        )
        st.rerun()

    except ValueError as error:
        st.toast(str(error), icon="❌")

    except Exception:
        # Do not show raw provider/system exception details in the UI.
        st.toast(
            "The upload could not be queued. Please check the application logs and try again.",
            icon="❌",
        )


# ============================================================
# Header
# ============================================================

st.title(
    "➕ Add Memory"
)

st.write(
    "Add a document, photo, or voice recording "
    "to your Memory Vault."
)


# ============================================================
# Back to Home
# ============================================================

if st.button(
    "← Back to Home",
    key="back_to_home",
):

    st.switch_page(
        "app.py"
    )


# ============================================================
# Add Memory Tabs
# ============================================================

st.divider()

document_tab, photo_tab, voice_tab = st.tabs(
    [
        "📄 Document",
        "🖼️ Photo",
        "🎙️ Voice",
    ]
)


# ============================================================
# Document
# ============================================================

with document_tab:

    st.subheader(
        "Add a Document"
    )

    st.write(
        "Supported formats: PDF, DOCX, TXT"
    )

    document_file = st.file_uploader(
        "Choose a document",
        type=[
            "pdf",
            "docx",
            "txt",
        ],
        key="document_uploader",
    )

    document_title = st.text_input(
        "Memory Title",
        placeholder=(
            "e.g. Delhi Travel Document"
        ),
        key="document_title",
    )

    if st.button(
        "Add Document",
        type="primary",
        key="add_document",
    ):

        if document_file is None:

            st.toast(
                "Please select a document.",
                icon="⚠️",
            )

        elif not document_title.strip():

            st.toast(
                "Please enter a memory title.",
                icon="⚠️",
            )

        else:

            process_memory_upload(
                document_file,
                document_title,
            )


# ============================================================
# Photo
# ============================================================

with photo_tab:

    st.subheader(
        "Add a Photo"
    )

    st.write(
        "Supported formats: JPG, JPEG, PNG, "
        "WEBP, GIF"
    )

    photo_file = st.file_uploader(
        "Choose a photo",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
            "gif",
        ],
        key="photo_uploader",
    )

    photo_title = st.text_input(
        "Memory Title",
        placeholder=(
            "e.g. Flight Ticket"
        ),
        key="photo_title",
    )

    if st.button(
        "Add Photo",
        type="primary",
        key="add_photo",
    ):

        if photo_file is None:

            st.toast(
                "Please select a photo.",
                icon="⚠️",
            )

        elif not photo_title.strip():

            st.toast(
                "Please enter a memory title.",
                icon="⚠️",
            )

        else:

            process_memory_upload(
                photo_file,
                photo_title,
            )


# ============================================================
# Voice
# ============================================================

with voice_tab:

    st.subheader(
        "Add a Voice Memory"
    )

    st.write(
        "Upload an existing audio file or "
        "record a new voice memory."
    )

    # ========================================================
    # Audio Upload
    # ========================================================

    st.markdown(
        "### Upload Audio"
    )

    audio_file = st.file_uploader(
        "Choose an audio file",
        type=[
            "mp3",
            "wav",
            "m4a",
            "mpeg",
            "mpga",
            "webm",
            "ogg",
            "flac",
        ],
        key="audio_uploader",
    )

    audio_title = st.text_input(
        "Memory Title",
        placeholder=(
            "e.g. Delhi Trip Voice Note"
        ),
        key="audio_title",
    )

    if st.button(
        "Add Audio",
        type="primary",
        key="add_audio",
    ):

        if audio_file is None:

            st.toast(
                "Please select an audio file.",
                icon="⚠️",
            )

        elif not audio_title.strip():

            st.toast(
                "Please enter a memory title.",
                icon="⚠️",
            )

        else:

            process_memory_upload(
                audio_file,
                audio_title,
            )

    # ========================================================
    # Browser Recording
    # ========================================================

    st.divider()

    st.markdown(
        "### Record Voice"
    )

    st.write(
        "Record a voice note directly "
        "from your browser."
    )

    recorded_audio = st.audio_input(
        "Record from microphone",
        key="memory_voice_recorder",
    )

    if recorded_audio is not None:

        recorded_bytes = (
            recorded_audio.getvalue()
        )

        if not recorded_bytes:

            st.error(
                (
                    "The recording is empty. "
                    "Please record again."
                )
            )

        else:

            # ------------------------------------------------
            # Audio Preview
            # ------------------------------------------------

            st.audio(
                recorded_bytes,
                format="audio/wav",
            )

            # ------------------------------------------------
            # Duration
            # ------------------------------------------------

            duration = (
                get_wav_duration_seconds(
                    recorded_bytes
                )
            )

            if duration > 0:

                st.caption(
                    (
                        "Recording duration: "
                        f"{duration:.1f} seconds"
                    )
                )

            # ------------------------------------------------
            # Recording Title
            # ------------------------------------------------

            recording_title = st.text_input(
                "Recording Title",
                placeholder=(
                    "e.g. Delhi Trip Reminder"
                ),
                key="recording_title",
            )

            # ------------------------------------------------
            # Use Recording
            # ------------------------------------------------

            if st.button(
                "Use Recording",
                type="primary",
                key="use_recording",
            ):

                if duration <= 0:

                    st.toast(
                        (
                            "The recording is empty "
                            "or could not be read. "
                            "Please record again."
                        ),
                        icon="❌",
                    )

                elif (
                    duration
                    < settings.MIN_RECORDING_SECONDS
                ):

                    st.toast(
                        (
                            "Recording is too short. "
                            "Please record at least "
                            f"{settings.MIN_RECORDING_SECONDS:.1f} "
                            "seconds."
                        ),
                        icon="❌",
                    )

                elif not recording_title.strip():

                    st.toast(
                        (
                            "Please enter a "
                            "recording title."
                        ),
                        icon="⚠️",
                    )

                else:

                    recorded_file = (
                        RecordedAudioFile(
                            data=recorded_bytes,
                        )
                    )

                    process_memory_upload(
                        recorded_file,
                        recording_title,
                    )

    else:

        st.caption(
            (
                "If microphone access is denied, "
                "allow microphone permission in "
                "your browser and try again."
            )
        )