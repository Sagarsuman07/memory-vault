"""Audio transcription with lazy provider loading and source metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ingestion.processing import clean_and_normalize_text
from ingestion.processing_models import ExtractionError, ProcessedContent

SUPPORTED_AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".m4a", ".mpeg", ".mpga", ".webm", ".ogg", ".flac"})


def get_groq_client():
    from config.settings import settings
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not configured.")
    try:
        from groq import Groq
    except ImportError as error:
        raise RuntimeError("The groq package is not installed in this environment.") from error
    return Groq(api_key=settings.GROQ_API_KEY)


def extract_audio_content(file_path: str) -> ProcessedContent:
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("Audio file does not exist or is not a regular file.")
    extension = path.suffix.lower()
    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported audio type: {extension or '(no extension)'}")
    from config.settings import settings
    client = get_groq_client()
    try:
        with path.open("rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(path.name, audio_file.read()),
                model=settings.GROQ_WHISPER_MODEL,
                response_format="json",
                temperature=0,
            )
        raw_text = transcription.text
    except Exception as error:
        raise ExtractionError("Audio transcription failed with the configured provider.") from error
    text = clean_and_normalize_text(raw_text or "")
    if not text:
        raise ExtractionError("Speech-to-text returned no usable transcript.")
    metadata: dict[str, Any] = {
        "document_type": "audio", "extraction_method": "groq_speech_to_text",
        "transcription_model": settings.GROQ_WHISPER_MODEL,
        "file_extension": extension, "file_size_bytes": path.stat().st_size,
        "character_count": len(text),
    }
    return ProcessedContent(text=text, metadata=metadata)


def extract_text_from_audio(file_path: str) -> str:
    """Backward-compatible entry point returning transcript text only."""
    return extract_audio_content(file_path).text
