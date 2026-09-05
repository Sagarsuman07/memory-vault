from pathlib import Path

from groq import Groq

from config.settings import settings


SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".mpeg",
    ".mpga",
    ".webm",
    ".ogg",
    ".flac",
}


def get_groq_client():
    if not settings.GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not configured."
        )

    return Groq(
        api_key=settings.GROQ_API_KEY
    )


def extract_text_from_audio(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {file_path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio type: {extension}"
        )

    client = get_groq_client()

    try:
        with open(path, "rb") as audio_file:
            transcription = (
                client.audio.transcriptions.create(
                    file=(
                        path.name,
                        audio_file.read()
                    ),
                    model=settings.GROQ_WHISPER_MODEL,
                    response_format="json",
                    temperature=0,
                )
            )

    except Exception as error:
        raise RuntimeError(
            f"Audio transcription failed: {error}"
        ) from error

    text = transcription.text

    if not text or not text.strip():
        raise ValueError(
            "Speech-to-text returned no transcript."
        )

    return text.strip()