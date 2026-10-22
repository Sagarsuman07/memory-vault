"""Vision-based image extraction with lazy provider loading and metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ingestion.processing import clean_and_normalize_text
from ingestion.processing_models import ExtractionError, ProcessedContent


def get_ollama_client():
    # Lazy import keeps local parsing/tests usable without the optional provider SDK.
    from config.settings import settings
    if not settings.OLLAMA_API_KEY:
        raise ValueError("OLLAMA_API_KEY is not configured.")
    if not settings.OLLAMA_MODEL:
        raise ValueError("OLLAMA_MODEL is not configured.")
    try:
        from ollama import Client
    except ImportError as error:
        raise RuntimeError("The ollama package is not installed in this environment.") from error
    return Client(host=settings.OLLAMA_HOST,
                  headers={"Authorization": f"Bearer {settings.OLLAMA_API_KEY}"})


def extract_image_content(file_path: str) -> ProcessedContent:
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("Image file does not exist or is not a regular file.")
    from config.settings import settings
    image_metadata: dict[str, Any] = {
        "document_type": "image", "extraction_method": "ollama_vision",
        "file_extension": path.suffix.lower(), "file_size_bytes": path.stat().st_size,
    }
    try:
        from PIL import Image
        with Image.open(path) as image:
            image_metadata["image_format"] = image.format
            image_metadata["image_width"] = image.width
            image_metadata["image_height"] = image.height
    except Exception:
        # Discovery already checks basic signatures. Some formats may not be
        # supported by Pillow; the vision provider can still decide if it can read them.
        image_metadata["image_dimensions_available"] = False
    client = get_ollama_client()
    prompt = (
        "You are extracting searchable information from a personal-memory image. "
        "Describe visible content factually and self-containedly. Transcribe readable "
        "text exactly, including names, dates, amounts, identifiers, addresses, and labels. "
        "Preserve relationships in tables/forms. For photos, describe relevant objects, "
        "setting, and actions. Do not guess unreadable content or invent facts; mark unclear "
        "text as illegible. Return plain text only."
    )
    try:
        response = client.chat(model=settings.OLLAMA_MODEL, messages=[{
            "role": "user", "content": prompt, "images": [str(path)]
        }])
        extracted = response.message.content
    except Exception as error:
        raise ExtractionError("Image vision extraction failed with the configured provider.") from error
    text = clean_and_normalize_text(extracted or "")
    if not text:
        raise ExtractionError("The vision model returned no usable image content.")
    image_metadata["character_count"] = len(text)
    return ProcessedContent(text=text, metadata=image_metadata)


def extract_text_from_image(file_path: str) -> str:
    """Backward-compatible entry point returning text only."""
    return extract_image_content(file_path).text
