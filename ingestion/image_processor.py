from pathlib import Path

from ollama import Client

from config.settings import settings


def get_ollama_client():
    if not settings.OLLAMA_API_KEY:
        raise ValueError(
            "OLLAMA_API_KEY is not configured."
        )

    if not settings.OLLAMA_MODEL:
        raise ValueError(
            "OLLAMA_MODEL is not configured."
        )

    return Client(
        host=settings.OLLAMA_HOST,
        headers={
            "Authorization": (
                f"Bearer {settings.OLLAMA_API_KEY}"
            )
        },
    )


def extract_text_from_image(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image file not found: {file_path}"
        )

    client = get_ollama_client()

    try:
        response = client.chat(
            model=settings.OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                            "You are a memory extraction system. Examine this image carefully "
                            "and describe its full content as natural, information-dense text "
                            "that will be stored and later searched by a user asking questions "
                            "about it.\n\n"
                            "Write in clear, self-contained sentences (not a bare list) so the "
                            "text makes sense on its own, without seeing the image. Include:\n"
                            "- A brief description of what the image is and what it shows "
                            "(e.g., a receipt, ID card, photo of people, screenshot, whiteboard, "
                            "ticket, etc.)\n"
                            "- All visible names, dates, times, locations, prices, IDs, booking "
                            "or reference numbers, product names, phone numbers, emails, and any "
                            "other readable text — transcribed exactly as shown, not reformatted.\n"
                            "- Relevant visual details for non-document images (people, objects, "
                            "setting, activity, colors, notable context) that someone might later "
                            "ask about.\n\n"
                            "Rules:\n"
                            "- Do not invent or infer information not visibly present in the image.\n"
                            "- If text is blurry, cut off, or ambiguous, say so explicitly rather "
                            "than guessing (e.g., 'partially legible number, possibly 4521...').\n"
                            "- Do not use filler phrases like 'the image shows' repeatedly or "
                            "generic hedging — write directly and factually.\n"
                            "- Output plain text only, no markdown, bullet points, or headers."
                    ),
                    "images": [str(path)],
                }
            ],
        )

    except Exception as error:
        raise RuntimeError(
            f"Image extraction failed: {error}"
        ) from error

    extracted_text = response.message.content

    if not extracted_text or not extracted_text.strip():
        raise ValueError(
            "The vision model returned no extracted text."
        )

    return extracted_text.strip()