from pathlib import Path

import fitz
from docx import Document


def extract_text_from_txt(
    file_path: str
) -> str:

    path = Path(file_path)

    try:

        return path.read_text(
            encoding="utf-8"
        )

    except UnicodeDecodeError:

        return path.read_text(
            encoding="latin-1"
        )


def extract_text_from_pdf(
    file_path: str
) -> str:

    document = fitz.open(
        file_path
    )

    pages = []

    try:

        for page in document:

            text = page.get_text()

            if text:

                pages.append(text)

    finally:

        document.close()


    return "\n\n".join(pages)


def extract_text_from_docx(
    file_path: str
) -> str:

    document = Document(
        file_path
    )

    paragraphs = []


    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:

            paragraphs.append(text)


    return "\n\n".join(paragraphs)


def extract_document_text(
    file_path: str
) -> str:

    extension = Path(
        file_path
    ).suffix.lower()


    if extension == ".txt":

        return extract_text_from_txt(
            file_path
        )


    if extension == ".pdf":

        return extract_text_from_pdf(
            file_path
        )


    if extension == ".docx":

        return extract_text_from_docx(
            file_path
        )


    raise ValueError(
        f"Unsupported document type: {extension}"
    )