"""Document extraction with page/section context and conservative validation.

Legacy extract_*_text functions remain available and still return strings.
Scanned-PDF OCR is optional and uses the already-configured Ollama vision
processor; it is disabled by default to avoid unexpected API calls/costs.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import fitz
from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from config.settings import settings
from ingestion.processing import clean_and_normalize_text
from ingestion.processing_models import ExtractionError, ProcessedContent


def extract_text_from_txt(file_path: str) -> str:
    return extract_txt_content(file_path).text


def extract_txt_content(file_path: str) -> ProcessedContent:
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("Text file does not exist or is not a regular file.")
    warnings: list[str] = []
    try:
        text = path.read_text(encoding="utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
            encoding = "latin-1"
            warnings.append("Text file was decoded using latin-1 because UTF-8 decoding failed.")
    cleaned = clean_and_normalize_text(text)
    if not cleaned:
        raise ExtractionError("The text file contains no usable text.")
    return ProcessedContent(
        text=cleaned,
        metadata={"document_type": "text", "extraction_method": "plain_text", "encoding": encoding,
                  "character_count": len(cleaned), "line_count": len(cleaned.splitlines())},
        warnings=tuple(warnings),
        status="partial" if warnings else "succeeded",
    )


def _pdf_metadata(document: fitz.Document) -> dict[str, Any]:
    source = document.metadata or {}
    result: dict[str, Any] = {
        "document_type": "pdf",
        "extraction_method": "pymupdf_text",
        "page_count": len(document),
    }
    for source_key, target_key in (("title", "title"), ("author", "author"),
                                   ("creationDate", "creation_date"), ("modDate", "modified_date"),
                                   ("producer", "producer")):
        value = source.get(source_key)
        if isinstance(value, str) and value.strip():
            result[target_key] = value.strip()
    return result


def _ocr_pdf_page(page: fitz.Page, page_number: int) -> str:
    """Run the existing image vision processor on a rendered page."""
    from ingestion.image_processor import extract_image_content
    dpi = max(72, min(int(getattr(settings, "PDF_OCR_DPI", 144)), 300))
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    with TemporaryDirectory(prefix="memory_vault_pdf_ocr_") as temp_dir:
        image_path = Path(temp_dir) / f"page_{page_number}.png"
        pixmap.save(str(image_path))
        result = extract_image_content(str(image_path))
        return result.text


def extract_pdf_content(file_path: str) -> ProcessedContent:
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("PDF file does not exist or is not a regular file.")
    warnings: list[str] = []
    page_records: list[dict[str, Any]] = []
    rendered_text: list[str] = []
    ocr_used = False
    ocr_enabled = bool(getattr(settings, "PDF_OCR_ENABLED", False))
    min_chars = max(0, int(getattr(settings, "PDF_OCR_MIN_PAGE_TEXT_CHARS", 20)))
    try:
        document = fitz.open(str(path))
    except Exception as error:
        raise ExtractionError("The PDF could not be opened by the PDF parser.") from error
    try:
        if len(document) == 0:
            raise ExtractionError("The PDF contains no pages.")
        metadata = _pdf_metadata(document)
        for index, page in enumerate(document, start=1):
            try:
                page_text = page.get_text("text") or ""
            except Exception:
                page_text = ""
                warnings.append(f"Text extraction failed on PDF page {index}.")
            page_text = clean_and_normalize_text(page_text)
            method = "pymupdf_text"
            if ocr_enabled and len(page_text) < min_chars:
                try:
                    ocr_text = clean_and_normalize_text(_ocr_pdf_page(page, index))
                    if ocr_text and len(ocr_text) > len(page_text):
                        page_text = ocr_text
                        method = "ollama_vision_ocr"
                        ocr_used = True
                    elif not page_text:
                        warnings.append(f"OCR did not produce usable text for PDF page {index}.")
                except Exception:
                    warnings.append(f"OCR could not process PDF page {index}; available PDF text was retained.")
            if page_text:
                rendered_text.append(f"[Page {index}]\n{page_text}")
            else:
                warnings.append(f"PDF page {index} contains no extractable text.")
            page_records.append({"page": index, "character_count": len(page_text), "extraction_method": method})
        metadata["pages"] = page_records
        metadata["pages_with_text"] = sum(1 for page in page_records if page["character_count"] > 0)
        metadata["extraction_method"] = "pymupdf_text+ollama_vision_ocr" if ocr_used else "pymupdf_text"
    finally:
        document.close()
    text = "\n\n".join(rendered_text)
    if not text.strip():
        if not ocr_enabled:
            raise ExtractionError("No selectable text was found in the PDF. Scanned-PDF OCR is disabled; set PDF_OCR_ENABLED=true and configure Ollama to enable it.")
        raise ExtractionError("No usable text could be extracted from the PDF, including the configured OCR fallback.")
    if page_records and any(page["character_count"] == 0 for page in page_records):
        warnings.append("One or more PDF pages did not yield text; inspect the source if those pages contain important content.")
    return ProcessedContent(text=text, metadata=metadata, warnings=tuple(dict.fromkeys(warnings)),
                            status="partial" if warnings else "succeeded")


def extract_text_from_pdf(file_path: str) -> str:
    return extract_pdf_content(file_path).text


def _iter_docx_blocks(document):
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def extract_docx_content(file_path: str) -> ProcessedContent:
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("Word document does not exist or is not a regular file.")
    try:
        document = Document(str(path))
    except Exception as error:
        raise ExtractionError("The Word document could not be parsed.") from error
    blocks: list[str] = []
    headings: list[str] = []
    paragraph_count = 0
    table_count = 0
    table_row_count = 0
    for block in _iter_docx_blocks(document):
        if isinstance(block, Paragraph):
            value = clean_and_normalize_text(block.text)
            if not value:
                continue
            paragraph_count += 1
            style_name = (block.style.name or "").strip() if block.style else ""
            if style_name.lower() == "title":
                blocks.append(f"# {value}")
                headings.append(value)
            elif style_name.lower().startswith("heading"):
                blocks.append(f"## {value}")
                headings.append(value)
            else:
                blocks.append(value)
        elif isinstance(block, Table):
            table_count += 1
            rows = []
            for row in block.rows:
                cells = [clean_and_normalize_text(cell.text) for cell in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            table_row_count += len(rows)
            if rows:
                blocks.append("[Table]")
                blocks.extend(rows)
                blocks.append("[/Table]")
    text = "\n\n".join(blocks).strip()
    if not text:
        raise ExtractionError("The Word document contains no usable paragraph or table text.")
    return ProcessedContent(
        text=text,
        metadata={"document_type": "docx", "extraction_method": "python_docx",
                  "paragraph_count": paragraph_count, "table_count": table_count,
                  "table_row_count": table_row_count, "headings": headings},
    )


def extract_text_from_docx(file_path: str) -> str:
    return extract_docx_content(file_path).text


def extract_document_content(file_path: str) -> ProcessedContent:
    extension = Path(file_path).suffix.lower()
    if extension == ".txt":
        return extract_txt_content(file_path)
    if extension == ".pdf":
        return extract_pdf_content(file_path)
    if extension == ".docx":
        return extract_docx_content(file_path)
    raise ExtractionError(f"Unsupported document type: {extension or '(no extension)'}")


def extract_document_text(file_path: str) -> str:
    """Backward-compatible entry point returning extracted text only."""
    return extract_document_content(file_path).text
