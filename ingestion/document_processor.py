"""Document extraction with page/section context and conservative validation.

Legacy extract_*_text functions remain available and still return strings.
Scanned-PDF OCR is optional and uses the already-configured Ollama vision
processor; it is disabled by default to avoid unexpected API calls/costs.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import logging
import random
import time
from typing import Any

import fitz
from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from config.logging_config import StageTimer, get_logger, log_event
from config.settings import settings
from ingestion.processing import clean_and_normalize_text
from ingestion.processing_models import ExtractionError, ProcessedContent

logger = get_logger(__name__)


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


def _render_ocr_pdf_page(
    page: fitz.Page,
    page_number: int,
    file_name: str,
    image_path: Path,
) -> None:
    """Render a PDF page on the owning thread; PyMuPDF objects stay thread-local."""
    dpi = max(72, min(int(getattr(settings, "PDF_OCR_DPI", 144)), 300))
    with StageTimer(logger, "pdf_ocr_page_render", file=file_name,
                    page=page_number, dpi=dpi) as timer:
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        pixmap.save(str(image_path))
        timer.set(width=pixmap.width, height=pixmap.height,
                  image_bytes=pixmap.stride * pixmap.height)


def _ocr_pdf_page(page_or_image_path: fitz.Page | str | Path,
                  page_number: int, file_name: str | None = None) -> str:
    """Run the configured vision processor for one rendered PDF page.

    Passing a page object remains supported for compatibility, but the parallel
    ingestion path passes an image path so PyMuPDF objects never cross threads.
    """
    from ingestion.image_processor import extract_image_content

    if isinstance(page_or_image_path, fitz.Page):
        with TemporaryDirectory(prefix="memory_vault_pdf_ocr_") as temp_dir:
            image_path = Path(temp_dir) / f"page_{page_number}.png"
            _render_ocr_pdf_page(page_or_image_path, page_number,
                                 file_name or "unknown", image_path)
            return extract_image_content(str(image_path)).text
    return extract_image_content(str(page_or_image_path)).text


def _exception_chain(error: BaseException):
    """Yield an exception and its cause/context chain without looping."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _http_status(error: BaseException) -> int | None:
    for current in _exception_chain(error):
        candidates = [current, getattr(current, "response", None)]
        for candidate in candidates:
            if candidate is None:
                continue
            for name in ("status_code", "status"):
                value = getattr(candidate, name, None)
                try:
                    if value is not None:
                        numeric = int(value)
                        if 100 <= numeric <= 599:
                            return numeric
                except (TypeError, ValueError):
                    continue
    return None


def _is_retryable_ocr_error(error: BaseException) -> bool:
    """Retry only likely-transient provider/transport failures."""
    status = _http_status(error)
    if status is not None:
        return status in {408, 425, 429, 500, 502, 503, 504}
    names: list[str] = []
    messages: list[str] = []
    for current in _exception_chain(error):
        names.append(type(current).__name__.lower())
        messages.append(str(current).lower())
    name_text = " ".join(names)
    message_text = " ".join(messages)
    retryable_names = (
        "timeout", "connecterror", "connectionerror", "remoteprotocolerror",
        "temporarilyunavailable", "ratelimit", "toomanyrequests",
        "networkerror", "readerror", "writeerror", "pooltimeout", "protocolerror",
    )
    if any(token in name_text for token in retryable_names):
        return True
    retryable_messages = (
        "rate limit", "rate_limit", "too many requests", "temporarily unavailable",
        "timed out", "timeout", "connection reset", "connection refused",
        "server disconnected", "try again",
    )
    return any(token in message_text for token in retryable_messages)


def _retry_after_seconds(error: BaseException) -> float | None:
    """Read a provider Retry-After header when the exception exposes one."""
    for current in _exception_chain(error):
        candidates = [current, getattr(current, "response", None)]
        for candidate in candidates:
            if candidate is None:
                continue
            headers = getattr(candidate, "headers", None)
            if not headers:
                continue
            try:
                value = headers.get("Retry-After") or headers.get("retry-after")
            except Exception:
                value = None
            if value is None:
                continue
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                try:
                    retry_at = parsedate_to_datetime(str(value))
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    continue
    return None


def _ocr_pdf_page_with_retries(
    image_path: str,
    page_number: int,
    file_name: str,
    max_retries: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
) -> tuple[str | None, int, str | None]:
    """OCR one pre-rendered page, retrying only transient failures.

    A result tuple is returned for both success and failure so one failed page
    does not cancel other pages. The exception message/body is never logged.
    """
    retries = 0
    while True:
        attempt = retries + 1
        try:
            text = _ocr_pdf_page(image_path, page_number, file_name)
            log_event(logger, "INGEST_PDF_OCR_PAGE_RESULT", file=file_name,
                      page=page_number, status="success" if text else "empty",
                      attempt=attempt, retries=retries, chars=len(text or ""))
            return (text or ""), retries, None
        except Exception as error:
            retryable = _is_retryable_ocr_error(error)
            if not retryable or retries >= max_retries:
                log_event(logger, "INGEST_PDF_PAGE_ERROR", logging.WARNING,
                          file=file_name, page=page_number, stage="ocr",
                          error=type(error).__name__, attempt=attempt,
                          retries=retries, retryable=retryable)
                log_event(logger, "INGEST_PDF_OCR_PAGE_RESULT", logging.WARNING,
                          file=file_name, page=page_number, status="failed",
                          attempt=attempt, retries=retries,
                          error=type(error).__name__)
                return None, retries, type(error).__name__

            exponential = min(max_delay_seconds,
                              base_delay_seconds * (2 ** retries))
            jitter = random.uniform(0.0, min(exponential * 0.25, 1.0))
            delay = min(max_delay_seconds, exponential + jitter)
            provider_delay = _retry_after_seconds(error)
            if provider_delay is not None:
                # Never retry earlier than the provider's explicit guidance.
                delay = max(delay, provider_delay)
            log_event(logger, "INGEST_PDF_OCR_RETRY", logging.WARNING,
                      file=file_name, page=page_number, attempt=attempt,
                      retry=retries + 1, max_retries=max_retries,
                      delay_seconds=delay, error=type(error).__name__)
            time.sleep(delay)
            retries += 1


def _log_pdf_page_progress(record: dict[str, Any], pages_total: int) -> None:
    """Emit each page's progress as soon as its own extraction/OCR work completes."""
    if record.get("progress_logged"):
        return
    page_text = record["text"]
    method = record["method"]
    if record["ocr_attempted"] and record["ocr_error"] is None:
        ocr_text = clean_and_normalize_text(record["ocr_text"] or "")
        if ocr_text and len(ocr_text) > len(page_text):
            page_text = ocr_text
            method = "ollama_vision_ocr"
    completed_at = record.get("page_completed_at") or time.perf_counter()
    log_event(logger, "INGEST_PDF_PAGE_PROGRESS", file=record["file_name"],
              page=record["page"], pages_total=pages_total, chars=len(page_text),
              method=method, ocr_attempted=record["ocr_attempted"],
              seconds=max(0.0, completed_at - record["page_started_at"]))
    record["progress_logged"] = True


def _collect_completed_ocr(
    completed: set[Future],
    pending: dict[Future, dict[str, Any]],
) -> None:
    """Transfer finished future results into their page records and release files."""
    for future in completed:
        record = pending.pop(future)
        try:
            text, retries, error_name = future.result()
        except Exception as error:
            text, retries, error_name = None, 0, type(error).__name__
            log_event(logger, "INGEST_PDF_PAGE_ERROR", logging.WARNING,
                      file=record["file_name"], page=record["page"],
                      stage="ocr_worker", error=error_name)
        record["ocr_text"] = text
        record["ocr_retries"] = retries
        record["ocr_error"] = error_name
        record["page_completed_at"] = time.perf_counter()
        _log_pdf_page_progress(record, record["pages_total"])
        try:
            record["image_path"].unlink(missing_ok=True)
        except OSError:
            log_event(logger, "INGEST_PDF_TEMP_IMAGE_CLEANUP_ERROR", logging.WARNING,
                      file=record["file_name"], page=record["page"])

def extract_pdf_content(file_path: str) -> ProcessedContent:
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("PDF file does not exist or is not a regular file.")
    started = time.perf_counter()
    warnings: list[str] = []
    page_records: list[dict[str, Any]] = []
    rendered_text: list[str] = []
    ocr_used = False
    ocr_enabled = bool(getattr(settings, "PDF_OCR_ENABLED", False))
    min_chars = max(0, int(getattr(settings, "PDF_OCR_MIN_PAGE_TEXT_CHARS", 20)))
    ocr_workers = max(1, int(getattr(settings, "PDF_OCR_WORKERS", 3)))
    max_retries = max(0, int(getattr(settings, "PDF_OCR_MAX_RETRIES", 2)))
    base_delay = max(0.0, float(getattr(settings, "PDF_OCR_RETRY_BASE_DELAY_SECONDS", 1.0)))
    max_delay = max(base_delay, float(getattr(settings, "PDF_OCR_RETRY_MAX_DELAY_SECONDS", 8.0)))
    ocr_eligible_pages = 0
    ocr_attempted_pages = 0
    ocr_successful_pages = 0
    ocr_failed_pages = 0
    ocr_retried_pages = 0
    try:
        with StageTimer(logger, "pdf_open", file=path.name) as timer:
            document = fitz.open(str(path))
            timer.set(pages=len(document))
    except Exception as error:
        raise ExtractionError("The PDF could not be opened by the PDF parser.") from error

    try:
        if len(document) == 0:
            raise ExtractionError("The PDF contains no pages.")
        metadata = _pdf_metadata(document)
        # Rendering remains on this thread because PyMuPDF document/page objects
        # are not passed to workers. Only independent provider calls run in parallel.
        temp_context = (
            TemporaryDirectory(prefix="memory_vault_pdf_ocr_")
            if ocr_enabled else nullcontext(None)
        )
        with temp_context as temp_dir:
            executor = (
                ThreadPoolExecutor(max_workers=ocr_workers, thread_name_prefix="pdf-ocr")
                if ocr_enabled else None
            )
            pending: dict[Future, dict[str, Any]] = {}
            try:
                if ocr_enabled:
                    log_event(logger, "INGEST_PDF_OCR_PARALLEL_START", file=path.name,
                              workers=ocr_workers, max_retries=max_retries,
                              eligible_threshold_chars=min_chars)
                for index, page in enumerate(document, start=1):
                    page_started = time.perf_counter()
                    text_extract_started = time.perf_counter()
                    text_extract_error = False
                    try:
                        page_text = page.get_text("text") or ""
                    except Exception as error:
                        page_text = ""
                        text_extract_error = True
                        warnings.append(f"Text extraction failed on PDF page {index}.")
                        log_event(logger, "INGEST_PDF_PAGE_ERROR", logging.WARNING, file=path.name,
                                  page=index, stage="text_extraction", error=type(error).__name__)
                    log_event(logger, "INGEST_TIMING", stage="pdf_page_text_extraction", file=path.name,
                              page=index, seconds=time.perf_counter() - text_extract_started,
                              status="error" if text_extract_error else "success")
                    page_text = clean_and_normalize_text(page_text)
                    page_ocr_eligible = len(page_text) < min_chars
                    if page_ocr_eligible:
                        ocr_eligible_pages += 1

                    record: dict[str, Any] = {
                        "page": index,
                        "pages_total": len(document),
                        "file_name": path.name,
                        "text": page_text,
                        "method": "pymupdf_text",
                        "page_started_at": page_started,
                        "page_completed_at": time.perf_counter(),
                        "ocr_eligible": page_ocr_eligible,
                        "ocr_attempted": False,
                        "ocr_text": None,
                        "ocr_retries": 0,
                        "ocr_error": None,
                        "image_path": None,
                    }
                    page_records.append(record)

                    if not (ocr_enabled and page_ocr_eligible):
                        _log_pdf_page_progress(record, len(document))
                        continue

                    record["ocr_attempted"] = True
                    ocr_attempted_pages += 1
                    image_path = Path(temp_dir) / f"page_{index}.png"
                    record["image_path"] = image_path
                    try:
                        _render_ocr_pdf_page(page, index, path.name, image_path)
                        if executor is None:
                            # Defensive fallback: OCR cannot run in parallel unless enabled.
                            record["ocr_text"] = _ocr_pdf_page_with_retries(
                                str(image_path), index, path.name, max_retries,
                                base_delay, max_delay
                            )[0]
                        else:
                            future = executor.submit(
                                _ocr_pdf_page_with_retries, str(image_path), index,
                                path.name, max_retries, base_delay, max_delay
                            )
                            pending[future] = record
                            record["page_completed_at"] = None
                            # Keep both queued requests and rendered files strictly bounded.
                            if len(pending) >= ocr_workers:
                                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                                _collect_completed_ocr(completed, pending)
                    except Exception as error:
                        record["ocr_error"] = type(error).__name__
                        record["ocr_retries"] = 0
                        record["page_completed_at"] = time.perf_counter()
                        try:
                            image_path.unlink(missing_ok=True)
                        except OSError:
                            pass
                        log_event(logger, "INGEST_PDF_PAGE_ERROR", logging.WARNING,
                                  file=path.name, page=index, stage="ocr_render",
                                  error=type(error).__name__)
                        _log_pdf_page_progress(record, len(document))

                    # Collect finished tasks promptly, even when the current
                    # document has fewer pending OCR pages than the worker limit.
                    if pending:
                        if len(pending) >= ocr_workers:
                            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                        else:
                            completed, _ = wait(pending, timeout=0, return_when=FIRST_COMPLETED)
                        if completed:
                            _collect_completed_ocr(completed, pending)

                while pending:
                    completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                    _collect_completed_ocr(completed, pending)
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=False)

        for record in page_records:
            index = record["page"]
            page_text = record["text"]
            if record["ocr_attempted"]:
                retry_count = int(record["ocr_retries"] or 0)
                if retry_count > 0:
                    ocr_retried_pages += 1
                ocr_text = clean_and_normalize_text(record["ocr_text"] or "")
                if record["ocr_error"] is not None:
                    ocr_failed_pages += 1
                    warnings.append(
                        f"OCR could not process PDF page {index}; available PDF text was retained."
                    )
                elif not ocr_text:
                    ocr_failed_pages += 1
                    if not page_text:
                        warnings.append(f"OCR did not produce usable text for PDF page {index}.")
                else:
                    ocr_successful_pages += 1
                    if len(ocr_text) > len(page_text):
                        page_text = ocr_text
                        record["method"] = "ollama_vision_ocr"
                        ocr_used = True
            record["text"] = page_text
            if page_text:
                rendered_text.append(f"[Page {index}]\n{page_text}")
            else:
                warnings.append(f"PDF page {index} contains no extractable text.")

        metadata["pages"] = [
            {"page": record["page"], "character_count": len(record["text"]),
             "extraction_method": record["method"]}
            for record in page_records
        ]
        metadata["pages_with_text"] = sum(
            1 for record in page_records if record["text"]
        )
        metadata["extraction_method"] = (
            "pymupdf_text+ollama_vision_ocr" if ocr_used else "pymupdf_text"
        )
    finally:
        document.close()

    with StageTimer(logger, "pdf_text_assembly", file=path.name,
                    pages=len(page_records)) as timer:
        text = "\n\n".join(rendered_text)
        timer.set(chars=len(text))
    log_event(logger, "INGEST_PDF_OCR_SUMMARY", file=path.name, enabled=ocr_enabled,
              eligible=ocr_eligible_pages, attempted=ocr_attempted_pages,
              succeeded=ocr_successful_pages, failed=ocr_failed_pages,
              retried=ocr_retried_pages, workers=ocr_workers if ocr_enabled else 0,
              max_retries=max_retries if ocr_enabled else 0)
    if not text.strip():
        if not ocr_enabled:
            raise ExtractionError("No selectable text was found in the PDF. Scanned-PDF OCR is disabled; set PDF_OCR_ENABLED=true and configure Ollama to enable it.")
        raise ExtractionError("No usable text could be extracted from the PDF, including the configured OCR fallback.")
    if page_records and any(not record["text"] for record in page_records):
        warnings.append("One or more PDF pages did not yield text; inspect the source if those pages contain important content.")
    log_event(logger, "INGEST_TIMING", stage="pdf_text_extraction", file=path.name,
              seconds=time.perf_counter() - started, pages=len(page_records), ocr_enabled=ocr_enabled,
              ocr_used=ocr_used, pages_with_text=metadata.get("pages_with_text"))
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
