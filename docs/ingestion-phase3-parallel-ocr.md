# Phase 3 — Bounded parallel PDF OCR

## Scope and compatibility

This phase only parallelizes OCR vision-provider calls for eligible PDF pages. It does not change the OCR eligibility threshold, rendering DPI, prompt, text cleaning, page markers, chunking, embeddings, Chroma writes, duplicate handling, or rollback behavior.

PDF text extraction and page rendering remain on the thread that owns the PyMuPDF document. Only rendered image paths are passed to worker threads, avoiding concurrent access to shared PyMuPDF page/document objects. Each provider call continues to create its own configured client through the existing image processor.

## Configuration

- `PDF_OCR_ENABLED`: remains disabled by default.
- `PDF_OCR_WORKERS`: maximum concurrent OCR provider calls; defaults to 3. Set to 1 for the sequential fallback.
- `PDF_OCR_MAX_RETRIES`: additional retries for transient errors; defaults to 2 (up to 3 total attempts).
- `PDF_OCR_RETRY_BASE_DELAY_SECONDS`: exponential backoff base; defaults to 1 second.
- `PDF_OCR_RETRY_MAX_DELAY_SECONDS`: cap for computed exponential backoff; defaults to 8 seconds. Provider `Retry-After` guidance is honored even if longer than this cap.

Retries are limited to common transient HTTP statuses (408, 425, 429, 500, 502, 503, 504) and recognizable transport/timeout errors. Permanent status errors such as HTTP 400 are not retried. Error payloads and exception messages are not written to logs.

## Bounded execution and ordering

- The configured worker count bounds both in-flight provider requests and temporary rendered page images.
- Completed image files are removed as their OCR task results are collected.
- Provider completions can occur out of order, but extracted content and per-page metadata are assembled in original page order.
- A page failure is isolated; other pages can still contribute text. Existing warnings for OCR failures and pages with no usable text are preserved.
- `PDF_OCR_WORKERS=1` retains a serial processing option.

## Tests

Run focused OCR and extraction tests:

`python -m unittest tests.test_parallel_pdf_ocr tests.test_phase3_processing tests.test_ingestion_logging`

The parallel OCR tests cover output order under out-of-order completion, worker bounds, retry-after handling, retry exhaustion, permanent-error non-retry, and page-level failure isolation. The end-to-end performance comparison remains part of final Phase 5 validation; no performance gain is claimed until measured on the same OCR-heavy PDF.