"""File-based, rotating logging for the Memory Vault ingestion pipeline.

* Logs go to ``logs/ingestion.log`` (project root) - not the terminal.
* The terminal only receives WARNING and above.
* The file rotates, so it cannot grow without bound
  (default: 5 MB per file, 5 old files kept => ~30 MB maximum).
* Only identifiers, counts and timings are logged. Never log extracted text,
  file contents, credentials, tokens or environment variables.

Optional environment overrides (all have safe defaults):
    INGESTION_LOG_DIR           default "logs"
    INGESTION_LOG_MAX_BYTES     default 5242880
    INGESTION_LOG_BACKUP_COUNT  default 5
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
# Package namespaces whose loggers write to the ingestion log.
_NAMESPACES = ("ingestion", "memory", "rag")

_SETUP_LOCK = threading.Lock()
_CONFIGURED = False
_WHITESPACE = re.compile(r"\s+")


def _env_int(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def get_log_file_path() -> Path:
    directory = Path(os.getenv("INGESTION_LOG_DIR", "logs"))
    if not directory.is_absolute():
        directory = _PROJECT_ROOT / directory
    return directory / "ingestion.log"


def setup_ingestion_logging() -> None:
    """Attach the rotating file handler once per process (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    with _SETUP_LOCK:
        if _CONFIGURED:
            return
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        console.setFormatter(formatter)

        file_handler = None
        try:
            path = get_log_file_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                path,
                maxBytes=_env_int("INGESTION_LOG_MAX_BYTES", 5 * 1024 * 1024, 10_000),
                backupCount=_env_int("INGESTION_LOG_BACKUP_COUNT", 5, 1),
                encoding="utf-8",
                delay=True,
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
        except OSError:
            # Logging must never stop ingestion. Fall back to console warnings only.
            file_handler = None

        for namespace in _NAMESPACES:
            target = logging.getLogger(namespace)
            target.setLevel(logging.INFO)
            target.propagate = False  # keep INFO lines off the terminal
            target.addHandler(console)
            if file_handler is not None:
                target.addHandler(file_handler)

        if file_handler is None:
            logging.getLogger("ingestion").warning(
                "Ingestion file logging is unavailable; could not open the log file."
            )
        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger wired to ``logs/ingestion.log`` (pass ``__name__``)."""
    setup_ingestion_logging()
    return logging.getLogger(name)


def _format_value(value) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:.3f}"
    text = _WHITESPACE.sub("_", str(value)).strip("_")
    return text[:200] if text else "-"


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields) -> None:
    """Write ``EVENT key=value key=value`` on one line. Never raises."""
    try:
        parts = " ".join(
            f"{key}={_format_value(value)}" for key, value in fields.items() if value is not None
        )
        logger.log(level, "%s %s", event, parts)
    except Exception:
        pass


class StageTimer:
    """Context manager that logs one ``INGEST_TIMING`` line for a stage.

    Usage::

        with StageTimer(logger, "chunking", file=name) as timer:
            chunks = split_text(text)
            timer.set(chars=len(text), chunks=len(chunks))

    If the block raises, the line is still written (WARNING, ``status=error``,
    exception class name only) and the exception propagates unchanged.
    """

    def __init__(self, logger: logging.Logger, stage: str, **fields):
        self._logger = logger
        self._stage = stage
        self._fields = dict(fields)
        self._start = 0.0
        self.seconds = 0.0

    def set(self, **fields) -> None:
        self._fields.update(fields)

    def __enter__(self) -> "StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.seconds = time.perf_counter() - self._start
        fields = dict(self._fields)
        ordered: dict = {"stage": self._stage}
        if "file" in fields:
            ordered["file"] = fields.pop("file")
        ordered["seconds"] = self.seconds
        ordered.update(fields)
        level = logging.INFO
        if exc_type is not None:
            ordered["status"] = "error"
            ordered["error"] = exc_type.__name__
            level = logging.WARNING
        log_event(self._logger, "INGEST_TIMING", level, **ordered)
        return False
