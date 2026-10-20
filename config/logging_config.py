"""File-first logging for Memory Vault ingestion diagnostics."""
from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False


def configure_logging() -> Path:
    """Log detailed ingestion diagnostics to logs/ingestion.log.

    The log rotates at 5 MiB and keeps three backups. Only warnings/errors from
    these application loggers are sent to the console to avoid terminal noise.
    Calling this repeatedly is safe (e.g. Streamlit reruns).
    """
    global _CONFIGURED
    project_root = Path(__file__).resolve().parents[1]
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "ingestion.log"
    logger_names = ("memory.background_worker", "memory.async_ingestion", "rag.vector_store")
    for name in logger_names:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not any(getattr(handler, "_memory_vault_file_handler", False) for handler in logger.handlers):
            file_handler = RotatingFileHandler(
                log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
            file_handler._memory_vault_file_handler = True
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logger.addHandler(file_handler)
        if not any(getattr(handler, "_memory_vault_console_handler", False) for handler in logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler._memory_vault_console_handler = True
            console_handler.setLevel(logging.WARNING)
            console_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
            logger.addHandler(console_handler)
    _CONFIGURED = True
    return log_path
