"""Thread-safe cached local embedding model used by the existing RAG index."""
import logging
import time
from threading import Lock

from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings
from config.logging_config import get_logger, log_event, timed_lock

MODEL_NAME = getattr(settings, "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_embeddings = None
_embeddings_lock = Lock()
logger = get_logger(__name__)


def get_embeddings():
    """Initialize the existing local model exactly once across worker threads."""
    global _embeddings
    if _embeddings is None:
        with timed_lock(logger, _embeddings_lock, "embedding_model_init", "initialize_model",
                        model=MODEL_NAME):
            if _embeddings is None:
                started = time.perf_counter()
                try:
                    model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
                except Exception as error:
                    log_event(logger, "INGEST_COMPONENT_INITIALIZATION", logging.WARNING,
                              component="embedding_model", model=MODEL_NAME, status="error",
                              error=type(error).__name__, seconds=time.perf_counter() - started)
                    raise
                _embeddings = model
                log_event(logger, "INGEST_COMPONENT_INITIALIZATION", component="embedding_model",
                          model=MODEL_NAME, status="success", seconds=time.perf_counter() - started)
    return _embeddings
