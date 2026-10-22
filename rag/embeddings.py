"""Thread-safe cached local embedding model used by the existing RAG index."""
from threading import Lock
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings

MODEL_NAME = getattr(settings, "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_embeddings = None
_embeddings_lock = Lock()

def get_embeddings():
    """Initialize the existing local model exactly once across worker threads."""
    global _embeddings
    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                _embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return _embeddings
