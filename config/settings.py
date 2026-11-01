import os

from dotenv import load_dotenv


load_dotenv()


class Settings:

    # ============================================================
    # Application
    # ============================================================

    APP_ENV = os.getenv(
        "APP_ENV",
        "development",
    )

    DEMO_USER_ID = os.getenv(
        "DEMO_USER_ID",
        "demo_user",
    )

    DATABASE_PATH = os.getenv(
        "DATABASE_PATH",
        "memory_vault.db",
    )

    # ============================================================
    # Supabase PostgreSQL
    # ============================================================

    # Read the connection string from environment/.env first, then
    # Streamlit secrets when deployed on Streamlit Community Cloud.
    @staticmethod
    def _read_secret(name, default=None):
        value = os.getenv(name)
        if value:
            return value
        try:
            import streamlit as st
            return st.secrets.get(name, default)
        except Exception:
            return default

    DATABASE_URL = _read_secret.__func__("DATABASE_URL")

    # Cloudflare R2 private object storage
    R2_ACCOUNT_ID = _read_secret.__func__("R2_ACCOUNT_ID")
    R2_ACCESS_KEY_ID = _read_secret.__func__("R2_ACCESS_KEY_ID")
    R2_SECRET_ACCESS_KEY = _read_secret.__func__("R2_SECRET_ACCESS_KEY")
    R2_BUCKET_NAME = _read_secret.__func__("R2_BUCKET_NAME")
    R2_ENDPOINT_URL = _read_secret.__func__("R2_ENDPOINT_URL")
    MAX_UPLOAD_SIZE_BYTES = int(_read_secret.__func__("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024)))

    DB_POOL_MIN_SIZE = int(_read_secret.__func__("DB_POOL_MIN_SIZE", "1"))
    DB_POOL_MAX_SIZE = int(_read_secret.__func__("DB_POOL_MAX_SIZE", "3"))
    DB_POOL_TIMEOUT_SECONDS = float(
        _read_secret.__func__("DB_POOL_TIMEOUT_SECONDS", "10")
    )
    DB_CONNECT_TIMEOUT_SECONDS = int(
        _read_secret.__func__("DB_CONNECT_TIMEOUT_SECONDS", "8")
    )
    DB_STATEMENT_TIMEOUT_MS = int(
        _read_secret.__func__("DB_STATEMENT_TIMEOUT_MS", "30000")
    )


    UPLOAD_DIR = os.getenv(
        "UPLOAD_DIR",
        "uploads",
    )

    # ============================================================
    # LangGraph Checkpoint Database
    # ============================================================

    CHECKPOINT_DATABASE_PATH = os.getenv(
        "CHECKPOINT_DATABASE_PATH",
        "memory_vault_checkpoints.sqlite",
    )

    # ============================================================
    # Phase 0.3 - Execution Limits
    # ============================================================

    # Maximum number of tools that the application is allowed
    # to ACTUALLY EXECUTE during one request.
    #
    # IMPORTANT:
    # The final LLM call is NOT counted as a tool call.
    #
    # Example:
    #
    # MAX_TOOL_CALLS=1
    #
    # search_memories -> allowed
    # search_web      -> blocked
    # final LLM       -> allowed
    #
    MAX_TOOL_CALLS = int(
        os.getenv(
            "MAX_TOOL_CALLS",
            "4",
        )
    )

    # Maximum LangGraph recursion depth.
    #
    # This protects against an agent repeatedly cycling through
    # agent -> tools -> agent.
    #
    # This is separate from MAX_TOOL_CALLS.
    MAX_GRAPH_RECURSION = int(
        os.getenv(
            "MAX_GRAPH_RECURSION",
            "20",
        )
    )

    # ============================================================
    # Knowledge Base Preparation (Phase 1 contracts/configuration)
    # ============================================================

    # Preserve the existing chunking defaults during the contract phase.
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Preserve the current embedding model; changing it may require re-indexing.
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    # ============================================================
    # Qdrant (vector database)
    # ============================================================

    # Set QDRANT_URL (+ QDRANT_API_KEY for Qdrant Cloud) to use a Qdrant
    # server. If QDRANT_URL is empty, an embedded local database is stored
    # in QDRANT_PATH (development only; single process).
    QDRANT_URL = os.getenv("QDRANT_URL") or None
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
    QDRANT_PATH = os.getenv("QDRANT_PATH", ".qdrant")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "memory_vault")

    # Must match EMBEDDING_MODEL (all-MiniLM-L6-v2 produces 384 dimensions).
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

    # "hybrid" (dense + BM25 fused with RRF), "dense" or "bm25".
    # "dense" / "bm25" exist mainly to compare strategies in evaluation.
    RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").lower()

    # Indexing and worker settings are configuration-only in Phase 1.
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    INGESTION_MAX_ATTEMPTS = int(os.getenv("INGESTION_MAX_ATTEMPTS", "3"))
    INGESTION_RETRY_DELAY_SECONDS = float(
        os.getenv("INGESTION_RETRY_DELAY_SECONDS", "5")
    )
    INGESTION_WORKER_CONCURRENCY = int(
        os.getenv("INGESTION_WORKER_CONCURRENCY", "2")
    )

    # Keep local embedding inference from running in several threads at once.
    # This is conservative for the current HuggingFaceEmbeddings implementation.
    EMBEDDING_CONCURRENCY = int(os.getenv("EMBEDDING_CONCURRENCY", "1"))

    # SQLite waits briefly for another writer instead of failing immediately.
    SQLITE_BUSY_TIMEOUT_MS = int(os.getenv("SQLITE_BUSY_TIMEOUT_MS", "30000"))

    # ============================================================
    # Safety
    # ============================================================

    SAFETY_ENABLED = os.getenv(
        "SAFETY_ENABLED",
        "true",
    ).lower() == "true"

    # ============================================================
    # Voice Recording
    # ============================================================

    MIN_RECORDING_SECONDS = float(
        os.getenv(
            "MIN_RECORDING_SECONDS",
            "1.0",
        )
    )

    # ============================================================
    # Ollama
    # ============================================================

    OLLAMA_HOST = os.getenv(
        "OLLAMA_HOST",
        "https://ollama.com",
    )

    OLLAMA_API_KEY = os.getenv(
        "OLLAMA_API_KEY"
    )

    OLLAMA_MODEL = os.getenv(
        "OLLAMA_MODEL"
    )

    # ============================================================
    # Groq
    # ============================================================

    GROQ_API_KEY = os.getenv(
        "GROQ_API_KEY"
    )

    GROQ_WHISPER_MODEL = os.getenv(
        "GROQ_WHISPER_MODEL",
        "whisper-large-v3-turbo",
    )

    GROQ_LLM_MODEL = os.getenv(
        "GROQ_LLM_MODEL",
        "openai/gpt-oss-120b",
    )

    # ============================================================
    # Tavily
    # ============================================================

    TAVILY_API_KEY = os.getenv(
        "TAVILY_API_KEY"
    )

    # ============================================================
    # LangSmith
    # ============================================================

    LANGSMITH_TRACING = os.getenv(
        "LANGSMITH_TRACING",
        "true",
    ).lower() == "true"

    LANGSMITH_API_KEY = os.getenv(
        "LANGSMITH_API_KEY"
    )

    LANGSMITH_PROJECT = os.getenv(
        "LANGSMITH_PROJECT",
        "memory-vault",
    )



    # ============================================================
    # Phase 3 - Document extraction options
    # ============================================================

    # OCR uses the configured Ollama vision provider. Disabled by default
    # to avoid unexpected external calls or costs for scanned PDFs.
    PDF_OCR_ENABLED = os.getenv("PDF_OCR_ENABLED", "false").lower() == "true"
    PDF_OCR_DPI = int(os.getenv("PDF_OCR_DPI", "144"))
    PDF_OCR_MIN_PAGE_TEXT_CHARS = int(os.getenv("PDF_OCR_MIN_PAGE_TEXT_CHARS", "20"))
    # OCR page rendering stays serial; only independent vision calls are parallel.
    # Set PDF_OCR_WORKERS=1 for the sequential OCR fallback.
    PDF_OCR_WORKERS = int(os.getenv("PDF_OCR_WORKERS", "3"))
    PDF_OCR_MAX_RETRIES = int(os.getenv("PDF_OCR_MAX_RETRIES", "2"))
    PDF_OCR_RETRY_BASE_DELAY_SECONDS = float(
        os.getenv("PDF_OCR_RETRY_BASE_DELAY_SECONDS", "1")
    )
    PDF_OCR_RETRY_MAX_DELAY_SECONDS = float(
        os.getenv("PDF_OCR_RETRY_MAX_DELAY_SECONDS", "8")
    )

    def validate_knowledge_base_settings(self):
        if self.MAX_UPLOAD_SIZE_BYTES <= 0:
            raise ValueError("MAX_UPLOAD_SIZE_BYTES must be greater than zero")
        if self.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be greater than zero")
        if self.CHUNK_OVERLAP < 0 or self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE")
        if self.EMBEDDING_BATCH_SIZE <= 0:
            raise ValueError("EMBEDDING_BATCH_SIZE must be greater than zero")
        if self.INGESTION_MAX_ATTEMPTS <= 0:
            raise ValueError("INGESTION_MAX_ATTEMPTS must be greater than zero")
        if self.INGESTION_RETRY_DELAY_SECONDS < 0:
            raise ValueError("INGESTION_RETRY_DELAY_SECONDS cannot be negative")
        if self.INGESTION_WORKER_CONCURRENCY <= 0:
            raise ValueError("INGESTION_WORKER_CONCURRENCY must be greater than zero")
        if self.EMBEDDING_CONCURRENCY <= 0:
            raise ValueError("EMBEDDING_CONCURRENCY must be greater than zero")
        if self.EMBEDDING_DIMENSION <= 0:
            raise ValueError("EMBEDDING_DIMENSION must be greater than zero")
        if self.RETRIEVAL_MODE not in {"hybrid", "dense", "bm25"}:
            raise ValueError("RETRIEVAL_MODE must be one of: hybrid, dense, bm25")
        if self.SQLITE_BUSY_TIMEOUT_MS < 0:
            raise ValueError("SQLITE_BUSY_TIMEOUT_MS cannot be negative")
        if self.PDF_OCR_WORKERS <= 0:
            raise ValueError("PDF_OCR_WORKERS must be greater than zero")
        if self.PDF_OCR_WORKERS > 20:
            raise ValueError("PDF_OCR_WORKERS cannot exceed 20")
        if self.PDF_OCR_MAX_RETRIES < 0:
            raise ValueError("PDF_OCR_MAX_RETRIES cannot be negative")
        if self.PDF_OCR_RETRY_BASE_DELAY_SECONDS < 0:
            raise ValueError("PDF_OCR_RETRY_BASE_DELAY_SECONDS cannot be negative")
        if self.PDF_OCR_RETRY_MAX_DELAY_SECONDS < self.PDF_OCR_RETRY_BASE_DELAY_SECONDS:
            raise ValueError("PDF_OCR_RETRY_MAX_DELAY_SECONDS must be >= PDF_OCR_RETRY_BASE_DELAY_SECONDS")


settings = Settings()
settings.validate_knowledge_base_settings()

