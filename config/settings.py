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
        if self.SQLITE_BUSY_TIMEOUT_MS < 0:
            raise ValueError("SQLITE_BUSY_TIMEOUT_MS cannot be negative")
        if self.PDF_OCR_WORKERS <= 0:
            raise ValueError("PDF_OCR_WORKERS must be greater than zero")
        if self.PDF_OCR_WORKERS > 8:
            raise ValueError("PDF_OCR_WORKERS cannot exceed 8")
        if self.PDF_OCR_MAX_RETRIES < 0:
            raise ValueError("PDF_OCR_MAX_RETRIES cannot be negative")
        if self.PDF_OCR_RETRY_BASE_DELAY_SECONDS < 0:
            raise ValueError("PDF_OCR_RETRY_BASE_DELAY_SECONDS cannot be negative")
        if self.PDF_OCR_RETRY_MAX_DELAY_SECONDS < self.PDF_OCR_RETRY_BASE_DELAY_SECONDS:
            raise ValueError("PDF_OCR_RETRY_MAX_DELAY_SECONDS must be >= PDF_OCR_RETRY_BASE_DELAY_SECONDS")


settings = Settings()
settings.validate_knowledge_base_settings()

