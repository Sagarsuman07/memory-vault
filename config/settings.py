import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    # ========================================================
    # Application
    # ========================================================

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

    # ========================================================
    # LangGraph Checkpoint Database
    # ========================================================

    CHECKPOINT_DATABASE_PATH = os.getenv(
        "CHECKPOINT_DATABASE_PATH",
        "memory_vault_checkpoints.sqlite",
    )

    # ========================================================
    # Safety
    # ========================================================

    SAFETY_ENABLED = os.getenv(
        "SAFETY_ENABLED",
        "true",
    ).lower() == "true"

    # ========================================================
    # Phase 13 - Voice Recording
    # ========================================================

    MIN_RECORDING_SECONDS = float(
        os.getenv(
            "MIN_RECORDING_SECONDS",
            "1.0",
        )
    )

    # ========================================================
    # Ollama
    # ========================================================

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

    # ========================================================
    # Groq
    # ========================================================

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

    # ========================================================
    # Tavily
    # ========================================================

    TAVILY_API_KEY = os.getenv(
        "TAVILY_API_KEY"
    )

    # ========================================================
    # LangSmith
    # ========================================================

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


settings = Settings()