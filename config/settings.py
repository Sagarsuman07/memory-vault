import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    APP_ENV = os.getenv("APP_ENV", "development")

    DEMO_USER_ID = os.getenv(
        "DEMO_USER_ID",
        "demo_user"
    )

    DATABASE_PATH = os.getenv(
        "DATABASE_PATH",
        "memory_vault.db"
    )

    UPLOAD_DIR = os.getenv(
        "UPLOAD_DIR",
        "uploads"
    )

    # Ollama Cloud
    OLLAMA_HOST = os.getenv(
        "OLLAMA_HOST",
        "https://ollama.com"
    )

    OLLAMA_API_KEY = os.getenv(
        "OLLAMA_API_KEY"
    )

    OLLAMA_MODEL = os.getenv(
        "OLLAMA_MODEL"
    )

    # Groq Speech-to-Text
    GROQ_API_KEY = os.getenv(
        "GROQ_API_KEY"
    )

    GROQ_WHISPER_MODEL = os.getenv(
        "GROQ_WHISPER_MODEL",
        "whisper-large-v3-turbo"
    )


settings = Settings()