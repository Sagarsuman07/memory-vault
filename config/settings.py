import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    APP_ENV = os.getenv("APP_ENV", "development")

    LLM_API_KEY = os.getenv("LLM_API_KEY")

    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_TRACING = os.getenv(
        "LANGSMITH_TRACING",
        "false"
    )

    LANGSMITH_PROJECT = os.getenv(
        "LANGSMITH_PROJECT",
        "memory-vault"
    )


settings = Settings()