import os

from dotenv import load_dotenv


load_dotenv()


class Settings:

    APP_ENV = os.getenv(
        "APP_ENV",
        "development"
    )

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


settings = Settings()