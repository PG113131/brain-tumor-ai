import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.logger import logger


class Settings(BaseSettings):
    """
    Central configuration for the Brain Tumor AI project.
    All project settings should be defined here.
    """

    # ==========================================================
    # Application Configuration
    # ==========================================================
    APP_NAME: str = "Brain Tumor AI Diagnostic Suite"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # ==========================================================
    # API Keys
    # ==========================================================
    GROQ_API_KEY: str

    # ==========================================================
    # Database
    # ==========================================================
    DATABASE_URL: str = "sqlite:///./data/brain_tumor.db"

    # ==========================================================
    # Directory Paths
    # ==========================================================
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    DATA_DIR: Path = BASE_DIR / "data"

    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    HEATMAP_DIR: Path = DATA_DIR / "heatmaps"

    LOGS_DIR: Path = BASE_DIR / "logs"

    MODEL_DIR: Path = BASE_DIR / "models"
    MODEL_PATH: Path = MODEL_DIR / "best_model.pth"

    # ==========================================================
    # Model Configuration
    # ==========================================================
    IMAGE_SIZE: int = 224

    NUM_CLASSES: int = 4

    CLASS_NAMES: list[str] = [
        "glioma",
        "meningioma",
        "notumor",
        "pituitary",
    ]

    # ==========================================================
    # Training Configuration
    # ==========================================================
    BATCH_SIZE: int = 32
    EPOCHS: int = 20

    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-2

    NUM_WORKERS: int = 0
    SEED: int = 42

    # ==========================================================
    # Pydantic Settings
    # ==========================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ==========================================================
    # Utility Methods
    # ==========================================================
    def create_required_directories(self) -> None:
        """
        Create required project directories if they do not exist.
        """

        directories = [
            self.DATA_DIR,
            self.UPLOAD_DIR,
            self.HEATMAP_DIR,
            self.LOGS_DIR,
            self.MODEL_DIR,
        ]

        logger.info("Checking project directories...")

        for directory in directories:

            os.makedirs(directory, exist_ok=True)

            logger.info(f"Ready: {directory}")

        logger.info("All required directories are available.")


# ==========================================================
# Singleton Settings Instance
# ==========================================================
logger.info("Loading application configuration...")

settings = Settings()

settings.create_required_directories()

logger.info(f"Application Name : {settings.APP_NAME}")
logger.info(f"Environment      : {settings.ENVIRONMENT}")
logger.info("Configuration loaded successfully.")