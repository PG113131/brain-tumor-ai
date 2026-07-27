from pathlib import Path
import sys

# Ensure project root is in Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.utils.logger import logger


def verify_weights() -> bool:
    """
    Verify that the trained model weights exist.

    Returns:
        bool: True if weights exist, False otherwise.
    """

    weights_path = Path(settings.MODEL_PATH)

    # Create models directory if it doesn't exist
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    if weights_path.is_file():
        logger.info(f"✅ Model weights found: {weights_path}")
        return True

    logger.error("❌ Model weights not found.")

    logger.info(
        f"""
Download the trained model by running:

    python scripts/download_weights.py

Or manually place the model here:

    {weights_path}
"""
    )

    return False


if __name__ == "__main__":
    verify_weights()