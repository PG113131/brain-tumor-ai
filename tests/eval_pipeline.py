import os
import sys
import argparse
from pathlib import Path
import torch
from PIL import Image

# Ensure project root is in Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.vision.predictor import Predictor
from src.utils.logger import logger


def evaluate_directory(image_dir: str):
    """
    Scans a directory of test MRI images and prints batch predictions.
    """
    target_path = Path(image_dir)
    if not target_path.exists() or not target_path.is_dir():
        logger.error(f"Provided path does not exist or is not a directory: {image_dir}")
        return

    logger.info(f"Loading vision model from settings: {settings.MODEL_WEIGHTS_PATH}")
    predictor = Predictor()

    supported_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = [f for f in target_path.iterdir() if f.suffix.lower() in supported_extensions]

    if not image_files:
        logger.warning(f"No image files found in {image_dir}")
        return

    logger.info(f"Evaluating {len(image_files)} MRI scans...")
    print("\n" + "=" * 85)
    print(f"{'Filename':<35} | {'Predicted Class':<18} | {'Confidence':<12}")
    print("=" * 85)

    for img_path in image_files:
        try:
            pil_img = Image.open(img_path).convert("RGB")
            pred_class, confidence, _, _ = predictor.predict_pil(pil_img)
            print(f"{img_path.name:<35} | {pred_class:<18} | {confidence * 100:.2f}%")
        except Exception as e:
            logger.error(f"Failed processing image {img_path.name}: {e}")

    print("=" * 85 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Brain MRI Vision Pipeline")
    parser.add_argument(
        "--dir",
        type=str,
        default="./data/test_samples",
        help="Path to directory containing sample MRI images."
    )
    args = parser.parse_args()
    
    evaluate_directory(args.dir)