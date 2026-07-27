import sys
import argparse
from pathlib import Path
from PIL import Image

# Ensure project root is in Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.logger import logger
from src.vision.predictor import Predictor
from scripts.verify_weights import verify_weights


def predict_directory(image_dir: str):
    """
    Predict all MRI images in a directory.
    """

    if not verify_weights():
        return

    target_dir = Path(image_dir)

    if not target_dir.exists():
        logger.error(f"Directory not found: {target_dir}")
        return

    if not target_dir.is_dir():
        logger.error(f"Not a directory: {target_dir}")
        return

    predictor = Predictor()

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    image_files = sorted(
        [
            f
            for f in target_dir.iterdir()
            if f.suffix.lower() in supported_extensions
        ]
    )

    if not image_files:
        logger.warning("No supported images found.")
        return

    print("\n")
    print("=" * 90)
    print(f"{'Filename':35} {'Prediction':18} {'Confidence':12}")
    print("=" * 90)

    success = 0

    for image_path in image_files:

        try:

            image = Image.open(image_path).convert("RGB")

            (
                predicted_class,
                confidence,
                _,
                _,
                _,
                _,
            ) = predictor.predict_pil(image)

            print(
                f"{image_path.name:35} "
                f"{predicted_class.upper():18} "
                f"{confidence*100:>7.2f}%"
            )

            success += 1

        except Exception as e:
            logger.error(f"{image_path.name}: {e}")

    print("=" * 90)
    print(f"Processed {success}/{len(image_files)} images successfully.")
    print("=" * 90)


def main():

    parser = argparse.ArgumentParser(
        description="Predict all MRI images in a directory."
    )

    parser.add_argument(
        "--dir",
        type=str,
        default="data/test_samples",
        help="Directory containing MRI images.",
    )

    args = parser.parse_args()

    predict_directory(args.dir)


if __name__ == "__main__":
    main()