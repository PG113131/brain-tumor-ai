import sys
import argparse
from PIL import Image

from src.config import settings
from src.utils.logger import logger
from src.vision.predictor import Predictor


def main():
    parser = argparse.ArgumentParser(description="Predict brain tumor class for a single MRI image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the brain MRI image file")
    args = parser.parse_args()

    try:
        pil_image = Image.open(args.image).convert("RGB")
    except Exception as e:
        logger.error(f"Could not load image at path '{args.image}': {e}")
        sys.exit(1)

    predictor = Predictor()
    (
        predicted_class,
        confidence,
        class_probs,
        heatmap,
        overlay,
        region,
    ) = predictor.predict_pil(pil_image)

    print("\n" + "=" * 50)
    print("      BRAIN TUMOR MRI PREDICTION RESULTS       ")
    print("=" * 50)
    print(f"File Path        : {args.image}")
    print(f"Predicted Class  : {predicted_class.upper()}")
    print(f"Confidence Score : {confidence * 100:.2f}%\n")
    print("Class Probabilities Breakdown:")
    for cls, prob in class_probs.items():
        print(f"  - {cls:<12}: {prob * 100:.2f}%")
    print("=" * 50 + "\n")
    print(f"Activated Region : {region}")
    print("Grad-CAM saved to outputs/gradcam_overlay.png")

if __name__ == "__main__":
    main()