import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from src.vision.gradcam import GradCAM
from src.config import settings
from src.utils.logger import logger
from src.vision.model import load_trained_model
from src.vision.transforms import get_inference_transforms


class Predictor:
    """
    Performs single-image inference using the trained
    EfficientNet Brain Tumor Classification model.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
    ) -> None:

        logger.info("=" * 60)
        logger.info("Initializing Predictor")
        logger.info("=" * 60)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info(f"Inference Device : {self.device}")

        self.model_path = model_path or str(settings.MODEL_PATH)

        logger.info(f"Model Path : {self.model_path}")

        self.transform = get_inference_transforms()

        self.model = None
        self.gradcam = None

    def load(self):

        if self.model is None:

            logger.info("Loading trained model...")

            self.model = load_trained_model(
                self.model_path,
                self.device,
            )

            self.gradcam = GradCAM(
                model=self.model,
                target_layer=self.model.features[-2],
            )

            logger.info("Model loaded successfully.")

    def predict_pil(
        self,
        image: Image.Image,
    ) -> Tuple[
        str,
        float,
        Dict[str, float],
        torch.Tensor,
    ]:
        """
        Predict a brain tumor class from a PIL image.

        Returns:
            predicted_class
            confidence
            class_probabilities
            input_tensor
        """

        if image is None:
            raise ValueError("Input image cannot be None.")

        if not isinstance(image, Image.Image):
            raise TypeError(
                "Input must be a PIL.Image.Image object."
            )

        self.load()

        logger.info("Starting prediction...")

        start_time = time.perf_counter()

        try:

            input_tensor = (
                self.transform(image)
                .unsqueeze(0)
                .to(self.device)
            )

            # ---------- Prediction ----------
            with torch.no_grad():

                outputs = self.model(input_tensor)

                probabilities = F.softmax(
                    outputs,
                    dim=1,
                )[0]

                confidence, predicted_index = torch.max(
                    probabilities,
                    dim=0,
                )

            predicted_class = settings.CLASS_NAMES[
                predicted_index.item()
            ]

            confidence = float(confidence.item())

            # ---------- Grad-CAM ----------
            target_layer = self.model.features[-2]

            gradcam = GradCAM(
                self.model,
                target_layer,
            )

            heatmap, region = self.gradcam.generate(
                input_tensor,
                predicted_index.item(),
            )

            overlay = self.gradcam.overlay_heatmap(
                image,
                heatmap,
            )


            class_probabilities = {
                class_name: float(probabilities[idx].item())
                for idx, class_name in enumerate(
                    settings.CLASS_NAMES
                )
            }

            inference_time = (
                time.perf_counter() - start_time
            )

            logger.info("=" * 60)
            logger.info("Prediction Completed")
            logger.info("=" * 60)

            logger.info(
                f"Predicted Class : {predicted_class}"
            )

            logger.info(
                f"Confidence      : {confidence:.4f}"
            )

            logger.info(
                f"Inference Time  : {inference_time:.4f} sec"
            )

            logger.info("Class Probabilities:")

            for class_name, probability in class_probabilities.items():
                logger.info(
                    f"{class_name:<15}: {probability:.4f}"
                )

            return (
                predicted_class,
                confidence,
                class_probabilities,
                overlay,
                region,
            )

        except Exception as e:

            logger.exception(
                "Prediction failed."
            )

            raise RuntimeError(
                "Unable to perform prediction."
            ) from e
    