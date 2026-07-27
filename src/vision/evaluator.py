import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from src.config import settings
from src.utils.logger import logger


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    save_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate the trained model on the test dataset.

    Args:
        model: Trained PyTorch model.
        test_loader: Test DataLoader.
        device: CPU or CUDA device.
        save_report_path: Optional JSON file path to save evaluation report.

    Returns:
        Dictionary containing evaluation metrics.
    """

    logger.info("=" * 60)
    logger.info("Starting Model Evaluation...")
    logger.info("=" * 60)

    if len(test_loader.dataset) == 0:
        logger.error("Test dataset is empty.")
        raise ValueError("Test dataset contains no samples.")

    model.eval()

    y_true = []
    y_pred = []

    try:

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                predictions = torch.argmax(outputs, dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())

        logger.info("Prediction completed successfully.")

        accuracy = accuracy_score(y_true, y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        report = classification_report(
            y_true,
            y_pred,
            target_names=settings.CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        results = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "report": report,
        }

        logger.info(f"Accuracy : {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall   : {recall:.4f}")
        logger.info(f"F1 Score : {f1:.4f}")

        if save_report_path:

            report_path = Path(save_report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_path, "w", encoding="utf-8") as file:
                json.dump(results, file, indent=4)

            logger.info(
                f"Evaluation report saved to: {report_path}"
            )

        logger.info("Model evaluation completed successfully.")

        return results

    except Exception as e:
        logger.exception("Error occurred during model evaluation.")
        raise RuntimeError(
            "Failed to evaluate the model."
        ) from e