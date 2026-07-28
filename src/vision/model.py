import os

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)

from src.config import settings
from src.utils.logger import logger


def build_model(
    num_classes: int = settings.NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build an EfficientNet-B0 model for Brain Tumor Classification.

    Args:
        num_classes:
            Number of output classes.

        pretrained:
            Load ImageNet pretrained weights.

    Returns:
        Configured EfficientNet model.
    """

    logger.info("=" * 60)
    logger.info("Building EfficientNet-B0 Model")
    logger.info("=" * 60)

    try:

        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        logger.info(
            f"Using pretrained weights: {pretrained}"
        )

        model = efficientnet_b0(weights=weights)

        in_features = model.classifier[1].in_features

        model.classifier = nn.Sequential(
            nn.Dropout(
                p=0.2,
                inplace=True,
            ),
            nn.Linear(
                in_features=in_features,
                out_features=num_classes,
            ),
        )

        total_params = sum(
            p.numel()
            for p in model.parameters()
        )

        trainable_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        logger.info(
            f"Total Parameters     : {total_params:,}"
        )

        logger.info(
            f"Trainable Parameters : {trainable_params:,}"
        )

        logger.info(
            f"Output Classes       : {num_classes}"
        )

        logger.info("Model built successfully.")

        return model

    except Exception as e:

        logger.exception(
            "Failed to build EfficientNet model."
        )

        raise RuntimeError(
            "Unable to build model."
        ) from e


def load_trained_model(
    model_path: str,
    device: torch.device,
) -> nn.Module:
    logger.info("=" * 60)
    logger.info("Loading Trained Model")
    logger.info("=" * 60)

    try:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file does not exist:\n{model_path}")

        logger.info(f"Loading checkpoint from:\n{model_path}")

        model = build_model(pretrained=False)

        checkpoint = torch.load(model_path, map_location=device)
        logger.info(type(checkpoint))
        
        if isinstance(checkpoint, dict):
            logger.info(checkpoint.keys())

        # Extract weight dictionary if loaded from Trainer checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        logger.info(f"Model loaded successfully on {device}.")
        return model

    except Exception as e:
        logger.exception("Failed to load trained model.")
        raise RuntimeError("Unable to load trained model.") from e
