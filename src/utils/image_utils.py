import io
import os
from os import PathLike
from typing import Union

import cv2
import numpy as np
from PIL import Image

from src.utils.logger import logger


def load_image_as_pil(
    image_source: Union[str, PathLike, bytes]
) -> Image.Image:
    """
    Load an image from a file path or bytes and convert it to RGB.
    """

    logger.info("Loading image...")

    try:
        if isinstance(image_source, (str, PathLike)):
            image = Image.open(image_source)

        elif isinstance(image_source, bytes):
            image = Image.open(io.BytesIO(image_source))

        else:
            raise ValueError("Unsupported image source type.")

        image = image.convert("RGB")

        logger.info(
            f"Image loaded successfully. Size: {image.size}"
        )

        return image

    except Exception as e:
        logger.exception("Failed to load image.")
        raise ValueError(f"Unable to load image: {e}") from e


def overlay_heatmap_on_image(
    original_pil: Image.Image,
    heatmap_np: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """
    Overlay a Grad-CAM heatmap onto the original image.

    Args:
        original_pil: Original PIL image.
        heatmap_np: 2D heatmap normalized between 0 and 1.
        alpha: Heatmap transparency.

    Returns:
        PIL.Image
    """

    logger.info("Generating Grad-CAM overlay...")

    if not isinstance(heatmap_np, np.ndarray):
        raise TypeError("heatmap_np must be a NumPy array.")

    if heatmap_np.ndim != 2:
        raise ValueError("Heatmap must be a 2D array.")

    alpha = float(np.clip(alpha, 0.0, 1.0))

    heatmap_np = np.clip(heatmap_np, 0, 1)

    original_np = np.array(original_pil)

    height, width = original_np.shape[:2]

    heatmap_uint8 = np.uint8(255 * heatmap_np)

    heatmap_resized = cv2.resize(
        heatmap_uint8,
        (width, height),
    )

    color_heatmap = cv2.applyColorMap(
        heatmap_resized,
        cv2.COLORMAP_JET,
    )

    color_heatmap = cv2.cvtColor(
        color_heatmap,
        cv2.COLOR_BGR2RGB,
    )

    blended = cv2.addWeighted(
        original_np,
        1 - alpha,
        color_heatmap,
        alpha,
        0,
    )

    logger.info("Grad-CAM overlay generated successfully.")

    return Image.fromarray(blended)


def save_pil_image(
    image: Image.Image,
    output_path: Union[str, PathLike],
) -> str:
    """
    Save a PIL image to disk.

    Args:
        image: PIL Image.
        output_path: Destination path.

    Returns:
        Saved image path.
    """

    logger.info(f"Saving image to: {output_path}")

    os.makedirs(
        os.path.dirname(str(output_path)),
        exist_ok=True,
    )

    image.save(output_path, quality=95)

    logger.info("Image saved successfully.")

    return str(output_path)