import os
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

from src.config import settings
from src.utils.logger import logger


class BrainTumorDataset(Dataset):
    """
    Custom PyTorch Dataset for Brain Tumor MRI Classification.

    Expected folder structure:

    dataset/
        glioma/
        meningioma/
        notumor/
        pituitary/
    """

    VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:

        logger.info("Initializing BrainTumorDataset...")

        self.data_dir = data_dir
        self.transform = transform

        self.class_to_idx = class_to_idx or {
            class_name: idx
            for idx, class_name in enumerate(settings.CLASS_NAMES)
        }

        self.image_paths: List[str] = []
        self.labels: List[int] = []

        self._load_dataset()

        logger.info(
            f"Dataset initialization completed with {len(self.image_paths)} images."
        )

    def _load_dataset(self) -> None:
        """
        Load all images and labels from dataset directory.
        """

        logger.info(f"Loading dataset from: {self.data_dir}")

        if not os.path.exists(self.data_dir):
            logger.error(f"Dataset directory does not exist: {self.data_dir}")
            raise FileNotFoundError(
                f"Dataset directory '{self.data_dir}' not found."
            )

        total_images = 0

        for class_name, class_idx in self.class_to_idx.items():

            class_folder = os.path.join(self.data_dir, class_name)

            if not os.path.isdir(class_folder):
                logger.warning(
                    f"Class folder not found: {class_folder}"
                )
                continue

            class_count = 0

            for filename in sorted(os.listdir(class_folder)):

                if filename.startswith("."):
                    continue

                if not filename.lower().endswith(self.VALID_EXTENSIONS):
                    continue

                image_path = os.path.join(class_folder, filename)

                self.image_paths.append(image_path)
                self.labels.append(class_idx)

                class_count += 1
                total_images += 1

            logger.info(
                f"Loaded {class_count} images from class '{class_name}'."
            )

        if total_images == 0:
            logger.error("No images were found in the dataset.")
            raise RuntimeError(
                f"No valid images found in '{self.data_dir}'."
            )

        logger.info(
            f"Successfully loaded {total_images} images "
            f"from {len(self.class_to_idx)} classes."
        )

    def __len__(self) -> int:
        """
        Returns total number of dataset samples.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[object, int]:
        """
        Returns:
            image: Transformed image tensor (or PIL image if no transform)
            label: Integer class label
        """

        image_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(image_path).convert("RGB")

        except Exception as exc:
            logger.exception(
                f"Failed to load image: {image_path}"
            )
            raise RuntimeError(
                f"Unable to load image '{image_path}'."
            ) from exc

        if self.transform is not None:
            image = self.transform(image)

        return image, label