import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.config import settings
from src.utils.logger import logger
from src.vision.dataset import BrainTumorDataset
from src.vision.transforms import (
    get_train_transforms,
    get_inference_transforms,
)
from src.vision.model import build_model
from src.vision.trainer import Trainer


def main():
    try:
        logger.info("=" * 60)
        logger.info("Brain Tumor AI - Training Pipeline Started")
        logger.info("=" * 60)

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------
        torch.manual_seed(42)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training Device : {device}")

        # ----------------------------------------------------
        # Dataset Path
        # ----------------------------------------------------
        train_dir = os.path.join(
            settings.BASE_DIR,
            "data",
            "dataset",
            "Training"
        )

        logger.info(f"Dataset Path : {train_dir}")

        if not os.path.exists(train_dir):
            logger.error(
                f"Training directory not found at '{train_dir}'.\n"
                "Please download the Kaggle Brain Tumor MRI dataset."
            )
            return

        # ----------------------------------------------------
        # Dataset Loading
        # ----------------------------------------------------
        logger.info("Loading training dataset...")

        full_dataset = BrainTumorDataset(
            data_dir=train_dir,
            transform=get_train_transforms()
        )

        logger.info("Dataset loaded successfully.")
        logger.info(f"Total Images : {len(full_dataset)}")

        # Optional (works only if dataset exposes class names)
        if hasattr(full_dataset, "classes"):
            logger.info(f"Classes : {full_dataset.classes}")

        # ----------------------------------------------------
        # Train / Validation Split
        # ----------------------------------------------------
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size

        train_set, val_set = random_split(
            full_dataset,
            [train_size, val_size]
        )

        logger.info(
            f"Training Samples   : {train_size}"
        )

        logger.info(
            f"Validation Samples : {val_size}"
        )

        # ----------------------------------------------------
        # DataLoaders
        # ----------------------------------------------------
        train_loader = DataLoader(
            train_set,
            batch_size=settings.BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_set,
            batch_size=settings.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        logger.info("DataLoaders created successfully.")

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------
        logger.info("Building EfficientNet model...")

        model = build_model(
            num_classes=settings.NUM_CLASSES,
            pretrained=True
        )

        logger.info("Model initialized successfully.")

        # ----------------------------------------------------
        # Loss Function
        # ----------------------------------------------------
        criterion = nn.CrossEntropyLoss()

        logger.info("Loss Function : CrossEntropyLoss")

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings.LEARNING_RATE,
            weight_decay=settings.WEIGHT_DECAY
        )

        logger.info("Optimizer : AdamW")

        # ----------------------------------------------------
        # Save Path
        # ----------------------------------------------------
        save_path = str(settings.MODEL_PATH)

        logger.info(f"Best model will be saved to:\n{save_path}")

        # ----------------------------------------------------
        # Training Configuration
        # ----------------------------------------------------
        logger.info("=" * 60)
        logger.info("Training Configuration")
        logger.info(f"Epochs        : {settings.EPOCHS}")
        logger.info(f"Batch Size    : {settings.BATCH_SIZE}")
        logger.info(f"Learning Rate : {settings.LEARNING_RATE}")
        logger.info(f"Weight Decay  : {settings.WEIGHT_DECAY}")
        logger.info("=" * 60)

        # ----------------------------------------------------
        # Trainer
        # ----------------------------------------------------
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            save_path=save_path,
        )

        # ----------------------------------------------------
        # Start Training
        # ----------------------------------------------------
        logger.info("Starting model training...")

        trainer.run(epochs=settings.EPOCHS)

        logger.info("Training completed successfully.")

        logger.info("=" * 60)
        logger.info("Brain Tumor AI Training Finished")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")

    except Exception as e:
        logger.exception(f"Unexpected error during training: {e}")


if __name__ == "__main__":
    main()