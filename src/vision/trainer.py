import time
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils.logger import logger

class Trainer:
    """Handles model training, validation, checkpointing, and early stopping."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        save_path: str,
        early_stopping_patience: int = 5,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_path = Path(save_path)
        self.best_val_loss = float("inf")
        self.patience = early_stopping_patience
        self.no_improvement_epochs = 0
        self.lr_scheduler = lr_scheduler
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Trainer initialized successfully.")

    def _run_epoch(self, train: bool) -> Dict[str, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0
        loader = self.train_loader if train else self.val_loader

        # Mixed precision for faster training on modern GPUs
        use_amp = self.device.type == "cuda"
        
        with torch.set_grad_enabled(train):
            for batch_idx, (images, labels) in enumerate(loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if train:
                    self.optimizer.zero_grad(set_to_none=True) # Faster than zero_grad()
                
                # Forward pass with optional autocast
                with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                # Metrics calculation
                running_loss += loss.item() * images.size(0)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                if batch_idx % 10 == 0:
                    logger.info(
                        f"{'Train' if train else 'Validation'} "
                        f"Batch {batch_idx + 1}/{len(loader)} | "
                        f"Loss: {loss.item():.4f}"
                    )

        # FIXED INDENTATION: Calculate metrics after the loop finishes completely
        epoch_loss = running_loss / total
        epoch_accuracy = correct / total
        
        return {
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
        }

    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: float) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        if self.lr_scheduler:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
            
        torch.save(checkpoint, self.save_path)
        logger.info(f"Best model checkpoint saved to:\n{self.save_path}")

    def run(self, epochs: int) -> None:
        logger.info("=" * 70)
        logger.info("Starting Training")
        logger.info("=" * 70)
        
        try:
            for epoch in range(1, epochs + 1):
                start_time = time.perf_counter()
                
                train_metrics = self._run_epoch(train=True)
                val_metrics = self._run_epoch(train=False)
                
                # Step the scheduler if present
                if self.lr_scheduler:
                    # Assumes a scheduler that steps per epoch (e.g., StepLR, MultiStepLR)
                    # If using ReduceLROnPlateau, use: self.lr_scheduler.step(val_metrics["loss"])
                    self.lr_scheduler.step()
                
                epoch_time = time.perf_counter() - start_time
                current_lr = self.optimizer.param_groups[0]["lr"]

                logger.info("-" * 70)
                logger.info(f"Epoch {epoch}/{epochs}")
                logger.info("-" * 70)
                logger.info(f"Train Loss     : {train_metrics['loss']:.4f}")
                logger.info(f"Train Accuracy : {train_metrics['accuracy']:.4f}")
                logger.info(f"Val Loss       : {val_metrics['loss']:.4f}")
                logger.info(f"Val Accuracy   : {val_metrics['accuracy']:.4f}")
                logger.info(f"Learning Rate  : {current_lr:.8f}")
                logger.info(f"Epoch Time     : {epoch_time:.2f} sec")

                # Early stopping check
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.no_improvement_epochs = 0
                    self._save_checkpoint(epoch, train_metrics["loss"], val_metrics["loss"])
                    logger.info("Validation loss improved.")
                else:
                    self.no_improvement_epochs += 1
                    logger.info(f"No improvement ({self.no_improvement_epochs}/{self.patience})")

                if self.no_improvement_epochs >= self.patience:
                    logger.info("Early stopping triggered.")
                    break

            logger.info("=" * 70)
            logger.info("Training Completed Successfully")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.exception("Training failed.")
            raise RuntimeError("Model training failed.") from e
