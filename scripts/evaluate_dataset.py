import os
import torch
from torch.utils.data import DataLoader

from src.config import settings
from src.utils.logger import logger
from src.vision.dataset import BrainTumorDataset
from src.vision.transforms import get_inference_transforms
from src.vision.model import load_trained_model
from src.vision.evaluator import evaluate_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device for evaluation: {device}")

    test_dir = os.path.join(settings.BASE_DIR, "data", "dataset", "Testing")
    
    if not os.path.exists(test_dir):
        logger.error(f"Testing directory not found at '{test_dir}'.")
        return

    # Load Test Set
    test_dataset = BrainTumorDataset(data_dir=test_dir, transform=get_inference_transforms())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Load Model
    model_path = str(settings.MODEL_PATH)
    model = load_trained_model(model_path, device)

    # Evaluate
    results = evaluate_model(model, test_loader, device)
    
    print("\n" + "=" * 50)
    print(f"Overall Accuracy: {results['accuracy'] * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()