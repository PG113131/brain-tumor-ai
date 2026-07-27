from pathlib import Path
import urllib.request

MODEL_URL = "https://github.com/PG113131/<YOUR_REPO>/releases/download/v1.0/best_model.pth"

MODEL_PATH = Path("models/best_model.pth")


def download_weights():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        print("Model already exists.")
        return

    print("Downloading model weights...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


if __name__ == "__main__":
    download_weights()