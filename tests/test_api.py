import io

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from main import app

# ==========================================================
# Test Client
# ==========================================================
client = TestClient(app)

HEALTH_ENDPOINT = "/api/v1/health"
PREDICT_ENDPOINT = "/api/v1/predict"


# ==========================================================
# Helper Functions
# ==========================================================
def create_dummy_image_bytes() -> bytes:
    """
    Creates a dummy RGB image in memory for API testing.
    """

    print("[TEST] Creating dummy MRI image...")

    image = Image.new("RGB", (224, 224), color=(128, 128, 128))

    buffer = io.BytesIO()

    image.save(buffer, format="JPEG")

    print("[SUCCESS] Dummy image created.")

    return buffer.getvalue()


# ==========================================================
# Health Endpoint Tests
# ==========================================================
def test_health_check():
    """
    Verify that the health endpoint is reachable.
    """

    print("\n[TEST] Health Check API")

    response = client.get(HEALTH_ENDPOINT)

    print(f"[INFO] Status Code: {response.status_code}")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

    print("[SUCCESS] Health endpoint is working.")


# ==========================================================
# Prediction Endpoint Tests
# ==========================================================
def test_predict_endpoint_missing_file():
    """
    Verify that no uploaded file returns HTTP 422.
    """

    print("\n[TEST] Predict API without file")

    response = client.post(PREDICT_ENDPOINT)

    print(f"[INFO] Status Code: {response.status_code}")

    assert response.status_code == 422

    print("[SUCCESS] Missing file validation passed.")


def test_predict_endpoint_invalid_file_type():
    """
    Verify that uploading a non-image file is rejected.
    """

    print("\n[TEST] Predict API with invalid file")

    files = {
        "file": (
            "test.txt",
            b"This is not an image.",
            "text/plain",
        )
    }

    response = client.post(
        PREDICT_ENDPOINT,
        files=files,
    )

    print(f"[INFO] Status Code: {response.status_code}")
    print(f"[INFO] Response: {response.json()}")

    assert response.status_code == 400
    assert "must be a valid image" in response.json()["detail"]

    print("[SUCCESS] Invalid file validation passed.")


# ==========================================================
# Future Test
# ==========================================================
@pytest.mark.skip(reason="Enable after prediction pipeline is implemented.")
def test_predict_endpoint_valid_image():
    """
    Verify successful prediction using a valid MRI image.
    """

    print("\n[TEST] Predict API with valid image")

    files = {
        "file": (
            "brain.jpg",
            create_dummy_image_bytes(),
            "image/jpeg",
        )
    }

    response = client.post(
        PREDICT_ENDPOINT,
        files=files,
    )

    print(f"[INFO] Status Code: {response.status_code}")

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "confidence" in data

    print("[SUCCESS] Prediction API returned a valid response.")