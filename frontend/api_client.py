import requests
from typing import Dict, Any, Optional

FASTAPI_BASE_URL = "http://localhost:8000"


class APIClient:
    """Helper client to handle HTTP requests between Streamlit and FastAPI."""

    def __init__(self, base_url: str = FASTAPI_BASE_URL) -> None:
        self.base_url = base_url
        self.session = requests.Session()

    def check_health(self) -> bool:
        """Verifies backend service availability."""
        try:
            response = requests.get(f"{self.base_url}/api/v1/health", timeout=3)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def predict_mri(
        self,
        image_bytes: bytes,
        filename: str,
        patient_code: str,
        name:str,
        age: Optional[str] = None,
        gender: Optional[str] = None
    ) -> Dict[str, Any]:
        """Sends the MRI image and metadata to the FastAPI backend."""
        url = f"{self.base_url}/api/v1/predict"
        
        files = {"file": (filename, image_bytes, "image/jpeg")}
        data = {
            "patient_code": patient_code,
            "name": name,
            "age": age or "",
            "gender": gender or ""
        }

        response = requests.post(url, files=files, data=data, timeout=30)
        
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"API Error ({response.status_code}): {response.text}")
    def get_history(self):
        response = requests.get(
            f"{self.base_url}/api/v1/history",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    def get_patient_history(self, patient_code):
        response = requests.get(
            f"{self.base_url}/api/v1/history/{patient_code}",
            timeout=40
        )
        response.raise_for_status()
        return response.json()
    def get_prediction(self, prediction_id):
        response = requests.get(
            f"{self.base_url}/api/v1/prediction/{prediction_id}",
            timeout=40
        )
        response.raise_for_status()
        return response.json()
