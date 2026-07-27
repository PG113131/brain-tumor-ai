from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class PatientInfo(BaseModel):
    patient_code: str = Field(..., example="PAT-10203")
    name: str = Field(..., example="Rahul Kumar")
    age: Optional[str] = Field(None, example="54")
    gender: Optional[str] = Field(None, example="Male")

class PredictionSummary(BaseModel):
    id: str
    image_url: str
    heatmap_url: Optional[str]
    predicted_class: str
    confidence_score: float
    class_probabilities: Dict[str, float]
    gradcam_region: Optional[str]
    created_at: datetime


class ReportSummary(BaseModel):
    id: str
    impression: str
    key_findings: List[str]
    certainty_analysis: str
    recommendations: List[str]
    created_at: datetime


class DiagnosticResponse(BaseModel):
    patient: PatientInfo
    prediction: PredictionSummary
    report: ReportSummary
class HistoryPatient(BaseModel):
    patient_code: str
    name: str
    age: str | None = None
    gender: str | None = None

class HistoryPrediction(BaseModel):
    id: str
    predicted_class: str
    confidence_score: float
    class_probabilities: dict
    gradcam_region: str | None = None
    image_url: str
    heatmap_url: str
    created_at: datetime


class HistoryReport(BaseModel):
    impression: str
    key_findings: List[str]
    certainty_analysis: str
    recommendations: List[str]


class HistoryItem(BaseModel):
    patient: HistoryPatient
    prediction: HistoryPrediction
    report: HistoryReport