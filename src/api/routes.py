import os
import uuid
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from sqlalchemy.orm import Session,joinedload
from typing import Optional

from src.config import settings
from src.database.connection import get_db
from src.database.models import Patient, Prediction, Report
from src.utils.image_utils import load_image_as_pil, overlay_heatmap_on_image, save_pil_image
from src.utils.logger import logger

from src.vision.predictor import Predictor
from src.vision.gradcam import GradCAM
from src.llm.report_generator import ReportGenerator
from src.api.schemas import HistoryItem, HistoryPatient, HistoryPrediction, HistoryReport,DiagnosticResponse, PatientInfo, PredictionSummary, ReportSummary

router = APIRouter(prefix="/api/v1", tags=["Diagnostic Suite"])

# Global singletons for model and LLM loaded at runtime
predictor_instance = Predictor()
report_gen_instance = ReportGenerator()


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy", "app": settings.APP_NAME}


@router.post("/predict", response_model=DiagnosticResponse, status_code=status.HTTP_201_CREATED)
async def analyze_mri_scan(
    file: UploadFile = File(...),
    patient_code: str = Form(...),
    name: str = Form(...),
    age: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Primary endpoint: Takes an uploaded MRI scan image, runs deep learning classification,
    generates a Grad-CAM explainability heatmap, generates an LLM report, and saves to DB.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a valid image."
        )

    try:
        # 1. Read & process raw uploaded image
        contents = await file.read()
        pil_image = load_image_as_pil(contents)

        filename = f"{uuid.uuid4().hex}_{file.filename}"
        upload_path = os.path.join(settings.UPLOAD_DIR, filename)
        save_pil_image(pil_image, upload_path)

        # 2. Run PyTorch Classifier Inference
        (
            predicted_class,
            confidence,
            class_probs,
            overlay_image,
            region_label,
        ) = predictor_instance.predict_pil(
            pil_image
        )

        heatmap_filename = f"gradcam_{filename}"

        heatmap_path = os.path.join(
            settings.HEATMAP_DIR,
            heatmap_filename,
        )

        save_pil_image(
            overlay_image,
            heatmap_path,
        )

        # 4. Invoke LLM Radiology Report Generator
        patient_data = {"patient_code": patient_code, "age": age, "gender": gender}
        prediction_data = {
            "predicted_class": predicted_class,
            "confidence_score": confidence,
            "class_probabilities": class_probs,
            "gradcam_region": region_label
        }
        
        llm_report = report_gen_instance.generate_report(patient_data, prediction_data)

        # 5. Persist into Database
        patient = db.query(Patient).filter(
            Patient.patient_code == patient_code
        ).first()

        if not patient:
            patient = Patient(
                patient_code=patient_code,
                name=name,
                age=age,
                gender=gender,
            )
            db.add(patient)
            db.flush()
        else:
            # Keep patient information up to date
            patient.name = name
            patient.age = age
            patient.gender = gender

        prediction_record = Prediction(
            patient_id=patient.id,
            image_path=upload_path,
            heatmap_path=heatmap_path,
            predicted_class=predicted_class,
            confidence_score=confidence,
            class_probabilities=class_probs,
            gradcam_region=region_label
        )
        db.add(prediction_record)
        db.flush()

        report_record = Report(
            prediction_id=prediction_record.id,
            impression=llm_report.impression,
            key_findings=llm_report.key_findings,
            certainty_analysis=llm_report.certainty_analysis,
            recommendations=llm_report.recommendations
        )
        db.add(report_record)
        db.commit()

        # 6. Build and return structured API response
        return DiagnosticResponse(
            patient=PatientInfo(
                patient_code=patient_code,
                name=name,
                age=age,
                gender=gender,
            ),
            prediction=PredictionSummary(
                id=prediction_record.id,
                image_url=f"/static/uploads/{filename}",
                heatmap_url=f"/static/heatmaps/{heatmap_filename}",
                predicted_class=predicted_class,
                confidence_score=confidence,
                class_probabilities=class_probs,
                gradcam_region=region_label,
                created_at=prediction_record.created_at
            ),
            report=ReportSummary(
                id=report_record.id,
                impression=report_record.impression,
                key_findings=report_record.key_findings,
                certainty_analysis=report_record.certainty_analysis,
                recommendations=report_record.recommendations,
                created_at=report_record.created_at
            )
        )
    except Exception as e:

        db.rollback()

        logger.exception(
            "Diagnostic pipeline failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Diagnostic pipeline failed.",
        )

@router.get(
    "/history",
    response_model=list[HistoryItem],
    status_code=status.HTTP_200_OK,
)
def get_prediction_history(db: Session = Depends(get_db)):
    """
    Returns all prediction history ordered by newest first.
    """

    predictions = (
        db.query(Prediction)
        .options(
            joinedload(Prediction.patient),
            joinedload(Prediction.report),
        )
        .order_by(Prediction.created_at.desc())
        .all()
    )

    history = []

    for pred in predictions:

        history.append(
            HistoryItem(
                patient=HistoryPatient(
                    patient_code=pred.patient.patient_code,
                    name=pred.patient.name,
                    age=pred.patient.age,
                    gender=pred.patient.gender,
                ),
                prediction=HistoryPrediction(
                    id=pred.id,
                    predicted_class=pred.predicted_class,
                    confidence_score=pred.confidence_score,
                    class_probabilities=pred.class_probabilities,
                    gradcam_region=pred.gradcam_region,
                    image_url=f"/static/uploads/{os.path.basename(pred.image_path)}",
                    heatmap_url=f"/static/heatmaps/{os.path.basename(pred.heatmap_path)}",
                    created_at=pred.created_at,
                ),
                report=HistoryReport(
                    impression=pred.report.impression,
                    key_findings=pred.report.key_findings,
                    certainty_analysis=pred.report.certainty_analysis,
                    recommendations=pred.report.recommendations,
                ),
            )
        )

    return history