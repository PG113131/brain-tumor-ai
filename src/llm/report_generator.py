from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from src.config import settings
from src.llm.prompts import get_radiology_report_prompt
from src.utils.logger import logger


class RadiologyReportSchema(BaseModel):
    """Pydantic schema enforcing structured LLM outputs."""
    impression: str = Field(
        description="Concise, one-sentence clinical impression summarizing the overall scan finding."
    )
    key_findings: List[str] = Field(
        description="List of specific diagnostic findings, including class probabilities and spatial location."
    )
    certainty_analysis: str = Field(
        description="Analysis of model confidence, potential differential diagnoses, or uncertainty factors."
    )
    recommendations: List[str] = Field(
        description="Actionable next clinical steps, such as contrast-enhanced MRI or neurosurgical consultation."
    )


class ReportGenerator:
    """Executes ChatGroq LLM chains to generate radiology reports."""

    def __init__(self) -> None:
        self.llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=0.2
        )
        self.structured_llm = self.llm.with_structured_output(RadiologyReportSchema)
        self.prompt_template = get_radiology_report_prompt()
        self.chain = self.prompt_template | self.structured_llm

    def generate_report(
        self,
        patient_info: Dict[str, Any],
        prediction_info: Dict[str, Any]
    ) -> RadiologyReportSchema:
        """
        Takes patient and prediction metadata, invokes ChatGroq, and returns a validated report.
        """
        try:
            input_payload = {
                "patient_code": patient_info.get("patient_code", "ANONYMOUS"),
                "patient_name": patient_info.get("name", "Unknown"),
                "age": patient_info.get("age", "Unknown"),
                "gender": patient_info.get("gender", "Unknown"),
                "predicted_class": prediction_info.get("predicted_class", "Unclassified").upper(),
                "confidence": prediction_info.get("confidence_score", 0.0) * 100,
                "class_probabilities": prediction_info.get("class_probabilities", {}),
                "gradcam_region": prediction_info.get("gradcam_region", "Unspecified Region")
            }

            logger.info(f"Invoking ChatGroq for patient {input_payload['patient_code']}...")
            report: RadiologyReportSchema = self.chain.invoke(input_payload)
            logger.info("Successfully generated structured radiology report.")
            return report

        except Exception as e:
            logger.error(f"Error generating LLM radiology report: {e}")
            raise e