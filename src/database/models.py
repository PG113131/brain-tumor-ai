import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from src.database.connection import Base


class Patient(Base):
    __tablename__ = "patients"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_code = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    age = Column(String, nullable=True)
    gender = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="patient", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, ForeignKey("patients.id"), nullable=True)
    
    image_path = Column(String, nullable=False)
    heatmap_path = Column(String, nullable=True)
    
    predicted_class = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    class_probabilities = Column(JSON, nullable=False)  # {"glioma": 0.94, ...}
    
    gradcam_region = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="predictions")
    report = relationship("Report", back_populates="prediction", uselist=False, cascade="all, delete-orphan")


class Report(Base):
    __tablename__ = "reports"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    prediction_id = Column(String, ForeignKey("predictions.id"), nullable=False, unique=True)
    
    impression = Column(Text, nullable=False)
    key_findings = Column(JSON, nullable=False)  # List of strings
    certainty_analysis = Column(Text, nullable=False)
    recommendations = Column(JSON, nullable=False)  # List of strings
    
    created_at = Column(DateTime, default=datetime.utcnow)

    prediction = relationship("Prediction", back_populates="report")