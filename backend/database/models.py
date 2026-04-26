"""Database models for the Calibration Confidence backend."""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, ForeignKey, Enum, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()


class ModelArchitecture(str, enum.Enum):
    """Supported model architectures."""
    MLP = "mlp"
    DEEP_MLP = "deep_mlp"
    LSTM = "lstm"
    VANILLA_RNN = "vanilla_rnn"
    RESIDUAL_MLP = "residual_mlp"


class AlertSeverity(str, enum.Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class FailureRisk(str, enum.Enum):
    """Failure risk levels."""
    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class Model(Base):
    """Model representation in database."""
    __tablename__ = "models"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)
    architecture = Column(String, nullable=False)  # Enum as string
    checkpoint_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    metrics = relationship("ModelMetric", back_populates="model", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="model", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="model", cascade="all, delete-orphan")


class ModelMetric(Base):
    """Time-series metrics for each model."""
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String, ForeignKey("models.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Health & Risk Metrics
    health_score = Column(Float, nullable=False)  # 0-100
    failure_risk = Column(String, nullable=False)  # Enum as string
    failure_time_predicted = Column(Integer, nullable=True)  # Steps until failure
    
    # ECE Metrics (Expected Calibration Error)
    ece_25 = Column(Float, nullable=False)  # 25th percentile
    ece_50 = Column(Float, nullable=False)  # Median
    ece_75 = Column(Float, nullable=False)  # 75th percentile
    calibration_error = Column(Float, nullable=False)  # Overall ECE
    
    # Performance Metrics
    rmse = Column(Float, nullable=False)
    r_square = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=False)
    brier_score = Column(Float, nullable=True)
    
    # Gradient Analysis
    grad_ece_correlation = Column(Float, nullable=False)  # Gradient-ECE correlation
    gradient_norm = Column(Float, nullable=True)
    max_gradient = Column(Float, nullable=True)
    
    # Stability
    ece_flat = Column(Boolean, nullable=False)  # Is ECE stable?
    calibration_drift = Column(Float, nullable=True)  # Drift over time
    
    # Relationships
    model = relationship("Model", back_populates="metrics")


class Alert(Base):
    """Alert events."""
    __tablename__ = "alerts"

    id = Column(String, primary_key=True, index=True)
    model_id = Column(String, ForeignKey("models.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    message = Column(Text, nullable=False)
    severity = Column(String, nullable=False)  # Enum as string
    resolved = Column(Boolean, default=False, nullable=False)
    resolved_at = Column(DateTime, nullable=True)
    
    # Relationships
    model = relationship("Model", back_populates="alerts")


class AppContext(Base):
    """Application context for models - capital exposure, KPIs, etc."""
    __tablename__ = "app_contexts"

    id = Column(String, primary_key=True, index=True)
    model_id = Column(String, ForeignKey("models.id"), nullable=False, unique=True, index=True)
    name = Column(String, nullable=False)  # e.g., "HFT Options Arbitrage"
    exposure = Column(Float, nullable=False)  # Capital at risk in millions
    metric_name = Column(String, nullable=False)  # KPI name: "Trades/sec"
    metric_value = Column(Float, nullable=False)  # Current KPI value
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class Prediction(Base):
    """Individual predictions and confidences."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String, ForeignKey("models.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    prediction = Column(Float, nullable=False)
    target = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)  # 0-1
    is_correct = Column(Boolean, nullable=False)
    loss = Column(Float, nullable=True)
    
    # Relationships
    model = relationship("Model", back_populates="predictions")
