"""Pydantic schemas for API validation and serialization."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class AppContextSchema(BaseModel):
    """Application context for a model."""
    name: str
    exposure: float
    metric_name: str
    metric_value: float

    class Config:
        from_attributes = True


class ModelMetricSchema(BaseModel):
    """Model metric data point."""
    timestamp: datetime
    health_score: float
    failure_risk: str
    failure_time_predicted: Optional[int] = None
    ece_25: float
    ece_50: float
    ece_75: float
    calibration_error: float
    rmse: float
    r_square: float
    accuracy: float
    brier_score: Optional[float] = None
    grad_ece_correlation: float
    gradient_norm: Optional[float] = None
    max_gradient: Optional[float] = None
    ece_flat: bool
    calibration_drift: Optional[float] = None

    class Config:
        from_attributes = True


class AlertSchema(BaseModel):
    """Alert event."""
    id: str
    timestamp: datetime
    message: str
    severity: str
    resolved: bool = False

    class Config:
        from_attributes = True


class PredictionSchema(BaseModel):
    """Individual prediction."""
    timestamp: datetime
    prediction: float
    target: float
    confidence: float
    is_correct: bool
    loss: Optional[float] = None

    class Config:
        from_attributes = True


class ModelDetailSchema(BaseModel):
    """Complete model data - matches frontend expectations."""
    id: str
    name: str
    architecture: str
    health_score: float
    failure_risk: str
    failure_time_predicted: Optional[int] = None
    ece_25: float
    ece_50: float
    ece_75: float
    calibration_error: float
    rmse: float
    r_square: float
    accuracy: float
    grad_ece_correlation: float
    ece_flat: bool
    insights: List[str] = []
    app_context: AppContextSchema
    updated_at: datetime

    class Config:
        from_attributes = True


class ModelListSchema(BaseModel):
    """Model for list responses."""
    id: str
    name: str
    architecture: str
    health_score: float
    failure_risk: str
    updated_at: datetime

    class Config:
        from_attributes = True


class ChartDataPointSchema(BaseModel):
    """Chart data point for frontend."""
    time: str  # "T-{timesteps_ago}"
    calibration_drift: float
    gradient_spike: float
    ece_base: float

    class Config:
        from_attributes = True


class ModelCreateSchema(BaseModel):
    """Create a new model."""
    id: str
    name: str
    architecture: str
    checkpoint_path: Optional[str] = None
    app_context: AppContextSchema


class MetricUpdateSchema(BaseModel):
    """Update model metrics."""
    health_score: float
    failure_risk: str
    failure_time_predicted: Optional[int] = None
    ece_25: float
    ece_50: float
    ece_75: float
    calibration_error: float
    rmse: float
    r_square: float
    accuracy: float
    grad_ece_correlation: float
    ece_flat: bool
    brier_score: Optional[float] = None
    gradient_norm: Optional[float] = None
    max_gradient: Optional[float] = None
    calibration_drift: Optional[float] = None


class AlertCreateSchema(BaseModel):
    """Create an alert."""
    model_id: str
    message: str
    severity: str


class WebSocketMessageSchema(BaseModel):
    """WebSocket message format."""
    type: str  # "metrics_update", "alert", "chart_update"
    data: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DashboardStateSchema(BaseModel):
    """Complete dashboard state for initial load."""
    models: List[ModelDetailSchema]
    alerts: List[AlertSchema]
    chart_data: List[ChartDataPointSchema]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
