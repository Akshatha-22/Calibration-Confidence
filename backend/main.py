"""Main FastAPI application."""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend.database.session import get_db, init_db
from backend.database.models import Model, ModelMetric, Alert, AppContext
from backend.schemas import (
    ModelListSchema, ModelDetailSchema, ModelCreateSchema,
    MetricUpdateSchema, AlertCreateSchema, DashboardStateSchema,
    AlertSchema, AppContextSchema
)
from backend.websocket_manager import manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle."""
    # Startup
    logger.info("Initializing database...")
    init_db()
    logger.info("Database ready!")
    yield
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Calibration Confidence API",
    description="Backend API for confidence calibration monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# MODELS ENDPOINTS
# ============================================================================

@app.get("/api/models", response_model=list[ModelListSchema])
async def list_models(db: Session = Depends(get_db)):
    """List all models with current health status."""
    models = db.query(Model).all()
    
    result = []
    for model in models:
        # Get latest metric
        latest_metric = (
            db.query(ModelMetric)
            .filter(ModelMetric.model_id == model.id)
            .order_by(ModelMetric.timestamp.desc())
            .first()
        )
        
        if latest_metric:
            result.append({
                "id": model.id,
                "name": model.name,
                "architecture": model.architecture,
                "health_score": latest_metric.health_score,
                "failure_risk": latest_metric.failure_risk,
                "updated_at": latest_metric.timestamp
            })
    
    return result


@app.get("/api/models/{model_id}", response_model=ModelDetailSchema)
async def get_model_detail(model_id: str, db: Session = Depends(get_db)):
    """Get detailed information about a specific model."""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Get latest metric
    latest_metric = (
        db.query(ModelMetric)
        .filter(ModelMetric.model_id == model_id)
        .order_by(ModelMetric.timestamp.desc())
        .first()
    )
    
    if not latest_metric:
        raise HTTPException(status_code=404, detail="No metrics found for model")
    
    # Get app context
    app_context = db.query(AppContext).filter(AppContext.model_id == model_id).first()
    if not app_context:
        raise HTTPException(status_code=404, detail="No app context found")
    
    # Generate insights
    insights = []
    if latest_metric.failure_time_predicted:
        insights.append(f"Failure predicted in {latest_metric.failure_time_predicted} timesteps")
    if latest_metric.grad_ece_correlation > 0.7:
        insights.append("High gradient-ECE correlation detected")
    if not latest_metric.ece_flat:
        insights.append("ECE is increasing - calibration degrading")
    if latest_metric.calibration_drift and latest_metric.calibration_drift > 0.1:
        insights.append(f"Calibration drift: {latest_metric.calibration_drift:.4f}")
    
    return {
        "id": model.id,
        "name": model.name,
        "architecture": model.architecture,
        "health_score": latest_metric.health_score,
        "failure_risk": latest_metric.failure_risk,
        "failure_time_predicted": latest_metric.failure_time_predicted,
        "ece_25": latest_metric.ece_25,
        "ece_50": latest_metric.ece_50,
        "ece_75": latest_metric.ece_75,
        "calibration_error": latest_metric.calibration_error,
        "rmse": latest_metric.rmse,
        "r_square": latest_metric.r_square,
        "accuracy": latest_metric.accuracy,
        "grad_ece_correlation": latest_metric.grad_ece_correlation,
        "ece_flat": latest_metric.ece_flat,
        "insights": insights,
        "app_context": {
            "name": app_context.name,
            "exposure": app_context.exposure,
            "metric_name": app_context.metric_name,
            "metric_value": app_context.metric_value
        },
        "updated_at": latest_metric.timestamp
    }


@app.post("/api/models", response_model=ModelDetailSchema)
async def create_model(
    model: ModelCreateSchema,
    db: Session = Depends(get_db)
):
    """Create a new model."""
    # Check if exists
    existing = db.query(Model).filter(Model.id == model.id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Model already exists")
    
    # Create model
    db_model = Model(
        id=model.id,
        name=model.name,
        architecture=model.architecture,
        checkpoint_path=model.checkpoint_path
    )
    db.add(db_model)
    
    # Create app context
    app_ctx = AppContext(
        id=f"{model.id}-ctx",
        model_id=model.id,
        name=model.app_context.name,
        exposure=model.app_context.exposure,
        metric_name=model.app_context.metric_name,
        metric_value=model.app_context.metric_value
    )
    db.add(app_ctx)
    
    db.commit()
    db.refresh(db_model)
    
    return await get_model_detail(model.id, db)


@app.get("/api/models/{model_id}/metrics")
async def get_model_metrics(
    model_id: str,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get historical metrics for a model."""
    metrics = (
        db.query(ModelMetric)
        .filter(ModelMetric.model_id == model_id)
        .order_by(ModelMetric.timestamp.desc())
        .limit(limit)
        .all()
    )
    
    return [
        {
            "timestamp": m.timestamp,
            "health_score": m.health_score,
            "calibration_error": m.calibration_error,
            "rmse": m.rmse,
            "ece_25": m.ece_25,
            "ece_75": m.ece_75,
            "grad_ece_correlation": m.grad_ece_correlation,
        }
        for m in reversed(metrics)
    ]


# ============================================================================
# METRICS ENDPOINTS
# ============================================================================

@app.post("/api/models/{model_id}/metrics")
async def update_model_metrics(
    model_id: str,
    metric: MetricUpdateSchema,
    db: Session = Depends(get_db)
):
    """Update metrics for a model."""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    db_metric = ModelMetric(
        model_id=model_id,
        health_score=metric.health_score,
        failure_risk=metric.failure_risk,
        failure_time_predicted=metric.failure_time_predicted,
        ece_25=metric.ece_25,
        ece_50=metric.ece_50,
        ece_75=metric.ece_75,
        calibration_error=metric.calibration_error,
        rmse=metric.rmse,
        r_square=metric.r_square,
        accuracy=metric.accuracy,
        brier_score=metric.brier_score,
        grad_ece_correlation=metric.grad_ece_correlation,
        gradient_norm=metric.gradient_norm,
        max_gradient=metric.max_gradient,
        ece_flat=metric.ece_flat,
        calibration_drift=metric.calibration_drift
    )
    db.add(db_metric)
    db.commit()
    
    # Broadcast update to connected WebSocket clients
    await manager.broadcast({
        "type": "metrics_update",
        "model_id": model_id,
        "health_score": metric.health_score,
        "failure_risk": metric.failure_risk,
        "timestamp": db_metric.timestamp.isoformat()
    })
    
    return {"status": "ok", "metric_id": db_metric.id}


# ============================================================================
# ALERTS ENDPOINTS
# ============================================================================

@app.get("/api/alerts")
async def get_alerts(db: Session = Depends(get_db)):
    """Get recent alerts."""
    alerts = (
        db.query(Alert)
        .order_by(Alert.timestamp.desc())
        .limit(100)
        .all()
    )
    
    return [
        {
            "id": a.id,
            "model_id": a.model_id,
            "timestamp": a.timestamp,
            "message": a.message,
            "severity": a.severity,
            "resolved": a.resolved
        }
        for a in alerts
    ]


@app.post("/api/alerts")
async def create_alert(
    alert: AlertCreateSchema,
    db: Session = Depends(get_db)
):
    """Create a new alert."""
    import uuid
    from datetime import datetime
    
    db_alert = Alert(
        id=str(uuid.uuid4()),
        model_id=alert.model_id,
        message=alert.message,
        severity=alert.severity,
        timestamp=datetime.utcnow()
    )
    db.add(db_alert)
    db.commit()
    
    # Broadcast alert to connected WebSocket clients
    await manager.broadcast({
        "type": "alert",
        "id": db_alert.id,
        "model_id": alert.model_id,
        "message": alert.message,
        "severity": alert.severity,
        "timestamp": db_alert.timestamp.isoformat()
    })
    
    return {"status": "ok", "alert_id": db_alert.id}


# ============================================================================
# DASHBOARD ENDPOINTS
# ============================================================================

@app.get("/api/dashboard", response_model=DashboardStateSchema)
async def get_dashboard_state(db: Session = Depends(get_db)):
    """Get complete dashboard state for initial load."""
    # Get all models with latest metrics
    models = db.query(Model).all()
    models_data = []
    
    for model in models:
        latest_metric = (
            db.query(ModelMetric)
            .filter(ModelMetric.model_id == model.id)
            .order_by(ModelMetric.timestamp.desc())
            .first()
        )
        
        if latest_metric:
            app_context = db.query(AppContext).filter(
                AppContext.model_id == model.id
            ).first()
            
            if app_context:
                models_data.append({
                    "id": model.id,
                    "name": model.name,
                    "architecture": model.architecture,
                    "health_score": latest_metric.health_score,
                    "failure_risk": latest_metric.failure_risk,
                    "failure_time_predicted": latest_metric.failure_time_predicted,
                    "ece_25": latest_metric.ece_25,
                    "ece_50": latest_metric.ece_50,
                    "ece_75": latest_metric.ece_75,
                    "calibration_error": latest_metric.calibration_error,
                    "rmse": latest_metric.rmse,
                    "r_square": latest_metric.r_square,
                    "accuracy": latest_metric.accuracy,
                    "grad_ece_correlation": latest_metric.grad_ece_correlation,
                    "ece_flat": latest_metric.ece_flat,
                    "insights": [],
                    "app_context": {
                        "name": app_context.name,
                        "exposure": app_context.exposure,
                        "metric_name": app_context.metric_name,
                        "metric_value": app_context.metric_value
                    },
                    "updated_at": latest_metric.timestamp
                })
    
    # Get recent alerts
    alerts = (
        db.query(Alert)
        .order_by(Alert.timestamp.desc())
        .limit(50)
        .all()
    )
    
    alerts_data = [
        {
            "id": a.id,
            "timestamp": a.timestamp,
            "message": a.message,
            "severity": a.severity,
            "resolved": a.resolved
        }
        for a in alerts
    ]
    
    return {
        "models": models_data,
        "alerts": alerts_data,
        "chart_data": []
    }


# ============================================================================
# SIMULATION ENDPOINTS
# ============================================================================

@app.post("/api/simulate/trigger-failure")
async def trigger_failure_scenario(model_id: str = "lstm", db: Session = Depends(get_db)):
    """Trigger a failure scenario for testing."""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Create critical metrics
    import uuid
    critical_metric = ModelMetric(
        model_id=model_id,
        health_score=35,
        failure_risk="critical",
        failure_time_predicted=37,
        ece_25=0.20,
        ece_50=0.45,
        ece_75=0.65,
        calibration_error=0.25,
        rmse=0.35,
        r_square=0.50,
        accuracy=0.55,
        grad_ece_correlation=0.92,
        ece_flat=False
    )
    db.add(critical_metric)
    
    # Create alert
    alert = Alert(
        id=str(uuid.uuid4()),
        model_id=model_id,
        message=f"CRITICAL: {model.name} showing pre-failure signature! Predicted failure in 37 timesteps.",
        severity="critical"
    )
    db.add(alert)
    db.commit()
    
    # Broadcast
    await manager.broadcast({
        "type": "failure_scenario",
        "model_id": model_id,
        "alert_id": alert.id,
        "health_score": 35,
        "failure_risk": "critical"
    })
    
    return {"status": "ok", "alert_id": alert.id}


# ============================================================================
# WEBSOCKET ENDPOINTS
# ============================================================================

@app.websocket("/ws/metrics")
async def websocket_metrics_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics updates."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back or process ping
            if data == "ping":
                await websocket.send_text("pong")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
