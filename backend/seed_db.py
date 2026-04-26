#!/usr/bin/env python
"""Initialize database with seed data."""

import os
import sys
from datetime import datetime

# Add repo root to path
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.database.session import SessionLocal, init_db
from backend.database.models import Model, AppContext, ModelMetric


def seed_database():
    """Seed database with initial model data."""
    db = SessionLocal()
    
    try:
        # Initialize tables
        init_db()
        print("✓ Database tables created")
        
        # Define models
        models_data = [
            {
                "id": "deep-mlp",
                "name": "Deep MLP Core",
                "architecture": "deep_mlp",
                "checkpoint_path": "checkpoints_deep_mlp/best_model.pth",
            },
            {
                "id": "lstm",
                "name": "LSTM Sequence",
                "architecture": "lstm",
                "checkpoint_path": "checkpoints_lstm/best_model.pth",
            },
            {
                "id": "mlp",
                "name": "Standard MLP",
                "architecture": "mlp",
                "checkpoint_path": "checkpoints_mlp/best_model.pth",
            },
            {
                "id": "res-mlp",
                "name": "Residual MLP",
                "architecture": "residual_mlp",
                "checkpoint_path": None,
            },
            {
                "id": "vanilla-rnn",
                "name": "Vanilla RNN",
                "architecture": "vanilla_rnn",
                "checkpoint_path": "checkpoints_vanilla_rnn/best_model.pth",
            },
        ]
        
        # Create models and app contexts
        for model_data in models_data:
            # Check if exists
            existing = db.query(Model).filter(Model.id == model_data["id"]).first()
            if existing:
                print(f"⊘ Model {model_data['id']} already exists, skipping")
                continue
            
            # Create model
            model = Model(**model_data)
            db.add(model)
            db.flush()
            
            # Create app context
            app_contexts = {
                "deep-mlp": {
                    "name": "HFT Options Arbitrage",
                    "exposure": 14.2,
                    "metric_name": "Trades/sec",
                    "metric_value": 1402,
                },
                "lstm": {
                    "name": "NatGas Futures Prediction",
                    "exposure": 42.5,
                    "metric_name": "Active Positions",
                    "metric_value": 840,
                },
                "mlp": {
                    "name": "Equity Block Trading",
                    "exposure": 110.1,
                    "metric_name": "Volume (Shares)",
                    "metric_value": 154000,
                },
                "res-mlp": {
                    "name": "FX Cross-currency Engine",
                    "exposure": 53.4,
                    "metric_name": "Latency (ms)",
                    "metric_value": 0.8,
                },
                "vanilla-rnn": {
                    "name": "Credit Risk Pricing",
                    "exposure": 12.8,
                    "metric_name": "Quotes/sec",
                    "metric_value": 345,
                },
            }
            
            ctx_data = app_contexts[model_data["id"]]
            app_ctx = AppContext(
                id=f"{model_data['id']}-ctx",
                model_id=model_data["id"],
                **ctx_data
            )
            db.add(app_ctx)
            
            # Create initial metric
            metrics_data = {
                "deep-mlp": {
                    "health_score": 92,
                    "failure_risk": "low",
                    "ece_25": 0.02,
                    "ece_50": 0.05,
                    "ece_75": 0.08,
                    "calibration_error": 0.04,
                    "rmse": 0.12,
                    "r_square": 0.89,
                    "accuracy": 0.91,
                    "grad_ece_correlation": 0.1,
                    "ece_flat": True,
                },
                "lstm": {
                    "health_score": 85,
                    "failure_risk": "elevated",
                    "ece_25": 0.04,
                    "ece_50": 0.09,
                    "ece_75": 0.15,
                    "calibration_error": 0.07,
                    "rmse": 0.18,
                    "r_square": 0.81,
                    "accuracy": 0.85,
                    "grad_ece_correlation": 0.4,
                    "ece_flat": False,
                },
                "mlp": {
                    "health_score": 95,
                    "failure_risk": "low",
                    "ece_25": 0.01,
                    "ece_50": 0.03,
                    "ece_75": 0.06,
                    "calibration_error": 0.03,
                    "rmse": 0.11,
                    "r_square": 0.92,
                    "accuracy": 0.94,
                    "grad_ece_correlation": 0.05,
                    "ece_flat": True,
                },
                "res-mlp": {
                    "health_score": 88,
                    "failure_risk": "low",
                    "ece_25": 0.03,
                    "ece_50": 0.06,
                    "ece_75": 0.09,
                    "calibration_error": 0.05,
                    "rmse": 0.14,
                    "r_square": 0.85,
                    "accuracy": 0.88,
                    "grad_ece_correlation": 0.2,
                    "ece_flat": True,
                },
                "vanilla-rnn": {
                    "health_score": 78,
                    "failure_risk": "high",
                    "ece_25": 0.09,
                    "ece_50": 0.15,
                    "ece_75": 0.25,
                    "calibration_error": 0.12,
                    "rmse": 0.25,
                    "r_square": 0.73,
                    "accuracy": 0.77,
                    "grad_ece_correlation": 0.65,
                    "ece_flat": False,
                    "failure_time_predicted": 85,
                },
            }
            
            metric = ModelMetric(
                model_id=model_data["id"],
                **metrics_data[model_data["id"]]
            )
            db.add(metric)
            
            print(f"✓ Created model: {model_data['name']}")
        
        db.commit()
        print("\n✓ Database seeding complete!")
        
    except Exception as e:
        db.rollback()
        print(f"✗ Error seeding database: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
