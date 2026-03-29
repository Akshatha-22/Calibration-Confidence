"""Hyperparameter tuning script for model variants.

Supports grid search, random search, and Optuna (if installed).
Results are saved under results/hyperparameter_tuning by default.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Ensure repo root is on sys.path when running this script directly.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import optuna  # type: ignore

    OPTUNA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    optuna = None  # type: ignore
    OPTUNA_AVAILABLE = False

from data.preprocessing.data_loaders import get_mlp_loaders, get_rnn_loaders
from data.preprocessing.split_data import create_splits
from experiments.train import (
    collect_classification_results,
    eval_epoch_classification,
    train_model,
)

MODEL_ALIASES = {
    "mlp": "mlp",
    "deep": "deep",
    "deep_mlp": "deep",
    "rnn": "rnn",
    "vanilla_rnn": "rnn",
    "lstm": "lstm",
    "residual": "residual",
    "residual_mlp": "residual",
}

MODEL_OUTPUT_DIR = {
    "mlp": "mlp",
    "deep": "deep_mlp",
    "rnn": "vanilla_rnn",
    "lstm": "lstm",
    "residual": "residual_mlp",
}

MODEL_INGEST_DIR = {
    "mlp": "mlp",
    "deep": "deep",
    "rnn": "rnn",
    "lstm": "lstm",
    "residual": "residual",
}


@dataclass
class TrialConfig:
    model: str
    seq_len: int
    lr: float
    batch_size: int
    epochs: int
    val_ratio: float
    split_seed: int | None
    dropout: float
    task: str
    train_path: str
    val_path: str
    test_path: str
    max_features: int
    max_seq_len: int
    mlp_hidden_sizes: Tuple[int, ...] | None
    deep_hidden_sizes: Tuple[int, ...] | None
    rnn_hidden_size: int
    rnn_num_layers: int
    lstm_hidden_size: int
    lstm_num_layers: int
    residual_hidden_size: int
    residual_num_blocks: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "model": self.model,
            "seq_len": self.seq_len,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "val_ratio": self.val_ratio,
            "split_seed": self.split_seed,
            "dropout": self.dropout,
            "task": self.task,
            "train_path": self.train_path,
            "val_path": self.val_path,
            "test_path": self.test_path,
            "max_features": self.max_features,
            "max_seq_len": self.max_seq_len,
            "mlp_hidden_sizes": list(self.mlp_hidden_sizes) if self.mlp_hidden_sizes else None,
            "deep_hidden_sizes": list(self.deep_hidden_sizes) if self.deep_hidden_sizes else None,
            "rnn_hidden_size": self.rnn_hidden_size,
            "rnn_num_layers": self.rnn_num_layers,
            "lstm_hidden_size": self.lstm_hidden_size,
            "lstm_num_layers": self.lstm_num_layers,
            "residual_hidden_size": self.residual_hidden_size,
            "residual_num_blocks": self.residual_num_blocks,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_search_space(model: str) -> Dict[str, Iterable]:
    """Return a dict of hyperparameter lists for the given model."""
    lrs = [1e-3, 1e-2, 1e-4, 5e-4]
    if model == "mlp":
        return {
            "hidden_size": [32, 64, 128, 256],
            "lr": lrs,
        }
    if model == "deep":
        return {
            "depth": [3, 4, 5, 6],
            "lr": lrs,
        }
    if model in ("rnn", "lstm"):
        return {
            "hidden_size": [32, 64, 128, 256],
            "lr": lrs,
            "seq_len": [10, 20, 30],
        }
    if model == "residual":
        return {
            "hidden_size": [32, 64, 128, 256],
            "lr": lrs,
        }
    raise ValueError(f"Unknown model '{model}'")


def grid_configs(
    model: str,
    base: Dict[str, object],
) -> List[TrialConfig]:
    space = build_search_space(model)
    keys = list(space.keys())
    combos = list(itertools.product(*[space[k] for k in keys]))

    configs: List[TrialConfig] = []
    for values in combos:
        params = dict(zip(keys, values))
        configs.append(make_config(model, base, params))
    return configs


def make_config(model: str, base: Dict[str, object], params: Dict[str, object]) -> TrialConfig:
    # Defaults for all models
    seq_len = int(params.get("seq_len", base["seq_len"]))
    max_seq_len = int(params.get("seq_len", base["max_seq_len"]))
    lr = float(params.get("lr", base["lr"]))
    hidden_size = int(params.get("hidden_size", base["hidden_size"]))
    depth = int(params.get("depth", base["depth"]))

    mlp_hidden_sizes = None
    deep_hidden_sizes = None
    rnn_hidden_size = hidden_size
    rnn_num_layers = int(base["rnn_num_layers"])
    lstm_hidden_size = hidden_size
    lstm_num_layers = int(base["lstm_num_layers"])
    residual_hidden_size = hidden_size
    residual_num_blocks = int(base["residual_num_blocks"])

    if model == "mlp":
        mlp_hidden_sizes = (hidden_size, hidden_size)
    elif model == "deep":
        deep_hidden_sizes = tuple([int(base["deep_base_hidden"])] * depth)

    return TrialConfig(
        model=model,
        seq_len=seq_len,
        lr=lr,
        batch_size=int(base["batch_size"]),
        epochs=int(base["epochs"]),
        val_ratio=float(base["val_ratio"]),
        split_seed=base["split_seed"],
        dropout=float(base["dropout"]),
        task=str(base["task"]),
        train_path=str(base["train_path"]),
        val_path=str(base["val_path"]),
        test_path=str(base["test_path"]),
        max_features=int(base["max_features"]),
        max_seq_len=max_seq_len,
        mlp_hidden_sizes=mlp_hidden_sizes,
        deep_hidden_sizes=deep_hidden_sizes,
        rnn_hidden_size=rnn_hidden_size,
        rnn_num_layers=rnn_num_layers,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        residual_hidden_size=residual_hidden_size,
        residual_num_blocks=residual_num_blocks,
    )


def summarize_history(history: Dict[str, List[float]]) -> Dict[str, float]:
    best_val_loss = float(np.min(history["val_loss"])) if history["val_loss"] else float("inf")
    best_epoch = int(np.argmin(history["val_loss"])) + 1 if history["val_loss"] else -1
    final_val_loss = float(history["val_loss"][-1]) if history["val_loss"] else float("inf")
    final_ece = float(history["ece"][-1]) if history.get("ece") else float("nan")
    final_train_loss = float(history["train_loss"][-1]) if history["train_loss"] else float("inf")
    final_grad_norm = (
        float(history["train_grad_norm"][-1]) if history.get("train_grad_norm") else float("nan")
    )
    final_rmse = float(history.get("val_rmse", [float("nan")])[-1])
    final_r2 = float(history.get("val_r2", [float("nan")])[-1])
    result = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_val_loss": final_val_loss,
        "final_train_loss": final_train_loss,
        "final_ece": final_ece,
        "final_grad_norm": final_grad_norm,
        "final_rmse": final_rmse,
        "final_r2": final_r2,
    }
    if history.get("val_accuracy"):
        result["final_val_accuracy"] = float(history["val_accuracy"][-1])
    if history.get("train_accuracy"):
        result["final_train_accuracy"] = float(history["train_accuracy"][-1])
    if history.get("val_precision"):
        result["final_val_precision"] = float(history["val_precision"][-1])
    if history.get("val_recall"):
        result["final_val_recall"] = float(history["val_recall"][-1])
    if history.get("val_f1"):
        result["final_val_f1"] = float(history["val_f1"][-1])
    if history.get("val_brier"):
        result["final_val_brier"] = float(history["val_brier"][-1])
    if history.get("val_avg_confidence"):
        result["final_val_avg_confidence"] = float(history["val_avg_confidence"][-1])
    if history.get("val_confidence_correct"):
        result["final_val_confidence_correct"] = float(history["val_confidence_correct"][-1])
    if history.get("val_confidence_incorrect"):
        result["final_val_confidence_incorrect"] = float(history["val_confidence_incorrect"][-1])
    if history.get("training_time"):
        result["training_time"] = float(history["training_time"][-1])
    if history.get("num_parameters"):
        result["num_parameters"] = float(history["num_parameters"][-1])
    return result


def run_trial(
    trial_id: int,
    cfg: TrialConfig,
    data_path: str,
    run_dir: str,
    run_name: str,
    early_stopping_patience: int,
    log_backend: str,
    log_dir: str,
    wandb_project: str | None,
) -> Dict[str, object]:
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_path = os.path.join(run_dir, "checkpoint.pt")
    trial_results_path = os.path.join(run_dir, "results.npz")

    model, history = train_model(
        model_name=cfg.model,
        data_path=data_path,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        lr=cfg.lr,
        val_ratio=cfg.val_ratio,
        split_seed=cfg.split_seed,
        task=cfg.task,
        train_path=cfg.train_path,
        val_path=cfg.val_path,
        test_path=cfg.test_path,
        max_features=cfg.max_features,
        max_seq_len=cfg.max_seq_len,
        mlp_hidden_sizes=cfg.mlp_hidden_sizes,
        deep_hidden_sizes=cfg.deep_hidden_sizes,
        rnn_hidden_size=cfg.rnn_hidden_size,
        rnn_num_layers=cfg.rnn_num_layers,
        lstm_hidden_size=cfg.lstm_hidden_size,
        lstm_num_layers=cfg.lstm_num_layers,
        residual_hidden_size=cfg.residual_hidden_size,
        residual_num_blocks=cfg.residual_num_blocks,
        dropout=cfg.dropout,
        checkpoint_path=checkpoint_path,
        resume=False,
        early_stopping_patience=early_stopping_patience,
        log_backend=log_backend,
        log_dir=log_dir,
        wandb_project=wandb_project,
        wandb_run_name=f"{cfg.model}_trial_{trial_id}",
        results_path=trial_results_path,
    )

    summary = summarize_history(history)
    hidden_size = cfg.rnn_hidden_size
    if cfg.model == "deep" and cfg.deep_hidden_sizes:
        hidden_size = cfg.deep_hidden_sizes[0]
    elif cfg.model == "mlp" and cfg.mlp_hidden_sizes:
        hidden_size = cfg.mlp_hidden_sizes[0]
    elif cfg.model == "lstm":
        hidden_size = cfg.lstm_hidden_size
    elif cfg.model == "residual":
        hidden_size = cfg.residual_hidden_size
    record: Dict[str, object] = {
        "trial_id": trial_id,
        "run_name": run_name,
        "run_dir": run_dir,
        **cfg.to_dict(),
        **summary,
        "learning_rate": cfg.lr,
        "hidden_size": hidden_size,
        "ECE": summary.get("final_ece"),
        "RMSE": summary.get("final_rmse"),
        "R2": summary.get("final_r2"),
        "grad_norm": summary.get("final_grad_norm"),
        "timestep": cfg.seq_len,
    }
    return record


def evaluate_best_on_test(
    cfg: TrialConfig,
    data_path: str,
    output_dir: str,
    early_stopping_patience: int,
    log_backend: str,
    log_dir: str,
    wandb_project: str | None,
) -> Dict[str, float]:
    """Train best config and evaluate on held-out test set."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "best_checkpoint.pt")
    val_results_path = os.path.join(output_dir, "val_results.npz")
    test_results_path = os.path.join(output_dir, "test_results.npz")

    model, _ = train_model(
        model_name=cfg.model,
        data_path=data_path,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        lr=cfg.lr,
        val_ratio=cfg.val_ratio,
        split_seed=cfg.split_seed,
        task=cfg.task,
        train_path=cfg.train_path,
        val_path=cfg.val_path,
        test_path=cfg.test_path,
        max_features=cfg.max_features,
        max_seq_len=cfg.max_seq_len,
        mlp_hidden_sizes=cfg.mlp_hidden_sizes,
        deep_hidden_sizes=cfg.deep_hidden_sizes,
        rnn_hidden_size=cfg.rnn_hidden_size,
        rnn_num_layers=cfg.rnn_num_layers,
        lstm_hidden_size=cfg.lstm_hidden_size,
        lstm_num_layers=cfg.lstm_num_layers,
        residual_hidden_size=cfg.residual_hidden_size,
        residual_num_blocks=cfg.residual_num_blocks,
        dropout=cfg.dropout,
        checkpoint_path=checkpoint_path,
        resume=False,
        early_stopping_patience=early_stopping_patience,
        log_backend=log_backend,
        log_dir=log_dir,
        wandb_project=wandb_project,
        wandb_run_name=f"{cfg.model}_best",
        results_path=val_results_path,
    )

    if cfg.model in ("rnn", "lstm"):
        _, _, test_loader = get_rnn_loaders(
            train_path=cfg.train_path,
            val_path=cfg.val_path,
            test_path=cfg.test_path,
            batch_size=cfg.batch_size,
            max_seq_len=cfg.max_seq_len,
        )
    else:
        _, _, test_loader = get_mlp_loaders(
            train_path=cfg.train_path,
            val_path=cfg.val_path,
            test_path=cfg.test_path,
            batch_size=cfg.batch_size,
            max_features=cfg.max_features,
        )

    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss()
    test_metrics = eval_epoch_classification(model, test_loader, loss_fn, device, cfg.model)
    collect_classification_results(
        model=model,
        loader=test_loader,
        device=device,
        model_name=cfg.model,
        save_path=test_results_path,
    )

    return {
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_ece": float(test_metrics["ece"]),
        "test_f1_score": float(test_metrics["f1"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_brier_score": float(test_metrics["brier"]),
        "test_avg_confidence": float(test_metrics["avg_confidence"]),
        "test_confidence_correct": float(test_metrics["confidence_correct"]),
        "test_confidence_incorrect": float(test_metrics["confidence_incorrect"]),
    }


def _format_lr(lr: float) -> str:
    text = f"{lr:.6f}"
    text = text.rstrip("0").rstrip(".")
    return text if text else "0"


def run_slug(cfg: TrialConfig, model: str) -> str:
    if model in ("mlp", "residual", "rnn", "lstm"):
        return f"h{cfg.rnn_hidden_size}_lr{_format_lr(cfg.lr)}"
    if model == "deep":
        depth = len(cfg.deep_hidden_sizes) if cfg.deep_hidden_sizes else 0
        return f"d{depth}_lr{_format_lr(cfg.lr)}"
    return f"trial_{_format_lr(cfg.lr)}"


def _parse_list_field(value: object) -> List[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str):
        raw = value.strip()
        if not raw or raw.lower() == "none":
            return None
        parts = [p for p in raw.split("|") if p]
        if not parts:
            return None
        return [int(float(p)) for p in parts]
    return None


def _run_slug_from_record(record: Dict[str, object], model: str) -> str:
    lr = float(record.get("lr", 0.0))
    if model == "deep":
        deep_sizes = _parse_list_field(record.get("deep_hidden_sizes"))
        depth = len(deep_sizes) if deep_sizes else 0
        return f"d{depth}_lr{_format_lr(lr)}"
    if model in ("mlp", "rnn", "lstm", "residual"):
        hidden_size = None
        mlp_sizes = _parse_list_field(record.get("mlp_hidden_sizes"))
        if mlp_sizes:
            hidden_size = mlp_sizes[0]
        for key in ("rnn_hidden_size", "lstm_hidden_size", "residual_hidden_size"):
            if hidden_size is None and record.get(key) is not None:
                hidden_size = int(float(record[key]))  # type: ignore[arg-type]
        if hidden_size is None:
            hidden_size = 0
        return f"h{hidden_size}_lr{_format_lr(lr)}"
    return f"trial_{_format_lr(lr)}"


def _read_csv_records(path: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    trial_map: Dict[int, TrialConfig] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records


def _read_jsonl_records(path: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _coerce_types(record: Dict[str, object]) -> Dict[str, object]:
    int_fields = {
        "trial_id",
        "seq_len",
        "batch_size",
        "epochs",
        "split_seed",
        "rnn_num_layers",
        "lstm_num_layers",
        "residual_num_blocks",
        "best_epoch",
        "max_features",
        "max_seq_len",
        "hidden_size",
        "timestep",
    }
    float_fields = {
        "lr",
        "learning_rate",
        "val_ratio",
        "dropout",
        "best_val_loss",
        "final_val_loss",
        "final_train_loss",
        "final_ece",
        "final_grad_norm",
        "final_rmse",
        "final_r2",
        "ECE",
        "RMSE",
        "R2",
        "grad_norm",
        "final_val_accuracy",
        "final_train_accuracy",
        "final_val_precision",
        "final_val_recall",
        "final_val_f1",
        "final_val_brier",
        "final_val_avg_confidence",
        "final_val_confidence_correct",
        "final_val_confidence_incorrect",
        "training_time",
        "num_parameters",
        "test_accuracy",
        "test_ece",
        "test_f1_score",
        "test_precision",
        "test_recall",
        "test_brier_score",
        "test_avg_confidence",
        "test_confidence_correct",
        "test_confidence_incorrect",
    }
    list_fields = {"mlp_hidden_sizes", "deep_hidden_sizes"}

    coerced: Dict[str, object] = {}
    for key, value in record.items():
        if isinstance(value, str):
            raw = value.strip()
            if raw.lower() == "none" or raw == "":
                coerced[key] = None
                continue
        if key in list_fields:
            coerced[key] = _parse_list_field(value)
            continue
        if key in int_fields and value is not None:
            coerced[key] = int(float(value))  # handles "42.0"
            continue
        if key in float_fields and value is not None:
            coerced[key] = float(value)
            continue
        coerced[key] = value
    return coerced


def ingest_results(
    model_key: str,
    ingest_from: str,
    ingest_run: str | None,
    output_dir: str,
) -> None:
    ingest_model_dir = os.path.join(ingest_from, MODEL_INGEST_DIR[model_key])
    if not os.path.isdir(ingest_model_dir):
        raise FileNotFoundError(f"No tuning directory found at {ingest_model_dir}")

    if ingest_run:
        run_dir = os.path.join(ingest_model_dir, ingest_run)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"No tuning run found at {run_dir}")
    else:
        run_dirs = [
            os.path.join(ingest_model_dir, d)
            for d in os.listdir(ingest_model_dir)
            if os.path.isdir(os.path.join(ingest_model_dir, d))
        ]
        if not run_dirs:
            raise FileNotFoundError(f"No tuning runs found under {ingest_model_dir}")
        run_dir = max(run_dirs, key=os.path.getmtime)

    csv_path = os.path.join(run_dir, "tuning_results.csv")
    jsonl_path = os.path.join(run_dir, "tuning_results.jsonl")
    if os.path.exists(csv_path):
        raw_records = _read_csv_records(csv_path)
    elif os.path.exists(jsonl_path):
        raw_records = _read_jsonl_records(jsonl_path)
    else:
        raise FileNotFoundError(f"No tuning results found in {run_dir}")

    model_dir = os.path.join(output_dir, MODEL_OUTPUT_DIR[model_key])
    runs_dir = os.path.join(model_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    records: List[Dict[str, object]] = []
    checkpoints_dir = os.path.join(run_dir, "checkpoints")

    for raw in raw_records:
        rec = _coerce_types(raw)
        trial_id = int(rec.get("trial_id") or 0)
        run_name = _run_slug_from_record(rec, model_key)
        new_run_dir = os.path.join(runs_dir, run_name)
        os.makedirs(new_run_dir, exist_ok=True)

        # Copy trial artifacts if present.
        old_trial_dir = os.path.join(run_dir, f"trial_{trial_id}")
        old_results = os.path.join(old_trial_dir, "results.npz")
        if os.path.exists(old_results):
            new_results = os.path.join(new_run_dir, "results.npz")
            if not os.path.exists(new_results):
                with open(old_results, "rb") as src, open(new_results, "wb") as dst:
                    dst.write(src.read())

        old_ckpt = os.path.join(checkpoints_dir, f"{model_key}_trial_{trial_id}.pt")
        new_ckpt = os.path.join(new_run_dir, "checkpoint.pt")
        if os.path.exists(old_ckpt) and not os.path.exists(new_ckpt):
            with open(old_ckpt, "rb") as src, open(new_ckpt, "wb") as dst:
                dst.write(src.read())

        rec["run_name"] = run_name
        rec["run_dir"] = new_run_dir
        records.append(rec)

    out_csv = os.path.join(model_dir, "results.csv")
    out_jsonl = os.path.join(model_dir, "results.jsonl")
    save_results(records, out_csv, out_jsonl)


def save_results(records: List[Dict[str, object]], out_csv: str, out_jsonl: str) -> None:
    if not records:
        return
    preferred_order = [
        "trial_id",
        "run_name",
        "run_dir",
        "model",
        "learning_rate",
        "hidden_size",
        "ECE",
        "RMSE",
        "R2",
        "grad_norm",
        "timestep",
        "task",
        "seq_len",
        "lr",
        "batch_size",
        "epochs",
        "val_ratio",
        "split_seed",
        "dropout",
        "train_path",
        "val_path",
        "test_path",
        "max_features",
        "max_seq_len",
        "mlp_hidden_sizes",
        "deep_hidden_sizes",
        "rnn_hidden_size",
        "rnn_num_layers",
        "lstm_hidden_size",
        "lstm_num_layers",
        "residual_hidden_size",
        "residual_num_blocks",
        "best_val_loss",
        "best_epoch",
        "final_val_loss",
        "final_train_loss",
        "final_ece",
        "final_grad_norm",
        "final_rmse",
        "final_r2",
        "final_val_accuracy",
        "final_train_accuracy",
        "final_val_precision",
        "final_val_recall",
        "final_val_f1",
        "final_val_brier",
        "final_val_avg_confidence",
        "final_val_confidence_correct",
        "final_val_confidence_incorrect",
        "training_time",
        "num_parameters",
        "test_accuracy",
        "test_ece",
        "test_f1_score",
        "test_precision",
        "test_recall",
        "test_brier_score",
        "test_avg_confidence",
        "test_confidence_correct",
        "test_confidence_incorrect",
    ]
    field_set = set()
    for rec in records:
        field_set.update(rec.keys())
    fieldnames = [f for f in preferred_order if f in field_set] + [
        f for f in sorted(field_set) if f not in preferred_order
    ]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(fieldnames) + "\n")
        for rec in records:
            row = []
            for name in fieldnames:
                value = rec.get(name)
                if isinstance(value, list):
                    value = "|".join(str(v) for v in value)
                row.append(str(value))
            f.write(",".join(row) + "\n")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for calibration-confidence models")
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=[
            "mlp",
            "deep",
            "deep_mlp",
            "rnn",
            "vanilla_rnn",
            "lstm",
            "residual",
            "residual_mlp",
        ],
    )
    parser.add_argument("--data-path", type=str, default="data/finsen/raw")
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "regression"])
    parser.add_argument("--train-path", type=str, default="data/finsen/processed/train.csv")
    parser.add_argument("--val-path", type=str, default="data/finsen/processed/val.csv")
    parser.add_argument("--test-path", type=str, default="data/finsen/processed/test.csv")
    parser.add_argument("--input-csv", type=str, default="data/finsen/raw/FinSen_US_Categorized.csv")
    parser.add_argument("--max-features", type=int, default=2000)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--rnn-num-layers", type=int, default=1)
    parser.add_argument("--lstm-num-layers", type=int, default=1)
    parser.add_argument("--residual-num-blocks", type=int, default=3)
    parser.add_argument("--deep-base-hidden", type=int, default=128, help="Base hidden size repeated across deep MLP depth")
    parser.add_argument("--method", type=str, default="grid", choices=["grid", "random", "optuna"])
    parser.add_argument("--n-trials", type=int, default=20, help="Number of trials for random/optuna")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--log-backend", type=str, default="none", choices=["none", "tensorboard", "wandb"])
    parser.add_argument("--log-dir", type=str, default="results/logs")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="results/hyperparameter_tuning")
    parser.add_argument(
        "--ingest-from",
        type=str,
        default=None,
        help="Ingest existing tuning outputs instead of training (e.g., results/tuning).",
    )
    parser.add_argument(
        "--ingest-run",
        type=str,
        default=None,
        help="Specific run directory under --ingest-from/<model> to ingest.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    model_key = MODEL_ALIASES.get(args.model)
    if model_key is None:
        raise ValueError(f"Unknown model '{args.model}'")
    model_dir_name = MODEL_OUTPUT_DIR[model_key]
    model_dir = os.path.join(args.results_dir, model_dir_name)
    runs_dir = os.path.join(model_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    if args.ingest_from:
        ingest_results(
            model_key=model_key,
            ingest_from=args.ingest_from,
            ingest_run=args.ingest_run,
            output_dir=args.results_dir,
        )
        print(f"Ingested tuning results into: {model_dir}")
        return

    if args.task == "classification":
        if not (os.path.exists(args.train_path) and os.path.exists(args.val_path) and os.path.exists(args.test_path)):
            print("Train/val/test splits not found. Creating new 70/15/15 split...")
            create_splits(
                input_csv=args.input_csv,
                output_dir=os.path.dirname(args.train_path),
                train_size=0.7,
                val_size=0.15,
                test_size=0.15,
                random_state=args.split_seed,
            )

    base = {
        "seq_len": args.seq_len,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "val_ratio": args.val_ratio,
        "split_seed": args.split_seed,
        "dropout": args.dropout,
        "task": args.task,
        "train_path": args.train_path,
        "val_path": args.val_path,
        "test_path": args.test_path,
        "max_features": args.max_features,
        "max_seq_len": args.max_seq_len,
        "rnn_num_layers": args.rnn_num_layers,
        "lstm_num_layers": args.lstm_num_layers,
        "residual_num_blocks": args.residual_num_blocks,
        "hidden_size": 128,
        "depth": 4,
        "deep_base_hidden": args.deep_base_hidden,
    }

    records: List[Dict[str, object]] = []

    if args.method == "grid":
        configs = grid_configs(model_key, base)
        for idx, cfg in enumerate(configs, start=1):
            run_name = run_slug(cfg, model_key)
            run_dir = os.path.join(runs_dir, run_name)
            rec = run_trial(
                trial_id=idx,
                cfg=cfg,
                data_path=args.data_path,
                run_dir=run_dir,
                run_name=run_name,
                early_stopping_patience=args.early_stopping_patience,
                log_backend=args.log_backend,
                log_dir=args.log_dir,
                wandb_project=args.wandb_project,
            )
            records.append(rec)
            trial_map[idx] = cfg
    elif args.method == "random":
        configs = grid_configs(model_key, base)
        random.shuffle(configs)
        configs = configs[: args.n_trials]
        for idx, cfg in enumerate(configs, start=1):
            run_name = run_slug(cfg, model_key)
            run_dir = os.path.join(runs_dir, run_name)
            rec = run_trial(
                trial_id=idx,
                cfg=cfg,
                data_path=args.data_path,
                run_dir=run_dir,
                run_name=run_name,
                early_stopping_patience=args.early_stopping_patience,
                log_backend=args.log_backend,
                log_dir=args.log_dir,
                wandb_project=args.wandb_project,
            )
            records.append(rec)
            trial_map[idx] = cfg
    else:
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna is not available. Install optuna or choose grid/random.")

        def objective(trial: "optuna.trial.Trial") -> float:
            params: Dict[str, object] = {}
            space = build_search_space(model_key)
            if "hidden_size" in space:
                params["hidden_size"] = trial.suggest_categorical("hidden_size", list(space["hidden_size"]))
            if "depth" in space:
                params["depth"] = trial.suggest_categorical("depth", list(space["depth"]))
            if "seq_len" in space:
                params["seq_len"] = trial.suggest_categorical("seq_len", list(space["seq_len"]))
            if "lr" in space:
                params["lr"] = trial.suggest_categorical("lr", list(space["lr"]))
            cfg = make_config(model_key, base, params)
            run_name = run_slug(cfg, model_key)
            run_dir = os.path.join(runs_dir, run_name)
            rec = run_trial(
                trial_id=trial.number + 1,
                cfg=cfg,
                data_path=args.data_path,
                run_dir=run_dir,
                run_name=run_name,
                early_stopping_patience=args.early_stopping_patience,
                log_backend=args.log_backend,
                log_dir=args.log_dir,
                wandb_project=args.wandb_project,
            )
            records.append(rec)
            trial_map[trial.number + 1] = cfg
            return float(rec["best_val_loss"])

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=args.n_trials)

    if records and args.task == "classification":
        best = min(records, key=lambda r: float(r["best_val_loss"]))
        best_trial_id = int(best.get("trial_id") or 0)
        best_cfg = trial_map.get(best_trial_id)
        if best_cfg is not None:
            best_output_dir = os.path.join(model_dir, "best_model")
            test_metrics = evaluate_best_on_test(
                cfg=best_cfg,
                data_path=args.data_path,
                output_dir=best_output_dir,
                early_stopping_patience=args.early_stopping_patience,
                log_backend=args.log_backend,
                log_dir=args.log_dir,
                wandb_project=args.wandb_project,
            )
            best.update(test_metrics)

    out_csv = os.path.join(model_dir, "results.csv")
    out_jsonl = os.path.join(model_dir, "results.jsonl")
    save_results(records, out_csv, out_jsonl)

    if records:
        best = min(records, key=lambda r: float(r["best_val_loss"]))
        print("Best trial:")
        for key, val in best.items():
            print(f"  {key}: {val}")
        print(f"Saved results to: {out_csv}")


if __name__ == "__main__":
    main()
