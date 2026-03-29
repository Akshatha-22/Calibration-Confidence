"""Training loop for models.

This script demonstrates how to wire together a dataset, model, and optimizer
for training a simple windowed forecast model.

Example:
    python experiments/train.py --data-path data/finsen/raw --seq-len 50 --epochs 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

# Ensure repo root is on sys.path when running this script directly.
# This allows importing from `data` and `models` even if the current working
# directory is not the repo root.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore
    TENSORBOARD_AVAILABLE = False

try:
    import wandb

    WANDB_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False

from data.preprocessing import FinSenDataset
from data.preprocessing.data_loaders import get_mlp_loaders, get_rnn_loaders
from models.mlp import build_mlp
from models.deep_mlp import build_deep_mlp
from models.vanilla_rnn import build_vanilla_rnn
from models.lstm import build_lstm
from models.residual_mlp import build_residual_mlp
from calibration.ece import expected_calibration_error, regression_calibration_error


def split_dataset(
    dataset: FinSenDataset, val_ratio: float = 0.2, seed: int | None = None
) -> Tuple[Subset, Subset]:
    """Split dataset into train and validation subsets."""
    n = len(dataset)
    val_size = int(n * val_ratio)
    train_size = n - val_size
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _classification_accuracy_counts(
    preds: torch.Tensor, targets: torch.Tensor
) -> Tuple[int, int] | None:
    """Return (correct, total) when predictions/targets define a classification task."""
    if preds.ndim == 0 or targets.ndim == 0:
        return None

    if targets.ndim > 1 and targets.shape[-1] == 1:
        targets = targets.squeeze(-1)
    if preds.ndim > 1 and preds.shape[-1] == 1:
        preds = preds.squeeze(-1)

    if targets.ndim != 1:
        return None

    if preds.ndim > 1 and preds.shape[-1] > 1:
        pred_labels = preds.argmax(dim=-1).reshape(-1)
    elif preds.ndim == 1:
        pred_labels = preds.reshape(-1)
    else:
        return None

    if pred_labels.shape != targets.shape:
        return None

    if targets.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        target_labels = targets
    elif targets.dtype.is_floating_point:
        if not torch.allclose(targets, targets.round(), atol=1e-4, rtol=1e-4):
            return None
        target_labels = targets.round()
    else:
        return None

    target_labels = target_labels.long()
    pred_labels = pred_labels.long()
    total = target_labels.numel()
    correct = int((pred_labels == target_labels).sum().item())
    return correct, total


def _prepare_sequence_batch(x: torch.Tensor, model_name: str) -> torch.Tensor:
    """Ensure sequence inputs are shaped for RNN/LSTM classifiers."""
    if model_name in ("rnn", "lstm"):
        if x.ndim == 2:
            # Token id sequences -> treat as single feature channel
            return x.float().unsqueeze(-1)
    return x


def _logits_to_probs(logits: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities for binary/multi-class."""
    logits = np.asarray(logits)
    if logits.ndim == 1:
        # (N,) -> binary logits
        probs_pos = 1.0 / (1.0 + np.exp(-logits))
        return np.stack([1.0 - probs_pos, probs_pos], axis=1)
    if logits.shape[1] == 1:
        probs_pos = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
        return np.stack([1.0 - probs_pos, probs_pos], axis=1)
    # Multi-class
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(np.clip(logits - logits_max, -50, 50))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def _binary_metrics_from_probs(
    probs: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    """Compute binary classification metrics from probs and labels."""
    labels = labels.astype(int).reshape(-1)
    probs_pos = probs[:, 1]
    preds = (probs_pos >= 0.5).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    total = len(labels)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    confidences = np.max(probs, axis=1)
    correct_mask = preds == labels
    incorrect_mask = ~correct_mask
    avg_confidence = float(np.mean(confidences)) if total > 0 else 0.0
    conf_correct = float(np.mean(confidences[correct_mask])) if np.any(correct_mask) else float("nan")
    conf_incorrect = (
        float(np.mean(confidences[incorrect_mask])) if np.any(incorrect_mask) else float("nan")
    )

    brier = float(np.mean((probs_pos - labels) ** 2)) if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier": brier,
        "avg_confidence": avg_confidence,
        "confidence_correct": conf_correct,
        "confidence_incorrect": conf_incorrect,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    task: str,
    model_name: str,
) -> Tuple[float, float, float | None]:
    model.train()
    total_loss_list = []
    total_grad_norm_list = []
    correct_list = []
    accuracy_samples_list = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if task == "classification":
            x = _prepare_sequence_batch(x, model_name)
        preds = model(x)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()

        # Aggregate global gradient norm across all parameters for this batch.
        total_sq = []
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_sq.append(float(param_norm.item()) ** 2)
        batch_grad_norm = float(sum(total_sq) ** 0.5)

        optimizer.step()

        total_loss_list.append(float(loss.item()) * x.shape[0])
        total_grad_norm_list.append(batch_grad_norm)

        if task == "classification":
            pred_labels = preds.argmax(dim=-1)
            correct_list.append(int((pred_labels == y).sum().item()))
            accuracy_samples_list.append(int(y.numel()))
        else:
            acc_counts = _classification_accuracy_counts(preds, y)
            if acc_counts is not None:
                correct_list.append(acc_counts[0])
                accuracy_samples_list.append(acc_counts[1])

    num_batches = len(total_loss_list)
    total_loss = sum(total_loss_list)
    total_grad_norm = sum(total_grad_norm_list)
    correct = sum(correct_list)
    accuracy_samples = sum(accuracy_samples_list)

    mean_grad_norm = total_grad_norm / max(num_batches, 1)
    accuracy = correct / accuracy_samples if accuracy_samples > 0 else None

    return total_loss / len(loader.dataset), mean_grad_norm, accuracy


def eval_epoch_regression(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Evaluation for regression tasks."""
    model.eval()
    total_loss_list = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = loss_fn(preds, y)
            if torch.isfinite(loss).all():
                total_loss_list.append(float(loss.item()) * x.shape[0])
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    total_loss = sum(total_loss_list)
    n = len(loader.dataset)
    val_loss = total_loss / max(n, 1)
    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    reg_ece = regression_calibration_error(preds_arr, targets_arr, n_bins=n_bins)
    mse = float(np.mean((preds_arr - targets_arr) ** 2)) if n > 0 else float("nan")
    rmse = float(np.sqrt(mse)) if np.isfinite(mse) else float("nan")
    targets_mean = float(np.mean(targets_arr)) if n > 0 else float("nan")
    ss_res = float(np.sum((preds_arr - targets_arr) ** 2)) if n > 0 else float("nan")
    ss_tot = float(np.sum((targets_arr - targets_mean) ** 2)) if n > 0 else float("nan")
    r2 = float("nan") if ss_tot == 0.0 else float(1.0 - (ss_res / ss_tot))
    return {
        "loss": float(val_loss),
        "ece": float(reg_ece),
        "rmse": rmse,
        "r2": r2,
    }


def eval_epoch_classification(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    model_name: str,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Evaluation for binary classification."""
    model.eval()
    total_loss_list = []
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = _prepare_sequence_batch(x, model_name)
            logits = model(x)
            loss = loss_fn(logits, y)
            if torch.isfinite(loss).all():
                total_loss_list.append(float(loss.item()) * x.shape[0])
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    total_loss = sum(total_loss_list)
    n = len(loader.dataset)
    val_loss = total_loss / max(n, 1)

    logits_arr = np.concatenate(all_logits, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    probs = _logits_to_probs(logits_arr)
    metrics = _binary_metrics_from_probs(probs, targets_arr)
    ece_val = expected_calibration_error(probs, targets_arr, n_bins=n_bins)
    metrics.update({"loss": float(val_loss), "ece": float(ece_val)})
    return metrics


def collect_results(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: str | None = None,
    history: Dict[str, List[float]] | None = None,
) -> Dict[str, np.ndarray]:
    """Run model on a dataloader and collect predictions/targets.

    For classification-style outputs (logits with shape (N, C), C > 1),
    this also computes probabilities, confidences, predicted labels,
    and ECE values. Optionally merges in history (losses, ECE over time).
    """
    model.eval()

    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(y.detach().cpu().numpy())

    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)

    # Align shapes: flatten to (N,) or keep (N, F) for per-example loss
    if preds_arr.ndim > 1 and preds_arr.shape[-1] == 1:
        preds_flat = np.squeeze(preds_arr)
    else:
        preds_flat = preds_arr
    if targets_arr.ndim > 1 and targets_arr.shape[-1] == 1:
        targets_flat = np.squeeze(targets_arr)
    else:
        targets_flat = targets_arr
    if preds_flat.ndim == 1:
        preds_flat = preds_flat.reshape(-1, 1)
    if targets_flat.ndim == 1:
        targets_flat = targets_flat.reshape(-1, 1)
    # Ensure same shape for elementwise loss
    if preds_flat.shape != targets_flat.shape:
        preds_flat = preds_arr.reshape(preds_arr.shape[0], -1)
        targets_flat = targets_arr.reshape(targets_arr.shape[0], -1)
    per_example_sq = (preds_flat - targets_flat) ** 2
    per_example_loss = np.mean(per_example_sq, axis=-1)
    if not np.isfinite(per_example_loss).all():
        per_example_loss = np.nan_to_num(per_example_loss, nan=0.0, posinf=0.0, neginf=0.0)

    results: Dict[str, np.ndarray] = {
        "predictions": preds_arr,
        "targets": targets_arr,
        "per_example_loss": per_example_loss,
    }

    # If outputs look like multi-class logits, compute calibration stats.
    if preds_arr.ndim == 2 and preds_arr.shape[1] > 1:
        logits = preds_arr
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(np.clip(logits - logits_max, -50, 50))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        confidences = np.max(probs, axis=1)
        pred_labels = np.argmax(probs, axis=1)
        results["probs"] = probs
        results["confidences"] = confidences
        results["pred_labels"] = pred_labels
        if targets_arr.ndim == 1 or (targets_arr.ndim == 2 and targets_arr.shape[1] == 1):
            labels = targets_arr.reshape(-1).astype(int)
            ece_val = expected_calibration_error(probs, labels, n_bins=10)
            results["ece"] = np.asarray([ece_val], dtype=np.float32)

    if history:
        n_epochs = len(history["train_loss"])
        results["epochs"] = np.arange(1, n_epochs + 1, dtype=np.int32)
        results["train_loss"] = np.asarray(history["train_loss"], dtype=np.float32)
        results["val_loss"] = np.asarray(history["val_loss"], dtype=np.float32)
        ece_list = history.get("ece", [])
        if len(ece_list) == n_epochs:
            results["ece_over_time"] = np.asarray(ece_list, dtype=np.float32)
        if history.get("train_grad_norm"):
            results["train_grad_norm"] = np.asarray(history["train_grad_norm"], dtype=np.float32)
        lr_list = history.get("learning_rate", [])
        if len(lr_list) == n_epochs:
            results["learning_rate"] = np.asarray(lr_list, dtype=np.float32)
        train_acc = history.get("train_accuracy", [])
        val_acc = history.get("val_accuracy", [])
        if len(train_acc) == n_epochs and len(val_acc) == n_epochs:
            results["train_accuracy"] = np.asarray(train_acc, dtype=np.float32)
            results["val_accuracy"] = np.asarray(val_acc, dtype=np.float32)

    if save_path is not None:
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        np.savez_compressed(save_path, **results)

    return results


def collect_classification_results(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
    save_path: str | None = None,
) -> Dict[str, np.ndarray]:
    """Collect logits/probs/labels for classification evaluation."""
    model.eval()
    all_logits: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = _prepare_sequence_batch(x, model_name)
            logits = model(x)
            all_logits.append(logits.detach().cpu().numpy())
            all_targets.append(y.detach().cpu().numpy())

    logits_arr = np.concatenate(all_logits, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    probs = _logits_to_probs(logits_arr)
    confidences = np.max(probs, axis=1)
    pred_labels = np.argmax(probs, axis=1)

    results: Dict[str, np.ndarray] = {
        "logits": logits_arr,
        "probs": probs,
        "confidences": confidences,
        "pred_labels": pred_labels,
        "targets": targets_arr,
    }

    if save_path is not None:
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        np.savez_compressed(save_path, **results)

    return results


def train_model(
    model_name: str,
    data_path: str,
    seq_len: int,
    batch_size: int,
    epochs: int,
    lr: float,
    val_ratio: float,
    split_seed: int | None = None,
    mlp_hidden_sizes: Tuple[int, ...] | None = None,
    deep_hidden_sizes: Tuple[int, ...] | None = None,
    rnn_hidden_size: int = 64,
    rnn_num_layers: int = 1,
    lstm_hidden_size: int = 64,
    lstm_num_layers: int = 1,
    residual_hidden_size: int = 128,
    residual_num_blocks: int = 3,
    dropout: float = 0.0,
    task: str = "regression",
    train_path: str = "data/finsen/processed/train.csv",
    val_path: str = "data/finsen/processed/val.csv",
    test_path: str = "data/finsen/processed/test.csv",
    max_features: int = 2000,
    max_seq_len: int = 128,
    device: torch.device | None = None,
    checkpoint_path: str | None = None,
    resume: bool = True,
    early_stopping_patience: int = 10,
    log_backend: str = "none",
    log_dir: str = "results/logs",
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    results_path: str | None = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Unified training loop for all supported models.

    This function wires together the dataset, model, optimizer and per-epoch
    train/validation loop for all 5 model variants:
    ``mlp``, ``deep``, ``rnn``, ``lstm``, and ``residual``.

    It can be reused by other experiment scripts (e.g. hyperparameter tuning,
    robustness tests) to avoid duplicating training boilerplate.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader | None = None
    num_features: int

    if task == "classification":
        if model_name in ("rnn", "lstm"):
            train_loader, val_loader, test_loader = get_rnn_loaders(
                train_path=train_path,
                val_path=val_path,
                test_path=test_path,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
            )
            num_features = 1
        else:
            train_loader, val_loader, test_loader = get_mlp_loaders(
                train_path=train_path,
                val_path=val_path,
                test_path=test_path,
                batch_size=batch_size,
                max_features=max_features,
            )
            # infer feature size from one batch
            sample_batch = next(iter(train_loader))[0]
            num_features = int(sample_batch.shape[-1])
    else:
        dataset = FinSenDataset(data_path=data_path, seq_length=seq_len)
        train_ds, val_ds = split_dataset(dataset, val_ratio=val_ratio, seed=split_seed)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        num_features = dataset.values.shape[1]

    output_size = 2 if task == "classification" else None

    if model_name == "deep":
        model = build_deep_mlp(
            seq_len=1 if task == "classification" else seq_len,
            num_features=num_features,
            hidden_sizes=deep_hidden_sizes or (256, 128, 64, 32),
            dropout=dropout,
            output_size=output_size,
        ).to(device)
    elif model_name == "rnn":
        model = build_vanilla_rnn(
            seq_len=seq_len,
            num_features=num_features,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=dropout,
            output_size=output_size,
        ).to(device)
    elif model_name == "lstm":
        model = build_lstm(
            seq_len=seq_len,
            num_features=num_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout,
            output_size=output_size,
        ).to(device)
    elif model_name == "residual":
        model = build_residual_mlp(
            seq_len=1 if task == "classification" else seq_len,
            num_features=num_features,
            hidden_size=residual_hidden_size,
            num_blocks=residual_num_blocks,
            dropout=dropout,
            output_size=output_size,
        ).to(device)
    else:
        # Default: shallow MLP
        model = build_mlp(
            seq_len=1 if task == "classification" else seq_len,
            num_features=num_features,
            hidden_sizes=mlp_hidden_sizes or (128, 64),
            dropout=dropout,
            output_size=output_size,
        ).to(device)

    loss_fn = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_grad_norm": [],
        "ece": [],
        "learning_rate": [],
        "val_rmse": [],
        "val_r2": [],
    }
    if task == "classification":
        history.update(
            {
                "train_accuracy": [],
                "val_accuracy": [],
                "val_precision": [],
                "val_recall": [],
                "val_f1": [],
                "val_brier": [],
                "val_avg_confidence": [],
                "val_confidence_correct": [],
                "val_confidence_incorrect": [],
            }
        )

    # Logging backends -----------------------------------------------------
    writer = None
    use_wandb = False

    log_backend = (log_backend or "none").lower()

    if log_backend == "tensorboard":
        if not TENSORBOARD_AVAILABLE:
            print("TensorBoard is not available; proceeding without it.")
        else:
            # Organize logs per model type.
            tb_log_dir = os.path.join(log_dir, "tensorboard", model_name)
            os.makedirs(tb_log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_log_dir)
            print(f"Logging to TensorBoard at {tb_log_dir}")
    elif log_backend == "wandb":
        if not WANDB_AVAILABLE:
            print("Weights & Biases is not available; proceeding without it.")
        else:
            if wandb_project is None:
                wandb_project = "calibration-confidence"
            config = {
                "model": model_name,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "lr": lr,
                "val_ratio": val_ratio,
            }
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=config,
            )
            use_wandb = True
            print(f"Logging to Weights & Biases project '{wandb_project}'")

    # Model checkpointing / resume support
    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if checkpoint_path is not None and resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        ckpt_model = ckpt.get("model_name")
        if ckpt_model == model_name:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            history = ckpt.get("history", history)
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("best_val_loss", best_val_loss)
            print(f"Resuming from epoch {start_epoch} (best_val_loss={best_val_loss:.6f})")
        else:
            print(f"Checkpoint is for model '{ckpt_model}', current is '{model_name}'; starting fresh.")

    start_time = time.perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_grad_norm, train_accuracy = train_epoch(
            model, train_loader, loss_fn, optimizer, device, task, model_name
        )
        if task == "classification":
            val_metrics = eval_epoch_classification(
                model, val_loader, loss_fn, device, model_name
            )
            val_loss = val_metrics["loss"]
            reg_ece = val_metrics["ece"]
            val_accuracy = val_metrics["accuracy"]
        else:
            val_metrics = eval_epoch_regression(model, val_loader, loss_fn, device)
            val_loss = val_metrics["loss"]
            reg_ece = val_metrics["ece"]
            val_accuracy = None
            history["val_rmse"].append(val_metrics.get("rmse", float("nan")))
            history["val_r2"].append(val_metrics.get("r2", float("nan")))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_grad_norm"].append(train_grad_norm)
        history["ece"].append(reg_ece)
        if task == "classification":
            history["train_accuracy"].append(train_accuracy or 0.0)
            history["val_accuracy"].append(val_accuracy or 0.0)
            history["val_precision"].append(val_metrics["precision"])
            history["val_recall"].append(val_metrics["recall"])
            history["val_f1"].append(val_metrics["f1"])
            history["val_brier"].append(val_metrics["brier"])
            history["val_avg_confidence"].append(val_metrics["avg_confidence"])
            history["val_confidence_correct"].append(val_metrics["confidence_correct"])
            history["val_confidence_incorrect"].append(val_metrics["confidence_incorrect"])
        else:
            if len(history["val_rmse"]) < len(history["train_loss"]):
                history["val_rmse"].append(float("nan"))
            if len(history["val_r2"]) < len(history["train_loss"]):
                history["val_r2"].append(float("nan"))
        current_lr = float(optimizer.param_groups[0].get("lr", lr))
        history["learning_rate"].append(current_lr)
        msg = f"Epoch {epoch:3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | ECE={reg_ece:.4f}"
        if task == "classification":
            msg += f" | acc={val_accuracy:.3f}"
        print(msg)

        # Scalar logging ---------------------------------------------------
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("grad/mean_norm", train_grad_norm, epoch)
            writer.add_scalar("lr/learning_rate", current_lr, epoch)
            if task == "classification":
                writer.add_scalar("accuracy/train", train_accuracy, epoch)
                writer.add_scalar("accuracy/val", val_accuracy, epoch)

        if use_wandb:
            log_data = {
                "epoch": epoch,
                "loss/train": train_loss,
                "loss/val": val_loss,
                "grad/mean_norm": train_grad_norm,
                "lr": current_lr,
            }
            if task == "classification":
                log_data["accuracy/train"] = train_accuracy
                log_data["accuracy/val"] = val_accuracy
            wandb.log(log_data)

        # Checkpointing on improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                dirpath = os.path.dirname(checkpoint_path)
                if dirpath:
                    os.makedirs(dirpath, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_name": model_name,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "history": history,
                        "best_val_loss": best_val_loss,
                    },
                    checkpoint_path,
                )
                print(f"Saved new best checkpoint to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                print(
                    f"Early stopping: no improvement in val loss for "
                    f"{epochs_without_improvement} epochs."
                )
                break

    training_time = time.perf_counter() - start_time
    history["training_time"] = [training_time]
    history["num_parameters"] = [sum(p.numel() for p in model.parameters())]

    if results_path:
        if task == "classification":
            collect_classification_results(
                model=model,
                loader=val_loader,
                device=device,
                model_name=model_name,
                save_path=results_path,
            )
        else:
            collect_results(
                model=model,
                loader=val_loader,
                device=device,
                save_path=results_path,
                history=history,
            )

    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a windowed model on FinSen data")
    parser.add_argument("--data-path", type=str, default="data/finsen/raw", help="Path to FinSen raw CSV files")
    parser.add_argument("--seq-len", type=int, default=50, help="Sequence length (window size)")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of data to use for validation")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "deep", "rnn", "lstm", "residual"], help="Model type: 'mlp', 'deep', 'rnn', 'lstm', 'residual'")
    parser.add_argument("--task", type=str, default="regression", choices=["regression", "classification"], help="Task type")
    parser.add_argument("--train-path", type=str, default="data/finsen/processed/train.csv", help="Train split CSV (classification)")
    parser.add_argument("--val-path", type=str, default="data/finsen/processed/val.csv", help="Validation split CSV (classification)")
    parser.add_argument("--test-path", type=str, default="data/finsen/processed/test.csv", help="Test split CSV (classification)")
    parser.add_argument("--max-features", type=int, default=2000, help="TF-IDF max features for MLP-like models")
    parser.add_argument("--max-seq-len", type=int, default=128, help="Max sequence length for RNN/LSTM inputs")
    parser.add_argument("--split-seed", type=int, default=None, help="Random seed for train/val split")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--mlp-hidden-sizes", type=str, default=None, help="Comma-separated hidden sizes for MLP (e.g. 128,64)")
    parser.add_argument("--deep-hidden-sizes", type=str, default=None, help="Comma-separated hidden sizes for deep MLP")
    parser.add_argument("--rnn-hidden-size", type=int, default=64, help="Hidden size for vanilla RNN")
    parser.add_argument("--rnn-num-layers", type=int, default=1, help="Number of layers for vanilla RNN")
    parser.add_argument("--lstm-hidden-size", type=int, default=64, help="Hidden size for LSTM")
    parser.add_argument("--lstm-num-layers", type=int, default=1, help="Number of layers for LSTM")
    parser.add_argument("--residual-hidden-size", type=int, default=128, help="Hidden size for residual MLP blocks")
    parser.add_argument("--residual-num-blocks", type=int, default=3, help="Number of residual blocks")
    parser.add_argument("--checkpoint-path", type=str, default="results/checkpoints/model.pt", help="Path to save model checkpoints")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from an existing checkpoint")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Early stopping patience in epochs")
    parser.add_argument("--log-backend", type=str, default="none", choices=["none", "tensorboard", "wandb"], help="Logging backend for metrics")
    parser.add_argument("--log-dir", type=str, default="results/logs", help="Base directory for TensorBoard logs")
    parser.add_argument("--wandb-project", type=str, default=None, help="Weights & Biases project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--results-path", type=str, default="results/first_results.npz", help="Path to save detailed predictions/confidences/ECE")
    args = parser.parse_args()

    def _parse_sizes(raw: str | None) -> Tuple[int, ...] | None:
        if raw is None:
            return None
        items = [s.strip() for s in raw.split(",") if s.strip()]
        if not items:
            return None
        return tuple(int(s) for s in items)

    # Call the unified training loop. We ignore the returned objects here,
    # but other experiment scripts can reuse ``train_model`` directly to
    # get the trained model and per-epoch loss history.
    train_model(
        model_name=args.model,
        data_path=args.data_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        task=args.task,
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        max_features=args.max_features,
        max_seq_len=args.max_seq_len,
        mlp_hidden_sizes=_parse_sizes(args.mlp_hidden_sizes),
        deep_hidden_sizes=_parse_sizes(args.deep_hidden_sizes),
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_num_layers=args.rnn_num_layers,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
        residual_hidden_size=args.residual_hidden_size,
        residual_num_blocks=args.residual_num_blocks,
        dropout=args.dropout,
        checkpoint_path=args.checkpoint_path,
        resume=not args.no_resume,
        early_stopping_patience=args.early_stopping_patience,
        log_backend=args.log_backend,
        log_dir=args.log_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        results_path=args.results_path,
    )


if __name__ == "__main__":
    main()
