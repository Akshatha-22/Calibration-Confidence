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
from models.mlp import build_mlp
from models.deep_mlp import build_deep_mlp
from models.vanilla_rnn import build_vanilla_rnn
from models.lstm import build_lstm
from models.residual_mlp import build_residual_mlp
from calibration.ece import expected_calibration_error, regression_calibration_error


def split_dataset(dataset: FinSenDataset, val_ratio: float = 0.2) -> Tuple[Subset, Subset]:
    """Split dataset into train and validation subsets."""
    n = len(dataset)
    val_size = int(n * val_ratio)
    train_size = n - val_size
    indices = torch.randperm(n).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    num_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()

        # Aggregate global gradient norm across all parameters for this batch.
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total += float(param_norm.item()) ** 2
        batch_grad_norm = float(total**0.5)

        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_grad_norm += batch_grad_norm
        num_batches += 1

    mean_grad_norm = total_grad_norm / max(num_batches, 1)

    return total_loss / len(loader.dataset), mean_grad_norm


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    n_bins: int = 10,
) -> Tuple[float, float]:
    """Returns (mean_val_loss, regression_calibration_error)."""
    model.eval()
    total_loss = 0.0
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = loss_fn(preds, y)
            if torch.isfinite(loss).all():
                total_loss += loss.item() * x.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    n = len(loader.dataset)
    val_loss = total_loss / max(n, 1)
    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    reg_ece = regression_calibration_error(preds_arr, targets_arr, n_bins=n_bins)
    return val_loss, float(reg_ece)


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

    dataset = FinSenDataset(data_path=data_path, seq_length=seq_len)
    train_ds, val_ds = split_dataset(dataset, val_ratio=val_ratio)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    num_features = dataset.values.shape[1]

    if model_name == "deep":
        model = build_deep_mlp(
            seq_len=seq_len,
            num_features=num_features,
            hidden_sizes=(256, 128, 64, 32),
            dropout=0.0,
        ).to(device)
    elif model_name == "rnn":
        model = build_vanilla_rnn(
            seq_len=seq_len,
            num_features=num_features,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
        ).to(device)
    elif model_name == "lstm":
        model = build_lstm(
            seq_len=seq_len,
            num_features=num_features,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
        ).to(device)
    elif model_name == "residual":
        model = build_residual_mlp(
            seq_len=seq_len,
            num_features=num_features,
            hidden_size=128,
            num_blocks=3,
            dropout=0.0,
        ).to(device)
    else:
        # Default: shallow MLP
        model = build_mlp(
            seq_len=seq_len,
            num_features=num_features,
            hidden_sizes=(128, 64),
            dropout=0.0,
        ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_grad_norm": [],
        "ece": [],
    }

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

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_grad_norm = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, reg_ece = eval_epoch(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_grad_norm"].append(train_grad_norm)
        history["ece"].append(reg_ece)
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | ECE={reg_ece:.4f}")

        # Scalar logging ---------------------------------------------------
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("grad/mean_norm", train_grad_norm, epoch)

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "loss/train": train_loss,
                    "loss/val": val_loss,
                    "grad/mean_norm": train_grad_norm,
                }
            )

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

    # Close logging backends
    if writer is not None:
        writer.close()
    if use_wandb:
        wandb.finish()

    # Optionally collect detailed results on the validation set for analysis.
    if results_path is not None:
        print(f"Collecting detailed results on validation set to {results_path}")
        collect_results(model, val_loader, device, save_path=results_path, history=history)

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
    parser.add_argument("--checkpoint-path", type=str, default="results/checkpoints/model.pt", help="Path to save model checkpoints")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from an existing checkpoint")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Early stopping patience in epochs")
    parser.add_argument("--log-backend", type=str, default="none", choices=["none", "tensorboard", "wandb"], help="Logging backend for metrics")
    parser.add_argument("--log-dir", type=str, default="results/logs", help="Base directory for TensorBoard logs")
    parser.add_argument("--wandb-project", type=str, default=None, help="Weights & Biases project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--results-path", type=str, default="results/first_results.npz", help="Path to save detailed predictions/confidences/ECE")
    args = parser.parse_args()

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
