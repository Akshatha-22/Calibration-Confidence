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
) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = loss_fn(preds, y)
            total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


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
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        history = ckpt.get("history", history)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        print(f"Resuming from epoch {start_epoch} (best_val_loss={best_val_loss:.6f})")

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_grad_norm = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = eval_epoch(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_grad_norm"].append(train_grad_norm)
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

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
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
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
    )


if __name__ == "__main__":
    main()
