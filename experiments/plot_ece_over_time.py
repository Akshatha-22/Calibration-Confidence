"""
Plot ECE (and losses) over time from saved results.

Run after training to visualize calibration over epochs. For RNN/LSTM we often
expect ECE (regression calibration error) to decay over time as the model
calibrates; MLPs may show different patterns.

Usage:
    python experiments/plot_ece_over_time.py results/first_results.npz
    python experiments/plot_ece_over_time.py results/mlp_first_results.npz results/rnn_first_results.npz --labels mlp rnn
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt


def load_history(npz_path: str) -> dict:
    """Load epoch-wise arrays from a results npz (from train.py --results-path)."""
    data = np.load(npz_path)
    out = {}
    if "epochs" in data:
        out["epochs"] = np.asarray(data["epochs"]).ravel()
    elif "train_loss" in data:
        t = np.asarray(data["train_loss"]).ravel()
        out["epochs"] = np.arange(1, len(t) + 1, dtype=np.int32)
    else:
        return out
    for key in ("train_loss", "val_loss", "ece_over_time", "train_grad_norm"):
        if key in data:
            out[key] = np.asarray(data[key]).ravel()
    for key in ("learning_rate", "train_accuracy", "val_accuracy"):
        if key in data:
            out[key] = np.asarray(data[key]).ravel()
    return out


def plot_ece_over_time(
    npz_paths: list[str],
    labels: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    """Plot loss, ECE, (optional) accuracy, and (optional) LR over epochs."""
    if labels is None:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in npz_paths]
    if len(labels) != len(npz_paths):
        labels = [f"run_{i}" for i in range(len(npz_paths))]

    histories: list[dict] = []
    for npz_path in npz_paths:
        if not os.path.isfile(npz_path):
            continue
        hist = load_history(npz_path)
        histories.append(hist or {})

    plot_accuracy = any(
        hist.get("train_accuracy") is not None and hist.get("val_accuracy") is not None
        for hist in histories
    )
    plot_lr = any("learning_rate" in hist for hist in histories)

    rows = 2 + int(plot_accuracy) + int(plot_lr)
    fig, axes = plt.subplots(rows, 1, figsize=(8, 3 * rows), sharex=True)
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = list(np.atleast_1d(axes))
    loss_ax, ece_ax = axes[0], axes[1]
    acc_ax = axes[2] if plot_accuracy else None
    lr_ax = axes[2 + int(plot_accuracy)] if plot_lr else None

    for npz_path, label in zip(npz_paths, labels):
        if not os.path.isfile(npz_path):
            print(f"Warning: not found {npz_path}, skipping.")
            continue
        hist = load_history(npz_path)
        if not hist or "epochs" not in hist:
            print(f"Warning: no epoch history in {npz_path}, skipping.")
            continue
        epochs = hist["epochs"]
        if "val_loss" in hist:
            loss_ax.plot(epochs, hist["val_loss"], label=label, marker="o", markersize=3)
        if "ece_over_time" in hist:
            ece_ax.plot(epochs, hist["ece_over_time"], label=label, marker="s", markersize=3)
        if plot_accuracy and acc_ax is not None:
            if "train_accuracy" in hist and "val_accuracy" in hist:
                acc_ax.plot(epochs, hist["train_accuracy"], label=f"{label} train", linestyle="--", markersize=3)
                acc_ax.plot(epochs, hist["val_accuracy"], label=f"{label} val", linestyle="-", markersize=3)
        if plot_lr and lr_ax is not None and "learning_rate" in hist:
            lr_ax.plot(epochs, hist["learning_rate"], label=label, marker="^", markersize=3)

    loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Validation loss over time")
    loss_ax.legend(loc="best", fontsize=8)
    loss_ax.grid(True, alpha=0.3)

    ece_ax.set_ylabel("ECE")
    ece_ax.set_title("ECE over time (lower = better calibrated)")
    ece_ax.legend(loc="best", fontsize=8)
    ece_ax.grid(True, alpha=0.3)

    if acc_ax is not None:
        acc_ax.set_ylabel("Accuracy")
        acc_ax.set_title("Accuracy over time")
        acc_ax.legend(loc="best", fontsize=8)
        acc_ax.grid(True, alpha=0.3)
    if lr_ax is not None:
        lr_ax.set_ylabel("Learning rate")
        lr_ax.set_title("Learning rate schedule")
        lr_ax.legend(loc="best", fontsize=8)
        lr_ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Epoch")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ECE and losses over time from results npz")
    parser.add_argument("npz_paths", nargs="+", help="Paths to results .npz files")
    parser.add_argument("--labels", nargs="+", default=None, help="Labels for each file (default: filename)")
    parser.add_argument("--save", type=str, default="results/figures/ece_over_time.png", help="Save figure path")
    args = parser.parse_args()
    plot_ece_over_time(args.npz_paths, labels=args.labels, save_path=args.save)


if __name__ == "__main__":
    main()
