"""Hyperparameter tuning script for model variants.

Supports grid search, random search, and Optuna (if installed).
Results are saved to CSV/JSONL under results/tuning by default.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

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

from experiments.train import train_model


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
    result = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_val_loss": final_val_loss,
        "final_train_loss": final_train_loss,
        "final_ece": final_ece,
    }
    if history.get("val_accuracy"):
        result["final_val_accuracy"] = float(history["val_accuracy"][-1])
    if history.get("train_accuracy"):
        result["final_train_accuracy"] = float(history["train_accuracy"][-1])
    return result


def run_trial(
    trial_id: int,
    cfg: TrialConfig,
    data_path: str,
    checkpoint_dir: str,
    results_dir: str,
    early_stopping_patience: int,
    log_backend: str,
    log_dir: str,
    wandb_project: str | None,
) -> Dict[str, object]:
    checkpoint_path = os.path.join(checkpoint_dir, f"{cfg.model}_trial_{trial_id}.pt")
    trial_results_path = os.path.join(results_dir, f"trial_{trial_id}", "results.npz")

    model, history = train_model(
        model_name=cfg.model,
        data_path=data_path,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        lr=cfg.lr,
        val_ratio=cfg.val_ratio,
        split_seed=cfg.split_seed,
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
    record: Dict[str, object] = {
        "trial_id": trial_id,
        **cfg.to_dict(),
        **summary,
    }
    return record


def save_results(records: List[Dict[str, object]], out_csv: str, out_jsonl: str) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
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
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "deep", "rnn", "lstm", "residual"])
    parser.add_argument("--data-path", type=str, default="data/finsen/raw")
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.0)
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
    parser.add_argument("--results-dir", type=str, default="results/tuning")
    args = parser.parse_args()

    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.results_dir, args.model, f"{args.method}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    base = {
        "seq_len": args.seq_len,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "val_ratio": args.val_ratio,
        "split_seed": args.split_seed,
        "dropout": args.dropout,
        "rnn_num_layers": args.rnn_num_layers,
        "lstm_num_layers": args.lstm_num_layers,
        "residual_num_blocks": args.residual_num_blocks,
        "hidden_size": 128,
        "depth": 4,
        "deep_base_hidden": args.deep_base_hidden,
    }

    records: List[Dict[str, object]] = []

    if args.method == "grid":
        configs = grid_configs(args.model, base)
        for idx, cfg in enumerate(configs, start=1):
            rec = run_trial(
                trial_id=idx,
                cfg=cfg,
                data_path=args.data_path,
                checkpoint_dir=checkpoint_dir,
                results_dir=run_dir,
                early_stopping_patience=args.early_stopping_patience,
                log_backend=args.log_backend,
                log_dir=args.log_dir,
                wandb_project=args.wandb_project,
            )
            records.append(rec)
    elif args.method == "random":
        configs = grid_configs(args.model, base)
        random.shuffle(configs)
        configs = configs[: args.n_trials]
        for idx, cfg in enumerate(configs, start=1):
            rec = run_trial(
                trial_id=idx,
                cfg=cfg,
                data_path=args.data_path,
                checkpoint_dir=checkpoint_dir,
                results_dir=run_dir,
                early_stopping_patience=args.early_stopping_patience,
                log_backend=args.log_backend,
                log_dir=args.log_dir,
                wandb_project=args.wandb_project,
            )
            records.append(rec)
    else:
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna is not available. Install optuna or choose grid/random.")

        def objective(trial: "optuna.trial.Trial") -> float:
            params: Dict[str, object] = {}
            space = build_search_space(args.model)
            if "hidden_size" in space:
                params["hidden_size"] = trial.suggest_categorical("hidden_size", list(space["hidden_size"]))
            if "depth" in space:
                params["depth"] = trial.suggest_categorical("depth", list(space["depth"]))
            if "seq_len" in space:
                params["seq_len"] = trial.suggest_categorical("seq_len", list(space["seq_len"]))
            if "lr" in space:
                params["lr"] = trial.suggest_categorical("lr", list(space["lr"]))
            cfg = make_config(args.model, base, params)
            rec = run_trial(
                trial_id=trial.number + 1,
                cfg=cfg,
                data_path=args.data_path,
                checkpoint_dir=checkpoint_dir,
                results_dir=run_dir,
                early_stopping_patience=args.early_stopping_patience,
                log_backend=args.log_backend,
                log_dir=args.log_dir,
                wandb_project=args.wandb_project,
            )
            records.append(rec)
            return float(rec["best_val_loss"])

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=args.n_trials)

    out_csv = os.path.join(run_dir, "tuning_results.csv")
    out_jsonl = os.path.join(run_dir, "tuning_results.jsonl")
    save_results(records, out_csv, out_jsonl)

    if records:
        best = min(records, key=lambda r: float(r["best_val_loss"]))
        print("Best trial:")
        for key, val in best.items():
            print(f"  {key}: {val}")
        print(f"Saved results to: {out_csv}")


if __name__ == "__main__":
    main()
