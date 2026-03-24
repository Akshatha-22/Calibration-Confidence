"""Sensitivity visualization utilities for hyperparameter tuning results.

This script loads per-model tuning CSVs and writes a set of plots that show
how performance changes across hyperparameters.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from sklearn.ensemble import RandomForestRegressor

MODEL_DIRS = {
    "mlp": "mlp",
    "deep_mlp": "deep_mlp",
    "vanilla_rnn": "vanilla_rnn",
    "lstm": "lstm",
    "residual_mlp": "residual_mlp",
}


def _parse_list_field(value: object) -> List[int] | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
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


def _extract_hidden_size(row: pd.Series) -> Optional[int]:
    mlp_sizes = _parse_list_field(row.get("mlp_hidden_sizes"))
    if mlp_sizes:
        return int(mlp_sizes[0])
    for key in ("rnn_hidden_size", "lstm_hidden_size", "residual_hidden_size"):
        val = row.get(key)
        if pd.notna(val):
            return int(float(val))
    return None


def _extract_depth(row: pd.Series) -> Optional[int]:
    deep_sizes = _parse_list_field(row.get("deep_hidden_sizes"))
    if deep_sizes:
        return int(len(deep_sizes))
    return None


def _ensure_metric(df: pd.DataFrame, metric: str) -> None:
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in columns: {list(df.columns)}")


def _metric_better_is_lower(metric: str) -> bool:
    return "loss" in metric or "ece" in metric


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hidden_size"] = df.apply(_extract_hidden_size, axis=1)
    df["depth"] = df.apply(_extract_depth, axis=1)
    df["lr"] = pd.to_numeric(df["lr"], errors="coerce")
    return df


def _select_hyperparams(df: pd.DataFrame) -> List[str]:
    cols = []
    if df["hidden_size"].notna().any():
        cols.append("hidden_size")
    if df["depth"].notna().any():
        cols.append("depth")
    if df["lr"].notna().any():
        cols.append("lr")
    return cols


def _save_plot(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_heatmap(df: pd.DataFrame, metric: str, out_path: str) -> None:
    if df["hidden_size"].isna().all() or df["lr"].isna().all():
        return
    pivot = df.pivot_table(index="hidden_size", columns="lr", values=metric, aggfunc="mean")
    plt.figure(figsize=(7, 5))
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(label=metric)
    plt.xticks(range(len(pivot.columns)), [f"{v:.4g}" for v in pivot.columns], rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Learning rate")
    plt.ylabel("Hidden size")
    plt.title("Heatmap: performance across hyperparameters")
    _save_plot(out_path)


def plot_hidden_size_effect(df: pd.DataFrame, metric: str, out_path: str) -> None:
    if df["hidden_size"].isna().all():
        return
    grouped = df.groupby("hidden_size")[metric].mean().reset_index()
    plt.figure(figsize=(7, 4))
    plt.plot(grouped["hidden_size"], grouped[metric], marker="o")
    plt.xlabel("Hidden size")
    plt.ylabel(metric)
    plt.title("Hidden size effect")
    plt.grid(True, alpha=0.3)
    _save_plot(out_path)


def plot_learning_rate_curves(model_dfs: Dict[str, pd.DataFrame], metric: str, out_path: str) -> None:
    plt.figure(figsize=(7, 4))
    for model_name, df in model_dfs.items():
        if df["lr"].isna().all():
            continue
        grouped = df.groupby("lr")[metric].mean().reset_index()
        plt.plot(grouped["lr"], grouped[metric], marker="o", label=model_name)
    plt.xlabel("Learning rate")
    plt.ylabel(metric)
    plt.title("Learning rate sensitivity")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    _save_plot(out_path)


def plot_parallel_coordinates(df: pd.DataFrame, metric: str, out_path: str) -> None:
    hp_cols = _select_hyperparams(df)
    if not hp_cols:
        return
    tmp = df[hp_cols + [metric]].copy()
    for col in hp_cols + [metric]:
        col_min = tmp[col].min()
        col_max = tmp[col].max()
        if col_max != col_min:
            tmp[col] = (tmp[col] - col_min) / (col_max - col_min)
        else:
            tmp[col] = 0.0
    tmp["performance_bin"] = pd.qcut(df[metric], q=3, labels=["low", "mid", "high"])
    plt.figure(figsize=(8, 4))
    parallel_coordinates(tmp, "performance_bin", color=["#d62728", "#ff7f0e", "#2ca02c"])
    plt.title("Parallel coordinates (normalized)")
    _save_plot(out_path)


def plot_sensitivity_ranking(df: pd.DataFrame, metric: str, out_path: str) -> None:
    hp_cols = _select_hyperparams(df)
    if not hp_cols:
        return
    X = df[hp_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = df[metric]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(6, 4))
    plt.bar([hp_cols[i] for i in order], importances[order])
    plt.ylabel("Importance")
    plt.title("Sensitivity ranking")
    _save_plot(out_path)


def plot_best_vs_worst(df: pd.DataFrame, metric: str, out_path: str) -> None:
    better_is_lower = _metric_better_is_lower(metric)
    best_idx = df[metric].idxmin() if better_is_lower else df[metric].idxmax()
    worst_idx = df[metric].idxmax() if better_is_lower else df[metric].idxmin()
    best = df.loc[best_idx]
    worst = df.loc[worst_idx]

    labels = ["Best", "Worst"]
    values = [best[metric], worst[metric]]
    plt.figure(figsize=(5, 4))
    plt.bar(labels, values, color=["#2ca02c", "#d62728"])
    plt.ylabel(metric)
    plt.title("Best vs worst configuration")
    _save_plot(out_path)


def plot_correlation_matrix(df: pd.DataFrame, metric: str, out_path: str) -> None:
    hp_cols = _select_hyperparams(df)
    if not hp_cols:
        return
    corr = df[hp_cols + [metric]].corr()
    plt.figure(figsize=(6, 5))
    plt.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation matrix")
    _save_plot(out_path)


def load_model_csv(input_dir: str, model_key: str) -> pd.DataFrame:
    model_dir = MODEL_DIRS[model_key]
    path = os.path.join(input_dir, model_dir, "results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing results CSV: {path}")
    df = pd.read_csv(path)
    df = _standardize_columns(df)
    return df


def _metric_direction(metric: str) -> str:
    return "lower_is_better" if _metric_better_is_lower(metric) else "higher_is_better"


def _summarize_model(df: pd.DataFrame, model_key: str, metric: str) -> Dict[str, object]:
    better_is_lower = _metric_better_is_lower(metric)
    best_idx = df[metric].idxmin() if better_is_lower else df[metric].idxmax()
    worst_idx = df[metric].idxmax() if better_is_lower else df[metric].idxmin()

    best = df.loc[best_idx]
    worst = df.loc[worst_idx]

    hp_cols = _select_hyperparams(df)
    X = df[hp_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = df[metric]
    if hp_cols:
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        top_params = [hp_cols[i] for i in order]
        top_scores = [float(importances[i]) for i in order]
    else:
        top_params = []
        top_scores = []

    summary: Dict[str, object] = {
        "model": model_key,
        "metric": metric,
        "metric_direction": _metric_direction(metric),
        "num_trials": int(len(df)),
        "best_value": float(best[metric]),
        "worst_value": float(worst[metric]),
        "best_run_name": best.get("run_name"),
        "worst_run_name": worst.get("run_name"),
        "best_hidden_size": best.get("hidden_size"),
        "best_depth": best.get("depth"),
        "best_lr": best.get("lr"),
        "worst_hidden_size": worst.get("hidden_size"),
        "worst_depth": worst.get("depth"),
        "worst_lr": worst.get("lr"),
    }

    for i in range(3):
        key_name = f"top_param_{i + 1}"
        key_score = f"top_param_{i + 1}_importance"
        summary[key_name] = top_params[i] if i < len(top_params) else None
        summary[key_score] = top_scores[i] if i < len(top_scores) else None

    return summary


def write_summary_table(model_dfs: Dict[str, pd.DataFrame], metric: str, out_path: str) -> None:
    rows = [_summarize_model(df, model_key, metric) for model_key, df in model_dfs.items()]
    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sensitivity plots from tuning CSVs")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="results/hyperparameter_tuning",
        help="Base directory containing per-model results.csv files.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="mlp,deep_mlp,vanilla_rnn,lstm,residual_mlp",
        help="Comma-separated model list.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="final_ece",
        help="Metric column to plot (e.g., final_ece, final_val_accuracy, final_val_loss).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/sensitivity_plots",
        help="Output directory for plots.",
    )
    args = parser.parse_args()

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    model_dfs: Dict[str, pd.DataFrame] = {}
    for model_key in model_keys:
        df = load_model_csv(args.input_dir, model_key)
        _ensure_metric(df, args.metric)
        model_dfs[model_key] = df

    for model_key, df in model_dfs.items():
        model_out = os.path.join(args.out_dir, model_key)
        plot_heatmap(df, args.metric, os.path.join(model_out, "plot1_heatmap.png"))
        plot_hidden_size_effect(df, args.metric, os.path.join(model_out, "plot2_hidden_size.png"))
        plot_parallel_coordinates(df, args.metric, os.path.join(model_out, "plot4_parallel_coords.png"))
        plot_sensitivity_ranking(df, args.metric, os.path.join(model_out, "plot5_sensitivity_rank.png"))
        plot_best_vs_worst(df, args.metric, os.path.join(model_out, "plot6_best_vs_worst.png"))
        plot_correlation_matrix(df, args.metric, os.path.join(model_out, "plot7_correlation.png"))

    plot_learning_rate_curves(model_dfs, args.metric, os.path.join(args.out_dir, "plot3_lr_curves.png"))
    write_summary_table(model_dfs, args.metric, os.path.join(args.out_dir, "sensitivity_summary.csv"))


if __name__ == "__main__":
    main()
