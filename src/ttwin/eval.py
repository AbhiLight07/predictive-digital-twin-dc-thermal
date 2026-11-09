"""Evaluation orchestration module for the thermal twin project.

Aggregates results across experiments, generates summary tables and plots.
"""

import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent.parent / "experiments"


def load_all_metrics(experiments_dir: Path = None) -> pd.DataFrame:
    """Load metrics.csv from all experiments and concatenate."""
    if experiments_dir is None:
        experiments_dir = EXPERIMENTS_DIR
    rows = []
    for metrics_file in sorted(experiments_dir.glob("*/metrics.csv")):
        df = pd.read_csv(metrics_file)
        # Parse config from experiment directory name
        parts = metrics_file.parent.name.split("_")
        df["exp_id"] = metrics_file.parent.name
        df["dataset"] = parts[0] if len(parts) > 0 else "?"
        df["split_type"] = parts[1] if len(parts) > 1 else "?"
        df["feature_set"] = parts[2] if len(parts) > 2 else "?"
        df["model"] = parts[3] if len(parts) > 3 else "?"
        df["horizon"] = parts[4] if len(parts) > 4 else "?"
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def make_ablation_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Create the main ablation table: feature_set x horizon x MAE/skill."""
    cols = ["feature_set", "model", "horizon", "mae_global", "rmse_global"]
    available = [c for c in cols if c in metrics_df.columns]
    tbl = (
        metrics_df[available]
        .dropna(subset=["mae_global"])
        .sort_values(["feature_set", "horizon"])
        .reset_index(drop=True)
    )
    return tbl


def compute_skill_scores(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Add skill_score column: 1 - MAE_model / MAE_persistence."""
    persistence = (
        metrics_df[metrics_df["feature_set"] == "S0"][["horizon", "mae_global"]]
        .rename(columns={"mae_global": "mae_persistence"})
    )
    merged = metrics_df.merge(persistence, on="horizon", how="left")
    merged["skill_score"] = 1 - merged["mae_global"] / merged["mae_persistence"]
    return merged


def plot_ablation_mae(metrics_df: pd.DataFrame, output_path: Path = None) -> Path:
    """Bar chart: MAE by feature set and horizon."""
    df = metrics_df[metrics_df["feature_set"] != "S0"].copy()
    df = df[df["split_label"] == "A"] if "split_label" in df.columns else df

    # Horizon display order
    horizon_order = ["10min", "1h", "6h"]
    fs_order = ["S1", "S2", "S3", "S4"]
    df = df[df["horizon"].isin(horizon_order) & df["feature_set"].isin(fs_order)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    palette = sns.color_palette("husl", len(fs_order))

    for ax, h in zip(axes, horizon_order):
        sub = df[df["horizon"] == h]
        # Add persistence baseline
        pers = metrics_df[(metrics_df["feature_set"] == "S0") & (metrics_df["horizon"] == h)]
        pers_mae = pers["mae_global"].values[0] if len(pers) > 0 else None

        bar_data = sub.groupby("feature_set")["mae_global"].mean().reindex(fs_order).dropna()
        bars = ax.bar(bar_data.index, bar_data.values, color=palette[:len(bar_data)], alpha=0.85, edgecolor="white")

        if pers_mae is not None:
            ax.axhline(pers_mae, color="crimson", linestyle="--", linewidth=1.5, label=f"Persistence ({pers_mae:.2f}°C)")
            ax.legend(fontsize=8)

        ax.set_title(f"Horizon: {h}", fontweight="bold")
        ax.set_xlabel("Feature Set")
        ax.set_ylabel("MAE (°C)")
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Ablation: MAE by Feature Set (Dataset 1, Split A)", fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path is None:
        output_path = EXPERIMENTS_DIR / "ablation_mae.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_per_server_mae_distribution(
    metrics_df: pd.DataFrame,
    feature_set: str = "S4",
    horizon: str = "1h",
    output_path: Path = None,
) -> Path:
    """Box plots of per-server MAE across servers for a given feature_set/horizon."""
    subset = metrics_df[
        (metrics_df["feature_set"] == feature_set) & (metrics_df["horizon"] == horizon)
    ]
    if len(subset) == 0:
        raise ValueError(f"No data for {feature_set}/{horizon}")

    # Load per-server results
    per_server_rows = []
    for exp_id in subset["exp_id"].unique():
        results_path = EXPERIMENTS_DIR / exp_id / "results.json"
        if not results_path.exists():
            continue
        with open(results_path) as f:
            results = json.load(f)
        for split in results.get("splits", []):
            ps_summary = split.get("per_server_summary", {})
            if ps_summary:
                per_server_rows.append({
                    "exp_id": exp_id,
                    "split_label": split.get("split_label", "?"),
                    "ps_mae_mean": ps_summary.get("mean"),
                    "ps_mae_std": ps_summary.get("std"),
                    "ps_mae_worst_decile": ps_summary.get("worst_decile"),
                })

    if not per_server_rows:
        raise ValueError("No per-server data found")

    df = pd.DataFrame(per_server_rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(df)), df["ps_mae_mean"], yerr=df["ps_mae_std"],
           capsize=4, color=sns.color_palette("husl")[2], alpha=0.8, edgecolor="white")
    ax.set_xlabel("Experiment Split")
    ax.set_ylabel("Per-server mean MAE (°C)")
    ax.set_title(f"{feature_set} at {horizon}: Per-server MAE distribution", fontweight="bold")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["split_label"], rotation=30, ha="right")
    plt.tight_layout()

    if output_path is None:
        output_path = EXPERIMENTS_DIR / f"per_server_{feature_set}_{horizon}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def print_results_table(metrics_df: pd.DataFrame) -> None:
    """Print a formatted results table to stdout."""
    cols = ["feature_set", "model", "horizon", "mae_global", "rmse_global"]
    available = [c for c in cols if c in metrics_df.columns]
    if "split_label" in metrics_df.columns:
        available = ["split_label"] + available

    tbl = metrics_df[available].dropna(subset=["mae_global"])
    tbl = tbl.sort_values(["horizon", "feature_set"])

    print(f"\n{'='*70}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*70}")
    print(tbl.to_string(index=False, float_format="%.3f"))
    print(f"{'='*70}\n")
