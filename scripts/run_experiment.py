#!/usr/bin/env python3
"""Run an experiment from a YAML config file or command-line arguments.

Usage:
    python scripts/run_experiment.py --config configs/experiments/<exp>.yaml
    python scripts/run_experiment.py --dataset dataset1 --feature-set S2 --model lightgbm --split A --horizon 1

This script:
  1. Loads data from Parquet
  2. Engineers features
  3. Splits data according to protocol
  4. Trains model
  5. Evaluates and saves results to experiments/<exp_id>/
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import yaml

from ttwin.io import load_parquet, load_all_servers
from ttwin.clean import clean_dataset
from ttwin.features import (
    build_features, get_feature_set, compute_effective_cpu_temp,
    add_calendar_features, add_lag_features, add_rolling_features, create_target,
)
from ttwin.splits import (
    split_chronological, split_leave_servers_out, split_strict,
    get_split_b_seeds,
)
from ttwin.metrics import (
    mae, rmse, skill_score_vs_persistence, persistence_predictions,
    per_server_metrics, summarize_per_server,
)
from ttwin.models import get_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Horizon map: timesteps (1 step = 10 min)
HORIZON_MAP = {
    "10min": 1,
    "1h": 6,
    "6h": 36,
}


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT)
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def generate_exp_id(config: dict) -> str:
    """Generate experiment ID from config."""
    parts = [
        config.get("dataset", "?"),
        config.get("split", "?"),
        config.get("feature_set", "?"),
        config.get("model", "?"),
        config.get("horizon", "?"),
    ]
    base = "_".join(parts)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}"


def run_single_experiment(config: dict) -> dict:
    """Run a single experiment and return results."""
    dataset = config["dataset"]
    split_type = config["split"]
    feature_set_name = config["feature_set"]
    model_name = config["model"]
    horizon_key = config["horizon"]
    seed = config.get("seed", 42)

    if horizon_key in HORIZON_MAP:
        horizon_steps = HORIZON_MAP[horizon_key]
    else:
        horizon_steps = int(horizon_key)
    target_col = "CPU_Temperature"
    target_name = f"{target_col}_target_{horizon_steps}"

    print(f"\n{'='*60}")
    print(f"Experiment: {dataset} | Split {split_type} | {feature_set_name} | {model_name} | {horizon_key}")
    print(f"{'='*60}")

    # Load data
    data_dir = INTERIM_DIR / dataset
    if not data_dir.exists():
        raise FileNotFoundError(f"Data not found: {data_dir}. Run make_parquet.py first.")

    combined_path = data_dir / "_combined.parquet"
    if combined_path.exists():
        print(f"Loading {combined_path}...")
        df = load_parquet(combined_path)
    else:
        print(f"Loading all servers from {data_dir}...")
        df = load_all_servers(data_dir)

    print(f"Loaded {len(df):,} rows, {df['Series'].nunique()} servers")

    # Feature engineering
    print("Building features...")
    df = build_features(
        df,
        target_col=target_col,
        horizons=[horizon_steps],
        lags=[1, 2, 3, 6, 12],
        rolling_windows=[6, 12, 36],
    )

    # Get feature columns
    feature_cols = get_feature_set(feature_set_name, df.columns.tolist())
    print(f"Feature set {feature_set_name}: {len(feature_cols)} features")

    if model_name == "persistence":
        # Special handling: persistence doesn't use features
        return run_persistence_experiment(df, config, horizon_steps, target_col, target_name)

    if len(feature_cols) == 0:
        raise ValueError(f"No features for set {feature_set_name}")

    # Split
    print(f"Splitting (type {split_type})...")
    results_list = []

    if split_type == "A":
        train, val, test = split_chronological(df, embargo_steps=horizon_steps)
        results_list.append(run_model_split(
            train, val, test, feature_cols, target_name, model_name, seed,
            split_label="A", config=config
        ))

    elif split_type == "B":
        seeds = config.get("seeds", get_split_b_seeds())
        for s in seeds:
            train_df, test_df, train_servers, test_servers = split_leave_servers_out(
                df, seed=s
            )
            # Carve validation from train (last 10% by time)
            train, val, _ = split_chronological(train_df, val_frac=0.1, embargo_steps=horizon_steps)
            results_list.append(run_model_split(
                train, val, test_df, feature_cols, target_name, model_name, s,
                split_label=f"B_seed{s}", config=config,
                extra={"test_servers": test_servers, "train_servers": train_servers}
            ))

    elif split_type == "C":
        seeds = config.get("seeds", get_split_b_seeds())
        for s in seeds:
            train, val, test = split_strict(df, embargo_steps=horizon_steps, seed=s)
            results_list.append(run_model_split(
                train, val, test, feature_cols, target_name, model_name, s,
                split_label=f"C_seed{s}", config=config
            ))

    # Aggregate results
    combined_results = {
        "config": config,
        "git_commit": get_git_commit(),
        "timestamp": datetime.now().isoformat(),
        "splits": results_list,
    }

    return combined_results


def run_persistence_experiment(df, config, horizon_steps, target_col, target_name):
    """Run persistence baseline (no model training needed)."""
    split_type = config["split"]

    df["persistence_pred"] = persistence_predictions(
        df.groupby("Series")[target_col].shift(0),  # current value as prediction
        horizon_steps=0  # persistence = current value for future
    )
    # Actually persistence at horizon h means: pred(t+h) = value(t)
    # The target is already shifted: target_name = value at t+h
    # So pred = current value = target_col at time t
    df["persistence_pred"] = df[target_col]

    results_list = []

    if split_type in ["A"]:
        train, val, test = split_chronological(df, embargo_steps=horizon_steps)
        valid = test[[target_name, "persistence_pred", "Series"]].dropna()
        ps_metrics = per_server_metrics(valid, target_name, "persistence_pred")
        summary = summarize_per_server(ps_metrics, "mae")
        results_list.append({
            "split_label": "A",
            "mae_global": mae(valid[target_name].values, valid["persistence_pred"].values),
            "rmse_global": rmse(valid[target_name].values, valid["persistence_pred"].values),
            "per_server_summary": summary,
            "n_test": len(valid),
        })

    elif split_type in ["B", "C"]:
        seeds = config.get("seeds", get_split_b_seeds())
        for s in seeds:
            if split_type == "B":
                _, test_df, _, _ = split_leave_servers_out(df, seed=s)
            else:
                _, _, test_df = split_strict(df, embargo_steps=horizon_steps, seed=s)
            valid = test_df[[target_name, "persistence_pred", "Series"]].dropna()
            if len(valid) == 0:
                continue
            ps_metrics = per_server_metrics(valid, target_name, "persistence_pred")
            summary = summarize_per_server(ps_metrics, "mae")
            results_list.append({
                "split_label": f"{split_type}_seed{s}",
                "mae_global": mae(valid[target_name].values, valid["persistence_pred"].values),
                "rmse_global": rmse(valid[target_name].values, valid["persistence_pred"].values),
                "per_server_summary": summary,
                "n_test": len(valid),
            })

    return {
        "config": config,
        "git_commit": get_git_commit(),
        "timestamp": datetime.now().isoformat(),
        "model": "persistence",
        "splits": results_list,
    }


def run_model_split(train, val, test, feature_cols, target_name, model_name, seed,
                    split_label, config, extra=None):
    """Train and evaluate a model on a single split."""
    # Drop rows with NaN in target or features
    train_clean = train[feature_cols + [target_name, "Series"]].dropna()
    val_clean = val[feature_cols + [target_name, "Series"]].dropna()
    test_clean = test[feature_cols + [target_name, "Series"]].dropna()

    X_train = train_clean[feature_cols].values
    y_train = train_clean[target_name].values
    X_val = val_clean[feature_cols].values if len(val_clean) > 0 else None
    y_val = val_clean[target_name].values if len(val_clean) > 0 else None
    X_test = test_clean[feature_cols].values
    y_test = test_clean[target_name].values

    print(f"  [{split_label}] Train: {len(X_train):,}, Val: {len(X_val) if X_val is not None else 0:,}, Test: {len(X_test):,}")

    if len(X_test) == 0:
        print(f"  [{split_label}] WARNING: No test data!")
        return {"split_label": split_label, "error": "no_test_data"}

    # Train
    t0 = time.time()
    model = get_model(model_name, seed=seed)
    if hasattr(model, 'fit'):
        if model_name in ("lightgbm",):
            model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)
        else:
            model.fit(X_train, y_train, X_val, y_val)
    train_time = time.time() - t0

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    mae_val = mae(y_test, y_pred)
    rmse_val = rmse(y_test, y_pred)

    # Persistence baseline on test set
    # Persistence = current temp predicts future temp, so we need the current value
    # We have the target (future value) and need the "current" value which is in the test data
    # But we'd need the original temperature column. For now, compute from test_clean
    test_clean = test_clean.copy()
    test_clean["y_pred"] = y_pred

    ps = per_server_metrics(test_clean, target_name, "y_pred")
    summary = summarize_per_server(ps, "mae")

    result = {
        "split_label": split_label,
        "model": model_name,
        "n_features": len(feature_cols),
        "n_train": len(X_train),
        "n_val": len(X_val) if X_val is not None else 0,
        "n_test": len(X_test),
        "mae_global": mae_val,
        "rmse_global": rmse_val,
        "per_server_summary": summary,
        "train_time_s": train_time,
    }

    if extra:
        result.update(extra)

    # Feature importance for tree models
    if hasattr(model, "feature_importance"):
        try:
            imp = model.feature_importance()
            result["top_features"] = imp.head(10).to_dict()
        except Exception:
            pass

    print(f"  [{split_label}] MAE: {mae_val:.3f} C, RMSE: {rmse_val:.3f} C, "
          f"train: {train_time:.1f}s")

    return result


def save_experiment(results: dict, exp_id: str) -> Path:
    """Save experiment results."""
    exp_dir = EXPERIMENTS_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = exp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(results.get("config", {}), f, default_flow_style=False)

    # Save metrics
    metrics_rows = []
    for split_result in results.get("splits", []):
        if isinstance(split_result, dict) and "error" not in split_result:
            row = {k: v for k, v in split_result.items()
                   if k not in ("per_server_summary", "top_features",
                                "test_servers", "train_servers")}
            if "per_server_summary" in split_result:
                for k2, v2 in split_result["per_server_summary"].items():
                    row[f"ps_{k2}"] = v2
            metrics_rows.append(row)

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(exp_dir / "metrics.csv", index=False)

    # Save full results as JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=convert)

    print(f"\nResults saved to {exp_dir}")
    return exp_dir


def main():
    parser = argparse.ArgumentParser(description="Run experiment.")
    parser.add_argument("--config", help="YAML config file")
    parser.add_argument("--dataset", default="dataset1")
    parser.add_argument("--feature-set", dest="feature_set", default="S2")
    parser.add_argument("--model", default="lightgbm")
    parser.add_argument("--split", default="A", choices=["A", "B", "C"])
    parser.add_argument("--horizon", default="10min", choices=["10min", "1h", "6h"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "dataset": args.dataset,
            "feature_set": args.feature_set,
            "model": args.model,
            "split": args.split,
            "horizon": args.horizon,
            "seed": args.seed,
        }
        if args.seeds:
            config["seeds"] = args.seeds

    exp_id = generate_exp_id(config)
    print(f"Experiment ID: {exp_id}")

    results = run_single_experiment(config)
    save_experiment(results, exp_id)


if __name__ == "__main__":
    main()
