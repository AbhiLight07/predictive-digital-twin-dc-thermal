#!/usr/bin/env python3
"""Run few-shot adaptation experiment (Phase 5 / E4 / RQ3).

Evaluates:
  1. Direct transfer (zero-shot)
  2. LightGBM fine-tune (warm-start from pretrained checkpoint)
  3. Ridge fine-tune
  4. LightGBM trained strictly from scratch on N days

Usage:
    python scripts/run_adaptation.py --dataset dataset1 --feature-set S4 --horizon 1h
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from ttwin.io import load_parquet
from ttwin.features import build_features, get_feature_set
from ttwin.splits import split_leave_servers_out, split_chronological
from ttwin.adapt import run_adaptation_experiment

HORIZON_MAP = {"10min": 1, "1h": 6, "6h": 36}
DATA_DIR = ROOT / "data" / "interim"
EXP_DIR = ROOT / "experiments"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset1")
    parser.add_argument("--feature-set", default="S4")
    parser.add_argument("--horizon", default="1h")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-servers", type=int, default=3, help="Number of target servers to evaluate")
    args = parser.parse_args()

    horizon_steps = HORIZON_MAP.get(args.horizon, 6)
    target_col = "CPU_Temperature"
    target_name = f"{target_col}_target_{horizon_steps}"

    # Load dataset
    combined = DATA_DIR / args.dataset / "_combined.parquet"
    print(f"Loading {combined}...")
    df = load_parquet(combined)
    print(f"Loaded {len(df):,} rows, {df['Series'].nunique()} servers")

    # Features
    print("Building features...")
    df = build_features(
        df,
        target_col=target_col,
        horizons=[horizon_steps],
        lags=[1, 2, 3, 6, 12],
        rolling_windows=[6, 12, 36],
    )
    feature_cols = get_feature_set(args.feature_set, df.columns.tolist())
    print(f"Feature set {args.feature_set}: {len(feature_cols)} features")

    # Split B: Leave out ~20% of servers
    train_full, test_full, train_servers, test_servers = split_leave_servers_out(
        df, seed=args.seed
    )
    print(f"Train servers: {len(train_servers)}, Test holdout servers: {len(test_servers)}")

    # Split chronological inside train for validation
    train_df, val_df, _ = split_chronological(train_full, val_frac=0.1, embargo_steps=horizon_steps)

    # Clean data
    train_clean = train_df[feature_cols + [target_name]].dropna()
    val_clean = val_df[feature_cols + [target_name]].dropna()

    X_train = train_clean[feature_cols].values
    y_train = train_clean[target_name].values
    X_val = val_clean[feature_cols].values
    y_val = val_clean[target_name].values

    # Train pretrained source model
    print(f"Training pretrained source LightGBM on {len(X_train):,} samples...")
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=feature_cols)

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "seed": args.seed,
    }
    source_model = lgb.train(
        params,
        dtrain,
        num_boost_round=400,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)],
    )
    print("Source model trained successfully.")

    # Select target servers
    eval_servers = test_servers[:args.n_servers]
    print(f"Evaluating adaptation on target servers: {eval_servers}")

    adapt_days = [1, 3, 7, 14, 30]
    all_results = []

    for srv in eval_servers:
        print(f"\n--- Adapting on Server: {srv} ---")
        df_srv = test_full[test_full["Series"] == srv].copy()
        res_df = run_adaptation_experiment(
            df_train=train_df,
            df_target_server=df_srv,
            source_model=source_model,
            feature_cols=feature_cols,
            target_col=target_name,
            adapt_days=adapt_days,
            seed=args.seed,
        )
        res_df["server"] = srv
        all_results.append(res_df)

    final_df = pd.concat(all_results, ignore_index=True)

    # Save results
    exp_id = f"adapt_{args.dataset}_{args.feature_set}_{args.horizon}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = EXP_DIR / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(out_dir / "adaptation_results.csv", index=False)

    # Aggregate by strategy and adapt_days
    summary = final_df.groupby(["strategy", "adapt_days"])["mae"].mean().reset_index()
    print("\n" + "=" * 60)
    print("ADAPTATION RESULTS (Mean MAE in deg C across target servers)")
    print("=" * 60)
    print(summary.to_string(index=False, float_format="%.3f"))

    # Plot adaptation curve
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(
            data=final_df,
            x="adapt_days",
            y="mae",
            hue="strategy",
            marker="o",
            ax=ax,
        )
        ax.set_title(f"Adaptation Learning Curves: MAE vs Local Telemetry ({args.horizon} horizon)", fontweight="bold")
        ax.set_xlabel("Adaptation Data (Days)")
        ax.set_ylabel("MAE (°C)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = out_dir / "adaptation_curves.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved adaptation curves to {plot_path}")
    except Exception as e:
        print(f"Plotting error: {e}")

    print(f"\nAll results saved to {out_dir}")


if __name__ == "__main__":
    main()
