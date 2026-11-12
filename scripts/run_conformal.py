#!/usr/bin/env python3
"""Run conformal prediction experiment (E5).

Usage:
    python scripts/run_conformal.py --dataset dataset1 --feature-set S4 --horizon 1h --split A
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

from ttwin.io import load_parquet
from ttwin.features import build_features, get_feature_set
from ttwin.splits import split_chronological, split_leave_servers_out
from ttwin.uncertainty import conformalize, coverage, average_width, run_conformal_experiment

HORIZON_MAP = {"10min": 1, "1h": 6, "6h": 36}
DATA_DIR = ROOT / "data" / "interim"
EXP_DIR = ROOT / "experiments"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset1")
    parser.add_argument("--feature-set", default="S4")
    parser.add_argument("--horizon", default="1h")
    parser.add_argument("--split", default="A", choices=["A", "B"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.horizon in HORIZON_MAP:
        horizon_steps = HORIZON_MAP[args.horizon]
    else:
        horizon_steps = int(args.horizon)

    target_col = "CPU_Temperature"
    target_name = f"{target_col}_target_{horizon_steps}"

    # Load data
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

    # Split — use val as calibration
    if args.split == "A":
        train_df, val_df, test_df = split_chronological(df)
    else:
        train_full, test_df, _, _ = split_leave_servers_out(df, seed=args.seed)
        train_df, val_df, _ = split_chronological(train_full, val_frac=0.15)

    # Drop NaN
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  [{name}] {len(part):,} rows")

    train_clean = train_df[feature_cols + [target_name]].dropna()
    val_clean = val_df[feature_cols + [target_name]].dropna()
    test_clean = test_df[feature_cols + [target_name]].dropna()

    X_train = train_clean[feature_cols].values
    y_train = train_clean[target_name].values
    X_val = val_clean[feature_cols].values
    y_val = val_clean[target_name].values
    X_test = test_clean[feature_cols].values
    y_test = test_clean[target_name].values

    print(f"After dropna — train: {len(X_train):,}, val(cal): {len(X_val):,}, test: {len(X_test):,}")

    # Train model
    print("Training LightGBM...")
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=feature_cols)

    params = {
        "objective": "regression", "metric": "mae",
        "learning_rate": 0.05, "num_leaves": 127,
        "feature_fraction": 0.7, "bagging_fraction": 0.8,
        "bagging_freq": 5, "verbosity": -1, "seed": args.seed,
    }
    model = lgb.train(
        params, dtrain, num_boost_round=500, valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    # Predictions
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # Conformal prediction at multiple alpha levels
    alphas = [0.05, 0.10, 0.20, 0.30]
    results = []

    print("\n" + "=" * 60)
    print(f"CONFORMAL PREDICTION RESULTS — {args.feature_set} {args.horizon} Split {args.split}")
    print("=" * 60)

    for alpha in alphas:
        lower, upper, q_hat = conformalize(y_val, y_pred_val, y_pred_test, alpha=alpha)
        cov = coverage(y_test, lower, upper)
        width = average_width(lower, upper)
        target_cov = 1 - alpha

        results.append({
            "alpha": alpha,
            "target_coverage": target_cov,
            "empirical_coverage": round(cov, 4),
            "average_width_C": round(width, 3),
            "q_hat": round(q_hat, 4),
            "coverage_gap": round(cov - target_cov, 4),
            "n_cal": len(y_val),
            "n_test": len(y_test),
        })

        status = "OK" if cov >= target_cov - 0.01 else "MISCOVERAGE"
        print(f"  alpha={alpha:.2f} | target={target_cov:.0%} | empirical={cov:.1%} | "
              f"width={width:.2f} deg C | q_hat={q_hat:.3f} [{status}]")

    # Save
    exp_id = f"conformal_{args.dataset}_{args.split}_{args.feature_set}_{args.horizon}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = EXP_DIR / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "conformal_results.csv", index=False)

    summary = {
        "exp_id": exp_id, "dataset": args.dataset, "split": args.split,
        "feature_set": args.feature_set, "horizon": args.horizon,
        "model_mae": float(np.mean(np.abs(y_test - y_pred_test))),
        "conformal": results,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
