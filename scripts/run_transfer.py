#!/usr/bin/env python3
"""Run cross-cluster hardware transfer experiment (Phase 7 / E3 / RQ5).

Source: Dataset 1 (Cluster 1, Dell R840, 81 servers)
Target: Dataset 2 (Cluster 2, Dell C6320, 64 servers)

Evaluates:
  1. Zero-shot transfer (Model trained on Cluster 1 -> evaluated on Cluster 2)
  2. Native benchmark (Model trained directly on Cluster 2 -> evaluated on Cluster 2)
  3. Transfer gap: MAE_transfer - MAE_native

Usage:
    python scripts/run_transfer.py --horizon 1h
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
from ttwin.features import build_features
from ttwin.splits import split_chronological
from ttwin.metrics import mae, rmse

HORIZON_MAP = {"10min": 1, "1h": 6, "6h": 36}
DATA_DIR = ROOT / "data" / "interim"
EXP_DIR = ROOT / "experiments"

SHARED_FEATURES = [
    "CPU_Load",
    "Ram_Used",
    "No_Of_Running_vms",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "CPU_Temperature_lag_1",
    "CPU_Temperature_lag_2",
    "CPU_Temperature_lag_6",
    "CPU_Temperature_rmean_6",
    "CPU_Temperature_rstd_6",
]


def prepare_cluster_data(dataset_name: str, horizon_steps: int):
    """Load and prepare shared features for a cluster."""
    path = DATA_DIR / dataset_name / "_combined.parquet"
    print(f"Loading {dataset_name} from {path}...")
    df = load_parquet(path)

    # Features
    df = build_features(
        df,
        target_col="CPU_Temperature",
        horizons=[horizon_steps],
        lags=[1, 2, 6],
        rolling_windows=[6],
    )

    target_name = f"CPU_Temperature_target_{horizon_steps}"
    available_features = [f for f in SHARED_FEATURES if f in df.columns]

    return df, available_features, target_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", default="1h")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    horizon_steps = HORIZON_MAP.get(args.horizon, 6)
    target_name = f"CPU_Temperature_target_{horizon_steps}"

    print("=" * 65)
    print("CROSS-CLUSTER TRANSFER EXPERIMENT (E3 / RQ5)")
    print("=" * 65)

    # Prepare Dataset 1 (Source: Cluster 1 Dell R840)
    df1, feat1, _ = prepare_cluster_data("dataset1", horizon_steps)

    # Prepare Dataset 2 (Target: Cluster 2 Dell C6320)
    df2, feat2, _ = prepare_cluster_data("dataset2", horizon_steps)

    shared_cols = sorted(list(set(feat1) & set(feat2)))
    print(f"Shared features across clusters ({len(shared_cols)}): {shared_cols}")

    # Splits
    train1, val1, test1 = split_chronological(df1, embargo_steps=horizon_steps)
    train2, val2, test2 = split_chronological(df2, embargo_steps=horizon_steps)

    # Drop NA
    clean_tr1 = train1[shared_cols + [target_name]].dropna()
    clean_val1 = val1[shared_cols + [target_name]].dropna()
    clean_tr2 = train2[shared_cols + [target_name]].dropna()
    clean_val2 = val2[shared_cols + [target_name]].dropna()
    clean_te2 = test2[shared_cols + [target_name]].dropna()

    X_tr1, y_tr1 = clean_tr1[shared_cols].values, clean_tr1[target_name].values
    X_val1, y_val1 = clean_val1[shared_cols].values, clean_val1[target_name].values

    X_tr2, y_tr2 = clean_tr2[shared_cols].values, clean_tr2[target_name].values
    X_val2, y_val2 = clean_val2[shared_cols].values, clean_val2[target_name].values

    X_te2, y_te2 = clean_te2[shared_cols].values, clean_te2[target_name].values

    # Train Model on Cluster 1 (Source)
    print(f"\nTraining Source Model on Cluster 1 ({len(X_tr1):,} rows)...")
    dtr1 = lgb.Dataset(X_tr1, label=y_tr1, feature_name=shared_cols)
    dv1 = lgb.Dataset(X_val1, label=y_val1, reference=dtr1, feature_name=shared_cols)

    params = {
        "objective": "regression", "metric": "mae",
        "learning_rate": 0.05, "num_leaves": 127,
        "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 5, "verbosity": -1, "seed": args.seed,
    }
    source_model = lgb.train(
        params, dtr1, num_boost_round=400, valid_sets=[dv1],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)],
    )

    # Evaluate Source Model on Cluster 2 Test Set (Zero-shot transfer)
    preds_transfer = source_model.predict(X_te2)
    mae_transfer = mae(y_te2, preds_transfer)
    rmse_transfer = rmse(y_te2, preds_transfer)

    # Train Native Model directly on Cluster 2
    print(f"\nTraining Native Model on Cluster 2 ({len(X_tr2):,} rows)...")
    dtr2 = lgb.Dataset(X_tr2, label=y_tr2, feature_name=shared_cols)
    dv2 = lgb.Dataset(X_val2, label=y_val2, reference=dtr2, feature_name=shared_cols)

    native_model = lgb.train(
        params, dtr2, num_boost_round=400, valid_sets=[dv2],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)],
    )

    # Evaluate Native Model on Cluster 2 Test Set
    preds_native = native_model.predict(X_te2)
    mae_native = mae(y_te2, preds_native)
    rmse_native = rmse(y_te2, preds_native)

    # Persistence baseline on Cluster 2
    clean_te2_series = test2[["Series", "CPU_Temperature", target_name]].dropna()
    mae_pers = mae(clean_te2_series[target_name].values, clean_te2_series["CPU_Temperature"].values)

    transfer_gap = mae_transfer - mae_native

    print("\n" + "=" * 65)
    print(f"CROSS-CLUSTER EVALUATION RESULTS on Cluster 2 (Horizon: {args.horizon})")
    print("=" * 65)
    print(f"  Persistence Baseline on Cluster 2:   MAE = {mae_pers:.3f} deg C")
    print(f"  Native Model (Trained on Cluster 2):  MAE = {mae_native:.3f} deg C | RMSE = {rmse_native:.3f} deg C")
    print(f"  Transferred Model (from Cluster 1):  MAE = {mae_transfer:.3f} deg C | RMSE = {rmse_transfer:.3f} deg C")
    print(f"  Transfer Gap (Degradation):          dMAE = {transfer_gap:+.3f} deg C")
    print("=" * 65)

    # Save results
    exp_id = f"transfer_c1_to_c2_{args.horizon}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = EXP_DIR / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "exp_id": exp_id,
        "source": "dataset1 (Dell R840)",
        "target": "dataset2 (Dell C6320)",
        "horizon": args.horizon,
        "shared_features": shared_cols,
        "n_test_cluster2": len(y_te2),
        "mae_persistence": float(mae_pers),
        "mae_native": float(mae_native),
        "rmse_native": float(rmse_native),
        "mae_transfer": float(mae_transfer),
        "rmse_transfer": float(rmse_transfer),
        "transfer_gap_mae": float(transfer_gap),
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    df_res = pd.DataFrame([{
        "model": "Persistence", "mae": mae_pers, "rmse": np.nan, "gap": np.nan
    }, {
        "model": "Native (Cluster 2)", "mae": mae_native, "rmse": rmse_native, "gap": 0.0
    }, {
        "model": "Transfer (Cluster 1 -> 2)", "mae": mae_transfer, "rmse": rmse_transfer, "gap": transfer_gap
    }])
    df_res.to_csv(out_dir / "metrics.csv", index=False)
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
