#!/usr/bin/env python3
"""Diagnostic script for Part A Question 5: Random 5-Fold CV vs GroupKFold on identical rows."""

import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ttwin.io import load_parquet
from ttwin.features import build_features, get_feature_set
from ttwin.splits import get_group_kfold_splits
from ttwin.metrics import mae, rmse

def main():
    print("=" * 80)
    print("PART A DIAGNOSTICS: QUESTION 5 (RANDOM K-FOLD VS GROUPKFOLD ON IDENTICAL ROWS)")
    print("=" * 80)

    path = ROOT / "data" / "interim" / "dataset1" / "_combined.parquet"
    print(f"Loading data from {path}...")
    df = load_parquet(path)
    df["Series"] = df["Series"].astype(str).str.strip()

    # Drop 5 servers without S4 columns for comparability across all S1-S4
    disqual = {"qh2-rcc196", "qh2-rcc197", "qh2-rcc198", "qh2-rcc199", "qh2-rcc200"}
    df = df[~df["Series"].isin(disqual)].copy()

    horizon_step = 6
    target_abs = f"CPU_Temperature_target_{horizon_step}"
    target_delta = f"CPU_Temperature_delta_{horizon_step}"

    print("Building features...")
    df = build_features(df, horizons=[horizon_step], lags=[1, 2, 3, 6, 12], rolling_windows=[6, 12, 36])

    feature_sets = {
        "S1": {"feats": list(dict.fromkeys(get_feature_set("S1", df.columns.tolist()))), "target": target_delta, "is_delta": True},
        "S4": {"feats": list(dict.fromkeys(get_feature_set("S4", df.columns.tolist()))), "target": target_delta, "is_delta": True},
        "S2": {"feats": list(dict.fromkeys(get_feature_set("S2", df.columns.tolist()))), "target": target_abs, "is_delta": False},
        "S3": {"feats": list(dict.fromkeys(get_feature_set("S3", df.columns.tolist()))), "target": target_abs, "is_delta": False},
    }

    # Require non-null values for all columns across S1, S2, S3, S4
    all_needed = list(dict.fromkeys(
        ["CPU_Temperature", target_abs, target_delta, "Series", "Time"] +
        feature_sets["S1"]["feats"] + feature_sets["S4"]["feats"] +
        feature_sets["S2"]["feats"] + feature_sets["S3"]["feats"]
    ))
    clean_df = df[all_needed].dropna().reset_index(drop=True)
    print(f"Total identical rows for evaluation: {len(clean_df):,} across {clean_df['Series'].nunique()} servers")

    # Set up 5-fold GroupKFold and 5-fold Random CV
    group_splits = get_group_kfold_splits(clean_df, n_splits=5, seed=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    random_splits = list(kf.split(clean_df))

    results = []

    for fs_name, fs_info in feature_sets.items():
        feats = fs_info["feats"]
        target_col = fs_info["target"]
        is_delta = fs_info["is_delta"]

        print(f"\nEvaluating {fs_name} ({'Delta target' if is_delta else 'Absolute target'}, {len(feats)} features)...")

        # 1. GroupKFold (Fold 1 to 5)
        g_fold_maes = []
        for fold_idx, (train_srvs, test_srvs) in enumerate(group_splits):
            tr = clean_df[clean_df["Series"].isin(train_srvs)]
            te = clean_df[clean_df["Series"].isin(test_srvs)]

            X_tr, y_tr = tr[feats].values, tr[target_col].values
            X_te = te[feats].values
            y_te_abs = te[target_abs].values
            y_te_curr = te["CPU_Temperature"].values

            dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feats)
            params = {"objective": "regression", "metric": "mae", "learning_rate": 0.05, "num_leaves": 63, "verbosity": -1, "seed": 42}
            booster = lgb.train(params, dtrain, num_boost_round=100)

            if is_delta:
                pred = y_te_curr + booster.predict(X_te)
            else:
                pred = booster.predict(X_te)

            fold_err = mae(y_te_abs, pred)
            g_fold_maes.append(fold_err)

        # 2. Random 5-Fold CV (Fold 1 to 5)
        r_fold_maes = []
        for fold_idx, (tr_idx, te_idx) in enumerate(random_splits):
            tr = clean_df.iloc[tr_idx]
            te = clean_df.iloc[te_idx]

            X_tr, y_tr = tr[feats].values, tr[target_col].values
            X_te = te[feats].values
            y_te_abs = te[target_abs].values
            y_te_curr = te["CPU_Temperature"].values

            dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feats)
            params = {"objective": "regression", "metric": "mae", "learning_rate": 0.05, "num_leaves": 63, "verbosity": -1, "seed": 42}
            booster = lgb.train(params, dtrain, num_boost_round=100)

            if is_delta:
                pred = y_te_curr + booster.predict(X_te)
            else:
                pred = booster.predict(X_te)

            fold_err = mae(y_te_abs, pred)
            r_fold_maes.append(fold_err)

        g_mean = float(np.mean(g_fold_maes))
        g_std = float(np.std(g_fold_maes))
        r_mean = float(np.mean(r_fold_maes))
        r_std = float(np.std(r_fold_maes))
        diff = g_mean - r_mean
        diff_pct = (diff / g_mean) * 100.0

        print(f"  GroupKFold MAE: {g_mean:.4f} ± {g_std:.4f}°C")
        print(f"  Random CV  MAE: {r_mean:.4f} ± {r_std:.4f}°C")
        print(f"  Difference (Group - Random): {diff:+.4f}°C ({diff_pct:+.2f}% leakage reduction)")

        results.append({
            "feature_set": fs_name,
            "target_type": "delta" if is_delta else "absolute",
            "group_kfold_mean_mae": g_mean,
            "group_kfold_std_mae": g_std,
            "random_cv_mean_mae": r_mean,
            "random_cv_std_mae": r_std,
            "difference_mae": diff,
            "leakage_underestimation_pct": diff_pct,
        })

    res_df = pd.DataFrame(results)
    out_path = ROOT / "experiments" / "diag_part_a_q5_leakage_comparison.csv"
    res_df.to_csv(out_path, index=False)
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: RANDOM 5-FOLD CV VS GROUPKFOLD (IDENTICAL ROWS)")
    print("=" * 80)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()
