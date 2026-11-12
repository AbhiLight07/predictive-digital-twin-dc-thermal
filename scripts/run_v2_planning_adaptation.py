#!/usr/bin/env python3
"""v2 Experiment E4: Planning Regime and Monitoring Cold-Start Adaptation.

Requirements:
- Planning regime (workload-only: S2, no temperature history) where persistence cannot be used.
- Meaningful baselines:
    (a) Cluster mean temperature
    (b) Local server mean estimated from first N hours
- Monitoring regime (S4, with T(t) and delta target) compared to persistence.
- Time-consistent pretraining: pretrain only on data BEFORE the adaptation test window.
- Budgets: 0, 1, 6, 24, 72, 168 hours.
- Sample size: >= 15 target servers (from GroupKFold).
- 95% bootstrap confidence interval over servers.

Outputs:
- experiments/v2_adaptation/planning_adaptation_results.csv
- experiments/v2_adaptation/monitoring_adaptation_results.csv
- experiments/v2_adaptation/summary_table.csv
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import os
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", "C:/tmp")

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ttwin.io import load_parquet
from ttwin.features import build_features, get_feature_set
from ttwin.splits import get_group_kfold_splits
from ttwin.metrics import mae, rmse

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--out-prefix", default="v2", help="Prefix for output experiment directory, e.g. v2 or v3")
args, _ = parser.parse_known_args()

OUT_DIR = ROOT / "experiments" / f"{args.out_prefix}_adaptation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data" / "interim"


def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=42):
    if len(values) <= 1:
        return float(values[0]), float(values[0])
    rng = np.random.RandomState(seed)
    boot_means = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    alpha = (1.0 - ci) / 2.0
    return float(np.percentile(boot_means, 100 * alpha)), float(np.percentile(boot_means, 100 * (1 - alpha)))


def main():
    print(f"\n{'='*70}\nRUNNING EXPERIMENT E4: PLANNING & MONITORING ADAPTATION\n{'='*70}")
    path = DATA_DIR / "dataset1" / "_combined.parquet"
    print(f"Loading data from {path}...")
    df = load_parquet(path)
    df["Series"] = df["Series"].astype(str).str.strip()

    # Drop 5 servers without S4 columns for comparability across both regimes
    disqual = {"qh2-rcc196", "qh2-rcc197", "qh2-rcc198", "qh2-rcc199", "qh2-rcc200"}
    df = df[~df["Series"].isin(disqual)].copy()

    # Build features for 1h horizon (6 steps)
    horizon_step = 6
    target_abs = f"CPU_Temperature_target_{horizon_step}"
    target_delta = f"CPU_Temperature_delta_{horizon_step}"

    print("Building features...")
    df = build_features(df, horizons=[horizon_step], lags=[1, 2, 3, 6, 12], rolling_windows=[6, 12, 36])

    s2_feats = get_feature_set("S2", df.columns.tolist())
    s4_feats = get_feature_set("S4", df.columns.tolist())

    # Get target servers: Fold 0 of 5-fold GroupKFold gives 15-16 servers
    splits = get_group_kfold_splits(df, n_splits=5, seed=42)
    source_servers, target_servers = splits[0]
    print(f"Source servers: {len(source_servers)} | Target servers: {len(target_servers)}")

    budgets_h = [0, 1, 6, 24, 72, 168]
    max_budget_h = 168
    embargo_h = 6

    # Time-consistent pretraining: use training data from earlier period
    df_source = df[df["Series"].isin(source_servers)].copy()
    split_time = df_source["Time"].min() + (df_source["Time"].max() - df_source["Time"].min()) * 0.70
    df_source_train = df_source[df_source["Time"] < split_time]

    print(f"Pretraining source models on {len(df_source_train):,} time-consistent rows...")

    # Pretrain Planning Model (S2, workload-only, predicting absolute temperature)
    clean_s2_src = df_source_train[s2_feats + [target_abs]].dropna()
    dtrain_s2 = lgb.Dataset(clean_s2_src[s2_feats].values, label=clean_s2_src[target_abs].values, feature_name=s2_feats)
    lgb_planning = lgb.train(
        {"objective": "regression", "metric": "mae", "learning_rate": 0.05, "num_leaves": 63, "verbosity": -1, "seed": 42},
        dtrain_s2, num_boost_round=200,
    )
    cluster_mean_temp = float(clean_s2_src[target_abs].mean())
    print(f"Cluster mean temperature: {cluster_mean_temp:.2f}°C")

    # Pretrain Monitoring Model (S4, full telemetry, predicting delta temperature)
    clean_s4_src = df_source_train[s4_feats + [target_delta]].dropna()
    dtrain_s4 = lgb.Dataset(clean_s4_src[s4_feats].values, label=clean_s4_src[target_delta].values, feature_name=s4_feats)
    lgb_monitoring = lgb.train(
        {"objective": "regression", "metric": "mae", "learning_rate": 0.05, "num_leaves": 63, "verbosity": -1, "seed": 42},
        dtrain_s4, num_boost_round=200,
    )

    planning_records = []
    monitoring_records = []

    for s_idx, srv in enumerate(target_servers):
        sdf = df[df["Series"] == srv].sort_values("Time").reset_index(drop=True)
        t_min = sdf["Time"].min()
        test_start = t_min + pd.Timedelta(hours=max_budget_h + embargo_h)
        sdf_test = sdf[sdf["Time"] >= test_start].copy()

        req_cols = set(["CPU_Temperature", target_abs, target_delta, "Series", "Time"] + s2_feats + s4_feats)
        clean_test = sdf_test[[c for c in req_cols if c in sdf_test.columns]].dropna()

        if len(clean_test) < 100:
            print(f"Skipping {srv}: insufficient test rows ({len(clean_test)})")
            continue

        y_test_abs = clean_test[target_abs].values
        y_test_curr = clean_test["CPU_Temperature"].values
        X_test_s2 = clean_test[s2_feats].values
        X_test_s4 = clean_test[s4_feats].values

        # Baseline: Persistence (Monitoring setting only)
        mae_pers = mae(y_test_abs, y_test_curr)
        monitoring_records.append({
            "server": srv, "budget_hours": 0, "strategy": "persistence",
            "regime": "monitoring", "mae": mae_pers,
        })

        # Baseline: Cluster mean (Planning setting)
        mae_cluster_mean = mae(y_test_abs, np.full_like(y_test_abs, cluster_mean_temp))
        planning_records.append({
            "server": srv, "budget_hours": 0, "strategy": "cluster_mean",
            "regime": "planning", "mae": mae_cluster_mean,
        })

        # Zero-shot Planning model
        pred_zs_plan = lgb_planning.predict(X_test_s2)
        planning_records.append({
            "server": srv, "budget_hours": 0, "strategy": "zero_shot",
            "regime": "planning", "mae": mae(y_test_abs, pred_zs_plan),
        })

        # Zero-shot Monitoring model
        pred_zs_mon_delta = lgb_monitoring.predict(X_test_s4)
        pred_zs_mon_abs = y_test_curr + pred_zs_mon_delta
        monitoring_records.append({
            "server": srv, "budget_hours": 0, "strategy": "zero_shot",
            "regime": "monitoring", "mae": mae(y_test_abs, pred_zs_mon_abs),
        })

        # Adaptation across hour-scale budgets
        for bh in [1, 6, 24, 72, 168]:
            t_adapt_end = t_min + pd.Timedelta(hours=bh)
            sdf_adapt = sdf[sdf["Time"] < t_adapt_end].copy()
            clean_adapt = sdf_adapt[[c for c in req_cols if c in sdf_adapt.columns]].dropna()
            n_adapt = len(clean_adapt)

            if n_adapt < 6:
                continue

            y_ad_abs = clean_adapt[target_abs].values
            y_ad_curr = clean_adapt["CPU_Temperature"].values
            y_ad_delta = clean_adapt[target_delta].values
            X_ad_s2 = clean_adapt[s2_feats].values
            X_ad_s4 = clean_adapt[s4_feats].values

            # ----------------- PLANNING REGIME -----------------
            # 1. Local server mean baseline
            server_mean_local = float(np.mean(y_ad_abs))
            planning_records.append({
                "server": srv, "budget_hours": bh, "strategy": "server_mean_local",
                "regime": "planning", "mae": mae(y_test_abs, np.full_like(y_test_abs, server_mean_local)),
            })

            # 2. Scalar residual offset correction
            ad_pred_plan = lgb_planning.predict(X_ad_s2)
            offset_plan = float(np.mean(y_ad_abs - ad_pred_plan))
            pred_plan_offset = pred_zs_plan + offset_plan
            planning_records.append({
                "server": srv, "budget_hours": bh, "strategy": "residual_offset",
                "regime": "planning", "mae": mae(y_test_abs, pred_plan_offset),
            })

            # 3. Standardised Ridge residual correction
            res_plan = y_ad_abs - ad_pred_plan
            ridge_pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=50.0))])
            ridge_pipe.fit(X_ad_s2, res_plan)
            pred_plan_ridge = pred_zs_plan + ridge_pipe.predict(X_test_s2)
            planning_records.append({
                "server": srv, "budget_hours": bh, "strategy": "residual_ridge",
                "regime": "planning", "mae": mae(y_test_abs, pred_plan_ridge),
            })

            # 4. Continued boosting (planning)
            dtrain_ad_plan = lgb.Dataset(X_ad_s2, label=y_ad_abs, feature_name=s2_feats)
            ws_plan = lgb.train(
                {"objective": "regression", "metric": "mae", "learning_rate": 0.01, "num_leaves": 15, "min_child_samples": 3, "verbosity": -1},
                dtrain_ad_plan, num_boost_round=min(30, max(5, n_adapt // 10)), init_model=lgb_planning,
            )
            planning_records.append({
                "server": srv, "budget_hours": bh, "strategy": "continued_boosting",
                "regime": "planning", "mae": mae(y_test_abs, ws_plan.predict(X_test_s2)),
            })

            # ----------------- MONITORING REGIME -----------------
            # 1. Scalar residual offset correction (on delta predictions)
            ad_pred_mon_delta = lgb_monitoring.predict(X_ad_s4)
            offset_mon = float(np.mean(y_ad_delta - ad_pred_mon_delta))
            pred_mon_offset = y_test_curr + (pred_zs_mon_delta + offset_mon)
            monitoring_records.append({
                "server": srv, "budget_hours": bh, "strategy": "residual_offset",
                "regime": "monitoring", "mae": mae(y_test_abs, pred_mon_offset),
            })

            # 2. Continued boosting (monitoring)
            dtrain_ad_mon = lgb.Dataset(X_ad_s4, label=y_ad_delta, feature_name=s4_feats)
            ws_mon = lgb.train(
                {"objective": "regression", "metric": "mae", "learning_rate": 0.01, "num_leaves": 15, "min_child_samples": 3, "verbosity": -1},
                dtrain_ad_mon, num_boost_round=min(30, max(5, n_adapt // 10)), init_model=lgb_monitoring,
            )
            pred_ws_mon_abs = y_test_curr + ws_mon.predict(X_test_s4)
            monitoring_records.append({
                "server": srv, "budget_hours": bh, "strategy": "continued_boosting",
                "regime": "monitoring", "mae": mae(y_test_abs, pred_ws_mon_abs),
            })

    # Save detailed records
    df_plan = pd.DataFrame(planning_records)
    df_plan.to_csv(OUT_DIR / "planning_adaptation_results.csv", index=False)

    df_mon = pd.DataFrame(monitoring_records)
    df_mon.to_csv(OUT_DIR / "monitoring_adaptation_results.csv", index=False)

    # Compute summary tables with 95% bootstrap CI over servers
    summary_rows = []
    for regime, df_sub in [("planning", df_plan), ("monitoring", df_mon)]:
        for (bh, strat), grp in df_sub.groupby(["budget_hours", "strategy"]):
            vals = grp["mae"].values
            ci_low, ci_high = bootstrap_ci(vals)
            summary_rows.append({
                "regime": regime, "strategy": strat, "budget_hours": bh,
                "n_servers": len(vals), "mean_mae": float(np.mean(vals)),
                "std_mae": float(np.std(vals)), "ci_95_low": ci_low, "ci_95_high": ci_high,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "summary_table.csv", index=False)

    print(f"\nAdaptation Summary Table (N={len(target_servers)} servers):")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
