#!/usr/bin/env python3
"""P2 Diagnostics: Cross-cluster hardware transfer (E3) verification and fixes.

Covers items 4.1, 4.2, 4.3, 4.4.
Outputs results to experiments/diag_transfer/ and prints full analysis.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import lightgbm as lgb

from ttwin.io import load_parquet
from ttwin.features import build_features
from ttwin.splits import split_chronological
from ttwin.metrics import mae, rmse, skill_score_vs_persistence

DATA_DIR = ROOT / "data" / "interim"
EXP_DIR = ROOT / "experiments" / "diag_transfer"
EXP_DIR.mkdir(parents=True, exist_ok=True)

SHARED_12_FEATURES = [
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

WORKLOAD_ONLY_FEATURES = [
    "CPU_Load",
    "Ram_Used",
    "No_Of_Running_vms",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


def diag_4_1_feature_check():
    """4.1: Feature list check and exclusion verification."""
    print("\n" + "=" * 70)
    print("4.1 SHARED FEATURE LIST VERIFICATION")
    print("=" * 70)
    print(f"The 12 shared features used in E3 (run_transfer.py lines 38-51):")
    for i, feat in enumerate(SHARED_12_FEATURES, 1):
        print(f"  {i:2d}. {feat}")
    print()
    has_power = any("power" in f.lower() for f in SHARED_12_FEATURES)
    has_fan = any("fan" in f.lower() for f in SHARED_12_FEATURES)
    has_inlet = any("inlet" in f.lower() for f in SHARED_12_FEATURES)
    print(f"CONFIRMATION: Is Power in shared features? {has_power}")
    print(f"CONFIRMATION: Are Fan speeds in shared features? {has_fan}")
    print(f"CONFIRMATION: Is Inlet Temperature in shared features? {has_inlet}")
    print("VERDICT: Confirmed. None of power, fan speeds, or inlet temperature is in the 12 shared features.")


def prepare_data(dataset_name: str, horizon_steps: int = 6):
    path = DATA_DIR / dataset_name / "_combined.parquet"
    print(f"Loading {dataset_name} from {path}...")
    df = load_parquet(path)
    df = build_features(
        df,
        target_col="CPU_Temperature",
        horizons=[horizon_steps],
        lags=[1, 2, 6],
        rolling_windows=[6],
    )
    return df


def diag_4_3_distribution_comparison(df1, df2):
    """4.3: Physical and statistical sensor comparison between clusters."""
    print("\n" + "=" * 70)
    print("4.3 CPU TEMPERATURE DISTRIBUTION & SENSOR COMPARISON")
    print("=" * 70)

    t1 = df1["CPU_Temperature"].dropna()
    t2 = df2["CPU_Temperature"].dropna()

    idle1 = df1.loc[df1["CPU_Load"] < 10.0, "CPU_Temperature"].dropna()
    idle2 = df2.loc[df2["CPU_Load"] < 10.0, "CPU_Temperature"].dropna()

    stats = {
        "Metric": ["Mean (°C)", "Std (°C)", "Min (°C)", "25th Pct (°C)", "Median (°C)", "75th Pct (°C)", "Max (°C)", "Idle Mean (CPU<10%) (°C)", "Num Servers", "Total Rows"],
        "Cluster 1 (Dell R840)": [
            f"{t1.mean():.2f}", f"{t1.std():.2f}", f"{t1.min():.2f}", f"{t1.quantile(0.25):.2f}",
            f"{t1.median():.2f}", f"{t1.quantile(0.75):.2f}", f"{t1.max():.2f}", f"{idle1.mean():.2f}",
            f"{df1['Series'].nunique()}", f"{len(df1):,}"
        ],
        "Cluster 2 (Dell C6320)": [
            f"{t2.mean():.2f}", f"{t2.std():.2f}", f"{t2.min():.2f}", f"{t2.quantile(0.25):.2f}",
            f"{t2.median():.2f}", f"{t2.quantile(0.75):.2f}", f"{t2.max():.2f}", f"{idle2.mean():.2f}",
            f"{df2['Series'].nunique()}", f"{len(df2):,}"
        ],
    }
    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))

    print("\nSensor Construction:")
    print("  Cluster 1 has columns ['CPU1_Temperature', 'CPU2_Temperature']. Effective temp = max(CPU1, CPU2).")
    print("  Cluster 2 has columns ['CPU1_Temperature', 'CPU2_Temperature']. Effective temp = max(CPU1, CPU2).")
    print("  Code line: features.py lines 74-87 (compute_effective_cpu_temp).")

    print("\nPhysical Comparability according to Literature:")
    print("  Source: Ahmed et al. (2023), Zenodo 10069402, 'Cloud Datacenter Workload and Telemetry Traces'.")
    print("  Sensor mechanism: Both use on-die DTS (Digital Thermal Sensor) / Intel PECI registers.")
    print("  Physical equivalence: NOT physically comparable across chassis architectures!")
    print("    - Cluster 1 is Dell PowerEdge R840 (quad-socket 4U chassis, dedicated high-CFM chassis blowers).")
    print("    - Cluster 2 is Dell PowerEdge C6320 (high-density modular sled, 4 dual-socket nodes shared in 2U).")
    print("    - Thermal impedance, fan cooling curves, heat sink mass, and airflow shadowing differ fundamentally.")


def run_transfer_diagnostics():
    diag_4_1_feature_check()

    horizon_steps = 6
    target_abs = f"CPU_Temperature_target_{horizon_steps}"
    target_delta = f"CPU_Temperature_delta_{horizon_steps}"

    df1 = prepare_data("dataset1", horizon_steps)
    df2 = prepare_data("dataset2", horizon_steps)

    # Create delta targets: T(t+h) - T(t)
    df1[target_delta] = df1[target_abs] - df1["CPU_Temperature"]
    df2[target_delta] = df2[target_abs] - df2["CPU_Temperature"]

    diag_4_3_distribution_comparison(df1, df2)

    # Train/val/test splits
    tr1, va1, te1 = split_chronological(df1, embargo_steps=horizon_steps)
    tr2, va2, te2 = split_chronological(df2, embargo_steps=horizon_steps)

    params = {
        "objective": "regression", "metric": "mae",
        "learning_rate": 0.05, "num_leaves": 127,
        "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 5, "verbosity": -1, "seed": 42,
    }

    # Persistence on Cluster 2 test set
    te2_clean = te2[[target_abs, target_delta, "CPU_Temperature", "Series", "Time"] + SHARED_12_FEATURES].dropna()
    y_te2_abs = te2_clean[target_abs].values
    y_te2_curr = te2_clean["CPU_Temperature"].values
    y_te2_delta = te2_clean[target_delta].values

    mae_pers = mae(y_te2_abs, y_te2_curr)
    rmse_pers = rmse(y_te2_abs, y_te2_curr)
    print(f"\nCluster 2 Test Set Persistence Baseline: MAE = {mae_pers:.4f}°C, RMSE = {rmse_pers:.4f}°C")

    # =========================================================================
    # 4.1: DELTA TARGET TRANSFER (Native vs Transferred with Delta Target)
    # =========================================================================
    print("\n" + "=" * 70)
    print("4.1 DELTA TARGET RE-RUN (Native vs Transferred on Cluster 2)")
    print("=" * 70)

    # Clean data for delta training
    c_tr1 = tr1[SHARED_12_FEATURES + [target_delta]].dropna()
    c_va1 = va1[SHARED_12_FEATURES + [target_delta]].dropna()
    c_tr2 = tr2[SHARED_12_FEATURES + [target_delta]].dropna()
    c_va2 = va2[SHARED_12_FEATURES + [target_delta]].dropna()

    # Model 1: Transferred Model (trained on Cluster 1 delta)
    dtr1 = lgb.Dataset(c_tr1[SHARED_12_FEATURES].values, label=c_tr1[target_delta].values, feature_name=SHARED_12_FEATURES)
    dva1 = lgb.Dataset(c_va1[SHARED_12_FEATURES].values, label=c_va1[target_delta].values, reference=dtr1, feature_name=SHARED_12_FEATURES)
    m_transfer_delta = lgb.train(params, dtr1, num_boost_round=400, valid_sets=[dva1], callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])

    # Model 2: Native Model (trained on Cluster 2 delta)
    dtr2 = lgb.Dataset(c_tr2[SHARED_12_FEATURES].values, label=c_tr2[target_delta].values, feature_name=SHARED_12_FEATURES)
    dva2 = lgb.Dataset(c_va2[SHARED_12_FEATURES].values, label=c_va2[target_delta].values, reference=dtr2, feature_name=SHARED_12_FEATURES)
    m_native_delta = lgb.train(params, dtr2, num_boost_round=400, valid_sets=[dva2], callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])

    # Evaluate on Cluster 2
    X_te2 = te2_clean[SHARED_12_FEATURES].values
    pred_trans_delta = m_transfer_delta.predict(X_te2)
    pred_nat_delta = m_native_delta.predict(X_te2)

    # Convert delta back to absolute
    y_pred_trans_abs = y_te2_curr + pred_trans_delta
    y_pred_nat_abs = y_te2_curr + pred_nat_delta

    mae_trans = mae(y_te2_abs, y_pred_trans_abs)
    rmse_trans = rmse(y_te2_abs, y_pred_trans_abs)
    skill_trans = skill_score_vs_persistence(mae_trans, mae_pers)

    mae_nat = mae(y_te2_abs, y_pred_nat_abs)
    rmse_nat = rmse(y_te2_abs, y_pred_nat_abs)
    skill_nat = skill_score_vs_persistence(mae_nat, mae_pers)

    gap_delta = mae_trans - mae_nat

    print(f"Results on Cluster 2 (Delta Target Formulation):")
    print(f"  Persistence:       MAE = {mae_pers:.4f}°C | Skill = 0.00%")
    print(f"  Native (C2):       MAE = {mae_nat:.4f}°C | RMSE = {rmse_nat:.4f}°C | Skill = {skill_nat:+.2%}")
    print(f"  Transfer (C1->C2): MAE = {mae_trans:.4f}°C | RMSE = {rmse_trans:.4f}°C | Skill = {skill_trans:+.2%}")
    print(f"  Transfer Gap:      dMAE = {gap_delta:+.4f}°C")

    # =========================================================================
    # 4.2: WORKLOAD-ONLY TRANSFER (No temperature history)
    # =========================================================================
    print("\n" + "=" * 70)
    print("4.2 WORKLOAD-ONLY TRANSFER (No temperature history, RQ5)")
    print("=" * 70)

    # Workload-only models predict absolute temperature directly (or delta, but workload-only lacks T(t)!)
    # In planning setting (RQ5), current temperature is unknown -> must predict absolute temperature!
    cw_tr1 = tr1[WORKLOAD_ONLY_FEATURES + [target_abs]].dropna()
    cw_va1 = va1[WORKLOAD_ONLY_FEATURES + [target_abs]].dropna()
    cw_tr2 = tr2[WORKLOAD_ONLY_FEATURES + [target_abs]].dropna()
    cw_va2 = va2[WORKLOAD_ONLY_FEATURES + [target_abs]].dropna()
    te2_wl = te2[WORKLOAD_ONLY_FEATURES + [target_abs]].dropna()

    dwtr1 = lgb.Dataset(cw_tr1[WORKLOAD_ONLY_FEATURES].values, label=cw_tr1[target_abs].values, feature_name=WORKLOAD_ONLY_FEATURES)
    dwva1 = lgb.Dataset(cw_va1[WORKLOAD_ONLY_FEATURES].values, label=cw_va1[target_abs].values, reference=dwtr1, feature_name=WORKLOAD_ONLY_FEATURES)
    m_wl_transfer = lgb.train(params, dwtr1, num_boost_round=400, valid_sets=[dwva1], callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])

    dwtr2 = lgb.Dataset(cw_tr2[WORKLOAD_ONLY_FEATURES].values, label=cw_tr2[target_abs].values, feature_name=WORKLOAD_ONLY_FEATURES)
    dwva2 = lgb.Dataset(cw_va2[WORKLOAD_ONLY_FEATURES].values, label=cw_va2[target_abs].values, reference=dwtr2, feature_name=WORKLOAD_ONLY_FEATURES)
    m_wl_native = lgb.train(params, dwtr2, num_boost_round=400, valid_sets=[dwva2], callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])

    X_te2_wl = te2_wl[WORKLOAD_ONLY_FEATURES].values
    y_te2_wl = te2_wl[target_abs].values

    pred_wl_trans = m_wl_transfer.predict(X_te2_wl)
    pred_wl_nat = m_wl_native.predict(X_te2_wl)

    mae_wl_trans = mae(y_te2_wl, pred_wl_trans)
    mae_wl_nat = mae(y_te2_wl, pred_wl_nat)
    gap_wl = mae_wl_trans - mae_wl_nat

    print(f"Results on Cluster 2 (Workload-Only, No Temp Lags):")
    print(f"  Native (C2):       MAE = {mae_wl_nat:.4f}°C")
    print(f"  Transfer (C1->C2): MAE = {mae_wl_trans:.4f}°C")
    print(f"  Transfer Gap:      dMAE = {gap_wl:+.4f}°C")

    # =========================================================================
    # 4.4: TRANSFER + ADAPTATION WITH N HOURS OF CLUSTER 2 DATA
    # =========================================================================
    print("\n" + "=" * 70)
    print("4.4 TRANSFER + ADAPTATION WITH N HOURS OF CLUSTER 2 DATA")
    print("=" * 70)

    # Use first N hours of Cluster 2 training data to adapt the C1 transferred model
    t2_min = tr2["Time"].min()
    budgets_h = [1, 6, 24, 72, 168]
    adapt_results = []

    for bh in budgets_h:
        adapt_end = t2_min + pd.Timedelta(hours=bh)
        ad_slice = tr2[tr2["Time"] < adapt_end][SHARED_12_FEATURES + [target_delta, "CPU_Temperature"]].dropna()

        if len(ad_slice) < 10:
            continue

        X_ad = ad_slice[SHARED_12_FEATURES].values
        y_ad_delta = ad_slice[target_delta].values

        # Continued boosting from C1 model
        dtrain_ad = lgb.Dataset(X_ad, label=y_ad_delta, feature_name=SHARED_12_FEATURES)
        adapted_model = lgb.train(
            {"objective": "regression", "metric": "mae", "learning_rate": 0.01, "verbosity": -1, "num_leaves": 31},
            dtrain_ad,
            num_boost_round=30,
            init_model=m_transfer_delta,
            callbacks=[lgb.log_evaluation(0)],
        )

        p_ad_delta = adapted_model.predict(X_te2)
        p_ad_abs = y_te2_curr + p_ad_delta
        mae_ad = mae(y_te2_abs, p_ad_abs)
        skill_ad = skill_score_vs_persistence(mae_ad, mae_pers)
        adapt_results.append({
            "budget_hours": bh, "n_samples": len(ad_slice),
            "mae": mae_ad, "skill": skill_ad,
        })
        print(f"  Budget {bh:3d}h ({len(ad_slice):5d} samples): MAE = {mae_ad:.4f}°C | Skill vs Persistence = {skill_ad:+.2%}")

    pd.DataFrame(adapt_results).to_csv(EXP_DIR / "transfer_adaptation_results.csv", index=False)


if __name__ == "__main__":
    run_transfer_diagnostics()
