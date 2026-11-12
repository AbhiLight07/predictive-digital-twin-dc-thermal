#!/usr/bin/env python3
"""v2 Experiment E3: Cross-Cluster Hardware Transfer (Dell R840 -> Dell C6320).

Protocol:
- Corrected features: includes current-state T(t) in monitoring setting.
- Delta target: Delta_T = T(t+h) - T(t) for monitoring setting.
- Persistence baseline on Cluster 2 test partition.
- Two operational settings:
    1. Monitoring regime (with temperature history and T(t))
    2. Planning regime (workload-only, no temperature history)
- Transfer + adaptation with N local hours of Cluster 2 data.
- Descriptive hardware temperature distribution comparison (without claiming unverified physical comparability).

Outputs:
- experiments/v2_transfer/cross_cluster_monitoring_results.csv
- experiments/v2_transfer/cross_cluster_planning_results.csv
- experiments/v2_transfer/transfer_adaptation_results.csv
- experiments/v2_transfer/hardware_distributions.csv
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
from ttwin.features import build_features
from ttwin.splits import split_chronological
from ttwin.metrics import mae, rmse, skill_score_vs_persistence

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--out-prefix", default="v2", help="Prefix for output experiment directory, e.g. v2 or v3")
args, _ = parser.parse_known_args()

OUT_DIR = ROOT / "experiments" / f"{args.out_prefix}_transfer"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data" / "interim"

# 13 shared features for monitoring (including current T(t))
SHARED_MONITORING_FEATURES = [
    "CPU_Load", "Ram_Used", "No_Of_Running_vms",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "CPU_Temperature",  # current temperature at t
    "CPU_Temperature_lag_1", "CPU_Temperature_lag_2", "CPU_Temperature_lag_6",
    "CPU_Temperature_rmean_6", "CPU_Temperature_rstd_6",
]

# 7 shared features for planning (workload only, no temperature)
SHARED_PLANNING_FEATURES = [
    "CPU_Load", "Ram_Used", "No_Of_Running_vms",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


def prepare_cluster(dataset_name: str, horizon_step: int = 6):
    path = DATA_DIR / dataset_name / "_combined.parquet"
    print(f"Loading {dataset_name} from {path}...")
    df = load_parquet(path)
    df["Series"] = df["Series"].astype(str).str.strip()

    if dataset_name == "dataset2":
        disqual = {"qh2-rcc263", "qh2-rcc264", "qh2-rcc226", "qh2-rcc205"}
        df = df[~df["Series"].isin(disqual)].copy()

    df = build_features(df, horizons=[horizon_step], lags=[1, 2, 6], rolling_windows=[6])
    return df


def descriptive_hardware_comparison(df1, df2):
    print("\n" + "=" * 70 + "\nDESCRIPTIVE TEMPERATURE DISTRIBUTIONS (NO PHYSICAL CLAIMS)\n" + "=" * 70)
    t1 = df1["CPU_Temperature"].dropna()
    t2 = df2["CPU_Temperature"].dropna()
    idle1 = df1.loc[df1["CPU_Load"] < 10.0, "CPU_Temperature"].dropna()
    idle2 = df2.loc[df2["CPU_Load"] < 10.0, "CPU_Temperature"].dropna()

    stats = [
        {"metric": "Mean (°C)", "cluster_1_dell_r840": t1.mean(), "cluster_2_dell_c6320": t2.mean()},
        {"metric": "Std (°C)", "cluster_1_dell_r840": t1.std(), "cluster_2_dell_c6320": t2.std()},
        {"metric": "Min (°C)", "cluster_1_dell_r840": t1.min(), "cluster_2_dell_c6320": t2.min()},
        {"metric": "Median (°C)", "cluster_1_dell_r840": t1.median(), "cluster_2_dell_c6320": t2.median()},
        {"metric": "Max (°C)", "cluster_1_dell_r840": t1.max(), "cluster_2_dell_c6320": t2.max()},
        {"metric": "Idle Mean (<10% CPU) (°C)", "cluster_1_dell_r840": idle1.mean(), "cluster_2_dell_c6320": idle2.mean()},
        {"metric": "Num Servers", "cluster_1_dell_r840": df1["Series"].nunique(), "cluster_2_dell_c6320": df2["Series"].nunique()},
        {"metric": "Total Rows", "cluster_1_dell_r840": len(df1), "cluster_2_dell_c6320": len(df2)},
    ]
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(OUT_DIR / "hardware_distributions.csv", index=False)
    print(stats_df.to_string(index=False))


def main():
    print(f"\n{'='*70}\nRUNNING EXPERIMENT E3: CROSS-CLUSTER HARDWARE TRANSFER\n{'='*70}")
    horizon_step = 6  # 1h
    target_abs = f"CPU_Temperature_target_{horizon_step}"
    target_delta = f"CPU_Temperature_delta_{horizon_step}"

    df1 = prepare_cluster("dataset1", horizon_step)
    df2 = prepare_cluster("dataset2", horizon_step)

    descriptive_hardware_comparison(df1, df2)

    tr1, va1, te1 = split_chronological(df1, train_frac=0.70, val_frac=0.10, embargo_steps=horizon_step)
    tr2, va2, te2 = split_chronological(df2, train_frac=0.70, val_frac=0.10, embargo_steps=horizon_step)

    # ---------------- 1. MONITORING REGIME (DELTA TARGET + T(t)) ----------------
    print("\n" + "=" * 70 + "\n1. MONITORING REGIME CROSS-CLUSTER TRANSFER (Delta Target)\n" + "=" * 70)
    req_cols_mon = list(dict.fromkeys(SHARED_MONITORING_FEATURES + [target_abs, target_delta, "CPU_Temperature"]))

    tr1_mon = tr1[req_cols_mon].dropna()
    va1_mon = va1[req_cols_mon].dropna()
    tr2_mon = tr2[req_cols_mon].dropna()
    va2_mon = va2[req_cols_mon].dropna()
    te2_mon = te2[req_cols_mon].dropna()

    y_te2_abs = te2_mon[target_abs].values
    y_te2_curr = te2_mon["CPU_Temperature"].values
    X_te2_mon = te2_mon[SHARED_MONITORING_FEATURES].values

    # Persistence baseline
    mae_pers = mae(y_te2_abs, y_te2_curr)
    rmse_pers = rmse(y_te2_abs, y_te2_curr)
    print(f"Cluster 2 Test Persistence: MAE = {mae_pers:.4f}°C | RMSE = {rmse_pers:.4f}°C")

    # Native Cluster 2 model (Delta target)
    dtr2_mon = lgb.Dataset(tr2_mon[SHARED_MONITORING_FEATURES].values, label=tr2_mon[target_delta].values, feature_name=SHARED_MONITORING_FEATURES)
    dva2_mon = lgb.Dataset(va2_mon[SHARED_MONITORING_FEATURES].values, label=va2_mon[target_delta].values, reference=dtr2_mon, feature_name=SHARED_MONITORING_FEATURES)
    m_native_mon = lgb.train(
        {"objective": "regression", "metric": "mae", "learning_rate": 0.05, "num_leaves": 63, "verbosity": -1, "seed": 42},
        dtr2_mon, num_boost_round=250, valid_sets=[dva2_mon], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    pred_nat_abs = y_te2_curr + m_native_mon.predict(X_te2_mon)
    mae_nat_mon = mae(y_te2_abs, pred_nat_abs)
    rmse_nat_mon = rmse(y_te2_abs, pred_nat_abs)
    skill_nat_mon = skill_score_vs_persistence(mae_nat_mon, mae_pers)

    # Transferred model (trained on Cluster 1 Delta target)
    dtr1_mon = lgb.Dataset(tr1_mon[SHARED_MONITORING_FEATURES].values, label=tr1_mon[target_delta].values, feature_name=SHARED_MONITORING_FEATURES)
    dva1_mon = lgb.Dataset(va1_mon[SHARED_MONITORING_FEATURES].values, label=va1_mon[target_delta].values, reference=dtr1_mon, feature_name=SHARED_MONITORING_FEATURES)
    m_trans_mon = lgb.train(
        {"objective": "regression", "metric": "mae", "learning_rate": 0.05, "num_leaves": 63, "verbosity": -1, "seed": 42},
        dtr1_mon, num_boost_round=250, valid_sets=[dva1_mon], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    pred_trans_abs = y_te2_curr + m_trans_mon.predict(X_te2_mon)
    mae_trans_mon = mae(y_te2_abs, pred_trans_abs)
    rmse_trans_mon = rmse(y_te2_abs, pred_trans_abs)
    skill_trans_mon = skill_score_vs_persistence(mae_trans_mon, mae_pers)
    gap_mon = mae_trans_mon - mae_nat_mon

    print(f"  Native (C2):       MAE = {mae_nat_mon:.4f}°C | RMSE = {rmse_nat_mon:.4f}°C | Skill = {skill_nat_mon:+.2%}")
    print(f"  Transfer (C1->C2): MAE = {mae_trans_mon:.4f}°C | RMSE = {rmse_trans_mon:.4f}°C | Skill = {skill_trans_mon:+.2%}")
    print(f"  Transfer Gap:      dMAE = {gap_mon:+.4f}°C")

    res_mon = [
        {"model": "Persistence", "mae": mae_pers, "rmse": rmse_pers, "skill_vs_pers": 0.0},
        {"model": "Native Cluster 2", "mae": mae_nat_mon, "rmse": rmse_nat_mon, "skill_vs_pers": skill_nat_mon},
        {"model": "Transferred (C1->C2)", "mae": mae_trans_mon, "rmse": rmse_trans_mon, "skill_vs_pers": skill_trans_mon},
        {"model": "Transfer Penalty (Gap)", "mae": gap_mon, "rmse": np.nan, "skill_vs_pers": np.nan},
    ]
    pd.DataFrame(res_mon).to_csv(OUT_DIR / "cross_cluster_monitoring_results.csv", index=False)

    # ---------------- 2. PLANNING REGIME (WORKLOAD-ONLY) ----------------
    print("\n" + "=" * 70 + "\n2. PLANNING REGIME CROSS-CLUSTER TRANSFER (Workload-Only)\n" + "=" * 70)
    req_cols_plan = list(dict.fromkeys(SHARED_PLANNING_FEATURES + [target_abs]))

    tr1_plan = tr1[req_cols_plan].dropna()
    va1_plan = va1[req_cols_plan].dropna()
    tr2_plan = tr2[req_cols_plan].dropna()
    va2_plan = va2[req_cols_plan].dropna()
    te2_plan = te2[req_cols_plan].dropna()

    y_te2_plan_abs = te2_plan[target_abs].values
    X_te2_plan = te2_plan[SHARED_PLANNING_FEATURES].values

    # Baselines: Cluster 2 train mean
    c2_mean = float(tr2_plan[target_abs].mean())
    mae_c2_mean = mae(y_te2_plan_abs, np.full_like(y_te2_plan_abs, c2_mean))
    print(f"Cluster 2 Mean Baseline: MAE = {mae_c2_mean:.4f}°C")

    # Native Cluster 2 planning model
    dtr2_plan = lgb.Dataset(tr2_plan[SHARED_PLANNING_FEATURES].values, label=tr2_plan[target_abs].values, feature_name=SHARED_PLANNING_FEATURES)
    dva2_plan = lgb.Dataset(va2_plan[SHARED_PLANNING_FEATURES].values, label=va2_plan[target_abs].values, reference=dtr2_plan, feature_name=SHARED_PLANNING_FEATURES)
    m_native_plan = lgb.train(
        {"objective": "regression", "metric": "mae", "learning_rate": 0.05, "num_leaves": 63, "verbosity": -1, "seed": 42},
        dtr2_plan, num_boost_round=250, valid_sets=[dva2_plan], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    pred_nat_plan = m_native_plan.predict(X_te2_plan)
    mae_nat_plan = mae(y_te2_plan_abs, pred_nat_plan)

    # Transferred planning model (Cluster 1 -> Cluster 2)
    dtr1_plan = lgb.Dataset(tr1_plan[SHARED_PLANNING_FEATURES].values, label=tr1_plan[target_abs].values, feature_name=SHARED_PLANNING_FEATURES)
    dva1_plan = lgb.Dataset(va1_plan[SHARED_PLANNING_FEATURES].values, label=va1_plan[target_abs].values, reference=dtr1_plan, feature_name=SHARED_PLANNING_FEATURES)
    m_trans_plan = lgb.train(
        {"objective": "regression", "metric": "mae", "learning_rate": 0.05, "num_leaves": 63, "verbosity": -1, "seed": 42},
        dtr1_plan, num_boost_round=250, valid_sets=[dva1_plan], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    pred_trans_plan = m_trans_plan.predict(X_te2_plan)
    mae_trans_plan = mae(y_te2_plan_abs, pred_trans_plan)
    gap_plan = mae_trans_plan - mae_nat_plan

    print(f"  Native (C2 Planning):       MAE = {mae_nat_plan:.4f}°C")
    print(f"  Transfer (C1->C2 Planning): MAE = {mae_trans_plan:.4f}°C")
    print(f"  Planning Transfer Gap:      dMAE = {gap_plan:+.4f}°C")

    res_plan = [
        {"model": "Cluster 2 Mean Baseline", "mae": mae_c2_mean},
        {"model": "Native Cluster 2 Planning", "mae": mae_nat_plan},
        {"model": "Transferred Planning", "mae": mae_trans_plan},
        {"model": "Planning Transfer Gap", "mae": gap_plan},
    ]
    pd.DataFrame(res_plan).to_csv(OUT_DIR / "cross_cluster_planning_results.csv", index=False)

    # ---------------- 3. TRANSFER + ADAPTATION (CLUSTER 2 DATA) ----------------
    print("\n" + "=" * 70 + "\n3. TRANSFER + ADAPTATION WITH N HOURS OF CLUSTER 2 DATA\n" + "=" * 70)
    # In Dataset 2, temperature sensors did not record valid values until 2019-02-07 16:00:00.
    # We define t2_min from the first timestamp with valid temperature telemetry.
    t2_min = tr2.dropna(subset=["CPU_Temperature", target_delta, "CPU_Temperature_lag_6"])["Time"].min()
    print(f"Beginning adaptation sampling from first valid timestamp: {t2_min}")
    adapt_results = []

    for bh in [1, 6, 24, 72, 168]:
        t_end = t2_min + pd.Timedelta(hours=bh)
        ad_slice = tr2[tr2["Time"] < t_end][req_cols_mon].dropna()
        if len(ad_slice) < 10:
            continue

        X_ad = ad_slice[SHARED_MONITORING_FEATURES].values
        y_ad_delta = ad_slice[target_delta].values

        # Scalar residual offset on monitoring transfer
        pred_ad_trans = m_trans_mon.predict(X_ad)
        offset = float(np.mean(y_ad_delta - pred_ad_trans))
        p_adapt_abs = y_te2_curr + (m_trans_mon.predict(X_te2_mon) + offset)
        mae_ad_offset = mae(y_te2_abs, p_adapt_abs)
        skill_ad_offset = skill_score_vs_persistence(mae_ad_offset, mae_pers)

        # Continued boosting
        dtr_ad = lgb.Dataset(X_ad, label=y_ad_delta, feature_name=SHARED_MONITORING_FEATURES)
        adapted_booster = lgb.train(
            {"objective": "regression", "metric": "mae", "learning_rate": 0.01, "num_leaves": 31, "min_child_samples": 5, "verbosity": -1},
            dtr_ad, num_boost_round=min(30, max(5, len(ad_slice) // 20)), init_model=m_trans_mon,
        )
        p_cb_abs = y_te2_curr + adapted_booster.predict(X_te2_mon)
        mae_ad_cb = mae(y_te2_abs, p_cb_abs)
        skill_ad_cb = skill_score_vs_persistence(mae_ad_cb, mae_pers)

        print(f"  Budget {bh:3d}h ({len(ad_slice):5d} samples): Offset MAE={mae_ad_offset:.4f}°C (Skill={skill_ad_offset:+.2%}) | Continued Boosting MAE={mae_ad_cb:.4f}°C (Skill={skill_ad_cb:+.2%})")

        adapt_results.append({
            "budget_hours": bh, "n_samples": len(ad_slice),
            "offset_mae": mae_ad_offset, "offset_skill": skill_ad_offset,
            "continued_boosting_mae": mae_ad_cb, "continued_boosting_skill": skill_ad_cb,
        })

    pd.DataFrame(adapt_results).to_csv(OUT_DIR / "transfer_adaptation_results.csv", index=False)
    print(f"\nExperiment E3 complete! Results saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
