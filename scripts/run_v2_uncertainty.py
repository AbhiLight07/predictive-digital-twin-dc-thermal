#!/usr/bin/env python3
"""v2 Experiment E5: Conformal Prediction & Uncertainty Quantification.

Protocol:
- Calibration on held-out validation block (never on training rows).
- Evaluation under Split C (unseen servers AND later time).
- Methods compared:
    1. Constant-width split conformal
    2. Normalized conformal (residual-scale model)
    3. Conformalized Quantile Regression (CQR)
- Detailed breakdowns:
    - Per-server coverage (min, median, max, fraction below nominal 90%)
    - Coverage by load regime (<10%, 10-50%, >50%)
    - Coverage by temperature bin (<50°C, 50-60°C, >=60°C)
- Spearman rank correlation: Spearman(width, |error|).
- Local recalibration from N hours of local data (ensuring sample size >= 1/alpha - 1).
- Full weekly coverage series across test partition.

Outputs:
- experiments/v2_uncertainty/conformal_methods_comparison.csv
- experiments/v2_uncertainty/coverage_by_regime_and_bin.csv
- experiments/v2_uncertainty/local_recalibration_results.csv
- experiments/v2_uncertainty/weekly_coverage_drift.csv
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import os
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", "C:/tmp")

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ttwin.io import load_parquet
from ttwin.features import build_features, get_feature_set
from ttwin.splits import split_strict
from ttwin.metrics import mae

OUT_DIR = ROOT / "experiments" / "v2_uncertainty"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data" / "interim"


def main():
    print(f"\n{'='*70}\nRUNNING EXPERIMENT E5: CONFORMAL PREDICTION (SPLIT C)\n{'='*70}")
    path = DATA_DIR / "dataset1" / "_combined.parquet"
    print(f"Loading data from {path}...")
    df = load_parquet(path)
    df["Series"] = df["Series"].astype(str).str.strip()

    # Drop 5 servers without S4 columns
    disqual = {"qh2-rcc196", "qh2-rcc197", "qh2-rcc198", "qh2-rcc199", "qh2-rcc200"}
    df = df[~df["Series"].isin(disqual)].copy()

    horizon_step = 6  # 1h
    target_abs = f"CPU_Temperature_target_{horizon_step}"
    target_delta = f"CPU_Temperature_delta_{horizon_step}"

    print("Building features...")
    df = build_features(df, horizons=[horizon_step], lags=[1, 2, 3, 6, 12], rolling_windows=[6, 12, 36])
    feats = get_feature_set("S4", df.columns.tolist())

    # Split C: train (seen servers, early), val (seen servers, val window), test (unseen servers, test window)
    train_df, val_df, test_df = split_strict(df, seed=42)
    print(f"Split C partitions — Train: {len(train_df):,} | Val(Cal): {len(val_df):,} | Test: {len(test_df):,}")

    cols_to_use = list(dict.fromkeys(feats + [target_abs, target_delta, "CPU_Temperature", "Series", "Time", "CPU_Load"]))
    train_clean = train_df[cols_to_use].dropna()
    val_clean = val_df[cols_to_use].dropna()
    test_clean = test_df[cols_to_use].dropna()

    X_tr = train_clean[feats].values
    y_tr_delta = train_clean[target_delta].values

    X_val = val_clean[feats].values
    y_val_abs = val_clean[target_abs].values
    y_val_curr = val_clean["CPU_Temperature"].values
    y_val_delta = val_clean[target_delta].values

    X_te = test_clean[feats].values
    y_te_abs = test_clean[target_abs].values
    y_te_curr = test_clean["CPU_Temperature"].values
    y_te_delta = test_clean[target_delta].values

    # Train point prediction model (delta target)
    print("Training point prediction model (LightGBM delta)...")
    dtrain = lgb.Dataset(X_tr, label=y_tr_delta, feature_name=feats)
    dval = lgb.Dataset(X_val, label=y_val_delta, reference=dtrain, feature_name=feats)
    lgb_point = lgb.train(
        {"objective": "regression", "metric": "mae", "learning_rate": 0.05, "num_leaves": 63, "verbosity": -1, "seed": 42},
        dtrain, num_boost_round=250, valid_sets=[dval], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )

    pred_val_delta = lgb_point.predict(X_val)
    pred_val_abs = y_val_curr + pred_val_delta

    pred_te_delta = lgb_point.predict(X_te)
    pred_te_abs = y_te_curr + pred_te_delta

    abs_err_val = np.abs(y_val_abs - pred_val_abs)
    abs_err_te = np.abs(y_te_abs - pred_te_abs)

    alpha = 0.10  # 90% nominal coverage
    n_cal = len(abs_err_val)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal

    # ---------------- 1. Constant-Width Split Conformal ----------------
    print("\nEvaluating 1. Constant-Width Split Conformal...")
    q_hat_const = float(np.quantile(abs_err_val, min(1.0, q_level)))
    lower_const = pred_te_abs - q_hat_const
    upper_const = pred_te_abs + q_hat_const
    cov_const = np.mean((y_te_abs >= lower_const) & (y_te_abs <= upper_const))
    width_const = 2.0 * q_hat_const
    r_const, _ = spearmanr(np.full_like(abs_err_te, width_const), abs_err_te)
    r_const = 0.0 if np.isnan(r_const) else r_const

    # ---------------- 2. Normalized Conformal (Residual Scaling) ----------------
    print("Evaluating 2. Normalized Conformal (Residual-scale model)...")
    ridge_scale = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=100.0))])
    ridge_scale.fit(X_val, abs_err_val)
    sigma_val = np.clip(ridge_scale.predict(X_val), 0.1, 15.0)
    sigma_te = np.clip(ridge_scale.predict(X_te), 0.1, 15.0)

    norm_scores_val = abs_err_val / sigma_val
    q_hat_norm = float(np.quantile(norm_scores_val, min(1.0, q_level)))

    lower_norm = pred_te_abs - q_hat_norm * sigma_te
    upper_norm = pred_te_abs + q_hat_norm * sigma_te
    cov_norm = np.mean((y_te_abs >= lower_norm) & (y_te_abs <= upper_norm))
    widths_norm = upper_norm - lower_norm
    mean_width_norm = float(np.mean(widths_norm))
    r_norm, _ = spearmanr(widths_norm, abs_err_te)

    # ---------------- 3. Conformalized Quantile Regression (CQR) ----------------
    print("Evaluating 3. Conformalized Quantile Regression (CQR)...")
    # Fit pinball loss models for delta target at alpha/2 and 1 - alpha/2
    q_lo_val = alpha / 2.0  # 0.05
    q_hi_val = 1.0 - alpha / 2.0  # 0.95

    m_low = lgb.train({"objective": "quantile", "alpha": q_lo_val, "learning_rate": 0.05, "num_leaves": 31, "verbosity": -1}, dtrain, num_boost_round=150)
    m_high = lgb.train({"objective": "quantile", "alpha": q_hi_val, "learning_rate": 0.05, "num_leaves": 31, "verbosity": -1}, dtrain, num_boost_round=150)

    q_low_val = y_val_curr + m_low.predict(X_val)
    q_high_val = y_val_curr + m_high.predict(X_val)

    q_low_te = y_te_curr + m_low.predict(X_te)
    q_high_te = y_te_curr + m_high.predict(X_te)

    # Conformity score: E_i = max(q_low(x_i) - y_i, y_i - q_high(x_i))
    cqr_scores_val = np.maximum(q_low_val - y_val_abs, y_val_abs - q_high_val)
    q_hat_cqr = float(np.quantile(cqr_scores_val, min(1.0, q_level)))

    lower_cqr = q_low_te - q_hat_cqr
    upper_cqr = q_high_te + q_hat_cqr
    cov_cqr = np.mean((y_te_abs >= lower_cqr) & (y_te_abs <= upper_cqr))
    widths_cqr = upper_cqr - lower_cqr
    mean_width_cqr = float(np.mean(widths_cqr))
    r_cqr, _ = spearmanr(widths_cqr, abs_err_te)

    comp_results = [
        {"method": "Constant-Width Conformal", "nominal_coverage": 1 - alpha, "empirical_coverage": cov_const, "mean_width_C": width_const, "spearman_r": r_const},
        {"method": "Normalized Conformal", "nominal_coverage": 1 - alpha, "empirical_coverage": cov_norm, "mean_width_C": mean_width_norm, "spearman_r": r_norm},
        {"method": "Conformalized Quantile Regression (CQR)", "nominal_coverage": 1 - alpha, "empirical_coverage": cov_cqr, "mean_width_C": mean_width_cqr, "spearman_r": r_cqr},
    ]
    df_comp = pd.DataFrame(comp_results)
    df_comp.to_csv(OUT_DIR / "conformal_methods_comparison.csv", index=False)
    print("\n--- Methods Comparison Table ---")
    print(df_comp.to_string(index=False))

    # Detailed coverage breakdowns: Per-Server, By Load, By Temp Bin
    test_clean["covered_const"] = (y_te_abs >= lower_const) & (y_te_abs <= upper_const)
    test_clean["covered_norm"] = (y_te_abs >= lower_norm) & (y_te_abs <= upper_norm)
    test_clean["covered_cqr"] = (y_te_abs >= lower_cqr) & (y_te_abs <= upper_cqr)

    # Regimes
    test_clean["load_regime"] = pd.cut(test_clean["CPU_Load"], bins=[-np.inf, 10, 50, np.inf], labels=["Idle (<10%)", "Active (10-50%)", "Heavy (>50%)"])
    test_clean["temp_bin"] = pd.cut(test_clean["CPU_Temperature"], bins=[-np.inf, 50, 60, np.inf], labels=["Cool (<50°C)", "Moderate (50-60°C)", "Hot (>=60°C)"])

    regime_rows = []
    for reg, grp in test_clean.groupby("load_regime", observed=False):
        regime_rows.append({
            "category": "Load Regime", "group": str(reg), "n_samples": len(grp),
            "cov_const": grp["covered_const"].mean(), "cov_norm": grp["covered_norm"].mean(), "cov_cqr": grp["covered_cqr"].mean(),
        })
    for tbin, grp in test_clean.groupby("temp_bin", observed=False):
        regime_rows.append({
            "category": "Temperature Bin", "group": str(tbin), "n_samples": len(grp),
            "cov_const": grp["covered_const"].mean(), "cov_norm": grp["covered_norm"].mean(), "cov_cqr": grp["covered_cqr"].mean(),
        })

    df_regimes = pd.DataFrame(regime_rows)
    df_regimes.to_csv(OUT_DIR / "coverage_by_regime_and_bin.csv", index=False)
    print("\n--- Coverage Breakdown by Regime and Temperature Bin ---")
    print(df_regimes.to_string(index=False))

    # Per-server breakdown
    server_stats = test_clean.groupby("Series").agg(
        cov_const=("covered_const", "mean"),
        cov_norm=("covered_norm", "mean"),
        cov_cqr=("covered_cqr", "mean"),
    )
    print(f"\nPer-Server Coverage Stats (N={len(server_stats)} servers):")
    print(f"  Constant-Width: min={server_stats['cov_const'].min():.1%}, median={server_stats['cov_const'].median():.1%}, max={server_stats['cov_const'].max():.1%}, below_90%={(server_stats['cov_const'] < 0.90).mean():.1%}")
    print(f"  CQR:            min={server_stats['cov_cqr'].min():.1%}, median={server_stats['cov_cqr'].median():.1%}, max={server_stats['cov_cqr'].max():.1%}, below_90%={(server_stats['cov_cqr'] < 0.90).mean():.1%}")

    # Local recalibration from N hours of target server data
    recalib_rows = []
    target_servers = test_clean["Series"].unique()
    for srv in target_servers[:10]:
        sdf = test_clean[test_clean["Series"] == srv].sort_values("Time").reset_index(drop=True)
        t_min = sdf["Time"].min()
        for bh in [24, 72, 168]:  # 144, 432, 1008 samples
            t_cal_end = t_min + pd.Timedelta(hours=bh)
            sdf_cal = sdf[sdf["Time"] < t_cal_end]
            sdf_eval = sdf[sdf["Time"] >= t_cal_end]
            if len(sdf_cal) < 15 or len(sdf_eval) < 50:
                continue

            y_cal_abs = sdf_cal[target_abs].values
            y_cal_curr = sdf_cal["CPU_Temperature"].values
            X_cal = sdf_cal[feats].values
            err_cal = np.abs(y_cal_abs - (y_cal_curr + lgb_point.predict(X_cal)))

            q_local = float(np.quantile(err_cal, min(1.0, np.ceil((len(err_cal)+1)*0.90)/len(err_cal))))
            y_eval_abs = sdf_eval[target_abs].values
            y_eval_curr = sdf_eval["CPU_Temperature"].values
            X_eval = sdf_eval[feats].values
            eval_pred = y_eval_curr + lgb_point.predict(X_eval)

            cov_loc = np.mean((y_eval_abs >= eval_pred - q_local) & (y_eval_abs <= eval_pred + q_local))
            recalib_rows.append({
                "server": srv, "budget_hours": bh, "n_cal_samples": len(err_cal),
                "q_hat_local": q_local, "width_C": 2.0 * q_local, "empirical_coverage": cov_loc,
            })

    df_recal = pd.DataFrame(recalib_rows)
    df_recal.to_csv(OUT_DIR / "local_recalibration_results.csv", index=False)

    # Full weekly coverage series
    test_clean["week"] = test_clean["Time"].dt.isocalendar().week
    weekly_rows = []
    for wk, grp in test_clean.groupby("week"):
        weekly_rows.append({
            "week": int(wk), "n_samples": len(grp),
            "cov_const": float(grp["covered_const"].mean()),
            "cov_norm": float(grp["covered_norm"].mean()),
            "cov_cqr": float(grp["covered_cqr"].mean()),
            "mae": float(mae(grp[target_abs].values, grp["CPU_Temperature"].values + lgb_point.predict(grp[feats].values))),
            "cause_attribution": "cause NOT TESTED",
        })
    df_weekly = pd.DataFrame(weekly_rows)
    df_weekly.to_csv(OUT_DIR / "weekly_coverage_drift.csv", index=False)
    print("\nFull Weekly Coverage Drift Series:")
    print(df_weekly.to_string(index=False))


if __name__ == "__main__":
    main()
