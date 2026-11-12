#!/usr/bin/env python3
"""v2 Experiment E5: Conformal Prediction & Uncertainty Quantification.

Protocol & Methodology:
- 4-way disjoint chronological partition under Split C:
    1. Training set (seen servers, earliest 70% time): model fitting only.
    2. Model-selection set (seen servers, next 5% time): early stopping & residual-scale fitting only.
    3. Calibration set (seen servers, next 5% time): calculate conformal nonconformity scores & exact quantiles only.
    4. 6-hour embargo gap: strictly excluded between calibration and test.
    5. Test set (unseen servers, later 20% time - 6h embargo): final evaluation only.
- Strict data independence:
    - Early stopping and model selection NEVER see calibration rows.
    - Conformal conformity scores are computed strictly on held-out calibration rows.
    - Variance/scale estimators for normalized conformal are fitted on model-selection rows, NEVER on calibration rows.
    - Test rows are strictly held out and evaluated only after all models and thresholds are frozen.
- Conformal quantile calculation:
    - Exact finite-sample order statistic: k = ceil((n + 1) * (1 - alpha)).
    - Returns +inf if k > n.
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
from ttwin.splits import (
    split_strict_4way,
    verify_no_timestamp_overlap,
    verify_embargo_enforced,
    verify_no_server_overlap,
)
from ttwin.uncertainty import compute_conformal_quantile, coverage, average_width
from ttwin.metrics import mae

import argparse
parser = argparse.ArgumentParser(description="v2 Experiment E5: Conformal Prediction & Uncertainty Quantification.")
parser.add_argument("--out-prefix", default="v2", help="Prefix for output experiment directory, e.g. v2 or v3")
parser.add_argument("--force", action="store_true", help="Force rerun of model training and calibration, overwriting existing results.")
parser.add_argument("--reuse-existing", action="store_true", help="Load and print existing result CSVs without rerunning.")
parser.add_argument("--dry-run", action="store_true", help="Validate data loading, 4-way disjoint partitioning, and shapes without training models.")
args, _ = parser.parse_known_args()

OUT_DIR = ROOT / "experiments" / f"{args.out_prefix}_uncertainty"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data" / "interim"


def main():
    print(f"\n{'='*70}\nRUNNING EXPERIMENT E5: CONFORMAL PREDICTION (SPLIT C - 4-WAY DISJOINT)\n{'='*70}")

    expected_files = [
        OUT_DIR / "conformal_methods_comparison.csv",
        OUT_DIR / "coverage_by_regime_and_bin.csv",
        OUT_DIR / "local_recalibration_results.csv",
        OUT_DIR / "weekly_coverage_drift.csv",
    ]
    existing_files = [f for f in expected_files if f.exists()]

    if args.reuse_existing:
        if len(existing_files) == len(expected_files):
            print(f"Loading existing results from {OUT_DIR} (--reuse-existing specified)...")
            for f in expected_files:
                print(f"\n--- {f.name} ---")
                print(pd.read_csv(f).head(10).to_string(index=False))
            return
        else:
            print(f"Cannot --reuse-existing: Missing files in {OUT_DIR}. Found {len(existing_files)}/{len(expected_files)} files.")
            sys.exit(1)

    if not args.force and not args.dry_run:
        if len(existing_files) > 0:
            print(
                f"ERROR: Existing experiment results found in {OUT_DIR}.\n"
                "Under the corrected calibration independence methodology (4-way partition: train / val_select / val_cal / test),\n"
                "existing results are preserved but reflect legacy calibration reuse.\n"
                "To safely inspect existing results without rerunning: python scripts/run_v2_uncertainty.py --reuse-existing\n"
                "To test the 4-way data pipeline without expensive training: python scripts/run_v2_uncertainty.py --dry-run\n"
                "To recompute and overwrite with new models: python scripts/run_v2_uncertainty.py --force"
            )
            sys.exit(1)

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

    # 4-way disjoint chronological partition:
    # 1. train: 70% timeline, seen servers -> model fitting
    # 2. val_select: 5% timeline, seen servers -> early stopping & variance scaling fit
    # 3. val_cal: 5% timeline, seen servers -> calculate conformal residual scores only
    # 4. test: 20% timeline (after 6h embargo), unseen servers -> final evaluation
    train_df, val_select_df, val_cal_df, test_df = split_strict_4way(
        df,
        train_frac=0.70,
        val_select_frac=0.05,
        val_cal_frac=0.05,
        holdout_frac=0.20,
        embargo_steps=36,
        seed=42,
    )

    # Verification checks
    assert verify_no_timestamp_overlap(train_df, val_select_df), "Train and Val-Select share timestamps"
    assert verify_no_timestamp_overlap(val_select_df, val_cal_df), "Val-Select and Val-Cal share timestamps"
    assert verify_no_timestamp_overlap(val_cal_df, test_df), "Val-Cal and Test share timestamps"
    assert verify_embargo_enforced(val_cal_df, test_df, expected_embargo_steps=36), "Embargo gap violated"
    assert verify_no_server_overlap(train_df, test_df), "Server leakage into test set"

    print(
        f"Split C 4-Way Partitions:\n"
        f"  Train:      {len(train_df):,} rows ({train_df['Time'].min()} to {train_df['Time'].max()})\n"
        f"  Val-Select: {len(val_select_df):,} rows ({val_select_df['Time'].min()} to {val_select_df['Time'].max()})\n"
        f"  Val-Cal:    {len(val_cal_df):,} rows ({val_cal_df['Time'].min()} to {val_cal_df['Time'].max()})\n"
        f"  Embargo:    6 hours (excluded)\n"
        f"  Test:       {len(test_df):,} rows ({test_df['Time'].min()} to {test_df['Time'].max()})"
    )

    cols_to_use = list(dict.fromkeys(feats + [target_abs, target_delta, "CPU_Temperature", "Series", "Time", "CPU_Load"]))
    train_clean = train_df[cols_to_use].dropna()
    val_select_clean = val_select_df[cols_to_use].dropna()
    val_cal_clean = val_cal_df[cols_to_use].dropna()
    test_clean = test_df[cols_to_use].dropna()

    X_tr = train_clean[feats].values
    y_tr_delta = train_clean[target_delta].values

    X_val_sel = val_select_clean[feats].values
    y_val_sel_abs = val_select_clean[target_abs].values
    y_val_sel_curr = val_select_clean["CPU_Temperature"].values
    y_val_sel_delta = val_select_clean[target_delta].values

    X_val_cal = val_cal_clean[feats].values
    y_val_cal_abs = val_cal_clean[target_abs].values
    y_val_cal_curr = val_cal_clean["CPU_Temperature"].values
    y_val_cal_delta = val_cal_clean[target_delta].values

    X_te = test_clean[feats].values
    y_te_abs = test_clean[target_abs].values
    y_te_curr = test_clean["CPU_Temperature"].values
    y_te_delta = test_clean[target_delta].values

    if args.dry_run:
        print("\n[DRY RUN] All 4-way partitions verified successfully.")
        print(f"  Train clean rows:      {len(X_tr):,}")
        print(f"  Val-Select clean rows: {len(X_val_sel):,}")
        print(f"  Val-Cal clean rows:    {len(X_val_cal):,}")
        print(f"  Test clean rows:       {len(X_te):,}")
        print("  Timestamps strictly disjoint and ordered; 6-hour embargo enforced.")
        print("[DRY RUN] Exiting without expensive model training.")
        return

    # Train point prediction model (delta target) with early stopping strictly on Val-Select
    print("Training point prediction model (LightGBM delta) with early stopping on Val-Select...")
    dtrain = lgb.Dataset(X_tr, label=y_tr_delta, feature_name=feats)
    dval_sel = lgb.Dataset(X_val_sel, label=y_val_sel_delta, reference=dtrain, feature_name=feats)
    lgb_point = lgb.train(
        {"objective": "regression", "metric": "mae", "learning_rate": 0.05, "num_leaves": 63, "verbosity": -1, "seed": 42},
        dtrain, num_boost_round=250, valid_sets=[dval_sel], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )

    pred_val_sel_delta = lgb_point.predict(X_val_sel)
    pred_val_sel_abs = y_val_sel_curr + pred_val_sel_delta
    abs_err_val_sel = np.abs(y_val_sel_abs - pred_val_sel_abs)

    pred_val_cal_delta = lgb_point.predict(X_val_cal)
    pred_val_cal_abs = y_val_cal_curr + pred_val_cal_delta
    abs_err_val_cal = np.abs(y_val_cal_abs - pred_val_cal_abs)

    pred_te_delta = lgb_point.predict(X_te)
    pred_te_abs = y_te_curr + pred_te_delta
    abs_err_te = np.abs(y_te_abs - pred_te_abs)

    alpha = 0.10  # 90% nominal coverage

    # ---------------- 1. Constant-Width Split Conformal ----------------
    print("\nEvaluating 1. Constant-Width Split Conformal (calibrated strictly on Val-Cal)...")
    q_hat_const = compute_conformal_quantile(abs_err_val_cal, alpha=alpha)
    lower_const = pred_te_abs - q_hat_const
    upper_const = pred_te_abs + q_hat_const
    cov_const = coverage(y_te_abs, lower_const, upper_const)
    width_const = 2.0 * q_hat_const if not np.isinf(q_hat_const) else float("inf")
    r_const, _ = spearmanr(np.full_like(abs_err_te, width_const), abs_err_te)
    r_const = 0.0 if np.isnan(r_const) else r_const

    # ---------------- 2. Normalized Conformal (Residual Scaling) ----------------
    print("Evaluating 2. Normalized Conformal (Scale fitted on Val-Select, calibrated on Val-Cal)...")
    ridge_scale = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=100.0))])
    ridge_scale.fit(X_val_sel, abs_err_val_sel)
    sigma_cal = np.clip(ridge_scale.predict(X_val_cal), 0.1, 15.0)
    sigma_te = np.clip(ridge_scale.predict(X_te), 0.1, 15.0)

    norm_scores_cal = abs_err_val_cal / sigma_cal
    q_hat_norm = compute_conformal_quantile(norm_scores_cal, alpha=alpha)

    lower_norm = pred_te_abs - q_hat_norm * sigma_te
    upper_norm = pred_te_abs + q_hat_norm * sigma_te
    cov_norm = coverage(y_te_abs, lower_norm, upper_norm)
    widths_norm = upper_norm - lower_norm
    mean_width_norm = float(np.mean(widths_norm))
    r_norm, _ = spearmanr(widths_norm, abs_err_te)

    # ---------------- 3. Conformalized Quantile Regression (CQR) ----------------
    print("Evaluating 3. Conformalized Quantile Regression (CQR)...")
    q_lo_val = alpha / 2.0  # 0.05
    q_hi_val = 1.0 - alpha / 2.0  # 0.95

    m_low = lgb.train({"objective": "quantile", "alpha": q_lo_val, "learning_rate": 0.05, "num_leaves": 31, "verbosity": -1}, dtrain, num_boost_round=150)
    m_high = lgb.train({"objective": "quantile", "alpha": q_hi_val, "learning_rate": 0.05, "num_leaves": 31, "verbosity": -1}, dtrain, num_boost_round=150)

    q_low_cal = y_val_cal_curr + m_low.predict(X_val_cal)
    q_high_cal = y_val_cal_curr + m_high.predict(X_val_cal)

    q_low_te = y_te_curr + m_low.predict(X_te)
    q_high_te = y_te_curr + m_high.predict(X_te)

    # Conformity score: E_i = max(q_low(x_i) - y_i, y_i - q_high(x_i)) strictly on Val-Cal
    cqr_scores_cal = np.maximum(q_low_cal - y_val_cal_abs, y_val_cal_abs - q_high_cal)
    q_hat_cqr = compute_conformal_quantile(cqr_scores_cal, alpha=alpha)

    lower_cqr = q_low_te - q_hat_cqr
    upper_cqr = q_high_te + q_hat_cqr
    cov_cqr = coverage(y_te_abs, lower_cqr, upper_cqr)
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

            q_local = compute_conformal_quantile(err_cal, alpha=0.10)
            y_eval_abs = sdf_eval[target_abs].values
            y_eval_curr = sdf_eval["CPU_Temperature"].values
            X_eval = sdf_eval[feats].values
            eval_pred = y_eval_curr + lgb_point.predict(X_eval)

            cov_loc = coverage(y_eval_abs, eval_pred - q_local, eval_pred + q_local)
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
