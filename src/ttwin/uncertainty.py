"""Uncertainty quantification via split conformal prediction (E5).

Split conformal prediction provides marginal coverage guarantees:
  P(Y_new in C(X_new)) >= 1 - alpha

for exchangeable calibration and test samples.

References:
  Angelopoulos & Bates (2021). "A Gentle Introduction to Conformal Prediction"
  Lei et al. (2018). "Distribution-Free Predictive Inference for Regression"
"""

import numpy as np
import pandas as pd
from typing import Tuple


def conformalize(
    y_cal: np.ndarray,
    y_pred_cal: np.ndarray,
    y_pred_test: np.ndarray,
    alpha: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Split conformal prediction intervals.

    Args:
        y_cal: true values on calibration set
        y_pred_cal: model predictions on calibration set
        y_pred_test: model predictions on test set
        alpha: miscoverage level (0.1 = 90% coverage target)

    Returns:
        (lower, upper, q_hat) where lower/upper are prediction intervals
        and q_hat is the conformal quantile.
    """
    # Conformity scores: absolute residuals
    scores = np.abs(y_cal - y_pred_cal)

    # Conformal quantile (finite-sample correction)
    n = len(scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    q_hat = float(np.quantile(scores, level))

    lower = y_pred_test - q_hat
    upper = y_pred_test + q_hat

    return lower, upper, q_hat


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical coverage: fraction of true values inside the interval."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def average_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Average interval width in degrees C."""
    return float(np.mean(upper - lower))


def conformalize_per_server(
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    server_col: str = "Series",
    alpha: float = 0.1,
    global_fallback: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Apply conformal prediction per-server on calibration, evaluate on test.

    Args:
        cal_df: calibration DataFrame with true and predicted values
        test_df: test DataFrame with predicted values (and true for evaluation)
        y_true_col: name of true value column
        y_pred_col: name of prediction column
        server_col: server identifier column
        alpha: miscoverage level
        global_fallback: if a server has <30 cal points, use global quantile

    Returns:
        (test_df_with_intervals, summary_dict)
    """
    # Compute global quantile as fallback
    global_scores = np.abs(cal_df[y_true_col].values - cal_df[y_pred_col].values)
    n_global = len(global_scores)
    level_global = min(np.ceil((n_global + 1) * (1 - alpha)) / n_global, 1.0)
    global_q = float(np.quantile(global_scores, level_global))

    test_df = test_df.copy()
    test_df["interval_lower"] = np.nan
    test_df["interval_upper"] = np.nan
    test_df["q_hat"] = np.nan

    cal_servers = set(cal_df[server_col].unique())

    for server, test_group in test_df.groupby(server_col):
        idx = test_group.index
        if server in cal_servers:
            cal_group = cal_df[cal_df[server_col] == server]
            if len(cal_group) >= 30:
                y_cal = cal_group[y_true_col].dropna().values
                y_pred_c = cal_group.loc[cal_group[y_true_col].notna(), y_pred_col].values
                min_len = min(len(y_cal), len(y_pred_c))
                y_cal, y_pred_c = y_cal[:min_len], y_pred_c[:min_len]
                scores = np.abs(y_cal - y_pred_c)
                n = len(scores)
                level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
                q = float(np.quantile(scores, level))
            else:
                q = global_q if global_fallback else np.nan
        else:
            q = global_q if global_fallback else np.nan

        preds = test_group[y_pred_col].values
        test_df.loc[idx, "interval_lower"] = preds - q
        test_df.loc[idx, "interval_upper"] = preds + q
        test_df.loc[idx, "q_hat"] = q

    # Evaluate
    valid = test_df[[y_true_col, "interval_lower", "interval_upper"]].dropna()
    cov = coverage(valid[y_true_col].values, valid["interval_lower"].values, valid["interval_upper"].values)
    width = average_width(valid["interval_lower"].values, valid["interval_upper"].values)

    summary = {
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "empirical_coverage": cov,
        "average_width_C": width,
        "global_q_hat": global_q,
        "n_cal": len(cal_df),
        "n_test": len(valid),
        "coverage_gap": cov - (1 - alpha),
    }

    return test_df, summary


def run_conformal_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model,
    feature_cols: list,
    target_col: str,
    server_col: str = "Series",
    alphas: list = None,
) -> pd.DataFrame:
    """Full conformal prediction experiment.

    Uses val_df as calibration set. Evaluates on test_df.

    Returns DataFrame with columns: alpha, target_coverage, empirical_coverage,
    average_width_C, coverage_gap.
    """
    if alphas is None:
        alphas = [0.05, 0.10, 0.20]

    # Get predictions on val (calibration) and test
    val_clean = val_df[feature_cols + [target_col, server_col]].dropna()
    test_clean = test_df[feature_cols + [target_col, server_col]].dropna()

    if len(val_clean) == 0 or len(test_clean) == 0:
        raise ValueError("Empty calibration or test set after dropna")

    X_val = val_clean[feature_cols].values
    X_test = test_clean[feature_cols].values

    val_clean = val_clean.copy()
    test_clean = test_clean.copy()
    val_clean["y_pred"] = model.predict(X_val)
    test_clean["y_pred"] = model.predict(X_test)

    rows = []
    for alpha in alphas:
        _, summary = conformalize_per_server(
            cal_df=val_clean,
            test_df=test_clean,
            y_true_col=target_col,
            y_pred_col="y_pred",
            server_col=server_col,
            alpha=alpha,
        )
        rows.append(summary)

    return pd.DataFrame(rows)
