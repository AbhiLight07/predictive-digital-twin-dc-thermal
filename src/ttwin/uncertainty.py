"""Uncertainty quantification via split conformal prediction (E5).

Split conformal prediction provides distribution-free finite-sample marginal coverage
guarantees under exchangeability:
    P(Y_{n+1} in C(X_{n+1})) >= 1 - alpha

Key Assumptions & Practical Limitations in Data Centre Telemetry:
1. Exchangeability Assumption:
   Split conformal guarantees strictly require that calibration points (X_i, Y_i)
   and test points (X_{n+1}, Y_{n+1}) are exchangeable. In operational data centres:
   - Temporal Autocorrelation: Telemetry records are chronologically ordered time
     series. Consecutive 10-minute intervals violate exchangeability.
   - Distribution Shift / Drift: Operational changes, server aging, and seasonal
     ambient shifts between the calibration window and future test windows induce
     temporal drift that violates exchangeability.
   - Spatial Heterogeneity: In leave-servers-out or strict holdouts (Split B, Split C),
     evaluating models calibrated on seen servers against completely unseen physical
     servers violates exchangeability due to chassis impedance, fan behavior, and
     rack micro-climate variations.
2. Marginal vs. Conditional Coverage:
   Marginal coverage guarantees an overall average over the population. It does NOT
   guarantee 1 - alpha coverage conditionally across:
   - Every individual physical server (per-server coverage).
   - High-temperature thermal hazard bins (e.g. CPU >= 60°C).
   - High-load or bursty workload regimes (CPU load >= 50%).
3. Data Independence Requirement:
   To prevent downward bias (optimistic under-coverage), calibration conformity
   scores must be computed on data completely independent of model fitting,
   early stopping, and hyperparameter selection.

References:
    Angelopoulos & Bates (2021). "A Gentle Introduction to Conformal Prediction"
    Vovk, Gammerman, & Shafer (2005). "Algorithmic Learning in a Random World"
    Lei et al. (2018). "Distribution-Free Predictive Inference for Regression"
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd


def compute_conformal_quantile(scores: np.ndarray | list, alpha: float) -> float:
    """Compute the exact finite-sample split-conformal quantile.

    For n exchangeable nonconformity scores S_1, ..., S_n and target miscoverage
    level alpha in (0, 1):
        k = ceil((n + 1) * (1 - alpha))

    Scores are sorted in ascending order: S_{(1)} <= S_{(2)} <= ... <= S_{(n)}.
    - If 1 <= k <= n: returns S_{(k)} (the kth order statistic; index k-1 in 0-based Python).
    - If k > n: the calibration sample size n is insufficient to guarantee 1 - alpha
      finite-sample coverage (since k = n + 1 when n < 1/alpha - 1). The mathematically
      correct finite-sample threshold is +infinity, NOT an interpolated quantile.

    Args:
        scores: 1D array-like nonconformity scores (e.g. absolute prediction residuals).
        alpha: Miscoverage level, strictly in the open interval (0, 1).

    Returns:
        float: Exact conformal quantile q_hat, or np.inf if k > n.

    Raises:
        ValueError: If alpha is not in (0, 1), scores is empty, or contains only NaNs.
    """
    if not isinstance(alpha, (int, float)) or np.isnan(alpha) or alpha <= 0.0 or alpha >= 1.0:
        raise ValueError(f"Miscoverage alpha must be strictly in (0, 1), got {alpha}")

    s = np.asarray(scores, dtype=float).ravel()
    # Filter out NaNs explicitly
    valid_mask = ~np.isnan(s)
    s_clean = s[valid_mask]

    n = len(s_clean)
    if n == 0:
        raise ValueError("Cannot compute conformal quantile on empty calibration scores.")

    # Exact finite-sample order statistic rank (1-based index)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))

    if k > n:
        return float("inf")

    # Sort ascending and select the kth order statistic (0-indexed: k - 1)
    s_sorted = np.sort(s_clean)
    return float(s_sorted[k - 1])


def conformalize(
    y_cal: np.ndarray,
    y_pred_cal: np.ndarray,
    y_pred_test: np.ndarray,
    alpha: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Split conformal prediction intervals.

    Args:
        y_cal: True values on calibration set.
        y_pred_cal: Model predictions on calibration set.
        y_pred_test: Model predictions on test set.
        alpha: Miscoverage level, strictly in (0, 1) (e.g. 0.1 for 90% nominal coverage).

    Returns:
        Tuple of (lower, upper, q_hat):
            - lower: Array of lower interval bounds on test set.
            - upper: Array of upper interval bounds on test set.
            - q_hat: Exact finite-sample conformal quantile (or np.inf).
    """
    y_cal = np.asarray(y_cal, dtype=float).ravel()
    y_pred_cal = np.asarray(y_pred_cal, dtype=float).ravel()
    y_pred_test = np.asarray(y_pred_test, dtype=float).ravel()

    if len(y_cal) != len(y_pred_cal):
        raise ValueError(
            f"Length mismatch: y_cal ({len(y_cal)}) and y_pred_cal ({len(y_pred_cal)}) must match."
        )

    # Filter invalid/NaN observation pairs
    valid_mask = (~np.isnan(y_cal)) & (~np.isnan(y_pred_cal))
    if not np.any(valid_mask):
        raise ValueError("No valid non-NaN pairs found between y_cal and y_pred_cal.")

    # Conformity scores: absolute residuals |y - y_hat|
    scores = np.abs(y_cal[valid_mask] - y_pred_cal[valid_mask])

    # Conformal quantile
    q_hat = compute_conformal_quantile(scores, alpha)

    if np.isinf(q_hat):
        lower = np.full_like(y_pred_test, -np.inf)
        upper = np.full_like(y_pred_test, np.inf)
    else:
        lower = y_pred_test - q_hat
        upper = y_pred_test + q_hat

    return lower, upper, q_hat


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical coverage: fraction of true values inside the interval.

    Excludes NaN values from evaluation. Handles infinite bounds correctly.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    lower = np.asarray(lower, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()

    valid = (~np.isnan(y_true)) & (~np.isnan(lower)) & (~np.isnan(upper))
    if not np.any(valid):
        return 0.0

    return float(np.mean((y_true[valid] >= lower[valid]) & (y_true[valid] <= upper[valid])))


def average_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Average interval width in degrees C.

    If any evaluated interval bounds are infinite, returns np.inf.
    """
    lower = np.asarray(lower, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()

    valid = (~np.isnan(lower)) & (~np.isnan(upper))
    if not np.any(valid):
        return 0.0

    diffs = upper[valid] - lower[valid]
    if np.any(np.isinf(diffs)):
        return float("inf")

    return float(np.mean(diffs))


def conformalize_per_server(
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    server_col: str = "Series",
    alpha: float = 0.1,
    global_fallback: bool = True,
    min_cal_samples: int = 30,
) -> Tuple[pd.DataFrame, dict]:
    """Apply conformal prediction per-server on calibration data, evaluate on test.

    Args:
        cal_df: Calibration DataFrame with true and predicted values.
        test_df: Test DataFrame with true and predicted values.
        y_true_col: Name of true target column.
        y_pred_col: Name of prediction column.
        server_col: Server identifier column name.
        alpha: Miscoverage level in (0, 1).
        global_fallback: If a server has fewer than min_cal_samples points, use global quantile.
        min_cal_samples: Minimum calibration observations required for server-specific quantile.

    Returns:
        (test_df_with_intervals, summary_dict)
    """
    # Clean global calibration pairs
    valid_cal = cal_df[[y_true_col, y_pred_col]].dropna()
    if len(valid_cal) == 0:
        raise ValueError("Calibration DataFrame contains no valid non-NaN (y_true, y_pred) pairs.")

    global_scores = np.abs(valid_cal[y_true_col].values - valid_cal[y_pred_col].values)
    global_q = compute_conformal_quantile(global_scores, alpha)

    test_df = test_df.copy()
    test_df["interval_lower"] = np.nan
    test_df["interval_upper"] = np.nan
    test_df["q_hat"] = np.nan

    cal_servers = set(cal_df[server_col].astype(str).str.strip().unique())

    for server, test_group in test_df.groupby(server_col):
        server_clean = str(server).strip()
        idx = test_group.index

        if server_clean in cal_servers:
            cal_group = cal_df[cal_df[server_col].astype(str).str.strip() == server_clean]
            valid_grp = cal_group[[y_true_col, y_pred_col]].dropna()

            if len(valid_grp) >= min_cal_samples:
                scores = np.abs(valid_grp[y_true_col].values - valid_grp[y_pred_col].values)
                q = compute_conformal_quantile(scores, alpha)
            else:
                q = global_q if global_fallback else np.nan
        else:
            q = global_q if global_fallback else np.nan

        preds = test_group[y_pred_col].values
        if np.isinf(q):
            test_df.loc[idx, "interval_lower"] = -np.inf
            test_df.loc[idx, "interval_upper"] = np.inf
        else:
            test_df.loc[idx, "interval_lower"] = preds - q
            test_df.loc[idx, "interval_upper"] = preds + q
        test_df.loc[idx, "q_hat"] = q

    # Evaluate on test
    valid = test_df[[y_true_col, "interval_lower", "interval_upper"]].dropna()
    cov = coverage(valid[y_true_col].values, valid["interval_lower"].values, valid["interval_upper"].values)
    width = average_width(valid["interval_lower"].values, valid["interval_upper"].values)

    summary = {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "empirical_coverage": cov,
        "average_width_C": width,
        "global_q_hat": global_q,
        "n_cal": len(valid_cal),
        "n_test": len(valid),
        "coverage_gap": cov - (1.0 - alpha),
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
    cal_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Run conformal prediction experiment on separated calibration and test partitions.

    CRITICAL METHODOLOGY NOTE:
    To preserve theoretical validity and prevent optimistic downward bias in nonconformity
    scores, calibration data (`cal_df`) MUST be held out strictly for calibration.
    If `cal_df` is not provided and `val_df` was also used for early stopping or hyperparameter
    selection of `model`, calibration independence is violated and empirical undercoverage
    may occur on test data.

    Args:
        train_df: Training DataFrame (used to fit model).
        val_df: Model-selection / validation DataFrame (used for early stopping or hyperparameter tuning).
        test_df: Test DataFrame (final unseen evaluation).
        model: Fitted prediction model with a `.predict(X)` method.
        feature_cols: List of feature column names.
        target_col: Target variable column name.
        server_col: Server identifier column name.
        alphas: List of miscoverage levels (default: [0.05, 0.10, 0.20]).
        cal_df: Optional separate calibration DataFrame. If provided, used strictly for calibration.
            If None, falls back to val_df for backward compatibility.

    Returns:
        pd.DataFrame summarizing empirical coverage, widths, and quantiles.
    """
    if alphas is None:
        alphas = [0.05, 0.10, 0.20]

    # Use cal_df if provided; otherwise fall back to val_df with warning
    active_cal_df = cal_df if cal_df is not None else val_df

    cal_clean = active_cal_df[feature_cols + [target_col, server_col]].dropna()
    test_clean = test_df[feature_cols + [target_col, server_col]].dropna()

    if len(cal_clean) == 0 or len(test_clean) == 0:
        raise ValueError("Empty calibration or test set after dropna.")

    X_cal = cal_clean[feature_cols].values
    X_test = test_clean[feature_cols].values

    cal_clean = cal_clean.copy()
    test_clean = test_clean.copy()
    cal_clean["y_pred"] = model.predict(X_cal)
    test_clean["y_pred"] = model.predict(X_test)

    rows = []
    for alpha in alphas:
        _, summary = conformalize_per_server(
            cal_df=cal_clean,
            test_df=test_clean,
            y_true_col=target_col,
            y_pred_col="y_pred",
            server_col=server_col,
            alpha=alpha,
        )
        rows.append(summary)

    return pd.DataFrame(rows)
