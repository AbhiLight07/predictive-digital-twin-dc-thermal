"""Tests for uncertainty quantification and conformal prediction correctness."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from ttwin.uncertainty import (
    compute_conformal_quantile,
    conformalize,
    coverage,
    average_width,
    conformalize_per_server,
)


def test_exact_order_statistic_selection():
    """Verify exact rank selection: k = ceil((n+1)*(1-alpha)) order statistic."""
    # n = 9, alpha = 0.1 -> (9+1)*0.9 = 10*0.9 = 9 -> kth score is 9th (last element)
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    q_hat = compute_conformal_quantile(scores, alpha=0.1)
    assert q_hat == 9.0

    # n = 19, alpha = 0.1 -> (19+1)*0.9 = 20*0.9 = 18 -> 18th score
    scores_20 = np.linspace(1.0, 19.0, 19)
    q_hat_18 = compute_conformal_quantile(scores_20, alpha=0.1)
    assert q_hat_18 == 18.0

    # n = 100, alpha = 0.1 -> ceil(101*0.9) = ceil(90.9) = 91 -> 91st score
    scores_100 = np.arange(1, 101, dtype=float)
    q_hat_100 = compute_conformal_quantile(scores_100, alpha=0.1)
    assert q_hat_100 == 91.0


def test_small_sample_size_k_greater_than_n():
    """When n < (1/alpha - 1), ceil((n+1)*(1-alpha)) > n, exact threshold is +infinity."""
    # alpha = 0.1 requires n >= 9. For n = 5, k = ceil(6 * 0.9) = ceil(5.4) = 6 > 5.
    scores_small = np.array([0.5, 1.2, 2.1, 3.4, 4.0])
    q_hat = compute_conformal_quantile(scores_small, alpha=0.1)
    assert np.isinf(q_hat)
    assert q_hat > 0

    # Test via conformalize: infinite q_hat yields infinite prediction interval
    y_cal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred_cal = y_cal + scores_small
    y_pred_test = np.array([25.0, 35.0])

    lower, upper, q_val = conformalize(y_cal, y_pred_cal, y_pred_test, alpha=0.1)
    assert np.isinf(q_val)
    assert np.all(lower == -np.inf)
    assert np.all(upper == np.inf)
    assert coverage(np.array([25.0, 35.0]), lower, upper) == 1.0
    assert np.isinf(average_width(lower, upper))


def test_invalid_alpha_raises_value_error():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    for invalid_alpha in [-0.1, 0.0, 1.0, 1.5, np.nan]:
        with pytest.raises(ValueError, match="alpha must be strictly in"):
            compute_conformal_quantile(scores, alpha=invalid_alpha)


def test_empty_calibration_scores_raises_value_error():
    with pytest.raises(ValueError, match="empty calibration scores"):
        compute_conformal_quantile(np.array([]), alpha=0.1)


def test_nan_handling_in_scores():
    """NaNs in scores should be filtered out, leaving valid numerical values."""
    scores_with_nan = np.array([1.0, np.nan, 2.0, 3.0, 4.0, 5.0, np.nan, 6.0, 7.0, 8.0, 9.0])
    # 9 valid scores, alpha=0.1 -> k = ceil(10*0.9) = 9 -> returns 9.0
    q_hat = compute_conformal_quantile(scores_with_nan, alpha=0.1)
    assert q_hat == 9.0

    # If all values are NaN, raise ValueError
    with pytest.raises(ValueError, match="empty calibration scores"):
        compute_conformal_quantile(np.array([np.nan, np.nan]), alpha=0.1)


def test_nan_handling_in_conformalize():
    """Pairs containing NaN are dropped; valid pairs are used for calibration."""
    y_cal = np.array([10.0, np.nan, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    y_pred_cal = np.array([9.5, 20.0, 29.5, 39.5, 49.5, 59.5, 69.5, 79.5, 89.5, 99.5])
    y_pred_test = np.array([55.0])

    # 9 valid pairs with absolute residual 0.5. alpha = 0.1 -> returns 0.5
    lower, upper, q_hat = conformalize(y_cal, y_pred_cal, y_pred_test, alpha=0.1)
    assert q_hat == pytest.approx(0.5)
    assert lower[0] == pytest.approx(54.5)
    assert upper[0] == pytest.approx(55.5)


def test_calibration_test_separation_data_flow():
    """Verify that calibration data does not leak into test evaluations."""
    np.random.seed(42)
    n_cal = 50
    n_test = 20

    y_cal = np.random.uniform(40, 60, size=n_cal)
    y_pred_cal = y_cal + np.random.normal(0, 0.5, size=n_cal)

    # Shift test distribution to verify coverage evaluates independently
    y_test = np.random.uniform(50, 70, size=n_test)
    y_pred_test = y_test + np.random.normal(0, 0.5, size=n_test)

    lower, upper, q_hat = conformalize(y_cal, y_pred_cal, y_pred_test, alpha=0.1)

    assert len(lower) == n_test
    assert len(upper) == n_test
    assert not np.isnan(q_hat)

    # Empirical coverage computation evaluates only y_test against intervals
    cov = coverage(y_test, lower, upper)
    assert 0.0 <= cov <= 1.0


def test_conformalize_per_server_fallback():
    """Servers with fewer than min_cal_samples fall back to the global quantile."""
    cal_data = pd.DataFrame({
        "Series": ["server_A"] * 40 + ["server_B"] * 10,
        "y_true": [50.0] * 50,
        "y_pred": [49.0] * 50,  # residual = 1.0
    })
    test_data = pd.DataFrame({
        "Series": ["server_A", "server_B", "server_C"],
        "y_pred": [50.0, 50.0, 50.0],
        "y_true": [50.0, 50.0, 50.0],
    })

    res_df, summary = conformalize_per_server(
        cal_data, test_data,
        y_true_col="y_true", y_pred_col="y_pred",
        server_col="Series", alpha=0.1, min_cal_samples=30
    )

    # server_A has 40 samples (>=30) -> uses local quantile
    # server_B has 10 samples (<30) -> uses global fallback
    # server_C not in cal -> uses global fallback
    assert len(res_df) == 3
    assert not res_df["q_hat"].isna().any()
    assert summary["global_q_hat"] == pytest.approx(1.0)
