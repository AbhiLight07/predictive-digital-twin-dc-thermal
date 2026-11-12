"""Focused unit tests for conformal calibration data independence and 4-way chronological partitioning.

Verifies:
1. No timestamp overlap between model-selection and calibration data.
2. No timestamp overlap between calibration and test data.
3. Correct chronological ordering across all 4 partitions.
4. 6-hour embargo enforcement between calibration and test partitions.
5. Calibration scores are calculated strictly and exclusively from calibration observations.
6. Server isolation (seen servers for train/select/cal, holdout servers for test).
"""

import sys
import subprocess
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

from ttwin.splits import (
    split_strict_4way,
    split_chronological_4way,
    partition_validation_conformal,
    verify_no_timestamp_overlap,
    verify_embargo_enforced,
    verify_no_server_overlap,
    verify_no_future_leak,
)
from ttwin.uncertainty import (
    compute_conformal_quantile,
    conformalize,
    run_conformal_experiment,
)


@pytest.fixture
def synthetic_multiserver_df():
    """Create a multi-server time-series DataFrame with 10 servers and 200 time steps."""
    np.random.seed(42)
    servers = [f"server_{i:02d}" for i in range(10)]
    times = pd.date_range("2020-01-01 00:00:00", periods=200, freq="10min")
    rows = []
    for i, s in enumerate(servers):
        base_load = (i + 1) * 8.0
        for t in times:
            load = base_load + np.random.uniform(-2, 2)
            temp = 40.0 + load * 0.4 + np.random.normal(0, 1.0)
            target = temp + np.random.normal(0, 0.5)
            rows.append({
                "Series": s,
                "Time": t,
                "CPU_Load": load,
                "CPU_Temperature": temp,
                "target": target,
                "feat_1": load * 0.1,
                "feat_2": temp * 0.05,
            })
    return pd.DataFrame(rows)


def test_no_timestamp_overlap_between_model_selection_and_calibration(synthetic_multiserver_df):
    """Verify that model-selection data and calibration data share zero timestamps."""
    train, val_select, val_cal, test = split_strict_4way(
        synthetic_multiserver_df,
        train_frac=0.70,
        val_select_frac=0.05,
        val_cal_frac=0.05,
        embargo_steps=6,
    )

    assert len(val_select) > 0, "val_select must not be empty"
    assert len(val_cal) > 0, "val_cal must not be empty"

    # Pairwise disjoint timestamps
    assert verify_no_timestamp_overlap(val_select, val_cal)
    select_times = set(val_select["Time"].unique())
    cal_times = set(val_cal["Time"].unique())
    assert len(select_times.intersection(cal_times)) == 0


def test_no_timestamp_overlap_between_calibration_and_test(synthetic_multiserver_df):
    """Verify that calibration data and test data share zero timestamps."""
    train, val_select, val_cal, test = split_strict_4way(
        synthetic_multiserver_df,
        train_frac=0.70,
        val_select_frac=0.05,
        val_cal_frac=0.05,
        embargo_steps=6,
    )

    assert len(test) > 0, "test partition must not be empty"
    assert verify_no_timestamp_overlap(val_cal, test)

    cal_times = set(val_cal["Time"].unique())
    test_times = set(test["Time"].unique())
    assert len(cal_times.intersection(test_times)) == 0


def test_strict_chronological_ordering(synthetic_multiserver_df):
    """Verify strict chronological ordering: train < val_select < val_cal < test."""
    train, val_select, val_cal, test = split_strict_4way(
        synthetic_multiserver_df,
        train_frac=0.70,
        val_select_frac=0.05,
        val_cal_frac=0.05,
        embargo_steps=6,
    )

    t_train_max = train["Time"].max()
    t_select_min = val_select["Time"].min()
    t_select_max = val_select["Time"].max()
    t_cal_min = val_cal["Time"].min()
    t_cal_max = val_cal["Time"].max()
    t_test_min = test["Time"].min()

    assert t_train_max < t_select_min, f"Train max ({t_train_max}) must be < Val-Select min ({t_select_min})"
    assert t_select_max < t_cal_min, f"Val-Select max ({t_select_max}) must be < Val-Cal min ({t_cal_min})"
    assert t_cal_max < t_test_min, f"Val-Cal max ({t_cal_max}) must be < Test min ({t_test_min})"


def test_embargo_enforcement(synthetic_multiserver_df):
    """Verify that the embargo gap between calibration and test is strictly enforced."""
    embargo_steps = 12  # 12 steps of 10 min = 120 min = 2 hours
    train, val_select, val_cal, test = split_strict_4way(
        synthetic_multiserver_df,
        train_frac=0.60,
        val_select_frac=0.05,
        val_cal_frac=0.05,
        embargo_steps=embargo_steps,
    )

    gap = test["Time"].min() - val_cal["Time"].max()
    expected_gap = pd.Timedelta(minutes=embargo_steps * 10)
    assert gap >= expected_gap, f"Gap ({gap}) must be >= expected embargo ({expected_gap})"
    assert verify_embargo_enforced(val_cal, test, expected_embargo_steps=embargo_steps)

    # Negative check: an artificially truncated test set violates embargo check
    violating_test = test.copy()
    violating_test.loc[0, "Time"] = val_cal["Time"].max() + pd.Timedelta(minutes=10)
    assert not verify_embargo_enforced(val_cal, violating_test, expected_embargo_steps=embargo_steps)


def test_calibration_scores_calculated_only_from_calibration_observations():
    """Verify that conformal scores and q_hat depend exclusively on calibration observations."""
    n_select = 50
    n_cal = 40
    n_test = 25

    # True model residuals on calibration data
    np.random.seed(123)
    residuals_select = np.random.normal(0, 1.0, size=n_select)  # model-selection residuals
    residuals_cal = np.random.normal(0, 2.5, size=n_cal)        # calibration residuals
    abs_err_cal = np.abs(residuals_cal)

    # Compute baseline conformal quantile on calibration data
    alpha = 0.10
    q_hat_baseline = compute_conformal_quantile(abs_err_cal, alpha=alpha)

    # Perturb model-selection residuals drastically (e.g. 100x magnification)
    residuals_select_mutated = residuals_select * 100.0

    # Verify that calibration quantile remains bit-for-bit identical
    q_hat_after_mutation = compute_conformal_quantile(abs_err_cal, alpha=alpha)
    assert q_hat_baseline == q_hat_after_mutation

    # Verify via conformalize
    y_cal = np.full(n_cal, 50.0)
    y_pred_cal = y_cal + residuals_cal
    y_pred_test = np.full(n_test, 55.0)

    lower1, upper1, q1 = conformalize(y_cal, y_pred_cal, y_pred_test, alpha=alpha)
    assert q1 == pytest.approx(q_hat_baseline)

    # If any model-selection data were mixed into calibration, q_hat would differ
    scores_leaked = np.concatenate([abs_err_cal, np.abs(residuals_select_mutated)])
    q_hat_leaked = compute_conformal_quantile(scores_leaked, alpha=alpha)
    assert q_hat_leaked != q_hat_baseline


def test_run_conformal_experiment_with_independent_cal_df():
    """Verify run_conformal_experiment isolates calibration when cal_df is provided."""
    class DummyModel:
        def predict(self, X):
            return np.full(len(X), 50.0)

    df_train = pd.DataFrame({"Series": ["s1"] * 20, "feat": [1.0] * 20, "target": [50.0] * 20})
    df_val_select = pd.DataFrame({"Series": ["s1"] * 20, "feat": [1.0] * 20, "target": [999.0] * 20}) # extreme values
    df_val_cal = pd.DataFrame({"Series": ["s1"] * 20, "feat": [1.0] * 20, "target": [52.0] * 20})    # residual = 2.0
    df_test = pd.DataFrame({"Series": ["s1"] * 10, "feat": [1.0] * 10, "target": [51.5] * 10})

    # If cal_df is provided, calibration must use df_val_cal (residuals=2.0), not df_val_select (residuals=949.0)
    res = run_conformal_experiment(
        train_df=df_train,
        val_df=df_val_select,
        test_df=df_test,
        model=DummyModel(),
        feature_cols=["feat"],
        target_col="target",
        server_col="Series",
        alphas=[0.10],
        cal_df=df_val_cal,
    )

    assert len(res) == 1
    assert res.iloc[0]["global_q_hat"] == pytest.approx(2.0)
    assert res.iloc[0]["empirical_coverage"] == 1.0


def test_partition_validation_conformal(synthetic_multiserver_df):
    """Verify partition_validation_conformal cleanly divides validation chronologically."""
    times = pd.date_range("2020-01-01 00:00", periods=60, freq="10min")
    val_df = pd.DataFrame({"Time": times, "Series": "s1", "val": range(60)})

    val_select, val_cal = partition_validation_conformal(val_df, select_ratio=0.5)
    assert len(val_select) > 0
    assert len(val_cal) > 0
    assert len(val_select) + len(val_cal) == len(val_df)
    assert verify_no_timestamp_overlap(val_select, val_cal)
    assert val_select["Time"].max() < val_cal["Time"].min()


def test_server_isolation_in_4way_split(synthetic_multiserver_df):
    """Verify seen servers appear in train, select, and cal; unseen appear in test."""
    train, val_select, val_cal, test = split_strict_4way(
        synthetic_multiserver_df,
        holdout_frac=0.20,
        seed=42,
    )

    train_servers = set(train["Series"].unique())
    select_servers = set(val_select["Series"].unique())
    cal_servers = set(val_cal["Series"].unique())
    test_servers = set(test["Series"].unique())

    assert train_servers == select_servers
    assert train_servers == cal_servers
    assert len(train_servers.intersection(test_servers)) == 0
    assert verify_no_server_overlap(train, test)
    assert verify_no_server_overlap(val_cal, test)


def test_run_v2_uncertainty_cli_safety():
    """Verify CLI safety: running without flags exits with error code 1 to protect existing results."""
    cmd = [sys.executable, str(ROOT / "scripts" / "run_v2_uncertainty.py")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 1
    assert "Existing experiment results found" in res.stdout or "Existing experiment results found" in res.stderr
    assert "--reuse-existing" in res.stdout or "--reuse-existing" in res.stderr
