"""Tests to verify feature engineering includes current-state t but forbids future values > t."""
import pytest
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, "d:/mainel5/src")

from ttwin.features import (
    add_lag_features, add_rolling_features, create_target,
    add_calendar_features, compute_effective_cpu_temp, get_feature_set,
    build_features,
)


@pytest.fixture
def sample_server_df():
    """Single server time series with deterministic values."""
    times = pd.date_range("2019-01-01", periods=50, freq="10min")
    return pd.DataFrame({
        "Series": "server_A",
        "Time": times,
        "CPU_Temperature": np.arange(50, dtype=float) + 30,
        "CPU1_Temperature": np.arange(50, dtype=float) + 30,
        "CPU2_Temperature": np.arange(50, dtype=float) + 28,
        "CPU_Load": np.random.uniform(0, 100, 50),
        "Inlet_Temperature": np.ones(50) * 22.0,
        "Power": np.ones(50) * 250.0,
        "Fan_speed1": np.ones(50) * 4500.0,
    })


def test_lag_features_no_future(sample_server_df):
    """Lag 1 uses observation at t-1, lag 3 uses observation at t-3."""
    df = add_lag_features(sample_server_df, ["CPU_Temperature"], [1, 3])
    for i in range(1, len(df)):
        assert df.iloc[i]["CPU_Temperature_lag_1"] == df.iloc[i - 1]["CPU_Temperature"]
    assert pd.isna(df.iloc[0]["CPU_Temperature_lag_1"])


def test_rolling_features_end_at_t(sample_server_df):
    """Rolling features end at time t: includes observation at t and w-1 preceding."""
    df = add_rolling_features(sample_server_df, ["CPU_Temperature"], [3])
    for i in range(2, len(df)):
        # Window of size 3 ending at i: [i-2, i-1, i]
        vals = sample_server_df["CPU_Temperature"].iloc[i - 2:i + 1].values
        expected = np.mean(vals)
        np.testing.assert_almost_equal(
            df.iloc[i]["CPU_Temperature_rmean_3"], expected, decimal=5
        )


def test_perturbing_future_leaves_features_at_t_unchanged(sample_server_df):
    """Perturbing any rows after t must leave all features at time t completely unchanged."""
    df_orig = build_features(sample_server_df.copy(), horizons=[1, 6])
    feature_cols = get_feature_set("S4", df_orig.columns.tolist())

    t_idx = 25  # evaluate at step 25
    features_at_t_orig = df_orig.iloc[t_idx][feature_cols].copy()

    # Perturb all future rows (> t_idx)
    df_perturbed = sample_server_df.copy()
    df_perturbed.loc[t_idx + 1:, "CPU_Temperature"] += 999.0
    df_perturbed.loc[t_idx + 1:, "CPU1_Temperature"] += 999.0
    df_perturbed.loc[t_idx + 1:, "CPU2_Temperature"] += 999.0
    df_perturbed.loc[t_idx + 1:, "CPU_Load"] = 100.0

    df_perturbed_feats = build_features(df_perturbed, horizons=[1, 6])
    features_at_t_pert = df_perturbed_feats.iloc[t_idx][feature_cols].copy()

    # Features at t must be strictly identical
    pd.testing.assert_series_equal(features_at_t_orig, features_at_t_pert)


def test_target_creation_correct_horizon_and_delta(sample_server_df):
    """Target creates absolute T(t+h) and delta T(t+h) - T(t)."""
    df = create_target(sample_server_df, "CPU_Temperature", horizon_steps=3)
    target_col = "CPU_Temperature_target_3"
    delta_col = "CPU_Temperature_delta_3"
    assert target_col in df.columns
    assert delta_col in df.columns
    for i in range(len(df) - 3):
        assert df.iloc[i][target_col] == sample_server_df.iloc[i + 3]["CPU_Temperature"]
        assert df.iloc[i][delta_col] == sample_server_df.iloc[i + 3]["CPU_Temperature"] - sample_server_df.iloc[i]["CPU_Temperature"]
    for i in range(len(df) - 3, len(df)):
        assert pd.isna(df.iloc[i][target_col])
        assert pd.isna(df.iloc[i][delta_col])


def test_feature_set_s1_and_s4_include_current_temp(sample_server_df):
    """S1 and S4 must include current temperature state at t."""
    df = build_features(sample_server_df)
    s1_cols = get_feature_set("S1", df.columns.tolist())
    s4_cols = get_feature_set("S4", df.columns.tolist())

    assert "CPU_Temperature" in s1_cols
    assert "CPU_Temperature" in s4_cols
    assert "CPU1_Temperature" in s1_cols
    assert "CPU2_Temperature" in s1_cols


def test_effective_cpu_temp_is_max(sample_server_df):
    df = compute_effective_cpu_temp(sample_server_df)
    expected = sample_server_df[["CPU1_Temperature", "CPU2_Temperature"]].max(axis=1)
    pd.testing.assert_series_equal(df["CPU_Temperature"], expected, check_names=False)


def test_calendar_features_correct(sample_server_df):
    df = add_calendar_features(sample_server_df)
    assert "hour" in df.columns
    assert df["hour"].iloc[0] == 0
    assert all(-1 <= df["hour_sin"]) and all(df["hour_sin"] <= 1)


def test_feature_set_s0_is_empty(sample_server_df):
    assert get_feature_set("S0", sample_server_df.columns.tolist()) == []


def test_feature_set_s2_excludes_power_and_fan():
    cols = ["CPU_Load", "Ram", "Power", "Fan_speed1", "hour", "dayofweek",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    feats = get_feature_set("S2", cols)
    assert "Power" not in feats
    assert "Fan_speed1" not in feats
    assert "CPU_Load" in feats


def test_feature_set_s4_deduplicates_and_preserves_order():
    cols = [
        "CPU_Load", "Ram", "Ram_Used", "CPU_cores", "CPU_cores_used",
        "Network_RX", "Network_TX", "No_Of_Running_vms",
        "hour", "dayofweek", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "Power", "Fan_speed1", "Fan_speed2", "Fan_speed3", "Fan_speed4",
        "Inlet_Temperature", "CPU_Temperature", "CPU1_Temperature", "CPU2_Temperature",
        "CPU_Temperature_lag_1", "CPU_Temperature_rmean_6",
    ]
    # Intentionally duplicate columns in available_columns input
    cols_duplicated = cols + ["CPU_Temperature", "Inlet_Temperature", "CPU_Temperature_lag_1", "CPU_Load"]
    feats = get_feature_set("S4", cols_duplicated)
    assert len(feats) == len(set(feats)), f"Duplicates found in S4: {feats}"
    # Verify order preservation
    expected_unique = []
    for c in cols_duplicated:
        if c in feats and c not in expected_unique:
            expected_unique.append(c)
    assert feats == expected_unique

