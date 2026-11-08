"""Tests to verify feature engineering does not introduce future information."""
import pytest
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, "d:/mainel5/src")

from ttwin.features import (
    add_lag_features, add_rolling_features, create_target,
    add_calendar_features, compute_effective_cpu_temp, get_feature_set,
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
    })


def test_lag_features_no_future(sample_server_df):
    df = add_lag_features(sample_server_df, ["CPU_Temperature"], [1, 3])
    for i in range(1, len(df)):
        assert df.iloc[i]["CPU_Temperature_lag_1"] == df.iloc[i - 1]["CPU_Temperature"]
    assert pd.isna(df.iloc[0]["CPU_Temperature_lag_1"])


def test_rolling_features_causal(sample_server_df):
    df = add_rolling_features(sample_server_df, ["CPU_Temperature"], [3])
    for i in range(4, len(df)):
        vals = sample_server_df["CPU_Temperature"].iloc[i - 3:i].values
        expected = np.mean(vals)
        np.testing.assert_almost_equal(
            df.iloc[i]["CPU_Temperature_rmean_3"], expected, decimal=5
        )


def test_target_creation_correct_horizon(sample_server_df):
    df = create_target(sample_server_df, "CPU_Temperature", horizon_steps=3)
    target_col = "CPU_Temperature_target_3"
    assert target_col in df.columns
    for i in range(len(df) - 3):
        assert df.iloc[i][target_col] == sample_server_df.iloc[i + 3]["CPU_Temperature"]
    for i in range(len(df) - 3, len(df)):
        assert pd.isna(df.iloc[i][target_col])


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
