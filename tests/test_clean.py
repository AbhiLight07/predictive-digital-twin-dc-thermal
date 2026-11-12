"""Tests for data cleaning pipeline and time-gap handling."""
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "d:/mainel5/src")

from ttwin.clean import clean_dataset, strip_host_names, fill_missing_timestamps
from ttwin.features import build_features


def test_clean_dataset_calls_strip_host_names():
    """Verify clean_dataset strips whitespace from host identifiers."""
    df = pd.DataFrame({
        "Series": [" qh2-rcc120 ", "qh2-rcc121\t", "  qh2-rcc122  "],
        "Time": ["2019-01-01 00:00:00", "2019-01-01 00:10:00", "2019-01-01 00:20:00"],
        "CPU_Temperature": [45.0, 50.0, 55.0],
    })
    cleaned = clean_dataset(df, host_col="Series", time_col="Time")
    expected_hosts = ["qh2-rcc120", "qh2-rcc121", "qh2-rcc122"]
    assert list(cleaned["Series"]) == expected_hosts


def test_consecutive_timestamps_differ_by_600s_after_filling():
    """Verify consecutive timestamps differ by exactly 600 seconds after grid filling."""
    df = pd.DataFrame({
        "Series": ["srv1", "srv1", "srv1"],
        "Time": ["2019-01-01 00:00:00", "2019-01-01 00:10:00", "2019-01-01 00:40:00"],
        "CPU_Temperature": [40.0, 42.0, 50.0],
    })
    filled = fill_missing_timestamps(df, host_col="Series", time_col="Time", freq="10min")
    assert len(filled) == 5
    diffs = filled["Time"].diff().dropna()
    assert (diffs == pd.Timedelta(seconds=600)).all()


def test_horizon_target_timestamp_exactly_h_times_10min_ahead():
    """Verify target timestamp is exactly h * 10 minutes ahead across horizons."""
    times = pd.date_range("2019-01-01 00:00:00", periods=50, freq="10min")
    keep_indices = list(range(0, 15)) + list(range(20, 50))
    df = pd.DataFrame({
        "Series": "srv1",
        "Time": times[keep_indices],
        "CPU_Temperature": np.arange(len(keep_indices), dtype=float) + 30.0,
    })
    df_feat = build_features(df, horizons=[1, 6, 36])

    diffs = df_feat["Time"].diff().dropna()
    assert (diffs == pd.Timedelta(seconds=600)).all()

    for h in [1, 6, 36]:
        target_col = f"CPU_Temperature_target_{h}"
        assert target_col in df_feat.columns
        for i in range(len(df_feat) - h):
            actual_target = df_feat.iloc[i][target_col]
            future_temp = df_feat.iloc[i + h]["CPU_Temperature"]
            future_time = df_feat.iloc[i + h]["Time"]
            curr_time = df_feat.iloc[i]["Time"]
            assert future_time - curr_time == pd.Timedelta(minutes=h * 10)
            if pd.isna(actual_target):
                assert pd.isna(future_temp)
            else:
                assert actual_target == future_temp
