"""Tests to verify no data leakage in train/test splits."""
import pytest
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, "d:/mainel5/src")

from ttwin.splits import (
    split_chronological, split_leave_servers_out, split_strict,
    verify_no_server_overlap, verify_no_future_leak, get_split_b_seeds,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    servers = ["server_A", "server_B", "server_C", "server_D", "server_E"]
    times = pd.date_range("2019-01-01", periods=100, freq="10min")
    rows = []
    for s in servers:
        for t in times:
            rows.append({
                "Series": s, "Time": t,
                "CPU_Load": np.random.uniform(0, 100),
                "CPU_Temperature": np.random.uniform(30, 80),
            })
    return pd.DataFrame(rows)


def test_no_future_timestamps_in_train(sample_df):
    train, val, test = split_chronological(sample_df, embargo_steps=3)
    assert len(test) > 0, "Test set should not be empty"
    assert verify_no_future_leak(train, test)
    assert train["Time"].max() < test["Time"].min()


def test_no_server_overlap_in_leave_out(sample_df):
    train, test, train_servers, test_servers = split_leave_servers_out(sample_df)
    assert verify_no_server_overlap(train, test)
    assert len(set(train_servers) & set(test_servers)) == 0


def test_strict_split_no_overlap(sample_df):
    train, val, test = split_strict(sample_df, embargo_steps=3)
    train_servers = set(train["Series"].unique())
    test_servers = set(test["Series"].unique())
    assert len(train_servers & test_servers) == 0
    assert len(test) > 0, "Test set should not be empty"
    assert train["Time"].max() < test["Time"].min()


def test_chronological_no_data_loss(sample_df):
    train, val, test = split_chronological(sample_df, embargo_steps=0)
    for s in sample_df["Series"].unique():
        in_any = (
            s in train["Series"].values
            or s in val["Series"].values
            or s in test["Series"].values
        )
        assert in_any


def test_leave_out_reproducibility(sample_df):
    _, _, _, test1 = split_leave_servers_out(sample_df, seed=42)
    _, _, _, test2 = split_leave_servers_out(sample_df, seed=42)
    assert test1 == test2


def test_leave_out_different_seeds(sample_df):
    _, _, _, test1 = split_leave_servers_out(sample_df, seed=42)
    _, _, _, test2 = split_leave_servers_out(sample_df, seed=123)
    assert len(test1) > 0
    assert len(test2) > 0


def test_split_b_has_5_seeds():
    assert len(get_split_b_seeds()) == 5
