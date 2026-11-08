"""Tests to verify no data leakage and correct stratification in train/test splits."""
import pytest
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, "d:/mainel5/src")

from ttwin.splits import (
    split_chronological, split_leave_servers_out, split_strict,
    verify_no_server_overlap, verify_no_future_leak, get_split_b_seeds,
    get_group_kfold_splits,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing with 10 servers and varying load."""
    np.random.seed(42)
    servers = [f"server_{i}" for i in range(10)]
    times = pd.date_range("2019-01-01", periods=100, freq="10min")
    rows = []
    for i, s in enumerate(servers):
        base_load = i * 10.0  # distinct load per server
        for t in times:
            rows.append({
                "Series": s, "Time": t,
                "CPU_Load": base_load + np.random.uniform(-5, 5),
                "CPU_Temperature": np.random.uniform(30, 80),
            })
    return pd.DataFrame(rows)


def test_no_future_timestamps_in_train(sample_df):
    train, val, test = split_chronological(sample_df, embargo_steps=3)
    assert len(test) > 0, "Test set should not be empty"
    assert verify_no_future_leak(train, test)
    assert train["Time"].max() < test["Time"].min()
    assert val["Time"].max() < test["Time"].min()


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


def test_leave_out_reproducibility(sample_df):
    _, _, _, test1 = split_leave_servers_out(sample_df, seed=42)
    _, _, _, test2 = split_leave_servers_out(sample_df, seed=42)
    assert test1 == test2


def test_leave_out_different_seeds_give_different_sets(sample_df):
    """Different random seeds must yield different held-out servers."""
    _, _, _, test1 = split_leave_servers_out(sample_df, seed=42)
    _, _, _, test2 = split_leave_servers_out(sample_df, seed=123)
    assert test1 != test2, f"Seed 42 and Seed 123 produced identical sets: {test1}"


def test_group_kfold_covers_all_servers_disjointly(sample_df):
    """5-fold GroupKFold by server must test every server exactly once."""
    splits = get_group_kfold_splits(sample_df, n_splits=5, seed=42)
    assert len(splits) == 5

    all_test_servers = []
    servers = set(sample_df["Series"].unique())
    for train_s, test_s in splits:
        assert len(set(train_s) & set(test_s)) == 0
        all_test_servers.extend(test_s)

    # Union must equal all servers, with no duplicates in test sets
    assert len(all_test_servers) == len(servers)
    assert set(all_test_servers) == servers


def test_split_b_has_5_seeds():
    assert len(get_split_b_seeds()) == 5
