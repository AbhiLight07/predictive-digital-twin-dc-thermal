"""Tests to verify no data leakage in train/test splits.

Checks:
- No future timestamps appear in training data for any test row
- No server overlap in leave-servers-out splits
- Lag/rolling features computed only from past data
- Scalers/encoders fit on training data only
"""
import pytest


def test_no_future_timestamps_in_train():
    """Verify no future timestamps leak into training data."""
    pytest.skip("Not implemented yet — requires processed data")


def test_no_server_overlap_in_leave_out():
    """Verify held-out servers do not appear in training data."""
    pytest.skip("Not implemented yet — requires split implementation")


def test_lag_features_no_future():
    """Verify lag and rolling features use only past data."""
    pytest.skip("Not implemented yet — requires feature implementation")


def test_scalers_fit_on_train_only():
    """Verify scalers and encoders are fit on training data only."""
    pytest.skip("Not implemented yet — requires pipeline implementation")
