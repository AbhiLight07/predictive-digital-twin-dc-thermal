"""Tests to verify feature engineering does not introduce future information."""
import pytest


def test_rolling_features_causal():
    """Verify rolling features are computed using only past and present data."""
    pytest.skip("Not implemented yet — requires feature implementation")


def test_target_creation_correct_horizon():
    """Verify prediction targets are correctly shifted by the specified horizon."""
    pytest.skip("Not implemented yet — requires feature implementation")
