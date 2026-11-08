"""Tests for evaluation metrics."""
import pytest
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, "d:/mainel5/src")

from ttwin.metrics import mae, rmse, skill_score_vs_persistence, per_server_metrics, summarize_per_server


def test_mae_computation():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    assert mae(y_true, y_pred) == pytest.approx(0.5)


def test_mae_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == pytest.approx(0.0)


def test_rmse_computation():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(1 / 3))


def test_rmse_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_skill_score_perfect():
    assert skill_score_vs_persistence(0.0, 1.0) == pytest.approx(1.0)


def test_skill_score_equal():
    assert skill_score_vs_persistence(1.0, 1.0) == pytest.approx(0.0)


def test_skill_score_worse():
    assert skill_score_vs_persistence(2.0, 1.0) == pytest.approx(-1.0)


def test_per_server_metrics():
    df = pd.DataFrame({
        "Series": ["A", "A", "B", "B"],
        "y_true": [1.0, 2.0, 3.0, 4.0],
        "y_pred": [1.1, 2.1, 3.2, 4.2],
    })
    result = per_server_metrics(df, "y_true", "y_pred")
    assert len(result) == 2
    assert "mae" in result.columns
    assert "rmse" in result.columns


def test_summarize_per_server():
    per_server = pd.DataFrame({"mae": [1.0, 2.0, 3.0, 4.0, 5.0]})
    summary = summarize_per_server(per_server, "mae")
    assert summary["mean"] == pytest.approx(3.0)
    assert summary["median"] == pytest.approx(3.0)
    assert summary["n_servers"] == 5
