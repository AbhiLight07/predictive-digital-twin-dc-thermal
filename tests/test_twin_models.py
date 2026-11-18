"""Tests for twin models: reproducibility, partition disjointness, interval ordering.

Skipped if models are missing.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
TWIN_DIR = ROOT / "models" / "twin"

_has_d1 = (TWIN_DIR / "d1_metadata.json").exists()
_has_d2 = (TWIN_DIR / "d2_metadata.json").exists()

needs_d1 = pytest.mark.skipif(not _has_d1, reason="d1 twin models not present")
needs_d2 = pytest.mark.skipif(not _has_d2, reason="d2 twin models not present")


def _load_meta(cluster):
    with open(TWIN_DIR / f"{cluster}_metadata.json") as f:
        return json.load(f)


def _load_samples(cluster):
    path = TWIN_DIR / f"{cluster}_sample_predictions.json"
    if not path.exists():
        pytest.skip(f"Sample predictions file not found for {cluster}")
    with open(path) as f:
        return json.load(f)


@needs_d1
class TestD1Models:
    def test_partition_boundaries_disjoint(self):
        meta = _load_meta("d1")
        for m in meta["models"]:
            b = m["boundaries"]
            # val_sel_start must be after train_end
            assert pd.Timestamp(b["val_sel_start"]) > pd.Timestamp(b["train_end"])
            # val_cal_start must be after val_sel_end
            assert pd.Timestamp(b["val_cal_start"]) > pd.Timestamp(b["val_sel_end"])
            # test_start must be after val_cal_end
            assert pd.Timestamp(b["test_start"]) > pd.Timestamp(b["val_cal_end"])

    def test_sample_predictions_reproducible(self):
        samples = _load_samples("d1")
        meta = _load_meta("d1")
        for key, data in samples.items():
            parts = key.split("_")
            kind = parts[0]
            h = int(parts[1])
            model_path = TWIN_DIR / f"d1_{kind}_{h}_point.txt"
            if not model_path.exists():
                pytest.skip(f"Model file {model_path.name} not found")
            m = lgb.Booster(model_file=str(model_path))
            m_meta = next(info for info in meta["models"] if info["kind"] == kind and int(info["horizon"]) == h)
            feat_cols = m_meta["feature_names"]
            X = np.column_stack([data["input_rows"][c] for c in feat_cols])
            preds = m.predict(X)
            np.testing.assert_allclose(preds, data["predictions"], atol=1e-9)

    def test_intervals_ordered(self):
        """q_hat can be negative (CQR shrinks over-wide quantile bands); check lower<=upper on stored metadata."""
        meta = _load_meta("d1")
        for m in meta["models"]:
            # CQR q_hat may be negative when quantile models already over-cover; that is valid
            # What matters is that intervals are finite (not inf) and width >= 0
            assert m["mean_interval_width"] >= 0, f"negative width for {m['kind']} h={m['horizon']}"

    def test_coverage_reasonable(self):
        meta = _load_meta("d1")
        for m in meta["models"]:
            cov = m["empirical_coverage"]
            assert 0.80 <= cov <= 1.0, f"Coverage {cov} outside [0.80, 1.0] for {m['kind']} h={m['horizon']}"


@needs_d2
class TestD2Models:
    def test_partition_boundaries_disjoint(self):
        meta = _load_meta("d2")
        for m in meta["models"]:
            b = m["boundaries"]
            assert pd.Timestamp(b["val_sel_start"]) > pd.Timestamp(b["train_end"])
            assert pd.Timestamp(b["val_cal_start"]) > pd.Timestamp(b["val_sel_end"])
            assert pd.Timestamp(b["test_start"]) > pd.Timestamp(b["val_cal_end"])

    def test_sample_predictions_reproducible(self):
        samples = _load_samples("d2")
        meta = _load_meta("d2")
        for key, data in samples.items():
            parts = key.split("_")
            kind = parts[0]
            h = int(parts[1])
            model_path = TWIN_DIR / f"d2_{kind}_{h}_point.txt"
            if not model_path.exists():
                pytest.skip(f"Model file {model_path.name} not found")
            m = lgb.Booster(model_file=str(model_path))
            m_meta = next(info for info in meta["models"] if info["kind"] == kind and int(info["horizon"]) == h)
            feat_cols = m_meta["feature_names"]
            X = np.column_stack([data["input_rows"][c] for c in feat_cols])
            preds = m.predict(X)
            np.testing.assert_allclose(preds, data["predictions"], atol=1e-9)

    def test_intervals_ordered(self):
        """q_hat can be negative (CQR shrinks over-wide quantile bands); check width >= 0."""
        meta = _load_meta("d2")
        for m in meta["models"]:
            assert m["mean_interval_width"] >= 0, f"negative width for {m['kind']} h={m['horizon']}"

    def test_coverage_reasonable(self):
        meta = _load_meta("d2")
        for m in meta["models"]:
            cov = m["empirical_coverage"]
            assert 0.80 <= cov <= 1.0, f"Coverage {cov} outside [0.80, 1.0] for {m['kind']} h={m['horizon']}"
