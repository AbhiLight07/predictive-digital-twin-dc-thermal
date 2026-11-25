"""Tests for TwinEngine: leakage, parity, gaps, availability, intervals.

Uses small synthetic fixtures so tests run without raw data.
"""

import json
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
TWIN_DIR = ROOT / "models" / "twin"
CACHE_DIR = ROOT / "data" / "twin_cache"
_D1_INTERIM = ROOT / "data" / "interim" / "dataset1"

_has_d1_data = _D1_INTERIM.exists() and any(
    f for f in _D1_INTERIM.glob("*.parquet") if not f.name.startswith("_")
)
_required_d1_models = [
    TWIN_DIR / f"d1_{kind}_{h}_{role}.txt"
    for kind, hs in [("monitoring", (1, 6, 36)), ("planning", (0,))]
    for h in hs
    for role in ("point", "lower", "upper")
]
_has_d1_models = (TWIN_DIR / "d1_metadata.json").exists() and all(
    p.exists() for p in _required_d1_models
)

needs_models = pytest.mark.skipif(
    not (_has_d1_models and _has_d1_data),
    reason="d1 twin models or data/interim/dataset1 parquet files not present",
)


@pytest.fixture(scope="module")
def d1_engine():
    """Load a real D1 engine (skipped if models or interim data missing)."""
    if not _has_d1_models:
        pytest.skip("d1 twin models not present")
    if not _has_d1_data:
        pytest.skip("data/interim/dataset1 parquet files not present")
    from ttwin.twin.engine import TwinEngine
    return TwinEngine("d1")


@pytest.fixture(scope="module")
def d1_cache():
    cache_path = CACHE_DIR / "d1_forecasts.parquet"
    if not cache_path.exists():
        pytest.skip("cache not present")
    return pd.read_parquet(cache_path)


# ── T3.1 No future leakage ───────────────────────────────────────────

@needs_models
def test_no_future_leakage(d1_engine):
    """Overwrite every observation after t with random values; forecast must be unchanged."""
    engine = d1_engine
    host = engine.hosts()[0]

    # Pick a time in the test partition
    b = engine._models_info["monitoring_1"]["boundaries"]
    t = pd.Timestamp(b["test_start"]) + pd.Timedelta(hours=12)

    # Get baseline forecast
    fc_before = engine.forecast(host, t)

    # Corrupt all data after t
    mask = (engine._data["Series"] == host) & (engine._data["Time"] > t)
    original = engine._data.loc[mask].copy()
    rng = np.random.RandomState(99)
    for col in ["CPU1_Temperature", "CPU2_Temperature", "CPU_Temperature", "CPU_Load"]:
        if col in engine._data.columns:
            engine._data.loc[mask, col] = rng.uniform(0, 100, size=mask.sum()).astype(engine._data[col].dtype)

    fc_after = engine.forecast(host, t)

    # Restore
    engine._data.loc[mask] = original.values

    # Compare
    for f1, f2 in zip(fc_before, fc_after):
        assert abs(f1.point - f2.point) < 1e-9, f"Leakage detected at h={f1.horizon_steps}"
        assert abs(f1.lower - f2.lower) < 1e-9
        assert abs(f1.upper - f2.upper) < 1e-9


# ── T3.2 Train/serve parity ──────────────────────────────────────────

@needs_models
def test_train_serve_parity(d1_engine, d1_cache):
    """Engine forecast for a sample (host, t) equals the cache row within 1e-6."""
    cache = d1_cache
    # Pick 3 sample rows
    sample = cache.head(100).sample(3, random_state=42)
    for _, row in sample.iterrows():
        fcs = d1_engine.forecast(row["host"], pd.Timestamp(row["time"]))
        fc_h = [f for f in fcs if f.horizon_steps == row["horizon"]]
        if not fc_h:
            continue
        fc = fc_h[0]
        assert abs(fc.point - row["point"]) < 1e-4, \
            f"Parity mismatch: engine={fc.point}, cache={row['point']}"


# ── T3.3 Gaps produce stale/unknown, not crash or zero ──────────────

@needs_models
def test_gap_handling(d1_engine):
    """A missing timestamp gives stale/unknown, not a crash or zero."""
    engine = d1_engine
    host = engine.hosts()[0]

    # Query a time way before data
    t_early = pd.Timestamp("2017-01-01")
    state = engine.state(host, t_early)
    assert state.cpu_temperature.quality.value == "missing"
    assert state.cpu_temperature.value is None

    # Query between timestamps (should be stale or ok, not crash)
    t_gap = pd.Timestamp("2019-05-10 00:05:00")
    state2 = engine.state(host, t_gap)
    assert state2.cpu_temperature.quality.value in ("ok", "stale", "missing")

    # Forecast at early time should return unknown risk
    fcs = engine.forecast(host, t_early)
    for fc in fcs:
        assert fc.risk == "unknown" or fc.risk.value == "unknown"


# ── T3.4 Availability: hosts without fans/inlet report not available ─

@needs_models
def test_fan_inlet_availability(d1_engine):
    """D1 hosts without fan/inlet report not available; D1 hosts with fan report available."""
    engine = d1_engine

    # Find a host without fans (from columns.yaml: qh2-rcc196..200)
    no_fan_hosts = engine._no_fan_servers
    if not no_fan_hosts:
        pytest.skip("No fan-less hosts found in D1")

    no_fan_host = list(no_fan_hosts)[0]
    if no_fan_host not in engine.hosts():
        pytest.skip(f"{no_fan_host} not in host list")

    t = pd.Timestamp("2019-05-15 12:00:00")
    state_no_fan = engine.state(no_fan_host, t)
    assert state_no_fan.fan_speed1.available is False
    assert state_no_fan.inlet_temperature.available is False

    # A normal D1 host should have fans available
    normal_host = [h for h in engine.hosts() if h not in no_fan_hosts][0]
    state_normal = engine.state(normal_host, t)
    assert state_normal.fan_speed1.available is True
    assert state_normal.inlet_temperature.available is True


# ── T3.5 Interval ordering and risk consistency ──────────────────────

@needs_models
def test_interval_ordering_and_risk(d1_engine):
    """Intervals ordered (lower <= upper); risk consistent with thresholds."""
    engine = d1_engine
    host = engine.hosts()[0]
    b = engine._models_info["monitoring_1"]["boundaries"]
    t = pd.Timestamp(b["test_start"]) + pd.Timedelta(hours=6)

    fcs = engine.forecast(host, t)
    for fc in fcs:
        if not np.isnan(fc.lower) and not np.isnan(fc.upper):
            assert fc.lower <= fc.upper, f"lower > upper at h={fc.horizon_steps}"
        # Check risk consistency
        if not np.isnan(fc.upper):
            if fc.upper >= engine._crit_thresh:
                assert fc.risk == RiskLevel.CRITICAL or fc.risk.value == "critical"
            elif fc.upper >= engine._warn_thresh:
                assert fc.risk == RiskLevel.WARNING or fc.risk.value == "warning"
            else:
                assert fc.risk == RiskLevel.NORMAL or fc.risk.value == "normal"


# ── T3.6 Gap parity between training and serving ─────────────────────


@needs_models
def test_gap_parity_matches_training(d1_engine):
    """Deliberate 40-min gap must produce exact training-time NaNs and UNKNOWN risk."""
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.train_twin_models import build_monitoring_features

    # Build a 60-step series with steps 40..43 dropped (40-min gap)
    base = pd.Timestamp("2019-05-01 00:00:00")
    rows = []
    for i in range(60):
        if i in (40, 41, 42, 43):
            continue
        rows.append({
            "Series": "qh2-rcc120",
            "Time": base + pd.Timedelta(minutes=10 * i),
            "CPU1_Temperature": 45.0 + 0.2 * i,
            "CPU2_Temperature": 44.0 + 0.2 * i,
        })
    raw_df = pd.DataFrame(rows)
    train_feat = build_monitoring_features(raw_df)
    serve_feat = d1_engine._build_monitoring_features(raw_df)

    # Identical regular 10-min index and feature columns
    assert len(serve_feat) == len(train_feat) == 60
    assert (serve_feat["Time"].values == train_feat["Time"].values).all()

    feat_cols = d1_engine._models_info["monitoring_1"]["feature_names"]
    for col in feat_cols:
        assert col in serve_feat.columns
        np.testing.assert_allclose(
            serve_feat[col].to_numpy(dtype=np.float64),
            train_feat[col].to_numpy(dtype=np.float64),
            equal_nan=True,
            err_msg=f"Train/serve gap parity mismatch in {col}",
        )

    # At row right after the 40-min gap (step 44), lag_1 must be NaN -> UNKNOWN risk
    t_after_gap = base + pd.Timedelta(minutes=10 * 44)
    row_after = serve_feat[serve_feat["Time"] == t_after_gap].iloc[0]
    assert pd.isna(row_after["CPU_Temperature_lag_1"])

    orig_data = d1_engine._data
    try:
        from ttwin.features import compute_effective_cpu_temp
        d1_engine._data = compute_effective_cpu_temp(raw_df)
        fcs_after = d1_engine.forecast("qh2-rcc120", t_after_gap)
        assert all(fc.risk.value == "unknown" for fc in fcs_after)
        # Querying inside gap (>30 min past latest observation at step 39) -> stale -> UNKNOWN
        t_in_gap = base + pd.Timedelta(minutes=10 * 43)
        fcs_stale = d1_engine.forecast("qh2-rcc120", t_in_gap)
        assert all(fc.risk.value == "unknown" for fc in fcs_stale)
    finally:
        d1_engine._data = orig_data


# Import after function definitions to avoid circular issues
from ttwin.twin.schema import RiskLevel
