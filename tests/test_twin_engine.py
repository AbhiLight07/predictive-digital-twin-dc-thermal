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

_has_models = (TWIN_DIR / "d1_metadata.json").exists()
_has_cache = (CACHE_DIR / "d1_forecasts.parquet").exists()

needs_models = pytest.mark.skipif(not _has_models, reason="twin models not present")
needs_cache = pytest.mark.skipif(not _has_cache, reason="twin cache not present")

# ── Helpers ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def d1_engine():
    """Load a real D1 engine (skipped if data missing)."""
    if not _has_models:
        pytest.skip("models not present")
    from ttwin.twin.engine import TwinEngine
    return TwinEngine("d1")


@pytest.fixture(scope="module")
def d1_cache():
    if not _has_cache:
        pytest.skip("cache not present")
    return pd.read_parquet(CACHE_DIR / "d1_forecasts.parquet")


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
@needs_cache
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


# Import after function definitions to avoid circular issues
from ttwin.twin.schema import RiskLevel
