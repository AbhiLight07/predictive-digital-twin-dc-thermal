"""Tests for ttwin.twin schema dataclasses."""

import enum
import pytest

from ttwin.twin.schema import (
    DataQuality,
    HostForecast,
    HostStatic,
    HostState,
    RiskLevel,
    SensorReading,
)


class TestRiskLevel:
    def test_is_enum(self):
        assert issubclass(RiskLevel, enum.Enum)

    def test_values(self):
        assert set(r.value for r in RiskLevel) == {"normal", "warning", "critical", "unknown"}

    def test_from_string(self):
        assert RiskLevel("warning") is RiskLevel.WARNING


class TestDataQuality:
    def test_values(self):
        assert set(q.value for q in DataQuality) == {"ok", "stale", "missing"}


class TestHostStatic:
    def test_construction(self):
        h = HostStatic(
            cluster="D1",
            host_id="qh2-rcc120",
            cores=80,
            ram_gb=1536.0,
            hardware_model="as reported in dataset paper (Ilager et al., UCC 2023)",
            has_fan=True,
            has_inlet=True,
        )
        assert h.cluster == "D1"
        assert h.host_id == "qh2-rcc120"
        assert h.has_fan is True

    def test_frozen(self):
        h = HostStatic("D2", "qh2-rcc201", 32, 128.0, "unverified", False, False)
        with pytest.raises(AttributeError):
            h.cluster = "D1"


class TestSensorReading:
    def test_defaults(self):
        s = SensorReading()
        assert s.value is None
        assert s.available is True
        assert s.quality is DataQuality.OK

    def test_missing(self):
        s = SensorReading(value=None, available=False, quality=DataQuality.MISSING)
        assert s.available is False


class TestHostState:
    def test_construction(self):
        state = HostState(cluster="D1", host_id="qh2-rcc120", timestamp="2019-01-01T00:00:00")
        assert state.cpu_load.quality is DataQuality.OK
        assert state.fan_speed1.value is None

    def test_d2_no_fans(self):
        state = HostState(
            cluster="D2",
            host_id="qh2-rcc201",
            timestamp="2019-03-01T00:00:00",
            fan_speed1=SensorReading(available=False, quality=DataQuality.MISSING),
            inlet_temperature=SensorReading(available=False, quality=DataQuality.MISSING),
        )
        assert state.fan_speed1.available is False
        assert state.inlet_temperature.available is False


class TestHostForecast:
    def test_construction(self):
        f = HostForecast(
            cluster="D1",
            host_id="qh2-rcc120",
            timestamp="2019-05-01T12:00:00",
            horizon_steps=6,
            point=68.5,
            lower=65.0,
            upper=72.0,
            persistence=67.0,
            risk=RiskLevel.NORMAL,
            model_version="v0.1",
            partition="TEST",
        )
        assert f.horizon_steps == 6
        assert f.risk is RiskLevel.NORMAL
        assert f.partition == "TEST"

    def test_default_risk_unknown(self):
        f = HostForecast("D1", "qh2-rcc120", "t", 1, 70.0, 68.0, 72.0, 69.0)
        assert f.risk is RiskLevel.UNKNOWN
