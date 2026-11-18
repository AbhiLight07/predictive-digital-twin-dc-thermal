"""Plain dataclasses for the predictive digital twin schema.

No heavy imports — only stdlib.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


class RiskLevel(enum.Enum):
    """Risk classification for a host temperature forecast."""

    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class DataQuality(enum.Enum):
    """Quality flag for a single sensor reading."""

    OK = "ok"
    STALE = "stale"
    MISSING = "missing"


@dataclass(frozen=True)
class HostStatic:
    """Time-invariant properties of a physical host."""

    cluster: str                # "D1" or "D2"
    host_id: str                # e.g. "qh2-rcc120"
    cores: int                  # from CPU_cores column
    ram_gb: float               # from Ram column
    hardware_model: str         # as reported in dataset paper (Ilager et al., UCC 2023)
    has_fan: bool               # True for most D1 hosts, False for D2
    has_inlet: bool             # True for most D1 hosts, False for D2


@dataclass
class SensorReading:
    """A single sensor value with availability and quality metadata."""

    value: Optional[float] = None
    available: bool = True
    quality: DataQuality = DataQuality.OK


@dataclass
class HostState:
    """Snapshot of all telemetry for one host at one time step."""

    cluster: str
    host_id: str
    timestamp: str              # ISO-8601

    cpu_load: SensorReading = field(default_factory=SensorReading)
    ram: SensorReading = field(default_factory=SensorReading)
    ram_used: SensorReading = field(default_factory=SensorReading)
    cpu_cores: SensorReading = field(default_factory=SensorReading)
    cpu_cores_used: SensorReading = field(default_factory=SensorReading)
    network_rx: SensorReading = field(default_factory=SensorReading)
    network_tx: SensorReading = field(default_factory=SensorReading)
    no_of_running_vms: SensorReading = field(default_factory=SensorReading)
    power: SensorReading = field(default_factory=SensorReading)
    fan_speed1: SensorReading = field(default_factory=SensorReading)
    fan_speed2: SensorReading = field(default_factory=SensorReading)
    fan_speed3: SensorReading = field(default_factory=SensorReading)
    fan_speed4: SensorReading = field(default_factory=SensorReading)
    inlet_temperature: SensorReading = field(default_factory=SensorReading)
    cpu1_temperature: SensorReading = field(default_factory=SensorReading)
    cpu2_temperature: SensorReading = field(default_factory=SensorReading)
    cpu_temperature: SensorReading = field(default_factory=SensorReading)  # max(CPU1, CPU2)


@dataclass
class HostForecast:
    """Temperature forecast for one host at one horizon."""

    cluster: str
    host_id: str
    timestamp: str              # ISO-8601 — base time of the forecast
    horizon_steps: int          # 1, 6, or 36
    point: float                # predicted effective CPU temp (°C)
    lower: float                # conformal interval lower bound
    upper: float                # conformal interval upper bound
    persistence: float          # persistence baseline value T(t)
    risk: RiskLevel = RiskLevel.UNKNOWN
    model_version: str = ""
    partition: str = ""         # TRAIN / VAL / TEST
