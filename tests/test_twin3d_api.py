"""Contract tests for the FastAPI 3D digital twin server (tests/test_twin3d_api.py).

Uses a self-contained synthetic engine & cache fixture so all tests run fast
without requiring raw telemetry files, while verifying:
  - /api/health, /api/meta, /api/snapshot, /api/compare, /api/series, /api/alerts, POST /api/whatif
  - Compare-mode strict time causality (twin value for t issued at t - h*10m)
  - Layout determinism and uniqueness of (rack, slot)
  - Unavailable sensors (D2 and D1 no-fan host) report null + available=False, never 0
  - Gaps > 30 min mark status as stale without fabricating/interpolating across gaps
  - HTTP 503 when models or clusters are missing
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from ttwin.twin.schema import DataQuality, HostForecast, HostState, HostStatic, RiskLevel, SensorReading  # noqa: E402
from ttwin.twin3d.server import Twin3DService, create_app  # noqa: E402


# ── Synthetic engine + cache fixture ─────────────────────────────────


class _SyntheticEngine:
    """Deterministic in-memory stand-in matching TwinEngine's public interface."""

    def __init__(self, cluster: str):
        self.cluster = cluster
        self._has_fan = cluster == "d1"
        self._has_inlet = cluster == "d1"
        self._no_fan_servers = {"qh2-rcc196"} if cluster == "d1" else set()
        self._warn_thresh = 73.93 if cluster == "d1" else 75.72
        self._crit_thresh = 79.93 if cluster == "d1" else 77.63
        self._horizons = [1, 6, 36]
        self._config = {"replay": {"stride_minutes": 30, "sampling_interval_minutes": 10}}

        if cluster == "d1":
            # Include >20 hosts to span multiple racks, plus one host without fans/inlet
            self._host_list = [f"qh2-rcc{120 + i}" for i in range(25)] + ["qh2-rcc196"]
        else:
            self._host_list = [f"qh2-rcc{201 + i}" for i in range(22)]

        # 30 timesteps at 10-min intervals starting 2019-05-01 00:00:00
        # Leave a 50-min gap before 04:00:00 on first host to test gap handling
        base = pd.Timestamp("2019-05-01 00:00:00")
        times = [base + pd.Timedelta(minutes=10 * k) for k in range(30)]
        rows = []
        for hi, host in enumerate(self._host_list):
            has_fan = self._has_fan and host not in self._no_fan_servers
            for ti, t in enumerate(times):
                # Inject a >30 min missing gap for host 0 at steps 20..23
                if hi == 0 and ti in (20, 21, 22, 23):
                    continue
                t_base = 45.0 + 0.3 * hi + 0.5 * ti
                rows.append({
                    "Series": host,
                    "Time": t,
                    "CPU_Load": 20.0 + hi + 0.4 * ti,
                    "Ram": 64.0 if cluster == "d1" else 192.0,
                    "Ram_Used": 32.0 + 0.1 * hi,
                    "CPU_cores": 64.0 if cluster == "d1" else 80.0,
                    "CPU_cores_used": 16.0 + (hi % 8),
                    "Network_RX": 100000.0 + 100.0 * ti,
                    "Network_TX": 80000.0 + 50.0 * ti,
                    "No_Of_Running_vms": float(4 + (hi % 5)),
                    "Power": 110.0 + 2.0 * hi + 0.5 * ti,
                    "Fan_speed1": 7000.0 + 10.0 * ti if has_fan else np.nan,
                    "Fan_speed2": 7050.0 + 10.0 * ti if has_fan else np.nan,
                    "Fan_speed3": 7020.0 + 10.0 * ti if has_fan else np.nan,
                    "Fan_speed4": 6980.0 + 10.0 * ti if has_fan else np.nan,
                    "Inlet_Temperature": 18.0 + 0.1 * hi if has_fan else np.nan,
                    "CPU1_Temperature": t_base,
                    "CPU2_Temperature": t_base - 1.0,
                    "CPU_Temperature": t_base,
                })
        self._data = pd.DataFrame(rows)
        self._models = {f"{cluster}_monitoring_1_point": object(), f"{cluster}_planning_0_point": object()}
        boundaries = {
            "train_end": "2019-05-01 01:00:00",
            "val_sel_start": "2019-05-01 01:10:00",
            "val_sel_end": "2019-05-01 01:30:00",
            "val_cal_start": "2019-05-01 01:40:00",
            "val_cal_end": "2019-05-01 01:50:00",
            "test_start": "2019-05-01 02:00:00",
            "test_end": "2019-05-01 04:50:00",
        }
        self._models_info = {
            "monitoring_1": {
                "kind": "monitoring",
                "horizon": 1,
                "boundaries": boundaries,
                "test_mae": 0.41,
                "persistence_mae": 0.42,
                "skill_score": 0.0238,
                "empirical_coverage": 0.90,
                "mean_interval_width": 1.70,
                "n_test_rows": 500,
                "q_hat": 0.1,
            },
            "monitoring_6": {
                "kind": "monitoring",
                "horizon": 6,
                "boundaries": boundaries,
                "test_mae": 0.82,
                "persistence_mae": 0.85,
                "skill_score": 0.035,
                "empirical_coverage": 0.895,
                "mean_interval_width": 3.40,
                "n_test_rows": 500,
                "q_hat": 0.2,
            },
            "monitoring_36": {
                "kind": "monitoring",
                "horizon": 36,
                "boundaries": boundaries,
                "test_mae": 1.50,
                "persistence_mae": 1.55,
                "skill_score": 0.032,
                "empirical_coverage": 0.88,
                "mean_interval_width": 6.80,
                "n_test_rows": 500,
                "q_hat": 0.3,
            },
            "planning_0": {
                "kind": "planning",
                "horizon": 0,
                "boundaries": boundaries,
                "test_mae": 4.20,
                "persistence_mae": None,
                "skill_score": None,
                "empirical_coverage": 0.84,
                "mean_interval_width": 16.0,
                "n_test_rows": 500,
                "q_hat": 0.5,
            },
        }
        self._meta = {
            "cluster": cluster,
            "git_commit": "abc1234",
            "models": list(self._models_info.values()),
        }

    def hosts(self):
        return list(self._host_list)

    def static(self, host: str) -> HostStatic:
        has_fan = self._has_fan and host not in self._no_fan_servers
        return HostStatic(
            cluster=self.cluster.upper(),
            host_id=host,
            cores=64 if self.cluster == "d1" else 80,
            ram_gb=64.0 if self.cluster == "d1" else 192.0,
            hardware_model="as reported in dataset paper (Ilager et al., UCC 2023)",
            has_fan=has_fan,
            has_inlet=has_fan,
        )

    def partition(self, t: pd.Timestamp) -> str:
        b = self._models_info["monitoring_1"]["boundaries"]
        ts = t.strftime("%Y-%m-%d %H:%M:%S")
        if ts <= b["train_end"]:
            return "TRAIN"
        if ts <= b["val_sel_end"]:
            return "VAL_SEL"
        if ts <= b["val_cal_end"]:
            return "VAL_CAL"
        return "TEST"

    def history(self, host: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        m = (self._data["Series"] == host) & (self._data["Time"] >= start) & (self._data["Time"] <= end)
        return self._data[m].copy()

    def state(self, host: str, t: pd.Timestamp) -> HostState:
        sub = self._data[(self._data["Series"] == host) & (self._data["Time"] <= t)]
        if len(sub) == 0:
            miss = SensorReading(value=None, available=False, quality=DataQuality.MISSING)
            return HostState(cluster=self.cluster.upper(), host_id=host, timestamp=str(t), cpu_temperature=miss)
        row = sub.iloc[-1]
        gap = (t - row["Time"]).total_seconds() / 60.0
        q = DataQuality.STALE if gap > 30.0 else DataQuality.OK
        has_fan = self._has_fan and host not in self._no_fan_servers
        return HostState(
            cluster=self.cluster.upper(),
            host_id=host,
            timestamp=str(row["Time"]),
            cpu_load=SensorReading(float(row["CPU_Load"]), True, q),
            ram=SensorReading(float(row["Ram"]), True, q),
            ram_used=SensorReading(float(row["Ram_Used"]), True, q),
            cpu_cores=SensorReading(float(row["CPU_cores"]), True, q),
            cpu_cores_used=SensorReading(float(row["CPU_cores_used"]), True, q),
            network_rx=SensorReading(float(row["Network_RX"]), True, q),
            network_tx=SensorReading(float(row["Network_TX"]), True, q),
            no_of_running_vms=SensorReading(float(row["No_Of_Running_vms"]), True, q),
            power=SensorReading(float(row["Power"]), True, q),
            fan_speed1=SensorReading(float(row["Fan_speed1"]) if has_fan else None, has_fan, q if has_fan else DataQuality.MISSING),
            fan_speed2=SensorReading(float(row["Fan_speed2"]) if has_fan else None, has_fan, q if has_fan else DataQuality.MISSING),
            fan_speed3=SensorReading(float(row["Fan_speed3"]) if has_fan else None, has_fan, q if has_fan else DataQuality.MISSING),
            fan_speed4=SensorReading(float(row["Fan_speed4"]) if has_fan else None, has_fan, q if has_fan else DataQuality.MISSING),
            inlet_temperature=SensorReading(float(row["Inlet_Temperature"]) if has_fan else None, has_fan, q if has_fan else DataQuality.MISSING),
            cpu1_temperature=SensorReading(float(row["CPU1_Temperature"]), True, q),
            cpu2_temperature=SensorReading(float(row["CPU2_Temperature"]), True, q),
            cpu_temperature=SensorReading(float(row["CPU_Temperature"]), True, q),
        )

    def forecast(self, host: str, t: pd.Timestamp):
        """Time-indexed synthetic forecast so issue time can be distinguished unambiguously."""
        minutes_since_midnight = int((t - pd.Timestamp("2019-05-01 00:00:00")).total_seconds() // 60)
        ti = minutes_since_midnight // 10
        hi = self._host_list.index(host)
        base_temp = 45.0 + 0.3 * hi + 0.5 * ti
        fcs = []
        for h in self._horizons:
            # Distinct forecast encoding issue time + horizon
            point = round(base_temp + 0.4 * h + 0.11, 4)
            lower = round(point - 2.0, 4)
            upper = round(point + 2.0, 4)
            # Trigger warning/critical alerts for the highest-index host
            if hi == len(self._host_list) - 2 and h in (6, 36):
                upper = self._warn_thresh + 0.5
                point = self._warn_thresh - 0.5
                lower = point - 2.0
                risk = RiskLevel.WARNING
            elif hi == len(self._host_list) - 1 and h in (6, 36):
                upper = self._crit_thresh + 0.8
                point = self._crit_thresh - 0.2
                lower = point - 2.0
                risk = RiskLevel.CRITICAL
            else:
                risk = RiskLevel.NORMAL
            fcs.append(
                HostForecast(
                    cluster=self.cluster.upper(),
                    host_id=host,
                    timestamp=str(t),
                    horizon_steps=h,
                    point=point,
                    lower=lower,
                    upper=upper,
                    persistence=round(base_temp, 4),
                    risk=risk,
                    model_version="abc1234",
                    partition=self.partition(t),
                )
            )
        return fcs

    def whatif(self, workload: dict, capacity: dict | None = None) -> dict:
        load = float(workload.get("CPU_Load", 0.0))
        vms = float(workload.get("No_Of_Running_vms", 0.0))
        pt = 38.0 + 0.35 * load + 0.4 * vms
        lo = pt - 3.0
        hi = pt + 3.0
        risk = RiskLevel.CRITICAL if hi >= self._crit_thresh else (RiskLevel.WARNING if hi >= self._warn_thresh else RiskLevel.NORMAL)
        return {"available": True, "point": round(pt, 4), "lower": round(lo, 4), "upper": round(hi, 4), "risk": risk.value}


def _build_synthetic_cache(eng: _SyntheticEngine) -> pd.DataFrame:
    rows = []
    # Build cache every 10 min starting 02:00:00 (test partition)
    base = pd.Timestamp("2019-05-01 02:00:00")
    for k in range(18):
        issue_t = base + pd.Timedelta(minutes=10 * k)
        for host in eng.hosts():
            for fc in eng.forecast(host, issue_t):
                target_t = issue_t + pd.Timedelta(minutes=fc.horizon_steps * 10)
                st_future = eng.state(host, target_t)
                rows.append({
                    "host": host,
                    "time": issue_t,
                    "horizon": fc.horizon_steps,
                    "point": fc.point,
                    "lower": fc.lower,
                    "upper": fc.upper,
                    "persistence": fc.persistence,
                    "risk": fc.risk.value,
                    "partition": "TEST",
                    "actual_future": st_future.cpu_temperature.value,
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_client():
    d1 = _SyntheticEngine("d1")
    d2 = _SyntheticEngine("d2")
    svc = Twin3DService(
        engines={"d1": d1, "d2": d2},
        caches={"d1": _build_synthetic_cache(d1), "d2": _build_synthetic_cache(d2)},
    )
    app = create_app(service=svc, web_dir=_REPO / "nonexistent_web_dir")
    return TestClient(app)


# ── Endpoint contract tests ──────────────────────────────────────────


def test_health_endpoint(synthetic_client):
    r = synthetic_client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert set(data["clusters"].keys()) == {"d1", "d2"}
    assert data["clusters"]["d1"]["loaded"] is True
    assert data["clusters"]["d2"]["host_count"] == 22


def test_meta_schema_and_layout(synthetic_client):
    r = synthetic_client.get("/api/meta")
    assert r.status_code == 200
    data = r.json()
    assert "schematic" in data["banner"].lower()
    assert data["horizons_steps"] == [0, 1, 6, 36]
    assert len(data["what_twin_does_not_know"]) >= 4

    for cid in ("d1", "d2"):
        cm = data["clusters"][cid]
        hosts = cm["hosts"]
        # Check deterministic ordering by numeric suffix and unique (rack, slot)
        nums = [int(h["id"].split("rcc")[-1]) for h in hosts]
        assert nums == sorted(nums)
        rack_slots = [(h["rack"], h["slot"]) for h in hosts]
        assert len(rack_slots) == len(set(rack_slots))
        for k in ("t_lo", "t_hi", "p_lo", "p_hi", "w_ref"):
            assert cm["legend"][k] is not None
        assert len(cm["models"]) >= 3


def test_snapshot_schema_and_unavailable_sensors(synthetic_client):
    r = synthetic_client.get("/api/snapshot?cluster=all&t=2019-05-01T02:30:00Z&horizon=6")
    assert r.status_code == 200
    data = r.json()
    hosts = data["hosts"]
    assert len(hosts) == 26 + 22

    # Check D1 normal host has fans & inlet, rounded floats, and forecast
    d1_first = next(h for h in hosts if h["id"] == "qh2-rcc120")
    assert d1_first["fans_available"] is True
    assert d1_first["fans"] is not None and len(d1_first["fans"]) == 4
    assert d1_first["inlet_available"] is True
    assert d1_first["inlet"] is not None
    assert d1_first["forecast"]["horizon"] == 6
    assert d1_first["forecast"]["issue_time"] == "2019-05-01T02:30:00Z"
    assert d1_first["forecast"]["target_time"] == "2019-05-01T03:30:00Z"
    assert round(d1_first["temp_eff"], 2) == d1_first["temp_eff"]

    # Check D1 host without fan/inlet (qh2-rcc196) reports null + available=False, never 0
    no_fan = next(h for h in hosts if h["id"] == "qh2-rcc196")
    assert no_fan["fans"] is None
    assert no_fan["fans_available"] is False
    assert no_fan["sensors"]["fans"]["available"] is False
    assert no_fan["sensors"]["fans"]["value"] is None
    assert no_fan["inlet"] is None
    assert no_fan["sensors"]["inlet"]["available"] is False

    # Check D2 hosts all report null + available=False for fans & inlet
    d2_hosts = [h for h in hosts if h["cluster"] == "d2"]
    assert all(h["fans"] is None and h["fans_available"] is False for h in d2_hosts)
    assert all(h["inlet"] is None and h["inlet_available"] is False for h in d2_hosts)


def test_snapshot_gap_marks_stale(synthetic_client):
    """Host qh2-rcc120 has missing steps 20..23 (03:20..03:50); at 03:50 gap is 40 min -> stale."""
    r = synthetic_client.get("/api/snapshot?cluster=d1&t=2019-05-01T03:50:00Z&horizon=0")
    assert r.status_code == 200
    hosts = r.json()["hosts"]
    h0 = next(h for h in hosts if h["id"] == "qh2-rcc120")
    assert h0["status"] == "stale"
    h1 = next(h for h in hosts if h["id"] == "qh2-rcc121")
    assert h1["status"] == "ok"


def test_compare_strict_causality_and_stats(synthetic_client):
    """Twin prediction for target_t=03:00Z at h=6 MUST be the forecast issued at 02:00Z."""
    target_t = "2019-05-01T03:00:00Z"
    r = synthetic_client.get(f"/api/compare?cluster=d1&t={target_t}&horizon=6")
    assert r.status_code == 200
    data = r.json()
    assert data["target_time"] == target_t
    assert data["issue_time"] == "2019-05-01T02:00:00Z"
    assert data["partition"] == "TEST"
    assert data["in_sample_warning"] is None

    # Fetch snapshot issued at 02:00Z h=6 and verify exact equality with compare's twin field
    snap_issue = synthetic_client.get("/api/snapshot?cluster=d1&t=2019-05-01T02:00:00Z&horizon=6").json()
    issue_fc_by_id = {h["id"]: h["forecast"] for h in snap_issue["hosts"]}

    h1 = next(h for h in data["hosts"] if h["id"] == "qh2-rcc121")
    assert h1["twin"] == issue_fc_by_id["qh2-rcc121"]["point"]
    assert h1["error"] == round(h1["twin"] - h1["reality"], 2)
    assert h1["inside"] == (h1["lower"] <= h1["reality"] <= h1["upper"])

    # Stats cross-check
    valid = [h for h in data["hosts"] if h["status"] == "ok"]
    expected_mae = round(sum(abs(h["error"]) for h in valid) / len(valid), 2)
    assert abs(data["stats"]["mae"] - expected_mae) <= 0.01
    assert data["stats"]["n_valid"] == len(valid)
    assert data["stats_24h"]["n_valid"] >= len(valid)


def test_compare_in_sample_warning(synthetic_client):
    """When issue_time falls in TRAIN/VAL, API must label in-sample warning."""
    r = synthetic_client.get("/api/compare?cluster=d1&t=2019-05-01T01:30:00Z&horizon=6")
    assert r.status_code == 200
    data = r.json()
    assert data["issue_time"] == "2019-05-01T00:30:00Z"
    assert data["partition"] == "TRAIN"
    assert data["in_sample_warning"] == "in-sample - not a fair test"


def test_series_and_alerts_endpoints(synthetic_client):
    r = synthetic_client.get(
        "/api/series?cluster=d1&host=qh2-rcc121&start=2019-05-01T00:00:00Z&end=2019-05-01T03:00:00Z&horizon=6"
    )
    assert r.status_code == 200
    s = r.json()
    n = len(s["times"])
    assert n > 0
    for key in ("actual", "cpu1", "cpu2", "inlet", "power", "load", "point", "lower", "upper"):
        assert len(s[key]) == n
    assert [fh["horizon_steps"] for fh in s["fan_horizons"]] == [1, 6, 36]

    ra = synthetic_client.get("/api/alerts?cluster=all&t=2019-05-01T02:30:00Z")
    assert ra.status_code == 200
    ad = ra.json()
    assert ad["count"] >= 4
    assert ad["alerts"][0]["risk"] == "critical"


def test_whatif_endpoint(synthetic_client):
    payload = {
        "cluster": "d1",
        "host": "qh2-rcc121",
        "workload": {"cpu_load": 70.0, "cores_used": 32.0, "ram_used": 40.0, "vms": 8.0, "net_rx": 120000.0, "net_tx": 90000.0},
    }
    r = synthetic_client.post("/api/whatif", json=payload)
    assert r.status_code == 200
    w = r.json()
    assert w["lower"] <= w["point"] <= w["upper"]
    assert w["note"] == "predictive, correlational - not causal"
    assert w["test_mae"] == 4.2
    assert len(w["sensitivity_curve"]) == 11
    # Increasing load monotonically increases point in synthetic model
    assert w["sensitivity_curve"][-1]["point"] > w["sensitivity_curve"][0]["point"]

    # Explicit hour/weekday calendar fields + new-server cluster-median defaults
    r_cal = synthetic_client.post(
        "/api/whatif",
        json={"cluster": "d2", "hour": 14, "weekday": 4, "workload": {"cpu_load": 50.0}},
    )
    assert r_cal.status_code == 200
    assert r_cal.json()["capacity"]["cores"] == 80.0
    assert r_cal.json()["capacity"]["ram_gb"] == 192.0
    assert synthetic_client.post("/api/whatif", json={"cluster": "d1", "hour": 25}).status_code == 422


def test_503_on_missing_model():
    def _failing_factory(c: str):
        raise FileNotFoundError(f"models/twin/{c}_metadata.json not found")

    svc = Twin3DService(engine_factory=_failing_factory)
    client = TestClient(create_app(service=svc, web_dir=_REPO / "nonexistent_web_dir"))

    assert client.get("/api/health").status_code == 503
    assert client.get("/api/meta").status_code == 503
    assert client.get("/api/snapshot?cluster=d1&t=2019-05-01T02:00:00Z&horizon=1").status_code == 503
    assert client.post("/api/whatif", json={"cluster": "d1"}).status_code == 503
