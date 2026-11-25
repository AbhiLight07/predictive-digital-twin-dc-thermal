"""FastAPI local API for the 3D predictive digital twin.

Serves recorded telemetry, model-driven CQR forecasts, compare-mode twin vs
reality diagnostics, alerts, time series, and planning-model what-if queries.
All numeric telemetry/predictions come strictly from TwinEngine, the saved
LightGBM models, and the precomputed forecast cache.
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ttwin.twin.engine import TwinEngine
from ttwin.twin.schema import DataQuality, RiskLevel

_REPO = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_WEB_DIR = _REPO / "twin3d" / "web"

_BANNER = (
    "Offline replay of recorded telemetry - schematic layout, not physical rack positions - thresholds illustrative"
)

_WHAT_TWIN_DOES_NOT_KNOW = [
    "Physical rack positions or room floorplan",
    "Cooling units, CRAC/CRAH setpoints, or chilled-water flow",
    "Room or outdoor ambient weather temperature",
    "Per-VM placement, resource quotas, or tenant workloads",
    "Live telemetry feed (offline historical replay only)",
]

_COLOURS_META = {
    "temperature_ramp": [
        {"t": 0.00, "hex": "#440154"},
        {"t": 0.25, "hex": "#3b528b"},
        {"t": 0.50, "hex": "#21918c"},
        {"t": 0.75, "hex": "#5ec962"},
        {"t": 1.00, "hex": "#fde725"},
    ],
    "cividis_ramp": [
        {"t": 0.00, "hex": "#00204c"},
        {"t": 0.25, "hex": "#414d6b"},
        {"t": 0.50, "hex": "#7c7b78"},
        {"t": 0.75, "hex": "#b9ac70"},
        {"t": 1.00, "hex": "#ffea46"},
    ],
    "risk": {
        "normal": "#2ca25f",
        "warning": "#fdae61",
        "critical": "#d7191c",
        "unknown": "#9e9e9e",
    },
    "error_diverging": {
        "below": "#2166ac",
        "zero": "#f7f7f7",
        "above": "#b2182b",
        "saturate_c": 3.0,
    },
}

_RACK_SIZE = 20
_RACK_PITCH_X = 0.9
_SLAB_W, _SLAB_H, _SLAB_D = 0.48, 0.0845, 0.72
_RACK_W, _RACK_D, _RACK_H = 0.6, 1.0, 1.87
_CLUSTER_Z = {"d1": 0.0, "d2": -4.0}
_STALE_NS = int(30 * 60 * 1_000_000_000)

_SIGNAL_UNITS = {
    "temp_eff": "°C",
    "cpu1": "°C",
    "cpu2": "°C",
    "inlet": "°C",
    "fans": "RPM",
    "power": "W",
    "load": "%",
    "ram_used_pct": "%",
    "ram_used_gb": "GB",
    "cores_used": "cores",
    "vms": "count",
    "net_rx": "B/s",
    "net_tx": "B/s",
}

_RAW_SIGNAL_COLS = [
    "CPU_Temperature",
    "CPU1_Temperature",
    "CPU2_Temperature",
    "Inlet_Temperature",
    "Fan_speed1",
    "Fan_speed2",
    "Fan_speed3",
    "Fan_speed4",
    "Power",
    "CPU_Load",
    "Ram",
    "Ram_Used",
    "CPU_cores",
    "CPU_cores_used",
    "No_Of_Running_vms",
    "Network_RX",
    "Network_TX",
]


# ── Helpers ──────────────────────────────────────────────────────────


def _host_sort_key(host_id: str) -> Tuple[int, str]:
    m = re.search(r"(\d+)$", host_id)
    return (int(m.group(1)) if m else 0, host_id)


def _r2(val: Any) -> Optional[float]:
    if val is None:
        return None
    f = float(val)
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, 2)


def _parse_ts(t_str: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(t_str)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid timestamp: {t_str}") from exc
    if pd.isna(ts):
        raise HTTPException(status_code=400, detail=f"Invalid timestamp: {t_str}")
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _to_iso(ts: Any) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts_to_ns(ts: pd.Timestamp) -> int:
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return int(ts.as_unit("ns").value)


def _layout(cluster: str, idx: int) -> dict:
    rack = idx // _RACK_SIZE
    slot = idx % _RACK_SIZE
    x = rack * _RACK_PITCH_X
    y = 0.05 + slot * 0.0889 + 0.0445
    z = _CLUSTER_Z[cluster]
    c_label = "C1" if cluster == "d1" else "C2"
    return {
        "rack": rack,
        "slot": slot,
        "rack_label": f"{c_label}-R{rack + 1}",
        "x": round(x, 4),
        "y": round(y, 4),
        "z": round(z, 4),
    }


def _sensor_entry(value: Any, available: bool, quality: str) -> dict:
    if not available:
        return {"value": None, "available": False, "quality": "missing"}
    if value is None:
        return {"value": None, "available": True, "quality": "missing"}
    return {"value": value, "available": True, "quality": quality}


# ── Pydantic schemas ─────────────────────────────────────────────────


class CapacityInput(BaseModel):
    cores: Optional[float] = None
    ram_gb: Optional[float] = None


class WorkloadInput(BaseModel):
    cpu_load: float = Field(default=30.0, ge=0.0, le=100.0)
    cores_used: float = Field(default=16.0, ge=0.0)
    ram_used: float = Field(default=32.0, ge=0.0)
    vms: float = Field(default=4.0, ge=0.0)
    net_rx: float = Field(default=100000.0, ge=0.0)
    net_tx: float = Field(default=100000.0, ge=0.0)


class WhatIfRequest(BaseModel):
    cluster: str = "d1"
    host: Optional[str] = None
    capacity: Optional[CapacityInput] = None
    workload: WorkloadInput = Field(default_factory=WorkloadInput)
    hour: Optional[int] = Field(default=None, ge=0, le=23)
    weekday: Optional[int] = Field(default=None, ge=0, le=6)
    include_curve: bool = True


def _load_twin_thresholds(cluster: str) -> Tuple[float, float]:
    cfg_path = _REPO / "configs" / "twin.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ds = "dataset1" if cluster == "d1" else "dataset2"
    rt = cfg["risk_thresholds"][ds]
    return float(rt["warning_celsius"]), float(rt["critical_celsius"])


# ── Service layer (indexed for <100 ms snapshots) ────────────────────


class _ClusterIndex:
    """Pre-indexed cluster telemetry, static metadata, layout, and forecast lookup."""

    def __init__(self, cluster: str, engine: Any, cache_df: Optional[pd.DataFrame]):
        self.cluster = cluster
        self.engine = engine
        self.cache_df = cache_df

        raw_hosts = list(engine.hosts())
        self.hosts: List[str] = sorted(raw_hosts, key=_host_sort_key)
        self.host_to_idx: Dict[str, int] = {h: i for i, h in enumerate(self.hosts)}
        self.layouts: Dict[str, dict] = {h: _layout(cluster, i) for i, h in enumerate(self.hosts)}

        self.statics: Dict[str, Any] = {h: engine.static(h) for h in self.hosts}
        cores_vals = [float(s.cores) for s in self.statics.values() if s.cores]
        ram_vals = [float(s.ram_gb) for s in self.statics.values() if s.ram_gb]
        self.default_cores = float(np.median(cores_vals)) if cores_vals else 0.0
        self.default_ram_gb = float(np.median(ram_vals)) if ram_vals else 0.0

        # Telemetry numpy index when underlying DataFrame is present
        self.has_df_index = False
        self.host_times_ns: Dict[str, np.ndarray] = {}
        self.host_cols: Dict[str, Dict[str, np.ndarray]] = {}
        self.host_df: Dict[str, pd.DataFrame] = {}

        data = getattr(engine, "_data", None)
        if isinstance(data, pd.DataFrame) and len(data) > 0 and "Series" in data.columns and "Time" in data.columns:
            self._build_df_index(data)

        # Forecast cache lookup: (horizon, ts_ns) -> dict[host, dict]
        self.cache_by_ht: Dict[Tuple[int, int], Dict[str, dict]] = {}
        self.cache_sorted_times_ns: Dict[int, np.ndarray] = {}
        self.cache_rows_count = 0
        if cache_df is not None and len(cache_df) > 0:
            self._build_cache_index(cache_df)

        # Legend percentiles (computed once from data & cache)
        self.legend = self._compute_legend(data, cache_df)

        # Replay range & partition boundaries
        self.time_min, self.time_max = self._compute_time_range(data, cache_df)
        self.meta_dict = getattr(engine, "_meta", {}) or {}
        self.models_info = getattr(engine, "_models_info", {}) or {}
        cfg_warn, cfg_crit = _load_twin_thresholds(cluster)
        self.warn_thresh = float(getattr(engine, "_warn_thresh", cfg_warn))
        self.crit_thresh = float(getattr(engine, "_crit_thresh", cfg_crit))

    def _build_df_index(self, data: pd.DataFrame) -> None:
        self.has_df_index = True
        avail_cols = [c for c in _RAW_SIGNAL_COLS if c in data.columns]
        for host, grp in data.groupby("Series", sort=False):
            if host not in self.host_to_idx:
                continue
            g = grp.sort_values("Time").reset_index(drop=True)
            t_col = pd.to_datetime(g["Time"])
            if getattr(t_col.dt, "tz", None) is not None:
                t_col = t_col.dt.tz_convert("UTC").dt.tz_localize(None)
            self.host_times_ns[host] = t_col.to_numpy(dtype="datetime64[ns]").view("int64")
            self.host_cols[host] = {c: g[c].to_numpy() for c in avail_cols}
            self.host_df[host] = g

    def _build_cache_index(self, cache_df: pd.DataFrame) -> None:
        df = cache_df.copy()
        t_col = pd.to_datetime(df["time"])
        if getattr(t_col.dt, "tz", None) is not None:
            t_col = t_col.dt.tz_convert("UTC").tz_localize(None)
        df["_ts_ns"] = t_col.to_numpy(dtype="datetime64[ns]").view("int64")
        self.cache_rows_count = len(df)

        hosts_arr = df["host"].astype(str).to_numpy()
        horizons_arr = df["horizon"].to_numpy(dtype=np.int64)
        ts_arr = df["_ts_ns"].to_numpy(dtype=np.int64)
        pt_arr = df["point"].to_numpy(dtype=np.float64)
        lo_arr = df["lower"].to_numpy(dtype=np.float64)
        hi_arr = df["upper"].to_numpy(dtype=np.float64)
        pe_arr = df["persistence"].to_numpy(dtype=np.float64)
        rk_arr = df["risk"].astype(str).to_numpy()
        af_arr = (
            df["actual_future"].to_numpy(dtype=np.float64)
            if "actual_future" in df.columns
            else np.full(len(df), np.nan)
        )

        by_ht: Dict[Tuple[int, int], Dict[str, dict]] = {}
        times_by_h: Dict[int, set] = {}
        for i in range(len(df)):
            h = int(horizons_arr[i])
            t_ns = int(ts_arr[i])
            key = (h, t_ns)
            bucket = by_ht.get(key)
            if bucket is None:
                bucket = {}
                by_ht[key] = bucket
            af = af_arr[i]
            bucket[hosts_arr[i]] = {
                "point": _r2(pt_arr[i]),
                "lower": _r2(lo_arr[i]),
                "upper": _r2(hi_arr[i]),
                "persistence": _r2(pe_arr[i]),
                "risk": rk_arr[i],
                "actual_future": _r2(af) if not np.isnan(af) else None,
                "_raw_point": float(pt_arr[i]),
                "_raw_lower": float(lo_arr[i]),
                "_raw_upper": float(hi_arr[i]),
                "_raw_af": None if np.isnan(af) else float(af),
            }
            times_by_h.setdefault(h, set()).add(t_ns)

        self.cache_by_ht = by_ht
        self.cache_sorted_times_ns = {
            h: np.array(sorted(ts_set), dtype=np.int64) for h, ts_set in times_by_h.items()
        }

    def _compute_legend(self, data: Optional[pd.DataFrame], cache_df: Optional[pd.DataFrame]) -> dict:
        legend: Dict[str, Optional[float]] = {
            "t_lo": None,
            "t_hi": None,
            "p_lo": None,
            "p_hi": None,
            "rpm_ref": None,
            "w_ref": None,
        }
        if isinstance(data, pd.DataFrame) and len(data) > 0:
            if "CPU_Temperature" in data.columns:
                vals = data["CPU_Temperature"].dropna().to_numpy()
                if len(vals) > 0:
                    legend["t_lo"] = _r2(np.nanpercentile(vals, 5))
                    legend["t_hi"] = _r2(np.nanpercentile(vals, 99.5))
            if "Power" in data.columns:
                pvals = data["Power"].dropna().to_numpy()
                if len(pvals) > 0:
                    legend["p_lo"] = _r2(np.nanpercentile(pvals, 5))
                    legend["p_hi"] = _r2(np.nanpercentile(pvals, 99.5))
            if getattr(self.engine, "_has_fan", False):
                fan_cols = [
                    c for c in ("Fan_speed1", "Fan_speed2", "Fan_speed3", "Fan_speed4")
                    if c in data.columns
                ]
                if fan_cols:
                    fan_vals = np.concatenate([data[c].dropna().to_numpy() for c in fan_cols])
                    if len(fan_vals) > 0:
                        legend["rpm_ref"] = _r2(np.nanpercentile(fan_vals, 95))

        if cache_df is not None and len(cache_df) > 0:
            widths = (cache_df["upper"] - cache_df["lower"]).dropna().to_numpy()
            if len(widths) > 0:
                legend["w_ref"] = _r2(np.nanpercentile(widths, 95))
        return legend

    def _compute_time_range(
        self, data: Optional[pd.DataFrame], cache_df: Optional[pd.DataFrame]
    ) -> Tuple[Optional[str], Optional[str]]:
        t_min, t_max = None, None
        if isinstance(data, pd.DataFrame) and len(data) > 0 and "Time" in data.columns:
            t_min = _to_iso(data["Time"].min())
            t_max = _to_iso(data["Time"].max())
        elif cache_df is not None and len(cache_df) > 0:
            t_min = _to_iso(cache_df["time"].min())
            t_max = _to_iso(cache_df["time"].max())
        return t_min, t_max


class Twin3DService:
    """Thread-safe / single-process coordinator for TwinEngine + forecast cache."""

    def __init__(
        self,
        engines: Optional[Dict[str, Any]] = None,
        caches: Optional[Dict[str, pd.DataFrame]] = None,
        load_errors: Optional[Dict[str, str]] = None,
        clusters: Tuple[str, ...] = ("d1", "d2"),
        engine_factory: Optional[Callable[[str], Any]] = None,
        cache_loader: Optional[Callable[[str], Optional[pd.DataFrame]]] = None,
    ):
        self.clusters = clusters
        self._engine_factory = engine_factory or (lambda c: TwinEngine(c))
        self._cache_loader = cache_loader or self._default_cache_loader
        self._indexes: Dict[str, _ClusterIndex] = {}
        self._load_errors: Dict[str, str] = dict(load_errors or {})
        self._live_fc_cache: OrderedDict[Tuple[str, int, int], Dict[str, dict]] = OrderedDict()

        if engines is not None:
            for c, eng in engines.items():
                if c in self._load_errors:
                    continue
                try:
                    cache_df = caches.get(c) if caches is not None else None
                    self._register_engine(c, eng, cache_df)
                except Exception as exc:
                    self._load_errors[c] = str(exc)

    @staticmethod
    def _default_cache_loader(cluster: str) -> Optional[pd.DataFrame]:
        path = _REPO / "data" / "twin_cache" / f"{cluster}_forecasts.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None

    def _register_engine(self, cluster: str, eng: Any, cache_df: Optional[pd.DataFrame]) -> _ClusterIndex:
        models_dict = getattr(eng, "_models", None)
        if isinstance(models_dict, dict) and len(models_dict) == 0:
            raise RuntimeError(f"No trained twin models loaded for cluster '{cluster}'")
        idx = _ClusterIndex(cluster, eng, cache_df)
        self._indexes[cluster] = idx
        return idx

    def ensure_cluster(self, cluster: str) -> _ClusterIndex:
        if cluster in self._load_errors:
            raise HTTPException(
                status_code=503,
                detail=f"Model or data not available for cluster '{cluster}': {self._load_errors[cluster]}",
            )
        if cluster in self._indexes:
            return self._indexes[cluster]
        if cluster not in self.clusters:
            raise HTTPException(status_code=400, detail=f"Unknown cluster '{cluster}' (expected d1 or d2)")
        try:
            eng = self._engine_factory(cluster)
            cache_df = self._cache_loader(cluster)
            return self._register_engine(cluster, eng, cache_df)
        except HTTPException:
            raise
        except Exception as exc:
            self._load_errors[cluster] = str(exc)
            raise HTTPException(
                status_code=503,
                detail=f"Failed to load twin engine/models for '{cluster}': {exc}",
            ) from exc

    def ensure_all(self) -> Dict[str, _ClusterIndex]:
        for c in self.clusters:
            if c not in self._indexes and c not in self._load_errors:
                try:
                    self.ensure_cluster(c)
                except HTTPException:
                    pass
        return self._indexes

    # ── Fast state lookup ────────────────────────────────────────────

    def _host_state_fast(self, cidx: _ClusterIndex, host: str, t_ns: int, t_ts: pd.Timestamp) -> dict:
        static = cidx.statics[host]
        has_fan = bool(static.has_fan)
        has_inlet = bool(static.has_inlet)

        if not cidx.has_df_index:
            st = cidx.engine.state(host, t_ts)
            quality = st.cpu_temperature.quality.value
            temp_val = _r2(st.cpu_temperature.value)
            status = "missing" if (quality == "missing" or temp_val is None) else ("stale" if quality == "stale" else "ok")
            fans_vals = (
                [
                    _r2(st.fan_speed1.value),
                    _r2(st.fan_speed2.value),
                    _r2(st.fan_speed3.value),
                    _r2(st.fan_speed4.value),
                ]
                if has_fan
                else None
            )
            ram_pct = None
            if st.ram_used.value is not None and st.ram.value and st.ram.value > 0:
                ram_pct = _r2(st.ram_used.value / st.ram.value * 100.0)
            return {
                "status": status,
                "temp_eff": _r2(st.cpu_temperature.value),
                "cpu1": _r2(st.cpu1_temperature.value),
                "cpu2": _r2(st.cpu2_temperature.value),
                "inlet": _r2(st.inlet_temperature.value) if has_inlet else None,
                "inlet_available": has_inlet,
                "fans": fans_vals,
                "fans_available": has_fan,
                "power": _r2(st.power.value),
                "load": _r2(st.cpu_load.value),
                "ram_used_pct": ram_pct,
                "ram_used_gb": _r2(st.ram_used.value),
                "cores_used": _r2(st.cpu_cores_used.value),
                "vms": _r2(st.no_of_running_vms.value),
                "net_rx": _r2(st.network_rx.value),
                "net_tx": _r2(st.network_tx.value),
                "_obs_quality": quality,
            }

        times = cidx.host_times_ns.get(host)
        if times is None or len(times) == 0:
            return self._empty_host_state(has_fan, has_inlet)

        pos = int(np.searchsorted(times, t_ns, side="right")) - 1
        if pos < 0:
            return self._empty_host_state(has_fan, has_inlet)

        gap_ns = t_ns - int(times[pos])
        base_quality = "stale" if gap_ns > _STALE_NS else "ok"
        cols = cidx.host_cols[host]

        def _val(col: str) -> Optional[float]:
            arr = cols.get(col)
            if arr is None:
                return None
            return _r2(arr[pos])

        temp_eff = _val("CPU_Temperature")
        cpu1 = _val("CPU1_Temperature")
        cpu2 = _val("CPU2_Temperature")
        if temp_eff is None and cpu1 is not None and cpu2 is not None:
            temp_eff = max(cpu1, cpu2)

        inlet = _val("Inlet_Temperature") if has_inlet else None
        fans = (
            [_val("Fan_speed1"), _val("Fan_speed2"), _val("Fan_speed3"), _val("Fan_speed4")]
            if has_fan
            else None
        )
        ram_total = cols["Ram"][pos] if "Ram" in cols else np.nan
        ram_used = cols["Ram_Used"][pos] if "Ram_Used" in cols else np.nan
        ram_pct = None
        if ram_total is not None and ram_used is not None and not np.isnan(ram_total) and not np.isnan(ram_used) and ram_total > 0:
            ram_pct = _r2(float(ram_used) / float(ram_total) * 100.0)

        status = "missing" if temp_eff is None else ("stale" if base_quality == "stale" else "ok")
        return {
            "status": status,
            "temp_eff": temp_eff,
            "cpu1": cpu1,
            "cpu2": cpu2,
            "inlet": inlet,
            "inlet_available": has_inlet,
            "fans": fans,
            "fans_available": has_fan,
            "power": _val("Power"),
            "load": _val("CPU_Load"),
            "ram_used_pct": ram_pct,
            "ram_used_gb": _val("Ram_Used"),
            "cores_used": _val("CPU_cores_used"),
            "vms": _val("No_Of_Running_vms"),
            "net_rx": _val("Network_RX"),
            "net_tx": _val("Network_TX"),
            "_obs_quality": base_quality,
        }

    @staticmethod
    def _empty_host_state(has_fan: bool, has_inlet: bool) -> dict:
        return {
            "status": "missing",
            "temp_eff": None,
            "cpu1": None,
            "cpu2": None,
            "inlet": None,
            "inlet_available": has_inlet,
            "fans": None,
            "fans_available": has_fan,
            "power": None,
            "load": None,
            "ram_used_pct": None,
            "ram_used_gb": None,
            "cores_used": None,
            "vms": None,
            "net_rx": None,
            "net_tx": None,
            "_obs_quality": "missing",
        }

    # ── Fast causal forecast lookup (cache + bounded on-demand batch) ─

    def _get_forecasts_at_issue_time(
        self, cidx: _ClusterIndex, issue_ts: pd.Timestamp, issue_ns: int, horizon: int
    ) -> Dict[str, dict]:
        if horizon == 0:
            return {}

        # 1. Exact cache lookup
        cached = cidx.cache_by_ht.get((horizon, issue_ns))
        if cached is not None:
            return cached

        # 2. Live LRU cache lookup
        lru_key = (cidx.cluster, issue_ns, horizon)
        if lru_key in self._live_fc_cache:
            self._live_fc_cache.move_to_end(lru_key)
            return self._live_fc_cache[lru_key]

        # 3. Compute causally using strictly data with Time <= issue_ts
        result = self._compute_live_batch_forecasts(cidx, issue_ts, issue_ns, horizon)
        self._live_fc_cache[lru_key] = result
        while len(self._live_fc_cache) > 128:
            self._live_fc_cache.popitem(last=False)
        return result

    def _compute_live_batch_forecasts(
        self, cidx: _ClusterIndex, issue_ts: pd.Timestamp, issue_ns: int, horizon: int
    ) -> Dict[str, dict]:
        eng = cidx.engine
        info = cidx.models_info.get(f"monitoring_{horizon}")
        models_dict = getattr(eng, "_models", None)

        # Fallback for lightweight synthetic engines without _build_monitoring_features
        if (
            not cidx.has_df_index
            or info is None
            or not isinstance(models_dict, dict)
            or not hasattr(eng, "_build_monitoring_features")
            or "feature_names" not in info
        ):
            out: Dict[str, dict] = {}
            for host in cidx.hosts:
                fcs = eng.forecast(host, issue_ts)
                for fc in fcs:
                    if fc.horizon_steps == horizon:
                        pt = _r2(fc.point)
                        lo = _r2(fc.lower)
                        hi = _r2(fc.upper)
                        pe = _r2(fc.persistence)
                        if pt is not None:
                            out[host] = {
                                "point": pt,
                                "lower": lo,
                                "upper": hi,
                                "persistence": pe,
                                "risk": fc.risk.value if hasattr(fc.risk, "value") else str(fc.risk),
                                "_raw_point": float(fc.point),
                                "_raw_lower": float(fc.lower),
                                "_raw_upper": float(fc.upper),
                            }
            return out

        m_pt = models_dict.get(f"{cidx.cluster}_monitoring_{horizon}_point")
        if m_pt is None:
            raise HTTPException(
                status_code=503,
                detail=f"Missing monitoring model '{cidx.cluster}_monitoring_{horizon}_point'",
            )
        m_lo = models_dict.get(f"{cidx.cluster}_monitoring_{horizon}_lower")
        m_hi = models_dict.get(f"{cidx.cluster}_monitoring_{horizon}_upper")

        # Gather strictly past 420-min windows (42 regular 10-min steps > max rolling=36 & lag=12)
        window_start_ns = issue_ns - int(420 * 60 * 1_000_000_000)
        tail_frames = []
        for host in cidx.hosts:
            times = cidx.host_times_ns.get(host)
            if times is None or len(times) == 0:
                continue
            end_pos = int(np.searchsorted(times, issue_ns, side="right"))
            if end_pos < 2:
                continue
            if (issue_ns - int(times[end_pos - 1])) > _STALE_NS:
                continue
            start_pos = int(np.searchsorted(times, window_start_ns, side="left"))
            tail_frames.append(cidx.host_df[host].iloc[start_pos:end_pos])

        if not tail_frames:
            return {}

        combined = pd.concat(tail_frames, ignore_index=True)
        feat_df = eng._build_monitoring_features(combined)
        last_rows = feat_df.groupby("Series", sort=False).tail(1)

        feat_names = info["feature_names"]
        needed = list(dict.fromkeys(feat_names + ["CPU_Temperature", "Series"]))
        avail_cols = [c for c in needed if c in last_rows.columns]
        if len(avail_cols) < len(needed):
            return {}
        valid = last_rows[needed].dropna()
        if len(valid) == 0:
            return {}

        X = valid[feat_names].to_numpy(dtype=np.float64)
        curr_temp = valid["CPU_Temperature"].to_numpy(dtype=np.float64)
        hosts_valid = valid["Series"].astype(str).to_numpy()

        q_hat = float(info.get("q_hat", 0.0))
        point_abs = curr_temp + m_pt.predict(X)
        if m_lo is not None and m_hi is not None:
            lower_abs = curr_temp + m_lo.predict(X) - q_hat
            upper_abs = curr_temp + m_hi.predict(X) + q_hat
        else:
            lower_abs = point_abs - abs(q_hat)
            upper_abs = point_abs + abs(q_hat)

        swap = lower_abs > upper_abs
        if np.any(swap):
            lower_abs[swap], upper_abs[swap] = upper_abs[swap], lower_abs[swap]

        out = {}
        for i, h_id in enumerate(hosts_valid):
            up = float(upper_abs[i])
            risk = eng._assess_risk(up).value
            out[h_id] = {
                "point": _r2(point_abs[i]),
                "lower": _r2(lower_abs[i]),
                "upper": _r2(up),
                "persistence": _r2(curr_temp[i]),
                "risk": risk,
                "_raw_point": float(point_abs[i]),
                "_raw_lower": float(lower_abs[i]),
                "_raw_upper": float(up),
            }
        return out

    # ── Endpoint handlers ────────────────────────────────────────────

    def health(self, cluster: Optional[str] = None) -> dict:
        if cluster:
            cidx = self.ensure_cluster(cluster)
            models_loaded = list(getattr(cidx.engine, "_models", {}).keys()) or list(cidx.models_info.keys())
            return {
                "status": "ok",
                "clusters": {
                    cluster: {
                        "loaded": True,
                        "host_count": len(cidx.hosts),
                        "models_count": len(models_loaded),
                        "cache_rows": cidx.cache_rows_count,
                    }
                },
                "has_simulate": hasattr(cidx.engine, "simulate"),
            }

        self.ensure_all()
        if not self._indexes:
            raise HTTPException(
                status_code=503,
                detail=f"No twin models loaded: {self._load_errors}",
            )
        clusters_info = {}
        has_sim = False
        for c in self.clusters:
            if c in self._indexes:
                cidx = self._indexes[c]
                models_loaded = list(getattr(cidx.engine, "_models", {}).keys()) or list(cidx.models_info.keys())
                has_sim = has_sim or hasattr(cidx.engine, "simulate")
                clusters_info[c] = {
                    "loaded": True,
                    "host_count": len(cidx.hosts),
                    "models_count": len(models_loaded),
                    "cache_rows": cidx.cache_rows_count,
                }
            else:
                clusters_info[c] = {
                    "loaded": False,
                    "error": self._load_errors.get(c, "not loaded"),
                }
        return {
            "status": "ok" if not self._load_errors else "degraded",
            "clusters": clusters_info,
            "has_simulate": has_sim,
        }

    def meta(self) -> dict:
        self.ensure_all()
        if not self._indexes:
            raise HTTPException(status_code=503, detail=f"Models not available: {self._load_errors}")

        clusters_meta = {}
        has_sim = False
        for cid, cidx in self._indexes.items():
            eng = cidx.engine
            has_sim = has_sim or hasattr(eng, "simulate")
            hosts_meta = []
            for h in cidx.hosts:
                st = cidx.statics[h]
                ly = cidx.layouts[h]
                hosts_meta.append({
                    "id": h,
                    "idx": cidx.host_to_idx[h],
                    **ly,
                    "cores": int(st.cores),
                    "ram_gb": _r2(st.ram_gb),
                    "hardware_model": st.hardware_model,
                    "sensors": {
                        "fans": bool(st.has_fan),
                        "inlet": bool(st.has_inlet),
                        "power": True,
                    },
                })

            mon1 = cidx.models_info.get("monitoring_1", {})
            boundaries = mon1.get("boundaries", {})
            norm_boundaries = {
                k: _to_iso(v) for k, v in boundaries.items() if v is not None
            }
            models_summary = []
            for mod in cidx.meta_dict.get("models", []):
                models_summary.append({
                    "kind": mod.get("kind"),
                    "horizon": mod.get("horizon"),
                    "test_mae": _r2(mod.get("test_mae")),
                    "persistence_mae": _r2(mod.get("persistence_mae")),
                    "skill_score": round(float(mod["skill_score"]), 4) if mod.get("skill_score") is not None else None,
                    "empirical_coverage": _r2(
                        mod["empirical_coverage"] * 100.0 if mod.get("empirical_coverage") is not None else None
                    ),
                    "mean_interval_width": _r2(mod.get("mean_interval_width")),
                    "n_test_rows": mod.get("n_test_rows"),
                })

            cfg = getattr(eng, "_config", {}) or {}
            replay_cfg = cfg.get("replay", {"stride_minutes": 30, "sampling_interval_minutes": 10})
            test_start_iso = norm_boundaries.get("test_start") or cidx.time_min

            clusters_meta[cid] = {
                "cluster": cid,
                "label": "Cluster 1 (D1)" if cid == "d1" else "Cluster 2 (D2)",
                "host_count": len(cidx.hosts),
                "hosts": hosts_meta,
                "legend": cidx.legend,
                "replay_range": {
                    "min": cidx.time_min,
                    "max": cidx.time_max,
                    "test_start": test_start_iso,
                    "stride_minutes": int(replay_cfg.get("stride_minutes", 30)),
                    "sampling_interval_minutes": int(replay_cfg.get("sampling_interval_minutes", 10)),
                },
                "partition_boundaries": norm_boundaries,
                "thresholds": {
                    "warning": _r2(cidx.warn_thresh),
                    "critical": _r2(cidx.crit_thresh),
                    "provenance": "Derived from recorded CPU_Temperature percentiles (p90/p99); illustrative, not vendor limits",
                },
                "model_version": str(cidx.meta_dict.get("git_commit", "")),
                "models": models_summary,
                "layout": {
                    "rack_size": _RACK_SIZE,
                    "rack_pitch_x": _RACK_PITCH_X,
                    "slab": {"w": _SLAB_W, "h": _SLAB_H, "d": _SLAB_D},
                    "rack": {"w": _RACK_W, "d": _RACK_D, "h": _RACK_H},
                    "cluster_z": _CLUSTER_Z[cid],
                },
            }

        return {
            "banner": _BANNER,
            "horizons_steps": [0, 1, 6, 36],
            "horizons_minutes": [0, 10, 60, 360],
            "colours": _COLOURS_META,
            "has_simulate": has_sim,
            "what_twin_does_not_know": _WHAT_TWIN_DOES_NOT_KNOW,
            "clusters": clusters_meta,
        }

    def snapshot(self, cluster: str, t_str: str, horizon: int = 1) -> dict:
        if horizon not in (0, 1, 6, 36):
            raise HTTPException(status_code=400, detail="horizon must be 0, 1, 6, or 36")

        t_ts = _parse_ts(t_str)
        t_ns = _ts_to_ns(t_ts)
        target_clusters = list(self.clusters) if cluster == "all" else [cluster]

        hosts_out = []
        for cid in target_clusters:
            cidx = self.ensure_cluster(cid)
            part = cidx.engine.partition(t_ts)
            fc_map = self._get_forecasts_at_issue_time(cidx, t_ts, t_ns, horizon) if horizon > 0 else {}
            issue_iso = _to_iso(t_ts)
            target_iso = _to_iso(t_ts + pd.Timedelta(minutes=horizon * 10)) if horizon > 0 else issue_iso

            for h in cidx.hosts:
                st = self._host_state_fast(cidx, h, t_ns, t_ts)
                q = st.pop("_obs_quality")
                ly = cidx.layouts[h]

                fc_obj = None
                if horizon > 0:
                    fc = fc_map.get(h)
                    if fc is not None and fc.get("point") is not None:
                        width = None
                        if fc.get("upper") is not None and fc.get("lower") is not None:
                            width = _r2(fc["upper"] - fc["lower"])
                        fc_obj = {
                            "horizon": horizon,
                            "issue_time": issue_iso,
                            "target_time": target_iso,
                            "point": fc["point"],
                            "lower": fc["lower"],
                            "upper": fc["upper"],
                            "width": width,
                            "persistence": fc["persistence"],
                            "risk": fc["risk"],
                        }

                sensors = {
                    "temp_eff": _sensor_entry(st["temp_eff"], True, q),
                    "cpu1": _sensor_entry(st["cpu1"], True, q),
                    "cpu2": _sensor_entry(st["cpu2"], True, q),
                    "inlet": _sensor_entry(st["inlet"], st["inlet_available"], q),
                    "fans": _sensor_entry(st["fans"], st["fans_available"], q),
                    "power": _sensor_entry(st["power"], True, q),
                    "load": _sensor_entry(st["load"], True, q),
                    "ram_used_pct": _sensor_entry(st["ram_used_pct"], True, q),
                    "cores_used": _sensor_entry(st["cores_used"], True, q),
                    "vms": _sensor_entry(st["vms"], True, q),
                    "net_rx": _sensor_entry(st["net_rx"], True, q),
                    "net_tx": _sensor_entry(st["net_tx"], True, q),
                }

                hosts_out.append({
                    "cluster": cid,
                    "idx": cidx.host_to_idx[h],
                    "id": h,
                    "status": "missing" if st["temp_eff"] is None else st["status"],
                    "partition": part,
                    **ly,
                    **st,
                    "sensors": sensors,
                    "forecast": fc_obj,
                })

        return {
            "cluster": cluster,
            "time": _to_iso(t_ts),
            "horizon": horizon,
            "units": _SIGNAL_UNITS,
            "hosts": hosts_out,
        }

    def compare(self, cluster: str, t_str: str, horizon: int = 1) -> dict:
        if horizon not in (1, 6, 36):
            raise HTTPException(status_code=400, detail="Compare mode requires horizon 1, 6, or 36")

        cidx = self.ensure_cluster(cluster)
        target_ts = _parse_ts(t_str)
        target_ns = _ts_to_ns(target_ts)
        issue_ts = target_ts - pd.Timedelta(minutes=horizon * 10)
        issue_ns = _ts_to_ns(issue_ts)

        issue_part = cidx.engine.partition(issue_ts)
        in_sample_warning = "in-sample - not a fair test" if issue_part in ("TRAIN", "VAL_SEL", "VAL_CAL") else None

        # Forecast for target_ts MUST be issued at issue_ts = target_ts - horizon * 10 min
        fc_map = self._get_forecasts_at_issue_time(cidx, issue_ts, issue_ns, horizon)

        hosts_out = []
        abs_errors: List[float] = []
        signed_errors: List[float] = []
        n_inside = 0
        n_valid = 0

        for h in cidx.hosts:
            st_real = self._host_state_fast(cidx, h, target_ns, target_ts)
            reality = st_real["temp_eff"]
            fc = fc_map.get(h)
            ly = cidx.layouts[h]

            if reality is None or st_real["status"] == "stale":
                status = "missing" if reality is None else "stale"
                twin = fc["point"] if fc else None
                lower = fc["lower"] if fc else None
                upper = fc["upper"] if fc else None
                persistence = fc.get("persistence") if fc else None
                width = _r2(upper - lower) if upper is not None and lower is not None else None
                risk = fc["risk"] if fc else "unknown"
                hosts_out.append({
                    "idx": cidx.host_to_idx[h],
                    "id": h,
                    **ly,
                    "status": status,
                    "horizon": horizon,
                    "reality": reality,
                    "twin": twin,
                    "lower": lower,
                    "upper": upper,
                    "width": width,
                    "persistence": persistence,
                    "error": None,
                    "inside": None,
                    "risk": risk,
                })
                continue

            if fc is None or fc.get("point") is None:
                hosts_out.append({
                    "idx": cidx.host_to_idx[h],
                    "id": h,
                    **ly,
                    "status": "unknown",
                    "horizon": horizon,
                    "reality": reality,
                    "twin": None,
                    "lower": None,
                    "upper": None,
                    "width": None,
                    "persistence": None,
                    "error": None,
                    "inside": None,
                    "risk": "unknown",
                })
                continue

            twin = fc["point"]
            lower = fc["lower"]
            upper = fc["upper"]
            persistence = fc.get("persistence")
            width = _r2(upper - lower) if upper is not None and lower is not None else None
            err = round(twin - reality, 2)
            inside = bool(lower is not None and upper is not None and lower <= reality <= upper)
            abs_errors.append(abs(err))
            signed_errors.append(err)
            n_valid += 1
            if inside:
                n_inside += 1

            hosts_out.append({
                "idx": cidx.host_to_idx[h],
                "id": h,
                **ly,
                "status": "ok",
                "horizon": horizon,
                "reality": reality,
                "twin": twin,
                "lower": lower,
                "upper": upper,
                "width": width,
                "persistence": persistence,
                "error": err,
                "inside": inside,
                "risk": fc["risk"],
            })

        frame_stats = {
            "mae": _r2(sum(abs_errors) / len(abs_errors)) if abs_errors else None,
            "mean_error": _r2(sum(signed_errors) / len(signed_errors)) if signed_errors else None,
            "coverage_pct": round(n_inside / n_valid * 100.0, 2) if n_valid > 0 else None,
            "n_valid": n_valid,
            "n_inside": n_inside,
        }

        stats_24h = self._compute_compare_24h_stats(cidx, issue_ns, horizon, abs_errors, n_inside, n_valid)

        return {
            "cluster": cluster,
            "target_time": _to_iso(target_ts),
            "issue_time": _to_iso(issue_ts),
            "horizon": horizon,
            "partition": issue_part,
            "in_sample_warning": in_sample_warning,
            "stats": frame_stats,
            "stats_24h": stats_24h,
            "hosts": hosts_out,
        }

    def _compute_compare_24h_stats(
        self,
        cidx: _ClusterIndex,
        issue_ns: int,
        horizon: int,
        fallback_abs_errors: List[float],
        fallback_inside: int,
        fallback_valid: int,
    ) -> dict:
        times_arr = cidx.cache_sorted_times_ns.get(horizon)
        if times_arr is not None and len(times_arr) > 0:
            win_start = issue_ns - int(24 * 3600 * 1_000_000_000)
            i0 = int(np.searchsorted(times_arr, win_start, side="right"))
            i1 = int(np.searchsorted(times_arr, issue_ns, side="right"))
            if i1 > i0:
                errs = []
                ins = 0
                tot = 0
                for idx in range(i0, i1):
                    bucket = cidx.cache_by_ht.get((horizon, int(times_arr[idx])), {})
                    for row in bucket.values():
                        af = row.get("_raw_af")
                        pt = row.get("_raw_point")
                        lo = row.get("_raw_lower")
                        hi = row.get("_raw_upper")
                        if af is None or pt is None:
                            continue
                        errs.append(abs(pt - af))
                        tot += 1
                        if lo is not None and hi is not None and lo <= af <= hi:
                            ins += 1
                if tot > 0:
                    return {
                        "mae": _r2(sum(errs) / tot),
                        "coverage_pct": round(ins / tot * 100.0, 2),
                        "n_valid": tot,
                        "n_inside": ins,
                    }

        return {
            "mae": _r2(sum(fallback_abs_errors) / len(fallback_abs_errors)) if fallback_abs_errors else None,
            "coverage_pct": round(fallback_inside / fallback_valid * 100.0, 2) if fallback_valid > 0 else None,
            "n_valid": fallback_valid,
            "n_inside": fallback_inside,
        }

    def series(
        self,
        cluster: str,
        host: str,
        start_str: Optional[str],
        end_str: Optional[str],
        horizon: int = 1,
    ) -> dict:
        if horizon not in (0, 1, 6, 36):
            raise HTTPException(status_code=400, detail="horizon must be 0, 1, 6, or 36")

        cidx = self.ensure_cluster(cluster)
        if host not in cidx.host_to_idx:
            raise HTTPException(status_code=404, detail=f"Host '{host}' not found in cluster '{cluster}'")

        if end_str:
            end_ts = _parse_ts(end_str)
        elif cidx.time_max:
            end_ts = _parse_ts(cidx.time_max)
        else:
            raise HTTPException(status_code=400, detail="end timestamp required")

        start_ts = _parse_ts(start_str) if start_str else (end_ts - pd.Timedelta(hours=24))
        df = cidx.engine.history(host, start_ts, end_ts)

        times: List[str] = []
        actual: List[Optional[float]] = []
        cpu1: List[Optional[float]] = []
        cpu2: List[Optional[float]] = []
        inlet: List[Optional[float]] = []
        power: List[Optional[float]] = []
        load: List[Optional[float]] = []
        point: List[Optional[float]] = []
        lower: List[Optional[float]] = []
        upper: List[Optional[float]] = []
        persistence: List[Optional[float]] = []

        has_inlet = bool(cidx.statics[host].has_inlet)
        h_steps = horizon if horizon > 0 else 6
        delta_ns = int(h_steps * 10 * 60 * 1_000_000_000)

        if len(df) > 0:
            t_col = pd.to_datetime(df["Time"])
            if getattr(t_col.dt, "tz", None) is not None:
                t_col = t_col.dt.tz_convert("UTC").tz_localize(None)
            t_ns_arr = t_col.to_numpy(dtype="datetime64[ns]").view("int64")
            for i, (_, row) in enumerate(df.iterrows()):
                tau_ns = int(t_ns_arr[i])
                times.append(_to_iso(row["Time"]))
                t_eff = _r2(row["CPU_Temperature"]) if "CPU_Temperature" in row.index else None
                c1 = _r2(row["CPU1_Temperature"]) if "CPU1_Temperature" in row.index else None
                c2 = _r2(row["CPU2_Temperature"]) if "CPU2_Temperature" in row.index else None
                if t_eff is None and c1 is not None and c2 is not None:
                    t_eff = max(c1, c2)
                actual.append(t_eff)
                cpu1.append(c1)
                cpu2.append(c2)
                inlet.append(_r2(row["Inlet_Temperature"]) if has_inlet and "Inlet_Temperature" in row.index else None)
                power.append(_r2(row["Power"]) if "Power" in row.index else None)
                load.append(_r2(row["CPU_Load"]) if "CPU_Load" in row.index else None)

                # Causally issued forecast at target time tau (issued at tau - h*10m)
                issue_ns = tau_ns - delta_ns
                fc_bucket = cidx.cache_by_ht.get((h_steps, issue_ns), {})
                fc = fc_bucket.get(host)
                if fc is not None:
                    point.append(fc["point"])
                    lower.append(fc["lower"])
                    upper.append(fc["upper"])
                    persistence.append(fc["persistence"])
                else:
                    point.append(None)
                    lower.append(None)
                    upper.append(None)
                    persistence.append(None)

        # Forward fan horizons issued at end_ts
        end_ns = _ts_to_ns(end_ts)
        fan_horizons = []
        for fh in (1, 6, 36):
            fc_map = self._get_forecasts_at_issue_time(cidx, end_ts, end_ns, fh)
            fc = fc_map.get(host)
            if fc is not None:
                fan_horizons.append({
                    "horizon_steps": fh,
                    "target_time": _to_iso(end_ts + pd.Timedelta(minutes=fh * 10)),
                    "point": fc["point"],
                    "lower": fc["lower"],
                    "upper": fc["upper"],
                    "persistence": fc["persistence"],
                    "risk": fc["risk"],
                })

        return {
            "cluster": cluster,
            "host": host,
            "start": _to_iso(start_ts),
            "end": _to_iso(end_ts),
            "horizon": horizon,
            "times": times,
            "actual": actual,
            "cpu1": cpu1,
            "cpu2": cpu2,
            "inlet": inlet,
            "power": power,
            "load": load,
            "point": point,
            "lower": lower,
            "upper": upper,
            "persistence": persistence,
            "fan_horizons": fan_horizons,
        }

    def alerts(self, cluster: str, t_str: str) -> dict:
        t_ts = _parse_ts(t_str)
        t_ns = _ts_to_ns(t_ts)
        target_clusters = list(self.clusters) if cluster == "all" else [cluster]
        alerts_out: List[dict] = []

        for cid in target_clusters:
            cidx = self.ensure_cluster(cid)
            fc6_map = self._get_forecasts_at_issue_time(cidx, t_ts, t_ns, 6)
            fc36_map = self._get_forecasts_at_issue_time(cidx, t_ts, t_ns, 36)

            for h in cidx.hosts:
                f6 = fc6_map.get(h)
                f36 = fc36_map.get(h)
                triggered = []
                for hz, fc in ((6, f6), (36, f36)):
                    if fc and fc.get("risk") in ("warning", "critical"):
                        triggered.append((hz, fc))
                if not triggered:
                    continue

                # Highest severity, then largest upper bound
                triggered.sort(
                    key=lambda item: (1 if item[1]["risk"] == "critical" else 0, item[1].get("upper") or 0.0),
                    reverse=True,
                )
                worst_hz, worst_fc = triggered[0]
                st = self._host_state_fast(cidx, h, t_ns, t_ts)
                ly = cidx.layouts[h]

                alerts_out.append({
                    "cluster": cid,
                    "id": h,
                    "idx": cidx.host_to_idx[h],
                    **ly,
                    "horizon": worst_hz,
                    "horizons_triggered": [hz for hz, _ in triggered],
                    "risk": worst_fc["risk"],
                    "point": worst_fc["point"],
                    "lower": worst_fc["lower"],
                    "upper": worst_fc["upper"],
                    "temp_eff": st["temp_eff"],
                    "threshold_warning": _r2(cidx.warn_thresh),
                    "threshold_critical": _r2(cidx.crit_thresh),
                })

        alerts_out.sort(
            key=lambda a: (1 if a["risk"] == "critical" else 0, a.get("upper") or 0.0),
            reverse=True,
        )
        return {
            "cluster": cluster,
            "time": _to_iso(t_ts),
            "count": len(alerts_out),
            "alerts": alerts_out,
        }

    def whatif(self, req: WhatIfRequest) -> dict:
        cidx = self.ensure_cluster(req.cluster)
        eng = cidx.engine

        # Resolve host capacity or cluster median defaults
        cores = cidx.default_cores
        ram_gb = cidx.default_ram_gb
        if req.host:
            if req.host not in cidx.statics:
                raise HTTPException(status_code=404, detail=f"Host '{req.host}' not found in '{req.cluster}'")
            st = cidx.statics[req.host]
            cores = float(st.cores)
            ram_gb = float(st.ram_gb)
        if req.capacity:
            if req.capacity.cores is not None:
                cores = float(req.capacity.cores)
            if req.capacity.ram_gb is not None:
                ram_gb = float(req.capacity.ram_gb)

        w = req.workload
        hr = 12 if req.hour is None else req.hour
        wd = 2 if req.weekday is None else req.weekday
        hour_sin = math.sin(2.0 * math.pi * hr / 24.0)
        hour_cos = math.cos(2.0 * math.pi * hr / 24.0)
        dow_sin = math.sin(2.0 * math.pi * wd / 7.0)
        dow_cos = math.cos(2.0 * math.pi * wd / 7.0)

        def _build_wl(cpu_load_val: float) -> dict:
            return {
                "CPU_Load": cpu_load_val,
                "CPU_cores": cores,
                "CPU_cores_used": w.cores_used,
                "Ram": ram_gb,
                "Ram_Used": w.ram_used,
                "No_Of_Running_vms": w.vms,
                "Network_RX": w.net_rx,
                "Network_TX": w.net_tx,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "dow_sin": dow_sin,
                "dow_cos": dow_cos,
                "CPU_Load_rmean_6": cpu_load_val,
                "CPU_Load_rmean_36": cpu_load_val,
                "Ram_Used_rmean_6": w.ram_used,
                "Ram_Used_rmean_36": w.ram_used,
                "No_Of_Running_vms_rmean_6": w.vms,
                "No_Of_Running_vms_rmean_36": w.vms,
                "Network_RX_rmean_6": w.net_rx,
                "Network_RX_rmean_36": w.net_rx,
                "Network_TX_rmean_6": w.net_tx,
                "Network_TX_rmean_36": w.net_tx,
            }

        cap = {"CPU_cores": cores, "Ram": ram_gb}
        res = eng.whatif(_build_wl(w.cpu_load), cap)
        if not res.get("available", False):
            raise HTTPException(
                status_code=503,
                detail=res.get("error", f"Planning model not available for '{req.cluster}'"),
            )

        plan_info = cidx.models_info.get("planning_0", {})
        model_id = f"{req.cluster}_planning_0 ({str(cidx.meta_dict.get('git_commit', ''))[:7]})".strip()
        test_mae = _r2(plan_info.get("test_mae"))
        cov_raw = plan_info.get("empirical_coverage")
        coverage = _r2(cov_raw * 100.0) if cov_raw is not None else None
        mean_w = _r2(plan_info.get("mean_interval_width"))

        curve: List[dict] = []
        if req.include_curve:
            loads = [float(v) for v in range(0, 101, 10)]
            models_dict = getattr(eng, "_models", None)
            if isinstance(models_dict, dict) and plan_info.get("feature_names"):
                feat_names = plan_info["feature_names"]
                q_hat = float(plan_info.get("q_hat", 0.0))
                X = np.array(
                    [[_build_wl(lv).get(f, 0.0) for f in feat_names] for lv in loads],
                    dtype=np.float64,
                )
                m_pt = models_dict.get(f"{req.cluster}_planning_0_point")
                m_lo = models_dict.get(f"{req.cluster}_planning_0_lower")
                m_hi = models_dict.get(f"{req.cluster}_planning_0_upper")
                if m_pt is None:
                    raise HTTPException(status_code=503, detail="Planning point model missing")
                pts = m_pt.predict(X)
                los = m_lo.predict(X) - q_hat if m_lo is not None else pts
                his = m_hi.predict(X) + q_hat if m_hi is not None else pts
                for i, lv in enumerate(loads):
                    lo_v, hi_v = float(los[i]), float(his[i])
                    if lo_v > hi_v:
                        lo_v, hi_v = hi_v, lo_v
                    curve.append({
                        "cpu_load": lv,
                        "point": _r2(pts[i]),
                        "lower": _r2(lo_v),
                        "upper": _r2(hi_v),
                    })
            else:
                for lv in loads:
                    cr = eng.whatif(_build_wl(lv), cap)
                    if cr.get("available"):
                        curve.append({
                            "cpu_load": lv,
                            "point": _r2(cr["point"]),
                            "lower": _r2(cr["lower"]),
                            "upper": _r2(cr["upper"]),
                        })

        pt = _r2(res["point"])
        lo = _r2(res["lower"])
        hi = _r2(res["upper"])
        width = _r2(hi - lo) if pt is not None and lo is not None and hi is not None else None

        return {
            "cluster": req.cluster,
            "host": req.host,
            "capacity": {"cores": round(cores, 2), "ram_gb": round(ram_gb, 2)},
            "workload": w.model_dump(),
            "point": pt,
            "lower": lo,
            "upper": hi,
            "width": width,
            "risk": str(res.get("risk", "unknown")),
            "model_id": model_id,
            "test_mae": test_mae,
            "coverage": coverage,
            "mean_interval_width": mean_w,
            "note": "predictive, correlational - not causal",
            "sensitivity_curve": curve,
        }


# ── App factory ──────────────────────────────────────────────────────


def create_app(
    service: Optional[Twin3DService] = None,
    web_dir: Optional[Path] = None,
) -> FastAPI:
    svc = service or Twin3DService()
    static_dir = Path(web_dir) if web_dir else _DEFAULT_WEB_DIR

    fastapi_app = FastAPI(title="Predictive Digital Twin 3D API", version="2.1.0")

    @fastapi_app.get("/api/health")
    def api_health(cluster: Optional[str] = Query(default=None)):
        return svc.health(cluster=cluster)

    @fastapi_app.get("/api/meta")
    def api_meta():
        return svc.meta()

    @fastapi_app.get("/api/snapshot")
    def api_snapshot(
        cluster: str = Query(...),
        t: str = Query(...),
        horizon: int = Query(default=1),
    ):
        return svc.snapshot(cluster=cluster, t_str=t, horizon=horizon)

    @fastapi_app.get("/api/compare")
    def api_compare(
        cluster: str = Query(...),
        t: str = Query(...),
        horizon: int = Query(default=1),
    ):
        return svc.compare(cluster=cluster, t_str=t, horizon=horizon)

    @fastapi_app.get("/api/series")
    def api_series(
        cluster: str = Query(...),
        host: str = Query(...),
        start: Optional[str] = Query(default=None),
        end: Optional[str] = Query(default=None),
        horizon: int = Query(default=1),
    ):
        return svc.series(cluster=cluster, host=host, start_str=start, end_str=end, horizon=horizon)

    @fastapi_app.get("/api/alerts")
    def api_alerts(
        cluster: str = Query(default="all"),
        t: str = Query(...),
    ):
        return svc.alerts(cluster=cluster, t_str=t)

    @fastapi_app.post("/api/whatif")
    def api_whatif(req: WhatIfRequest):
        return svc.whatif(req)

    # Conditionally expose /api/simulate only if engine implements simulate
    @fastapi_app.post("/api/simulate")
    def api_simulate(payload: dict):
        cluster = payload.get("cluster", "d1")
        cidx = svc.ensure_cluster(cluster)
        if not hasattr(cidx.engine, "simulate"):
            raise HTTPException(status_code=404, detail="Simulator not enabled on engine")
        return cidx.engine.simulate(**{k: v for k, v in payload.items() if k != "cluster"})

    if static_dir.exists():
        fastapi_app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="twin3d_web")

    return fastapi_app


app = create_app()
