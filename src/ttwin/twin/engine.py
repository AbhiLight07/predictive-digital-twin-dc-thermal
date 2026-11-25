"""TwinEngine — offline predictive digital twin engine.

Loads trained LightGBM models and interim data to produce HostState,
HostForecast, and fleet snapshots. Uses only data at or before the query time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from ttwin.twin.schema import (
    DataQuality,
    HostForecast,
    HostStatic,
    HostState,
    RiskLevel,
    SensorReading,
)
from ttwin.clean import fill_missing_timestamps
from ttwin.features import (
    compute_effective_cpu_temp,
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
)

_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # repo root

# Signals that exist per-row in interim data
_ALL_SIGNALS = [
    "CPU_Load", "Ram", "Ram_Used", "CPU_cores", "CPU_cores_used",
    "Network_RX", "Network_TX", "No_Of_Running_vms", "Power",
    "Fan_speed1", "Fan_speed2", "Fan_speed3", "Fan_speed4",
    "Inlet_Temperature", "CPU1_Temperature", "CPU2_Temperature",
]

_STALE_MINUTES = 30

# Planning rolling columns and windows (must match train_twin_models.py)
_PLAN_ROLLING_COLS = ["CPU_Load", "Ram_Used", "No_Of_Running_vms", "Network_RX", "Network_TX"]
_PLAN_ROLLING_WINDOWS = [6, 36]


class TwinEngine:
    """Offline predictive digital twin engine for one cluster."""

    def __init__(
        self,
        cluster: str,
        models_dir: str | Path | None = None,
        config_path: str | Path | None = None,
        data_dir: str | Path | None = None,
        planning_cal_mode: str = "static",
    ):
        self.cluster = cluster  # "d1" or "d2"
        self._dataset = "dataset1" if cluster == "d1" else "dataset2"
        self._planning_cal_mode = planning_cal_mode

        models_dir = Path(models_dir) if models_dir else _ROOT / "models" / "twin"
        config_path = Path(config_path) if config_path else _ROOT / "configs" / "twin.yaml"
        data_dir = Path(data_dir) if data_dir else _ROOT / "data" / "interim" / self._dataset

        # Load config
        with open(config_path) as f:
            self._config = yaml.safe_load(f)

        thresholds = self._config["risk_thresholds"][self._dataset]
        self._warn_thresh = thresholds["warning_celsius"]
        self._crit_thresh = thresholds["critical_celsius"]
        self._horizons = self._config["horizons"]["steps"]  # [1, 6, 36]

        # Load metadata
        meta_path = models_dir / f"{cluster}_metadata.json"
        with open(meta_path) as f:
            self._meta = json.load(f)

        # Optional sidecar for adaptive planning calibration (leaves .txt models untouched)
        self._planning_calibration: Optional[dict] = None
        cal_path = models_dir / "planning_calibration.json"
        if cal_path.exists():
            with open(cal_path, encoding="utf-8") as f:
                cal_all = json.load(f)
            self._planning_calibration = cal_all.get("clusters", {}).get(cluster)

        self._host_list: List[str] = self._meta["host_list"]
        self._models_info: Dict[str, dict] = {}
        for m in self._meta["models"]:
            key = f"{m['kind']}_{m['horizon']}"
            self._models_info[key] = m

        # Load LightGBM models
        self._models: Dict[str, lgb.Booster] = {}
        for kind in ["monitoring", "planning"]:
            horizons_to_load = self._horizons if kind == "monitoring" else [0]
            for h in horizons_to_load:
                for role in ["point", "lower", "upper"]:
                    name = f"{cluster}_{kind}_{h}_{role}"
                    path = models_dir / f"{name}.txt"
                    if path.exists():
                        self._models[name] = lgb.Booster(model_file=str(path))

        # Load and prepare interim data
        self._data: pd.DataFrame = self._load_data(data_dir)

        # Cluster sensor availability
        self._has_fan = cluster == "d1"
        self._has_inlet = cluster == "d1"

        # Columns config
        cols_path = _ROOT / "configs" / "columns.yaml"
        self._cols_config = {}
        if cols_path.exists():
            with open(cols_path) as f:
                self._cols_config = yaml.safe_load(f)

        # Servers without fan/inlet in D1
        self._no_fan_servers = set()
        if cluster == "d1" and self._dataset in self._cols_config:
            self._no_fan_servers = set(
                self._cols_config[self._dataset].get("servers_no_inlet_fan", [])
            )

    def _load_data(self, data_dir: Path) -> pd.DataFrame:
        """Load all interim parquets, filter to host_list, sort."""
        frames = []
        for f in sorted(data_dir.glob("*.parquet")):
            if f.name.startswith("_"):
                continue
            df = pd.read_parquet(f)
            frames.append(df)
        if not frames:
            raise FileNotFoundError(f"No parquet files in {data_dir}")
        df = pd.concat(frames, ignore_index=True)
        df["Series"] = df["Series"].str.strip()
        df = df[df["Series"].isin(self._host_list)].copy()
        df["Time"] = pd.to_datetime(df["Time"])
        df = df.sort_values(["Series", "Time"]).reset_index(drop=True)
        df = compute_effective_cpu_temp(df)
        return df

    # ── Public API ────────────────────────────────────────────────────

    def hosts(self) -> List[str]:
        return list(self._host_list)

    def static(self, host: str) -> HostStatic:
        hdata = self._data[self._data["Series"] == host]
        cores = int(hdata["CPU_cores"].dropna().iloc[0]) if len(hdata) > 0 and "CPU_cores" in hdata.columns else 0
        ram = float(hdata["Ram"].dropna().iloc[0]) if len(hdata) > 0 and "Ram" in hdata.columns else 0.0
        has_fan = self._has_fan and host not in self._no_fan_servers
        has_inlet = self._has_inlet and host not in self._no_fan_servers
        return HostStatic(
            cluster=self.cluster.upper(),
            host_id=host,
            cores=cores,
            ram_gb=ram,
            hardware_model="as reported in dataset paper (Ilager et al., UCC 2023)",
            has_fan=has_fan,
            has_inlet=has_inlet,
        )

    def state(self, host: str, t: pd.Timestamp) -> HostState:
        """Return HostState at the latest observation at or before t."""
        hdata = self._data[self._data["Series"] == host]
        past = hdata[hdata["Time"] <= t]

        if len(past) == 0:
            return self._missing_state(host, t)

        row = past.iloc[-1]
        obs_time = row["Time"]
        gap_min = (t - obs_time).total_seconds() / 60.0
        base_quality = DataQuality.STALE if gap_min > _STALE_MINUTES else DataQuality.OK

        has_fan = self._has_fan and host not in self._no_fan_servers
        has_inlet = self._has_inlet and host not in self._no_fan_servers

        def _sr(col, available=True):
            if not available or col not in row.index:
                return SensorReading(value=None, available=False, quality=DataQuality.MISSING)
            val = row[col]
            if pd.isna(val):
                return SensorReading(value=None, available=True, quality=DataQuality.MISSING)
            return SensorReading(value=float(val), available=True, quality=base_quality)

        cpu_temp_val = None
        if not pd.isna(row.get("CPU_Temperature", np.nan)):
            cpu_temp_val = float(row["CPU_Temperature"])

        return HostState(
            cluster=self.cluster.upper(),
            host_id=host,
            timestamp=str(obs_time),
            cpu_load=_sr("CPU_Load"),
            ram=_sr("Ram"),
            ram_used=_sr("Ram_Used"),
            cpu_cores=_sr("CPU_cores"),
            cpu_cores_used=_sr("CPU_cores_used"),
            network_rx=_sr("Network_RX"),
            network_tx=_sr("Network_TX"),
            no_of_running_vms=_sr("No_Of_Running_vms"),
            power=_sr("Power"),
            fan_speed1=_sr("Fan_speed1", has_fan),
            fan_speed2=_sr("Fan_speed2", has_fan),
            fan_speed3=_sr("Fan_speed3", has_fan),
            fan_speed4=_sr("Fan_speed4", has_fan),
            inlet_temperature=_sr("Inlet_Temperature", has_inlet),
            cpu1_temperature=_sr("CPU1_Temperature"),
            cpu2_temperature=_sr("CPU2_Temperature"),
            cpu_temperature=SensorReading(
                value=cpu_temp_val,
                available=True,
                quality=base_quality if cpu_temp_val is not None else DataQuality.MISSING,
            ),
        )

    def _missing_state(self, host: str, t: pd.Timestamp) -> HostState:
        missing = SensorReading(value=None, available=False, quality=DataQuality.MISSING)
        return HostState(
            cluster=self.cluster.upper(), host_id=host, timestamp=str(t),
            cpu_load=missing, ram=missing, ram_used=missing,
            cpu_cores=missing, cpu_cores_used=missing,
            network_rx=missing, network_tx=missing,
            no_of_running_vms=missing, power=missing,
            fan_speed1=missing, fan_speed2=missing,
            fan_speed3=missing, fan_speed4=missing,
            inlet_temperature=missing,
            cpu1_temperature=missing, cpu2_temperature=missing,
            cpu_temperature=missing,
        )

    def history(self, host: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        hdata = self._data[self._data["Series"] == host]
        return hdata[(hdata["Time"] >= start) & (hdata["Time"] <= end)].copy()

    def partition(self, t: pd.Timestamp) -> str:
        """Return train/val_sel/val_cal/test for a given timestamp."""
        # Use monitoring_1 boundaries as reference (all share the same split)
        b = self._models_info["monitoring_1"]["boundaries"]
        ts = str(t)
        if ts <= b["train_end"]:
            return "TRAIN"
        elif ts <= b["val_sel_end"]:
            return "VAL_SEL"
        elif ts <= b["val_cal_end"]:
            return "VAL_CAL"
        else:
            return "TEST"

    def forecast(self, host: str, t: pd.Timestamp) -> List[HostForecast]:
        """Produce forecasts at 10m, 1h, 6h horizons using only data at or before t."""
        hdata = self._data[self._data["Series"] == host]
        past = hdata[hdata["Time"] <= t].copy()

        if len(past) < 2:
            return [self._unknown_forecast(host, t, h) for h in self._horizons]

        gap_min = (t - past["Time"].max()).total_seconds() / 60.0
        if gap_min > _STALE_MINUTES:
            return [self._unknown_forecast(host, t, h) for h in self._horizons]

        # Build monitoring features on past data only (aligned to 10-min grid)
        feat_df = self._build_monitoring_features(past)
        if len(feat_df) == 0:
            return [self._unknown_forecast(host, t, h) for h in self._horizons]

        last_row = feat_df.iloc[-1]
        current_temp = last_row.get("CPU_Temperature", np.nan)
        if pd.isna(current_temp):
            return [self._unknown_forecast(host, t, h) for h in self._horizons]

        part = self.partition(t)
        forecasts = []

        for h in self._horizons:
            info = self._models_info.get(f"monitoring_{h}")
            if info is None:
                forecasts.append(self._unknown_forecast(host, t, h))
                continue

            feat_names = info["feature_names"]
            # Check all features exist and are non-NaN (e.g., after a gap)
            missing_feats = [f for f in feat_names if f not in last_row.index]
            if missing_feats or pd.isna(last_row[feat_names]).any():
                forecasts.append(self._unknown_forecast(host, t, h))
                continue

            X = last_row[feat_names].values.reshape(1, -1).astype(np.float64)

            m_point = self._models.get(f"{self.cluster}_monitoring_{h}_point")
            m_lower = self._models.get(f"{self.cluster}_monitoring_{h}_lower")
            m_upper = self._models.get(f"{self.cluster}_monitoring_{h}_upper")

            if m_point is None:
                forecasts.append(self._unknown_forecast(host, t, h))
                continue

            delta_point = float(m_point.predict(X)[0])
            point_abs = current_temp + delta_point
            persistence = current_temp

            q_hat = info.get("q_hat", 0.0)

            if m_lower and m_upper:
                delta_lo = float(m_lower.predict(X)[0])
                delta_hi = float(m_upper.predict(X)[0])
                lower_abs = current_temp + delta_lo - q_hat
                upper_abs = current_temp + delta_hi + q_hat
            else:
                lower_abs = point_abs - abs(q_hat)
                upper_abs = point_abs + abs(q_hat)

            # Ensure ordering
            if lower_abs > upper_abs:
                lower_abs, upper_abs = upper_abs, lower_abs

            risk = self._assess_risk(upper_abs)

            forecasts.append(HostForecast(
                cluster=self.cluster.upper(),
                host_id=host,
                timestamp=str(t),
                horizon_steps=h,
                point=round(point_abs, 4),
                lower=round(lower_abs, 4),
                upper=round(upper_abs, 4),
                persistence=round(persistence, 4),
                risk=risk,
                model_version=self._meta.get("git_commit", ""),
                partition=part,
            ))

        return forecasts

    def whatif(
        self,
        workload: dict,
        capacity: Optional[dict] = None,
        t: Optional[pd.Timestamp] = None,
        cal_mode: Optional[str] = None,
    ) -> dict:
        """Planning-model prediction: workload → temperature estimate + CQR interval."""
        info = self._models_info.get("planning_0")
        if info is None:
            return {"available": False, "error": "planning model not loaded"}

        feat_names = info["feature_names"]
        q_hat = float(info.get("q_hat", 0.0))
        mode = cal_mode or self._planning_cal_mode
        if mode == "rolling_7d" and self._planning_calibration:
            daily = self._planning_calibration.get("rolling_7d_daily_q_hats", {})
            if t is not None and daily:
                day_str = pd.Timestamp(t).strftime("%Y-%m-%d")
                if day_str in daily:
                    q_hat = float(daily[day_str])
                else:
                    prior_days = [d for d in sorted(daily) if d <= day_str]
                    q_hat = float(daily[prior_days[-1]]) if prior_days else float(
                        self._planning_calibration.get("rolling_7d_q_hat_latest", q_hat)
                    )
            else:
                q_hat = float(self._planning_calibration.get("rolling_7d_q_hat_latest", q_hat))

        # Build feature vector from workload dict
        row = {}
        for f in feat_names:
            if f in workload:
                row[f] = workload[f]
            elif capacity and f in capacity:
                row[f] = capacity[f]
            else:
                row[f] = 0.0

        X = np.array([[row[f] for f in feat_names]], dtype=np.float64)

        m_pt = self._models.get(f"{self.cluster}_planning_0_point")
        m_lo = self._models.get(f"{self.cluster}_planning_0_lower")
        m_hi = self._models.get(f"{self.cluster}_planning_0_upper")

        if m_pt is None:
            return {"available": False, "error": "planning point model not found"}

        point = float(m_pt.predict(X)[0])
        lower = float(m_lo.predict(X)[0]) - q_hat if m_lo else point
        upper = float(m_hi.predict(X)[0]) + q_hat if m_hi else point

        if lower > upper:
            lower, upper = upper, lower

        return {
            "available": True,
            "point": round(point, 4),
            "lower": round(lower, 4),
            "upper": round(upper, 4),
            "risk": self._assess_risk(upper).value,
        }

    def fleet_snapshot(self, t: pd.Timestamp) -> pd.DataFrame:
        """One row per host with state summary and latest forecasts."""
        rows = []
        for host in self._host_list:
            st = self.state(host, t)
            fcs = self.forecast(host, t)
            row = {
                "host": host,
                "time": t,
                "cpu_temperature": st.cpu_temperature.value,
                "cpu_load": st.cpu_load.value,
                "quality": st.cpu_temperature.quality.value,
            }
            for fc in fcs:
                h = fc.horizon_steps
                row[f"point_{h}"] = fc.point
                row[f"lower_{h}"] = fc.lower
                row[f"upper_{h}"] = fc.upper
                row[f"risk_{h}"] = fc.risk.value
            rows.append(row)
        return pd.DataFrame(rows)

    # ── Vectorised forecast for cache building ────────────────────────

    def forecast_vectorised(self, host: str) -> pd.DataFrame:
        """Compute forecasts for all test-partition timestamps using causal features.
        Returns a DataFrame with columns: host, time, horizon, point, lower, upper,
        persistence, risk, partition, actual_future.
        """
        hdata = self._data[self._data["Series"] == host].copy()
        if len(hdata) < 2:
            return pd.DataFrame()

        # Build features on full series (causal by construction)
        feat_df = self._build_monitoring_features(hdata)

        # Identify test partition rows
        b = self._models_info["monitoring_1"]["boundaries"]
        test_start = pd.Timestamp(b["test_start"])
        test_mask = feat_df["Time"] >= test_start

        stride_steps = self._config["replay"]["stride_minutes"] // 10
        test_idx = feat_df.index[test_mask]
        if stride_steps > 1:
            test_idx = test_idx[::stride_steps]

        if len(test_idx) == 0:
            return pd.DataFrame()

        rows = []
        for h in self._horizons:
            info = self._models_info.get(f"monitoring_{h}")
            if info is None:
                continue

            feat_names = info["feature_names"]
            q_hat = info.get("q_hat", 0.0)

            # Filter to rows with all features available
            needed = list(dict.fromkeys(feat_names + ["CPU_Temperature", "Time"]))
            avail = feat_df.loc[test_idx, needed].dropna()
            if len(avail) == 0:
                continue

            X = avail[feat_names].values.astype(np.float64)
            current_temp = avail["CPU_Temperature"].values
            times = avail["Time"].values

            m_pt = self._models.get(f"{self.cluster}_monitoring_{h}_point")
            m_lo = self._models.get(f"{self.cluster}_monitoring_{h}_lower")
            m_hi = self._models.get(f"{self.cluster}_monitoring_{h}_upper")

            if m_pt is None:
                continue

            delta_pt = m_pt.predict(X)
            point_abs = current_temp + delta_pt
            persistence = current_temp.copy()

            if m_lo and m_hi:
                delta_lo = m_lo.predict(X)
                delta_hi = m_hi.predict(X)
                lower_abs = current_temp + delta_lo - q_hat
                upper_abs = current_temp + delta_hi + q_hat
            else:
                lower_abs = point_abs - abs(q_hat)
                upper_abs = point_abs + abs(q_hat)

            # Ensure ordering
            swap = lower_abs > upper_abs
            lower_abs[swap], upper_abs[swap] = upper_abs[swap], lower_abs[swap]

            # Risk levels
            risk_vals = np.where(
                upper_abs >= self._crit_thresh, "critical",
                np.where(upper_abs >= self._warn_thresh, "warning", "normal")
            )

            # Actual future temperature (O(1) dict lookup per timestamp)
            horizon_td = pd.Timedelta(minutes=h * 10)
            temp_by_time = dict(zip(hdata["Time"], hdata["CPU_Temperature"]))
            actual_future = np.array(
                [temp_by_time.get(pd.Timestamp(t_val) + horizon_td, np.nan) for t_val in times],
                dtype=np.float64,
            )

            for i in range(len(avail)):
                rows.append({
                    "host": host,
                    "time": pd.Timestamp(times[i]),
                    "horizon": h,
                    "point": round(float(point_abs[i]), 4),
                    "lower": round(float(lower_abs[i]), 4),
                    "upper": round(float(upper_abs[i]), 4),
                    "persistence": round(float(persistence[i]), 4),
                    "risk": risk_vals[i],
                    "partition": "TEST",
                    "actual_future": actual_future[i] if not np.isnan(actual_future[i]) else None,
                })

        return pd.DataFrame(rows)

    # ── Internal helpers ──────────────────────────────────────────────

    def _build_monitoring_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build S1 monitoring features on a regular 10-min grid (matching training)."""
        df = fill_missing_timestamps(df, host_col="Series", time_col="Time", freq="10min")
        df = compute_effective_cpu_temp(df)
        df = add_calendar_features(df, "Time")
        temp_cols = [c for c in ["CPU_Temperature", "CPU1_Temperature", "CPU2_Temperature"] if c in df.columns]
        df = add_lag_features(df, temp_cols, [1, 2, 3, 6, 12], "Series")
        df = add_rolling_features(df, temp_cols, [6, 12, 36], "Series")
        return df

    def _assess_risk(self, upper_bound: float) -> RiskLevel:
        if pd.isna(upper_bound):
            return RiskLevel.UNKNOWN
        if upper_bound >= self._crit_thresh:
            return RiskLevel.CRITICAL
        if upper_bound >= self._warn_thresh:
            return RiskLevel.WARNING
        return RiskLevel.NORMAL

    def _unknown_forecast(self, host: str, t: pd.Timestamp, h: int) -> HostForecast:
        return HostForecast(
            cluster=self.cluster.upper(), host_id=host, timestamp=str(t),
            horizon_steps=h, point=np.nan, lower=np.nan, upper=np.nan,
            persistence=np.nan, risk=RiskLevel.UNKNOWN,
        )
