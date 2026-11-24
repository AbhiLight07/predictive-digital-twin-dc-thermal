"""twin3d API server — Flask app serving the 3D digital twin.

Endpoints:
  GET /api/meta          — legend, time range, cluster info
  GET /api/snapshot      — per-host telemetry + forecast at time t
  GET /api/compare       — twin vs reality at target time t for horizon h
  GET /api/whatif        — planning model what-if
  GET /api/history/<id>  — time-series for one host
  GET /                  — serves the 3D web UI

All floats rounded to 2 dp.  Times in UTC ISO-8601.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

# Repo root — twin3d/api/server.py -> repo root is ../../
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

from ttwin.twin.engine import TwinEngine  # noqa: E402
from ttwin.twin.schema import RiskLevel, DataQuality  # noqa: E402

app = Flask(__name__, static_folder=None)

# ── Lazy engine loading ──────────────────────────────────────────────

_engines: dict[str, TwinEngine] = {}


def _get_engine(cluster: str) -> TwinEngine:
    if cluster not in _engines:
        _engines[cluster] = TwinEngine(cluster)
    return _engines[cluster]


def _get_engines() -> dict[str, TwinEngine]:
    for c in ("d1", "d2"):
        _get_engine(c)
    return _engines


# ── Cache-backed forecast lookup ─────────────────────────────────────

_cache: dict[str, pd.DataFrame] = {}


def _load_cache(cluster: str) -> pd.DataFrame | None:
    if cluster in _cache:
        return _cache[cluster]
    cache_path = _REPO / "data" / "twin_cache" / f"{cluster}_forecasts.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        df["time"] = pd.to_datetime(df["time"])
        _cache[cluster] = df
        return df
    return None


# ── Layout helpers ───────────────────────────────────────────────────

_RACK_SIZE = 20
_RACK_PITCH_X = 0.9
_SLAB_W, _SLAB_H, _SLAB_D = 0.48, 0.0845, 0.72
_RACK_W, _RACK_D, _RACK_H = 0.6, 1.0, 1.87
_CLUSTER_Z = {"d1": 0.0, "d2": -4.0}


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


# ── Legend / meta helpers ────────────────────────────────────────────

def _compute_legend(engine: TwinEngine) -> dict:
    data = engine._data
    t_lo = float(np.nanpercentile(data["CPU_Temperature"].dropna(), 5))
    t_hi = float(np.nanpercentile(data["CPU_Temperature"].dropna(), 99.5))
    p_lo = float(np.nanpercentile(data["Power"].dropna(), 5))
    p_hi = float(np.nanpercentile(data["Power"].dropna(), 99.5))

    legend = {
        "t_lo": round(t_lo, 2),
        "t_hi": round(t_hi, 2),
        "p_lo": round(p_lo, 2),
        "p_hi": round(p_hi, 2),
        "w_ref": None,
        "rpm_ref": None,
    }

    # w_ref from cache if available
    cache_df = _load_cache(engine.cluster)
    if cache_df is not None:
        widths = cache_df["upper"] - cache_df["lower"]
        legend["w_ref"] = round(float(np.nanpercentile(widths.dropna(), 95)), 2)

    # rpm_ref for d1
    if engine._has_fan:
        fan_cols = [c for c in ["Fan_speed1", "Fan_speed2", "Fan_speed3", "Fan_speed4"]
                    if c in data.columns]
        if fan_cols:
            all_fans = pd.concat([data[c].dropna() for c in fan_cols])
            legend["rpm_ref"] = round(float(np.nanpercentile(all_fans, 95)), 2)

    return legend


def _time_range(engine: TwinEngine) -> dict:
    times = engine._data["Time"]
    return {
        "min": str(times.min()),
        "max": str(times.max()),
        "step_minutes": 10,
    }


# ── Helper: build one host snapshot dict ──────────────────────────────

def _r2(v):
    """Round to 2 dp, or None."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), 2)


def _host_snapshot(engine: TwinEngine, host: str, idx: int,
                   t: pd.Timestamp, horizon: int,
                   cache_df: pd.DataFrame | None) -> dict:
    st = engine.state(host, t)
    pos = _layout(engine.cluster, idx)
    part = engine.partition(t)
    has_fan = engine._has_fan and host not in engine._no_fan_servers
    has_inlet = engine._has_inlet and host not in engine._no_fan_servers

    status = "ok"
    if st.cpu_temperature.quality == DataQuality.MISSING:
        status = "missing"
    elif st.cpu_temperature.quality == DataQuality.STALE:
        status = "stale"

    obj: dict = {
        "idx": idx,
        "id": host,
        "status": status,
        "partition": part,
        "temp_eff": _r2(st.cpu_temperature.value),
        "cpu1": _r2(st.cpu1_temperature.value),
        "cpu2": _r2(st.cpu2_temperature.value),
        "inlet": _r2(st.inlet_temperature.value) if has_inlet else None,
        "inlet_available": has_inlet,
        "fans": (
            [_r2(getattr(st, f"fan_speed{i}").value) for i in range(1, 5)]
            if has_fan else None
        ),
        "fans_available": has_fan,
        "power": _r2(st.power.value),
        "load": _r2(st.cpu_load.value),
        "ram_used_pct": _r2(
            (st.ram_used.value / st.ram.value * 100)
            if st.ram_used.value is not None and st.ram.value and st.ram.value > 0
            else None
        ),
        "cores_used": _r2(st.cpu_cores_used.value),
        "vms": _r2(st.no_of_running_vms.value),
        "net_rx": _r2(st.network_rx.value),
        "net_tx": _r2(st.network_tx.value),
        "forecast": None,
        **pos,
    }

    # Attach forecast from cache
    if cache_df is not None and horizon > 0:
        fc_rows = cache_df[
            (cache_df["host"] == host) &
            (cache_df["time"] == t) &
            (cache_df["horizon"] == horizon)
        ]
        if len(fc_rows) > 0:
            fc = fc_rows.iloc[0]
            obj["forecast"] = {
                "horizon": int(fc["horizon"]),
                "point": _r2(fc["point"]),
                "lower": _r2(fc["lower"]),
                "upper": _r2(fc["upper"]),
                "persistence": _r2(fc["persistence"]),
                "risk": fc["risk"] if isinstance(fc["risk"], str) else str(fc["risk"]),
            }
    elif horizon == 0:
        # No forecast at h=0, just current state
        pass

    return obj


# ── API Endpoints ────────────────────────────────────────────────────

@app.route("/api/meta")
def api_meta():
    engines = _get_engines()
    clusters = {}
    for cid, eng in engines.items():
        clusters[cid] = {
            "hosts": eng.hosts(),
            "host_count": len(eng.hosts()),
            "legend": _compute_legend(eng),
            "time_range": _time_range(eng),
            "has_fan": eng._has_fan,
            "has_inlet": eng._has_inlet,
            "thresholds": {
                "warning": eng._warn_thresh,
                "critical": eng._crit_thresh,
            },
            "horizons": eng._horizons,
            "layout": {
                "rack_size": _RACK_SIZE,
                "rack_pitch_x": _RACK_PITCH_X,
                "slab": {"w": _SLAB_W, "h": _SLAB_H, "d": _SLAB_D},
                "rack": {"w": _RACK_W, "d": _RACK_D, "h": _RACK_H},
                "cluster_z": _CLUSTER_Z[cid],
            },
        }
    return jsonify({
        "clusters": clusters,
        "banner": "Offline replay of recorded telemetry - schematic layout, not physical rack positions - thresholds illustrative",
        "horizons_steps": [1, 6, 36],
        "horizons_minutes": [10, 60, 360],
        "colours": {
            "temperature_ramp": [
                {"t": 0.00, "hex": "#440154"},
                {"t": 0.25, "hex": "#3b528b"},
                {"t": 0.50, "hex": "#21918c"},
                {"t": 0.75, "hex": "#5ec962"},
                {"t": 1.00, "hex": "#fde725"},
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
            },
        },
    })


@app.route("/api/snapshot")
def api_snapshot():
    t_str = request.args.get("t")
    cluster = request.args.get("cluster")
    horizon = int(request.args.get("horizon", 1))

    if not t_str or not cluster:
        return jsonify({"error": "t and cluster required"}), 400
    if cluster not in ("d1", "d2"):
        return jsonify({"error": "cluster must be d1 or d2"}), 400

    try:
        engine = _get_engine(cluster)
    except Exception as e:
        return jsonify({"error": str(e)}), 503

    t = pd.Timestamp(t_str)
    cache_df = _load_cache(cluster)
    hosts_list = engine.hosts()

    snapshot = []
    for idx, host in enumerate(hosts_list):
        snapshot.append(_host_snapshot(engine, host, idx, t, horizon, cache_df))

    return jsonify({
        "cluster": cluster,
        "time": str(t),
        "horizon": horizon,
        "hosts": snapshot,
    })


@app.route("/api/compare")
def api_compare():
    """Compare mode: twin forecast vs reality at target_time for horizon h."""
    t_str = request.args.get("t")  # target time
    cluster = request.args.get("cluster")
    horizon = int(request.args.get("horizon", 1))

    if not t_str or not cluster:
        return jsonify({"error": "t and cluster required"}), 400
    if cluster not in ("d1", "d2"):
        return jsonify({"error": "cluster must be d1 or d2"}), 400

    try:
        engine = _get_engine(cluster)
    except Exception as e:
        return jsonify({"error": str(e)}), 503

    target_t = pd.Timestamp(t_str)
    issue_t = target_t - pd.Timedelta(minutes=horizon * 10)

    cache_df = _load_cache(cluster)
    hosts_list = engine.hosts()

    results = []
    n_inside = 0
    n_valid = 0
    errors = []

    for idx, host in enumerate(hosts_list):
        # Reality at target time
        reality_state = engine.state(host, target_t)
        reality_temp = reality_state.cpu_temperature.value

        # Twin: forecast made at issue_time for this horizon
        twin_point = None
        twin_lower = None
        twin_upper = None
        twin_risk = "unknown"

        if cache_df is not None:
            fc_rows = cache_df[
                (cache_df["host"] == host) &
                (cache_df["time"] == issue_t) &
                (cache_df["horizon"] == horizon)
            ]
            if len(fc_rows) > 0:
                fc = fc_rows.iloc[0]
                twin_point = _r2(fc["point"])
                twin_lower = _r2(fc["lower"])
                twin_upper = _r2(fc["upper"])
                twin_risk = fc["risk"] if isinstance(fc["risk"], str) else str(fc["risk"])

        error = None
        inside = None
        status = "ok"

        if reality_temp is None:
            status = "missing"
        elif twin_point is None:
            status = "unknown"
        else:
            error = round(twin_point - reality_temp, 2)
            errors.append(error)
            inside = (twin_lower is not None and twin_upper is not None and
                      twin_lower <= reality_temp <= twin_upper)
            if inside:
                n_inside += 1
            n_valid += 1

        issue_part = engine.partition(issue_t)
        in_sample = issue_part in ("TRAIN", "VAL_SEL", "VAL_CAL")

        pos = _layout(engine.cluster, idx)
        results.append({
            "idx": idx,
            "id": host,
            "status": status,
            "reality": _r2(reality_temp),
            "twin": twin_point,
            "lower": twin_lower,
            "upper": twin_upper,
            "error": error,
            "inside": inside,
            "risk": twin_risk,
            **pos,
        })

    # Aggregate stats
    mae = round(np.mean([abs(e) for e in errors]), 2) if errors else None
    coverage = round(n_inside / n_valid * 100, 1) if n_valid > 0 else None

    return jsonify({
        "cluster": cluster,
        "target_time": str(target_t),
        "issue_time": str(issue_t),
        "horizon": horizon,
        "partition": engine.partition(issue_t),
        "in_sample_warning": "in-sample - not a fair test" if engine.partition(issue_t) in ("TRAIN", "VAL_SEL", "VAL_CAL") else None,
        "stats": {
            "mae": mae,
            "coverage_pct": coverage,
            "n_valid": n_valid,
            "n_inside": n_inside,
        },
        "hosts": results,
    })


@app.route("/api/whatif")
def api_whatif():
    cluster = request.args.get("cluster")
    if not cluster or cluster not in ("d1", "d2"):
        return jsonify({"error": "cluster must be d1 or d2"}), 400

    try:
        engine = _get_engine(cluster)
    except Exception as e:
        return jsonify({"error": str(e)}), 503

    # Accept workload params from query string
    workload = {}
    for key in request.args:
        if key != "cluster":
            try:
                workload[key] = float(request.args[key])
            except ValueError:
                pass

    result = engine.whatif(workload)
    return jsonify(result)


@app.route("/api/history/<host_id>")
def api_history(host_id):
    cluster = request.args.get("cluster")
    start = request.args.get("start")
    end = request.args.get("end")

    if not cluster or cluster not in ("d1", "d2"):
        return jsonify({"error": "cluster must be d1 or d2"}), 400
    if not start or not end:
        return jsonify({"error": "start and end required"}), 400

    try:
        engine = _get_engine(cluster)
    except Exception as e:
        return jsonify({"error": str(e)}), 503

    df = engine.history(host_id, pd.Timestamp(start), pd.Timestamp(end))
    records = []
    for _, row in df.iterrows():
        rec = {"time": str(row["Time"])}
        for col in ["CPU_Temperature", "CPU_Load", "Power",
                     "CPU1_Temperature", "CPU2_Temperature",
                     "Inlet_Temperature", "Fan_speed1"]:
            if col in row.index:
                v = row[col]
                rec[col] = _r2(v) if not pd.isna(v) else None
        records.append(rec)

    return jsonify({"host": host_id, "cluster": cluster, "data": records})


# ── Static file serving ──────────────────────────────────────────────

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@app.route("/")
def index():
    return send_from_directory(str(_WEB_DIR), "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(str(_WEB_DIR), filename)


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("TWIN3D_PORT", 5000))
    print(f"Loading engines... (this takes a moment)")
    _get_engines()  # preload
    print(f"Twin3D API ready on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
