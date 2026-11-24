"""Tests for twin3d API — schema, causality, layout, sensors, error codes.

Also includes source-code checks (no Math.random, no hard-coded temps).
"""

import json
import math
import os
import re
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

# ── Conditionally import the Flask app ───────────────────────────────

_HAS_MODELS = (
    (_REPO / "models" / "twin" / "d1_metadata.json").exists()
    and (_REPO / "data" / "interim" / "dataset1").exists()
)

needs_models = pytest.mark.skipif(
    not _HAS_MODELS, reason="twin models or interim data not available"
)


@pytest.fixture(scope="module")
def client():
    """Flask test client — engines loaded once per module."""
    sys.path.insert(0, str(_REPO / "twin3d" / "api"))
    from server import app  # noqa: E402

    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── /api/meta ────────────────────────────────────────────────────────


@needs_models
def test_meta_schema(client):
    r = client.get("/api/meta")
    assert r.status_code == 200
    data = r.get_json()

    assert "banner" in data
    assert "schematic" in data["banner"].lower()
    assert "clusters" in data
    for cid in ("d1", "d2"):
        c = data["clusters"][cid]
        assert "legend" in c
        leg = c["legend"]
        for key in ("t_lo", "t_hi", "p_lo", "p_hi"):
            assert key in leg
            assert isinstance(leg[key], (int, float))
        assert "time_range" in c
        assert "thresholds" in c
        assert c["thresholds"]["warning"] > 0
        assert c["thresholds"]["critical"] > c["thresholds"]["warning"]


@needs_models
def test_meta_colours(client):
    r = client.get("/api/meta")
    data = r.get_json()
    ramp = data["colours"]["temperature_ramp"]
    assert len(ramp) == 5
    assert ramp[0]["hex"] == "#440154"
    assert ramp[-1]["hex"] == "#fde725"


# ── /api/snapshot ────────────────────────────────────────────────────


@needs_models
def test_snapshot_schema(client):
    # Get a valid time from meta
    meta = client.get("/api/meta").get_json()
    t_min = meta["clusters"]["d1"]["time_range"]["min"]

    r = client.get(f"/api/snapshot?cluster=d1&t={t_min}&horizon=1")
    assert r.status_code == 200
    data = r.get_json()

    assert data["cluster"] == "d1"
    assert "hosts" in data
    assert len(data["hosts"]) > 0

    h = data["hosts"][0]
    # Required fields from spec §6
    for field in ("idx", "id", "status", "partition", "temp_eff",
                  "cpu1", "cpu2", "power", "load", "ram_used_pct",
                  "cores_used", "vms", "net_rx", "net_tx"):
        assert field in h, f"missing field {field}"
    # Layout fields
    for field in ("rack", "slot", "rack_label", "x", "y", "z"):
        assert field in h, f"missing layout field {field}"


@needs_models
def test_snapshot_floats_rounded(client):
    meta = client.get("/api/meta").get_json()
    t_min = meta["clusters"]["d1"]["time_range"]["min"]
    r = client.get(f"/api/snapshot?cluster=d1&t={t_min}&horizon=1")
    data = r.get_json()
    for h in data["hosts"]:
        for key in ("temp_eff", "cpu1", "cpu2", "power", "load"):
            v = h[key]
            if v is not None:
                # At most 2 decimal places
                assert round(v, 2) == v, f"{key}={v} not rounded to 2dp"


@needs_models
def test_snapshot_bad_cluster(client):
    r = client.get("/api/snapshot?cluster=d3&t=2019-01-01")
    assert r.status_code == 400


@needs_models
def test_snapshot_missing_params(client):
    r = client.get("/api/snapshot")
    assert r.status_code == 400


# ── Layout determinism and uniqueness ────────────────────────────────


@needs_models
def test_layout_determinism(client):
    meta = client.get("/api/meta").get_json()
    t_min = meta["clusters"]["d1"]["time_range"]["min"]

    r1 = client.get(f"/api/snapshot?cluster=d1&t={t_min}&horizon=1")
    r2 = client.get(f"/api/snapshot?cluster=d1&t={t_min}&horizon=1")
    h1 = r1.get_json()["hosts"]
    h2 = r2.get_json()["hosts"]

    for a, b in zip(h1, h2):
        assert a["rack"] == b["rack"]
        assert a["slot"] == b["slot"]
        assert a["x"] == b["x"]
        assert a["y"] == b["y"]


@needs_models
def test_layout_unique_rack_slot(client):
    meta = client.get("/api/meta").get_json()
    t_min = meta["clusters"]["d1"]["time_range"]["min"]
    r = client.get(f"/api/snapshot?cluster=d1&t={t_min}&horizon=1")
    hosts = r.get_json()["hosts"]
    positions = [(h["rack"], h["slot"]) for h in hosts]
    assert len(positions) == len(set(positions)), "Duplicate (rack, slot) found"


# ── Unavailable sensors ─────────────────────────────────────────────


@needs_models
def test_d2_no_fans_inlet(client):
    """Cluster 2 has no fans or inlet — must be null, not 0."""
    meta = client.get("/api/meta").get_json()
    t_min = meta["clusters"]["d2"]["time_range"]["min"]
    r = client.get(f"/api/snapshot?cluster=d2&t={t_min}&horizon=1")
    hosts = r.get_json()["hosts"]
    for h in hosts:
        assert h["fans"] is None, f"{h['id']} should have null fans"
        assert h["fans_available"] is False
        assert h["inlet"] is None, f"{h['id']} should have null inlet"
        assert h["inlet_available"] is False


# ── Compare / causality ─────────────────────────────────────────────


@needs_models
def test_compare_schema(client):
    meta = client.get("/api/meta").get_json()
    # Pick a time in test partition
    tr = meta["clusters"]["d1"]["time_range"]
    r = client.get(f"/api/compare?cluster=d1&t={tr['max']}&horizon=1")
    assert r.status_code == 200
    data = r.get_json()
    assert "stats" in data
    assert "hosts" in data
    assert "issue_time" in data
    assert "target_time" in data


@needs_models
def test_compare_causality(client):
    """issue_time must be target_time - horizon * 10 min."""
    meta = client.get("/api/meta").get_json()
    tr = meta["clusters"]["d1"]["time_range"]
    for h in (1, 6, 36):
        r = client.get(f"/api/compare?cluster=d1&t={tr['max']}&horizon={h}")
        data = r.get_json()
        from datetime import datetime, timedelta
        target = datetime.fromisoformat(data["target_time"])
        issue = datetime.fromisoformat(data["issue_time"])
        expected_diff = timedelta(minutes=h * 10)
        assert target - issue == expected_diff, (
            f"h={h}: target-issue={target-issue}, expected={expected_diff}"
        )


@needs_models
def test_compare_stats_from_api(client):
    """Stats (MAE, coverage) must be computed from API data, not invented."""
    meta = client.get("/api/meta").get_json()
    tr = meta["clusters"]["d1"]["time_range"]
    r = client.get(f"/api/compare?cluster=d1&t={tr['max']}&horizon=1")
    data = r.get_json()
    s = data["stats"]
    if s["n_valid"] > 0:
        # Recompute from host data
        errors = []
        n_inside = 0
        for h in data["hosts"]:
            if h["status"] == "ok" and h["error"] is not None:
                errors.append(abs(h["error"]))
                if h["inside"]:
                    n_inside += 1
        if errors:
            assert abs(s["mae"] - round(sum(errors) / len(errors), 2)) < 0.02
            assert s["n_inside"] == n_inside


# ── What-if ──────────────────────────────────────────────────────────


@needs_models
def test_whatif(client):
    r = client.get("/api/whatif?cluster=d1&CPU_Load=50&No_Of_Running_vms=5")
    assert r.status_code == 200
    data = r.get_json()
    assert data["available"] is True
    assert "point" in data
    assert "lower" in data
    assert "upper" in data
    assert data["lower"] <= data["upper"]


# ── 503 on missing model ────────────────────────────────────────────


def test_503_on_bad_cluster():
    """Direct engine instantiation with invalid cluster should fail."""
    from ttwin.twin.engine import TwinEngine
    with pytest.raises(Exception):
        TwinEngine("d99")


# ── Source-code checks (no Math.random, no hard-coded temps) ─────────


def test_no_math_random_in_web():
    web_dir = _REPO / "twin3d" / "web"
    if not web_dir.exists():
        pytest.skip("twin3d/web not found")
    for f in web_dir.rglob("*"):
        if f.suffix in (".js", ".html", ".ts"):
            content = f.read_text(encoding="utf-8")
            assert "Math.random" not in content, (
                f"{f.name} contains Math.random — data must come from the API"
            )


def test_no_hardcoded_temps_in_web():
    web_dir = _REPO / "twin3d" / "web"
    if not web_dir.exists():
        pytest.skip("twin3d/web not found")
    # Match suspicious float literals that look like temperatures (40-100 °C range)
    temp_re = re.compile(r'\b(4[0-9]|[5-9][0-9]|100)\.\d+\b')
    for f in web_dir.rglob("*"):
        if f.suffix in (".js", ".ts"):
            content = f.read_text(encoding="utf-8")
            # Exclude known non-temperature constants (camera positions, geometry, etc.)
            # Only flag if it appears in a context suggesting temperature
            lines = content.split("\n")
            for i, line in enumerate(lines):
                # Skip comments, geometry constants, camera positions
                stripped = line.strip()
                if stripped.startswith("//") or stripped.startswith("*"):
                    continue
                if any(kw in line.lower() for kw in ["temp", "celsius", "degree", "threshold"]):
                    matches = temp_re.findall(line)
                    if matches:
                        assert False, (
                            f"{f.name}:{i+1} has suspicious temperature literal: {matches} "
                            f"— temperatures must come from the API"
                        )
