# 3D Predictive Digital Twin — User Guide

> Offline, model-driven predictive digital twin prototype replaying the University of Melbourne research-cloud trace (Ilager et al., UCC 2023). Not live, not bidirectional, and no cooling control.

---

## 1. Prerequisites & Installation

```powershell
# Clone and enter repository
git clone https://github.com/AbhiLight07/predictive-digital-twin-dc-thermal.git
cd predictive-digital-twin-dc-thermal

# Create / activate virtual environment (Windows)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install core project + 3D API dependencies (includes FastAPI, uvicorn, httpx, playwright)
pip install -r requirements.txt
```

All Three.js modules are vendored locally under `twin3d/web/vendor/three/` (`three@0.164.1`); **no internet connection or CDN is required at runtime**.

---

## 2. Prepare Data, Models & Forecast Cache

*(Note: All 24 trained LightGBM models and metadata in `models/twin/` are already tracked in Git. The steps below that read/write `data/raw/`, `data/interim/`, or `data/twin_cache/` **require the Zenodo raw dataset**.)*

1. **Telemetry data *(requires Zenodo raw CSVs)*:** Download the University of Melbourne trace (`10.5281/zenodo.10069402`) into `data/raw/` and convert to interim parquet (`data/interim/dataset1/` and `data/interim/dataset2/`):
   ```powershell
   python scripts/make_parquet.py
   ```
   See `data/README.md`.
2. **Train twin models *(optional — already committed in `models/twin/`; requires interim data)*:**
   ```powershell
   python scripts/train_twin_models.py --cluster d1 --horizons 1 6 36 --out models/twin
   python scripts/train_twin_models.py --cluster d2 --horizons 1 6 36 --out models/twin
   ```
   Saves LightGBM point/lower/upper models, Conformalized Quantile Regression (CQR) calibration `q_hat`, and test metrics into `models/twin/`.
3. **Build forecast cache *(requires interim data)*:**
   ```powershell
   python scripts/build_twin_cache.py
   ```
   Writes 30-minute stride test-partition forecasts (`h = 1, 6, 36` steps) to `data/twin_cache/d1_forecasts.parquet` (550,848 rows, 9.2 MB) and `data/twin_cache/d2_forecasts.parquet` (205,671 rows, 4.3 MB). Measured runtime: **69.5 s (D1) + 33.1 s (D2) = 108.9 s total** on Windows.

---

## 3. Run the 3D Digital Twin *(requires interim data + cache)*

```powershell
python scripts/run_twin3d.py
```

Options:
- `--host 127.0.0.1` — bind address
- `--port 8000` — HTTP port (opens `http://127.0.0.1:8000` automatically)
- `--no-browser` — start the FastAPI server without launching a browser window

Wait for initial model/cache indexing (~15–25s on startup); subsequent snapshot requests run under 100 ms.

---

## 4. Controls & Views

| Control | Function |
|---|---|
| **Cluster** (`All / D1 / D2`) | Filters visible telemetry and provenance metrics |
| **Horizon** (`Now / +10m / +1h / +6h`) | Switches glow between observed `temp_eff` and causal forecast `point` with CQR uncertainty halos |
| **Temp / Risk** | Colours glow strips by temperature scale or CQR upper-bound risk level |
| **Viridis / Cividis** | Toggles between colour-blind-safe temperature ramps |
| **Camera** (`Overview / C1 / C2 / Selected`) | Smoothly flies camera to preset viewpoints (orbit/pan/zoom via mouse) |
| **Timeline / Play** | Scrubs 30-minute frames across train/val/test partitions; `TEST` frames are out-of-sample |
| **Compare / Error / Single** | Linked dual WebGL scissor viewports: Left = observed reality at `t`; Right = Twin prediction issued at `t - h` (+ optional ±3°C diverging error overlay and single-viewport toggle) |
| **Alerts List** | Lists hosts crossing warning/critical upper bounds at `h=1h` or `h=6h`; clicking flies to host |
| **What-If Sliders** | Queries saved planning models (`POST /api/whatif`) for a selected host or new server, updating a semi-transparent ghost server & sensitivity curve (*correlational, not causal*) |
| **Inspect server** | Enters **Server Anatomy View**: dual glowing CPU1/CPU2 heatsinks, scaled front fans (D1 only; D2 shows static grey `"no fan sensor"`), inlet airflow streaks (D1 only; D2 shows `"no inlet sensor"`), Power LED, and 24h chart |

---

## 5. Verification & Evidence Script

```powershell
# Works in a fresh clone (synthetic engine + offline Playwright WebGL; no raw data needed)
pytest tests/test_twin3d_api.py tests/test_twin3d_ui.py -v

# Requires interim data & forecast cache
python scripts/validate_twin3d.py
```
