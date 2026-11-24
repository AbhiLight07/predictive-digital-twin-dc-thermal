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

# Install core project + 3D API dependencies
pip install -r requirements.txt
pip install fastapi uvicorn httpx playwright
```

All Three.js modules are vendored locally under `twin3d/web/vendor/three/` (`three@0.164.1`); **no internet connection or CDN is required at runtime**.

---

## 2. Prepare Data, Models & Forecast Cache

1. **Telemetry data**: Download the University of Melbourne trace (Zenodo `10.5281/zenodo.10069402`) and convert raw files to interim parquet (`data/interim/dataset1/` and `data/interim/dataset2/`) using `scripts/make_parquet.py`. See `data/README.md`.
2. **Train twin models**:
   ```powershell
   python scripts/train_twin_models.py
   ```
   Saves LightGBM point/lower/upper models, Conformalized Quantile Regression (CQR) calibration `q_hat`, and test metrics into `models/twin/`.
3. **Build forecast cache**:
   ```powershell
   python scripts/build_twin_cache.py
   ```
   Writes 30-minute stride test-partition forecasts (`h = 1, 6, 36` steps) to `data/twin_cache/d1_forecasts.parquet` and `data/twin_cache/d2_forecasts.parquet`.

---

## 3. Run the 3D Digital Twin

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
pytest tests/test_twin3d_api.py tests/test_twin3d_ui.py -v
python scripts/validate_twin3d.py
```
