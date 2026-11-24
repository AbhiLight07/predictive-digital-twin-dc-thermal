# 3D Predictive Digital Twin — 5-Minute Demo Script

Target runtime: **~5 minutes** (`python scripts/run_twin3d.py`). Keep the **Model Accuracy & Provenance** panel visible throughout.

---

## 0:00 – 0:45 | 1. Overview & Honesty Framing
- **State**: Default load on `All Clusters`, `Horizon = +1 hour (h=6)`, timeline anchored at common test start (`2019-05-25 03:50 UTC`).
- **Action**: Orbit slightly (`Overview` preset) to show both schematic rows (`C1-R1..R4` and `C2-R1..R3`).
- **What to say**:
  > "This is an offline, model-driven predictive digital twin of the University of Melbourne research cloud — 76 servers in Cluster 1 and 60 in Cluster 2. Notice the top banner: the dataset contains no floorplan, CRAC cooling units, or ambient weather, so this 42U rack arrangement is **schematic** and thresholds are data-derived percentiles, not vendor limits."
  > "All colours, halos, and predictions come live from our FastAPI server reading the saved LightGBM + CQR models — zero mock data."

## 0:45 – 1:30 | 2. Warning Server & Alerts
- **Action**: Scrub/jump to **`2019-05-27 16:20 UTC`** (high-alert window) -> inspect the pulsing amber/red beacons on servers and rack tops -> click the top host in the **Alerts** panel.
- **What to say**:
  > "At this timestamp, several hosts trigger warning or critical beacons because their **CQR upper bound** crosses our p90/p99 thresholds at 1h or 6h ahead. Clicking an alert flies directly to that server and opens its telemetry, partition status, and 24-hour trace."

## 1:30 – 2:15 | 3. Forecast Horizons & Uncertainty Halos
- **Action**: Switch **Horizon** `Now (h=0)` → `+10 min (h=1)` → `+1 hour (h=6)` → `+6 hours (h=36)`. Point at the translucent halos around slabs and the **Horizon fan** mini-chart in the inspector.
- **What to say**:
  > "When horizon increases from 10 minutes to 6 hours, glow switches to the causal forecast point, and each translucent halo expands proportionally to the calibrated CQR interval width (`w_ref`). As shown in the Accuracy panel, short horizons match persistence accuracy closely due to thermal inertia, while CQR gives empirically calibrated uncertainty bounds (~86–90% test coverage)."

## 2:15 – 3:15 | 4. Compare Mode — "The twin predicted this one hour ago"
- **Action**: Set **Horizon** back to `+1 hour (h=6)`, select `Cluster 1` camera, check **Compare**. Inspect Left vs Right viewports, the bottom stats overlay (`Frame MAE`, `% Inside CQR`, `Last 24h MAE`), then toggle **Error** overlay (`±3°C` diverging ramp) briefly.
- **What to say**:
  > "Compare mode is the core validation test. Both viewports share one camera. On the left is **observed reality** at time `t`; on the right is **what the twin predicted at `t - 1 hour` using strictly past telemetry**. Notice the stats box updates directly from `/api/compare` with exact frame and 24-hour MAE and empirical CQR coverage. Toggling Error highlights above/below deviations in blue/red without hiding misses."

## 3:15 – 4:00 | 5. Live What-If on a New Server
- **Action**: Uncheck **Compare**. Under **What-If (Planning Model)** switch target from `Selected host` to `New server` (`Cluster 1 (D1)`), then drag **CPU Load** from `20%` → `90%` and **VMs** up to `25`. Show the semi-transparent **ghost server**, its colour/halo shift, and the **CPU load sensitivity curve**.
- **What to say**:
  > "Using our steady-state workload planning model, operators can evaluate hypothetical workload allocations on an existing host or a new server bay. Notice the disclaimer badge: **`predictive, correlational - not causal`** — there are no fan RPM overrides or cooling actuation sliders because the trace has no cooling-intervention experiments."

## 4:00 – 4:40 | 6. Server Anatomy (D1 vs D2)
- **Action**: Select a D1 server (e.g., `qh2-rcc120`) and click **Inspect server**. Point out glowing `CPU1`/`CPU2` heatsinks, 4 spinning front fans (note visual cap `12 rad/s`), inlet airflow streaks, Power LED, and the bottom 24h chart. Click **Back to hall**, switch cluster filter to **Cluster 2 (D2)**, select `qh2-rcc201`, and click **Inspect server** again.
- **What to say**:
  > "In anatomy view, component mappings reflect real sensors — and equally important, **honesty about missing sensors**: Cluster 2 servers have real CPU1/CPU2 glows, but grey static parts labelled **`no fan sensor`** and **`no inlet sensor`**, never animated from fabricated values."

## 4:40 – 5:00 | 7. Provenance & Limitations Close
- **Action**: Click **Back to hall**, scroll briefly to **Model Accuracy & Provenance** (`What the twin does not know`).
- **What to say**:
  > "Every metric on screen—MAE vs persistence, CQR coverage, and model version—stays visible alongside what the twin does not know: no physical rack positions, no CRAC/weather data, and no live bidirectional control."
