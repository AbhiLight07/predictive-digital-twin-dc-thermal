# 3D Digital Twin — Validation Report

All measurements below were collected automatically by `scripts/validate_twin3d.py` against the real D1 (76 hosts) and D2 (60 hosts) engines, saved LightGBM models, and precomputed forecast caches.

## 1. Static Source Hygiene & Offline Execution
- Non-vendor web files inspected (`twin3d/web/`, excluding `vendor/`): **9**
- `Math.random` occurrences: **0**
- External/CDN URL occurrences (`http://`, `https://`, `cdn.jsdelivr`, `unpkg`): **0**
- Hard-coded temperature literals: **0**
- Playwright offline route test (non-localhost requests blocked): **0 external requests attempted**, **0 console errors**

## 2. API Latency & Payload Size (10 warm calls per endpoint)
| Endpoint | p50 (ms) | p95 (ms) | Payload (bytes) |
|---|---|---|---|
| `GET /api/health` | 3.65 | 4.75 | 199 |
| `GET /api/meta` | 11.37 | 15.21 | 37936 |
| `GET /api/snapshot (d1, h=6)` | 35.74 | 48.03 | 101429 |
| `GET /api/snapshot (all, h=6)` | 59.27 | 67.59 | 178511 |
| `GET /api/compare (d1, h=6)` | 17.60 | 18.26 | 17197 |
| `GET /api/series (d1, 24h)` | 66.08 | 71.43 | 12040 |
| `GET /api/alerts (all)` | 9.67 | 11.38 | 9022 |
| `POST /api/whatif (d1)` | 8.65 | 9.64 | 1084 |

## 3. 60-Second Replay Soak Test (Headless Chrome SwiftShader WebGL, 136 hosts)
> Note: Headless Chromium uses software ANGLE/SwiftShader rendering, so FPS below reflects CPU WebGL rasterisation rather than hardware GPU throughput.

| Elapsed (s) | FPS | Draw Calls | Hosts Rendered | JS Heap (MB) |
|---|---|---|---|---|
| 0 | 2 | 1 | 136 | 10.95 |
| 30 | 3 | 12 | 136 | 13.80 |
| 60 | 3 | 12 | 136 | 14.92 |

- Observed JS heap delta (`t=0s` to `t=60s`): **+3.97 MB**
- Measured compare-mode frame rate / draw calls: **2 fps · 12 draws**

## 4. Compare Mode Stats Across 3 Test-Period Days (`horizon=6` steps / 1 h)
| Cluster | Target Time | Issue Time (`t - 1h`) | Partition | Frame MAE (°C) | Inside CQR (%) | n_inside / n_valid |
|---|---|---|---|---|---|---|
| D1 | `2019-05-05T12:20:00Z` | `2019-05-05T11:20:00Z` | TEST | 0.61 | 90.79% | 69 / 76 |
| D1 | `2019-05-25T12:20:00Z` | `2019-05-25T11:20:00Z` | TEST | 0.64 | 90.79% | 69 / 76 |
| D1 | `2019-06-15T12:20:00Z` | `2019-06-15T11:20:00Z` | TEST | 0.5 | 97.37% | 74 / 76 |
| D2 | `2019-05-28T12:20:00Z` | `2019-05-28T11:20:00Z` | TEST | 0.89 | 87.27% | 48 / 55 |
| D2 | `2019-06-06T12:20:00Z` | `2019-06-06T11:20:00Z` | TEST | 0.51 | 89.8% | 44 / 49 |
| D2 | `2019-06-15T12:20:00Z` | `2019-06-15T11:20:00Z` | TEST | 0.83 | 90.91% | 50 / 55 |

## 5. Visual Evidence Captures
| File | Size (bytes) | RGB Pixel Std (non-blank > 4.0) |
|---|---|---|
| `docs/twin3d_screens/01_overview.png` | 246872 | 34.59 |
| `docs/twin3d_screens/06_alerts.png` | 250728 | 36.49 |
| `docs/twin3d_screens/02_forecast_halos.png` | 199952 | 51.12 |
| `docs/twin3d_screens/03_compare_mode.png` | 202042 | 37.8 |
| `docs/twin3d_screens/04_whatif_ghost.png` | 230053 | 45.88 |
| `docs/twin3d_screens/05_anatomy.png` | 137518 | 30.12 |
- Walkthrough video (`docs/twin3d_demo.webm`): **PASS** (Recorded 24304.4 KB webm)
