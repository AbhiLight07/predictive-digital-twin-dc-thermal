# Predictive Digital Twin — Specification

> Offline, data-driven predictive digital twin replaying the University of Melbourne
> research-cloud trace (Ilager et al., UCC 2023).

---

## 1. Entities

### HostStatic
| Field | Description |
|---|---|
| `cluster` | `"D1"` or `"D2"` |
| `host_id` | e.g. `qh2-rcc120` |
| `cores` | Total CPU cores (from `CPU_cores` column) |
| `ram_gb` | Total RAM (from `Ram` column) |
| `hardware_model` | As reported in the dataset paper (Ilager et al., UCC 2023) |
| `has_fan` | `True` for D1 (except 5 servers), `False` for D2 |
| `has_inlet` | `True` for D1 (except 5 servers), `False` for D2 |

### HostState (at time *t*)
Each signal carries:
- `value`: float or NaN
- `availability`: bool — sensor exists for this cluster
- `quality`: enum `ok | stale | missing`

Signals: CPU_Load, Ram, Ram_Used, CPU_cores, CPU_cores_used, Network_RX,
Network_TX, No_Of_Running_vms, Power, Fan_speed1–4 (D1 only),
Inlet_Temperature (D1 only), CPU1_Temperature, CPU2_Temperature,
CPU_Temperature (effective = max(CPU1, CPU2)).

### HostForecast
| Field | Description |
|---|---|
| `horizon` | 10 min (1 step) · 1 h (6 steps) · 6 h (36 steps) |
| `point` | Predicted effective CPU temperature (°C) |
| `lower` / `upper` | Conformal prediction interval bounds |
| `persistence` | Persistence baseline value T(t) |
| `risk` | `RiskLevel` enum (normal / warning / critical / unknown) |
| `model_version` | Model identifier string |
| `partition` | `TRAIN` / `VAL` / `TEST` |

### RiskLevel (enum)
`normal` · `warning` · `critical` · `unknown`

---

## 2. Views

| View | Purpose |
|---|---|
| **Fleet** | All hosts, current state, risk heat-map (schematic layout, not physical) |
| **Server Detail** | Single host: time-series, forecast fans, intervals |
| **Alerts** | Hosts crossing warning / critical thresholds |
| **What-If** | Correlational workload scenarios (not causal) |
| **Provenance** | Model version, training split, data partition, metric lineage |

---

## 3. Signal Availability

| Signal | Available | Not Available |
|---|---|---|
| CPU_Load | ✓ | |
| Ram, Ram_Used | ✓ | |
| CPU_cores, CPU_cores_used | ✓ | |
| Network_RX, Network_TX | ✓ | |
| No_Of_Running_vms | ✓ | |
| Power | ✓ | |
| CPU1_Temperature, CPU2_Temperature | ✓ | |
| Fan_speed1–4 | D1 only | D2 |
| Inlet_Temperature | D1 only | D2 |
| Rack position / layout | | ✗ — not in dataset |
| Cooling unit / CRAC telemetry | | ✗ — not in dataset |
| Room / outdoor temperature | | ✗ — not in dataset |
| Per-VM workload breakdown | | ✗ — not in dataset |

---

## 4. Mandatory Honest Labels

1. **Offline replay** of recorded historical data — not a live system.
2. **Schematic layout** — no physical rack positions are known.
3. **Thresholds are illustrative** — derived from data percentiles, not vendor limits.
4. **What-if is correlational**, not causal — no physical intervention data exists.
5. **No cooling control** — the twin cannot actuate fans, CRAC, or chillers.

---

## 5. Acceptance Criteria for Steps 2–6

| Step | Criterion |
|---|---|
| **Step 2 — Replay engine** | Streams test-period data at configurable stride; emits `HostState` per host per tick; handles missing sensors gracefully. |
| **Step 3 — Forecasting** | Produces `HostForecast` at all three horizons with calibrated conformal intervals; persistence baseline included; skill score vs persistence ≥ 0 on test set. |
| **Step 4 — Risk scoring** | Applies twin.yaml thresholds; `RiskLevel` assigned correctly; alerts fire when crossing warning/critical. |
| **Step 5 — Streamlit UI** | Fleet, detail, alert, what-if, and provenance views render; schematic and honest labels visible; runs without errors on test replay. |
| **Step 6 — Integration tests** | End-to-end replay produces forecasts, risk levels, and alerts consistent with schema; pytest passes; no regressions. |
