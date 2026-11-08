# Data Audit Report

**Date:** 2026-11-08  
**Script:** `scripts/data_audit.py`  
**Git commit:** e3b2154  

---

## Executive Summary

| Property | Dataset 1 | Dataset 2 |
|---|---|---|
| File | `dataset1_combined.csv` | `dataset2_combined.csv` |
| Size | 584.8 MB | 175.6 MB |
| Rows | 2,748,525 | 1,281,024 |
| Servers | 81 (`qh2-rcc120`--`qh2-rcc200`) | 64 (`qh2-rcc201`--`qh2-rcc264`) |
| Columns | 18 | 13 |
| Sampling interval | 10 min (confirmed) | 10 min (confirmed) |
| Coverage period | 2018-10-09 to 2019-06-20 (~254 days) | 2019-02-01 to 2019-06-20 (~139 days) |
| Has fan speed | Yes (4 fans) | No |
| Has inlet temperature | Yes | No |
| CPU temperature missing | ~0.00% | ~5--14% (varies by server) |
| Gate qualifying servers | **76** | **2** |
| **Gate verdict** | **PASS** | FAIL (standalone) |

**Overall gate verdict: PASS.** Dataset 1 alone provides 76 servers with >=60 days coverage and <10% missing in temperature and workload columns, well above the 40-server threshold.

---

## 1. Cluster Hardware (reported, to be verified)

- **Dataset 1 (Cluster 1):** Dell R840 servers. 81 servers, richer instrumentation (inlet temp, 4 fan speeds).
- **Dataset 2 (Cluster 2):** Dell C6320 servers. 64 servers, fewer sensors (no fan, no inlet).

Both clusters share the same naming convention (`qh2-rcc*`), non-overlapping server IDs.

---

## 2. Column Inventory

### Dataset 1 -- 18 columns

| Column | Dtype | Category | Missing % | Notes |
|---|---|---|---|---|
| `Series` | str | host ID | 0.00% | Server identifier |
| `Time` | str | timestamp | 0.00% | Format: `YYYY-MM-DD HH:MM:SS` |
| `CPU_Load` | float64 | workload | 0.00% | CPU utilisation |
| `Power` | float64 | power | 0.00% | Server power draw |
| `Network_RX` | float64 | network | 0.00% | Bytes received |
| `Network_TX` | float64 | network | 0.00% | Bytes transmitted |
| `Inlet_Temperature` | float64 | temperature | 3.05% | Air intake temp; 5 servers (rcc196--200) have 100% missing |
| `CPU1_Temperature` | float64 | temperature | 0.00% | CPU socket 1 |
| `CPU2_Temperature` | float64 | temperature | 0.00% | CPU socket 2 |
| `Fan_speed1` | float64 | fan | 3.05% | Same 5 servers missing as Inlet_Temperature |
| `Fan_speed2` | float64 | fan | 3.05% | |
| `Fan_speed3` | float64 | fan | 3.05% | |
| `Fan_speed4` | float64 | fan | 3.05% | |
| `Ram` | float64 | memory | 0.38% | Total RAM |
| `Ram_Used` | float64 | memory | 0.38% | RAM in use |
| `No_Of_Running_vms` | float64 | VM | 0.38% | Active VM count |
| `CPU_cores` | float64 | workload | 0.38% | Total CPU cores |
| `CPU_cores_used` | float64 | workload | 0.40% | Cores in use |

### Dataset 2 -- 13 columns

| Column | Dtype | Category | Missing % | Notes |
|---|---|---|---|---|
| `Series` | str | host ID | 0.00% | Server identifier |
| `Time` | str | timestamp | 0.00% | |
| `CPU_Load` | float64 | workload | 7.39% | |
| `CPU1_Temperature` | float64 | temperature | 13.87% | High missingness across many servers |
| `CPU2_Temperature` | float64 | temperature | 13.86% | |
| `Ram` | float64 | memory | 15.81% | |
| `Ram_Used` | float64 | memory | 15.81% | |
| `CPU_cores` | float64 | workload | 15.81% | |
| `CPU_cores_used` | float64 | workload | 15.84% | |
| `Power` | float64 | power | 13.84% | |
| `Network_RX` | float64 | network | 7.83% | |
| `Network_TX` | float64 | network | 7.83% | |
| `No_Of_Running_vms` | float64 | VM | 15.81% | |

**Missing columns in Dataset 2 vs Dataset 1:** `Inlet_Temperature`, `Fan_speed1`--`Fan_speed4`. This confirms the brief's expectation. Feature set S4 (full monitoring) applies only to Cluster 1.

---

## 3. Shared Columns (for cross-cluster experiments, E3)

Both datasets share these columns with identical names:

| Column | Category |
|---|---|
| `Series` | host ID |
| `Time` | timestamp |
| `CPU_Load` | workload |
| `CPU1_Temperature` | temperature |
| `CPU2_Temperature` | temperature |
| `Ram` | memory |
| `Ram_Used` | memory |
| `CPU_cores` | workload |
| `CPU_cores_used` | workload |
| `Power` | power |
| `Network_RX` | network |
| `Network_TX` | network |
| `No_Of_Running_vms` | VM |

Temperature columns use the same names and units (degrees C, plausible ranges 25--82), so cross-cluster transfer (E3) is feasible if the higher missingness in Dataset 2 is handled.

---

## 4. Per-Server Coverage

### Dataset 1

- **76 of 81** servers have >=60 days and <10% missing
- Most servers span 253.9 days (2018-10-09 to 2019-06-20)
- 5 servers (`rcc196`--`rcc200`) started later (2019-02-01) with ~118 days and have **no inlet temp or fan speed data** (100% missing in those columns). They still have CPU temperatures.
- 3 servers (`rcc184`, `rcc186`, `rcc187`) started 2018-11-28 with ~204 days
- Some servers have fewer rows than expected (e.g., `rcc121` has 28,792 vs typical ~35,870), indicating sampling gaps
- Max sampling gap observed: 3 days 23:30:00

### Dataset 2

- **Only 2 of 64** servers qualify for the gate (`rcc206`, `rcc207`)
- All 64 servers span 139.0 days (2019-02-01 to 2019-06-20)
- Uniform row count: 20,016 per server
- High missingness in most columns: temperature ~5--50%, RAM/cores ~16%
- 2 servers (`rcc263`, `rcc264`) have 100% missing temperature -- essentially dead sensors
- 1 server (`rcc226`) has ~79% missing temperature

---

## 5. Temperature Statistics

### Dataset 1 -- CPU temperature ranges (degrees C)
- **CPU1_Temperature:** mean 39.6--68.2, min 29.1--56.3, max 48.8--82.0
- **CPU2_Temperature:** mean 35.7--62.0, min 25.5--51.0, max 44.8--78.0
- **Inlet_Temperature:** mean 4.2--20.8, min 4.0--17.2, max 17.9--25.8
  - Note: several servers have inlet min of 4.0 C which is suspiciously low (possible sensor error or unheated room)

### Dataset 2 -- CPU temperature ranges (degrees C)
- **CPU1_Temperature:** mean 35.1--70.1, min 26.7--43.0, max 42.4--79.2
- **CPU2_Temperature:** mean 33.7--73.4, min 25.6--46.0, max 41.8--81.4

Temperature ranges are physically comparable across clusters (same units, similar magnitudes).

---

## 6. Sampling Interval

Both datasets: **10 minutes** (median), confirmed.

Horizons from the brief (10 min, 1 h, 6 h) correspond to 1, 6, and 36 timesteps.

---

## 7. Gate Verdict

| Criterion | Threshold | Result |
|---|---|---|
| Qualifying servers (>=60 days, <10% missing temp+workload) | >=40 | **76** (Dataset 1) |
| Licence permits publication | CC BY 4.0 | **Yes** |
| Temperature columns present | Yes | **Yes** (CPU1, CPU2 in both; Inlet in D1) |

**GATE: PASS**

---

## 8. Key Observations and Implications

1. **Dataset 1 is the primary workhorse.** 76 clean servers with rich instrumentation; all core experiments (E1--E5) can run on it.

2. **Dataset 2 has quality issues.** High missingness limits its standalone use, but it remains valuable for cross-cluster transfer (E3) -- provided we handle missing data carefully and limit the experiment to the ~40 servers with <15% temperature missingness.

3. **5 servers in Dataset 1 lack inlet/fan data.** These (`rcc196`--`rcc200`) can still participate in S0--S3 experiments but not S4.

4. **Inlet temperature anomalies.** Several servers show inlet temps as low as 4.0 C. Investigate whether these are sensor errors or real readings. Could affect the inlet-to-CPU temperature difference target.

5. **No disk/storage columns** in either dataset. The brief mentions disk/data usage for S2 features -- this is unavailable. S2 will use: CPU_Load, Ram, Ram_Used, CPU_cores, CPU_cores_used, Network_RX, Network_TX, No_Of_Running_vms, plus calendar features.

6. **Column order differs** between datasets but names match. Code should reference by name, not position.

---

## 9. Decisions Needed

1. **Missingness handling for Dataset 2:** Accept higher missingness (relax the 10% threshold for D2 servers) for cross-cluster experiments, or use only the 2 qualifying servers?
   - *Recommendation:* Use all D2 servers with <20% temperature missingness (~40 servers) for E3, documenting the relaxed threshold.

2. **Inlet temperature outliers:** Flag 4.0 C readings as suspect or accept them?
   - *Recommendation:* Investigate in EDA; if they appear as sustained low readings (not transient), treat as valid but note the anomaly.

3. **Missing disk/storage features:** Acknowledged -- not available. S2 feature set adjusted accordingly.
