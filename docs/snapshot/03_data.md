# 03 Data Evidence

## 1. Raw Files in `data/raw/`

Command:
```powershell
Get-ChildItem data\raw -Recurse -File | ForEach-Object {
    $h = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
    "$($_.Name) | SHA256: $h | Bytes: $($_.Length)"
}
```

Raw Output:
```text
dataset1_combined.csv | SHA256: 18592C5A1A9EDF021CB16E5775CD12BFB1479D244C10E4D913BF18CEFE7B0256 | Bytes: 613,212,066
dataset2_combined.csv | SHA256: 2B5709F16E6AE94D274D1F2B111EF67A38EDCCA8E1D22C6C02182308042A08BB | Bytes: 184,178,675
```

---

## 2. Dataset Properties After Interim Cleaning (`data/interim/`)

Measured directly from `data/interim/dataset1/_combined.parquet` and `data/interim/dataset2/_combined.parquet`:

| Metric | Dataset 1 (`dataset1_combined.csv`) | Dataset 2 (`dataset2_combined.csv`) |
|---|---|---|
| **Interim Rows** | 2,748,039 | 1,280,640 |
| **Unique Server Identifiers** | 81 (`qh2-rcc120` to `qh2-rcc200`) | 64 (`qh2-rcc201` to `qh2-rcc264`) |
| **First Timestamp** | `2018-10-09 11:20:00` | `2019-02-01 11:10:00` |
| **Last Timestamp** | `2019-06-20 10:00:00` | `2019-06-20 10:00:00` |
| **Temporal Span** | 253.94 days | 138.95 days |
| **Sampling Interval (Median / Mode)** | 10 minutes (`0 days 00:10:00`) | 10 minutes (`0 days 00:10:00`) |
| **Nominal Interval Steps / Server** | ~36,568 expected steps | ~20,010 expected steps |

### 2.1 Host ID Trailing Whitespace Evidence
In the raw CSV `dataset1_combined.csv`, host names for servers 120 through 195 were recorded with a trailing space character (`"qh2-rcc120 "` through `"qh2-rcc195 "`), whereas servers 196 through 200 had no trailing space (`"qh2-rcc196"` through `"qh2-rcc200"`).

Raw Inspection Output:
- Servers with trailing whitespace in Dataset 1 Parquet: 76 hosts (`qh2-rcc120 ` to `qh2-rcc195 `).
- Servers with trailing whitespace in Dataset 2: 0 hosts (all 64 are cleanly stripped).
- Code behavior: `src/ttwin/clean.py` defines `strip_host_names()`, but `clean_dataset()` in `clean.py` omitted calling it. However, `src/ttwin/splits.py` explicitly executes `.str.strip()` inside `get_server_list()`, `split_leave_servers_out()`, `get_group_kfold_splits()`, and `split_strict()`. Furthermore, all v2 experiment scripts (`run_v2_e1_ablation.py`, `run_v2_planning_adaptation.py`, `run_v2_uncertainty.py`) explicitly execute `df["Series"] = df["Series"].astype(str).str.strip()` immediately after load.
- Deduplication check: No physical server appears under two distinct names (e.g., `"qh2-rcc120"` and `"qh2-rcc120 "` do not co-exist within the same file). The stripped set has exactly 81 unique servers for D1 and 64 for D2.

---

## 3. `configs/columns.yaml` (Verbatim)

```yaml
# Column mappings for the Melbourne cloud trace dataset
# Verified from data audit on 2026-11-08
#
# Dataset 1: 81 servers (qh2-rcc120 to qh2-rcc200), 18 columns
# Dataset 2: 64 servers (qh2-rcc201 to qh2-rcc264), 13 columns

dataset1:
  file: dataset1_combined.csv
  hardware: "Dell R840 (reported, unverified)"
  host_col: Series
  time_col: Time
  n_servers: 81
  n_rows: 2748525
  coverage_days: 253.9
  date_range: ["2018-10-09", "2019-06-20"]
  sampling_interval_min: 10

  target_cols:
    - CPU1_Temperature
    - CPU2_Temperature

  workload_cols:
    - CPU_Load
    - Ram
    - Ram_Used
    - CPU_cores
    - CPU_cores_used
    - Network_RX
    - Network_TX
    - No_Of_Running_vms

  power_cols:
    - Power

  fan_cols:
    - Fan_speed1
    - Fan_speed2
    - Fan_speed3
    - Fan_speed4

  temperature_cols:
    - Inlet_Temperature
    - CPU1_Temperature
    - CPU2_Temperature

  # Servers with 100% missing inlet/fan data (still have CPU temps)
  servers_no_inlet_fan:
    - qh2-rcc196
    - qh2-rcc197
    - qh2-rcc198
    - qh2-rcc199
    - qh2-rcc200

dataset2:
  file: dataset2_combined.csv
  hardware: "Dell C6320 (reported, unverified)"
  host_col: Series
  time_col: Time
  n_servers: 64
  n_rows: 1281024
  coverage_days: 139.0
  date_range: ["2019-02-01", "2019-06-20"]
  sampling_interval_min: 10

  target_cols:
    - CPU1_Temperature
    - CPU2_Temperature

  workload_cols:
    - CPU_Load
    - Ram
    - Ram_Used
    - CPU_cores
    - CPU_cores_used
    - Network_RX
    - Network_TX
    - No_Of_Running_vms

  power_cols:
    - Power

  fan_cols: []  # Not available in this cluster

  temperature_cols:
    - CPU1_Temperature
    - CPU2_Temperature
    # No Inlet_Temperature in this cluster

  # Servers with severe data quality issues (>50% temp missing or dead sensors)
  servers_exclude:
    - qh2-rcc263  # 100% missing temperature
    - qh2-rcc264  # 100% missing temperature
    - qh2-rcc226  # ~79% missing temperature

# Shared columns across both clusters (for cross-cluster E3)
shared_columns:
  host: Series
  time: Time
  workload:
    - CPU_Load
    - Ram
    - Ram_Used
    - CPU_cores
    - CPU_cores_used
    - Network_RX
    - Network_TX
    - No_Of_Running_vms
  power:
    - Power
  temperature:
    - CPU1_Temperature
    - CPU2_Temperature

# Feature sets (ablation ladder)
feature_sets:
  S0_persistence:
    description: "Next value = last value"
    features: []  # No features; prediction = last observed temperature

  S1_history_only:
    description: "Lagged temperature only"
    features:
      - CPU1_Temperature_lag_*
      - CPU2_Temperature_lag_*

  S2_workload_only:
    description: "Planning setting -- workload + calendar, no power/fan/temp history"
    features:
      - CPU_Load
      - Ram
      - Ram_Used
      - CPU_cores
      - CPU_cores_used
      - Network_RX
      - Network_TX
      - No_Of_Running_vms
      - hour
      - dayofweek
    note: "No disk/storage columns available in this dataset"

  S3_workload_power:
    description: "S2 + power"
    features:
      - S2_workload_only
      - Power

  S4_full_monitoring:
    description: "S3 + fan speeds + inlet temp + temperature lags (Dataset 1 only)"
    features:
      - S3_workload_power
      - Fan_speed1
      - Fan_speed2
      - Fan_speed3
      - Fan_speed4
      - Inlet_Temperature
      - CPU1_Temperature_lag_*
      - CPU2_Temperature_lag_*
    note: "Only applicable to Dataset 1; 5 servers lack fan/inlet data"
```

---

## 4. Missingness of Columns Actually Used

Computed strictly across rows of `data/interim/*/_combined.parquet`:

### 4.1 Dataset 1 (Dell PowerEdge R840, 2,748,039 total rows)
| Column Name | Category | Missing Count | Missing Percentage | Operational Handling |
|---|---|---|---|---|
| `CPU1_Temperature` | Target / State | 86 | 0.003% | Dropped via `.dropna()` |
| `CPU2_Temperature` | Target / State | 91 | 0.003% | Dropped via `.dropna()` |
| `CPU_Load` | Workload | 85 | 0.003% | Dropped via `.dropna()` |
| `Power` | Workload / Power | 83 | 0.003% | Dropped via `.dropna()` |
| `Network_RX` | Workload | 83 | 0.003% | Dropped via `.dropna()` |
| `Network_TX` | Workload | 86 | 0.003% | Dropped via `.dropna()` |
| `Ram` | Workload | 10,489 | 0.382% | Dropped via `.dropna()` |
| `Ram_Used` | Workload | 10,489 | 0.382% | Dropped via `.dropna()` |
| `No_Of_Running_vms`| Workload | 10,489 | 0.382% | Dropped via `.dropna()` |
| `CPU_cores` | Static capacity| 10,489 | 0.382% | Dropped via `.dropna()` |
| `CPU_cores_used` | Workload | 10,944 | 0.398% | Dropped via `.dropna()` |
| `Inlet_Temperature`| Telemetry (S4)| 83,769 | 3.048% | Concentrated on 5 servers (100% missing) |
| `Fan_speed1` | Cooling (S4) | 83,770 | 3.048% | Concentrated on 5 servers (100% missing) |
| `Fan_speed2` | Cooling (S4) | 83,769 | 3.048% | Concentrated on 5 servers (100% missing) |
| `Fan_speed3` | Cooling (S4) | 83,763 | 3.048% | Concentrated on 5 servers (100% missing) |
| `Fan_speed4` | Cooling (S4) | 83,763 | 3.048% | Concentrated on 5 servers (100% missing) |

*Excluded Servers in Dataset 1 for S4 experiments:*
- `qh2-rcc196`, `qh2-rcc197`, `qh2-rcc198`, `qh2-rcc199`, `qh2-rcc200` (5 servers, representing all 83,763 missing fan/inlet records). These servers lack inlet and fan telemetry hardware channels entirely in the raw trace. They are excluded from S4 experiments and included in S1/S2/S3.

### 4.2 Dataset 2 (Dell PowerEdge C6320, 1,280,640 total rows)
| Column Name | Category | Missing Count | Missing Percentage | Operational Handling |
|---|---|---|---|---|
| `CPU1_Temperature` | Target / State | 177,627 | 13.87% | Severe sensor outages on 4 servers |
| `CPU2_Temperature` | Target / State | 177,451 | 13.86% | Severe sensor outages on 4 servers |
| `CPU_Load` | Workload | 94,717 | 7.40% | Telemetry gaps |
| `Power` | Workload / Power | 177,208 | 13.84% | Sensor outage periods |
| `Network_RX` | Workload | 100,353 | 7.84% | Telemetry gaps |
| `Network_TX` | Workload | 100,331 | 7.83% | Telemetry gaps |
| `Ram` | Workload | 202,577 | 15.82% | Telemetry gaps |
| `Ram_Used` | Workload | 202,581 | 15.82% | Telemetry gaps |
| `No_Of_Running_vms`| Workload | 202,579 | 15.82% | Telemetry gaps |
| `CPU_cores` | Static capacity| 202,576 | 15.82% | Telemetry gaps |
| `CPU_cores_used` | Workload | 202,964 | 15.85% | Telemetry gaps |
| `Inlet_Temperature`| Telemetry | Structurally Absent | 100.0% | Channel does not exist on C6320 |
| `Fan_speed1..4` | Cooling | Structurally Absent | 100.0% | Channels do not exist on C6320 |

*Excluded Servers in Dataset 2 for all v2 experiments:*
- `qh2-rcc263`: 100.0% missing CPU temperature (dead sensor).
- `qh2-rcc264`: 100.0% missing CPU temperature (dead sensor).
- `qh2-rcc226`: 79.45% missing CPU temperature.
- `qh2-rcc205`: 41.68% missing CPU temperature.
Removing these 4 servers leaves **60 qualifying servers** for Dataset 2.

---

## 5. Final Analysis Server Sets per Experiment

Every server list below is stored as an exact CSV under `docs/snapshot/server_sets/`:

| Server Set CSV File | Experiment / Protocol Context | Server Count | Description & Exclusion Logic |
|---|---|---|---|
| `dataset1_all_servers.csv` | Full Dataset 1 interim inventory | 81 | All servers in raw CSV (`qh2-rcc120` to `qh2-rcc200`). |
| `dataset1_s4_servers.csv` | Dataset 1 S4 full-telemetry evaluation | 76 | Excludes 5 hosts (`rcc196–rcc200`) missing fan/inlet channels. |
| `dataset1_no_fan_servers.csv`| Excluded D1 hosts | 5 | `qh2-rcc196`, `qh2-rcc197`, `qh2-rcc198`, `qh2-rcc199`, `qh2-rcc200`. |
| `dataset1_e1_split_a_servers.csv`| E1 Split A Unified Ablation | 76 | 76 qualifying hosts evaluated on identical test rows. |
| `dataset1_split_c_test_servers.csv`| E1 Split C & E5 Conformal Test Set | 15 | 15 stratified held-out hosts evaluated in later calendar window. |
| `dataset1_e4_target_servers.csv` | E4 Cold-Start Adaptation Target Hosts | 16 | Fold 0 held-out hosts from 5-fold GroupKFold. |
| `dataset2_all_servers.csv` | Full Dataset 2 interim inventory | 64 | All servers in raw CSV (`qh2-rcc201` to `qh2-rcc264`). |
| `dataset2_e1_split_a_servers.csv`| Dataset 2 E1 Unified Ablation | 60 | Excludes 4 servers with severe temperature outages (`rcc263, 264, 226, 205`). |
| `dataset2_split_c_test_servers.csv`| Dataset 2 Split C Test Set | 12 | 12 stratified held-out hosts evaluated in test window. |
| `dataset2_e3_cross_cluster_servers.csv`| E3 Cross-Cluster Target Fleet | 60 | 60 qualifying Cluster 2 servers evaluated for C1 $\rightarrow$ C2 transfer. |

### Explanation of Discrepancies Across Experiments
1. **Dataset 1 (81 vs 76 vs 16 vs 15):**
   - 81 is the raw hardware inventory.
   - 76 servers have physical fan tachometer and inlet thermistor instrumentation. When running S4 or comparing S1-S4 fairly, 76 servers are used.
   - 16 servers were selected for E4 cold-start adaptation because Fold 0 of 5-Fold GroupKFold assigns `76 / 5 = 15.2` (rounded to 16 servers) to Fold 0.
   - 15 servers were selected for Split C because `split_strict(seed=42)` samples `int(76 * 0.20) = 15` stratified holdout servers.
2. **Dataset 2 (64 vs 60 vs 12):**
   - 64 is the raw hardware inventory.
   - 60 servers qualify after excluding dead/severely broken sensors (`rcc263, 264, 226, 205`).
   - 12 servers are selected for Split C because `split_strict(seed=42)` samples `int(60 * 0.20) = 12` holdout servers.
   - Legacy text in earlier draft reports erroneously stated "59 servers" for D2 Split A; re-running the exact code confirms `df["Series"].isin(qual_60)` contains exactly 60 unique hosts.
