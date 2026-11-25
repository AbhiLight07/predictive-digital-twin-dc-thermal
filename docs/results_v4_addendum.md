# Results v4 Addendum — Planning Baselines & Cross-Cluster Reconciliation

> Addendum to `docs/results_v3.md` (`scripts/run_v4_planning_addendum.py`). Existing v3 files and `experiments/v3_experiments/` are untouched.

## 1. Fleet-Wide Planning Adaptation with Load-Aware Baselines ($N=81$ Servers)

Evaluated on the exact same 81 Dataset 1 servers, 5 GroupKFold folds (`seed=42`), S2 features, and post-174h evaluation windows as `docs/results_v3.md` §4, adding two zero-shot load-aware baselines fit on each fold's training servers:
- `ridge_cpu_load`: Ridge regression (`alpha=1.0`) on `CPU_Load` alone.
- `load_bin_mean`: mean training temperature per 10%-wide `CPU_Load` bin (`[0,10), ..., [90,100]`).

| Strategy | Budget (h) | n_servers | Mean MAE (°C) | Std (°C) | 95% Bootstrap CI (°C) |
|---|---:|---:|---:|---:|---|
| `zero_shot` | 0 | 81 | 5.0347 | 1.5905 | [4.7103, 5.3950] |
| `load_bin_mean` | 0 | 81 | 5.6658 | 2.1411 | [5.2143, 6.1260] |
| `ridge_cpu_load` | 0 | 81 | 5.7000 | 2.1520 | [5.2395, 6.1706] |
| `fleet_mean` | 0 | 81 | 8.8659 | 2.9927 | [8.2264, 9.4982] |
| `continued_boosting` | 1 | 81 | 5.0347 | 1.5905 | [4.7103, 5.3950] |
| `residual_offset` | 1 | 81 | 5.5193 | 2.5000 | [5.0128, 6.0842] |
| `continued_boosting` | 6 | 81 | 5.0347 | 1.5905 | [4.7103, 5.3950] |
| `residual_offset` | 6 | 81 | 5.5365 | 2.5913 | [5.0298, 6.1122] |
| `continued_boosting` | 24 | 81 | 4.9311 | 1.5200 | [4.6220, 5.2803] |
| `residual_offset` | 24 | 81 | 5.4673 | 2.3372 | [4.9917, 5.9844] |
| `continued_boosting` | 72 | 81 | 4.9181 | 1.5090 | [4.6138, 5.2615] |
| `residual_offset` | 72 | 81 | 5.3250 | 2.2496 | [4.8761, 5.8424] |
| `continued_boosting` | 168 | 81 | 4.8717 | 1.4326 | [4.5807, 5.1921] |
| `residual_offset` | 168 | 81 | 5.0168 | 2.2149 | [4.5517, 5.5184] |

### Paired Zero-Shot (`S2` LightGBM) Gain Over Baselines ($N=81$ Servers)

| Baseline | Overall Pooled Reduction [95% CI] | Mean Per-Server Reduction [95% CI] |
|---|---|---|
| `fleet_mean` (8.87°C) | **+43.21%** [+38.57%, +47.30%] | +35.85% [+27.76%, +42.20%] |
| `ridge_cpu_load` (5.70°C) | **+11.67%** [+5.37%, +17.56%] | +5.80% [-0.71%, +11.67%] |
| `load_bin_mean` (5.67°C) | **+11.14%** [+4.24%, +17.02%] | +5.01% [-1.45%, +10.77%] |

**Impact on the "43.2% vs fleet mean" claim:** Against an unconditioned `fleet_mean` (8.87°C), zero-shot `S2` LightGBM (5.03°C) reduces error by **43.2%** (95% CI `[38.6%, 47.3%]`). However, simple load-aware baselines already capture most of that reduction (`ridge_cpu_load`: **5.70°C**; `load_bin_mean`: **5.67°C**), leaving S2 LightGBM with a modest **+11.67%** gain over Ridge on CPU load and **+11.14%** over 10%-binned load mean.

---

## 2. Cross-Cluster Adaptation (`C1 -> C2`): Server Count & 6.85°C vs 4.88°C Reconciliation

In the v3 cross-cluster adaptation experiment (`docs/results_v3.md` §5), **`N = 12` target servers** in Cluster 2 met the Split C test holdout and post-174h evaluation criteria (`>= 150` rows and `>= 30` post-174h rows):

| Budget (h) | Strategy | n_servers | Mean MAE (°C) | 95% CI (°C) |
|---:|---|---:|---:|---|
| 0 | `zero_shot` | 12 | 6.8518 | [2.9162, 13.4668] |
| 1 | `residual_offset` | 12 | 3.7883 | [2.4026, 5.2735] |
| 1 | `continued_boosting` | 12 | 6.8518 | [2.9162, 13.4668] |
| 6 | `residual_offset` | 12 | 3.1606 | [2.1302, 4.3042] |
| 6 | `continued_boosting` | 12 | 6.8518 | [2.9162, 13.4668] |
| 24 | `residual_offset` | 12 | 3.1687 | [1.9008, 4.8095] |
| 24 | `continued_boosting` | 12 | 6.0719 | [2.6914, 11.7068] |
| 72 | `residual_offset` | 12 | 2.9219 | [1.6388, 4.8064] |
| 72 | `continued_boosting` | 12 | 6.0997 | [2.7206, 11.7339] |
| 168 | `residual_offset` | 12 | 3.0332 | [1.7042, 4.7507] |
| 168 | `continued_boosting` | 12 | 6.0500 | [2.6967, 11.7058] |

### Reconciling Zero-Shot MAE: `6.85°C` (v3) vs `4.97°C` (v2)

| Protocol Variant | C2 Servers | Eval Rows | Zero-Shot MAE (°C) | Notes |
|---|---:|---:|---:|---|
| v2 exact (Split A chron, 70% C1 train + early stop, pooled rows) | 59 | 215,984 | **4.9697** | unweighted server mean = 5.1786 C |
| v3 model (100% C1, 80 trees) on v2 Split A chron (pooled rows) | 59 | 215,984 | **5.4328** | Isolates C1 training set/rounds difference on identical C2 Split A rows |
| v2 model on v3 Split C strict held-out servers (unweighted server mean) | 12 | 34,985 | **6.6657** | Isolates C2 held-out server subset + post-174h time window |
| v3 exact (Split C strict, unweighted server mean) | 12 | 34,985 | **6.8518** | pooled row-weighted MAE on same Split C rows = 6.8525 C; hosts = qh2-rcc204, qh2-rcc206, qh2-rcc212, qh2-rcc220, qh2-rcc232, qh2-rcc243, qh2-rcc247, qh2-rcc248, qh2-rcc252, qh2-rcc254, qh2-rcc255, qh2-rcc256 |

**Why v3 (`6.85°C`) differs from v2 (`4.88°C`):**
1. **Evaluation Split & Server Subset:** v2 evaluated chronological Split A across all `53` qualifying C2 servers (`4.88°C` pooled), whereas v3 used Split C (`seed=42`) evaluating strictly held-out C2 servers (`N = 12`) after a 174-hour adaptation buffer.
2. **C1 Training Configuration & Aggregation:** v2 trained on 70% of C1 with early stopping (`best_iteration` trees), whereas v3 trained on 100% of C1 for 80 rounds and averaged unweighted per-server MAEs (giving high-variance held-out C2 hosts equal weight).

Source artifacts: `experiments/v4_planning_addendum/` (`planning_baselines_summary.csv`, `planning_paired_gains.csv`, `cross_cluster_server_counts.csv`, `cross_cluster_reconciliation.csv`).
