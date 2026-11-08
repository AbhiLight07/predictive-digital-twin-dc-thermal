# Literature Review

| # | Citation | What it did | Relevance to our work | Verification status |
|---|----------|-------------|----------------------|--------------------|
| 1 | Ilager, Ramamohanarao, Buyya — "Thermal Prediction for Efficient Energy Management of Clouds using ML", IEEE TPDS 2021 (arXiv 2011.03649) | Per-host XGBoost for CPU temperature; 10-fold CV (no time-ordered splits mentioned); inputs include fan speed (possible circularity); near-idle hosts produced extreme predictions patched with bounds; thermal-aware scheduling | Primary baseline paper; our evaluation protocol addresses its CV setup; fan-speed circularity to be tested in ablation | `read-full` |
| 2 | Ilager, Toosi, Jha, Brandic, Buyya — "A Data-driven Analysis of a Cloud Data Center", UCC 2023 | Dataset paper for our primary data; random 70/30 splits; per-cluster models; next-interval CPU load and power prediction | Provides our primary dataset; evaluation section to be re-examined | `read-partial` |
| 3 | Veigas, Chinnici et al. — "Towards Energy Efficiency of HPC Data Centers", Electronics 2025, 14(16), 3170 | CRESCO7: 144 nodes, LightGBM, Streamlit dashboard, per-node models, what-if sliders on fan speed and supply air | Dashboard inspiration; does not measure data needed for adaptation; fan-speed what-if is causally unsound | `read-full` |
| 4 | Borghesi et al. — "M100 ExaData", Scientific Data 10:288 (2023) | 980+ nodes, 2.5 years, 49.9 TB; monthly Zenodo entries in partitioned Parquet | Optional scale-up dataset for unseen-node replication | `read-full` |
| 5 | Seyedkazemi Ardebili et al. — "HazardNet", FGCS 155 (2024) 340–353 | Room/rack-level hazard prediction on Marconi A2 data | Related thermal hazard work; different granularity (room/rack vs server) | `read-full` |
| 6 | Brännvall, Gustafsson, Sandin — "Modular and Transferable ML for Heat Management", Energies 2023, 16(5), 2255 | Fine-tuning pretrained physics-informed RNN on ~5h beat training from scratch (lab, 6→1 server) | Closest prior work on pretrained-beats-scratch; we must not claim that finding as new | `read-full` |
| 7 | Mo et al. — cross-workload power prediction with transfer GPR, IEEE Trans. Cloud Computing 13(3), 2025 | Transfer Gaussian process regression for power prediction across workloads | Related transfer learning approach for DC telemetry | `abstract-only` |
| 8 | Wu et al. — "VM Level Temperature Profiling and Prediction", ICDCS 2016 | VM-level temperature prediction in cloud datacenters | Earlier work on temperature prediction at finer granularity | `abstract-only` |

### Unverified — must verify before citing

| Citation | Notes | Status |
|----------|-------|--------|
| Zhang et al., IEEE TPDS 2018 | ML temperature prediction for runtime thermal management | `reference-list-only` |
| Aransay et al., IEEE CLOUD 2015 | Self-organising maps for thermal anomalies | `reference-list-only` |
| ThermoCast (2011) | | `unverified` |
| Tong et al. (2023) | Time-varying state-space rack temperature prediction | `unverified` |
| Deng et al. (2026) | CFD + ML for 180-rack server room | `unverified` |
| Digital Twin-Driven Energy Consumption Management... (2024) | Integrated heat pipe cooling | `unverified` |
| Intelligent Thermal Management Strategy... (2025) | Digital twin technology | `unverified` |
| Federated Transfer Learning for Anomaly Detection in HPC (2025/26) | | `unverified` |

### Search topics still to cover
- Server/rack temperature forecasting
- Thermal anomaly and hazard prediction
- Cross-server / cross-cluster generalisation
- Domain adaptation and transfer learning for regression
- Few-shot / cold-start time-series
- Conformal prediction and quantile regression under distribution shift
- Digital-twin thermal management
- Leakage and evaluation pitfalls in time-series ML
