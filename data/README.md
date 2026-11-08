# Data Directory

**This directory is git-ignored.** Raw data files must be downloaded manually.

## Primary Dataset — University of Melbourne Research-Cloud Trace

- **Source:** Zenodo record [10.5281/zenodo.10069402](https://zenodo.org/records/10069402)
- **Paper:** Ilager, Toosi, Jha, Brandic, Buyya — "A Data-driven Analysis of a Cloud Data Center", UCC 2023
- **Licence:** CC BY 4.0 — attribute and cite the UCC 2023 paper
- **Files to download:**
  - `dataset1_combined.csv` (~613 MB) → place in `data/raw/`
  - `dataset2_combined.csv` (~184 MB) → place in `data/raw/`

### Attribution

This project uses data from:
> S. Ilager, A. N. Toosi, J. Jha, S. Brandic, R. Buyya, "A Data-driven Analysis of a Cloud Data Center: Statistical Characterization of Workload, Energy and Temperature", Proceedings of the 16th IEEE/ACM International Conference on Utility and Cloud Computing (UCC 2023), Taormina, Italy, December 4–7, 2023.

## Directory Structure

- `raw/` — Original downloaded files (never modified)
- `interim/` — Intermediate processing results (e.g., chunked Parquet)
- `processed/` — Final feature-engineered datasets ready for modelling

## Checksums

*(To be filled after download — run `certutil -hashfile <file> SHA256` on Windows)*
