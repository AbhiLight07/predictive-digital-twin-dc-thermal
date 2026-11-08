# Predictive Digital Twin for Data Centre Thermal Management

**Generalisation, Limited-Data Adaptation and Uncertainty-Aware Server Temperature Prediction**

[![SDG 9](https://img.shields.io/badge/SDG%209-Industry%2C%20Innovation%20%26%20Infrastructure-orange)](https://sdgs.un.org/goals/goal9)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-TBD-lightgrey)]()

---

## Problem Statement

> This project investigates how reliably server-temperature prediction models generalise to unseen servers and unfamiliar hardware, how much local data is needed to adapt a pretrained model to a new server, and whether prediction intervals can flag unreliable predictions. The models are integrated into a lightweight, offline, data-driven predictive digital twin for server-level thermal monitoring.

## Research Questions

- **RQ1.** How does prediction performance change from chronological testing (seen servers) to unseen-server testing, and how much does it depend on which sensors are used?
- **RQ2.** Can a model trained on existing servers predict a new server's temperature with zero local data?
- **RQ3.** How much local data from a new server is needed to adapt a pretrained model, and does adaptation beat training from scratch?
- **RQ4.** Are prediction intervals reliable on unseen servers, and can a few hours of local data recalibrate them?
- **RQ5 (conditional).** Does a model transfer across hardware clusters using only shared, physically comparable features?

## Quick Start

### Prerequisites
- Python 3.10 or later
- The primary dataset files (see [data/README.md](data/README.md) for download instructions)

### Setup

```bash
# Create and activate virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data

Download the dataset files from [Zenodo (10.5281/zenodo.10069402)](https://zenodo.org/records/10069402):
- `dataset1_combined.csv` → place in `data/raw/`
- `dataset2_combined.csv` → place in `data/raw/`

See [data/README.md](data/README.md) for full details, attribution requirements, and checksums.

### Run the Data Audit

```bash
python scripts/data_audit.py data/raw/dataset1_combined.csv
python scripts/data_audit.py data/raw/dataset2_combined.csv
```

### Run Tests

```bash
pytest tests/ -v
```

### Run the Digital Twin Demo

```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```
├── README.md                   This file
├── PROJECT_BRIEF.md            Full project brief
├── requirements.txt            Python dependencies
├── .gitignore                  Git ignore rules
├── configs/                    Configuration files
│   ├── columns.yaml            Dataset column mappings
│   ├── protocol.yaml           Evaluation protocol (frozen after Phase 2)
│   └── experiments/            Experiment config files
├── data/                       (git-ignored) Dataset files
│   ├── raw/                    Original downloads
│   ├── interim/                Intermediate processing
│   ├── processed/              Feature-engineered data
│   └── README.md               Download instructions and attribution
├── src/ttwin/                  Core library
│   ├── io.py                   Data I/O
│   ├── clean.py                Data cleaning
│   ├── features.py             Feature engineering
│   ├── splits.py               Train/val/test split logic
│   ├── metrics.py              Evaluation metrics
│   ├── models.py               Model definitions
│   ├── adapt.py                Cold-start adaptation
│   ├── uncertainty.py          Uncertainty quantification
│   └── eval.py                 Experiment evaluation
├── scripts/                    Runnable scripts
│   ├── data_audit.py           Dataset audit (Phase 1)
│   ├── make_parquet.py         CSV → Parquet conversion
│   └── run_experiment.py       Experiment runner
├── experiments/                Experiment results (small files only)
├── app/                        Streamlit digital twin
│   └── streamlit_app.py
├── docs/                       Documentation
│   ├── literature.md           Literature review table
│   ├── protocol.md             Evaluation protocol description
│   ├── decisions.md            Decision log
│   ├── limitations.md          Known limitations
│   ├── data_audit_report.md    Data audit findings
│   └── paper/                  Paper draft
├── tests/                      Test suite
│   ├── test_splits_no_leakage.py
│   ├── test_features_no_future.py
│   └── test_metrics.py
├── notebooks/                  EDA notebooks (results must come from scripts)
└── logger/                     (Phase 5) College server data collection
```

## Dataset Attribution

This project uses the University of Melbourne research-cloud trace dataset:

> S. Ilager, A. N. Toosi, J. Jha, S. Brandic, R. Buyya, "A Data-driven Analysis of a Cloud Data Center: Statistical Characterization of Workload, Energy and Temperature", Proceedings of the 16th IEEE/ACM International Conference on Utility and Cloud Computing (UCC 2023), Taormina, Italy, December 4–7, 2023.

Dataset available at: [Zenodo 10.5281/zenodo.10069402](https://zenodo.org/records/10069402) under CC BY 4.0.

## SDG 9 — Industry, Innovation and Infrastructure

Energy-efficient, resilient digital infrastructure. Servers that run hot raise cooling cost and failure risk; reliable thermal prediction supports efficient and resilient data centres.

## Team

RVCE (Bengaluru) undergraduate research team — Experiential Learning project, Semester 5.

## Terminology

This project builds an **offline, data-driven predictive digital twin prototype** ("digital shadow with predictive capability"). It is not a production, real-time, or bidirectional digital twin.
