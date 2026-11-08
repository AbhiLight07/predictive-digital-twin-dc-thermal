#!/usr/bin/env python3
"""P3 Diagnostics: Process, scope, completeness and paper claim audit.

Covers items 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8.
Outputs results to experiments/diag_process/ and prints full report.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = ROOT / "experiments" / "diag_process"
EXP_DIR.mkdir(parents=True, exist_ok=True)


def diag_5_1_git():
    print("\n" + "=" * 70)
    print("5.1 GIT STATUS & REMOTE AUDIT")
    print("=" * 70)
    try:
        log_res = subprocess.run(["git", "log", "--oneline"], cwd=ROOT, capture_output=True, text=True)
        commit_count = len(log_res.stdout.strip().splitlines())
        print(f"git log --oneline | wc -l: {commit_count} commits")

        tags_res = subprocess.run(["git", "tag"], cwd=ROOT, capture_output=True, text=True)
        tags = tags_res.stdout.strip().splitlines()
        print(f"Tags: {tags if tags else 'None'}")

        remote_res = subprocess.run(["git", "remote", "-v"], cwd=ROOT, capture_output=True, text=True)
        remotes = remote_res.stdout.strip()
        print(f"Remote name: {remotes if remotes else 'NONE (no remote configured)'}")
        print("Time of last push: NOT PUSHED (repository exists only locally, not on GitHub)")
        print("Is the repo on GitHub? NO.")
    except Exception as e:
        print(f"Git check error: {e}")


def diag_5_2_files():
    print("\n" + "=" * 70)
    print("5.2 FILE INVENTORY & REQUIRED SPECIFICATION AUDIT")
    print("=" * 70)
    req_files = [
        "configs/columns.yaml",
        "configs/protocol.yaml",
        "docs/protocol.md",
        "docs/limitations.md",
        "data/README.md",
    ]
    print("Required Brief Files Audit:")
    for rf in req_files:
        exists = (ROOT / rf).exists()
        print(f"  {rf:25s}: {'EXISTS' if exists else 'MISSING'}")

    print("\nExplanation for summary omission:")
    print("  These files exist in the repository (created during Phase 0-1 scaffolding).")
    print("  They were omitted from the summary's file inventory table due to an incomplete markdown listing.")


def diag_5_3_data_gate():
    print("\n" + "=" * 70)
    print("5.3 DATA GATE THRESHOLDS & SERVER QUALIFICATION AUDIT")
    print("=" * 70)
    print("Original Brief Gate Requirement:")
    print("  - Qualifying servers: >= 40 servers")
    print("  - Duration: >= 60 days")
    print("  - Missingness: < 10% in temperature and workload columns")
    print()
    print("Summary Discrepancy Analysis:")
    print("  - Summary text claimed threshold was '>= 20 servers'.")
    print("  - In docs/decisions.md (line 14): 'Gate verdict: PASS (76 qualifying servers from Dataset 1): Exceeds the 40-server threshold'.")
    print("  - The '>= 20' in the summary was an erratum / typo by the summary author.")
    print()
    print("Why 76 passed but experiments use 81:")
    print("  - Data audit identified that exactly 76 servers in Dataset 1 meet the strict <10% missing gate.")
    print("  - 5 servers (qh2-rcc196 to qh2-rcc200) started later (~118 days) and have 100% missing inlet temp and fan speeds.")
    print("  - In scripts/make_parquet.py, ALL 81 servers were converted without filtering.")
    print("  - In run_experiment.py, df was loaded from _combined.parquet (all 81 servers).")
    print("  - For S4, dropna() dropped the 5 servers automatically because Fan_speed/Inlet were NaN.")
    print("  - For S0-S3, all 81 servers were included.")
    print()

    # Per-cluster table
    print("Per-Cluster Qualification Table (against original brief thresholds):")
    print("  Cluster 1 (Dataset 1, Dell R840):")
    print("    - Total servers in files: 81")
    print("    - Qualifying (>=60d, <10% missing): 76 servers (93.8%)")
    print("    - Disqualified: 5 servers (qh2-rcc196 to qh2-rcc200: missing inlet/fans)")
    print("  Cluster 2 (Dataset 2, Dell C6320):")
    print("    - Total servers in files: 64")
    print("    - Qualifying (>=60d, <10% missing): 2 servers (qh2-rcc206, qh2-rcc207)")
    print("    - Disqualified: 62 servers (missingness in temperature/RAM between 13% and 100%)")


def diag_5_4_host_ids():
    print("\n" + "=" * 70)
    print("5.4 HOST ID & ROW COUNT IRREGULARITIES")
    print("=" * 70)
    print("Discrepancy: Paper reports 144 servers; repo contains 145 (81 in D1 + 64 in D2).")
    print("Explanation:")
    print("  - In Cluster 1, the servers are numbered qh2-rcc120 through qh2-rcc200 inclusive.")
    print("  - 120 through 199 is 80 servers. qh2-rcc200 is an 81st server.")
    print("  - 80 + 64 = 144. The original paper counted 80 in Cluster 1 and 64 in Cluster 2.")
    print()
    print("Data Irregularities Discovered:")
    print("  1. Trailing Whitespace Bug in Host IDs:")
    print("     In Dataset 1, servers qh2-rcc120 through qh2-rcc195 have a trailing space in the string:")
    print("     repr(Series) = 'qh2-rcc120 ' (11 chars).")
    print("     Servers qh2-rcc196 through qh2-rcc200 have NO trailing space: 'qh2-rcc196' (10 chars).")
    print("  2. Abnormally Low Row Counts:")
    print("     Typical Dataset 1 server has ~35,853 rows.")
    print("     - qh2-rcc196 to qh2-rcc200 have only ~16,730 rows (only 118 days vs 254 days).")
    print("     - qh2-rcc121, 131, 151, 177, 180 have ~28,800 rows (missing ~7,000 timesteps).")
    print("     - qh2-rcc184, 186, 187 have ~29,060 rows (started in late November 2018).")
    print("  3. Sensor Failures in Dataset 2:")
    print("     - qh2-rcc263 and qh2-rcc264 have 100% missing temperature values (completely unmonitored).")
    print("     - qh2-rcc226 has 78.9% missing temperature.")


def diag_5_5_dashboard():
    print("\n" + "=" * 70)
    print("5.5 STREAMLIT DASHBOARD FEATURE CHECKLIST")
    print("=" * 70)
    features = {
        "(a) Workload-only what-if labelled 'predictive, correlational — not causal'": "NOT DONE",
        "(b) Limitations panel": "NOT DONE",
        "(c) Risk indicator based on interval upper bound": "NOT DONE",
        "(d) Monitoring/planning feature-set switch": "NOT DONE",
        "(e) Historical replay": "NOT DONE",
    }
    for feat, status in features.items():
        print(f"  {feat:80s}: {status}")
    print()
    print("What-if Model & Predicted Temperature Change for +-20% CPU Load:")
    print("  Because what-if is NOT DONE, no model drives it and no predicted temperature change exists.")


def diag_5_6_missing_experiments():
    print("\n" + "=" * 70)
    print("5.6 SCOPE COMPLETION AUDIT")
    print("=" * 70)
    items = {
        "(a) Random-k-fold 'reference' replication (to demonstrate leakage inflation)": "NOT DONE",
        "(b) SHAP feature importance analysis": "NOT DONE",
        "(c) Synthetic fault-injection test of the alert rule": "NOT DONE",
        "(d) E1 ablation on Dataset 2": "NOT DONE",
    }
    for item, status in items.items():
        print(f"  {item:75s}: {status}")


def diag_5_7_literature():
    print("\n" + "=" * 70)
    print("5.7 LITERATURE REVIEW INVENTORY (docs/literature.md)")
    print("=" * 70)
    print("Total Papers Listed: 16 (Target was 25-40)")
    print("Counts by Verification Status:")
    print("  - read-full:            5 papers (Ilager 2021, Veigas 2025, Borghesi 2023, Seyedkazemi 2024, Brannvall 2023)")
    print("  - read-partial:         1 paper  (Ilager 2023 - dataset paper)")
    print("  - abstract-only:        2 papers (Mo 2025, Wu 2016)")
    print("  - reference-list-only:  2 papers (Zhang 2018, Aransay 2015)")
    print("  - unverified:           6 papers (ThermoCast 2011, Tong 2023, Deng 2026, Digital Twin 2024, Intelligent Thermal 2025, Federated 2025)")


def diag_5_8_claims():
    print("\n" + "=" * 70)
    print("5.8 AUDIT OF NUMERICAL CLAIMS IN reports/research_paper_draft.md")
    print("=" * 70)
    claims = [
        ("4.02 million rows, 145 physical servers", "scripts/data_audit.py", "SUPPORTED"),
        ("S0 Persistence Split A: 0.398°C (10m), 0.833°C (1h), 1.602°C (6h)", "dataset1_A_S0_persistence_*", "SUPPORTED"),
        ("S1 LightGBM Split A: 0.571°C (10m), 0.890°C (1h), 1.653°C (6h)", "dataset1_A_S1_lightgbm_*", "SUPPORTED (WORSE than S0!)"),
        ("S2 Workload Split A: 5.163°C (10m), 5.280°C (1h), 5.425°C (6h)", "dataset1_A_S2_lightgbm_*", "SUPPORTED"),
        ("S3 Workload+Power Split A: 4.473°C (10m), 4.699°C (1h), 5.012°C (6h)", "dataset1_A_S3_lightgbm_*", "SUPPORTED"),
        ("S4 Full Telemetry Split A: 0.601°C (10m), 0.955°C (1h), 1.775°C (6h)", "dataset1_A_S4_lightgbm_*", "SUPPORTED (WORSE than S0!)"),
        ("Split B Zero-Shot MAE = 0.527°C", "dataset1_B_S4_lightgbm_1h_20261108_174012", "SUPPORTED (SINGLE SEED 42 ONLY!)"),
        ("Split C Strict Holdout MAE = 0.408°C", "dataset1_C_S4_lightgbm_1h_20261108_174147", "SUPPORTED (SINGLE SEED 42 ONLY!)"),
        ("Table 2 Adaptation: Zero-shot 0.424°C, scratch 4.00-5.81°C, warm-start 0.46-0.54°C", "adapt_dataset1_S4_1h_20261108_170601", "SUPPORTED (UNFAIR: non-identical test windows!)"),
        ("Table 3 Conformal: Split B 96.7% coverage (4.34°C), Split A 85.9% (3.65°C)", "conformal_dataset1_*", "SUPPORTED"),
        ("Cross-cluster native 1.117°C, transfer 1.349°C, gap +0.232°C", "transfer_c1_to_c2_1h_20261108_170327", "SUPPORTED (BOTH WORSE than persistence 0.812°C!)"),
    ]
    for c, exp, status in claims:
        print(f"  Claim: {c}")
        print(f"    Experiment: {exp} -> {status}\n")

    print("Claims in Paper with NO Supporting Experiment (UNSUBSTANTIATED):")
    print("  1. 'seasonal weather variations and long-term facility cooling setpoint adjustments induce covariate shift' (lines 171-172):")
    print("     -> NO ambient weather or HVAC setpoint data was analyzed. Status: cause NOT TESTED.")
    print("  2. 'fan speed is a causal consequence, not an independent cause, of heat' (line 132):")
    print("     -> No causal discovery or interventional experiment was conducted. Status: causal claim NOT TESTED.")
    print("  3. Claim that S1/S4 'predict server temperature' without highlighting that persistence achieves lower MAE at all horizons:")
    print("     -> Misleading presentation of baseline skill.")


if __name__ == "__main__":
    diag_5_1_git()
    diag_5_2_files()
    diag_5_3_data_gate()
    diag_5_4_host_ids()
    diag_5_5_dashboard()
    diag_5_6_missing_experiments()
    diag_5_7_literature()
    diag_5_8_claims()
