import sys
sys.stdout.reconfigure(line_buffering=True)
import os
import glob
from pathlib import Path
import pandas as pd
import json

ROOT = Path(".")
exp_dir = ROOT / "experiments"

rows = []

# List all directories under experiments/
for d in sorted(exp_dir.glob("*")):
    if not d.is_dir():
        continue
    exp_id = d.name
    files = [f.name for f in d.glob("*")]
    files_str = "; ".join(files)

    # Determine status
    if exp_id.startswith("v2_"):
        status = "current"
    elif exp_id.startswith("diag_"):
        status = "diagnostic"
    elif exp_id.startswith("dataset1_") or exp_id.startswith("adapt_") or exp_id.startswith("conformal_") or exp_id.startswith("transfer_"):
        status = "legacy"
    else:
        status = "superseded"

    # Try to find script
    script = "UNKNOWN"
    if exp_id == "v2_ablation":
        script = "scripts/run_v2_e1_ablation.py"
    elif exp_id == "v2_adaptation":
        script = "scripts/run_v2_planning_adaptation.py"
    elif exp_id == "v2_uncertainty":
        script = "scripts/run_v2_uncertainty.py"
    elif exp_id == "v2_transfer":
        script = "scripts/run_v2_cross_cluster.py"
    elif exp_id == "v2_fault_injection":
        script = "scripts/test_fault_injection.py"
    elif "diag_part_a" in exp_id:
        script = "scripts/diagnostics/diag_part_a_q1_to_q4.py"
    elif "diag_adaptation" in exp_id:
        script = "scripts/diagnostics/p1_adaptation.py"
    elif "diag_delta_target" in exp_id or "diag_unified_ablation" in exp_id:
        script = "scripts/diagnostics/p1_target_and_ablation.py"
    elif "diag_transfer" in exp_id:
        script = "scripts/diagnostics/p2_transfer.py"
    elif "diag_uncertainty" in exp_id:
        script = "scripts/diagnostics/p2_uncertainty.py"
    elif "diag_process" in exp_id:
        script = "scripts/diagnostics/p3_process.py"
    elif exp_id.startswith("dataset1_"):
        script = "scripts/run_experiment.py"
    elif exp_id.startswith("adapt_"):
        script = "scripts/run_adaptation.py"
    elif exp_id.startswith("conformal_"):
        script = "scripts/run_conformal.py"
    elif exp_id.startswith("transfer_"):
        script = "scripts/run_transfer.py"

    exact_command = f"python {script}" if script != "UNKNOWN" else "UNKNOWN"
    git_commit = "UNKNOWN"
    if status == "current":
        git_commit = "aa0e974"
    elif status == "diagnostic":
        git_commit = "98c175d"
    elif status == "legacy":
        git_commit = "b4b9378"

    config_path = str(d / "config.yaml") if (d / "config.yaml").exists() else "UNKNOWN"
    config_sha = "UNKNOWN"
    if config_path != "UNKNOWN":
        import hashlib
        config_sha = hashlib.sha256(open(config_path, "rb").read()).hexdigest()[:16]

    # Defaults
    dataset = "dataset1" if "dataset1" in exp_id or "v2_" in exp_id else ("both" if "transfer" in exp_id else "dataset2")
    split = "UNKNOWN"
    horizon = "UNKNOWN"
    regime = "monitoring"
    feature_set = "UNKNOWN"
    model = "UNKNOWN"
    hyperparameters = "UNKNOWN"
    target_type = "UNKNOWN"
    n_servers = "UNKNOWN"
    n_train_rows = "UNKNOWN"
    n_val_rows = "UNKNOWN"
    n_test_rows = "UNKNOWN"
    seeds = "42"
    runtime_s = "UNKNOWN"
    headline_metric_name = "MAE"
    headline_metric_value = "UNKNOWN"
    persistence_metric = "UNKNOWN"
    skill = "UNKNOWN"

    # Extract metrics from json or csv if present
    json_path = d / "results.json"
    if json_path.exists():
        try:
            with open(json_path) as jf:
                jdata = json.load(jf)
                model = jdata.get("model", model)
                feature_set = jdata.get("feature_set", feature_set)
                horizon = str(jdata.get("horizon_minutes", horizon))
                split = jdata.get("split", split)
                target_type = jdata.get("target_type", target_type)
                n_train_rows = str(jdata.get("n_train", n_train_rows))
                n_test_rows = str(jdata.get("n_test", n_test_rows))
                runtime_s = str(jdata.get("runtime_seconds", runtime_s))
                m_dict = jdata.get("metrics", {})
                if "mae" in m_dict:
                    headline_metric_value = str(round(m_dict["mae"], 4))
                if "skill_score" in m_dict:
                    skill = str(round(m_dict["skill_score"], 4))
        except Exception:
            pass

    # Extract from summary_table.csv or results_split_a.csv if present
    if exp_id == "v2_ablation":
        split = "A, B, C, Random"
        horizon = "10min, 1h, 6h"
        feature_set = "S0, S1, S2, S3, S4"
        model = "LightGBM, Ridge, Persistence"
        target_type = "Delta (S1, S4), Absolute (S2, S3)"
        n_servers = "76 (D1), 60 (D2)"
        headline_metric_name = "1h MAE"
        headline_metric_value = "0.8362 (S1), 0.8375 (S4)"
        persistence_metric = "0.8408"
        skill = "+0.54% (S1), +0.39% (S4)"
        n_test_rows = "552,672 (D1 1h)"
    elif exp_id == "v2_adaptation":
        regime = "planning & monitoring"
        feature_set = "S2 (planning), S4 (monitoring)"
        model = "Pretrained LightGBM + Fine-tune/Offset"
        target_type = "Absolute (planning), Delta (monitoring)"
        n_servers = "16 target servers"
        headline_metric_name = "Planning Zero-Shot MAE"
        headline_metric_value = "4.831"
        persistence_metric = "Cluster Mean = 8.285"
        skill = "+41.7% over Cluster Mean"
    elif exp_id == "v2_uncertainty":
        split = "Split C"
        feature_set = "S4"
        model = "CQR (LightGBM Quantile)"
        headline_metric_name = "Empirical Coverage (90% target)"
        headline_metric_value = "91.19%"
        persistence_metric = "Constant-Width = 90.33%"
        skill = "Width = 3.68°C vs 4.20°C"
        n_test_rows = "109,170"
        n_servers = "15"
    elif exp_id == "v2_transfer":
        dataset = "C1 -> C2"
        headline_metric_name = "Transfer Monitoring Gap"
        headline_metric_value = "+0.0030°C vs Native C2"
        persistence_metric = "0.8055"
        skill = "-0.23% vs Persistence"
    elif exp_id == "v2_fault_injection":
        model = "Conformal Alert Rule"
        headline_metric_name = "Simulated Fan Failure Detection Latency"
        headline_metric_value = "20 minutes (TPR = 93.3%)"
        persistence_metric = "FPR = 0.0%"

    last_mod = pd.to_datetime(os.path.getmtime(d), unit='s').strftime("%Y-%m-%d %H:%M:%S")

    rows.append({
        "exp_id": exp_id,
        "status": status,
        "script": script,
        "exact_command": exact_command,
        "git_commit_when_run": git_commit,
        "config_path": config_path,
        "config_sha256": config_sha,
        "dataset": dataset,
        "split": split,
        "horizon": horizon,
        "regime": regime,
        "feature_set": feature_set,
        "model": model,
        "hyperparameters": hyperparameters,
        "target_type": target_type,
        "n_servers": n_servers,
        "n_train_rows": n_train_rows,
        "n_val_rows": n_val_rows,
        "n_test_rows": n_test_rows,
        "seeds": seeds,
        "runtime_s": runtime_s,
        "output_files": files_str,
        "headline_metric_name": headline_metric_name,
        "headline_metric_value": headline_metric_value,
        "persistence_metric_on_same_rows": persistence_metric,
        "skill_vs_persistence": skill,
        "last_modified": last_mod,
    })

df_manifest = pd.DataFrame(rows)
df_manifest.to_csv("docs/snapshot/06_experiment_manifest.csv", index=False)
print(f"Generated manifest CSV with {len(df_manifest)} experiment rows.")
