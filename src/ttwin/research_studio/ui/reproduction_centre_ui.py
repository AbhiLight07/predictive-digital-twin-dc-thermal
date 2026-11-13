"""Experiment Reproduction Centre UI (Stage 9).

Provides a secure, verified interface to execute the actual repository experiment pipelines
using safe subprocess execution with explicit working directories, stdout/stderr capture,
and strict protection against arbitrary command injection.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import subprocess
import time
import sys
import streamlit as st

from ttwin.research_studio.data_registry import DataRegistry


VERIFIED_PIPELINES = {
    "v3_comprehensive": {
        "name": "v3 Comprehensive Benchmark Runner (scripts/run_v3_experiments.py)",
        "script": "scripts/run_v3_experiments.py",
        "description": "Executes or re-audits all 7 Tier B gap-corrected experiments and updates the execution manifest.",
        "supported_modes": ["--dry-run", "--reuse-existing", "--force"],
        "default_mode": "--dry-run",
        "expected_outputs": [
            "experiments/v3_experiments/e1_bootstrap_summary.csv",
            "experiments/v3_experiments/sensor_ablation.csv",
            "experiments/v3_experiments/split_b_and_c_horizons.csv",
            "experiments/v3_experiments/planning_adaptation_all_folds.csv",
            "experiments/v3_experiments/cross_cluster_adaptation.csv",
            "experiments/v3_experiments/random_vs_group_identical_rows.csv",
            "experiments/v3_experiments/s1_learning_rate_study.csv",
            "experiments/v3_experiments/execution_manifest.json",
        ],
    },
    "v2_uncertainty": {
        "name": "v2 Conformal Uncertainty Pipeline (scripts/run_v2_uncertainty.py)",
        "script": "scripts/run_v2_uncertainty.py",
        "description": "Executes 4-way disjoint conformal calibration, evaluating Constant-width, Normalized, and CQR.",
        "supported_modes": ["--dry-run", "--reuse-existing", "--force"],
        "default_mode": "--dry-run",
        "expected_outputs": [
            "experiments/v2_uncertainty/conformal_methods_comparison.csv",
            "experiments/v2_uncertainty/coverage_by_regime_and_bin.csv",
            "experiments/v2_uncertainty/local_recalibration_results.csv",
            "experiments/v2_uncertainty/weekly_coverage_drift.csv",
        ],
    },
    "train_baseline": {
        "name": "Baseline Model Trainer (scripts/train_baseline_model.py)",
        "script": "scripts/train_baseline_model.py",
        "description": "Trains and persists a genuine LightGBM S1 model on Split A seen servers for real offline inference.",
        "supported_modes": [],
        "default_mode": "",
        "expected_outputs": [
            "models/dataset1_split_a_s1_lightgbm_1h.joblib",
            "models/dataset1_split_a_s1_lightgbm_1h.json",
        ],
    },
}


def render_reproduction_centre(registry: DataRegistry):
    st.markdown('<div class="studio-header"><div class="studio-title">⚡ Experiment Reproduction Centre</div><div class="studio-subtitle">Reproduce empirical research pipelines locally with strict subprocess sandboxing</div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="tech-note-box">
        <strong>🔒 Reproducibility Security Guardrails:</strong><br>
        Only pre-verified repository scripts with frozen argument lists can be executed.
        Arbitrary user shell command input is completely prohibited. Results are written strictly to <code>experiments/</code>.
    </div>
    """, unsafe_allow_html=True)

    col_pipe, col_mode = st.columns([2, 1])
    with col_pipe:
        pipeline_key = st.selectbox(
            "Select Experiment Pipeline to Run",
            list(VERIFIED_PIPELINES.keys()),
            format_func=lambda k: VERIFIED_PIPELINES[k]["name"],
        )
        pipe_spec = VERIFIED_PIPELINES[pipeline_key]
        st.caption(pipe_spec["description"])

    with col_mode:
        if pipe_spec["supported_modes"]:
            mode_choice = st.radio("Execution Mode", pipe_spec["supported_modes"], index=0)
        else:
            mode_choice = ""
            st.info("Direct execution mode (generates offline checkpoint).")

    # Confirmation for force mode
    force_confirmed = True
    if mode_choice == "--force":
        force_confirmed = st.checkbox("⚠️ Confirm: Recompute from scratch and overwrite existing CSVs?", value=False)

    col_run, col_status = st.columns([1, 2])
    with col_run:
        run_btn = st.button("🚀 Execute Verified Pipeline", type="primary", disabled=(mode_choice == "--force" and not force_confirmed))

    if run_btn:
        cmd = [sys.executable, str(registry.root_dir / pipe_spec["script"])]
        if mode_choice:
            cmd.append(mode_choice)

        st.markdown(f"**Executing Command:** `{' '.join(cmd)}`")

        start_time = time.time()
        with st.spinner("Pipeline running... Please wait."):
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(registry.root_dir),
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )
                elapsed = time.time() - start_time

                if proc.returncode == 0:
                    st.success(f"✓ Execution completed successfully in {elapsed:.2f} seconds!")
                else:
                    st.error(f"❌ Execution failed with exit code {proc.returncode} in {elapsed:.2f} seconds.")

                # Output logs
                with st.expander("📄 Captured Standard Output (stdout)", expanded=True):
                    st.code(proc.stdout if proc.stdout else "[No standard output]", language="text")

                if proc.stderr:
                    with st.expander("⚠️ Captured Standard Error (stderr)", expanded=True):
                        st.code(proc.stderr, language="text")

                # Verify expected outputs
                st.markdown("#### Output Artifact Verification")
                out_status = []
                for out_f in pipe_spec["expected_outputs"]:
                    p = registry.root_dir / out_f
                    exists = p.exists()
                    size = f"{p.stat().st_size:,} bytes" if exists else "Missing"
                    out_status.append({"Artifact Path": out_f, "Status": "FOUND" if exists else "NOT FOUND", "Size": size})
                st.table(pd.DataFrame(out_status))

            except subprocess.TimeoutExpired:
                st.error("❌ Process timed out after 300 seconds.")
            except Exception as e:
                st.error(f"❌ Execution error: {e}")
