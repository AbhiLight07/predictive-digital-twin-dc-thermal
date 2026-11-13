"""Real Model Inference Lab UI (Stage 4).

Executes genuine offline inference on actual telemetry observations.
No mock calculations, no synthetic noise, no faking.
Displays current temperature, model prediction, ground truth future outcome,
persistence baseline, and exact error margins.
"""

from typing import Dict, List, Optional, Any
import subprocess
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from ttwin.research_studio.data_registry import DataRegistry
from ttwin.research_studio.dataset_loader import DatasetLoader
from ttwin.research_studio.model_loader import ModelLoader
from ttwin.research_studio.inference import ModelInferenceEngine


def render_inference_lab(registry: DataRegistry, loader: DatasetLoader):
    st.markdown('<div class="studio-header"><div class="studio-title">🎯 Real Model Inference Lab</div><div class="studio-subtitle">Point prediction and error benchmarking against actual future sensor ground truth</div></div>', unsafe_allow_html=True)

    m_loader = ModelLoader(registry)
    engine = ModelInferenceEngine(m_loader, loader)

    # 1. Model Availability Status
    available_models = m_loader.list_available_models()
    if not available_models:
        st.markdown("""
        <div class="tech-alert-amber">
            <h4>⚠️ Trained Checkpoint Unavailable</h4>
            No serialized model checkpoint was found in <code>models/</code> or <code>checkpoints/</code>.<br>
            Per the research studio protocol, <strong>the system will never fabricate predictions or use mock heuristics as a substitute for model inference</strong>.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Reproducible Training Pipeline Command")
        st.code("python scripts/train_baseline_model.py", language="bash")
        st.caption("This script trains a genuine LightGBM S1 model on Split A seen servers (70% train / 10% val with early stopping) and persists the verified booster and metadata to `models/`.")

        col_tr1, col_tr2 = st.columns([1, 2])
        with col_tr1:
            if st.button("🚀 Train Verified Baseline Checkpoint Now", type="primary"):
                with st.spinner("Training genuine LightGBM S1 model on Split A seen servers..."):
                    try:
                        from scripts.train_baseline_model import train_s1_model
                        saved_path = train_s1_model(horizon_steps=6)
                        st.success(f"✓ Model trained and persisted to: `{saved_path}`")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Training failed: {e}")
        return

    # 2. Controls when model is available
    st.markdown('<div class="lab-section-title">Inference Configuration & Observation Selection</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sel_dataset = st.selectbox("Telemetry Dataset", ["dataset1", "dataset2"])
    with c2:
        servers = loader.get_interim_servers(sel_dataset)
        sel_server = st.selectbox("Select Target Server", servers, index=0)
    with c3:
        sel_model_desc = st.selectbox(
            "Select Trained Model",
            available_models,
            format_func=lambda m: f"{m.name} ({m.feature_set}, {m.horizon_label})",
        )
    with c4:
        horizon_steps = sel_model_desc.horizon_steps
        st.metric("Model Prediction Horizon", sel_model_desc.horizon_label, f"{horizon_steps * 10} min ahead")

    # Load observation list for chosen server
    df_srv = loader.load_interim_server(sel_dataset, sel_server, columns=["Time", "CPU_Temperature" if "CPU_Temperature" in loader.get_interim_metadata(sel_dataset)["columns"] else "CPU1_Temperature"])
    df_srv["Time"] = pd.to_datetime(df_srv["Time"])

    # Compute Split A chronological boundaries for this telemetry trace
    t_min = df_srv["Time"].min()
    t_max = df_srv["Time"].max()
    dur = t_max - t_min
    train_end = t_min + dur * 0.70
    val_end = train_end + dur * 0.10
    test_start = val_end + pd.Timedelta(hours=6)  # 6-hour embargo buffer

    part_choice = st.radio(
        "Select Target Observation Partition:",
        ["Held-Out Test Window (Split A Test — Recommended)", "Validation Window (Model Selection)", "Train Window (In-Sample)", "All Observations"],
        horizontal=True,
    )

    if "Held-Out Test Window" in part_choice:
        valid_window = df_srv[(df_srv["Time"] >= test_start) & (df_srv["Time"] <= t_max - pd.Timedelta(minutes=horizon_steps * 10 + 20))].copy()
    elif "Validation Window" in part_choice:
        valid_window = df_srv[(df_srv["Time"] >= train_end) & (df_srv["Time"] < val_end - pd.Timedelta(minutes=horizon_steps * 10 + 20))].copy()
    elif "Train Window" in part_choice:
        valid_window = df_srv[(df_srv["Time"] >= t_min + pd.Timedelta(hours=6)) & (df_srv["Time"] < train_end - pd.Timedelta(minutes=horizon_steps * 10 + 20))].copy()
    else:
        valid_window = df_srv.iloc[1000 : max(1001, len(df_srv) - horizon_steps - 10)].copy()

    if valid_window.empty:
        valid_window = df_srv.iloc[1000 : max(1001, len(df_srv) - horizon_steps - 10)].copy()

    col_ts, col_btn = st.columns([3, 1])
    with col_ts:
        timestamp_options = valid_window["Time"].tolist()
        sel_timestamp = st.selectbox(
            f"Select Telemetry Timestamp ({len(timestamp_options):,} available in selected partition)",
            timestamp_options,
            index=min(50, len(timestamp_options) - 1),
            format_func=lambda t: t.strftime("%Y-%m-%d %H:%M:%S UTC"),
        )
    with col_btn:
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        run_btn = st.button("⚡ Run Model Prediction", type="primary")

    st.markdown("---")

    # 3. Execution of Genuine Inference
    res = engine.run_inference(
        dataset_name=sel_dataset,
        server_name=sel_server,
        timestamp=sel_timestamp,
        checkpoint_name=sel_model_desc.name,
        horizon_steps=horizon_steps,
        feature_set=sel_model_desc.feature_set,
    )

    if not res.get("available", False):
        st.error(f"Inference Error: {res.get('error')}")
        return

    # 4. Display Results
    st.markdown('<div class="lab-section-title">Inference Results & Error Comparison</div>', unsafe_allow_html=True)

    part_color = "badge-verified" if "TEST" in res["partition"] else "badge-warning"
    st.markdown(f"""
    <div>
        <strong>Evaluation Partition:</strong> <span class="{part_color}">{res['partition']}</span>
        &nbsp;|&nbsp; <strong>Target Server:</strong> <code>{res['server']}</code>
        &nbsp;|&nbsp; <strong>Origin Time:</strong> <code>{res['timestamp']}</code>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding-top: 12px;'></div>", unsafe_allow_html=True)

    res_cols = st.columns(4)
    with res_cols[0]:
        st.metric("Observed Temp at t", f"{res['current_temp']:.2f} °C")
    with res_cols[1]:
        st.metric(
            f"Predicted Temp (t+{res['horizon_label']})",
            f"{res['predicted_future_temp']:.2f} °C",
            f"Δ = {res['predicted_delta']:+.2f} °C",
        )
    with res_cols[2]:
        st.metric(
            f"Actual Observed Temp (t+{res['horizon_label']})",
            f"{res['actual_future_temp']:.2f} °C",
        )
    with res_cols[3]:
        delta_color = "normal" if res["improvement_over_persistence"] >= 0 else "inverse"
        st.metric(
            "Model Absolute Error",
            f"{res['model_abs_error']:.3f} °C",
            f"{res['improvement_over_persistence']:+.3f} °C vs Persistence",
            delta_color=delta_color,
        )

    st.markdown("#### Comparison with Persistence Baseline")
    comp_df = pd.DataFrame([
        {
            "Predictor": "Persistence Baseline (T(t+h) ≈ T(t))",
            "Predicted Temp": f"{res['persistence_pred']:.2f} °C",
            "Actual Temp": f"{res['actual_future_temp']:.2f} °C",
            "Absolute Error": f"{res['persistence_abs_error']:.3f} °C",
            "Skill vs Baseline": "0.00% (Reference)",
        },
        {
            "Predictor": f"Trained {res['model_name']}",
            "Predicted Temp": f"{res['predicted_future_temp']:.2f} °C",
            "Actual Temp": f"{res['actual_future_temp']:.2f} °C",
            "Absolute Error": f"{res['model_abs_error']:.3f} °C",
            "Skill vs Baseline": f"{((res['persistence_abs_error'] - res['model_abs_error']) / max(0.001, res['persistence_abs_error'])) * 100:+.2f}%",
        },
    ])
    st.table(comp_df)

    # 5. Trajectory Plot
    st.markdown("#### Observed Trajectory & Prediction Alignment")
    # Extract historical window around timestamp
    time_origin = pd.to_datetime(res["timestamp"])
    time_future = time_origin + pd.Timedelta(minutes=res["horizon_steps"] * 10)

    window_mask = (df_srv["Time"] >= time_origin - pd.Timedelta(hours=4)) & (df_srv["Time"] <= time_future + pd.Timedelta(hours=2))
    df_window = df_srv[window_mask]

    cur_col = "CPU_Temperature" if "CPU_Temperature" in df_window.columns else "CPU1_Temperature"

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_window["Time"], df_window[cur_col], label="Actual Telemetry Trajectory", color="#38bdf8", linewidth=1.5)
    ax.scatter([time_origin], [res["current_temp"]], color="#ffffff", s=70, zorder=5, label=f"Origin t ({res['current_temp']:.1f}°C)")
    ax.scatter([time_future], [res["actual_future_temp"]], color="#10b981", s=90, marker="o", zorder=6, label=f"Actual Ground Truth ({res['actual_future_temp']:.1f}°C)")
    ax.scatter([time_future], [res["predicted_future_temp"]], color="#f59e0b", s=90, marker="^", zorder=6, label=f"Model Prediction ({res['predicted_future_temp']:.1f}°C)")

    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Thermal Forecast Verification — {res['server']} at horizon {res['horizon_label']}", fontweight="bold", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

    # 6. Feature Vector & Download Record
    with st.expander("🔍 Inspect Input Feature Vector (First 10 Features Fed to Model)"):
        st.json(res["feature_summary"])

    pred_record = json.dumps(res, indent=2).encode("utf-8")
    st.download_button(
        "📥 Download Verified Prediction Record (JSON)",
        data=pred_record,
        file_name=f"prediction_{res['server']}_{res['timestamp'].replace(' ', '_').replace(':', '-')}.json",
        mime="application/json",
    )
