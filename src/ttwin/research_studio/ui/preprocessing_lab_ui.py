"""Preprocessing and Feature Engineering Lab UI (Stage 3).

Visualizes the step-by-step transformation from raw telemetry into model-ready tensors,
explaining lag features, rolling windows, target formulations (Monitoring vs Planning),
and real feature correlation heatmaps.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from ttwin.research_studio.data_registry import DataRegistry
from ttwin.research_studio.dataset_loader import DatasetLoader
from ttwin.research_studio.feature_view import FeatureView


def render_preprocessing_lab(registry: DataRegistry, loader: DatasetLoader):
    st.markdown('<div class="studio-header"><div class="studio-title">🧪 Preprocessing & Feature Engineering Lab</div><div class="studio-subtitle">Verifiable data lineage from raw sensor streams to causal, leak-free feature vectors</div></div>', unsafe_allow_html=True)

    fview = FeatureView(loader)

    tab_pipe, tab_feat, tab_corr = st.tabs([
        "🔄 Pipeline Architecture & Lineage",
        "📐 Feature Sets & Target Formulation",
        "🔥 Correlation Heatmap & Leakage Prevention",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: Pipeline Architecture
    # ══════════════════════════════════════════════════════════════════════════
    with tab_pipe:
        st.markdown('<div class="lab-section-title">The 7-Stage Verified Telemetry Pipeline</div>', unsafe_allow_html=True)

        stages = fview.get_pipeline_stages()
        for s in stages:
            with st.expander(f"Stage {s['stage']}: {s['name']} — `{s['function']}`", expanded=(s['stage'] in [2, 5, 6])):
                st.markdown(f"""
                - **Source File:** `{s['source_file']}`
                - **Function:** `{s['function']}`
                - **Input:** {s['inputs']}
                - **Output:** {s['outputs']}
                - **Scientific Rationale:** {s['description']}
                """)

        st.markdown('<div class="lab-section-title">Actual Preprocessing Execution Trace</div>', unsafe_allow_html=True)
        st.markdown("""
        | Stage | Input Records | Cleaned Output | Transformation Applied | Responsible Module |
        |:---|:---|:---|:---|:---|
        | **D1 Raw $\\to$ Clean** | 2,748,525 | 2,748,039 | Deduplication (-486 rows), Temp Clip $[-10^\\circ\\text{C}, 110^\\circ\\text{C}]$ | `src/ttwin/clean.py` |
        | **D2 Raw $\\to$ Clean** | 1,281,024 | 1,280,640 | Deduplication (-384 rows), Temp Clip $[-10^\\circ\\text{C}, 110^\\circ\\text{C}]$ | `src/ttwin/clean.py` |
        | **Grid Alignment** | Cleaned series | Uniform $\\Delta t = 10\\text{ min}$ | NaN insertion for unrecorded intervals | `fill_missing_timestamps()` |
        | **Hotspot Temp** | Dual-socket readings | $T_{\\text{eff}} = \\max(T_1, T_2)$ | Physics hotspot tracking | `compute_effective_cpu_temp()` |
        | **Feature Matrix** | 10-min time series | Lag & Rolling stats | Windows strictly end at $t$ (zero leakage) | `build_features()` |
        """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: Feature Sets & Target Formulation
    # ══════════════════════════════════════════════════════════════════════════
    with tab_feat:
        st.markdown('<div class="lab-section-title">Feature Set Rationale (S0 through S4)</div>', unsafe_allow_html=True)

        fs_specs = fview.get_feature_set_descriptions()
        c_fs = st.columns(len(fs_specs))
        for idx, (k, spec) in enumerate(fs_specs.items()):
            with c_fs[idx]:
                st.markdown(f"""
                <div class="metric-card-container">
                    <div class="card-val">{k}</div>
                    <div class="card-lbl">{spec['feature_count']} Features</div>
                    <div class="card-sub">{spec['regime']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="lab-section-title">Critical Operational Target Distinction</div>', unsafe_allow_html=True)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("""
            <div class="tech-alert-emerald">
                <h4>🟢 MONITORING REGIME (S1, S4)</h4>
                <strong>Formulation: Incremental Delta Target</strong><br>
                $$\\Delta T(t+h) = T(t+h) - T(t)$$
                <strong>Prediction Reconstruction:</strong><br>
                $$\\hat{T}(t+h) = T(t) + \\Delta \\hat{T}(t+h)$$
                <ul>
                    <li>Evaluated against <strong>Persistence Baseline</strong> ($T(t+h) \\approx T(t)$).</li>
                    <li>Utilizes current temperature $T(t)$ and autoregressive lag memory.</li>
                    <li>Model learns strictly the thermal departure caused by transient workload fluctuations.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col_t2:
            st.markdown("""
            <div class="tech-alert-amber">
                <h4>🟡 PLANNING REGIME (S2, S3)</h4>
                <strong>Formulation: Absolute Temperature Target</strong><br>
                $$\\hat{T}(t+h) = f(\\text{Workload}, \\text{Calendar}, [\\text{Power}])$$
                <strong>Direct Output: Absolute Degrees Celsius</strong><br>
                <ul>
                    <li>Evaluated against <strong>Cluster Mean</strong> and <strong>Local Server Mean</strong> baselines.</li>
                    <li>Persistence <strong>CANNOT</strong> be evaluated because current temperature $T(t)$ is assumed unrecorded.</li>
                    <li>Directly powers scheduler 'what-if' placement where digital twin projects absolute heat before committing jobs.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="lab-section-title">Live Server Feature Vector Extraction</div>', unsafe_allow_html=True)

        servers = loader.get_interim_servers("dataset1")
        sel_srv = st.selectbox("Select Server for Feature Matrix Inspection", servers, index=0)
        df_feat_sample = fview.generate_server_feature_sample("dataset1", sel_srv, n_rows=15)

        st.dataframe(df_feat_sample, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: Correlation Heatmap & Leakage Prevention
    # ══════════════════════════════════════════════════════════════════════════
    with tab_corr:
        st.markdown('<div class="lab-section-title">Empirical Feature Correlation Matrix</div>', unsafe_allow_html=True)
        st.caption("Calculated dynamically from real interim telemetry rows for the selected server.")

        fs_corr = st.selectbox("Select Feature Set to Correlate", ["S1", "S2", "S3", "S4"], index=0)
        df_feat_corr = fview.generate_server_feature_sample("dataset1", sel_srv, n_rows=100)
        corr_matrix = fview.compute_feature_correlations(df_feat_corr, feature_set=fs_corr)

        if not corr_matrix.empty:
            fig_c, ax_c = plt.subplots(figsize=(9, 6))
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="vlag",
                center=0,
                cbar=True,
                ax=ax_c,
                annot_kws={"size": 7},
            )
            ax_c.set_title(f"Feature Pearson Correlation Heatmap ({fs_corr} — {sel_srv})", fontweight="bold", fontsize=10)
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            st.pyplot(fig_c)
            plt.close(fig_c)
        else:
            st.info("Insufficient observation variance to generate correlation matrix.")

        st.markdown('<div class="lab-section-title">Data Leakage Prevention Checklist</div>', unsafe_allow_html=True)
        st.markdown("""
        1. **Strictly Causal Rolling Statistics:** Rolling mean and std calculations end strictly at observation time $t$. Windows never incorporate $t+1$.
        2. **Continuous Grid Alignment:** Grid filling inserts NaNs for unrecorded steps, ensuring a lag of 1 step is strictly 10 minutes in wall-clock time.
        3. **Embargo Buffers:** A 6-hour embargo gap (36 continuous 10-min steps) is excluded between training/validation and test to prevent autoregressive bleed.
        4. **Target Isolation:** Target shifts are computed strictly prior to train/test partitioning, and the target row is completely removed from input feature sets.
        """)
