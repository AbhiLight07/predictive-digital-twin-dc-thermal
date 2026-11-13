"""Generalisation Lab UI (Stage 6).

Demonstrates and explains the three generalisation protocols:
- Split A: Chronological 70/10/20 with 6h embargo
- Split B: 5-Fold GroupKFold by physical server
- Split C: Strict joint spatiotemporal holdout
- Random KFold vs GroupKFold leakage demonstration (`random_vs_group_identical_rows.csv`)
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from ttwin.research_studio.data_registry import DataRegistry
from ttwin.research_studio.dataset_loader import DatasetLoader
from ttwin.research_studio.experiment_loader import ExperimentLoader


def render_generalisation_lab(registry: DataRegistry, loader: DatasetLoader):
    st.markdown('<div class="studio-header"><div class="studio-title">🌐 Spatial & Temporal Generalisation Lab</div><div class="studio-subtitle">Rigorous cross-server validation, embargo enforcement, and data leakage diagnostics</div></div>', unsafe_allow_html=True)

    exp_loader = ExperimentLoader(registry)

    tab_splits, tab_leakage, tab_seeds = st.tabs([
        "📐 Split Protocols & Evaluation Architecture",
        "🚨 Random KFold vs GroupKFold (Leakage Proof)",
        "🎲 Split C 5-Seed Sensitivity Analysis",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: Split Protocols
    # ══════════════════════════════════════════════════════════════════════════
    with tab_splits:
        st.markdown('<div class="lab-section-title">The Three Spatial & Temporal Evaluation Protocols</div>', unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("""
            <div class="metric-card-container">
                <div class="card-val">Split A</div>
                <div class="card-lbl">Chronological Holdout</div>
                <div class="card-sub">70% Train / 10% Val / 6h Embargo / 20% Test</div>
                <hr style="border-color: #1e324d; margin: 8px 0;">
                <p style="font-size: 0.8rem; color: #cbd5e1; margin: 0;">
                    Simulates forecasting into the future for known, instrumented server nodes.
                    Guarantees zero future information enters model training.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown("""
            <div class="metric-card-container">
                <div class="card-val">Split B</div>
                <div class="card-lbl">5-Fold GroupKFold by Server</div>
                <div class="card-sub">100% of Servers Evaluated Once as Unseen</div>
                <hr style="border-color: #1e324d; margin: 8px 0;">
                <p style="font-size: 0.8rem; color: #cbd5e1; margin: 0;">
                    All telemetry rows for a given server are kept exclusively in either train or test folds.
                    Stratified by mean CPU load to balance thermal regimes.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col_c:
            st.markdown("""
            <div class="metric-card-container">
                <div class="card-val">Split C</div>
                <div class="card-lbl">Strict Spatiotemporal Holdout</div>
                <div class="card-sub">Held-Out Servers AND Strictly Later Time</div>
                <hr style="border-color: #1e324d; margin: 8px 0;">
                <p style="font-size: 0.8rem; color: #cbd5e1; margin: 0;">
                    The ultimate stress test: predicts future operations on physical hardware never seen during training or early stopping.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="lab-section-title">Actual Split B & C Evaluation Across Horizons</div>', unsafe_allow_html=True)
        st.markdown('<span class="source-tag">Source: experiments/v3_experiments/split_b_and_c_horizons.csv</span>', unsafe_allow_html=True)

        df_splits, _ = exp_loader.load_split_generalisation()
        st.dataframe(df_splits, use_container_width=True)

        st.markdown("""
        > **Key Generalisation Finding:** In GroupKFold (Split B), error scales with forecast horizon:
        > - 10 min: **0.435°C MAE**
        > - 1 hour: **0.918°C MAE**
        > - 6 hours: **1.817°C MAE**
        > Evaluating unseen servers incurs a spatial penalty compared to seen servers (~0.92°C vs ~0.83°C at 1h).
        """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: Leakage Demonstration
    # ══════════════════════════════════════════════════════════════════════════
    with tab_leakage:
        st.markdown('<div class="lab-section-title">Leakage Diagnosis: Random K-Fold vs GroupKFold</div>', unsafe_allow_html=True)
        st.markdown('<span class="source-tag">Source: experiments/v3_experiments/random_vs_group_identical_rows.csv</span>', unsafe_allow_html=True)
        st.caption("Direct empirical proof comparing random row splitting versus GroupKFold on strictly identical telemetry rows.")

        df_leak, _ = exp_loader.load_leakage_comparison()
        st.dataframe(df_leak, use_container_width=True)

        # Plot leakage underestimation
        fig_l, ax_l = plt.subplots(figsize=(8, 4))
        bars = ax_l.bar(
            df_leak["feature_set"] + " (" + df_leak["regime"] + ")",
            df_leak["leakage_underestimation_pct"],
            color=["#38bdf8", "#f59e0b", "#f59e0b", "#38bdf8"],
            alpha=0.85,
        )
        ax_l.set_ylabel("Leakage Optimism Bias (% Underestimated Error)")
        ax_l.set_title("Optimism Bias of Naive Random Row Splitting", fontweight="bold", fontsize=10)
        ax_l.grid(True, alpha=0.2)
        for bar in bars:
            ax_l.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{bar.get_height():.2f}%", ha="center", fontsize=8)
        st.pyplot(fig_l)
        plt.close(fig_l)

        st.markdown("""
        ### Why Random Row Splitting Produces False Optimism
        1. **In Monitoring (S1, S4):** Because current temperature $T(t)$ is provided, the difference is small (+0.24% to +0.54%) because local thermal inertia anchors both models.
        2. **In Planning (S2, S3):** Naive random splitting underestimates error by **17.60% (S2)** and **18.54% (S3)**!
           - **Temporal leakage:** Row $t-1$ and row $t+1$ for the same physical server leak through random sampling.
           - **Spatial leakage:** The model memorizes server-specific baseline thermal offsets instead of learning genuine generalizable workload dynamics.
        """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: Split C Sensitivity
    # ══════════════════════════════════════════════════════════════════════════
    with tab_seeds:
        st.markdown('<div class="lab-section-title">Split C Robustness Across 5 Random Server Draws</div>', unsafe_allow_html=True)
        st.markdown("""
        To verify that Split C results are not an artifact of an unusually easy server holdout,
        Split C was evaluated across **5 distinct random seed draws** (`[42, 123, 456, 789, 1024]`):
        - **10 min Horizon:** Persistence MAE = 0.456°C ± 0.102°C | S1 MAE = 0.456°C ± 0.101°C
        - **1 hour Horizon:** Persistence MAE = 0.940°C ± 0.179°C | S1 MAE = 0.936°C ± 0.175°C
        - **6 hours Horizon:** Persistence MAE = 1.710°C ± 0.218°C | S1 MAE = 1.699°C ± 0.209°C

        The variance across seed draws (std ~0.10°C to 0.22°C) confirms that inter-server thermal heterogeneity exists, but the overall accuracy envelope remains stable.
        """)
