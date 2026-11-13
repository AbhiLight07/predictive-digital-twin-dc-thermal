"""Model Benchmark and Ablation Lab UI (Stage 5).

Interactive explorer for committed benchmark outputs (v2 and v3):
- Filter by dataset, split, horizon, feature set, model
- Metrics: MAE, RMSE, Skill vs Persistence
- Hardware Sensor Ablation (S4 vs without Fans vs without Power vs S1)
- Scientific explanations of why persistence is a meaningful baseline
- Source CSV file displayed alongside every result
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from ttwin.research_studio.data_registry import DataRegistry
from ttwin.research_studio.dataset_loader import DatasetLoader
from ttwin.research_studio.experiment_loader import ExperimentLoader


def render_benchmark_lab(registry: DataRegistry, loader: DatasetLoader):
    st.markdown('<div class="studio-header"><div class="studio-title">📊 Model Benchmark & Ablation Lab</div><div class="studio-subtitle">Verified empirical performance tables and sensor ablation from committed experiment outputs</div></div>', unsafe_allow_html=True)

    exp_loader = ExperimentLoader(registry)

    tab_ablation, tab_sensor, tab_explain = st.tabs([
        "📈 Primary Ablation Explorer",
        "🛠️ Sensor & Hardware Ablation (v3)",
        "📖 Baseline Rationale & Metric Definitions",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: Primary Ablation Explorer
    # ══════════════════════════════════════════════════════════════════════════
    with tab_ablation:
        st.markdown('<div class="lab-section-title">Split A Unified Benchmark Table</div>', unsafe_allow_html=True)
        st.markdown('<span class="source-tag">Source: experiments/v2_ablation/results_split_a.csv</span>', unsafe_allow_html=True)

        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            f_dataset = st.selectbox("Dataset", ["all", "dataset1", "dataset2"])
        with col_f2:
            f_horizon = st.selectbox("Prediction Horizon", ["all", "10min", "1h", "6h"], index=2)  # default 1h
        with col_f3:
            f_feat = st.selectbox("Feature Set", ["all", "S0", "S1", "S2", "S3", "S4"])
        with col_f4:
            f_model = st.selectbox("Model Family", ["all", "persistence", "lightgbm", "ridge"])

        df_bench, src_bench = exp_loader.load_unified_ablation_benchmarks(
            dataset=f_dataset,
            horizon=f_horizon,
            feature_set=f_feat,
            model=f_model,
        )

        st.dataframe(df_bench, use_container_width=True)

        # Plot MAE comparison across feature sets
        if not df_bench.empty and "mae" in df_bench.columns and "feature_set" in df_bench.columns:
            st.markdown("#### MAE Comparison Across Models & Horizons")
            fig, ax = plt.subplots(figsize=(10, 4))
            sub_plot = df_bench.sort_values("mae")
            bars = ax.barh(
                sub_plot["feature_set"] + " (" + sub_plot["model"] + ", " + sub_plot["horizon"] + ")",
                sub_plot["mae"],
                color="#38bdf8",
                alpha=0.85,
            )
            ax.set_xlabel("Mean Absolute Error (°C)")
            ax.set_title("Forecast Accuracy by Configuration (Split A Chronological)", fontweight="bold", fontsize=10)
            ax.grid(True, alpha=0.2)
            for bar in bars:
                ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.2f}°C", va="center", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: Sensor & Hardware Ablation (v3)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_sensor:
        st.markdown('<div class="lab-section-title">Isolating Hardware Sensors: Fans, Power, and Inlet (v3)</div>', unsafe_allow_html=True)
        st.markdown('<span class="source-tag">Source: experiments/v3_experiments/sensor_ablation.csv</span>', unsafe_allow_html=True)
        st.caption("Evaluated on strictly identical test rows (552,672 rows on Dataset 1, 1-hour horizon).")

        df_sensor, src_sensor = exp_loader.load_sensor_ablation()
        st.dataframe(df_sensor, use_container_width=True)

        st.markdown("#### Sensor Impact Insights")
        st.markdown("""
        - **Cooling Fan Speed Telemetry:** Removing cooling fan telemetry changes test MAE by less than **0.0001°C** (0.83807°C vs 0.83807°C). Fan speeds operate under firmware closed-loop control (BMC/PID), making them reactive to heat rather than proactively explanatory.
        - **Power Telemetry:** Removing server power draw changes test MAE by only **0.0003°C** (0.8381°C vs 0.8383°C).
        - **Lean S1 Superiority:** The lean S1 feature set (36 features of pure temperature memory) achieves **0.8360°C MAE**, slightly outperforming the 56-feature multimodal model (0.8381°C). Additional hardware sensors introduce slight overfitting variance.
        """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: Baseline Rationale & Metric Definitions
    # ══════════════════════════════════════════════════════════════════════════
    with tab_explain:
        st.markdown('<div class="lab-section-title">Scientific Metric Definitions & Baseline Rationale</div>', unsafe_allow_html=True)

        st.markdown("""
        ### Why Persistence is the Authoritative Monitoring Baseline
        In short-horizon data centre telemetry (10 min to 1 hour), servers exhibit substantial **thermal inertia** (copper heat sinks, heat pipes, and chassis mass). The simple naive persistence baseline:
        $$\\hat{T}(t+h) = T(t)$$
        accounts for **>99% of short-term thermal variance**.
        
        Therefore:
        - Reporting an $R^2$ of 0.98 or an MAE of 0.84°C *without comparing against persistence* is scientifically misleading.
        - The meaningful metric is **Skill Score vs Persistence**:
          $$\\text{Skill} = 1 - \\frac{\\text{MAE}_{\\text{model}}}{\\text{MAE}_{\\text{persistence}}}$$
        - Positive skill ($>0$) means the machine learning model genuinely extracts predictive signal beyond thermal memory.
        - Zero or negative skill indicates the model fails to beat a trivial static wire.

        ### Why a 0.5% – 2.8% Improvement Must Not Be Overstated
        In empirical experiments:
        - LightGBM achieves **+0.59% skill** on Dataset 1 and **+2.83% skill** on Dataset 2.
        - However, **server-level 95% bootstrap confidence intervals cross zero** on both datasets (`[-0.38%, +0.17%]` on D1).
        - Across individual physical server nodes, predictive superiority over persistence is **not statistically significant**.
        """)
