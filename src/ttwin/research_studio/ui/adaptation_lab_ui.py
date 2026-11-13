"""Adaptation Lab UI (Stage 7).

Demonstrates few-shot and limited-data adaptation across budgets [0, 1, 6, 24, 72, 168] hours:
- Zero-shot vs Residual Offset vs Continued Boosting
- Fleet-wide planning adaptation (81 servers, all 5 folds)
- Cross-cluster hardware transfer (Dell R840 -> Dell C6320)
- Interactive budget scrubber and 95% bootstrap confidence intervals
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from ttwin.research_studio.data_registry import DataRegistry
from ttwin.research_studio.dataset_loader import DatasetLoader
from ttwin.research_studio.experiment_loader import ExperimentLoader


def render_adaptation_lab(registry: DataRegistry, loader: DatasetLoader):
    st.markdown('<div class="studio-header"><div class="studio-title">🔄 Few-Shot & Cold-Start Adaptation Lab</div><div class="studio-subtitle">Empirical performance scaling with limited local calibration budgets (0h to 168h)</div></div>', unsafe_allow_html=True)

    exp_loader = ExperimentLoader(registry)

    tab_fleet, tab_cross, tab_methods = st.tabs([
        "🏙️ Fleet-Wide Planning Adaptation (81 Servers)",
        "🔀 Cross-Cluster Hardware Transfer (C1 -> C2)",
        "🧠 Adaptation Strategies Explained",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: Fleet-Wide Planning Adaptation
    # ══════════════════════════════════════════════════════════════════════════
    with tab_fleet:
        st.markdown('<div class="lab-section-title">Fleet-Wide Planning Adaptation Across All 5 Folds</div>', unsafe_allow_html=True)
        st.markdown('<span class="source-tag">Source: experiments/v3_experiments/planning_adaptation_all_folds.csv</span>', unsafe_allow_html=True)
        st.caption("Evaluated on all 81 physical servers in Dataset 1 across budgets [0h, 1h, 6h, 24h, 72h, 168h] in the planning regime.")

        df_fleet, _ = exp_loader.load_adaptation_fleet()

        # Interactive budget scrubber
        budgets = sorted(df_fleet["budget_hours"].unique().tolist())
        sel_budget = st.select_slider("Select Local Calibration Budget (Hours)", options=budgets, value=24)

        sub_fleet = df_fleet[df_fleet["budget_hours"] == sel_budget]

        st.markdown(f"**Results at Budget = {sel_budget} hours:**")
        st.dataframe(sub_fleet, use_container_width=True)

        # Plot adaptation curves across budgets
        st.markdown("#### Adaptation Trajectory Across Budgets")
        fig, ax = plt.subplots(figsize=(10, 4))

        for strat, color, marker in [
            ("continued_boosting", "#38bdf8", "o"),
            ("residual_offset", "#f59e0b", "s"),
            ("zero_shot", "#10b981", "D"),
        ]:
            strat_df = df_fleet[df_fleet["strategy"] == strat].sort_values("budget_hours")
            if not strat_df.empty:
                ax.plot(strat_df["budget_hours"], strat_df["mean_mae"], label=strat, color=color, marker=marker, linewidth=1.5)
                # Plot confidence band if available
                if "ci_95_low" in strat_df.columns and "ci_95_high" in strat_df.columns:
                    ax.fill_between(strat_df["budget_hours"], strat_df["ci_95_low"], strat_df["ci_95_high"], color=color, alpha=0.15)

        # Fleet mean baseline
        fmean = df_fleet[df_fleet["strategy"] == "fleet_mean"]["mean_mae"].iloc[0]
        ax.axhline(fmean, color="#ef4444", linestyle="--", label=f"Fleet Mean Baseline ({fmean:.2f}°C)")

        ax.set_xlabel("Local Calibration Budget (Hours)")
        ax.set_ylabel("MAE (°C)")
        ax.set_title("Fleet-Wide Planning Adaptation Curve (95% Bootstrap CI over Servers)", fontweight="bold", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        > **Core Insight on Planning Adaptation:**  
        > - **Zero-shot planning** achieves **5.03°C MAE** (95% CI: `[4.71°C, 5.39°C]`), delivering a **43.2% error reduction** over the unconditioned fleet mean (8.87°C).
        > - **Continued boosting** gradually refines accuracy: 5.03°C (0h) $\to$ **4.87°C at 168 hours** (7 days).
        > - Longer adaptation budgets do not monotonically improve accuracy for every server because local operational drift varies across physical nodes.
        """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: Cross-Cluster Hardware Transfer
    # ══════════════════════════════════════════════════════════════════════════
    with tab_cross:
        st.markdown('<div class="lab-section-title">Cross-Cluster Hardware Adaptation (Cluster 1 $\\to$ Cluster 2)</div>', unsafe_allow_html=True)
        st.markdown('<span class="source-tag">Source: experiments/v3_experiments/cross_cluster_adaptation.csv</span>', unsafe_allow_html=True)
        st.caption("Transfer from Dell PowerEdge R840 (Cluster 1) to Dell PowerEdge C6320 (Cluster 2) target nodes.")

        df_cross, _ = exp_loader.load_cross_cluster_adaptation()
        st.dataframe(df_cross, use_container_width=True)

        st.markdown("""
        ### Cross-Cluster Transfer Findings
        - **Zero-Shot Transfer Uncertainty:** Applying the Cluster 1 model directly to Cluster 2 yields a mean MAE of **6.85°C**, but exhibits wide confidence bounds (`[2.92°C, 13.47°C]`).
        - **Rapid Bias Correction via Residual Offset:** A simple scalar offset estimated from just **6 hours of local data** drops error to **3.16°C** (95% CI: `[2.13°C, 4.30°C]`).
        - **Continued Boosting Limitation:** Continued tree boosting underperforms residual offset (6.05°C vs 3.03°C at 168h) because tree split constraints on small adaptation slices restrict parameter adaptation.
        """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: Adaptation Strategies Explained
    # ══════════════════════════════════════════════════════════════════════════
    with tab_methods:
        st.markdown('<div class="lab-section-title">The Three Adaptation Paradigms Explained</div>', unsafe_allow_html=True)

        st.markdown("""
        1. **Zero-Shot Transfer:**
           - Deploys the pretrained model directly to the new physical server with **zero local calibration observations**.
           - **Pro:** Instant zero-downtime deployment; avoids days of cold-start idling.
           - **Con:** Incurs spatial offset bias if the target server has distinct chassis airflow impedance or thermal paste aging.
        2. **Scalar Residual Offset Adaptation:**
           - Runs zero-shot predictions on the first $N$ hours of local telemetry, computes the scalar mean residual bias:
             $$\\hat{b} = \\frac{1}{N} \\sum_{i=1}^{N} (T_i - \\hat{T}_i)$$
             and shifts future predictions by $\\hat{b}$.
           - **Pro:** Highly robust against overfitting even with tiny sample sizes ($N=6\\text{ hours}$).
           - **Con:** Adjusts only the intercept, cannot adapt to altered non-linear cooling fan curves.
        3. **Continued Boosting (Warm-Start Fine-Tuning):**
           - Uses the pretrained LightGBM booster as an initial model (`init_model`), growing additional trees on local adaptation observations with reduced learning rate.
           - **Pro:** Can learn complex local non-linearities given sufficient data ($N \\ge 72\\text{ hours}$).
           - **Con:** Vulnerable to severe leaf overfitting at small sample sizes (e.g. 1h to 6h).
        """)
