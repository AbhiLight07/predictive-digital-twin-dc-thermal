"""Uncertainty and Calibration Lab UI (Stage 8).

Inspects split conformal prediction, finite-sample guarantees,
4-way disjoint calibration partitions, conditional coverage breakdowns,
and exchangeability violations in operational data centre telemetry.
Explicitly distinguishes legacy overlapping validation from corrected pipelines.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from ttwin.research_studio.data_registry import DataRegistry
from ttwin.research_studio.dataset_loader import DatasetLoader
from ttwin.research_studio.experiment_loader import ExperimentLoader


def render_uncertainty_lab(registry: DataRegistry, loader: DatasetLoader):
    st.markdown('<div class="studio-header"><div class="studio-title">📐 Conformal Uncertainty & Calibration Lab</div><div class="studio-subtitle">Distribution-free finite-sample coverage guarantees and distribution-shift stress testing</div></div>', unsafe_allow_html=True)

    exp_loader = ExperimentLoader(registry)

    tab_methods, tab_regimes, tab_drift, tab_theory = st.tabs([
        "📊 Conformal Methods Benchmark",
        "🌡️ Conditional Coverage by Regime & Temp",
        "⏱️ Weekly Temporal Coverage Drift",
        "📐 Conformal Math & Exchangeability Limits",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: Methods Benchmark
    # ══════════════════════════════════════════════════════════════════════════
    with tab_methods:
        st.markdown('<div class="lab-section-title">Split Conformal Methods Comparison (Split C Unseen Servers)</div>', unsafe_allow_html=True)
        st.markdown('<span class="source-tag">Source: experiments/v2_uncertainty/conformal_methods_comparison.csv</span>', unsafe_allow_html=True)

        st.markdown("""
        <div class="tech-alert-emerald">
            <strong>✓ Corrected 4-Way Disjoint Partitioning Enforced:</strong><br>
            All v2 uncertainty benchmarks enforce strict separation:
            <strong>1. Train (70%)</strong> $\\to$ <strong>2. Val-Select (5%)</strong> $\\to$ <strong>3. Val-Cal (5%)</strong> $\\to$ <strong>4. 6h Embargo</strong> $\\to$ <strong>5. Test (20% - 6h)</strong>.<br>
            Calibration conformity scores were computed exclusively on held-out calibration rows never seen by early stopping or model fitting.
        </div>
        """, unsafe_allow_html=True)

        df_comp, _ = exp_loader.load_uncertainty_conformal()
        st.dataframe(df_comp, use_container_width=True)

        # Plot Coverage vs Nominal
        st.markdown("#### Empirical Coverage vs 90% Nominal Target")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        bars = ax.bar(
            df_comp["method"],
            df_comp["empirical_coverage"].str.rstrip("%").astype(float),
            color=["#38bdf8", "#f59e0b", "#10b981"],
            alpha=0.85,
        )
        ax.axhline(90.0, color="#ef4444", linestyle="--", linewidth=1.5, label="Nominal Target (90.0%)")
        ax.set_ylabel("Empirical Coverage (%)")
        ax.set_ylim(80, 95)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.2)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{bar.get_height():.2f}%", ha="center", fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        * **Constant-Width Split Conformal:** Meets 90.33% marginal coverage, but interval width is fixed at 4.20°C regardless of load.
        * **Normalized Conformal:** Adaptively scales intervals by predicted residual variance, but under-covers at 86.44% due to variance estimator bias.
        * **Conformalized Quantile Regression (CQR):** Achieves **91.19% empirical coverage** with a tighter mean interval width of **3.68°C** and strong error adaptivity ($r=0.470$).
        """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: Conditional Coverage by Regime
    # ══════════════════════════════════════════════════════════════════════════
    with tab_regimes:
        st.markdown('<div class="lab-section-title">Conditional Coverage: Why Marginal Guarantees Fail Locally</div>', unsafe_allow_html=True)
        st.markdown('<span class="source-tag">Source: experiments/v2_uncertainty/coverage_by_regime_and_bin.csv</span>', unsafe_allow_html=True)
        st.caption("Marginal coverage guarantees an overall average over the population; it does NOT guarantee 90% coverage conditionally in every operating slice.")

        df_reg, _ = exp_loader.load_uncertainty_regimes()
        st.dataframe(df_reg, use_container_width=True)

        st.markdown(r"""
        <div class="tech-alert-amber">
            <h4>⚠️ Key Safety Finding on Thermal Hazard Bins</h4>
            <ul>
                <li><strong>Constant-Width Under-Coverage on Active Loads:</strong> Drops to <strong>83.06% coverage</strong> on active workloads (10–50% CPU), precisely when thermal risk increases!</li>
                <li><strong>Constant-Width Under-Coverage on Hot Temperatures:</strong> Drops to <strong>86.97% coverage</strong> on hot temperatures ($\ge 60^\circ\text{C}$).</li>
                <li><strong>CQR Superiority:</strong> CQR maintains $\ge 90.8\%$ coverage across all workload regimes and achieves 92.67% on hot servers, but slightly dips to <strong>89.09% in the moderate 50–60°C bin</strong>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: Weekly Drift
    # ══════════════════════════════════════════════════════════════════════════
    with tab_drift:
        st.markdown('<div class="lab-section-title">Temporal Coverage Drift Across 8 Test Weeks</div>', unsafe_allow_html=True)
        st.markdown('<span class="source-tag">Source: experiments/v2_uncertainty/weekly_coverage_drift.csv</span>', unsafe_allow_html=True)

        df_drift, _ = exp_loader.load_weekly_drift()
        st.dataframe(df_drift, use_container_width=True)

        # Plot drift over weeks
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(df_drift["iso_week"], df_drift["constant_width_coverage"].str.rstrip("%").astype(float), marker="o", label="Constant-Width", color="#38bdf8")
        ax.plot(df_drift["iso_week"], df_drift["cqr_coverage"].str.rstrip("%").astype(float), marker="s", label="CQR", color="#10b981")
        ax.axhline(90.0, color="#ef4444", linestyle="--", label="Nominal 90%")
        ax.set_ylabel("Weekly Coverage (%)")
        ax.set_title("Temporal Uncertainty Coverage Stability Over Test Deployment", fontweight="bold", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4: Conformal Math & Exchangeability Limits
    # ══════════════════════════════════════════════════════════════════════════
    with tab_theory:
        st.markdown('<div class="lab-section-title">Exact Mathematical Formulation & Limitations</div>', unsafe_allow_html=True)

        st.markdown("""
        ### Exact Finite-Sample Quantile Formula
        Given $n$ exchangeable calibration nonconformity scores $S_1, \\dots, S_n$ and miscoverage target $\\alpha \\in (0, 1)$:
        $$k = \\left\\lceil (n + 1)(1 - \\alpha) \\right\\rceil$$
        
        Sorting scores in ascending order $S_{(1)} \\le S_{(2)} \\le \\dots \\le S_{(n)}$:
        - If $1 \\le k \\le n$: $\\hat{q} = S_{(k)}$ (the $k$-th order statistic).
        - **If $k > n$:** The sample size $n < \\frac{1}{\\alpha} - 1$ is insufficient to guarantee coverage. The mathematically correct finite-sample threshold is **$+\\infty$**, NOT an ad-hoc interpolated quantile.

        ### Practical Violations of Exchangeability in Data Centres
        Split conformal theory strictly guarantees $P(Y_{n+1} \\in C(X_{n+1})) \\ge 1 - \\alpha$ under **exchangeability**.
        In physical facilities, exchangeability is routinely violated by:
        1. **Temporal Autocorrelation:** Consecutive 10-minute records form autocorrelated time series.
        2. **Operational Drift:** Operational workload shift and component aging between calibration and test.
        3. **Spatial Heterogeneity:** Unseen physical servers differ in thermal mass, fan RPM curves, and rack microclimate.
        """)
