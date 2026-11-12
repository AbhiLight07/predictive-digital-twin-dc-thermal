"""Predictive Digital Twin — Streamlit Dashboard (v2).

Features:
- (a) Workload-only what-if labelled "predictive, correlational — not causal"
- (b) Limitations panel
- (c) Risk indicator based on interval upper bound (75°C threshold)
- (d) Monitoring / Planning feature-set switch
- (e) Historical replay time-scrubber

Run with:
    streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ttwin.eval import load_all_metrics, compute_skill_scores

EXPERIMENTS_DIR = ROOT / "experiments"
INTERIM_DIR = ROOT / "data" / "interim"

# ── Page config (MUST be the first Streamlit command) ────────────────────────
st.set_page_config(
    page_title="DC Thermal Twin v2",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.warning("⚠️ PRELIMINARY — offline prototype. Results come from pre-computed CSV files.")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .metric-card {
        background: linear-gradient(135deg, #1e2a3a 0%, #0f1923 100%);
        border: 1px solid #2d4a6e;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin: 4px;
    }
    .metric-value { font-size: 2.0rem; font-weight: 700; color: #4fc3f7; }
    .metric-label { font-size: 0.85rem; color: #90caf9; margin-top: 4px; }
    .risk-badge-normal {
        background-color: #1b5e20; color: #a5d6a7; padding: 6px 14px;
        border-radius: 20px; font-weight: 600; display: inline-block;
    }
    .risk-badge-elevated {
        background-color: #e65100; color: #ffe0b2; padding: 6px 14px;
        border-radius: 20px; font-weight: 600; display: inline-block;
    }
    .risk-badge-critical {
        background-color: #b71c1c; color: #ffcdd2; padding: 6px 14px;
        border-radius: 20px; font-weight: 600; display: inline-block;
    }
    .disclaimer-box {
        background-color: #261b0d; border-left: 5px solid #ff9800;
        padding: 14px 18px; border-radius: 6px; margin: 16px 0;
        color: #ffe0b2; font-size: 0.95rem;
    }
    .section-header {
        font-size: 1.25rem; font-weight: 600; color: #e3f2fd;
        border-left: 4px solid #4fc3f7; padding-left: 10px;
        margin: 20px 0 12px 0;
    }
    [data-testid="stSidebar"] { background: #0d1b2a; }
</style>
""", unsafe_allow_html=True)


def load_server_data(dataset: str, server: str) -> pd.DataFrame | None:
    p = INTERIM_DIR / dataset / f"{server}.parquet"
    if not p.exists():
        p_raw = list((INTERIM_DIR / dataset).glob(f"*{server}*.parquet"))
        if p_raw:
            p = p_raw[0]
        else:
            return None
    return pd.read_parquet(p)


# ── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.title("🌡️ Thermal Twin v2")
st.sidebar.markdown("**Predictive Digital Shadow & Uncertainty Monitor**")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Overview & Ablation",
        "🔮 What-If Simulator (Planning)",
        "⏱️ Historical Replay & Risk",
        "📐 Conformal Coverage",
        "⚠️ Limitations Panel",
        "ℹ️ System Architecture",
    ],
)

st.sidebar.markdown("---")
regime_mode = st.sidebar.radio(
    "Active Operational Mode",
    ["Monitoring Mode (Telemetry + Temp Lags)", "Planning Mode (Workload Only)"],
)
st.sidebar.caption("Switch between active operational telemetry and planning-regime what-if projections.")


# ── PAGE 1: Overview & Ablation ──────────────────────────────────────────────
if page == "📊 Overview & Ablation":
    st.title("📊 Model Evaluation & Ablation Overview")
    st.markdown("""
    Evaluation of the Predictive Digital Twin across forecast horizons (10 min, 1 hour, 6 hours)
    under strict chronological, spatial GroupKFold, and strict spatiotemporal splits.
    """)

    # Look for v2 ablation results
    v2_split_a = EXPERIMENTS_DIR / "v2_ablation" / "results_split_a.csv"
    if v2_split_a.exists():
        df_a = pd.read_csv(v2_split_a)
        st.markdown('<div class="section-header">Split A (Chronological 70/10/20) — Benchmark Results</div>', unsafe_allow_html=True)

        kpis = st.columns(4)
        pers_1h = df_a[(df_a["feature_set"] == "S0") & (df_a["horizon"] == "1h")]["mae"].values
        lgb_s1_1h = df_a[(df_a["feature_set"] == "S1") & (df_a["model"] == "lightgbm") & (df_a["horizon"] == "1h")]["mae"].values
        lgb_s4_1h = df_a[(df_a["feature_set"] == "S4") & (df_a["model"] == "lightgbm") & (df_a["horizon"] == "1h")]["mae"].values

        with kpis[0]:
            p_val = f"{pers_1h[0]:.3f}°C" if len(pers_1h) else "—"
            st.markdown(f'<div class="metric-card"><div class="metric-value">{p_val}</div><div class="metric-label">Persistence (1h MAE)</div></div>', unsafe_allow_html=True)
        with kpis[1]:
            s1_val = f"{lgb_s1_1h[0]:.3f}°C" if len(lgb_s1_1h) else "—"
            st.markdown(f'<div class="metric-card"><div class="metric-value">{s1_val}</div><div class="metric-label">S1 LightGBM (1h MAE)</div></div>', unsafe_allow_html=True)
        with kpis[2]:
            s4_val = f"{lgb_s4_1h[0]:.3f}°C" if len(lgb_s4_1h) else "—"
            st.markdown(f'<div class="metric-card"><div class="metric-value">{s4_val}</div><div class="metric-label">S4 LightGBM (1h MAE)</div></div>', unsafe_allow_html=True)
        with kpis[3]:
            st.markdown(f'<div class="metric-card"><div class="metric-value">0.0% to +0.2%</div><div class="metric-label">Skill vs Persistence</div></div>', unsafe_allow_html=True)

        st.dataframe(df_a, use_container_width=True)
    else:
        st.info("Ablation results are running in background. Displaying legacy metrics:")
        metrics_df = load_all_metrics(EXPERIMENTS_DIR)
        if len(metrics_df) > 0:
            st.dataframe(metrics_df, use_container_width=True)


# ── PAGE 2: What-If Simulator (Planning Regime) ──────────────────────────────
elif page == "🔮 What-If Simulator (Planning)":
    st.title("🔮 Workload What-If Simulator (Planning Regime)")

    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ ILLUSTRATIVE PROJECTION — SIMPLIFIED DEMONSTRATION HEURISTIC</strong><br>
        This what-if simulator demonstrates UI interaction using an <strong>illustrative linear correlation formula</strong>
        (derived from exploratory S2 feature weights: 45°C + 0.28*CPU + 0.015*RAM + 0.1*VMs).
        It is <strong>NOT a live trained model checkpoint inference</strong>, is not an interventional causal model,
        and does not model active cooling dynamics or fan controller responses.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("Simulate thermal impacts of planned workload shifts when temperature history is unavailable (brand-new server or hypothetical workload):")

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        base_cpu = st.slider("Baseline CPU Load (%)", 0.0, 100.0, 35.0, 1.0)
        cpu_delta = st.slider("Workload Delta (±20% Shift)", -20.0, 20.0, 0.0, 1.0)
        sim_cpu = np.clip(base_cpu + cpu_delta, 0.0, 100.0)
    with col_w2:
        ram_used = st.slider("RAM Used (GB)", 10.0, 256.0, 64.0, 2.0)
        vms = st.slider("Running VMs", 0, 50, 8, 1)

    # Correlational planning model heuristic derived from S2 workload-only model
    # Baseline idle ~ 45°C + ~0.28°C per 1% CPU load + RAM contribution
    # ILLUSTRATIVE — not model output
    pred_base_temp = 45.0 + 0.28 * base_cpu + 0.015 * ram_used + 0.1 * vms
    pred_sim_temp = 45.0 + 0.28 * sim_cpu + 0.015 * ram_used + 0.1 * vms
    delta_temp = pred_sim_temp - pred_base_temp

    st.markdown('<div class="section-header">Projected Thermal Outcome</div>', unsafe_allow_html=True)
    res_cols = st.columns(3)
    with res_cols[0]:
        st.metric("Base Workload Temp", f"{pred_base_temp:.1f} °C", f"{base_cpu:.0f}% CPU")
    with res_cols[1]:
        st.metric("Simulated Workload Temp", f"{pred_sim_temp:.1f} °C", f"{sim_cpu:.0f}% CPU")
    with res_cols[2]:
        color = "inverse" if delta_temp > 0 else "normal"
        st.metric("Predicted Temp Shift", f"{delta_temp:+.2f} °C", f"{cpu_delta:+.0f}% ΔCPU", delta_color=color)

    # Uncertainty band around planning projection (+/- 4.8°C based on S2 MAE)
    st.caption("Planning uncertainty: Illustrative ±4.8°C band shown (matches typical S2 planning MAE scale in offline experiments).")


# ── PAGE 3: Historical Replay & Risk Monitor ─────────────────────────────────
elif page == "⏱️ Historical Replay & Risk":
    st.title("⏱️ Historical Replay & Offline Thermal Risk Monitor")
    st.markdown("""
    Scrub through historical telemetry step-by-step. The **Risk Indicator** assesses
    offline thermal safety margins by comparing the **90% Conformal Interval Upper Bound**
    against the critical safety threshold ($75.0^\circ\text{C}$).
    """)

    st.info("ℹ️ **ILLUSTRATIVE REPLAY PROTOTYPE**: Predictions and interval bounds in this scrubber evaluate an illustrative demonstration function (q_hat = 1.80°C with sinusoidal perturbation). These are synthetic display values for UI prototyping, not verified offline model inferences. This prototype has no live telemetry feed and no physical actuator connection.")

    ds_dir = INTERIM_DIR / "dataset1"
    servers = sorted([f.stem for f in ds_dir.glob("*.parquet") if not f.stem.startswith("_")]) if ds_dir.exists() else []

    sel_server = st.selectbox("Select Target Server", servers if servers else ["qh2-rcc120"])
    df_srv = load_server_data("dataset1", sel_server)

    if df_srv is not None and len(df_srv) > 500:
        df_srv["Time"] = pd.to_datetime(df_srv["Time"])
        df_slice = df_srv.iloc[1000:1500].reset_index(drop=True)

        step_idx = st.slider("Timeline Step (Historical Replay)", 0, len(df_slice) - 1, 250, 1)

        cur_row = df_slice.iloc[step_idx]
        cur_time = cur_row["Time"]
        cur_temp = cur_row["CPU_Temperature"]
        cur_load = cur_row["CPU_Load"]

        # Conformal interval simulation (+/- 1.8°C at 90%)
        # ILLUSTRATIVE — not model output; q_hat is fixed scalar, not from conformal calibration
        q_hat = 1.80
        pred_temp = cur_temp + np.sin(step_idx / 10.0) * 0.3
        upper_bound = pred_temp + q_hat
        lower_bound = pred_temp - q_hat

        # Risk Indicator Evaluation
        CRITICAL_THRESHOLD = 75.0
        ELEVATED_THRESHOLD = 70.0

        if upper_bound >= CRITICAL_THRESHOLD:
            risk_status = "CRITICAL RISK"
            risk_class = "risk-badge-critical"
            risk_msg = f"Upper bound ({upper_bound:.1f}°C) exceeds critical threshold ({CRITICAL_THRESHOLD}°C). Informational alert — prototype has no physical cooling actuator connection."
        elif upper_bound >= ELEVATED_THRESHOLD:
            risk_status = "ELEVATED"
            risk_class = "risk-badge-elevated"
            risk_msg = f"Upper bound ({upper_bound:.1f}°C) exceeds warning threshold ({ELEVATED_THRESHOLD}°C)."
        else:
            risk_status = "NORMAL"
            risk_class = "risk-badge-normal"
            risk_msg = "Operating within safe thermal margins."

        st.markdown(f"""
        ### Current Status: <span class="{risk_class}">{risk_status}</span>
        **Timestamp:** `{cur_time}` | {risk_msg}
        """, unsafe_allow_html=True)

        kpis = st.columns(4)
        with kpis[0]:
            st.metric("Observed Temp", f"{cur_temp:.1f} °C")
        with kpis[1]:
            st.metric("Predicted Temp (1h)", f"{pred_temp:.1f} °C")
        with kpis[2]:
            st.metric("90% Upper Bound", f"{upper_bound:.1f} °C")
        with kpis[3]:
            st.metric("CPU Load", f"{cur_load:.1f}%")

        # Interactive trajectory plot with conformal band
        fig, ax = plt.subplots(figsize=(11, 4))
        history_window = df_slice.iloc[max(0, step_idx - 60):min(len(df_slice), step_idx + 30)]

        ax.plot(history_window["Time"], history_window["CPU_Temperature"], label="Actual CPU Temp", color="#4fc3f7", linewidth=1.5)
        ax.axhline(CRITICAL_THRESHOLD, color="#e57373", linestyle="--", linewidth=1.2, label=f"Critical Threshold ({CRITICAL_THRESHOLD}°C)")

        # Current point and prediction interval
        ax.scatter([cur_time], [cur_temp], color="#ffffff", s=60, zorder=5)
        ax.errorbar([cur_time], [pred_temp], yerr=[[pred_temp - lower_bound], [upper_bound - pred_temp]],
                    fmt='o', color="#ffb74d", capsize=5, label="Illustrative 1h Conformal Band (q_hat=1.8°C)", zorder=6)

        ax.set_ylabel("Temperature (°C)")
        ax.set_title(f"Historical Telemetry Trajectory with Illustrative Uncertainty Envelope — {sel_server}")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)


# ── PAGE 4: Conformal Coverage ───────────────────────────────────────────────
elif page == "📐 Conformal Coverage":
    st.title("📐 Conformal Prediction & Calibration Monitor")
    st.markdown("""
    Rigorous verification of uncertainty coverage guarantees under distribution shift.
    """)

    v2_unc_dir = EXPERIMENTS_DIR / "v2_uncertainty"
    comp_file = v2_unc_dir / "conformal_methods_comparison.csv"
    if comp_file.exists():
        st.markdown('<div class="section-header">Methods Comparison (Split C Unseen Servers)</div>', unsafe_allow_html=True)
        st.dataframe(pd.read_csv(comp_file), use_container_width=True)

        reg_file = v2_unc_dir / "coverage_by_regime_and_bin.csv"
        if reg_file.exists():
            st.markdown('<div class="section-header">Coverage Breakdown by Operating Regime & Temperature Bin</div>', unsafe_allow_html=True)
            st.dataframe(pd.read_csv(reg_file), use_container_width=True)


# ── PAGE 5: Limitations Panel ────────────────────────────────────────────────
elif page == "⚠️ Limitations Panel":
    st.title("⚠️ Methodological & Engineering Limitations")
    st.markdown("""
    The following limitations must be acknowledged when interpreting or deploying this Predictive Digital Twin:

    1. **Autoregressive Dominance & Persistence Parity:**
       * In short-to-medium horizon monitoring (10 min to 1 hour), server temperature is heavily governed by thermal inertia.
       * Point prediction models using delta targets achieve approximate parity with the simple persistence baseline ($T(t+h) \approx T(t)$), but do not provide substantial accuracy margins over persistence in stationary regimes.
    2. **Unmeasured Facility & Meteorological Telemetry:**
       * The Melbourne research cloud dataset contains no ambient outdoor weather feeds (ambient wet-bulb/dry-bulb temperatures) nor chiller loop / BMS HVAC setpoint telemetry.
       * All temporal drift must be described as operational drift; attributing changes to "seasonal weather" is **empirically unverified**.
    3. **Non-Causal Nature of Fan Telemetry:**
       * In modern servers, chassis fan speeds operate under autonomous firmware closed-loop control (PID / BMC).
       * Fan speed increases *in response* to heat; treating fan speed as an independent explanatory knob without physical intervention is causally unsound.
    4. **Cross-Hardware Transfer Barriers:**
       * Dell PowerEdge R840 (dense 2U 4-socket server) and Dell PowerEdge C6320 (2U multi-node modular sled) have fundamentally distinct aerodynamic paths and thermal masses.
       * Raw transfer without local fine-tuning incurs significant bias in planning settings.
    5. **Conformal Coverage Disparities on Extreme Regimes:**
       * Constant-width conformal intervals suffer under-coverage during high-load ($10\text{--}50\%$) and high-temperature ($\ge 60^\circ\text{C}$) spikes.
       * Adaptive normalized intervals or CQR are strictly necessary to avoid false security during thermal hazard events.
    """)


# ── PAGE 6: System Architecture ──────────────────────────────────────────────
elif page == "ℹ️ System Architecture":
    st.title("ℹ️ System Architecture & Data Lineage")
    st.markdown("""
    ### Pipeline Specifications
    * **Feature Engineering:** Current-state $T(t)$, historical lags $[1, 2, 3, 6, 12]$, rolling causal windows ending at $t$.
    * **Target Formulation:** Incremental delta $\Delta T = T(t+h) - T(t)$ for monitoring; absolute temperature for planning.
    * **Evaluation Protocol:** GroupKFold 5-fold cross-validation by server; strict chronological 70/10/20 holdouts with 6-hour embargo buffers.
    * **Hardware Telemetry Trace:** Melbourne Cloud Data Center (Zenodo 10069402, CC BY 4.0; Ilager et al., UCC 2023).
    """)
