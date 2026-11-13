"""Conclusions and Methodological Limitations UI (Stage 10).

Summarizes core scientific findings, research questions, and explicit engineering disclosures:
- Offline telemetry prototype status (no live feeds, no physical cooling actuators)
- Distinguishing illustrative replay from verified model inferences
- Provenance clarifications (Melbourne Cloud trace vs CRESCO7)
- Meteorological disclosures (no outdoor ambient weather data)
"""

import streamlit as st
from ttwin.research_studio.data_registry import DataRegistry


def render_conclusions_and_limitations(registry: DataRegistry):
    st.markdown('<div class="studio-header"><div class="studio-title">📑 Research Conclusions & Methodological Disclosures</div><div class="studio-subtitle">Authoritative scientific synthesis, verified findings, and mandatory operational limitations</div></div>', unsafe_allow_html=True)

    tab_findings, tab_disclosures, tab_future = st.tabs([
        "🔬 Key Research Findings",
        "⚠️ Explicit Disclosures & Disclaimers",
        "🚀 Research Roadmap & Future Directions",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: Key Research Findings
    # ══════════════════════════════════════════════════════════════════════════
    with tab_findings:
        st.markdown('<div class="lab-section-title">Summary of Core Scientific Findings</div>', unsafe_allow_html=True)

        st.markdown("""
        ### 1. Thermal Persistence Dominance in Monitoring
        * **Finding:** In operational monitoring regimes (10 min to 1 hour), physical thermal inertia accounts for **>99% of short-term thermal variance**.
        * **Empirical Metric:** Machine learning twins achieve modest skill over naive persistence: **+0.59%** on Dataset 1 (0.828°C vs 0.833°C MAE) and **+2.83%** on Dataset 2 (0.781°C vs 0.804°C MAE).
        * **Statistical Caveat:** Server-level 95% bootstrap confidence intervals cross zero on both datasets (`[-0.38%, +0.17%]` on D1). Across individual physical nodes, predictive superiority is **not statistically significant**.

        ### 2. High Operational Value for Workload Planning
        * **Finding:** In planning regimes where current temperatures are unavailable (e.g. brand-new servers or scheduler what-if projections), machine learning twins provide substantial operational utility.
        * **Empirical Metric:** Pretrained zero-shot planning achieves **5.03°C MAE**, delivering a **43.2% error reduction** over the unconditioned fleet mean baseline (8.87°C MAE).

        ### 3. Hardware Telemetry & Sensor Utility
        * **Finding:** Cooling fan speeds and server power telemetry contribute negligibly to short-term predictive accuracy (<0.0003°C MAE difference).
        * **Physical Cause:** Chassis fans operate under firmware closed-loop control (BMC/PID) without facility BMS coordination, making fan speed reactive to past heat rather than proactively informative.

        ### 4. Generalisation & Data Leakage Penalties
        * **Finding:** Naive random row splitting leaks temporal autocorrelation and spatial identity, underestimating planning error by **17.6% (S2)** and **18.5% (S3)**.
        * **Protocol:** 5-fold GroupKFold by server correctly models deployment to unseen hardware, scaling from **0.435°C** (10m) $\\to$ **0.918°C** (1h) $\\to$ **1.817°C** (6h).

        ### 5. Uncertainty Quantification Under Distribution Shift
        * **Finding:** Constant-width conformal intervals under-cover severely during thermal hazard regimes (**83.06% coverage** on active workloads).
        * **Solution:** Conformalized Quantile Regression (CQR) achieves **91.19% aggregate coverage** with a narrower average width (**3.68°C**) and superior conditional coverage across hot servers (92.67%).
        """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: Explicit Disclosures
    # ══════════════════════════════════════════════════════════════════════════
    with tab_disclosures:
        st.markdown('<div class="lab-section-title">Mandatory Technical & Scope Disclosures</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="tech-alert-amber">
            <h4>⚠️ MANDATORY SCIENTIFIC & OPERATIONAL DISCLOSURES</h4>
            <ol>
                <li><strong>Offline Research Prototype Status:</strong> This platform is strictly an offline research demonstrator evaluating historical trace datasets. It is <strong>NOT a production data centre control system</strong>.</li>
                <li><strong>No Live Telemetry Integration:</strong> No live telemetry feeds, real-time BMS streams, or college campus data centre sensors are connected to this dashboard.</li>
                <li><strong>Zero Physical Cooling Actuation:</strong> The system has <strong>NO physical cooling control</strong>. It cannot adjust chiller setpoints, throttle server fans, modulate CRAC units, or actuate physical valves.</li>
                <li><strong>Legacy Illustrative Replay Distinctions:</strong> The what-if simulator and historical replay scrubber in the original <code>streamlit_app.py</code> utilize illustrative synthetic formulas ($45.0 + 0.28\\cdot\\text{CPU}$ and sinusoidal offsets). They must not be confused with trained model checkpoint inference.</li>
                <li><strong>Data Provenance (Ilager et al., UCC 2023):</strong> The telemetry originates from the University of Melbourne Research Cloud Trace (Zenodo 10069402). It must <strong>NOT be confused with CRESCO7</strong> or any unverified private traces.</li>
                <li><strong>Absence of Outdoor Weather Measurements:</strong> The dataset contains no outdoor meteorological measurements (ambient wet-bulb/dry-bulb temperatures). Any temporal drift across months must be attributed strictly to operational and workload shifts; claiming weather effects is <strong>empirically unverified</strong>.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: Research Roadmap
    # ══════════════════════════════════════════════════════════════════════════
    with tab_future:
        st.markdown('<div class="lab-section-title">Future Research & Engineering Directions</div>', unsafe_allow_html=True)

        st.markdown("""
        1. **Facility-Coupled Spatiotemporal Graph Models:**
           - Integrating server chassis telemetry with facility chilled-water loop telemetry and CFD (Computational Fluid Dynamics) boundary conditions.
        2. **Causal Fan Intervention Modeling:**
           - Decoupling autonomous BMC fan controller responses from external thermal disturbances using interventional causal inference.
        3. **Adaptive Online Conformal Prediction:**
           - Implementing time-decayed rolling conformal calibration (e.g. Gibbs & Candès, 2021) to eliminate weekly coverage drift under operational non-stationarity.
        4. **Closed-Loop Workload Scheduler Integration:**
           - Connecting the planning regime digital twin directly to Kubernetes / Slurm scheduler queues to avoid proactive thermal hotspots before job dispatch.
        """)
