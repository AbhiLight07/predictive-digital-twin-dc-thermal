"""Predictive Digital Twin — Research Demonstration Studio.

A dedicated, scientific research application demonstrating actual datasets,
verifiable preprocessing pipelines, trained model inference, benchmark results,
generalisation protocols, cold-start adaptation, and conformal uncertainty quantification.

Run with:
    streamlit run app/research_studio.py
"""

import os
import sys
from pathlib import Path
import streamlit as st

# Configure Matplotlib temporary cache directory for Windows compatibility
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", "C:/tmp")

# ── 1. Page Configuration (MUST be the first Streamlit command) ───────────────
st.set_page_config(
    page_title="Thermal Twin — Research Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Path Setup ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ttwin.research_studio.data_registry import DataRegistry
from ttwin.research_studio.dataset_loader import DatasetLoader
from ttwin.research_studio.ui.styles import STUDIO_CSS
from ttwin.research_studio.ui.project_overview_ui import render_project_overview
from ttwin.research_studio.ui.dataset_explorer_ui import render_dataset_explorer
from ttwin.research_studio.ui.preprocessing_lab_ui import render_preprocessing_lab
from ttwin.research_studio.ui.inference_lab_ui import render_inference_lab
from ttwin.research_studio.ui.benchmark_lab_ui import render_benchmark_lab
from ttwin.research_studio.ui.generalisation_lab_ui import render_generalisation_lab
from ttwin.research_studio.ui.adaptation_lab_ui import render_adaptation_lab
from ttwin.research_studio.ui.uncertainty_lab_ui import render_uncertainty_lab
from ttwin.research_studio.ui.reproduction_centre_ui import render_reproduction_centre
from ttwin.research_studio.ui.conclusions_limitations_ui import render_conclusions_and_limitations

# Apply custom dark research theme CSS
st.markdown(STUDIO_CSS, unsafe_allow_html=True)

# ── 3. Initialize Shared Data Layer ───────────────────────────────────────────
@st.cache_resource
def get_registry_and_loader():
    registry = DataRegistry(root_dir=ROOT)
    loader = DatasetLoader(registry)
    return registry, loader

registry, loader = get_registry_and_loader()

# ── 4. Sidebar Navigation & Controls ──────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding: 10px 0 16px 0;">
    <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; display: flex; align-items: center; gap: 8px;">
        <span>🔬</span> THERMAL TWIN
    </div>
    <div style="font-size: 0.78rem; color: #38bdf8; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">
        Research Studio v3.0
    </div>
</div>
""", unsafe_allow_html=True)

presentation_mode = st.sidebar.checkbox("🎯 Evaluator Presentation Mode", value=False)
if presentation_mode:
    st.sidebar.caption("Presentation mode active: Recommended walkthrough order highlighted.")

pages = [
    "1. 🔬 Project Overview & Hypotheses",
    "2. 🗂️ Telemetry Dataset Explorer",
    "3. 🧪 Preprocessing & Feature Lab",
    "4. 🎯 Real Model Inference Lab",
    "5. 📊 Model Benchmark & Ablation Lab",
    "6. 🌐 Spatial Generalisation Lab",
    "7. 🔄 Adaptation & Transfer Lab",
    "8. 📐 Conformal Uncertainty Lab",
    "9. ⚡ Experiment Reproduction Centre",
    "10. 📑 Conclusions & Disclosures",
]

page_choice = st.sidebar.radio("Navigation", pages, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.76rem; color: #64748b; line-height: 1.4;">
    <strong>Verification Status:</strong><br>
    • Data: Zenodo 10069402 (UCC 2023)<br>
    • Protocol: Frozen (Split A/B/C)<br>
    • Checkpoints: Genuine offline inference<br>
    • ACTUATION: Zero physical cooling control
</div>
""", unsafe_allow_html=True)

# ── 5. Page Dispatcher ────────────────────────────────────────────────────────
if "1. 🔬" in page_choice:
    render_project_overview(registry)
elif "2. 🗂️" in page_choice:
    render_dataset_explorer(registry, loader)
elif "3. 🧪" in page_choice:
    render_preprocessing_lab(registry, loader)
elif "4. 🎯" in page_choice:
    render_inference_lab(registry, loader)
elif "5. 📊" in page_choice:
    render_benchmark_lab(registry, loader)
elif "6. 🌐" in page_choice:
    render_generalisation_lab(registry, loader)
elif "7. 🔄" in page_choice:
    render_adaptation_lab(registry, loader)
elif "8. 📐" in page_choice:
    render_uncertainty_lab(registry, loader)
elif "9. ⚡" in page_choice:
    render_reproduction_centre(registry)
elif "10. 📑" in page_choice:
    render_conclusions_and_limitations(registry)
