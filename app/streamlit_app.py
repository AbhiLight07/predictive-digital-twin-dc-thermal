"""Deprecated legacy mock dashboard."""

MSG = "Deprecated - run `python scripts/run_twin3d.py` or `streamlit run app/research_studio.py`"
print(MSG)

try:
    import streamlit as st

    st.set_page_config(page_title="Deprecated Dashboard", layout="centered")
    st.warning(MSG)
except Exception:
    pass
