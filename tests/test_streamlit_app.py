"""Lightweight test to verify Streamlit dashboard script configuration."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_PATH = ROOT / "app" / "streamlit_app.py"


def test_set_page_config_is_first_st_command():
    """Verify that st.set_page_config is the very first Streamlit call in app/streamlit_app.py."""
    assert APP_PATH.exists()
    content = APP_PATH.read_text(encoding="utf-8")

    st_calls = []
    for line_no, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("st.") and not stripped.startswith("#"):
            st_calls.append((line_no, stripped))

    assert len(st_calls) > 0, "No Streamlit calls found"
    first_line_no, first_call = st_calls[0]
    assert first_call.startswith("st.set_page_config"), (
        f"First Streamlit call at line {first_line_no} is '{first_call}', expected st.set_page_config"
    )


def test_no_hardcoded_windows_paths_in_app():
    """Verify no hardcoded Windows drive paths (d:/ or D:\\) in app/streamlit_app.py."""
    content = APP_PATH.read_text(encoding="utf-8")
    assert "d:/" not in content.lower()
    assert "d:\\" not in content.lower()


def test_no_cooling_throttle_claims():
    """Verify no claims of physical cooling throttle or physical actuation."""
    content = APP_PATH.read_text(encoding="utf-8")
    assert "cooling throttle required" not in content.lower()
