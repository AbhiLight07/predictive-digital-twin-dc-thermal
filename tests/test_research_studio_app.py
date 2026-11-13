"""Tests for the Research Studio Streamlit application (app/research_studio.py).

Verifies:
1. File exists and st.set_page_config is the first Streamlit command.
2. No hardcoded Windows paths (d:/ or D:\\) anywhere in app or research_studio source files.
3. No claims of physical cooling throttle or physical actuation.
4. All UI modules import cleanly and their render functions exist.
5. app/streamlit_app.py is strictly unmodified.
"""

from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parent.parent
STUDIO_PATH = ROOT / "app" / "research_studio.py"
LEGACY_APP_PATH = ROOT / "app" / "streamlit_app.py"


def test_research_studio_exists():
    assert STUDIO_PATH.exists(), f"Missing {STUDIO_PATH}"


def test_set_page_config_is_first_st_call_in_studio():
    """Verify that st.set_page_config is the very first Streamlit call in app/research_studio.py."""
    content = STUDIO_PATH.read_text(encoding="utf-8")

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


def test_no_hardcoded_windows_paths():
    """Verify no hardcoded Windows drive letters (d:/ or D:\\) in studio or research_studio src."""
    studio_files = [STUDIO_PATH] + list((ROOT / "src" / "ttwin" / "research_studio").glob("**/*.py"))

    for f in studio_files:
        content = f.read_text(encoding="utf-8")
        assert "d:/" not in content.lower(), f"Hardcoded 'd:/' found in {f}"
        assert "d:\\" not in content.lower(), f"Hardcoded 'd:\\' found in {f}"


def test_no_physical_actuation_claims():
    """Verify no claims of physical cooling control or real-time closed loop in research studio."""
    content = STUDIO_PATH.read_text(encoding="utf-8")
    assert "cooling throttle required" not in content.lower()
    assert "physical actuation connected" not in content.lower()


def test_legacy_streamlit_app_is_unchanged():
    """Verify that git status shows zero modifications to app/streamlit_app.py."""
    res = subprocess.run(
        ["git", "diff", "--exit-code", str(LEGACY_APP_PATH)],
        cwd=str(ROOT),
        capture_output=True,
    )
    assert res.returncode == 0, "app/streamlit_app.py has been modified!"


def test_ui_components_importable():
    """Verify that all research studio UI modules import cleanly."""
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

    assert callable(render_project_overview)
    assert callable(render_dataset_explorer)
    assert callable(render_preprocessing_lab)
    assert callable(render_inference_lab)
    assert callable(render_benchmark_lab)
    assert callable(render_generalisation_lab)
    assert callable(render_adaptation_lab)
    assert callable(render_uncertainty_lab)
    assert callable(render_reproduction_centre)
    assert callable(render_conclusions_and_limitations)
