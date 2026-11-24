"""Playwright UI and static hygiene tests for twin3d/web (3D-B & 3D-C).

Runs against an in-process FastAPI + synthetic engine server on an ephemeral port,
using installed Chrome with headless WebGL (ANGLE SwiftShader) and offline routing.
Never opens/views screenshots directly; validates pixel non-blankness in Python.
"""

from __future__ import annotations

import io
import os
import re
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import uvicorn

_REPO = Path(__file__).resolve().parent.parent
_WEB_DIR = _REPO / "twin3d" / "web"
_SCREENS_DIR = _REPO / "docs" / "twin3d_screens"

sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_twin3d_api import _SyntheticEngine, _build_synthetic_cache  # noqa: E402
from ttwin.twin3d.server import Twin3DService, create_app  # noqa: E402


# ── Static source hygiene checks (no browser needed) ─────────────────


def _iter_web_sources():
    for f in _WEB_DIR.rglob("*"):
        if not f.is_file():
            continue
        if "vendor" in f.parts:
            continue
        if f.suffix in (".js", ".html", ".css", ".ts"):
            yield f


def test_no_math_random_in_web():
    for f in _iter_web_sources():
        text = f.read_text(encoding="utf-8")
        assert "Math.random" not in text, f"Forbidden Math.random in {f.relative_to(_REPO)}"


def test_no_cdn_urls_in_web():
    cdn_re = re.compile(r"https?://|cdn\.jsdelivr|unpkg\.com", re.IGNORECASE)
    for f in _iter_web_sources():
        text = f.read_text(encoding="utf-8")
        m = cdn_re.search(text)
        assert m is None, f"Forbidden external URL '{m.group(0)}' in {f.relative_to(_REPO)}"


def test_no_hardcoded_temps_in_web():
    temp_re = re.compile(r"\b(4[0-9]|[5-9][0-9]|100)\.\d+\b")
    for f in _iter_web_sources():
        if f.suffix not in (".js", ".ts"):
            continue
        for idx, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            st = line.strip()
            if st.startswith("//") or st.startswith("*"):
                continue
            if any(k in line.lower() for k in ("temp", "celsius", "threshold")):
                assert not temp_re.search(line), f"Hard-coded temp in {f.name}:{idx}: {line}"


def test_vendor_three_files_present():
    vendor = _WEB_DIR / "vendor" / "three"
    assert (vendor / "three.module.js").exists()
    assert (vendor / "addons" / "controls" / "OrbitControls.js").exists()
    assert (vendor / "addons" / "renderers" / "CSS2DRenderer.js").exists()
    assert (vendor / "addons" / "postprocessing" / "EffectComposer.js").exists()
    assert (vendor / "addons" / "postprocessing" / "UnrealBloomPass.js").exists()


# ── Playwright WebGL fixtures ────────────────────────────────────────


def _chrome_available() -> bool:
    for p in (
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ):
        if Path(p).exists():
            return True
    return False


needs_playwright = pytest.mark.skipif(
    not _chrome_available(), reason="System Chrome/Edge not found for Playwright"
)


@pytest.fixture(scope="module")
def live_server_url():
    d1 = _SyntheticEngine("d1")
    d2 = _SyntheticEngine("d2")
    svc = Twin3DService(
        engines={"d1": d1, "d2": d2},
        caches={"d1": _build_synthetic_cache(d1), "d2": _build_synthetic_cache(d2)},
    )
    app = create_app(service=svc, web_dir=_WEB_DIR)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait until server responds
    import urllib.request
    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            with urllib.request.urlopen(f"{base}/api/health", timeout=1) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(0.1)
    yield base
    server.should_exit = True
    thread.join(timeout=5)


@needs_playwright
def test_3db_and_3dc_end_to_end_webgl(live_server_url):
    from playwright.sync_api import sync_playwright

    _SCREENS_DIR.mkdir(parents=True, exist_ok=True)
    console_errors = []
    page_errors = []
    blocked_external = []
    api_requests = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome",
            headless=True,
            args=[
                "--use-gl=angle",
                "--use-angle=swiftshader",
                "--enable-webgl",
                "--ignore-gpu-blocklist",
            ],
        )
        context = browser.new_context(viewport={"width": 1280, "height": 800})

        # Enforce strict offline test: abort any request leaving localhost
        def _route_handler(route):
            url = route.request.url
            if url.startswith(live_server_url) or url.startswith("data:") or url.startswith("blob:"):
                if "/api/" in url:
                    api_requests.append((route.request.method, url))
                route.continue_()
            else:
                blocked_external.append(url)
                route.abort()

        context.route("**/*", _route_handler)
        page = context.new_page()
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        page.goto(live_server_url, wait_until="networkidle")
        page.wait_for_function(
            "() => document.getElementById('status-overlay').style.display === 'none'",
            timeout=15000,
        )
        page.wait_for_timeout(1200)

        # 1. Banner & no blocked external CDN requests
        banner_txt = page.locator("#banner").inner_text()
        assert "schematic layout" in banner_txt.lower()
        assert blocked_external == [], f"External requests attempted: {blocked_external}"

        # 2. Canvas non-blank pixel variance check (using PIL / raw PNG via Playwright)
        canvas_shot = page.locator("#canvas-container canvas").first.screenshot()
        from PIL import Image
        img = Image.open(io.BytesIO(canvas_shot)).convert("RGB")
        arr = np.asarray(img, dtype=np.float32)
        assert arr.std() > 4.0, f"WebGL canvas looks blank (std={arr.std():.2f})"

        # Save initial 3D-B overview screenshot (do not open/read it manually)
        page.screenshot(path=str(_SCREENS_DIR / "01_overview.png"))
        assert (_SCREENS_DIR / "01_overview.png").stat().st_size > 10_000

        # 3. Host detail panel populated & sparkline drawn
        detail_txt = page.locator("#host-detail").inner_text()
        assert "Temp effective" in detail_txt
        assert "°C" in detail_txt

        # 4. 3D-C: Horizon switch changes API request
        reqs_before = len(api_requests)
        page.select_option("#sel-horizon", "36")
        page.wait_for_timeout(500)
        new_urls = [u for _, u in api_requests[reqs_before:]]
        assert any("horizon=36" in u for u in new_urls), f"Expected horizon=36 request, got {new_urls}"

        # 5. 3D-C: Enable compare mode -> /api/compare called & stats box reflects API numbers
        page.check("#chk-compare")
        page.wait_for_timeout(800)
        stats_js = page.evaluate("() => window.__twin3d.getStats()")
        assert stats_js["compareMode"] is True
        assert stats_js["compareStats"] is not None
        mae_dom = page.locator("#cmp-mae").inner_text()
        inside_dom = page.locator("#cmp-inside").inner_text()
        assert str(stats_js["compareStats"]["mae"]) in mae_dom
        assert str(stats_js["compareStats"]["coverage_pct"]) in inside_dom
        page.uncheck("#chk-compare")
        page.wait_for_timeout(400)

        # 6. 3D-C: Move What-If slider -> POST /api/whatif sent & ghost colour changes
        page.locator("#wi-load").fill("10")
        page.locator("#wi-load").dispatch_event("input")
        page.wait_for_timeout(400)
        ghost_before = page.evaluate("() => window.__twin3d.getStats()")

        wi_before = len(api_requests)
        page.locator("#wi-load").fill("95")
        page.locator("#wi-load").dispatch_event("input")
        page.wait_for_timeout(500)
        wi_reqs = [(m, u) for m, u in api_requests[wi_before:] if m == "POST" and "/api/whatif" in u]
        assert len(wi_reqs) >= 1, "Moving what-if slider did not trigger POST /api/whatif"
        ghost_after = page.evaluate("() => window.__twin3d.getStats()")
        assert ghost_after["ghostVisible"] is True
        assert ghost_after["ghostColorHex"] != ghost_before["ghostColorHex"], (
            f"Ghost colour did not change ({ghost_before['ghostColorHex']} -> {ghost_after['ghostColorHex']})"
        )
        wi_text = page.locator("#whatif-result").inner_text()
        assert "correlational - not causal" in wi_text.lower()

        browser.close()

    assert console_errors == [], f"Console errors: {console_errors}"
    assert page_errors == [], f"Page errors: {page_errors}"
