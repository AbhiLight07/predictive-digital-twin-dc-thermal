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
import pandas as pd
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


def test_no_mock_formulas_in_app_or_web():
    """Fail if legacy mock heuristics (0.28 *, np.sin(, or hard-coded q_hat) appear in app/ or twin3d/web/."""
    q_hat_re = re.compile(r"\bq_hat\s*=\s*[-+]?\d+(?:\.\d+)?")
    targets = []
    app_dir = _REPO / "app"
    if app_dir.exists():
        targets.extend(f for f in app_dir.rglob("*") if f.is_file() and f.suffix in (".py", ".js", ".ts", ".html"))
    targets.extend(f for f in _iter_web_sources() if f.suffix in (".py", ".js", ".ts", ".html"))
    for f in targets:
        text = f.read_text(encoding="utf-8")
        rel = f.relative_to(_REPO)
        assert "0.28 *" not in text, f"Forbidden mock formula '0.28 *' in {rel}"
        assert "np.sin(" not in text, f"Forbidden mock 'np.sin(' in {rel}"
        m = q_hat_re.search(text)
        assert m is None, f"Forbidden hard-coded '{m.group(0)}' in {rel}"


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
def test_3db_and_3dc_end_to_end_webgl(live_server_url, tmp_path):
    from playwright.sync_api import sync_playwright

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

        # Save smoke overview screenshot in tmp_path (preserve real-data docs/twin3d_screens/)
        smoke_png = tmp_path / "01_overview.png"
        page.screenshot(path=str(smoke_png))
        assert smoke_png.stat().st_size > 10_000

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
        page.wait_for_function(
            "() => window.__twin3d.getStats().compareMode && window.__twin3d.getStats().compareStats !== null",
            timeout=10000,
        )
        stats_js = page.evaluate("() => window.__twin3d.getStats()")
        assert stats_js["compareMode"] is True
        assert stats_js["compareStats"] is not None
        mae_dom = page.locator("#cmp-mae").inner_text()
        inside_dom = page.locator("#cmp-inside").inner_text()
        assert str(stats_js["compareStats"]["mae"]) in mae_dom
        assert str(stats_js["compareStats"]["coverage_pct"]) in inside_dom
        page.uncheck("#chk-compare")
        page.wait_for_function("() => !window.__twin3d.getStats().compareMode", timeout=10000)
        page.wait_for_timeout(400)

        # 6. 3D-C: Move What-If slider -> POST /api/whatif sent & ghost colour changes
        is_whatif_post = lambda r: r.request.method == "POST" and "/api/whatif" in r.url
        with page.expect_response(is_whatif_post, timeout=15000) as resp_10:
            page.locator("#wi-load").fill("10")
            page.locator("#wi-load").dispatch_event("input")
        assert resp_10.value.status == 200
        page.wait_for_timeout(150)
        ghost_before = page.evaluate("() => window.__twin3d.getStats()")

        with page.expect_response(is_whatif_post, timeout=15000) as resp_95:
            page.locator("#wi-load").fill("95")
            page.locator("#wi-load").dispatch_event("input")
        assert resp_95.value.status == 200
        page.wait_for_function(
            "(before) => window.__twin3d.getStats().ghostVisible && window.__twin3d.getStats().ghostColorHex !== before",
            arg=ghost_before["ghostColorHex"],
            timeout=10000,
        )
        ghost_after = page.evaluate("() => window.__twin3d.getStats()")
        assert ghost_after["ghostVisible"] is True
        assert ghost_after["ghostColorHex"] != ghost_before["ghostColorHex"], (
            f"Ghost colour did not change ({ghost_before['ghostColorHex']} -> {ghost_after['ghostColorHex']})"
        )
        wi_text = page.locator("#whatif-result").inner_text()
        assert "correlational - not causal" in wi_text.lower()

        # 6b. What-If hour & weekday selector override sends explicit calendar params
        with page.expect_response(
            lambda r: is_whatif_post(r) and (r.request.post_data_json or {}).get("hour") == 18,
            timeout=15000,
        ) as resp_hour:
            page.select_option("#wi-hour", "18")
        assert resp_hour.value.status == 200

        with page.expect_response(
            lambda r: is_whatif_post(r) and (r.request.post_data_json or {}).get("weekday") == 5,
            timeout=15000,
        ) as resp_wd:
            page.select_option("#wi-weekday", "5")
        assert resp_wd.value.status == 200
        cal_payload = resp_wd.value.request.post_data_json or {}
        assert cal_payload.get("hour") == 18 and cal_payload.get("weekday") == 5, (
            f"Calendar override missing in POST /api/whatif payload: {cal_payload}"
        )

        # 7. 3D-D: Anatomy view on D1 host (real fans + inlet) then D2 host (static grey "no fan sensor" / "no inlet sensor")
        page.click("#btn-inspect-server")
        page.wait_for_function("() => window.__twin3d.getStats().anatomyMode === true", timeout=10000)
        d1_anat = page.evaluate("() => window.__twin3d.getStats()")
        assert d1_anat["anatomyMode"] is True
        assert d1_anat["anatomyHasFans"] is True
        assert d1_anat["anatomyHasInlet"] is True
        assert "not airflow physics" in page.locator("#anatomy-overlay").inner_text().lower()
        assert page.locator("#anatomy-chart-wrap").is_visible()

        page.click("#btn-back-hall")
        page.wait_for_function("() => window.__twin3d.getStats().anatomyMode === false", timeout=10000)
        page.select_option("#sel-cluster", "d2")
        page.wait_for_function(
            """() => {
                const s = window.__twin3d.getStats();
                return s.selectedHostKey && s.selectedHostKey.startsWith('d2|')
                    && document.getElementById('legend-extra').innerText.includes('D2: no fan sensors');
            }""",
            timeout=15000,
        )
        assert "D2: no fan sensors" in page.locator("#legend-extra").inner_text()

        page.click("#btn-inspect-server")
        page.wait_for_function(
            "() => window.__twin3d.getStats().anatomyMode === true && !window.__twin3d.getStats().anatomyHasFans",
            timeout=10000,
        )
        d2_anat = page.evaluate("() => window.__twin3d.getStats()")
        assert d2_anat["anatomyMode"] is True
        assert d2_anat["anatomyHasFans"] is False
        assert d2_anat["anatomyHasInlet"] is False
        assert d2_anat["anatomyFanLabel"] == "no fan sensor"
        assert d2_anat["anatomyInletLabel"] == "no inlet sensor"

        # 8. 3D-D: Palette switch (viridis -> cividis)
        page.select_option("#sel-palette", "cividis")
        page.wait_for_timeout(300)
        assert page.evaluate("() => window.__twin3d.getStats().palette") == "cividis"

        browser.close()

    assert console_errors == [], f"Console errors: {console_errors}"
    assert page_errors == [], f"Page errors: {page_errors}"


@needs_playwright
def test_panel_no_undefined_nan_null_across_views(tmp_path):
    """Scan #panel text across fleet, forecast, compare, what-if, anatomy, and missing-obs host for 'undefined', 'NaN', or 'null'."""
    from playwright.sync_api import sync_playwright

    d1 = _SyntheticEngine("d1")
    d2 = _SyntheticEngine("d2")
    c1 = _build_synthetic_cache(d1)
    # Remove observations for qh2-rcc121 at/before 03:00Z while keeping cache forecast with persistence=NaN
    miss_host = "qh2-rcc121"
    t_target = pd.Timestamp("2019-05-01 03:00:00")
    t_issue = pd.Timestamp("2019-05-01 02:00:00")
    d1._data = d1._data[~((d1._data["Series"] == miss_host) & (d1._data["Time"] <= t_target))].copy()
    c1.loc[(c1["host"] == miss_host) & (c1["time"] == t_issue) & (c1["horizon"] == 6), "persistence"] = np.nan

    svc = Twin3DService(engines={"d1": d1, "d2": d2}, caches={"d1": c1, "d2": _build_synthetic_cache(d2)})
    app = create_app(service=svc, web_dir=_WEB_DIR)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    import urllib.request
    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            with urllib.request.urlopen(f"{base}/api/health", timeout=1) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(0.1)

    bad_re = re.compile(r"\b(undefined|NaN|null)\b", re.IGNORECASE)

    def _assert_panel_clean(page, view_name):
        txt = page.locator("#panel").inner_text()
        m = bad_re.search(txt)
        assert m is None, f"Found '{m.group(0)}' in #panel during {view_name} view:\n{txt}"
        return txt

    try:
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
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.goto(base, wait_until="networkidle")
            page.wait_for_function("() => document.getElementById('status-overlay').style.display === 'none'", timeout=15000)
            page.evaluate("() => window.__twin3d.jumpToTime('2019-05-01T03:00:00Z')")
            page.wait_for_timeout(400)

            # 1. Fleet view (horizon = 0)
            page.select_option("#sel-horizon", "0")
            page.wait_for_timeout(350)
            _assert_panel_clean(page, "fleet(h=0)")

            # 2. Forecast view (horizon = 6) + select missing-obs host
            page.select_option("#sel-horizon", "6")
            page.wait_for_timeout(350)
            page.evaluate("() => window.__twin3d.selectHost('d1|qh2-rcc121')")
            page.wait_for_timeout(350)
            fc_txt = _assert_panel_clean(page, "forecast(h=6, missing host)")
            assert "missing" in fc_txt.lower()
            assert "not available" in fc_txt.lower()

            # 3. Compare view (missing-obs host with null persistence)
            page.check("#chk-compare")
            page.wait_for_function("() => window.__twin3d.getStats().compareMode && window.__twin3d.getStats().compareStats !== null", timeout=10000)
            page.evaluate("() => window.__twin3d.selectHost('d1|qh2-rcc121')")
            page.wait_for_timeout(350)
            cmp_txt = _assert_panel_clean(page, "compare(missing host)")
            detail_txt = page.locator("#host-detail").inner_text()
            assert "missing" in detail_txt.lower(), f"Expected status missing, got:\n{detail_txt}"
            assert "not available" in detail_txt.lower()
            assert "n/a" in detail_txt.lower()
            assert "ok" not in detail_txt.lower().splitlines()[0]

            # Also check an ok host in compare mode
            page.evaluate("() => window.__twin3d.selectHost('d1|qh2-rcc122')")
            page.wait_for_timeout(350)
            _assert_panel_clean(page, "compare(ok host)")

            # 4. What-If view
            page.uncheck("#chk-compare")
            page.wait_for_timeout(250)
            page.locator("#wi-load").fill("80")
            page.locator("#wi-load").dispatch_event("input")
            page.wait_for_timeout(350)
            _assert_panel_clean(page, "what-if")

            # 5. Anatomy view
            page.click("#btn-inspect-server")
            page.wait_for_function("() => window.__twin3d.getStats().anatomyMode === true", timeout=10000)
            page.wait_for_timeout(250)
            _assert_panel_clean(page, "anatomy")

            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=5)
