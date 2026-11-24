"""Offline evidence capture and validation suite for the 3D Predictive Digital Twin.

Loads the real D1/D2 Twin3DService, starts an in-process uvicorn server,
captures 6 screenshots into docs/twin3d_screens/, records docs/twin3d_demo.webm,
runs offline + 60s soak + API latency + 3-day compare benchmarks, and writes
measured results into docs/twin3d_validation.md.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import uvicorn
from PIL import Image
from playwright.sync_api import sync_playwright

_REPO = Path(__file__).resolve().parent.parent
_WEB_DIR = _REPO / "twin3d" / "web"
_SCREENS_DIR = _REPO / "docs" / "twin3d_screens"
_VIDEO_PATH = _REPO / "docs" / "twin3d_demo.webm"
_VALIDATION_DOC = _REPO / "docs" / "twin3d_validation.md"

sys.path.insert(0, str(_REPO / "src"))
from ttwin.twin3d.server import Twin3DService, create_app  # noqa: E402


def _check_static_hygiene() -> dict:
    math_random = 0
    cdn_urls = 0
    hardcoded_temps = 0
    files_checked = 0
    cdn_re = re.compile(r"https?://|cdn\.jsdelivr|unpkg\.com", re.IGNORECASE)
    temp_re = re.compile(r"\b(4[0-9]|[5-9][0-9]|100)\.\d+\b")

    for f in _WEB_DIR.rglob("*"):
        if not f.is_file() or "vendor" in f.parts:
            continue
        if f.suffix not in (".js", ".html", ".css", ".ts"):
            continue
        files_checked += 1
        text = f.read_text(encoding="utf-8")
        if "Math.random" in text:
            math_random += 1
        if cdn_re.search(text):
            cdn_urls += 1
        if f.suffix in (".js", ".ts"):
            for line in text.splitlines():
                st = line.strip()
                if st.startswith("//") or st.startswith("*"):
                    continue
                if any(k in line.lower() for k in ("temp", "celsius", "threshold")) and temp_re.search(line):
                    hardcoded_temps += 1

    return {
        "files_checked": files_checked,
        "math_random": math_random,
        "cdn_urls": cdn_urls,
        "hardcoded_temps": hardcoded_temps,
    }


def _http_json(url: str, payload: dict | None = None) -> tuple[float, dict, int]:
    t0 = time.perf_counter()
    if payload is not None:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    else:
        req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as r:
        raw = r.read()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return dt_ms, json.loads(raw.decode("utf-8")), len(raw)


def _measure_api_latencies(base: str) -> list[dict]:
    endpoints = [
        ("GET /api/health", f"{base}/api/health", None),
        ("GET /api/meta", f"{base}/api/meta", None),
        ("GET /api/snapshot (d1, h=6)", f"{base}/api/snapshot?cluster=d1&t=2019-05-25T03:50:00Z&horizon=6", None),
        ("GET /api/snapshot (all, h=6)", f"{base}/api/snapshot?cluster=all&t=2019-05-25T03:50:00Z&horizon=6", None),
        ("GET /api/compare (d1, h=6)", f"{base}/api/compare?cluster=d1&t=2019-05-25T09:50:00Z&horizon=6", None),
        (
            "GET /api/series (d1, 24h)",
            f"{base}/api/series?cluster=d1&host=qh2-rcc120&start=2019-05-24T03:50:00Z&end=2019-05-25T03:50:00Z&horizon=6",
            None,
        ),
        ("GET /api/alerts (all)", f"{base}/api/alerts?cluster=all&t=2019-05-27T16:20:00Z", None),
        (
            "POST /api/whatif (d1)",
            f"{base}/api/whatif",
            {
                "cluster": "d1",
                "host": "qh2-rcc120",
                "workload": {"cpu_load": 65.0, "cores_used": 32.0, "ram_used": 40.0, "vms": 8.0, "net_rx": 150000.0, "net_tx": 120000.0},
            },
        ),
    ]
    rows = []
    for name, url, body in endpoints:
        # Warmup
        _http_json(url, body)
        times = []
        size_bytes = 0
        for _ in range(10):
            dt_ms, _, sz = _http_json(url, body)
            times.append(dt_ms)
            size_bytes = sz
        rows.append({
            "endpoint": name,
            "p50_ms": round(float(np.percentile(times, 50)), 2),
            "p95_ms": round(float(np.percentile(times, 95)), 2),
            "bytes": size_bytes,
        })
    return rows


def _measure_compare_days(base: str) -> list[dict]:
    queries = [
        ("d1", "2019-05-05T12:20:00Z", 6),
        ("d1", "2019-05-25T12:20:00Z", 6),
        ("d1", "2019-06-15T12:20:00Z", 6),
        ("d2", "2019-05-28T12:20:00Z", 6),
        ("d2", "2019-06-06T12:20:00Z", 6),
        ("d2", "2019-06-15T12:20:00Z", 6),
    ]
    results = []
    for cid, ts, h in queries:
        _, data, _ = _http_json(f"{base}/api/compare?cluster={cid}&t={ts}&horizon={h}")
        st = data["stats"]
        results.append({
            "cluster": cid.upper(),
            "target_time": ts,
            "issue_time": data["issue_time"],
            "partition": data["partition"],
            "mae": st["mae"],
            "coverage_pct": st["coverage_pct"],
            "n_inside": st["n_inside"],
            "n_valid": st["n_valid"],
        })
    return results


def _verify_png_nonblank(path: Path) -> float:
    raw = path.read_bytes()
    assert len(raw) > 10_000, f"Screenshot too small: {path.name} ({len(raw)} bytes)"
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    std = float(np.asarray(img, dtype=np.float32).std())
    assert std > 4.0, f"Screenshot appears blank: {path.name} (std={std:.2f})"
    return round(std, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the 3D predictive digital twin.")
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Re-record docs/twin3d_demo.webm (disabled by default to avoid git bloat).",
    )
    args = parser.parse_args()

    print("Loading real D1/D2 Twin3DService...")
    svc = Twin3DService()
    svc.ensure_all()
    app = create_app(service=svc, web_dir=_WEB_DIR)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"{base}/api/health", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(0.2)

    try:
        hygiene = _check_static_hygiene()
        latencies = _measure_api_latencies(base)
        compare_days = _measure_compare_days(base)

        _SCREENS_DIR.mkdir(parents=True, exist_ok=True)
        video_tmp_dir = _REPO / "docs" / "_tmp_video"
        if video_tmp_dir.exists():
            shutil.rmtree(video_tmp_dir)
        video_tmp_dir.mkdir(parents=True, exist_ok=True)

        console_errors = []
        blocked_external = []
        screen_checks = []
        soak_samples = []
        video_ok = False
        video_note = ""

        chrome_args = [
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--enable-webgl",
            "--ignore-gpu-blocklist",
            "--enable-precise-memory-info",
        ]

        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True, args=chrome_args)

            # 1. Walkthrough + Screenshots (+ optional Video recording via --record-video)
            use_video = bool(args.record_video)
            if use_video:
                try:
                    ctx = browser.new_context(
                        viewport={"width": 1440, "height": 900},
                        record_video_dir=str(video_tmp_dir),
                        record_video_size={"width": 1280, "height": 800},
                    )
                    page = ctx.new_page()
                except Exception as vid_exc:
                    use_video = False
                    video_note = f"Video recording skipped ({vid_exc})"
                    ctx = browser.new_context(viewport={"width": 1440, "height": 900})
                    page = ctx.new_page()
            else:
                ctx = browser.new_context(viewport={"width": 1440, "height": 900})
                page = ctx.new_page()
                if _VIDEO_PATH.exists():
                    video_ok = True
                    video_note = f"Kept existing {round(_VIDEO_PATH.stat().st_size / 1024, 1)} KB webm (--record-video omitted)"
                else:
                    video_note = "Skipped (--record-video not passed)"

            def _route(route):
                url = route.request.url
                if url.startswith(base) or url.startswith("data:") or url.startswith("blob:"):
                    route.continue_()
                else:
                    blocked_external.append(url)
                    route.abort()

            ctx.route("**/*", _route)
            page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)

            page.goto(base, wait_until="networkidle")
            page.wait_for_function("() => document.getElementById('status-overlay').style.display === 'none'", timeout=20000)
            page.wait_for_timeout(3000)

            # Shot 1: Overview at 2019-05-25T03:50:00Z
            p1 = _SCREENS_DIR / "01_overview.png"
            page.screenshot(path=str(p1))
            screen_checks.append(("01_overview.png", _verify_png_nonblank(p1), p1.stat().st_size))

            # Jump to alert-rich timestamp 2019-05-27T16:20:00Z (28 active alerts across D1/D2)
            page.evaluate("() => window.__twin3d.jumpToTime('2019-05-27T16:20:00Z')")
            page.wait_for_timeout(4000)
            p6 = _SCREENS_DIR / "06_alerts.png"
            page.screenshot(path=str(p6))
            screen_checks.append(("06_alerts.png", _verify_png_nonblank(p6), p6.stat().st_size))

            # Click first alert to fly camera & show forecast halos at h=36
            if page.locator(".alert-item").count() > 0:
                page.locator(".alert-item").first.click()
                page.wait_for_timeout(2000)
            page.select_option("#sel-horizon", "36")
            page.wait_for_timeout(4000)
            p2 = _SCREENS_DIR / "02_forecast_halos.png"
            page.screenshot(path=str(p2))
            screen_checks.append(("02_forecast_halos.png", _verify_png_nonblank(p2), p2.stat().st_size))

            # Shot 3: Compare mode (h=6) + error overlay glimpse
            page.select_option("#sel-horizon", "6")
            page.wait_for_timeout(1500)
            page.select_option("#sel-camera", "c1")
            page.check("#chk-compare")
            page.wait_for_timeout(4500)
            p3 = _SCREENS_DIR / "03_compare_mode.png"
            page.screenshot(path=str(p3))
            screen_checks.append(("03_compare_mode.png", _verify_png_nonblank(p3), p3.stat().st_size))
            cmp_perf = page.evaluate("() => window.__twin3d.getStats()")

            # Toggle error overlay briefly for video
            page.check("#chk-error-overlay")
            page.wait_for_timeout(3500)
            page.uncheck("#chk-error-overlay")
            page.uncheck("#chk-compare")
            page.wait_for_timeout(1500)

            # Shot 4: What-If Ghost on new/high-load server
            page.locator("#wi-load").fill("92")
            page.locator("#wi-load").dispatch_event("input")
            page.locator("#wi-vms").fill("25")
            page.locator("#wi-vms").dispatch_event("input")
            page.wait_for_timeout(4000)
            p4 = _SCREENS_DIR / "04_whatif_ghost.png"
            page.screenshot(path=str(p4))
            screen_checks.append(("04_whatif_ghost.png", _verify_png_nonblank(p4), p4.stat().st_size))

            # Shot 5: D1 Server Anatomy View
            page.select_option("#sel-cluster", "d1")
            page.wait_for_timeout(1500)
            page.click("#btn-inspect-server")
            page.wait_for_timeout(5000)
            p5 = _SCREENS_DIR / "05_anatomy.png"
            page.screenshot(path=str(p5))
            screen_checks.append(("05_anatomy.png", _verify_png_nonblank(p5), p5.stat().st_size))

            # Brief D2 anatomy glimpse in video ("no fan sensor" / "no inlet sensor")
            page.click("#btn-back-hall")
            page.wait_for_timeout(1500)
            page.select_option("#sel-cluster", "d2")
            page.wait_for_timeout(1500)
            page.click("#btn-inspect-server")
            page.wait_for_timeout(4000)
            page.click("#btn-back-hall")
            page.wait_for_timeout(1500)

            ctx.close()

            # Save recorded video
            if use_video:
                vids = list(video_tmp_dir.glob("*.webm"))
                if vids and vids[0].stat().st_size > 10_000:
                    shutil.move(str(vids[0]), str(_VIDEO_PATH))
                    video_ok = True
                    video_note = f"Recorded {round(_VIDEO_PATH.stat().st_size / 1024, 1)} KB webm"
                else:
                    video_note = "Playwright video file missing or empty"
            shutil.rmtree(video_tmp_dir, ignore_errors=True)

            # 2. Dedicated 60-second soak test (continuous replay of All Clusters)
            soak_ctx = browser.new_context(viewport={"width": 1280, "height": 800})
            soak_ctx.route("**/*", _route)
            spage = soak_ctx.new_page()
            spage.goto(base, wait_until="networkidle")
            spage.wait_for_function("() => document.getElementById('status-overlay').style.display === 'none'", timeout=20000)
            spage.select_option("#sel-speed", "600")
            spage.click("#btn-play")

            for sec in (0, 30, 60):
                if sec > 0:
                    spage.wait_for_timeout(30_000)
                st = spage.evaluate("""() => ({
                    ...window.__twin3d.getStats(),
                    heapMB: performance && performance.memory
                        ? Math.round(performance.memory.usedJSHeapSize / (1024 * 1024) * 100) / 100
                        : null
                })""")
                soak_samples.append({"t_sec": sec, **st})
            soak_ctx.close()
            browser.close()

        # Write docs/twin3d_validation.md with strictly measured values
        lines = [
            "# 3D Digital Twin — Validation Report",
            "",
            "All measurements below were collected automatically by `scripts/validate_twin3d.py` against the real D1 (76 hosts) and D2 (60 hosts) engines, saved LightGBM models, and precomputed forecast caches.",
            "",
            "## 1. Static Source Hygiene & Offline Execution",
            f"- Non-vendor web files inspected (`twin3d/web/`, excluding `vendor/`): **{hygiene['files_checked']}**",
            f"- `Math.random` occurrences: **{hygiene['math_random']}**",
            f"- External/CDN URL occurrences (`http://`, `https://`, `cdn.jsdelivr`, `unpkg`): **{hygiene['cdn_urls']}**",
            f"- Hard-coded temperature literals: **{hygiene['hardcoded_temps']}**",
            f"- Playwright offline route test (non-localhost requests blocked): **{len(blocked_external)} external requests attempted**, **{len(console_errors)} console errors**",
            "",
            "## 2. API Latency & Payload Size (10 warm calls per endpoint)",
            "| Endpoint | p50 (ms) | p95 (ms) | Payload (bytes) |",
            "|---|---|---|---|",
        ]
        for r in latencies:
            lines.append(f"| `{r['endpoint']}` | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['bytes']} |")

        lines += [
            "",
            "## 3. 60-Second Replay Soak Test (Headless Chrome SwiftShader WebGL, 136 hosts)",
            "> Note: Headless Chromium uses software ANGLE/SwiftShader rendering, so FPS below reflects CPU WebGL rasterisation rather than hardware GPU throughput.",
            "",
            "| Elapsed (s) | FPS | Draw Calls | Hosts Rendered | JS Heap (MB) |",
            "|---|---|---|---|---|",
        ]
        for s in soak_samples:
            heap_str = f"{s['heapMB']:.2f}" if s["heapMB"] is not None else "n/a"
            lines.append(f"| {s['t_sec']} | {s['fps']} | {s['drawCalls']} | {s['hostCount']} | {heap_str} |")

        if soak_samples[0]["heapMB"] is not None and soak_samples[-1]["heapMB"] is not None:
            delta_mb = round(soak_samples[-1]["heapMB"] - soak_samples[0]["heapMB"], 2)
            lines.append(f"\n- Observed JS heap delta (`t=0s` to `t=60s`): **{delta_mb:+.2f} MB**")
        lines.append(f"- Measured compare-mode frame rate / draw calls: **{cmp_perf['fps']} fps · {cmp_perf['drawCalls']} draws**")

        lines += [
            "",
            "## 4. Compare Mode Stats Across 3 Test-Period Days (`horizon=6` steps / 1 h)",
            "| Cluster | Target Time | Issue Time (`t - 1h`) | Partition | Frame MAE (°C) | Inside CQR (%) | n_inside / n_valid |",
            "|---|---|---|---|---|---|---|",
        ]
        for c in compare_days:
            lines.append(
                f"| {c['cluster']} | `{c['target_time']}` | `{c['issue_time']}` | {c['partition']} | {c['mae']} | {c['coverage_pct']}% | {c['n_inside']} / {c['n_valid']} |"
            )

        lines += [
            "",
            "## 5. Visual Evidence Captures",
            "| File | Size (bytes) | RGB Pixel Std (non-blank > 4.0) |",
            "|---|---|---|",
        ]
        for fname, std, sz in screen_checks:
            lines.append(f"| `docs/twin3d_screens/{fname}` | {sz} | {std} |")
        lines.append(f"- Walkthrough video (`docs/twin3d_demo.webm`): **{'PASS' if video_ok else 'FAIL'}** ({video_note})")

        _VALIDATION_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("VALIDATION COMPLETE:", _VALIDATION_DOC)
        print("SOAK:", json.dumps(soak_samples))
        print("VIDEO:", video_ok, video_note)
    finally:
        server.should_exit = True
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
