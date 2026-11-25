#!/usr/bin/env python3
"""Measure 3D Digital Twin WebGL frame rate in compare mode (headed GPU vs software SwiftShader).

Starts an in-process FastAPI + Twin3DService server on an ephemeral port, launches
Chromium via Playwright in headed mode (default GPU flags) and headless SwiftShader
mode for comparison, runs compare mode for 30 s each, samples window.__twinStats,
and writes docs/twin3d_fps_gpu.md.
"""

from __future__ import annotations

import socket
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

import uvicorn
from playwright.sync_api import sync_playwright

_REPO = Path(__file__).resolve().parent.parent
_WEB_DIR = _REPO / "twin3d" / "web"
_OUT_DOC = _REPO / "docs" / "twin3d_fps_gpu.md"

sys.path.insert(0, str(_REPO / "src"))
from ttwin.twin3d.server import Twin3DService, create_app  # noqa: E402


def _measure_mode(base_url: str, headed: bool, duration_s: int = 30) -> dict:
    label = "Headed Chromium (GPU)" if headed else "Headless Chromium (SwiftShader)"
    args = [] if headed else [
        "--use-gl=angle",
        "--use-angle=swiftshader",
        "--enable-webgl",
        "--ignore-gpu-blocklist",
    ]
    samples = []
    renderer_name = "unknown"
    vendor_name = "unknown"

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=not headed, args=args)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.goto(base_url, wait_until="networkidle")
            page.wait_for_function(
                "() => document.getElementById('status-overlay').style.display === 'none'",
                timeout=30000,
            )
            page.check("#chk-compare")
            page.wait_for_function(
                "() => window.__twinStats && window.__twinStats.compareMode === true",
                timeout=15000,
            )
            # Warmup 2s so first 1s FPS bucket reflects steady compare rendering
            time.sleep(2.0)

            for _ in range(duration_s):
                st = page.evaluate("() => window.__twinStats")
                samples.append(st)
                renderer_name = st.get("renderer") or renderer_name
                vendor_name = st.get("vendor") or vendor_name
                time.sleep(1.0)
        finally:
            browser.close()

    fps_vals = [float(s["fps"]) for s in samples if s and s.get("fps") is not None]
    draw_vals = [int(s["drawCalls"]) for s in samples if s and s.get("drawCalls") is not None]
    last = samples[-1] if samples else {}
    return {
        "label": label,
        "ok": True,
        "n_samples": len(fps_vals),
        "renderer": renderer_name,
        "vendor": vendor_name,
        "fps_min": min(fps_vals) if fps_vals else None,
        "fps_median": round(statistics.median(fps_vals), 1) if fps_vals else None,
        "fps_mean": round(statistics.mean(fps_vals), 1) if fps_vals else None,
        "fps_max": max(fps_vals) if fps_vals else None,
        "draws_median": round(statistics.median(draw_vals), 1) if draw_vals else None,
        "single_viewport_final": last.get("singleViewport"),
        "bloom_final": last.get("bloomEnabled"),
        "halos_final": last.get("halosEnabled"),
    }


def main() -> None:
    print("Loading Twin3DService (d1 + d2)...")
    svc = Twin3DService()
    svc.ensure_all()
    app = create_app(service=svc, web_dir=_WEB_DIR)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            with urllib.request.urlopen(f"{base_url}/api/health", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(0.2)

    results = []
    try:
        for headed in (True, False):
            tag = "headed GPU" if headed else "headless SwiftShader"
            print(f"Measuring 30s compare mode in {tag}...")
            try:
                res = _measure_mode(base_url, headed=headed, duration_s=30)
                print(
                    f"  {res['label']}: median={res['fps_median']} fps "
                    f"(min={res['fps_min']}, max={res['fps_max']}), "
                    f"draws={res['draws_median']}, renderer={res['renderer']}"
                )
                results.append(res)
            except Exception as exc:
                print(f"  {tag} failed: {exc}")
                results.append({
                    "label": "Headed Chromium (GPU)" if headed else "Headless Chromium (SwiftShader)",
                    "ok": False,
                    "error": str(exc),
                })
    finally:
        server.should_exit = True
        thread.join(timeout=5)

    lines = [
        "# 3D Digital Twin Frame Rate — Headed GPU vs Software WebGL",
        "",
        "Measured via `scripts/measure_twin3d_fps.py` in **Compare mode** (`h=6`, `1280×800` viewport, real D1+D2 `Twin3DService`, 30 s per run sampled at 1 Hz from `window.__twinStats`).",
        "",
        "| Mode | Status | Median FPS | Mean FPS | Min–Max FPS | Median Draw Calls | Single-View Fallback | WebGL Renderer |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    headed_failed = False
    for r in results:
        if r.get("ok"):
            lines.append(
                f"| {r['label']} | OK ({r['n_samples']} samples) | {r['fps_median']} | {r['fps_mean']} | "
                f"{r['fps_min']}–{r['fps_max']} | {r['draws_median']} | {r['single_viewport_final']} | `{r['renderer']}` |"
            )
        else:
            headed_failed = headed_failed or ("Headed" in r["label"])
            lines.append(f"| {r['label']} | FAILED | n/a | n/a | n/a | n/a | n/a | `{r.get('error', '')}` |")

    if headed_failed:
        lines.extend([
            "",
            "## Manual GPU Measurement Steps (Headed Mode Unavailable)",
            "1. Start the server: `python scripts/run_twin3d.py`",
            "2. Enable **Compare** in the bottom control bar.",
            "3. Open browser DevTools console and read `window.__twinStats` over 30 seconds.",
        ])

    _OUT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {_OUT_DOC}")


if __name__ == "__main__":
    main()
