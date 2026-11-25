# 3D Digital Twin Frame Rate — Headed GPU vs Software WebGL

Measured via `scripts/measure_twin3d_fps.py` in **Compare mode** (`h=6`, `1280×800` viewport, real D1+D2 `Twin3DService`, 30 s per run sampled at 1 Hz from `window.__twinStats`).

| Mode | Status | Median FPS | Mean FPS | Min–Max FPS | Median Draw Calls | Single-View Fallback | WebGL Renderer |
|---|---|---:|---:|---:|---:|---|---|
| Headed Chromium (GPU) | OK (30 samples) | 60.0 | 60.3 | 60.0–61.0 | 13.0 | False | `ANGLE (Intel, Intel(R) UHD Graphics (0x00008A56) Direct3D11 vs_5_0 ps_5_0, D3D11)` |
| Headless Chromium (SwiftShader) | OK (30 samples) | 6.0 | 6.4 | 3.0–7.0 | 13.0 | True | `ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)` |
