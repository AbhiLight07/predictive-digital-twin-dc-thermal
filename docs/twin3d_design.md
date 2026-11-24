# docs/twin3d_design.md - 3D digital twin design (supplied by the human; do not rewrite)

## 1. Purpose and honesty
A 3D, offline, model-driven predictive digital twin of the Melbourne research-cloud trace. Colours, halos, forecasts and what-if responses come from the saved models through the local API. The data has no rack positions, cooling units, room/outdoor temperature or per-VM data, so the layout is **schematic**. Fan spin, glow and airflow streaks are artistic encodings of recorded sensor values, not airflow physics. Banner (always visible): "Offline replay of recorded telemetry - schematic layout, not physical rack positions - thresholds illustrative".

## 2. Signal -> visual mapping
| Source (API field) | Visual element | Scaling | Real or artistic |
|---|---|---|---|
| `temp_eff` (max of CPU1, CPU2), or forecast `point` when horizon > 0 | Glow strip colour on the server slab | `t = clamp((T - legend.t_lo) / (legend.t_hi - legend.t_lo), 0, 1)` through the temperature ramp | value real, colour artistic |
| `forecast.lower`, `forecast.upper` | Halo shell around the slab | `halo = 1 + 0.35 * clamp(width / legend.w_ref, 0, 1)`, `width = upper - lower` | width real, shell artistic |
| `forecast.risk` (upper bound vs thresholds) | Beacon ring on server and rack top | normal none, warning amber pulse, critical red pulse | real rule, artistic ring |
| `power` | Power LED intensity | `0.15 + 0.85 * clamp((P - p_lo) / (p_hi - p_lo), 0, 1)` | value real |
| `fans[0..3]` (Cluster 1 only) | Four fan discs | `omega = clamp(rpm / rpm_ref, 0, 1) * 12 rad/s` (visual cap, shown in legend as "visual scaling, not true RPM") | rpm real, speed scaled |
| `inlet` (Cluster 1 only) | Intake streak colour | same temperature ramp | value real |
| mean fan rpm | Airflow streak speed | proportional, capped | artistic |
| `cpu1`, `cpu2` | Two CPU blocks in anatomy view | same temperature ramp | value real |
| missing or stale | Grey, no animation | never zero | - |
| compare error (twin - reality) | Overlay colour | diverging ramp, +-3 C saturates | real |
Cluster 2 has no fans or inlet: grey, static parts labelled "no fan sensor" / "no inlet sensor".

## 3. Colours
- Temperature ramp (colour-blind-safe, viridis-like): 0.00 #440154, 0.25 #3b528b, 0.50 #21918c, 0.75 #5ec962, 1.00 #fde725.
- Risk: normal #2ca25f, warning #fdae61, critical #d7191c, unknown #9e9e9e.
- Error (diverging): twin below reality #2166ac, zero #f7f7f7, twin above #b2182b.
- `/api/meta` returns `legend` per cluster: `t_lo` (5th percentile), `t_hi` (99.5th percentile), `p_lo`, `p_hi`, `rpm_ref`, `w_ref` (95th percentile of interval widths), computed from data, not hard-coded.

## 4. Layout (deterministic, schematic)
Hosts sorted by numeric id. Racks of 20 hosts, 2U each (40U of a 42U rack). Rack index `r = floor(i / 20)`, slot `s = i mod 20` (bottom to top). Units in metres: rack pitch 0.9 along x; rack frame 0.6 wide, 1.0 deep, 1.87 tall; server slab 0.48 x 0.0845 x 0.72; slab centre y = `0.05 + s * 0.0889 + 0.0445`. Cluster 1 row at z = 0, Cluster 2 row at z = -4; fronts face +z. Rack labels "C1-R1" and so on. The API returns `rack` and `slot` so the front end and tests agree. Camera presets: overview, cluster 1, cluster 2, selected server.

## 5. Time causality (compare mode)
For target time t and horizon h steps (10 min each): reality(t) = observed value at t; twin(t) = forecast row with `issue_time = t - h*10 min`, `horizon = h`, `target_time = t`, built only from data up to `issue_time`; error = twin - reality; `inside = lower <= reality <= upper`. Reality missing -> status missing (grey). Issuing history insufficient -> unknown. Partition badge uses the partition of `issue_time`; train or validation -> "in-sample - not a fair test".

## 6. API (see the 3D-A prompt for the endpoint list)
Per-host snapshot object: `idx, id, status, partition, temp_eff, cpu1, cpu2, inlet|null, fans|null, power, load, ram_used_pct, cores_used, vms, net_rx, net_tx, forecast|null`. Unavailable sensors are `null` with `available: false`, never 0. Floats rounded to 2 decimals; times in UTC ISO.

## 7. Rendering and performance budget
At least 50 fps for about 145 hosts on an integrated GPU; snapshot request under 100 ms (about 40 KB JSON); no per-frame allocation.
- One `InstancedMesh` for slab bodies; one for glow strips (per-instance colour written into a Float32Array when data changes); one for halos (transparent, `depthWrite` false, scale updated only when data changes); one for fan discs with an instanced `omega` attribute and a shader time uniform (no per-frame matrix updates); rack frames in one `LineSegments`; CSS2D labels only for racks and the hovered host.
- Ease colours between snapshots over about 300 ms; label "frames every N min; colours eased".
- Compare mode: scissor viewports sharing geometry and one set of orbit controls. Fall back to a single-viewport toggle below 30 fps.
- Bloom only on glow strips; disable bloom and halos automatically if fps < 30.

## 8. Tests
API schema and causality; layout determinism and unique (rack, slot); unavailable sensors are null; 503 on missing model; Playwright: canvas non-blank, no console errors, selection fills the panel, horizon switch changes the request, compare stats come from the API, what-if slider triggers a request; source check: no `Math.random` or hard-coded temperatures in `twin3d/web/`.

## 9. Risks and fallbacks
Weak GPU: disable bloom and halos. Cache gaps: grey, never interpolated across gaps longer than 30 minutes. Few hosts alerting: show "no alerts", not placeholder rows. If the simulator is low fidelity: visible warning. Never present visuals without the accuracy panel (test MAE vs persistence, coverage).
