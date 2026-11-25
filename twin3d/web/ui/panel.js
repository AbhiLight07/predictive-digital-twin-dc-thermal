/**
 * ui/panel.js — Right inspector panel: Provenance & Accuracy, Fleet Summary,
 * Alerts list, Host Signals & 24h Sparkline + Fan Chart, Live What-If simulator.
 */

function fmt(val, unit = "") {
  if (val == null || val === "" || (typeof val === "number" && Number.isNaN(val))) {
    return '<span class="muted">not available</span>';
  }
  const n = typeof val === "number" ? val : Number(val);
  if (typeof val === "number" && !Number.isFinite(n)) {
    return '<span class="muted">not available</span>';
  }
  return unit ? `${val} ${unit}` : String(val);
}

export class PanelController {
  constructor(callbacks) {
    this.cb = callbacks;
    this.meta = null;
    this.lastCompareData = null;
    this.whatIfDebounce = null;
    this.whatIfTarget = "selected";

    this.elSummary = document.getElementById("fleet-summary");
    this.elAccuracy = document.getElementById("accuracy-panel");
    this.elProvenance = document.getElementById("provenance-unknowns");
    this.elAlerts = document.getElementById("alert-list");
    this.elHostTitle = document.getElementById("host-detail-title");
    this.elHostDetail = document.getElementById("host-detail");
    this.elBtnInspect = document.getElementById("btn-inspect-server");
    this.elCanvasSpark = document.getElementById("canvas-sparkline");
    this.elCanvasFan = document.getElementById("canvas-fan");
    this.elCanvasAnatomy = document.getElementById("canvas-anatomy-chart");

    // What-If controls
    this.elWiTarget = document.getElementById("sel-whatif-target");
    this.elWiCluster = document.getElementById("sel-whatif-cluster");
    this.elWiHour = document.getElementById("wi-hour");
    this.elWiWeekday = document.getElementById("wi-weekday");
    this.elWiLoad = document.getElementById("wi-load");
    this.elWiCores = document.getElementById("wi-cores");
    this.elWiRam = document.getElementById("wi-ram");
    this.elWiVms = document.getElementById("wi-vms");
    this.elWiNet = document.getElementById("wi-net");
    this.elWiResult = document.getElementById("whatif-result");
    this.elCanvasSens = document.getElementById("canvas-sensitivity");

    this._bindEvents();
  }

  _bindEvents() {
    this.elBtnInspect.addEventListener("click", () => this.cb.onInspectSelected());

    this.elWiTarget.addEventListener("change", () => {
      this.whatIfTarget = this.elWiTarget.value;
      this.elWiCluster.disabled = this.whatIfTarget !== "new_server";
      this._triggerWhatIf();
    });
    this.elWiCluster.addEventListener("change", () => this._triggerWhatIf());
    this.elWiHour.addEventListener("change", () => this._triggerWhatIf());
    this.elWiWeekday.addEventListener("change", () => this._triggerWhatIf());

    for (const el of [this.elWiLoad, this.elWiCores, this.elWiRam, this.elWiVms, this.elWiNet]) {
      el.addEventListener("input", () => {
        this._syncSliderLabels();
        clearTimeout(this.whatIfDebounce);
        this.whatIfDebounce = setTimeout(() => this._triggerWhatIf(), 120);
      });
    }
    this._syncSliderLabels();
  }

  _syncSliderLabels() {
    document.getElementById("wi-load-val").textContent = `${this.elWiLoad.value}%`;
    document.getElementById("wi-cores-val").textContent = this.elWiCores.value;
    document.getElementById("wi-ram-val").textContent = `${this.elWiRam.value} GB`;
    document.getElementById("wi-vms-val").textContent = this.elWiVms.value;
    document.getElementById("wi-net-val").textContent = `${Math.round(Number(this.elWiNet.value) / 1000)} KB/s`;
  }

  _triggerWhatIf() {
    const netBytes = Number(this.elWiNet.value);
    this.cb.onWhatIfChange({
      target: this.whatIfTarget,
      cluster: this.elWiCluster.value,
      hour: this.elWiHour.value,
      weekday: this.elWiWeekday.value,
      workload: {
        cpu_load: Number(this.elWiLoad.value),
        cores_used: Number(this.elWiCores.value),
        ram_used: Number(this.elWiRam.value),
        vms: Number(this.elWiVms.value),
        net_rx: netBytes,
        net_tx: netBytes,
      },
    });
  }

  seedWhatIfFromHost(host) {
    if (!host || this.whatIfTarget !== "selected") return;
    if (host.load != null) this.elWiLoad.value = String(Math.round(Math.min(100, Math.max(0, host.load))));
    if (host.cores_used != null) this.elWiCores.value = String(Math.round(host.cores_used));
    if (host.ram_used_gb != null) this.elWiRam.value = String(Math.round(host.ram_used_gb));
    if (host.vms != null) this.elWiVms.value = String(Math.round(host.vms));
    this._syncSliderLabels();
  }

  setMeta(meta, activeCluster = "d1") {
    this.meta = meta;
    this.renderProvenance(activeCluster);
  }

  renderProvenance(activeCluster = "d1") {
    if (!this.meta) return;
    const cid = activeCluster === "all" ? "d1" : activeCluster;
    const cm = this.meta.clusters[cid];
    if (!cm) return;
    const mRows = cm.models
      .map((m) => {
        const tag = m.kind === "planning" ? "Plan" : `h=${m.horizon ?? "n/a"}`;
        const pStr = m.persistence_mae != null ? ` / persist ${m.persistence_mae}` : "";
        const maeStr = m.test_mae != null ? `${m.test_mae}°C` : "n/a";
        const covStr = m.empirical_coverage != null ? `${m.empirical_coverage}%` : "n/a";
        const rollStr = m.kind === "planning" && m.rolling_7d_coverage != null ? ` (roll-7d ${m.rolling_7d_coverage}%)` : "";
        return `<div class="row"><span class="label">${tag} MAE / cov</span><span class="val">${maeStr}${pStr} · ${covStr}${rollStr}</span></div>`;
      })
      .join("");
    const b = cm.partition_boundaries || {};
    this.elAccuracy.innerHTML = `
      <div class="row"><span class="label">Cluster / Version</span><span class="val">${cid.toUpperCase()} · ${cm.model_version || "saved"}</span></div>
      <div class="row"><span class="label">Train end</span><span class="val">${(b.train_end || "--").slice(0, 10)}</span></div>
      <div class="row"><span class="label">Test period</span><span class="val">${(b.test_start || "--").slice(0, 10)} → ${(b.test_end || "--").slice(0, 10)}</span></div>
      ${mRows}
    `;
    this.elProvenance.innerHTML = (this.meta.what_twin_does_not_know || [])
      .map((item) => `<li>${item}</li>`)
      .join("");
  }

  updateFleetSummary(hosts, horizonSteps) {
    const nOk = hosts.filter((h) => h.status === "ok").length;
    const nWarn = hosts.filter((h) => (h.alert_risk || h.forecast?.risk) === "warning").length;
    const nCrit = hosts.filter((h) => (h.alert_risk || h.forecast?.risk) === "critical").length;
    const hLabel = horizonSteps === 0 ? "Now (observed)" : `+${horizonSteps * 10} min (h=${horizonSteps})`;
    this.elSummary.innerHTML = `
      <div class="row"><span class="label">Hosts shown / OK</span><span class="val">${hosts.length} / ${nOk}</span></div>
      <div class="row"><span class="label">Horizon</span><span class="val">${hLabel}</span></div>
      <div class="row"><span class="label"><span class="risk-dot" style="background:#fdae61"></span>Warning</span><span class="val">${nWarn}</span></div>
      <div class="row"><span class="label"><span class="risk-dot" style="background:#d7191c"></span>Critical</span><span class="val">${nCrit}</span></div>
    `;
  }

  updateAlerts(alertsData) {
    const alerts = alertsData?.alerts || [];
    if (alerts.length === 0) {
      this.elAlerts.innerHTML = '<div class="muted">no alerts</div>';
      return;
    }
    this.elAlerts.innerHTML = alerts
      .map(
        (a) => `<div class="alert-item ${a.risk || "unknown"}" data-cluster="${a.cluster || ""}" data-id="${a.id || ""}">
          <strong>${a.id || "n/a"}</strong> (${(a.cluster || "--").toUpperCase()} · ${a.rack_label || "--"}) — ${a.risk || "unknown"}, h=${a.horizon ?? "n/a"}, upper ${a.upper != null ? `${a.upper}°C` : "n/a"}
        </div>`
      )
      .join("");
    this.elAlerts.querySelectorAll(".alert-item").forEach((el) => {
      el.addEventListener("click", () => {
        this.cb.onSelectHostKey(`${el.dataset.cluster}|${el.dataset.id}`, true);
      });
    });
  }

  updateHostDetail(host) {
    if (!host) {
      this.elHostTitle.textContent = "Select a server";
      this.elHostDetail.innerHTML = '<div class="muted">Click any server slab in the 3D racks.</div>';
      this.elBtnInspect.disabled = true;
      return;
    }
    this.elBtnInspect.disabled = false;
    const clusterTag = (host.cluster || "--").toUpperCase();
    const rackLabel = host.rack_label ?? "--";
    const slotLabel = host.slot ?? "--";
    this.elHostTitle.textContent = `${host.id} (${clusterTag} · ${rackLabel} slot ${slotLabel})`;

    const hasObs = host.temp_eff != null || host.reality != null;
    const effStatus = !hasObs ? "missing" : (host.status || "missing");

    const partBadge =
      host.partition && host.partition !== "TEST"
        ? `<span class="badge-warn">${host.partition}: in-sample - not a fair test</span>`
        : `<span class="badge-test">${host.partition || "--"}</span>`;

    const fansArr = Array.isArray(host.fans) ? host.fans : [];
    const fansHtml = host.fans_available
      ? (fansArr.some((v) => v != null && !Number.isNaN(Number(v)))
          ? fansArr.map((v) => (v != null && !Number.isNaN(Number(v)) ? v : "n/a")).join(", ") + " RPM"
          : '<span class="muted">not available</span>')
      : '<span class="muted">no fan sensor (not available)</span>';

    const inletHtml = host.inlet_available
      ? fmt(host.inlet, "°C")
      : '<span class="muted">no inlet sensor (not available)</span>';

    let fcHtml = '<div class="muted">Select horizon > 0 for forecast bounds</div>';
    if (host.forecast) {
      const fc = host.forecast;
      const hNum = Number(fc.horizon);
      const hLabel = fc.horizon != null && Number.isFinite(hNum) ? `+${hNum * 10}m` : "n/a";
      const hasBounds =
        fc.lower != null &&
        fc.upper != null &&
        !Number.isNaN(Number(fc.lower)) &&
        !Number.isNaN(Number(fc.upper));
      const boundsHtml = hasBounds
        ? `[${fc.lower}, ${fc.upper}]°C`
        : '<span class="muted">not available</span>';
      const persistHtml =
        fc.persistence != null && !Number.isNaN(Number(fc.persistence))
          ? `${fc.persistence}°C`
          : '<span class="muted">n/a</span>';
      fcHtml = `
        <div class="row"><span class="label">Point (${hLabel})</span><span class="val">${fmt(fc.point, "°C")}</span></div>
        <div class="row"><span class="label">CQR Interval</span><span class="val">${boundsHtml}</span></div>
        <div class="row"><span class="label">Persistence T(t)</span><span class="val">${persistHtml}</span></div>
        <div class="row"><span class="label">Risk</span><span class="val">${fc.risk || "unknown"}</span></div>
      `;
    }

    this.elHostDetail.innerHTML = `
      <div class="row"><span class="label">Status / Partition</span><span class="val">${effStatus} · ${partBadge}</span></div>
      <div class="row"><span class="label">Temp effective</span><span class="val">${fmt(host.temp_eff, "°C")}</span></div>
      <div class="row"><span class="label">CPU1 / CPU2</span><span class="val">${fmt(host.cpu1, "°C")} / ${fmt(host.cpu2, "°C")}</span></div>
      <div class="row"><span class="label">Inlet temp</span><span class="val">${inletHtml}</span></div>
      <div class="row"><span class="label">Fans 1–4</span><span class="val">${fansHtml}</span></div>
      <div class="row"><span class="label">Power</span><span class="val">${fmt(host.power, "W")}</span></div>
      <div class="row"><span class="label">CPU load / Cores</span><span class="val">${fmt(host.load, "%")} / ${fmt(host.cores_used, "")}</span></div>
      <div class="row"><span class="label">RAM used / VMs</span><span class="val">${fmt(host.ram_used_pct, "%")} / ${fmt(host.vms, "")}</span></div>
      <div class="row"><span class="label">Net RX / TX</span><span class="val">${fmt(host.net_rx, "B/s")} / ${fmt(host.net_tx, "B/s")}</span></div>
      <div class="subhead">Forecast & Fan</div>
      ${fcHtml}
    `;
  }

  drawSeriesCharts(seriesData) {
    this._drawSparkline(this.elCanvasSpark, seriesData);
    this._drawFanChart(this.elCanvasFan, seriesData);
    if (this.elCanvasAnatomy) {
      this._drawSparkline(this.elCanvasAnatomy, seriesData);
    }
  }

  _drawSparkline(canvas, s) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#10131a";
    ctx.fillRect(0, 0, w, h);

    if (!s || !s.times || s.times.length === 0) {
      ctx.fillStyle = "#777";
      ctx.font = "11px sans-serif";
      ctx.fillText("No 24h series", 8, 20);
      return;
    }

    const vals = [];
    for (const arr of [s.actual, s.lower, s.upper, s.point]) {
      for (const v of arr || []) if (v != null) vals.push(v);
    }
    if (vals.length === 0) return;
    const minV = Math.min(...vals) - 1;
    const maxV = Math.max(...vals) + 1;
    const n = s.times.length;
    const xOf = (i) => 6 + (i / Math.max(1, n - 1)) * (w - 12);
    const yOf = (v) => h - 14 - ((v - minV) / Math.max(0.1, maxV - minV)) * (h - 24);

    // Draw forecast interval band where present
    ctx.fillStyle = "rgba(94, 201, 98, 0.22)";
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < n; i++) {
      if (s.upper[i] != null && s.lower[i] != null) {
        if (!started) { ctx.moveTo(xOf(i), yOf(s.upper[i])); started = true; }
        else ctx.lineTo(xOf(i), yOf(s.upper[i]));
      }
    }
    for (let i = n - 1; i >= 0; i--) {
      if (s.upper[i] != null && s.lower[i] != null) {
        ctx.lineTo(xOf(i), yOf(s.lower[i]));
      }
    }
    ctx.closePath();
    ctx.fill();

    // Actual trace
    ctx.strokeStyle = "#fde725";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let move = true;
    for (let i = 0; i < n; i++) {
      if (s.actual[i] == null) { move = true; continue; }
      if (move) { ctx.moveTo(xOf(i), yOf(s.actual[i])); move = false; }
      else ctx.lineTo(xOf(i), yOf(s.actual[i]));
    }
    ctx.stroke();

    ctx.fillStyle = "#aaa";
    ctx.font = "10px sans-serif";
    ctx.fillText(`24h Observed (${minV.toFixed(0)}–${maxV.toFixed(0)}°C)`, 6, 11);
  }

  _drawFanChart(canvas, s) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#10131a";
    ctx.fillRect(0, 0, w, h);
    const fhs = s?.fan_horizons || [];
    if (fhs.length === 0) {
      ctx.fillStyle = "#777";
      ctx.font = "10px sans-serif";
      ctx.fillText("Forecast fan not available", 8, 18);
      return;
    }

    const xs = [0, ...fhs.map((f) => f.horizon_steps)];
    const curActual = [...(s.actual || [])].reverse().find((v) => v != null) ?? fhs[0].persistence;
    const pts = [curActual, ...fhs.map((f) => f.point)];
    const los = [curActual, ...fhs.map((f) => f.lower)];
    const his = [curActual, ...fhs.map((f) => f.upper)];
    const all = [...pts, ...los, ...his].filter((v) => v != null);
    const minV = Math.min(...all) - 1;
    const maxV = Math.max(...all) + 1;
    const xOf = (step) => 12 + (step / 36) * (w - 24);
    const yOf = (v) => h - 14 - ((v - minV) / Math.max(0.1, maxV - minV)) * (h - 24);

    // Fan polygon
    ctx.fillStyle = "rgba(33, 145, 140, 0.30)";
    ctx.beginPath();
    xs.forEach((st, idx) => (idx === 0 ? ctx.moveTo(xOf(st), yOf(his[idx])) : ctx.lineTo(xOf(st), yOf(his[idx]))));
    for (let i = xs.length - 1; i >= 0; i--) ctx.lineTo(xOf(xs[i]), yOf(los[i]));
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = "#5ec962";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    xs.forEach((st, idx) => (idx === 0 ? ctx.moveTo(xOf(st), yOf(pts[idx])) : ctx.lineTo(xOf(st), yOf(pts[idx]))));
    ctx.stroke();

    ctx.fillStyle = "#aaa";
    ctx.font = "10px sans-serif";
    ctx.fillText("Horizon fan (+10m, +1h, +6h)", 6, 11);
  }

  updateWhatIfResult(data) {
    if (!data) return;
    const hasBounds =
      data.lower != null &&
      data.upper != null &&
      !Number.isNaN(Number(data.lower)) &&
      !Number.isNaN(Number(data.upper));
    this.elWiResult.innerHTML = `
      <div class="row"><span class="label">Predicted temp</span><span class="val"><strong>${fmt(data.point, "°C")}</strong></span></div>
      <div class="row"><span class="label">CQR interval</span><span class="val">${hasBounds ? `[${data.lower}, ${data.upper}]°C` : '<span class="muted">not available</span>'}</span></div>
      <div class="row"><span class="label">Risk / Test MAE</span><span class="val">${data.risk || "unknown"} · ${data.test_mae ?? "--"}°C (cov ${data.coverage ?? "--"}%)</span></div>
      <div class="badge-warn">${data.note || ""}</div>
    `;
    this._drawSensitivityCurve(this.elCanvasSens, data.sensitivity_curve || []);
  }

  _drawSensitivityCurve(canvas, curve) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#10131a";
    ctx.fillRect(0, 0, w, h);
    if (!curve.length) return;

    const all = curve.flatMap((c) => [c.lower, c.upper, c.point]);
    const minV = Math.min(...all) - 1;
    const maxV = Math.max(...all) + 1;
    const xOf = (load) => 10 + (load / 100) * (w - 20);
    const yOf = (v) => h - 14 - ((v - minV) / Math.max(0.1, maxV - minV)) * (h - 24);

    ctx.fillStyle = "rgba(253, 174, 97, 0.22)";
    ctx.beginPath();
    curve.forEach((c, i) => (i === 0 ? ctx.moveTo(xOf(c.cpu_load), yOf(c.upper)) : ctx.lineTo(xOf(c.cpu_load), yOf(c.upper))));
    for (let i = curve.length - 1; i >= 0; i--) ctx.lineTo(xOf(curve[i].cpu_load), yOf(curve[i].lower));
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = "#fdae61";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    curve.forEach((c, i) => (i === 0 ? ctx.moveTo(xOf(c.cpu_load), yOf(c.point)) : ctx.lineTo(xOf(c.cpu_load), yOf(c.point))));
    ctx.stroke();

    ctx.fillStyle = "#aaa";
    ctx.font = "10px sans-serif";
    ctx.fillText("CPU load sensitivity (0–100%)", 6, 11);
  }
}
