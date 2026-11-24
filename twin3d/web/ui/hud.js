/**
 * ui/hud.js — Timeline construction (30-min grid anchored at test_start),
 * controls binding, legend rendering, FPS/draw-call budget monitor, and compare HUD.
 */

export class HudController {
  constructor(callbacks) {
    this.cb = callbacks;
    this.timeSteps = [];
    this.currentTimeIdx = 0;
    this.playing = false;
    this.playTimer = null;
    this.playMs = 1400;

    // Elements
    this.elCluster = document.getElementById("sel-cluster");
    this.elHorizon = document.getElementById("sel-horizon");
    this.elColorMode = document.getElementById("sel-colormode");
    this.elPalette = document.getElementById("sel-palette");
    this.elCamera = document.getElementById("sel-camera");
    this.elPrev = document.getElementById("btn-prev");
    this.elNext = document.getElementById("btn-next");
    this.elPlay = document.getElementById("btn-play");
    this.elSpeed = document.getElementById("sel-speed");
    this.elSlider = document.getElementById("time-slider");
    this.elTimeDisplay = document.getElementById("time-display");
    this.elCompare = document.getElementById("chk-compare");
    this.elErrorOverlay = document.getElementById("chk-error-overlay");
    this.elSingleView = document.getElementById("chk-single-viewport");
    this.elSingleSide = document.getElementById("sel-single-side");
    this.elFps = document.getElementById("fps-counter");
    this.elCompareOverlay = document.getElementById("compare-overlay");
    this.elCompareStats = document.getElementById("compare-stats");
    this.elLegendBar = document.getElementById("legend-bar");
    this.elLegendLo = document.getElementById("legend-lo");
    this.elLegendHi = document.getElementById("legend-hi");
    this.elLegendExtra = document.getElementById("legend-extra");

    this._bindEvents();
  }

  _bindEvents() {
    this.elCluster.addEventListener("change", () => this.cb.onClusterChange(this.elCluster.value));
    this.elHorizon.addEventListener("change", () => this.cb.onHorizonChange(parseInt(this.elHorizon.value, 10)));
    this.elColorMode.addEventListener("change", () => this.cb.onColorModeChange(this.elColorMode.value));
    this.elPalette.addEventListener("change", () => this.cb.onPaletteChange(this.elPalette.value));
    this.elCamera.addEventListener("change", () => this.cb.onCameraPreset(this.elCamera.value));

    this.elPrev.addEventListener("click", () => this.stepBy(-1));
    this.elNext.addEventListener("click", () => this.stepBy(1));
    this.elPlay.addEventListener("click", () => this.togglePlay());
    this.elSpeed.addEventListener("change", () => {
      this.playMs = parseInt(this.elSpeed.value, 10);
      if (this.playing) {
        this._stopPlayTimer();
        this._startPlayTimer();
      }
    });

    this.elSlider.addEventListener("input", () => {
      this.currentTimeIdx = parseInt(this.elSlider.value, 10);
      this.syncTimeDisplay();
      this.cb.onTimeChange(this.getCurrentTimeIso());
    });

    this.elCompare.addEventListener("change", () => {
      this.elErrorOverlay.disabled = !this.elCompare.checked;
      this.elSingleView.disabled = !this.elCompare.checked;
      this.elSingleSide.disabled = !this.elCompare.checked || !this.elSingleView.checked;
      this.cb.onCompareToggle(this.elCompare.checked);
    });
    this.elErrorOverlay.addEventListener("change", () =>
      this.cb.onErrorOverlayToggle(this.elErrorOverlay.checked)
    );
    this.elSingleView.addEventListener("change", () => {
      this.elSingleSide.disabled = !this.elSingleView.checked;
      this.cb.onSingleViewportToggle(this.elSingleView.checked, this.elSingleSide.value);
    });
    this.elSingleSide.addEventListener("change", () =>
      this.cb.onSingleViewportToggle(this.elSingleView.checked, this.elSingleSide.value)
    );
  }

  buildTimeline(meta, clusterFilter) {
    const c1 = meta.clusters.d1;
    const c2 = meta.clusters.d2;
    const active = clusterFilter === "d2" && c2 ? c2 : c1 || c2;
    const strideMin = active.replay_range.stride_minutes || 30;
    const strideMs = strideMin * 60 * 1000;

    const minMs = Date.parse(active.replay_range.min);
    const maxMs = Date.parse(active.replay_range.max);
    // Align grid cleanly to test_start (so :20/:50 timestamps hit cached forecasts directly)
    let anchorIso = active.replay_range.test_start;
    if (clusterFilter === "all" && c1 && c2) {
      const t1 = Date.parse(c1.replay_range.test_start);
      const t2 = Date.parse(c2.replay_range.test_start);
      anchorIso = t2 >= t1 ? c2.replay_range.test_start : c1.replay_range.test_start;
    }
    const anchorMs = Date.parse(anchorIso);
    const offset = ((minMs - anchorMs) % strideMs + strideMs) % strideMs;
    const startMs = minMs + (offset === 0 ? 0 : strideMs - offset);

    this.timeSteps = [];
    let defaultIdx = 0;
    for (let t = startMs; t <= maxMs; t += strideMs) {
      const iso = new Date(t).toISOString().replace(".000Z", "Z");
      if (t >= anchorMs && defaultIdx === 0) {
        defaultIdx = this.timeSteps.length;
      }
      this.timeSteps.push(iso);
    }

    this.elSlider.min = "0";
    this.elSlider.max = String(Math.max(0, this.timeSteps.length - 1));
    this.setTimeIndex(defaultIdx);
  }

  setTimeIndex(idx) {
    this.currentTimeIdx = Math.max(0, Math.min(this.timeSteps.length - 1, idx));
    this.elSlider.value = String(this.currentTimeIdx);
    this.syncTimeDisplay();
  }

  setTimeToIso(iso) {
    const idx = this.timeSteps.indexOf(iso);
    if (idx >= 0) {
      this.setTimeIndex(idx);
      this.cb.onTimeChange(iso);
    }
  }

  getCurrentTimeIso() {
    return this.timeSteps[this.currentTimeIdx] || null;
  }

  getPrefetchTimes(count = 2) {
    const out = [];
    for (let k = 1; k <= count; k++) {
      if (this.currentTimeIdx + k < this.timeSteps.length) {
        out.push(this.timeSteps[this.currentTimeIdx + k]);
      }
    }
    return out;
  }

  syncTimeDisplay() {
    const iso = this.getCurrentTimeIso() || "--";
    this.elTimeDisplay.textContent = iso.replace("T", " ").replace("Z", " UTC");
  }

  stepBy(delta) {
    const nxt = Math.max(0, Math.min(this.timeSteps.length - 1, this.currentTimeIdx + delta));
    if (nxt !== this.currentTimeIdx) {
      this.setTimeIndex(nxt);
      this.cb.onTimeChange(this.getCurrentTimeIso());
    }
  }

  togglePlay(force = null) {
    this.playing = force !== null ? force : !this.playing;
    this.elPlay.textContent = this.playing ? "Pause" : "Play";
    this._stopPlayTimer();
    if (this.playing) this._startPlayTimer();
  }

  _startPlayTimer() {
    this.playTimer = setInterval(() => {
      if (this.currentTimeIdx >= this.timeSteps.length - 1) {
        this.togglePlay(false);
        return;
      }
      this.stepBy(1);
    }, this.playMs);
  }

  _stopPlayTimer() {
    if (this.playTimer) {
      clearInterval(this.playTimer);
      this.playTimer = null;
    }
  }

  updateLegend(legend, colorMapper, strideMinutes = 30) {
    this.elLegendBar.style.background = colorMapper.getCssGradient();
    this.elLegendLo.textContent = legend?.t_lo != null ? `${legend.t_lo}°C (p5)` : "--";
    this.elLegendHi.textContent = legend?.t_hi != null ? `${legend.t_hi}°C (p99.5)` : "--";
    const wStr = legend?.w_ref != null ? `w_ref=${legend.w_ref}°C` : "w_ref=--";
    const rpmStr = legend?.rpm_ref != null ? `fan visual cap 12 rad/s @ ${legend.rpm_ref} RPM (not true RPM)` : "D2: no fan sensors";
    this.elLegendExtra.textContent = `Halo ${wStr} · ${rpmStr} · frames every ${strideMinutes} min; colours eased`;
  }

  updatePerf(fps, drawCalls, bloomEnabled, halosEnabled) {
    const extras = [];
    if (!bloomEnabled) extras.push("bloom off");
    if (!halosEnabled) extras.push("halos off");
    const suffix = extras.length ? ` (${extras.join(", ")})` : "";
    this.elFps.textContent = `${fps} fps · ${drawCalls} draws${suffix}`;
  }

  updateCompareHud(active, compareData, singleViewport, singleSide) {
    this.elCompareOverlay.style.display = active ? "flex" : "none";
    this.elCompareStats.style.display = active ? "block" : "none";
    if (!active || !compareData) return;

    const s = compareData.stats || {};
    const s24 = compareData.stats_24h || {};
    const warn = compareData.in_sample_warning
      ? `<span class="badge-warn">${compareData.in_sample_warning}</span>`
      : `<span class="badge-test">Partition: ${compareData.partition}</span>`;

    const modeTag = singleViewport ? `Single-view: ${singleSide.toUpperCase()}` : "Dual scissor";
    this.elCompareStats.innerHTML = `
      <div><strong>Compare (${compareData.cluster.toUpperCase()}, h=${compareData.horizon})</strong> · ${modeTag} ${warn}</div>
      <div>Target <code>${compareData.target_time}</code> · Issued <code>${compareData.issue_time}</code></div>
      <div>Frame MAE: <strong id="cmp-mae">${s.mae ?? "--"}°C</strong> · Inside CQR: <strong id="cmp-inside">${s.coverage_pct ?? "--"}%</strong> (${s.n_inside ?? 0}/${s.n_valid ?? 0})</div>
      <div>Last 24h MAE: <strong>${s24.mae ?? "--"}°C</strong> · 24h Inside: <strong>${s24.coverage_pct ?? "--"}%</strong> (${s24.n_inside ?? 0}/${s24.n_valid ?? 0})</div>
    `;
  }
}
