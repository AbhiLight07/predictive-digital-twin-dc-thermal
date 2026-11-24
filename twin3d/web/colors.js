/**
 * colors.js — Viridis, Cividis, Risk, and Diverging Error ramps from /api/meta.
 * Zero hard-coded temperatures; all domains come from server legend metadata.
 */
import * as THREE from "three";

export const GREY_COLOR = new THREE.Color(0x555860);
export const LED_OFF_COLOR = new THREE.Color(0x2a2d34);

const DEFAULT_VIRIDIS = [
  { t: 0.0, hex: "#440154" },
  { t: 0.25, hex: "#3b528b" },
  { t: 0.5, hex: "#21918c" },
  { t: 0.75, hex: "#5ec962" },
  { t: 1.0, hex: "#fde725" },
];

const DEFAULT_CIVIDIS = [
  { t: 0.0, hex: "#00204c" },
  { t: 0.25, hex: "#414d6b" },
  { t: 0.5, hex: "#7c7b78" },
  { t: 0.75, hex: "#b9ac70" },
  { t: 1.0, hex: "#ffea46" },
];

const RISK_HEX = {
  normal: "#2ca25f",
  warning: "#fdae61",
  critical: "#d7191c",
  unknown: "#9e9e9e",
};

const ERROR_STOPS = [
  new THREE.Color("#2166ac"),
  new THREE.Color("#f7f7f7"),
  new THREE.Color("#b2182b"),
];

export class ColorMapper {
  constructor(coloursMeta = null) {
    this.palette = "viridis";
    this.stopsViridis = (coloursMeta?.temperature_ramp || DEFAULT_VIRIDIS).map((s) => ({
      t: s.t,
      color: new THREE.Color(s.hex),
    }));
    this.stopsCividis = (coloursMeta?.cividis_ramp || DEFAULT_CIVIDIS).map((s) => ({
      t: s.t,
      color: new THREE.Color(s.hex),
    }));
    const rk = coloursMeta?.risk || RISK_HEX;
    this.riskColors = {
      normal: new THREE.Color(rk.normal),
      warning: new THREE.Color(rk.warning),
      critical: new THREE.Color(rk.critical),
      unknown: new THREE.Color(rk.unknown),
    };
    const ed = coloursMeta?.error_diverging;
    if (ed) {
      ERROR_STOPS[0].set(ed.below);
      ERROR_STOPS[1].set(ed.zero);
      ERROR_STOPS[2].set(ed.above);
      this.errorSatC = ed.saturate_c || 3.0;
    } else {
      this.errorSatC = 3.0;
    }
  }

  setPalette(name) {
    this.palette = name === "cividis" ? "cividis" : "viridis";
  }

  getActiveStops() {
    return this.palette === "cividis" ? this.stopsCividis : this.stopsViridis;
  }

  getCssGradient() {
    const stops = this.getActiveStops();
    return `linear-gradient(to right, ${stops.map((s) => `#${s.color.getHexString()}`).join(", ")})`;
  }

  tempToColor(temp, legend, out = new THREE.Color()) {
    if (temp == null || Number.isNaN(temp) || !legend || legend.t_lo == null || legend.t_hi == null) {
      return out.copy(GREY_COLOR);
    }
    const span = Math.max(0.01, legend.t_hi - legend.t_lo);
    const t = Math.max(0, Math.min(1, (temp - legend.t_lo) / span));
    return this.sampleRamp(t, out);
  }

  sampleRamp(tNorm, out = new THREE.Color()) {
    const stops = this.getActiveStops();
    const t = Math.max(0, Math.min(1, tNorm));
    let idx = 0;
    for (; idx < stops.length - 2; idx++) {
      if (t <= stops[idx + 1].t) break;
    }
    const s0 = stops[idx];
    const s1 = stops[idx + 1];
    const localT = s1.t === s0.t ? 0 : (t - s0.t) / (s1.t - s0.t);
    return out.copy(s0.color).lerp(s1.color, localT);
  }

  riskToColor(risk, out = new THREE.Color()) {
    const key = typeof risk === "string" ? risk.toLowerCase() : "unknown";
    const c = this.riskColors[key] || this.riskColors.unknown;
    return out.copy(c);
  }

  errorToColor(err, out = new THREE.Color()) {
    if (err == null || Number.isNaN(err)) return out.copy(GREY_COLOR);
    const norm = Math.max(-1, Math.min(1, err / this.errorSatC));
    if (norm < 0) {
      return out.copy(ERROR_STOPS[1]).lerp(ERROR_STOPS[0], -norm);
    }
    return out.copy(ERROR_STOPS[1]).lerp(ERROR_STOPS[2], norm);
  }

  powerToLedColor(power, legend, out = new THREE.Color()) {
    if (power == null || Number.isNaN(power) || !legend || legend.p_lo == null || legend.p_hi == null) {
      return out.copy(LED_OFF_COLOR);
    }
    const span = Math.max(0.01, legend.p_hi - legend.p_lo);
    const norm = Math.max(0, Math.min(1, (power - legend.p_lo) / span));
    const intensity = 0.15 + 0.85 * norm;
    return out.setRGB(0.15 * intensity, 0.95 * intensity, 0.72 * intensity);
  }
}
