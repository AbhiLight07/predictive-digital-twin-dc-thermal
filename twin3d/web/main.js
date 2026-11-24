/**
 * twin3d/web/main.js — 3D predictive digital twin visualisation.
 *
 * Uses Three.js InstancedMesh for performance (50+ fps on integrated GPU).
 * All data comes from /api/* endpoints; no hard-coded temperatures.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";

// ── State ───────────────────────────────────────────────────────────

let meta = null;
let currentCluster = "d1";
let currentHorizon = 1;
let currentTimeIdx = 0;
let timeSteps = [];
let snapshotData = null;
let selectedHostIdx = -1;
let playing = false;
let playInterval = null;
let compareMode = false;

// Performance
let frameCount = 0;
let lastFpsTime = performance.now();
let currentFps = 60;
let bloomEnabled = true;
let halosEnabled = true;

// Colour easing
let prevColours = null;
let nextColours = null;
let easeStart = 0;
const EASE_MS = 300;

// ── Temperature ramp (viridis-like, colour-blind safe) ──────────────

const RAMP_STOPS = [
  { t: 0.0, r: 0x44 / 255, g: 0x01 / 255, b: 0x54 / 255 },
  { t: 0.25, r: 0x3b / 255, g: 0x52 / 255, b: 0x8b / 255 },
  { t: 0.5, r: 0x21 / 255, g: 0x91 / 255, b: 0x8c / 255 },
  { t: 0.75, r: 0x5e / 255, g: 0xc9 / 255, b: 0x62 / 255 },
  { t: 1.0, r: 0xfd / 255, g: 0xe7 / 255, b: 0x25 / 255 },
];

const RISK_COLOURS = {
  normal: new THREE.Color(0x2ca25f),
  warning: new THREE.Color(0xfdae61),
  critical: new THREE.Color(0xd7191c),
  unknown: new THREE.Color(0x9e9e9e),
};

const GREY = new THREE.Color(0x444444);

function tempToColour(T, tLo, tHi) {
  if (T == null || isNaN(T)) return GREY.clone();
  const t = Math.max(0, Math.min(1, (T - tLo) / (tHi - tLo)));
  // Interpolate through ramp stops
  let i = 0;
  for (; i < RAMP_STOPS.length - 1; i++) {
    if (t <= RAMP_STOPS[i + 1].t) break;
  }
  const s0 = RAMP_STOPS[i], s1 = RAMP_STOPS[Math.min(i + 1, RAMP_STOPS.length - 1)];
  const f = s0.t === s1.t ? 0 : (t - s0.t) / (s1.t - s0.t);
  return new THREE.Color(
    s0.r + f * (s1.r - s0.r),
    s0.g + f * (s1.g - s0.g),
    s0.b + f * (s1.b - s0.b)
  );
}

// ── Three.js setup ──────────────────────────────────────────────────

const container = document.getElementById("canvas-container");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setClearColor(0x0a0a0a);
container.appendChild(renderer.domElement);

const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(container.clientWidth, container.clientHeight);
labelRenderer.domElement.style.position = "absolute";
labelRenderer.domElement.style.top = "0";
labelRenderer.domElement.style.pointerEvents = "none";
container.appendChild(labelRenderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
camera.position.set(4, 2, 6);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

// Lights
scene.add(new THREE.AmbientLight(0xffffff, 0.4));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
dirLight.position.set(5, 8, 5);
scene.add(dirLight);

// Post-processing (bloom on glow strips only via layers)
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(container.clientWidth, container.clientHeight),
  0.4, 0.3, 0.85
);
composer.addPass(bloomPass);

// ── Instanced geometry pools ────────────────────────────────────────

const MAX_HOSTS = 150;
const SLAB_GEO = new THREE.BoxGeometry(0.48, 0.0845, 0.72);
const GLOW_GEO = new THREE.BoxGeometry(0.50, 0.005, 0.74);
const HALO_GEO = new THREE.BoxGeometry(0.52, 0.09, 0.76);

const slabMat = new THREE.MeshStandardMaterial({ color: 0x333333, roughness: 0.7, metalness: 0.3 });
const glowMat = new THREE.MeshStandardMaterial({ emissive: 0xffffff, emissiveIntensity: 1.0, transparent: true, opacity: 0.9 });
const haloMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.15, depthWrite: false });

const slabs = new THREE.InstancedMesh(SLAB_GEO, slabMat, MAX_HOSTS);
const glows = new THREE.InstancedMesh(GLOW_GEO, glowMat, MAX_HOSTS);
const halos = new THREE.InstancedMesh(HALO_GEO, haloMat, MAX_HOSTS);

slabs.count = 0;
glows.count = 0;
halos.count = 0;

// Glow strip uses per-instance colour
glows.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(MAX_HOSTS * 3), 3);

scene.add(slabs);
scene.add(glows);
scene.add(halos);

// Rack frames group
const rackGroup = new THREE.Group();
scene.add(rackGroup);

// CSS2D labels
const labelGroup = new THREE.Group();
scene.add(labelGroup);
const rackLabels = [];
let hoverLabel = null;

// ── Raycasting for host selection ───────────────────────────────────

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

renderer.domElement.addEventListener("click", (e) => {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObject(slabs);
  if (hits.length > 0) {
    selectedHostIdx = hits[0].instanceId;
    updatePanel();
  }
});

// ── Build rack frames ───────────────────────────────────────────────

function buildRacks(clusterMeta) {
  // Clear old
  while (rackGroup.children.length) rackGroup.remove(rackGroup.children[0]);
  rackLabels.forEach(l => { if (l.parent) l.parent.remove(l); });
  rackLabels.length = 0;

  for (const [cid, cmeta] of Object.entries(clusterMeta)) {
    const nHosts = cmeta.host_count;
    const nRacks = Math.ceil(nHosts / 20);
    const ly = cmeta.layout;
    const cz = ly.cluster_z;
    const cLabel = cid === "d1" ? "C1" : "C2";

    for (let r = 0; r < nRacks; r++) {
      const x = r * ly.rack_pitch_x;
      // Wireframe rack
      const edges = new THREE.EdgesGeometry(
        new THREE.BoxGeometry(ly.rack.w, ly.rack.h, ly.rack.d)
      );
      const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x555555 }));
      line.position.set(x, ly.rack.h / 2, cz);
      rackGroup.add(line);

      // Label
      const div = document.createElement("div");
      div.textContent = `${cLabel}-R${r + 1}`;
      div.style.cssText = "color:#888;font-size:11px;font-family:monospace";
      const lbl = new CSS2DObject(div);
      lbl.position.set(x, ly.rack.h + 0.08, cz);
      labelGroup.add(lbl);
      rackLabels.push(lbl);
    }
  }
}

// ── Position instances from snapshot data ────────────────────────────

const _mat4 = new THREE.Matrix4();
const _pos = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _scale = new THREE.Vector3(1, 1, 1);
const _haloScale = new THREE.Vector3(1, 1, 1);
const _col = new THREE.Color();

function positionInstances(hosts, legend) {
  const n = hosts.length;
  slabs.count = n;
  glows.count = n;
  halos.count = halosEnabled ? n : 0;

  const tLo = legend.t_lo, tHi = legend.t_hi;
  const wRef = legend.w_ref || 5;

  // Prepare target colours for easing
  const targetCols = new Float32Array(n * 3);

  for (let i = 0; i < n; i++) {
    const h = hosts[i];
    _pos.set(h.x, h.y, h.z);
    _mat4.compose(_pos, _quat, _scale);
    slabs.setMatrixAt(i, _mat4);

    // Glow strip slightly above slab
    _pos.y += 0.045;
    _mat4.compose(_pos, _quat, _scale);
    glows.setMatrixAt(i, _mat4);

    // Colour: forecast point (if available + horizon > 0), else temp_eff
    let temp = h.temp_eff;
    if (h.forecast && h.forecast.point != null) {
      temp = h.forecast.point;
    }
    const col = tempToColour(temp, tLo, tHi);
    targetCols[i * 3] = col.r;
    targetCols[i * 3 + 1] = col.g;
    targetCols[i * 3 + 2] = col.b;

    // Halo: scale by interval width
    _pos.y = h.y;
    let haloS = 1;
    if (h.forecast && h.forecast.upper != null && h.forecast.lower != null) {
      const width = h.forecast.upper - h.forecast.lower;
      haloS = 1 + 0.35 * Math.max(0, Math.min(1, width / wRef));
    }
    _haloScale.set(haloS, haloS, haloS);
    _mat4.compose(_pos, _quat, _haloScale);
    halos.setMatrixAt(i, _mat4);
  }

  // Set up colour easing
  if (prevColours === null) {
    prevColours = new Float32Array(targetCols);
  } else {
    prevColours = new Float32Array(nextColours || targetCols);
  }
  nextColours = targetCols;
  easeStart = performance.now();

  slabs.instanceMatrix.needsUpdate = true;
  glows.instanceMatrix.needsUpdate = true;
  halos.instanceMatrix.needsUpdate = true;
}

function updateGlowColours(now) {
  if (!prevColours || !nextColours) return;
  const t = Math.min(1, (now - easeStart) / EASE_MS);
  const n = glows.count;
  const attr = glows.instanceColor;
  for (let i = 0; i < n; i++) {
    const r = prevColours[i * 3] + t * (nextColours[i * 3] - prevColours[i * 3]);
    const g = prevColours[i * 3 + 1] + t * (nextColours[i * 3 + 1] - prevColours[i * 3 + 1]);
    const b = prevColours[i * 3 + 2] + t * (nextColours[i * 3 + 2] - prevColours[i * 3 + 2]);
    attr.setXYZ(i, r, g, b);
  }
  attr.needsUpdate = true;
}

// ── Camera presets ──────────────────────────────────────────────────

const CAMERA_PRESETS = {
  overview: { pos: [4, 3, 8], target: [2, 0.8, -2] },
  c1: { pos: [2, 1.5, 4], target: [2, 0.8, 0] },
  c2: { pos: [2, 1.5, -1], target: [2, 0.8, -4] },
};

function setCamera(preset) {
  const p = CAMERA_PRESETS[preset];
  if (!p) return;
  camera.position.set(...p.pos);
  controls.target.set(...p.target);
  controls.update();
}

// ── Data fetching ───────────────────────────────────────────────────

async function fetchMeta() {
  const r = await fetch("/api/meta");
  meta = await r.json();

  // Populate legend
  const leg = meta.clusters[currentCluster].legend;
  document.getElementById("legend-lo").textContent = `${leg.t_lo}°C`;
  document.getElementById("legend-hi").textContent = `${leg.t_hi}°C`;

  // Build time steps from time_range
  const tr = meta.clusters[currentCluster].time_range;
  const tMin = new Date(tr.min);
  const tMax = new Date(tr.max);
  timeSteps = [];
  // Use stride from API (30 min = 3 steps of 10 min)
  const strideMs = 30 * 60 * 1000;
  for (let t = tMin.getTime(); t <= tMax.getTime(); t += strideMs) {
    timeSteps.push(new Date(t).toISOString().replace("Z", ""));
  }
  // Start from roughly 80% through (test partition)
  currentTimeIdx = Math.floor(timeSteps.length * 0.8);

  buildRacks(meta.clusters);
  return meta;
}

async function fetchSnapshot() {
  const t = timeSteps[currentTimeIdx];
  document.getElementById("time-display").textContent = t.replace("T", " ").slice(0, 19);

  const url = `/api/snapshot?cluster=${currentCluster}&t=${encodeURIComponent(t)}&horizon=${currentHorizon}`;
  const r = await fetch(url);
  if (!r.ok) return;
  snapshotData = await r.json();

  const legend = meta.clusters[currentCluster].legend;
  positionInstances(snapshotData.hosts, legend);
  updatePanel();
  updateAlerts();
}

async function fetchCompare() {
  const t = timeSteps[currentTimeIdx];
  const url = `/api/compare?cluster=${currentCluster}&t=${encodeURIComponent(t)}&horizon=${currentHorizon}`;
  const r = await fetch(url);
  if (!r.ok) return;
  const data = await r.json();
  updateCompareStats(data);
}

// ── UI updates ──────────────────────────────────────────────────────

function updatePanel() {
  if (!snapshotData) return;
  const hosts = snapshotData.hosts;

  // Fleet summary
  const nOk = hosts.filter(h => h.status === "ok").length;
  const nWarning = hosts.filter(h => h.forecast && h.forecast.risk === "warning").length;
  const nCritical = hosts.filter(h => h.forecast && h.forecast.risk === "critical").length;
  document.getElementById("fleet-summary").innerHTML = `
    <div class="row"><span class="label">Total hosts</span><span class="val">${hosts.length}</span></div>
    <div class="row"><span class="label">Status OK</span><span class="val">${nOk}</span></div>
    <div class="row"><span class="label"><span class="risk-dot" style="background:#fdae61"></span>Warning</span><span class="val">${nWarning}</span></div>
    <div class="row"><span class="label"><span class="risk-dot" style="background:#d7191c"></span>Critical</span><span class="val">${nCritical}</span></div>
    <div class="row"><span class="label">Horizon</span><span class="val">${currentHorizon * 10} min</span></div>
  `;

  // Host detail
  if (selectedHostIdx >= 0 && selectedHostIdx < hosts.length) {
    const h = hosts[selectedHostIdx];
    document.getElementById("host-detail-title").textContent = `🖥️ ${h.id}`;
    let html = `
      <div class="row"><span class="label">Rack</span><span class="val">${h.rack_label} slot ${h.slot}</span></div>
      <div class="row"><span class="label">Status</span><span class="val">${h.status}</span></div>
      <div class="row"><span class="label">Partition</span><span class="val">${h.partition}</span></div>
      <div class="row"><span class="label">Temp (eff)</span><span class="val">${h.temp_eff != null ? h.temp_eff + '°C' : '—'}</span></div>
      <div class="row"><span class="label">CPU1 / CPU2</span><span class="val">${h.cpu1 ?? '—'} / ${h.cpu2 ?? '—'}</span></div>
      <div class="row"><span class="label">Power</span><span class="val">${h.power != null ? h.power + ' W' : '—'}</span></div>
      <div class="row"><span class="label">Load</span><span class="val">${h.load != null ? h.load + '%' : '—'}</span></div>
      <div class="row"><span class="label">RAM used</span><span class="val">${h.ram_used_pct != null ? h.ram_used_pct + '%' : '—'}</span></div>
      <div class="row"><span class="label">VMs</span><span class="val">${h.vms ?? '—'}</span></div>
    `;
    if (h.fans_available && h.fans) {
      html += `<div class="row"><span class="label">Fans</span><span class="val">${h.fans.map(f => f ?? '—').join(', ')}</span></div>`;
    } else if (!h.fans_available) {
      html += `<div class="row"><span class="label">Fans</span><span class="val" style="color:#666">no fan sensor</span></div>`;
    }
    if (h.inlet_available) {
      html += `<div class="row"><span class="label">Inlet</span><span class="val">${h.inlet != null ? h.inlet + '°C' : '—'}</span></div>`;
    } else {
      html += `<div class="row"><span class="label">Inlet</span><span class="val" style="color:#666">no inlet sensor</span></div>`;
    }
    if (h.forecast) {
      const fc = h.forecast;
      html += `
        <h3>📊 Forecast (h=${fc.horizon})</h3>
        <div class="row"><span class="label">Point</span><span class="val">${fc.point}°C</span></div>
        <div class="row"><span class="label">Interval</span><span class="val">[${fc.lower}, ${fc.upper}]°C</span></div>
        <div class="row"><span class="label">Persistence</span><span class="val">${fc.persistence}°C</span></div>
        <div class="row"><span class="label">Risk</span><span class="val"><span class="risk-dot" style="background:${RISK_COLOURS[fc.risk] ? '#' + RISK_COLOURS[fc.risk].getHexString() : '#999'}"></span>${fc.risk}</span></div>
      `;
    }
    document.getElementById("host-detail").innerHTML = html;
  }
}

function updateAlerts() {
  if (!snapshotData) return;
  const alertList = document.getElementById("alert-list");
  const alerts = snapshotData.hosts.filter(
    h => h.forecast && (h.forecast.risk === "warning" || h.forecast.risk === "critical")
  );
  if (alerts.length === 0) {
    alertList.innerHTML = "<em>no alerts</em>";
    return;
  }
  alertList.innerHTML = alerts.map(h => {
    const risk = h.forecast.risk;
    return `<div class="alert-item ${risk}" data-idx="${h.idx}">
      <strong>${h.id}</strong> — ${risk} (${h.forecast.point}°C, upper ${h.forecast.upper}°C)
    </div>`;
  }).join("");
  // Click to select
  alertList.querySelectorAll(".alert-item").forEach(el => {
    el.style.cursor = "pointer";
    el.addEventListener("click", () => {
      selectedHostIdx = parseInt(el.dataset.idx);
      updatePanel();
    });
  });
}

function updateCompareStats(data) {
  const s = data.stats;
  document.getElementById("metric-mae").textContent = `MAE: ${s.mae != null ? s.mae + '°C' : '—'}`;
  document.getElementById("metric-coverage").textContent = `Coverage: ${s.coverage_pct != null ? s.coverage_pct + '%' : '—'} (${s.n_inside}/${s.n_valid})`;
  if (data.in_sample_warning) {
    document.getElementById("metric-persist").textContent = data.in_sample_warning;
    document.getElementById("metric-persist").style.color = "#fdae61";
  } else {
    document.getElementById("metric-persist").textContent = `Partition: ${data.partition}`;
    document.getElementById("metric-persist").style.color = "#999";
  }
}

// ── What-If ─────────────────────────────────────────────────────────

document.getElementById("whatif-load").addEventListener("input", async (e) => {
  const load = parseFloat(e.target.value);
  const url = `/api/whatif?cluster=${currentCluster}&CPU_Load=${load}&No_Of_Running_vms=5&Ram_Used=50`;
  const r = await fetch(url);
  if (!r.ok) return;
  const data = await r.json();
  if (data.available) {
    document.getElementById("whatif-result").innerHTML = `
      <div class="row"><span class="label">Predicted</span><span class="val">${data.point}°C</span></div>
      <div class="row"><span class="label">Interval</span><span class="val">[${data.lower}, ${data.upper}]°C</span></div>
      <div class="row"><span class="label">Risk</span><span class="val">${data.risk}</span></div>
    `;
  } else {
    document.getElementById("whatif-result").innerHTML = `<em>${data.error}</em>`;
  }
});

// ── Controls ────────────────────────────────────────────────────────

document.getElementById("sel-cluster").addEventListener("change", (e) => {
  currentCluster = e.target.value;
  prevColours = null;
  nextColours = null;
  fetchSnapshot();
  if (compareMode) fetchCompare();
  // Update legend
  if (meta) {
    const leg = meta.clusters[currentCluster].legend;
    document.getElementById("legend-lo").textContent = `${leg.t_lo}°C`;
    document.getElementById("legend-hi").textContent = `${leg.t_hi}°C`;
  }
});

document.getElementById("sel-horizon").addEventListener("change", (e) => {
  currentHorizon = parseInt(e.target.value);
  fetchSnapshot();
  if (compareMode) fetchCompare();
});

document.getElementById("sel-camera").addEventListener("change", (e) => {
  if (e.target.value === "selected" && selectedHostIdx >= 0 && snapshotData) {
    const h = snapshotData.hosts[selectedHostIdx];
    camera.position.set(h.x + 1, h.y + 0.5, h.z + 2);
    controls.target.set(h.x, h.y, h.z);
    controls.update();
  } else {
    setCamera(e.target.value);
  }
});

document.getElementById("btn-prev").addEventListener("click", () => {
  if (currentTimeIdx > 0) { currentTimeIdx--; fetchSnapshot(); if (compareMode) fetchCompare(); }
});
document.getElementById("btn-next").addEventListener("click", () => {
  if (currentTimeIdx < timeSteps.length - 1) { currentTimeIdx++; fetchSnapshot(); if (compareMode) fetchCompare(); }
});
document.getElementById("btn-play").addEventListener("click", () => {
  playing = !playing;
  document.getElementById("btn-play").textContent = playing ? "⏸ Pause" : "▶ Play";
  if (playing) {
    playInterval = setInterval(() => {
      if (currentTimeIdx < timeSteps.length - 1) {
        currentTimeIdx++;
        fetchSnapshot();
        if (compareMode) fetchCompare();
      } else {
        playing = false;
        document.getElementById("btn-play").textContent = "▶ Play";
        clearInterval(playInterval);
      }
    }, 2000);
  } else {
    clearInterval(playInterval);
  }
});

document.getElementById("chk-compare").addEventListener("change", (e) => {
  compareMode = e.target.checked;
  if (compareMode) fetchCompare();
});

// ── Render loop ─────────────────────────────────────────────────────

function animate(now) {
  requestAnimationFrame(animate);
  controls.update();

  // Colour easing
  updateGlowColours(now);

  // FPS counting
  frameCount++;
  if (now - lastFpsTime >= 1000) {
    currentFps = frameCount;
    frameCount = 0;
    lastFpsTime = now;
    document.getElementById("fps-counter").textContent = `${currentFps} fps`;

    // Auto-disable bloom/halos if fps < 30
    if (currentFps < 30 && bloomEnabled) {
      bloomEnabled = false;
      bloomPass.enabled = false;
    }
    if (currentFps < 30 && halosEnabled) {
      halosEnabled = false;
      halos.count = 0;
    }
  }

  if (bloomEnabled) {
    composer.render();
  } else {
    renderer.render(scene, camera);
  }
  labelRenderer.render(scene, camera);
}

// ── Resize ──────────────────────────────────────────────────────────

window.addEventListener("resize", () => {
  const w = container.clientWidth, h = container.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  labelRenderer.setSize(w, h);
  composer.setSize(w, h);
});

// ── Boot ────────────────────────────────────────────────────────────

async function init() {
  await fetchMeta();
  await fetchSnapshot();
  if (compareMode) await fetchCompare();
  setCamera("overview");
  animate(performance.now());
}

init().catch(console.error);
