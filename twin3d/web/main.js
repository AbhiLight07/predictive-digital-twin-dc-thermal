/**
 * main.js — Three.js application orchestrator for the 3D Predictive Digital Twin.
 * Zero CDN dependencies, zero hard-coded telemetry, zero per-frame allocations.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

import { TwinAPI } from "./api.js";
import { ColorMapper } from "./colors.js";
import { HallScene } from "./scene/hall.js";
import { ServerFleetScene } from "./scene/servers.js";
import { ServerAnatomyScene } from "./scene/anatomy.js";
import { HudController } from "./ui/hud.js";
import { PanelController } from "./ui/panel.js";

const container = document.getElementById("canvas-container");
const statusOverlay = document.getElementById("status-overlay");

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setClearColor(0x0a0c10, 1.0);
container.appendChild(renderer.domElement);

const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(container.clientWidth, container.clientHeight);
labelRenderer.domElement.style.position = "absolute";
labelRenderer.domElement.style.top = "0";
labelRenderer.domElement.style.left = "0";
labelRenderer.domElement.style.pointerEvents = "none";
container.appendChild(labelRenderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0c10);
scene.fog = new THREE.FogExp2(0x0a0c10, 0.032);

const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 80);
camera.position.set(4.2, 2.8, 4.6);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(1.4, 0.9, -1.6);
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.75);
dirLight.position.set(5, 8, 6);
scene.add(dirLight);

const composer = new EffectComposer(renderer);
const renderPass = new RenderPass(scene, camera);
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(container.clientWidth, container.clientHeight),
  0.35,
  0.35,
  0.85
);
const outputPass = new OutputPass();
composer.addPass(renderPass);
composer.addPass(bloomPass);
composer.addPass(outputPass);

// State
const api = new TwinAPI();
const colors = new ColorMapper();
const hall = new HallScene(scene);
const fleet = new ServerFleetScene(scene, colors);
const anatomy = new ServerAnatomyScene(scene, colors);

const state = {
  meta: null,
  clusterFilter: "all",
  horizon: 6,
  colorMode: "temp",
  compareMode: false,
  errorOverlay: false,
  singleViewport: false,
  singleSide: "right",
  anatomyMode: false,
  bloomEnabled: true,
  selectedHostKey: null,
  lastSnapshotHosts: [],
  lastCompareData: null,
  legendsByCluster: {},
  alertsMap: new Map(),
};

// Camera flight interpolation (zero per-frame allocations)
const camFlight = {
  active: false,
  startMs: 0,
  durationMs: 450,
  fromPos: new THREE.Vector3(),
  toPos: new THREE.Vector3(),
  fromTarget: new THREE.Vector3(),
  toTarget: new THREE.Vector3(),
};

function flyCamera(posArr, targetArr) {
  camFlight.active = true;
  camFlight.startMs = performance.now();
  camFlight.fromPos.copy(camera.position);
  camFlight.toPos.set(posArr[0], posArr[1], posArr[2]);
  camFlight.fromTarget.copy(controls.target);
  camFlight.toTarget.set(targetArr[0], targetArr[1], targetArr[2]);
}

function getCameraPreset(preset) {
  if (preset === "c1") return { pos: [1.5, 1.4, 3.8], target: [1.35, 0.9, 0.0] };
  if (preset === "c2") return { pos: [1.1, 1.4, -0.4], target: [0.9, 0.9, -4.0] };
  if (preset === "selected" && state.selectedHostKey) {
    const idx = fleet.hostByKey.get(state.selectedHostKey);
    if (idx != null) {
      const h = fleet.hosts[idx];
      return { pos: [h.x + 0.9, h.y + 0.35, h.z + 1.8], target: [h.x, h.y, h.z] };
    }
  }
  return { pos: [4.2, 2.8, 4.6], target: [1.4, 0.9, -1.6] };
}

const panel = new PanelController({
  onSelectHostKey: (key, fly = true) => selectHostByKey(key, fly),
  onWhatIfChange: (wiReq) => runWhatIf(wiReq),
  onInspectSelected: () => enterAnatomyMode(),
});

const hud = new HudController({
  onClusterChange: (c) => {
    if (state.anatomyMode) exitAnatomyMode();
    state.clusterFilter = c;
    hud.buildTimeline(state.meta, c);
    panel.renderProvenance(c);
    refreshLegend();
    refreshFrame();
  },
  onHorizonChange: (h) => {
    state.horizon = h;
    refreshFrame();
  },
  onColorModeChange: (m) => {
    state.colorMode = m;
    renderCurrentData();
  },
  onPaletteChange: (p) => {
    colors.setPalette(p);
    refreshLegend();
    renderCurrentData();
  },
  onCameraPreset: (preset) => {
    const p = getCameraPreset(preset);
    flyCamera(p.pos, p.target);
  },
  onTimeChange: () => refreshFrame(),
  onCompareToggle: (on) => {
    state.compareMode = on;
    if (on && state.horizon === 0) {
      state.horizon = 6;
      hud.elHorizon.value = "6";
    }
    if (!on) {
      const w = container.clientWidth, h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setViewport(0, 0, w, h);
    }
    refreshFrame();
  },
  onErrorOverlayToggle: (on) => {
    state.errorOverlay = on;
    hud.updateCompareHud(state.compareMode, state.lastCompareData, state.singleViewport, state.singleSide);
  },
  onSingleViewportToggle: (single, side) => {
    state.singleViewport = single;
    state.singleSide = side;
    hud.updateCompareHud(state.compareMode, state.lastCompareData, state.singleViewport, state.singleSide);
  },
});

document.getElementById("btn-back-hall").addEventListener("click", () => exitAnatomyMode());

function enterAnatomyMode() {
  if (!state.selectedHostKey) return;
  const idx = fleet.hostByKey.get(state.selectedHostKey);
  if (idx == null) return;
  const host = fleet.hosts[idx];
  if (state.compareMode) {
    state.compareMode = false;
    hud.elCompare.checked = false;
    hud.updateCompareHud(false, null, false, "right");
  }
  state.anatomyMode = true;
  hall.setVisible(false);
  fleet.setVisible(false);
  const leg = state.legendsByCluster[host.cluster || "d1"];
  anatomy.setHost(host, leg);
  anatomy.setVisible(true);
  document.getElementById("anatomy-title").textContent = `Anatomy: ${host.id} (${(host.cluster || "d1").toUpperCase()})`;
  document.getElementById("anatomy-overlay").style.display = "flex";
  document.getElementById("anatomy-chart-wrap").style.display = "block";
  document.getElementById("anatomy-chart-caption").textContent = `${host.id} — Last 24h Observed (°C) & CQR Forecast Band`;
  flyCamera([1.35, 0.92, 1.75], [0.0, 0.25, 0.0]);
}

function exitAnatomyMode() {
  if (!state.anatomyMode) return;
  state.anatomyMode = false;
  anatomy.setVisible(false);
  hall.setVisible(true);
  fleet.setVisible(true);
  document.getElementById("anatomy-overlay").style.display = "none";
  document.getElementById("anatomy-chart-wrap").style.display = "none";
  const p = getCameraPreset("overview");
  flyCamera(p.pos, p.target);
}

function refreshLegend() {
  if (!state.meta) return;
  const cid = state.clusterFilter === "d2" ? "d2" : "d1";
  const cm = state.meta.clusters[cid];
  if (cm) hud.updateLegend(cm.legend, colors, cm.replay_range.stride_minutes || 30);
}

async function selectHostByKey(key, fly = false) {
  state.selectedHostKey = key;
  const idx = fleet.hostByKey.get(key);
  if (idx != null) {
    fleet.setSelectedIndex(idx);
    const host = fleet.hosts[idx];
    panel.updateHostDetail(host);
    panel.seedWhatIfFromHost(host);
    if (fly) {
      const p = getCameraPreset("selected");
      flyCamera(p.pos, p.target);
    }
    const tIso = hud.getCurrentTimeIso();
    const startIso = new Date(Date.parse(tIso) - 24 * 3600 * 1000).toISOString().replace(".000Z", "Z");
    try {
      const series = await api.getSeries(host.cluster || "d1", host.id, startIso, tIso, state.horizon || 6);
      panel.drawSeriesCharts(series);
    } catch {
      // Ignore transient series error
    }
    panel._triggerWhatIf();
  }
}

async function runWhatIf(wiReq) {
  try {
    let cluster = wiReq.cluster;
    let hostId = null;
    if (wiReq.target === "selected" && state.selectedHostKey) {
      const [c, id] = state.selectedHostKey.split("|");
      cluster = c;
      hostId = id;
    }
    const tIso = hud.getCurrentTimeIso();
    let hour = 12;
    let weekday = 2;
    if (tIso) {
      const dt = new Date(tIso);
      if (!Number.isNaN(dt.getTime())) {
        hour = dt.getUTCHours();
        weekday = (dt.getUTCDay() + 6) % 7;
      }
    }
    const res = await api.postWhatIf({
      cluster,
      host: hostId,
      workload: wiReq.workload,
      hour,
      weekday,
      include_curve: true,
    });
    panel.updateWhatIfResult(res);
    const leg = state.legendsByCluster[cluster];
    if (hostId && fleet.hostByKey.has(`${cluster}|${hostId}`)) {
      const h = fleet.hosts[fleet.hostByKey.get(`${cluster}|${hostId}`)];
      fleet.updateGhost(true, { x: h.x + 0.56, y: h.y, z: h.z }, res, leg);
    } else {
      const bayX = cluster === "d1" ? 3.8 : 2.9;
      const bayZ = cluster === "d1" ? 0.0 : -4.0;
      fleet.updateGhost(true, { x: bayX, y: 0.45, z: bayZ }, res, leg);
    }
  } catch (err) {
    document.getElementById("whatif-result").innerHTML = `<span class="badge-warn">${err.message}</span>`;
  }
}

async function refreshFrame() {
  const tIso = hud.getCurrentTimeIso();
  if (!tIso || !state.meta) return;

  try {
    const snapReq = api.getSnapshot(state.clusterFilter, tIso, state.horizon);
    const alertReq = api.getAlerts(state.clusterFilter, tIso);
    const cmpCluster = state.clusterFilter === "d2" ? "d2" : "d1";
    const cmpReq = state.compareMode && state.horizon > 0
      ? api.getCompare(cmpCluster, tIso, state.horizon)
      : Promise.resolve(null);

    const [snap, alertsData, cmpData] = await Promise.all([snapReq, alertReq, cmpReq]);

    state.alertsMap.clear();
    for (const a of alertsData.alerts || []) {
      state.alertsMap.set(`${a.cluster}|${a.id}`, a.risk);
    }
    for (const h of snap.hosts) {
      h.alert_risk = state.alertsMap.get(`${h.cluster}|${h.id}`) || null;
    }

    state.lastSnapshotHosts = snap.hosts;
    state.lastCompareData = cmpData;

    renderCurrentData();
    panel.updateFleetSummary(snap.hosts, state.horizon);
    panel.updateAlerts(alertsData);
    hud.updateCompareHud(state.compareMode, cmpData, state.singleViewport, state.singleSide);

    if (state.selectedHostKey && fleet.hostByKey.has(state.selectedHostKey)) {
      await selectHostByKey(state.selectedHostKey, false);
    } else if (snap.hosts.length > 0) {
      await selectHostByKey(`${snap.hosts[0].cluster}|${snap.hosts[0].id}`, false);
    }

    api.prefetchSnapshots(state.clusterFilter, hud.getPrefetchTimes(2), state.horizon);
  } catch (err) {
    statusOverlay.textContent = err.status === 503 ? `Model not available: ${err.message}` : err.message;
    statusOverlay.style.display = "flex";
  }
}

function renderCurrentData() {
  const hosts = state.compareMode && state.lastCompareData ? state.lastCompareData.hosts.map((ch) => ({
    ...ch,
    cluster: state.lastCompareData.cluster,
    forecast: { lower: ch.lower, upper: ch.upper, point: ch.twin, risk: ch.risk },
  })) : state.lastSnapshotHosts;

  fleet.updateSnapshot(hosts, {
    horizon: state.horizon,
    colorMode: state.colorMode,
    legendsByCluster: state.legendsByCluster,
    isCompareMode: state.compareMode,
    errorOverlay: state.errorOverlay,
    showHalos: state.horizon > 0,
    defaultCluster: state.clusterFilter === "d2" ? "d2" : "d1",
  });

  if (state.anatomyMode && state.selectedHostKey && fleet.hostByKey.has(state.selectedHostKey)) {
    const idx = fleet.hostByKey.get(state.selectedHostKey);
    const h = fleet.hosts[idx];
    anatomy.setHost(h, state.legendsByCluster[h.cluster || "d1"]);
  }
}

// Pointer hover & click selection
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

renderer.domElement.addEventListener("pointermove", (ev) => {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObject(fleet.slabs);
  fleet.setHover(hits.length > 0 ? hits[0].instanceId : -1);
});

renderer.domElement.addEventListener("click", (ev) => {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObject(fleet.slabs);
  if (hits.length > 0) {
    const h = fleet.hosts[hits[0].instanceId];
    if (h) selectHostByKey(`${h.cluster || "d1"}|${h.id}`, false);
  }
});

window.addEventListener("resize", () => {
  const w = container.clientWidth, h = container.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  labelRenderer.setSize(w, h);
  composer.setSize(w, h);
});

window.__twin3d = {
  getStats: () => ({
    fps: currentFps,
    drawCalls: renderer.info.render.calls,
    hostCount: fleet.hosts.length,
    selectedHostKey: state.selectedHostKey,
    compareMode: state.compareMode,
    compareStats: state.lastCompareData?.stats || null,
    horizon: state.horizon,
    ghostVisible: fleet.ghostGroup.visible,
    ghostColorHex: fleet.ghostMat.color.getHexString(),
    palette: colors.palette,
    anatomyMode: state.anatomyMode,
    anatomyHostId: anatomy.host?.id || null,
    anatomyHasFans: anatomy.hasFans,
    anatomyHasInlet: anatomy.hasInlet,
    anatomyFanLabel: anatomy.fanStatusText,
    anatomyInletLabel: anatomy.inletStatusText,
  }),
  jumpToTime: (iso) => hud.setTimeToIso(iso),
};

// Render loop & FPS monitor
let lastFrameMs = performance.now();
let fpsFrames = 0;
let fpsLastCheck = performance.now();
let currentFps = 60;
let lowFpsStreak = 0;

function animate(now) {
  requestAnimationFrame(animate);
  const dt = Math.min(0.1, (now - lastFrameMs) / 1000);
  lastFrameMs = now;

  if (camFlight.active) {
    const t = Math.min(1, (now - camFlight.startMs) / camFlight.durationMs);
    const s = t * t * (3 - 2 * t);
    camera.position.lerpVectors(camFlight.fromPos, camFlight.toPos, s);
    controls.target.lerpVectors(camFlight.fromTarget, camFlight.toTarget, s);
    if (t >= 1) camFlight.active = false;
  }
  controls.update();

  if (state.compareMode && state.lastCompareData) {
    fleet._uTime.value += dt;
    const w = container.clientWidth, h = container.clientHeight;
    renderer.setScissorTest(true);
    if (state.singleViewport) {
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setViewport(0, 0, w, h);
      renderer.setScissor(0, 0, w, h);
      fleet.halos.visible = state.singleSide === "right" && state.horizon > 0;
      fleet.applyCompareSide(state.singleSide, state.errorOverlay, state.legendsByCluster);
      renderer.render(scene, camera);
    } else {
      const halfW = Math.floor(w / 2);
      camera.aspect = Math.max(1, halfW) / Math.max(1, h);
      camera.updateProjectionMatrix();

      renderer.setViewport(0, 0, halfW, h);
      renderer.setScissor(0, 0, halfW, h);
      fleet.halos.visible = false;
      fleet.applyCompareSide("left", false, state.legendsByCluster);
      renderer.render(scene, camera);

      renderer.setViewport(halfW, 0, w - halfW, h);
      renderer.setScissor(halfW, 0, w - halfW, h);
      fleet.halos.visible = state.horizon > 0 && fleet.halosEnabled;
      fleet.applyCompareSide("right", state.errorOverlay, state.legendsByCluster);
      renderer.render(scene, camera);
    }
    renderer.setScissorTest(false);
  } else if (state.anatomyMode) {
    anatomy.tick(now / 1000, dt);
    if (state.bloomEnabled) {
      composer.render();
    } else {
      renderer.render(scene, camera);
    }
  } else {
    fleet.tick(now, dt);
    fleet.halos.visible = state.horizon > 0 && fleet.halosEnabled;
    if (state.bloomEnabled) {
      composer.render();
    } else {
      renderer.render(scene, camera);
    }
  }
  labelRenderer.render(scene, camera);

  fpsFrames++;
  if (now - fpsLastCheck >= 1000) {
    currentFps = fpsFrames;
    fpsFrames = 0;
    fpsLastCheck = now;
    if (currentFps < 30) {
      lowFpsStreak++;
      if (lowFpsStreak >= 3) {
        state.bloomEnabled = false;
        bloomPass.enabled = false;
        fleet.halosEnabled = false;
        if (state.compareMode && !state.singleViewport) {
          state.singleViewport = true;
          hud.elSingleView.checked = true;
        }
      }
    } else {
      lowFpsStreak = 0;
    }
    hud.updatePerf(currentFps, renderer.info.render.calls, state.bloomEnabled, fleet.halosEnabled);
  }
}

async function boot() {
  try {
    const meta = await api.getMeta();
    state.meta = meta;
    for (const [cid, cm] of Object.entries(meta.clusters)) {
      state.legendsByCluster[cid] = cm.legend;
    }
    hall.buildRacks(meta.clusters);
    hud.buildTimeline(meta, state.clusterFilter);
    panel.setMeta(meta, state.clusterFilter);
    refreshLegend();
    await refreshFrame();
    statusOverlay.style.display = "none";
    requestAnimationFrame(animate);
  } catch (err) {
    statusOverlay.textContent = `model not available (${err.message})`;
  }
}

boot();
