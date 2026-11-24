/**
 * scene/servers.js — Zero-allocation instanced rendering for all hosts across D1 & D2.
 *
 * Draw calls:
 *  - 1 InstancedMesh: server slabs
 *  - 1 InstancedMesh: emissive glow strips (colour-eased over 300ms)
 *  - 1 InstancedMesh: power LEDs
 *  - 1 InstancedMesh: uncertainty halos (transparent, depthWrite:false)
 *  - 1 InstancedMesh: pulsing risk beacons (server + rack top)
 *  - 1 InstancedMesh: fan blades with instanced aOmega + uTime shader (no per-frame matrices)
 *  - 1 LineSegments: selection highlight box
 *  - 1 Mesh + halo: What-If ghost server
 */
import * as THREE from "three";
import { CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { GREY_COLOR, LED_OFF_COLOR } from "../colors.js";

const MAX_HOSTS = 150;
const MAX_RACKS = 16;
const MAX_BEACONS = MAX_HOSTS + MAX_RACKS;
const EASE_DURATION_MS = 300;
const FAN_RAD_CAP = 12.0;

function createFanBladeGeometry() {
  // Crisp cross-blade fan rotor in XY plane facing +Z
  const positions = [];
  const addBlade = (ang) => {
    const r0 = 0.005, r1 = 0.031, w = 0.008;
    const c = Math.cos(ang), s = Math.sin(ang);
    const nx = -s * w, ny = c * w;
    const ax = c * r0, ay = s * r0;
    const bx = c * r1, by = s * r1;
    // Two triangles per blade
    positions.push(
      ax - nx, ay - ny, 0,
      bx - nx, by - ny, 0,
      bx + nx, by + ny, 0,
      ax - nx, ay - ny, 0,
      bx + nx, by + ny, 0,
      ax + nx, ay + ny, 0
    );
  };
  for (let b = 0; b < 4; b++) addBlade((b * Math.PI) / 2);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.computeVertexNormals();
  return geo;
}

export class ServerFleetScene {
  constructor(scene, colorMapper) {
    this.scene = scene;
    this.colors = colorMapper;
    this.group = new THREE.Group();
    this.scene.add(this.group);

    this.hosts = [];
    this.hostByKey = new Map();
    this.selectedIdx = -1;
    this.hoveredIdx = -1;
    this.halosEnabled = true;

    // Reusable scratch math objects (zero per-frame GC)
    this._mat4 = new THREE.Matrix4();
    this._pos = new THREE.Vector3();
    this._quat = new THREE.Quaternion();
    this._scale = new THREE.Vector3(1, 1, 1);
    this._col = new THREE.Color();

    // Easing buffers
    this._prevGlow = new Float32Array(MAX_HOSTS * 3);
    this._nextGlow = new Float32Array(MAX_HOSTS * 3);
    this._hasPrevGlow = false;
    this._easeStart = 0;

    // Shader time uniform for fans & beacons
    this._uTime = { value: 0 };

    // 1. Slabs
    this.slabGeo = new THREE.BoxGeometry(0.48, 0.0845, 0.72);
    this.slabMat = new THREE.MeshStandardMaterial({
      color: 0x262a34,
      roughness: 0.65,
      metalness: 0.35,
    });
    this.slabs = new THREE.InstancedMesh(this.slabGeo, this.slabMat, MAX_HOSTS);
    this.slabs.count = 0;
    this.slabs.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.group.add(this.slabs);

    // 2. Emissive glow strips (toneMapped:false + HDR scalar for selective bloom)
    this.glowGeo = new THREE.BoxGeometry(0.46, 0.018, 0.735);
    this.glowMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(1.35, 1.35, 1.35),
      toneMapped: false,
    });
    this.glows = new THREE.InstancedMesh(this.glowGeo, this.glowMat, MAX_HOSTS);
    this.glows.count = 0;
    this.glows.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(MAX_HOSTS * 3), 3);
    this.group.add(this.glows);

    // 3. Power LEDs
    this.ledGeo = new THREE.BoxGeometry(0.024, 0.024, 0.024);
    this.ledMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(1.4, 1.4, 1.4),
      toneMapped: false,
    });
    this.leds = new THREE.InstancedMesh(this.ledGeo, this.ledMat, MAX_HOSTS);
    this.leds.count = 0;
    this.leds.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(MAX_HOSTS * 3), 3);
    this.group.add(this.leds);

    // 4. Uncertainty halos
    this.haloGeo = new THREE.BoxGeometry(0.50, 0.095, 0.74);
    this.haloMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.16,
      depthWrite: false,
    });
    this.halos = new THREE.InstancedMesh(this.haloGeo, this.haloMat, MAX_HOSTS);
    this.halos.count = 0;
    this.halos.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(MAX_HOSTS * 3), 3);
    this.group.add(this.halos);

    // 5. Risk beacons (host front top + rack top rings)
    this.beaconGeo = new THREE.TorusGeometry(0.055, 0.012, 8, 20);
    this.beaconGeo.rotateX(Math.PI / 2);
    this.beaconMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(1.4, 1.4, 1.4),
      transparent: true,
      opacity: 0.9,
      toneMapped: false,
    });
    this.beacons = new THREE.InstancedMesh(this.beaconGeo, this.beaconMat, MAX_BEACONS);
    this.beacons.count = 0;
    this.beacons.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(MAX_BEACONS * 3), 3);
    this.group.add(this.beacons);

    // 6. Fan rotors (4 per host) with GPU shader spin via aOmega attribute + uTime uniform
    this.fanGeo = createFanBladeGeometry();
    this._aOmega = new Float32Array(MAX_HOSTS * 4);
    this.fanGeo.setAttribute("aOmega", new THREE.InstancedBufferAttribute(this._aOmega, 1));
    this.fanMat = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });
    this.fanMat.onBeforeCompile = (shader) => {
      shader.uniforms.uTime = this._uTime;
      shader.vertexShader = shader.vertexShader
        .replace(
          "#include <common>",
          "#include <common>\nattribute float aOmega;\nuniform float uTime;"
        )
        .replace(
          "#include <begin_vertex>",
          `#include <begin_vertex>
          float _fang = -aOmega * uTime;
          float _fc = cos(_fang);
          float _fs = sin(_fang);
          transformed.xy = vec2(transformed.x * _fc - transformed.y * _fs, transformed.x * _fs + transformed.y * _fc);`
        );
    };
    this.fans = new THREE.InstancedMesh(this.fanGeo, this.fanMat, MAX_HOSTS * 4);
    this.fans.count = 0;
    this.fans.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(MAX_HOSTS * 4 * 3), 3);
    this.group.add(this.fans);

    // 7. Selection wireframe box
    const selEdges = new THREE.EdgesGeometry(new THREE.BoxGeometry(0.53, 0.11, 0.77));
    this.selectionBox = new THREE.LineSegments(
      selEdges,
      new THREE.LineBasicMaterial({ color: 0xffffff, toneMapped: false })
    );
    this.selectionBox.visible = false;
    this.group.add(this.selectionBox);

    // 8. What-If semi-transparent ghost server + halo
    this.ghostGroup = new THREE.Group();
    this.ghostMat = new THREE.MeshBasicMaterial({
      color: 0x21918c,
      transparent: true,
      opacity: 0.58,
      depthWrite: false,
      toneMapped: false,
    });
    this.ghostMesh = new THREE.Mesh(new THREE.BoxGeometry(0.48, 0.0845, 0.72), this.ghostMat);
    this.ghostHaloMat = new THREE.MeshBasicMaterial({
      color: 0x21918c,
      transparent: true,
      opacity: 0.22,
      depthWrite: false,
    });
    this.ghostHalo = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.10, 0.76), this.ghostHaloMat);
    const ghostEdges = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(0.49, 0.088, 0.73)),
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.7 })
    );
    const ghostDiv = document.createElement("div");
    ghostDiv.className = "hover-label";
    ghostDiv.textContent = "What-If Ghost";
    this.ghostLabel = new CSS2DObject(ghostDiv);
    this.ghostLabel.position.set(0, 0.12, 0);
    this.ghostGroup.add(this.ghostMesh, this.ghostHalo, ghostEdges, this.ghostLabel);
    this.ghostGroup.visible = false;
    this.group.add(this.ghostGroup);

    // 9. Single reusable hover tooltip
    const hDiv = document.createElement("div");
    hDiv.className = "hover-label";
    this.hoverLabel = new CSS2DObject(hDiv);
    this.hoverLabel.visible = false;
    this.group.add(this.hoverLabel);
  }

  /**
   * Update visual instances from snapshot hosts array.
   * options: { horizon, colorMode ('temp'|'risk'|'error'), legendsByCluster, compareMode, showHalos }
   */
  updateSnapshot(hosts, options) {
    this.hosts = hosts;
    this.hostByKey.clear();
    const n = Math.min(hosts.length, MAX_HOSTS);

    this.slabs.count = n;
    this.glows.count = n;
    this.leds.count = n;
    this.halos.count = this.halosEnabled && options.showHalos ? n : 0;
    this.fans.count = n * 4;

    const fanOffsets = [-0.15, -0.05, 0.05, 0.15];
    let beaconCount = 0;
    const rackAlertWorst = new Map(); // key `${cluster}|${rack}` -> {x, y, z, risk}

    for (let i = 0; i < n; i++) {
      const h = hosts[i];
      const key = `${h.cluster || options.defaultCluster || "d1"}|${h.id}`;
      this.hostByKey.set(key, i);

      const leg = options.legendsByCluster[h.cluster || options.defaultCluster || "d1"];
      const tEff = options.isCompareRight ? h.twin : (options.isCompareLeft ? h.reality : (h.forecast && options.horizon > 0 ? h.forecast.point : h.temp_eff));
      const fc = h.forecast;
      const risk = h.risk || (fc ? fc.risk : "unknown");

      // Matrix: slab + glow strip
      this._pos.set(h.x, h.y, h.z);
      this._scale.set(1, 1, 1);
      this._mat4.compose(this._pos, this._quat, this._scale);
      this.slabs.setMatrixAt(i, this._mat4);

      this._pos.set(h.x, h.y + 0.034, h.z + 0.005);
      this._mat4.compose(this._pos, this._quat, this._scale);
      this.glows.setMatrixAt(i, this._mat4);

      // Power LED at front-left
      this._pos.set(h.x - 0.21, h.y, h.z + 0.365);
      this._mat4.compose(this._pos, this._quat, this._scale);
      this.leds.setMatrixAt(i, this._mat4);
      if (h.status === "missing" || h.status === "stale") {
        this.leds.setColorAt(i, LED_OFF_COLOR);
      } else {
        this.colors.powerToLedColor(h.power, leg, this._col);
        this.leds.setColorAt(i, this._col);
      }

      // Glow color (target)
      if (h.status === "missing" || h.status === "stale" || h.status === "unknown") {
        this._col.copy(GREY_COLOR);
      } else if (options.errorOverlay && options.isCompareMode) {
        this.colors.errorToColor(h.error, this._col);
      } else if (options.colorMode === "risk") {
        this.colors.riskToColor(risk, this._col);
      } else {
        this.colors.tempToColor(tEff, leg, this._col);
      }
      this._nextGlow[i * 3] = this._col.r;
      this._nextGlow[i * 3 + 1] = this._col.g;
      this._nextGlow[i * 3 + 2] = this._col.b;

      // Halo shell scale & color
      const lower = h.lower ?? fc?.lower;
      const upper = h.upper ?? fc?.upper;
      const wRef = leg?.w_ref || 5.0;
      let haloScale = 1.0;
      if (options.showHalos && lower != null && upper != null && h.status === "ok") {
        const width = Math.max(0, upper - lower);
        haloScale = 1.0 + 0.35 * Math.max(0, Math.min(1, width / wRef));
        this._scale.set(haloScale, 1.0 + (haloScale - 1.0) * 1.4, haloScale);
        this._col.setRGB(this._nextGlow[i * 3], this._nextGlow[i * 3 + 1], this._nextGlow[i * 3 + 2]);
      } else {
        this._scale.set(0.001, 0.001, 0.001);
        this._col.copy(GREY_COLOR);
      }
      this._pos.set(h.x, h.y, h.z);
      this._mat4.compose(this._pos, this._quat, this._scale);
      this.halos.setMatrixAt(i, this._mat4);
      this.halos.setColorAt(i, this._col);

      // Fans (4 per host)
      const hasFans = Boolean(h.fans_available && Array.isArray(h.fans) && h.status === "ok");
      const rpmRef = leg?.rpm_ref || 10000;
      for (let f = 0; f < 4; f++) {
        const fi = i * 4 + f;
        this._pos.set(h.x + fanOffsets[f], h.y, h.z + 0.365);
        this._scale.set(1, 1, 1);
        this._mat4.compose(this._pos, this._quat, this._scale);
        this.fans.setMatrixAt(fi, this._mat4);
        if (hasFans && h.fans[f] != null) {
          this._aOmega[fi] = Math.max(0, Math.min(1, h.fans[f] / rpmRef)) * FAN_RAD_CAP;
          this._col.setHex(0x8bb8e8);
        } else {
          this._aOmega[fi] = 0.0;
          this._col.setHex(0x4b4e54);
        }
        this.fans.setColorAt(fi, this._col);
      }

      // Risk beacons on server & rack top (only when warning/critical)
      const alertRisk = h.alert_risk || risk;
      if ((alertRisk === "warning" || alertRisk === "critical") && beaconCount < MAX_HOSTS) {
        this._pos.set(h.x + 0.21, h.y + 0.05, h.z + 0.36);
        this._scale.set(0.75, 0.75, 0.75);
        this._mat4.compose(this._pos, this._quat, this._scale);
        this.beacons.setMatrixAt(beaconCount, this._mat4);
        this.colors.riskToColor(alertRisk, this._col);
        this.beacons.setColorAt(beaconCount, this._col);
        beaconCount++;

        const rkKey = `${h.cluster || "d1"}|${h.rack}`;
        const prevRk = rackAlertWorst.get(rkKey);
        if (!prevRk || alertRisk === "critical") {
          rackAlertWorst.set(rkKey, { x: h.x, z: h.z, risk: alertRisk });
        }
      }
    }

    // Rack-top beacons
    for (const rk of rackAlertWorst.values()) {
      if (beaconCount >= MAX_BEACONS) break;
      this._pos.set(rk.x, 1.93, rk.z);
      this._scale.set(3.2, 2.0, 3.2);
      this._mat4.compose(this._pos, this._quat, this._scale);
      this.beacons.setMatrixAt(beaconCount, this._mat4);
      this.colors.riskToColor(rk.risk, this._col);
      this.beacons.setColorAt(beaconCount, this._col);
      beaconCount++;
    }
    this.beacons.count = beaconCount;

    if (!this._hasPrevGlow) {
      this._prevGlow.set(this._nextGlow);
      this._hasPrevGlow = true;
    } else {
      // Snap current eased colours into _prevGlow so mid-ease transitions stay smooth
      this._prevGlow.set(this.glows.instanceColor.array);
    }
    this._easeStart = performance.now();

    this.slabs.instanceMatrix.needsUpdate = true;
    this.glows.instanceMatrix.needsUpdate = true;
    this.leds.instanceMatrix.needsUpdate = true;
    if (this.leds.instanceColor) this.leds.instanceColor.needsUpdate = true;
    this.halos.instanceMatrix.needsUpdate = true;
    if (this.halos.instanceColor) this.halos.instanceColor.needsUpdate = true;
    this.fans.instanceMatrix.needsUpdate = true;
    if (this.fans.instanceColor) this.fans.instanceColor.needsUpdate = true;
    this.fanGeo.attributes.aOmega.needsUpdate = true;
    this.beacons.instanceMatrix.needsUpdate = true;
    if (this.beacons.instanceColor) this.beacons.instanceColor.needsUpdate = true;

    this.slabs.computeBoundingSphere();
    this._syncSelectionBox();
  }

  /** Trigger immediate recolour without moving geometry (e.g. compare left vs right scissor pass). */
  applyCompareSide(side, errorOverlay, legendsByCluster) {
    const n = this.slabs.count;
    const attr = this.glows.instanceColor;
    for (let i = 0; i < n; i++) {
      const h = this.hosts[i];
      if (!h) continue;
      const leg = legendsByCluster[h.cluster || "d1"];
      if (h.status === "missing" || h.status === "stale" || h.status === "unknown") {
        this._col.copy(GREY_COLOR);
      } else if (errorOverlay && side === "right") {
        this.colors.errorToColor(h.error, this._col);
      } else {
        const val = side === "left" ? h.reality : h.twin;
        this.colors.tempToColor(val, leg, this._col);
      }
      attr.setXYZ(i, this._col.r, this._col.g, this._col.b);
    }
    attr.needsUpdate = true;
  }

  setSelectedIndex(idx) {
    this.selectedIdx = idx;
    this._syncSelectionBox();
  }

  _syncSelectionBox() {
    if (this.selectedIdx >= 0 && this.selectedIdx < this.hosts.length) {
      const h = this.hosts[this.selectedIdx];
      this.selectionBox.position.set(h.x, h.y, h.z);
      this.selectionBox.visible = true;
    } else {
      this.selectionBox.visible = false;
    }
  }

  setHover(idx) {
    this.hoveredIdx = idx;
    if (idx >= 0 && idx < this.hosts.length) {
      const h = this.hosts[idx];
      const tStr = h.temp_eff != null ? `${h.temp_eff.toFixed(1)} °C` : h.status;
      this.hoverLabel.element.textContent = `${h.id} · ${tStr}`;
      this.hoverLabel.position.set(h.x, h.y + 0.11, h.z + 0.42);
      this.hoverLabel.visible = true;
    } else {
      this.hoverLabel.visible = false;
    }
  }

  updateGhost(visible, pos, whatIfData, legend) {
    this.ghostGroup.visible = Boolean(visible);
    if (!visible || !whatIfData) return;
    this.ghostGroup.position.set(pos.x, pos.y, pos.z);
    this.colors.tempToColor(whatIfData.point, legend, this._col);
    this.ghostMat.color.copy(this._col);
    this.ghostHaloMat.color.copy(this._col);
    const wRef = legend?.w_ref || 16.0;
    const width = whatIfData.width ?? (whatIfData.upper != null && whatIfData.lower != null ? whatIfData.upper - whatIfData.lower : 0);
    const hs = 1.0 + 0.35 * Math.max(0, Math.min(1, width / wRef));
    this.ghostHalo.scale.set(hs, 1 + (hs - 1) * 1.4, hs);
  }

  tick(nowMs, dtSec) {
    this._uTime.value += dtSec;

    // Ease glow colours over 300ms
    const n = this.glows.count;
    if (n > 0 && this._hasPrevGlow) {
      const t = Math.min(1, (nowMs - this._easeStart) / EASE_DURATION_MS);
      const attr = this.glows.instanceColor;
      const arr = attr.array;
      for (let i = 0; i < n * 3; i++) {
        arr[i] = this._prevGlow[i] + t * (this._nextGlow[i] - this._prevGlow[i]);
      }
      attr.needsUpdate = true;
    }

    // Subtle pulse on risk beacons via material opacity
    if (this.beacons.count > 0) {
      this.beaconMat.opacity = 0.55 + 0.45 * (0.5 + 0.5 * Math.sin(this._uTime.value * 4.2));
    }
  }

  setVisible(visible) {
    this.group.visible = visible;
    if (!visible) {
      this.hoverLabel.visible = false;
      this.ghostLabel.visible = false;
    } else if (this.ghostGroup.visible) {
      this.ghostLabel.visible = true;
    }
  }
}
