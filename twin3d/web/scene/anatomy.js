/**
 * scene/anatomy.js — Single-server close-up anatomy view.
 *
 * Shows translucent chassis, CPU1 & CPU2 heatsink blocks (real CPU1/CPU2 temps),
 * 4 rotating front fans (D1 omega = clamp(rpm/rpm_ref, 0, 1) * 12 rad/s),
 * deterministic airflow streaks coloured by inlet temp, and Power LED.
 * Cluster 2 (or hosts missing fan/inlet sensors): static grey parts labelled
 * "no fan sensor" and "no inlet sensor", zero animation or fabricated values.
 */
import * as THREE from "three";
import { CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { GREY_COLOR, LED_OFF_COLOR } from "../colors.js";

const FAN_RAD_CAP = 12.0;
const STREAK_COUNT = 12;

export class ServerAnatomyScene {
  constructor(scene, colorMapper) {
    this.scene = scene;
    this.colors = colorMapper;
    this.group = new THREE.Group();
    this.group.visible = false;
    this.scene.add(this.group);

    this.host = null;
    this.fanOmegas = [0, 0, 0, 0];
    this.streakSpeed = 0;
    this.hasFans = false;
    this.hasInlet = false;
    this.fanStatusText = "";
    this.inletStatusText = "";
    this.streakPhases = [];
    for (let i = 0; i < STREAK_COUNT; i++) {
      this.streakPhases.push(i / STREAK_COUNT);
    }

    // Chassis base + translucent lid + wireframe
    const chassisGeo = new THREE.BoxGeometry(1.0, 0.24, 1.38);
    const chassisMat = new THREE.MeshStandardMaterial({
      color: 0x262c38,
      transparent: true,
      opacity: 0.26,
      roughness: 0.4,
      depthWrite: false,
    });
    const chassis = new THREE.Mesh(chassisGeo, chassisMat);
    chassis.position.set(0, 0.26, 0);
    const chassisEdges = new THREE.LineSegments(
      new THREE.EdgesGeometry(chassisGeo),
      new THREE.LineBasicMaterial({ color: 0x6b7892 })
    );
    chassisEdges.position.copy(chassis.position);
    const floorTray = new THREE.Mesh(
      new THREE.BoxGeometry(0.98, 0.02, 1.36),
      new THREE.MeshStandardMaterial({ color: 0x1a1e28, roughness: 0.75 })
    );
    floorTray.position.set(0, 0.145, 0);
    this.group.add(floorTray, chassis, chassisEdges);

    // Two CPU blocks + heatsink fins (CPU1 left, CPU2 right)
    this.cpu1Mat = new THREE.MeshBasicMaterial({ color: new THREE.Color(GREY_COLOR), toneMapped: false });
    this.cpu2Mat = new THREE.MeshBasicMaterial({ color: new THREE.Color(GREY_COLOR), toneMapped: false });
    this._buildCpuAssembly(-0.22, 0.21, -0.08, this.cpu1Mat);
    this._buildCpuAssembly(+0.22, 0.21, -0.08, this.cpu2Mat);

    // Four fan modules on front (+Z = +0.66)
    this.fanMats = [];
    this.fanRotors = [];
    const fanXs = [-0.33, -0.11, +0.11, +0.33];
    for (let i = 0; i < 4; i++) {
      const fGroup = new THREE.Group();
      fGroup.position.set(fanXs[i], 0.26, 0.65);

      const shroud = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(0.18, 0.18, 0.04)),
        new THREE.LineBasicMaterial({ color: 0x5c667a })
      );
      fGroup.add(shroud);

      const rotor = new THREE.Group();
      const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color(GREY_COLOR) });
      const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.024, 0.024, 0.02, 12).rotateX(Math.PI / 2), mat);
      rotor.add(hub);
      for (let b = 0; b < 4; b++) {
        const blade = new THREE.Mesh(new THREE.BoxGeometry(0.072, 0.018, 0.012), mat);
        const ang = (b * Math.PI) / 2;
        blade.position.set(Math.cos(ang) * 0.044, Math.sin(ang) * 0.044, 0);
        blade.rotation.z = ang + 0.3;
        rotor.add(blade);
      }
      fGroup.add(rotor);
      this.group.add(fGroup);
      this.fanMats.push(mat);
      this.fanRotors.push(rotor);
    }

    // Intake bar at front bottom (+Z = +0.71)
    this.intakeMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(GREY_COLOR) });
    const intakeBar = new THREE.Mesh(new THREE.BoxGeometry(0.92, 0.04, 0.03), this.intakeMat);
    intakeBar.position.set(0, 0.155, 0.70);
    this.group.add(intakeBar);

    // Deterministic airflow streaks (12 evenly distributed lanes along X/Y grid)
    this.streakMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(GREY_COLOR),
      transparent: true,
      opacity: 0.75,
      toneMapped: false,
    });
    this.streaks = [];
    const streakGeo = new THREE.BoxGeometry(0.016, 0.016, 0.14);
    for (let i = 0; i < STREAK_COUNT; i++) {
      const col = i % 4;
      const row = Math.floor(i / 4);
      const m = new THREE.Mesh(streakGeo, this.streakMat);
      m.position.set(-0.33 + col * 0.22, 0.21 + row * 0.055, 0.75);
      this.group.add(m);
      this.streaks.push(m);
    }

    // Power LED on front left
    this.ledMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(LED_OFF_COLOR), toneMapped: false });
    const led = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.04, 0.04), this.ledMat);
    led.position.set(-0.46, 0.34, 0.69);
    this.group.add(led);

    // CSS2D annotations
    this.lblCpu1 = this._makeLabel(-0.22, 0.36, -0.08);
    this.lblCpu2 = this._makeLabel(+0.22, 0.36, -0.08);
    this.lblFans = this._makeLabel(0, 0.42, 0.65);
    this.lblInlet = this._makeLabel(0, 0.12, 0.76);
    this.lblPower = this._makeLabel(-0.46, 0.41, 0.69);
    this.labels = [this.lblCpu1, this.lblCpu2, this.lblFans, this.lblInlet, this.lblPower];
  }

  _buildCpuAssembly(x, y, z, mat) {
    const base = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.04, 0.24), mat);
    base.position.set(x, y, z);
    this.group.add(base);
    for (let f = -2; f <= 2; f++) {
      const fin = new THREE.Mesh(new THREE.BoxGeometry(0.23, 0.06, 0.012), mat);
      fin.position.set(x, y + 0.05, z + f * 0.045);
      this.group.add(fin);
    }
  }

  _makeLabel(x, y, z) {
    const div = document.createElement("div");
    div.className = "hover-label";
    const lbl = new CSS2DObject(div);
    lbl.position.set(x, y, z);
    lbl.visible = false;
    this.group.add(lbl);
    return lbl;
  }

  setHost(host, legend) {
    this.host = host;
    if (!host) return;

    const c = new THREE.Color();
    // CPU1 & CPU2 colours
    if (host.status === "ok" || host.status === "stale") {
      this.colors.tempToColor(host.cpu1, legend, c).multiplyScalar(1.25);
      this.cpu1Mat.color.copy(c);
      this.colors.tempToColor(host.cpu2, legend, c).multiplyScalar(1.25);
      this.cpu2Mat.color.copy(c);
      this.colors.powerToLedColor(host.power, legend, c);
      this.ledMat.color.copy(c);
    } else {
      this.cpu1Mat.color.copy(GREY_COLOR);
      this.cpu2Mat.color.copy(GREY_COLOR);
      this.ledMat.color.copy(LED_OFF_COLOR);
    }

    this.lblCpu1.element.textContent = host.cpu1 != null ? `CPU1: ${host.cpu1}°C` : "CPU1: missing";
    this.lblCpu2.element.textContent = host.cpu2 != null ? `CPU2: ${host.cpu2}°C` : "CPU2: missing";
    this.lblPower.element.textContent = host.power != null ? `Power: ${host.power} W` : "Power: missing";

    // Fans (real on D1, grey static "no fan sensor" on D2 or unavailable)
    this.hasFans = Boolean(host.fans_available && Array.isArray(host.fans) && host.status === "ok");
    const rpmRef = legend?.rpm_ref || 10000;
    let rpmSum = 0;
    let rpmCount = 0;
    for (let i = 0; i < 4; i++) {
      if (this.hasFans && host.fans[i] != null) {
        const rpm = host.fans[i];
        this.fanOmegas[i] = Math.max(0, Math.min(1, rpm / rpmRef)) * FAN_RAD_CAP;
        this.fanMats[i].color.setHex(0x8bb8e8);
        rpmSum += rpm;
        rpmCount++;
      } else {
        this.fanOmegas[i] = 0;
        this.fanMats[i].color.setHex(0x555860);
      }
    }

    if (host.fans_available && rpmCount > 0) {
      const meanRpm = rpmSum / rpmCount;
      this.fanStatusText = `Fans mean ${Math.round(meanRpm)} RPM (visual cap 12 rad/s)`;
      this.streakSpeed = 0.4 + 1.4 * Math.max(0, Math.min(1, meanRpm / rpmRef));
    } else {
      this.fanStatusText = "no fan sensor";
      this.streakSpeed = 0;
    }
    this.lblFans.element.textContent = this.fanStatusText;

    // Inlet & airflow streaks
    this.hasInlet = Boolean(host.inlet_available && host.inlet != null && host.status === "ok");
    if (this.hasInlet) {
      this.colors.tempToColor(host.inlet, legend, c);
      this.intakeMat.color.copy(c);
      this.streakMat.color.copy(c);
      this.inletStatusText = `Inlet: ${host.inlet}°C`;
      for (const s of this.streaks) s.visible = this.streakSpeed > 0;
    } else {
      this.intakeMat.color.copy(GREY_COLOR);
      this.streakMat.color.copy(GREY_COLOR);
      this.inletStatusText = "no inlet sensor";
      for (const s of this.streaks) s.visible = false;
    }
    this.lblInlet.element.textContent = this.inletStatusText;
  }

  setVisible(visible) {
    this.group.visible = visible;
    for (const lbl of this.labels) {
      lbl.visible = visible;
    }
    if (!visible) {
      for (const s of this.streaks) s.visible = false;
    }
  }

  tick(nowSec, dtSec) {
    if (!this.group.visible) return;
    for (let i = 0; i < 4; i++) {
      if (this.fanOmegas[i] > 0) {
        this.fanRotors[i].rotation.z -= this.fanOmegas[i] * dtSec;
      }
    }
    if (this.hasInlet && this.streakSpeed > 0) {
      for (let i = 0; i < this.streaks.length; i++) {
        const travel = (nowSec * this.streakSpeed + this.streakPhases[i] * 1.4) % 1.4;
        this.streaks[i].position.z = 0.72 - travel;
      }
    }
  }
}
