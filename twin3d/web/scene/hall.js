/**
 * scene/hall.js — Schematic data hall floor, grid, merged rack wireframes (1 LineSegments),
 * and CSS2D cluster/rack labels.
 */
import * as THREE from "three";
import { CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

export class HallScene {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    this.scene.add(this.group);

    this.rackLines = null;
    this.labels = [];

    // Floor and schematic reference grid
    const floorGeo = new THREE.PlaneGeometry(18, 16);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x0f1218,
      roughness: 0.92,
      metalness: 0.08,
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(1.6, 0, -2.0);
    this.group.add(floor);

    const grid = new THREE.GridHelper(16, 32, 0x262d3d, 0x171b24);
    grid.position.set(1.6, 0.002, -2.0);
    this.group.add(grid);
  }

  buildRacks(clustersMeta) {
    if (this.rackLines) {
      this.group.remove(this.rackLines);
      this.rackLines.geometry.dispose();
      this.rackLines = null;
    }
    for (const lbl of this.labels) {
      if (lbl.parent) lbl.parent.remove(lbl);
    }
    this.labels.length = 0;

    // Collect line segments for all racks into ONE BufferGeometry / LineSegments (1 draw call)
    const boxEdgeCache = new Map();
    const positions = [];

    for (const [cid, cm] of Object.entries(clustersMeta)) {
      const ly = cm.layout;
      const rw = ly.rack.w;
      const rh = ly.rack.h;
      const rd = ly.rack.d;
      const cz = ly.cluster_z;
      const rackSize = ly.rack_size || 20;
      const nRacks = Math.max(1, Math.ceil(cm.host_count / rackSize));
      const cLabel = cid === "d1" ? "C1" : "C2";

      let baseEdges = boxEdgeCache.get(cid);
      if (!baseEdges) {
        const eg = new THREE.EdgesGeometry(new THREE.BoxGeometry(rw, rh, rd));
        baseEdges = eg.attributes.position.array;
        boxEdgeCache.set(cid, baseEdges);
      }

      // Row header label
      const midX = ((nRacks - 1) * ly.rack_pitch_x) * 0.5;
      const rowDiv = document.createElement("div");
      rowDiv.className = "row-label";
      rowDiv.textContent = cid === "d1" ? "Cluster 1 (D1) — schematic row" : "Cluster 2 (D2) — schematic row (no fan/inlet sensors)";
      const rowLbl = new CSS2DObject(rowDiv);
      rowLbl.position.set(midX, rh + 0.42, cz);
      this.group.add(rowLbl);
      this.labels.push(rowLbl);

      for (let r = 0; r < nRacks; r++) {
        const rx = r * ly.rack_pitch_x;
        const ry = rh * 0.5;
        for (let k = 0; k < baseEdges.length; k += 3) {
          positions.push(baseEdges[k] + rx, baseEdges[k + 1] + ry, baseEdges[k + 2] + cz);
        }

        const rDiv = document.createElement("div");
        rDiv.className = "rack-label";
        rDiv.textContent = `${cLabel}-R${r + 1}`;
        const rLbl = new CSS2DObject(rDiv);
        rLbl.position.set(rx, rh + 0.14, cz);
        this.group.add(rLbl);
        this.labels.push(rLbl);
      }
    }

    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    const lineMat = new THREE.LineBasicMaterial({ color: 0x455066, transparent: true, opacity: 0.72 });
    this.rackLines = new THREE.LineSegments(lineGeo, lineMat);
    this.group.add(this.rackLines);
  }

  setVisible(visible) {
    this.group.visible = visible;
    for (const lbl of this.labels) {
      lbl.visible = visible;
    }
  }
}
