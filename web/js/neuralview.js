// Neural view — the real FlyWire brain (139,255 neurons from FAFB v783).
import * as THREE from "three";

const UM2U = 0.25;   // µm → world units (814 µm brain → ~200 u)

// super_class palette (by code index, codebooks come from the layout header)
const PALETTE = {
  optic: 0x5f9eff, central: 0xffb454, sensory: 0x7dffb2,
  visual_projection: 0xc88bff, ascending: 0xff8bd0, descending: 0xff6b5b,
  sensory_ascending: 0xffd38b, visual_centrifugal: 0x9bffea,
  motor: 0xffe14f, endocrine: 0xbaffc5, "": 0x777777,
};

export class NeuralView {
  constructor(container) {
    this.container = container;
    const w = container.clientWidth || 800, h = container.clientHeight || 600;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x07090d);
    this.camera = new THREE.PerspectiveCamera(55, w / h, 0.1, 3000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    container.appendChild(this.renderer.domElement);
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.0));

    this.orbit = { theta: 2.4, phi: 1.25, r: 210, cx: 100, cy: 0, cz: 50 };
    this._bindOrbit();
    this._bindPick();

    this.loaded = false;
    this.activePts = null;
    this.prevActive = new Map();
    this.selObj = null;
    this.traceGroup = null;
    this.traceAnim = null;
    this.highlight = "none";
  }

  async load() {
    const resp = await fetch("/api/layout");
    if (!resp.ok) throw new Error("layout fetch failed");
    const buf = await resp.arrayBuffer();
    const nl = new Uint8Array(buf, 0, 1).indexOf(10);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 0, nl)));
    const n = header.n;
    let off = nl + 1;
    this.pos = new Float32Array(buf, off, n * 3); off += n * 12;
    this.sup = new Int16Array(buf, off, n); off += n * 2;
    this.side = new Int16Array(buf, off, n); off += n * 2;
    this.nt = new Int16Array(buf, off, n);
    this.supNames = header.super_class;
    this.sideNames = header.side;
    this.ntNames = header.nt;
    this.n = n;

    // positions Float32 [x,y,z] in µm → centre & scale
    const P = new Float32Array(n * 3);
    let cx = 0, cy = 0, cz = 0, cnt = 0;
    for (let i = 0; i < n; i++) {
      const f = Number.isFinite(this.pos[i * 3]);
      if (f) { cx += this.pos[i * 3]; cy += this.pos[i * 3 + 1]; cz += this.pos[i * 3 + 2]; cnt++; }
    }
    cx /= cnt; cy /= cnt; cz /= cnt;
    this.center = [cx, cy, cz];
    let maxr = 0;
    for (let i = 0; i < n; i++) {
      const x = (this.pos[i * 3] - cx) * UM2U, y = (this.pos[i * 3 + 1] - cy) * UM2U,
            z = (this.pos[i * 3 + 2] - cz) * UM2U;
      P[i * 3] = x; P[i * 3 + 1] = y; P[i * 3 + 2] = z;
      maxr = Math.max(maxr, Math.hypot(x, y, z));
    }
    this.posW = P;
    this.orbit.r = maxr * 1.9;
    this.orbit.cx = 0; this.orbit.cy = 0; this.orbit.cz = 0;

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(P, 3));
    this.colors = new Float32Array(n * 3);
    this._recolorAll();
    geo.setAttribute("color", new THREE.BufferAttribute(this.colors, 3));
    const mat = new THREE.PointsMaterial({
      size: 1.1, vertexColors: true, transparent: true, opacity: 0.85,
      sizeAttenuation: true, depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.cloud = new THREE.Points(geo, mat);
    this.scene.add(this.cloud);

    // activity overlay (points rebuilt per frame)
    this.actGeo = new THREE.BufferGeometry();
    this.actGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3), 3));
    this.actGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(3), 3));
    this.activePts = new THREE.Points(this.actGeo, new THREE.PointsMaterial({
      size: 3.2, vertexColors: true, transparent: true, opacity: 0.95,
      sizeAttenuation: true, depthWrite: false, blending: THREE.AdditiveBlending }));
    this.activePts.frustumCulled = false;
    this.scene.add(this.activePts);

    // selected-neuron marker + connection lines
    this.selMarker = new THREE.Mesh(
      new THREE.SphereGeometry(2.2, 14, 10),
      new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true }));
    this.selMarker.visible = false;
    this.scene.add(this.selMarker);
    this.connLines = null;

    this.loaded = true;
  }

  recolorBy(mode) {
    this.highlight = mode;
    this._recolorAll();
    this.cloud.geometry.attributes.color.needsUpdate = true;
  }

  _recolorAll() {
    const n = this.n;
    for (let i = 0; i < n; i++) {
      let hex;
      if (this.highlight === "nt") {
        const nm = this.ntNames[this.nt[i]];
        hex = { ACH: 0x67d6ff, GABA: 0xff6ba8, GLUT: 0xd597ff, SER: 0xffe08a,
                DA: 0x9dff5e, OCT: 0xffb08a, "": 0x666a71 }[nm] ?? 0x666a71;
      } else if (this.highlight === "side") {
        hex = { left: 0x6fc9ff, right: 0xffb02e, center: 0xd9ffe0, "": 0x777777 }[this.sideNames[this.side[i]]];
      } else {
        hex = PALETTE[this.supNames[this.sup[i]]] ?? 0x888888;
      }
      const c = new THREE.Color(hex);
      this.colors[i * 3] = c.r * 0.55; this.colors[i * 3 + 1] = c.g * 0.55; this.colors[i * 3 + 2] = c.b * 0.55;
    }
  }

  updateActivity(indices, values) {
    const k = Math.min(indices.length, 16000);
    const P = new Float32Array(k * 3), C = new Float32Array(k * 3);
    for (let j = 0; j < k; j++) {
      const i = indices[j];
      P[j * 3] = this.posW[i * 3]; P[j * 3 + 1] = this.posW[i * 3 + 1]; P[j * 3 + 2] = this.posW[i * 3 + 2];
      const v = Math.max(0, Math.min(1, values[j] ?? 0));
      C[j * 3] = 0.35 + v * 0.65; C[j * 3 + 1] = v * 0.25 + 0.9 * v * v; C[j * 3 + 2] = 0.08;
    }
    this.actGeo.setAttribute("position", new THREE.BufferAttribute(P, 3));
    this.actGeo.setAttribute("color", new THREE.BufferAttribute(C, 3));
  }

  selectNeuron(idx, info) {
    if (idx < 0) { this.selMarker.visible = false; return; }
    this.selMarker.position.set(this.posW[idx * 3], this.posW[idx * 3 + 1], this.posW[idx * 3 + 2]);
    this.selMarker.visible = true;
    // connection lines (outgoing cyan-ish, incoming magenta-ish)
    if (this.connLines) { this.scene.remove(this.connLines); this.connLines.geometry.dispose(); }
    const pts = [];
    const push = (nbs, color) => {
      for (const nb of (nbs || []).slice(0, 25)) {
        const j = nb.idx;
        pts.push(
          this.posW[idx * 3], this.posW[idx * 3 + 1], this.posW[idx * 3 + 2],
          this.posW[j * 3], this.posW[j * 3 + 1], this.posW[j * 3 + 2]);
      }
    };
    push(info?.out_n, 0); push(info?.in_n, 0);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pts), 3));
    this.connLines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({
      color: 0x6fd3ff, transparent: true, opacity: 0.5 }));
    this.scene.add(this.connLines);
  }

  // layered trace visual; pulses animated in render()
  showTrace(layers, seedIdx) {
    if (this.traceGroup) this.scene.remove(this.traceGroup);
    this.traceGroup = new THREE.Group();
    this.traceNodes = [];
    const geos = [];
    layers.forEach((layer, d) => {
      const col = new THREE.Color().setHSL(0.55 - d * 0.13, 0.9, 0.6);
      layer.slice(0, 120).forEach(([idx]) => {
        const m = new THREE.Mesh(
          new THREE.SphereGeometry(0.9, 8, 6),
          new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.0 }));
        m.position.set(this.posW[idx * 3], this.posW[idx * 3 + 1], this.posW[idx * 3 + 2]);
        this.traceGroup.add(m);
        this.traceNodes.push({ mesh: m, depth: d });
      });
    });
    this.scene.add(this.traceGroup);
    this.traceAnim = { t0: performance.now(), perLayerMs: 600 };
  }

  clearTrace() {
    if (this.traceGroup) { this.scene.remove(this.traceGroup); this.traceGroup = null; this.traceNodes = []; this.traceAnim = null; }
  }

  _bindPick() {
    const el = this.renderer.domElement;
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    el.addEventListener("click", e => {
      if (!this.loaded || !this.onPick) return;
      const r = el.getBoundingClientRect();
      this.mouse.x = ((e.clientX - r.left) / r.width) * 2 - 1;
      this.mouse.y = -((e.clientY - r.top) / r.height) * 2 + 1;
      this.raycaster.setFromCamera(this.mouse, this.camera);
      this.raycaster.params.Points.threshold = 1.4;
      const hits = this.raycaster.intersectObject(this.cloud);
      this.onPick(hits.length ? hits[0].index : -1);
    });
  }

  _bindOrbit() {
    const el = this.renderer.domElement;
    let drag = null;
    el.addEventListener("pointerdown", e => { drag = { x: e.clientX, y: e.clientY, moved: false }; el.setPointerCapture(e.pointerId); });
    el.addEventListener("pointerup", () => drag = null);
    el.addEventListener("pointermove", e => {
      if (!drag) return;
      const o = this.orbit;
      o.theta -= (e.clientX - drag.x) * 0.005;
      o.phi = Math.min(Math.PI - 0.1, Math.max(0.1, o.phi - (e.clientY - drag.y) * 0.005));
      drag = { x: e.clientX, y: e.clientY, moved: true };
    });
    el.addEventListener("wheel", e => {
      e.preventDefault();
      this.orbit.r = Math.min(900, Math.max(12, this.orbit.r * (1 + e.deltaY * 0.0011)));
    }, { passive: false });
  }

  render() {
    const o = this.orbit;
    const x = o.cx + o.r * Math.sin(o.phi) * Math.cos(o.theta);
    const z = o.cz + o.r * Math.sin(o.phi) * Math.sin(o.theta);
    const y = o.cy + o.r * Math.cos(o.phi);
    this.camera.position.set(x, y, z);
    this.camera.lookAt(o.cx, o.cy, o.cz);
    // trace animation
    if (this.traceAnim && this.traceNodes.length) {
      const t = performance.now() - this.traceAnim.t0;
      for (const tn of this.traceNodes) {
        const local = t - tn.depth * this.traceAnim.perLayerMs;
        const s = local < 0 ? 0 : Math.max(0.25, 1 - local / (this.traceAnim.perLayerMs * 1.6));
        tn.mesh.material.opacity = Math.min(0.95, s);
        tn.mesh.scale.setScalar(0.5 + s * 1.8 * Math.max(0, Math.sin(Math.min(Math.PI, local / 500))));
      }
      const total = (Math.max(...this.traceNodes.map(t => t.depth)) + 2) * this.traceAnim.perLayerMs;
      if (t > total + 4000) this.traceAnim = null;   // leave nodes lit, stop animating
    }
    this.renderer.render(this.scene, this.camera);
  }

  resize(w, h) {
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }
}
