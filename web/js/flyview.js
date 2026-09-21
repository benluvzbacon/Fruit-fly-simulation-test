// Normal view — the embodied fly in its 3D world.
import * as THREE from "three";

const M2U = 0.2;             // mm → world units (600 mm arena → 120 u)

export class FlyView {
  constructor(container) {
    this.container = container;
    const w = container.clientWidth || 800, h = container.clientHeight || 600;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0d11);
    this.scene.fog = new THREE.Fog(0x0b0d11, 220, 520);
    this.camera = new THREE.PerspectiveCamera(55, w / h, 0.05, 2000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    container.appendChild(this.renderer.domElement);

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    this.sun = new THREE.DirectionalLight(0xfff6dd, 1.1);
    this.sun.position.set(80, 140, 60);
    this.scene.add(this.sun);

    // orbit
    this.orbit = { theta: -2.2, phi: 1.05, r: 95, cx: 60, cz: 60, cy: 0, follow: true };
    this._bindOrbit();

    this.worldGroup = new THREE.Group();
    this.scene.add(this.worldGroup);
    this.fly = this._buildFly();
    this.scene.add(this.fly.group);
    this._arenaBuilt = false;
  }

  // ---------------------------------------------------------------- world
  buildWorld(world) {
    // (re)create environment meshes from the simulator's serialised world
    while (this.worldGroup.children.length) this.worldGroup.remove(this.worldGroup.children[0]);
    if (this._grid) this.scene.remove(this._grid);
    const B = world.bounds;
    const W = (B.x[1] - B.x[0]) * M2U, D = (B.y[1] - B.y[0]) * M2U, H = B.z[1] * M2U;

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(W, D),
      new THREE.MeshPhongMaterial({ color: 0x151a20 }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(B.x[0] * M2U + W / 2, 0, B.y[0] * M2U + D / 2);
    this.worldGroup.add(floor);
    this._grid = new THREE.GridHelper(W, 30, 0x2a3242, 0x1c2330);
    this._grid.position.set(W / 2, 0.02, D / 2);
    this.worldGroup.add(this._grid);

    const wallMat = new THREE.MeshPhongMaterial({
      color: 0x1e2733, transparent: true, opacity: 0.16, side: THREE.DoubleSide });
    const mkWall = (w, h, x, y, z, ry) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), wallMat);
      m.position.set(x, h / 2, z); m.rotation.y = ry;
      this.worldGroup.add(m);
    };
    mkWall(W, H, W / 2, 0, 0, 0); mkWall(W, H, W / 2, 0, D, 0);
    mkWall(D, H, 0, 0, D / 2, Math.PI / 2); mkWall(D, H, W, 0, D / 2, Math.PI / 2);

    // obstacles
    const obsGeo = new THREE.CylinderGeometry(1, 1, 1, 24);
    const obsMat = new THREE.MeshPhongMaterial({ color: 0x39434f, flatShading: true });
    this.obstacleMeshes = world.obstacles.map(([cx, cy, r, h]) => {
      const m = new THREE.Mesh(obsGeo, obsMat);
      m.scale.set(r * M2U, h * M2U, r * M2U);
      m.position.set(cx * M2U, h * M2U / 2, cy * M2U);
      this.worldGroup.add(m);
      return m;
    });

    // light marker
    const L = world.light;
    const sunBall = new THREE.Mesh(
      new THREE.SphereGeometry(4.5, 20, 20),
      new THREE.MeshBasicMaterial({ color: L.on ? 0xffe58a : 0x4a4633 }));
    sunBall.position.set(L.x * M2U, L.z * M2U, L.y * M2U);
    this.worldGroup.add(sunBall);
    this.sunMesh = sunBall;

    // heat zone
    const hz = new THREE.Mesh(
      new THREE.CircleGeometry(world.heat.r * M2U, 40),
      new THREE.MeshBasicMaterial({ color: 0xff5a3c, transparent: true, opacity: 0.13, side: THREE.DoubleSide }));
    hz.rotation.x = -Math.PI / 2;
    hz.position.set(world.heat.x * M2U, 0.05, world.heat.y * M2U);
    this.worldGroup.add(hz);

    this.sourceMeshes = [];
    this._arenaBuilt = true;
  }

  syncSources(world) {
    // remove stale
    const want = world.sources.length;
    while (this.sourceMeshes.length > want) {
      const s = this.sourceMeshes.pop();
      this.worldGroup.remove(s);
    }
    world.sources.forEach((s, i) => {
      let m = this.sourceMeshes[i];
      const color = s.kind === "food" ? 0x59ffa3 : 0xff5b4d;
      if (!m) {
        m = new THREE.Group();
        const blob = new THREE.Mesh(
          new THREE.SphereGeometry(2.4, 16, 16),
          new THREE.MeshPhongMaterial({ color, shininess: 90 }));
        blob.position.y = 1.2;
        const halo = new THREE.Mesh(
          new THREE.CircleGeometry(s.r * M2U, 40),
          new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.07, side: THREE.DoubleSide }));
        halo.rotation.x = -Math.PI / 2; halo.position.y = 0.04;
        m.add(blob); m.add(halo);
        this.worldGroup.add(m);
        this.sourceMeshes.push(m);
      }
      m.position.set(s.x * M2U, 0, s.y * M2U);
      m.children[0].material.color.setHex(color);
      m.children[0].scale.setScalar(Math.max(0.25, s.amount));
    });
  }

  // ---------------------------------------------------------------- fly
  _buildFly() {
    const g = new THREE.Group();
    const L = 5.0;                            // fly length in world units (~2.5 mm)
    const mat = new THREE.MeshPhongMaterial({ color: 0x7a5b3a, shininess: 24 });
    const darkMat = new THREE.MeshPhongMaterial({ color: 0x4a3826, shininess: 18 });

    const abdomen = new THREE.Mesh(new THREE.SphereGeometry(1.0, 20, 16), mat);
    abdomen.scale.set(L * 0.34, L * 0.20, L * 0.24);
    abdomen.position.x = -L * 0.36;
    const thorax = new THREE.Mesh(new THREE.SphereGeometry(1.0, 20, 16), darkMat);
    thorax.scale.set(L * 0.24, L * 0.20, L * 0.22);
    const head = new THREE.Group();

    const skull = new THREE.Mesh(new THREE.SphereGeometry(L * 0.15, 18, 14), mat);
    head.add(skull);
    const eyeMat = new THREE.MeshPhongMaterial({ color: 0xa11719, shininess: 200 });
    for (const s of [-1, 1]) {
      const eye = new THREE.Mesh(new THREE.SphereGeometry(L * 0.10, 14, 12), eyeMat);
      eye.scale.set(0.8, 1.15, 0.9);
      eye.position.set(L * 0.04, s * L * 0.085, 0);
      head.add(eye);
    }
    // antennae
    const antMat = new THREE.MeshPhongMaterial({ color: 0x5d4730 });
    for (const s of [-1, 1]) {
      const a = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.05, L * 0.3, 5), antMat);
      a.position.set(L * 0.15, s * L * 0.05, L * 0.06);
      a.rotation.z = -1.0; a.rotation.y = s * 0.35;
      head.add(a);
    }
    head.position.x = L * 0.30;
    g.add(abdomen, thorax, head);

    // legs — 6, two-joint, tripod-gaited
    const legMat = new THREE.MeshPhongMaterial({ color: 0x3c2f1f });
    const legs = [];
    const mounts = [L * 0.22, L * 0.08, -L * 0.06];       // x along body
    for (const s of [-1, 1]) {
      for (let i = 0; i < 3; i++) {
        const hip = new THREE.Group();
        hip.position.set(mounts[i], s * L * 0.16, -L * 0.05);
        const up = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.06, L * 0.30, 6), legMat);
        up.position.y = -L * 0.15; up.position.z = s * L * 0.10;
        up.rotation.x = Math.PI / 2; up.rotation.y = 0;
        up.rotation.x = 0; up.rotation.z = s * 0.9;
        hip.add(up);
        // lower segment hangs further out+down
        const knee = new THREE.Group();
        knee.position.set(0, -L * 0.30, 0);
        const low = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.03, L * 0.42, 6), legMat);
        low.position.y = -L * 0.21;
        knee.add(low);
        hip.add(knee);
        g.add(hip);
        legs.push({ hip, knee, side: s, i, phase: (i % 2 === 0 ? 0 : Math.PI) + (s > 0 ? Math.PI : 0) });
      }
    }

    // wings
    const wingMat = new THREE.MeshPhongMaterial({ color: 0xcfe6ef, transparent: true,
      opacity: 0.4, side: THREE.DoubleSide });
    const wings = [];
    for (const s of [-1, 1]) {
      const w = new THREE.Mesh(new THREE.PlaneGeometry(L * 0.55, L * 0.16), wingMat);
      w.geometry.translate(-L * 0.28, 0, 0);
      const pivot = new THREE.Group();
      pivot.add(w);
      pivot.position.set(L * 0.12, s * L * 0.05, L * 0.16);
      pivot.rotation.y = s * 0.5;
      pivot.rotation.z = -0.15;
      g.add(pivot);
      wings.push({ pivot, side: s });
    }

    // proboscis (feeding)
    const prob = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.02, L * 0.14, 6),
      new THREE.MeshPhongMaterial({ color: 0x54401f }));
    prob.geometry.translate(0, -L * 0.07, 0);
    prob.position.set(L * 0.42, 0, -L * 0.02);
    prob.rotation.z = 2.4;
    g.add(prob);

    return { group: g, abdomen, thorax, head, legs, wings, prob, L };
  }

  // ---------------------------------------------------------------- frame
  update(frame) {
    const b = frame.body;
    const [px, py, pz] = b.pos;
    const g = this.fly.group;
    g.position.set(px * M2U, pz * M2U + 0.8, py * M2U);
    g.rotation.set(0, -b.yaw, b.pitch || 0, "YXZ");

    // leg gait (tripod)
    for (const leg of this.fly.legs) {
      const ph = b.walk_phase + leg.phase;
      leg.hip.rotation.z = leg.side * (0.9 + 0.35 * Math.sin(ph));
      leg.hip.rotation.y = 0.45 * Math.cos(ph);
      leg.knee.rotation.z = leg.side * (0.35 + 0.3 * Math.sin(ph + 1.2));
    }
    // wings
    for (const w of this.fly.wings) {
      if (b.flying) {
        const flap = Math.sin(b.wing_phase * 8.0);
        w.pivot.rotation.z = w.side * 0.3 + w.side * flap * 0.9;
        w.pivot.rotation.y = w.side * (0.5 - 0.3 * Math.sin(b.wing_phase * 8.0));
      } else {
        w.pivot.rotation.z = w.side * -0.08;
        w.pivot.rotation.y = w.side * 0.5;
      }
    }
    this.fly.prob.rotation.z = 2.4 - (b.feed || 0) * 1.8;

    // camera follow
    const o = this.orbit;
    if (o.follow) { o.cx += (g.position.x - o.cx) * 0.08; o.cz += (g.position.z - o.cz) * 0.08; o.cy += (g.position.y - o.cy) * 0.08; }
    this.sunMesh.material.color.setHex(frame.world.light.on ? 0xffe58a : 0x4a4633);
    this.syncSources(frame.world);
  }

  _bindOrbit() {
    const el = this.renderer.domElement;
    let drag = null, pinch = 0;
    el.addEventListener("pointerdown", e => { drag = { x: e.clientX, y: e.clientY }; el.setPointerCapture(e.pointerId); });
    el.addEventListener("pointerup", () => drag = null);
    el.addEventListener("pointermove", e => {
      if (!drag) return;
      const o = this.orbit;
      o.theta -= (e.clientX - drag.x) * 0.005;
      o.phi = Math.min(Math.PI - 0.15, Math.max(0.15, o.phi - (e.clientY - drag.y) * 0.005));
      drag = { x: e.clientX, y: e.clientY };
    });
    el.addEventListener("wheel", e => {
      e.preventDefault();
      this.orbit.r = Math.min(320, Math.max(8, this.orbit.r * (1 + e.deltaY * 0.001)));
    }, { passive: false });
    el.addEventListener("touchmove", e => {
      if (e.touches.length === 2) {
        const d = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY);
        if (pinch) this.orbit.r = Math.min(320, Math.max(8, this.orbit.r * pinch / d));
        pinch = d;
      }
    });
    el.addEventListener("dblclick", () => { this.orbit.follow = !this.orbit.follow; });
  }

  render() {
    const o = this.orbit;
    const x = o.cx + o.r * Math.sin(o.phi) * Math.cos(o.theta);
    const z = o.cz + o.r * Math.sin(o.phi) * Math.sin(o.theta);
    const y = o.cy + o.r * Math.cos(o.phi);
    this.camera.position.set(x, y, z);
    this.camera.lookAt(o.cx, o.cy, o.cz);
    this.renderer.render(this.scene, this.camera);
  }

  resize(w, h) {
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }
}
