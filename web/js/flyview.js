// Normal view — the embodied fly in its 3D world.
//
// Coordinate conventions
//   simulator : x/y = ground plane (mm), z = height (mm), yaw = CCW around +z
//   three.js  : X = sim x, Y = up, Z = sim y     →  rotation.y = -yaw
//   fly model : built UPRIGHT — forward +X, up +Y, lateral ±Z, origin on the
//               thorax midline.  (The first version was built Z-up and never
//               converted, so the fly rendered lying on its side.)
import * as THREE from "three";

const M2U = 0.2;             // mm → world units (600 mm arena → 120 u)
const TAU = Math.PI * 2;
const lerpAngle = (a, b, t) => {
  let d = (b - a) % TAU;
  if (d > Math.PI) d -= TAU; else if (d < -Math.PI) d += TAU;
  return a + d * t;
};

export class FlyView {
  constructor(container) {
    this.container = container;
    const w = container.clientWidth || 800, h = container.clientHeight || 600;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x10141b);
    this.scene.fog = new THREE.Fog(0x10141b, 260, 640);
    this.camera = new THREE.PerspectiveCamera(55, w / h, 0.05, 2000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
    this.renderer.setSize(w, h);
    container.appendChild(this.renderer.domElement);

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    this.sun = new THREE.DirectionalLight(0xfff3d8, 1.15);
    this.sun.position.set(80, 140, 60);
    this.scene.add(this.sun);
    this.fill = new THREE.DirectionalLight(0x9db8ff, 0.25);
    this.fill.position.set(-60, 40, -80);
    this.scene.add(this.fill);

    // orbit
    this.orbit = { theta: -2.35, phi: 1.12, r: 60, cx: 60, cz: 60, cy: 2, follow: true };
    this._bindOrbit();

    this.worldGroup = new THREE.Group();
    this.scene.add(this.worldGroup);
    this.fly = this._buildFly();
    this.scene.add(this.fly.root);
    // soft blob shadow grounding the fly (cheap — no real shadow maps)
    this.blobShadow = new THREE.Mesh(
      new THREE.CircleGeometry(3.4, 24),
      new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.28, depthWrite: false }));
    this.blobShadow.rotation.x = -Math.PI / 2;
    this.scene.add(this.blobShadow);
    this._arenaBuilt = false;

    // interpolated presentation state (polls arrive ~5×/s; we render at 60 fps)
    this.cur = { pos: new THREE.Vector3(60, this.fly.bodyY, 60), yaw: 0, pitch: 0, bank: 0 };
    this.tgt = { pos: this.cur.pos.clone(), yaw: 0, pitch: 0 };
    this.speed = 0; this.flying = false; this.feed = 0;
    this.gait = 0; this.gaitTgt = 0;          // unwrapped continuous gait phase
    this.wing = 0; this.wingTgt = 0;
    this.flyLevel = 0;
    this._tLast = 0;
    this._clock = 0;
  }

  // procedural substrate texture: mottled soil + leaf-litter speckle
  _floorTexture() {
    const c = document.createElement("canvas");
    c.width = c.height = 256;
    const g = c.getContext("2d");
    g.fillStyle = "#232a22";
    g.fillRect(0, 0, 256, 256);
    for (let i = 0; i < 2600; i++) {
      const x = Math.random() * 256, y = Math.random() * 256, r = Math.random() * 2.6 + 0.4;
      const t = Math.random();
      g.fillStyle = t < 0.55 ? "rgba(46,58,40,0.5)" : t < 0.8 ? "rgba(30,36,28,0.6)" : "rgba(78,66,42,0.4)";
      g.beginPath(); g.arc(x, y, r, 0, 7); g.fill();
    }
    for (let i = 0; i < 220; i++) {       // fine grit
      g.fillStyle = Math.random() < 0.5 ? "rgba(120,110,70,0.25)" : "rgba(12,16,12,0.35)";
      g.fillRect(Math.random() * 256, Math.random() * 256, 1.2, 1.2);
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(7, 7);
    return tex;
  }

  // ---------------------------------------------------------------- world
  buildWorld(world) {
    while (this.worldGroup.children.length) this.worldGroup.remove(this.worldGroup.children[0]);
    if (this._grid) this.scene.remove(this._grid);
    const B = world.bounds;
    const W = (B.x[1] - B.x[0]) * M2U, D = (B.y[1] - B.y[0]) * M2U, H = B.z[1] * M2U;
    this._W = W; this._D = D;

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(W, D),
      new THREE.MeshPhongMaterial({ map: this._floorTexture(), color: 0xbfc7b4, shininess: 4 }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(B.x[0] * M2U + W / 2, 0, B.y[0] * M2U + D / 2);
    this.worldGroup.add(floor);
    this._grid = new THREE.GridHelper(W, 24, 0x39463a, 0x27332a);
    this._grid.position.set(W / 2, 0.02, D / 2);
    this._grid.material.transparent = true; this._grid.material.opacity = 0.35;
    this.worldGroup.add(this._grid);

    const wallMat = new THREE.MeshPhongMaterial({
      color: 0x243040, transparent: true, opacity: 0.15, side: THREE.DoubleSide });
    const mkWall = (w, h, x, z, ry) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), wallMat);
      m.position.set(x, h / 2, z); m.rotation.y = ry;
      this.worldGroup.add(m);
    };
    mkWall(W, H, W / 2, 0, 0); mkWall(W, H, W / 2, D, 0);
    mkWall(D, H, 0, D / 2, Math.PI / 2); mkWall(D, H, W, D / 2, Math.PI / 2);

    // obstacles
    const obsGeo = new THREE.CylinderGeometry(1, 1, 1, 24);
    const obsMat = new THREE.MeshPhongMaterial({ color: 0x4c5347, flatShading: true });
    this.obstacleMeshes = world.obstacles.map(([cx, cy, r, h]) => {
      const m = new THREE.Mesh(obsGeo, obsMat);
      m.scale.set(r * M2U, h * M2U, r * M2U);
      m.position.set(cx * M2U, h * M2U / 2, cy * M2U);
      this.worldGroup.add(m);
      return m;
    });

    // light marker + volumetric-ish glow cone down to the floor
    const L = world.light;
    const sunBall = new THREE.Mesh(
      new THREE.SphereGeometry(4.5, 20, 20),
      new THREE.MeshBasicMaterial({ color: L.on ? 0xffe58a : 0x4a4633 }));
    sunBall.position.set(L.x * M2U, L.z * M2U, L.y * M2U);
    this.worldGroup.add(sunBall);
    this.sunMesh = sunBall;
    const coneH = L.z * M2U;
    this.lampCone = new THREE.Mesh(
      new THREE.ConeGeometry(coneH * 0.55, coneH, 32, 1, true),
      new THREE.MeshBasicMaterial({ color: 0xfff0b8, transparent: true,
        opacity: 0.07, side: THREE.DoubleSide, depthWrite: false,
        blending: THREE.AdditiveBlending }));
    this.lampCone.rotation.x = Math.PI;          // apex up at the lamp
    this.lampCone.position.set(L.x * M2U, coneH / 2, L.y * M2U);
    this.worldGroup.add(this.lampCone);
    this._scatterDetails(W, D, world);

    // heat zone
    const hz = new THREE.Mesh(
      new THREE.CircleGeometry(world.heat.r * M2U, 40),
      new THREE.MeshBasicMaterial({ color: 0xff5a3c, transparent: true, opacity: 0.13, side: THREE.DoubleSide }));
    hz.rotation.x = -Math.PI / 2;
    hz.position.set(world.heat.x * M2U, 0.05, world.heat.y * M2U);
    this.worldGroup.add(hz);

    this.sourceMeshes = [];
    this._arenaBuilt = true;
    this.orbit.cx = this.cur.pos.x; this.orbit.cz = this.cur.pos.z;
  }

  // cheap dressing: pebbles, grass tufts, fallen leaves — static instanced/
  // small meshes so it costs almost nothing per frame
  _scatterDetails(W, D, world) {
    const rng = (a, b) => a + Math.random() * (b - a);
    const away = (x, z, r) => world.obstacles.some(([cx, cy, or]) =>
      Math.hypot(x - cx * M2U, z - cy * M2U) < (or * M2U + r));

    // pebbles
    const pebGeo = new THREE.IcosahedronGeometry(1, 0);
    const pebMat = new THREE.MeshPhongMaterial({ color: 0x6b6f66, flatShading: true });
    const peb = new THREE.InstancedMesh(pebGeo, pebMat, 42);
    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), sc = new THREE.Vector3();
    let n = 0;
    for (let i = 0; i < 120 && n < 42; i++) {
      const x = rng(4, W - 4), z = rng(4, D - 4);
      if (away(x, z, 3)) continue;
      const s = rng(0.35, 1.5);
      q.setFromEuler(new THREE.Euler(rng(0, 3), rng(0, 3), rng(0, 3)));
      m4.compose(new THREE.Vector3(x, s * 0.35, z), q, sc.set(s, s * rng(0.4, 0.7), s));
      peb.setMatrixAt(n++, m4);
    }
    peb.count = n; peb.instanceMatrix.needsUpdate = true;
    this.worldGroup.add(peb);

    // grass tufts — two crossed blades each
    const bladeMat = new THREE.MeshPhongMaterial({ color: 0x3f6b35,
      side: THREE.DoubleSide, transparent: true, opacity: 0.9 });
    const bladeGeo = new THREE.PlaneGeometry(0.9, 2.6);
    bladeGeo.translate(0, 1.3, 0);
    for (let i = 0; i < 26; i++) {
      const x = rng(5, W - 5), z = rng(5, D - 5);
      if (away(x, z, 2.5)) continue;
      const tuft = new THREE.Group();
      for (let k = 0; k < 3; k++) {
        const b = new THREE.Mesh(bladeGeo, bladeMat);
        b.rotation.y = k * Math.PI / 3 + rng(0, 0.6);
        b.rotation.z = rng(-0.22, 0.22);
        b.scale.setScalar(rng(0.7, 1.5));
        tuft.add(b);
      }
      tuft.position.set(x, 0, z);
      this.worldGroup.add(tuft);
    }

    // fallen leaves
    const leafMat = new THREE.MeshPhongMaterial({ color: 0x5d4a26,
      side: THREE.DoubleSide, shininess: 8 });
    for (let i = 0; i < 5; i++) {
      const leaf = new THREE.Mesh(new THREE.CircleGeometry(rng(2.2, 4.2), 9), leafMat);
      leaf.rotation.x = -Math.PI / 2 + rng(-0.12, 0.12);
      leaf.rotation.z = rng(0, 6.3);
      leaf.scale.set(1, rng(0.55, 0.75), 1);
      const x = rng(8, W - 8), z = rng(8, D - 8);
      if (away(x, z, 3)) continue;
      leaf.position.set(x, 0.06, z);
      this.worldGroup.add(leaf);
    }
  }

  syncSources(world) {
    const want = world.sources.length;
    while (this.sourceMeshes.length > want) {
      const s = this.sourceMeshes.pop();
      this.worldGroup.remove(s);
    }
    world.sources.forEach((s, i) => {
      let m = this.sourceMeshes[i];
      const color = s.kind === "food" ? 0xffab4d : 0xa8352b;
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
  // Upright fruit fly ~2.5 mm (5 world units).  All proportions / colours are
  // presentation-only (the REAL data lives in the brain, not the mesh).
  _buildFly() {
    const L = 5.0;
    const root = new THREE.Group();            // sim-driven transform (pos/yaw)
    const model = new THREE.Group();           // pitch/bank/bob live here
    root.add(model);

    const cuticle = new THREE.MeshPhongMaterial({ color: 0x8a6a42, shininess: 30 });
    const dark = new THREE.MeshPhongMaterial({ color: 0x4f3b24, shininess: 20 });
    const darker = new THREE.MeshPhongMaterial({ color: 0x2e2416, shininess: 14 });
    const legMat = new THREE.MeshPhongMaterial({ color: 0x3a2c1c, shininess: 10 });

    // -- abdomen: tapered with dark tergite bands -------------------------
    const abdomen = new THREE.Group();
    const ab = new THREE.Mesh(new THREE.SphereGeometry(1, 22, 18), cuticle);
    ab.scale.set(L * 0.36, L * 0.185, L * 0.225);
    abdomen.add(ab);
    for (let i = 0; i < 3; i++) {
      const t = (i + 1) / 4;
      const r = Math.sqrt(Math.max(0.2, 1 - (t - 0.25) * (t - 0.25) * 1.2));
      const band = new THREE.Mesh(new THREE.TorusGeometry(1, 0.045, 8, 28), darker);
      band.scale.set(r * L * 0.225, r * L * 0.185, r * L * 0.225);  // rot.y π/2 maps X→Z
      band.rotation.y = Math.PI / 2;
      band.position.x = -L * 0.36 * (t * 1.5 - 0.35);
      abdomen.add(band);
    }
    abdomen.position.set(-L * 0.33, L * 0.01, 0);
    model.add(abdomen);

    // -- thorax (mesonotum + scutellum) -----------------------------------
    const thorax = new THREE.Mesh(new THREE.SphereGeometry(1, 22, 18), dark);
    thorax.scale.set(L * 0.245, L * 0.20, L * 0.215);
    thorax.position.set(L * 0.02, L * 0.03, 0);
    model.add(thorax);
    const hump = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), darker);
    hump.scale.set(L * 0.16, L * 0.10, L * 0.15);
    hump.position.set(L * 0.05, L * 0.185, 0);
    model.add(hump);

    // -- head: skull, compound eyes, ocelli, antennae ----------------------
    const head = new THREE.Group();
    const skull = new THREE.Mesh(new THREE.SphereGeometry(L * 0.145, 20, 16), cuticle);
    head.add(skull);
    const eyeMat = new THREE.MeshPhongMaterial({ color: 0xb01f1c, shininess: 220,
      emissive: 0x330404 });
    for (const s of [-1, 1]) {
      const eye = new THREE.Mesh(new THREE.SphereGeometry(L * 0.105, 16, 14), eyeMat);
      eye.scale.set(0.8, 1.05, 1.15);
      eye.position.set(L * 0.035, L * 0.02, s * L * 0.095);
      head.add(eye);
    }
    for (const o of [[0.02, 0.135, 0], [0.05, 0.125, 0.03], [0.05, 0.125, -0.03]]) {
      const oc = new THREE.Mesh(new THREE.SphereGeometry(L * 0.014, 8, 6),
        new THREE.MeshBasicMaterial({ color: 0x1c0f08 }));
      oc.position.set(o[0] * L, o[1] * L, o[2] * L);
      head.add(oc);
    }
    const antMat = new THREE.MeshPhongMaterial({ color: 0x6a5030 });
    const antennae = [];
    for (const s of [-1, 1]) {
      const base = new THREE.Group();
      base.position.set(L * 0.125, -L * 0.01, s * L * 0.045);
      const seg1 = new THREE.Mesh(new THREE.CylinderGeometry(L * 0.016, L * 0.012, L * 0.09, 5), antMat);
      seg1.position.set(L * 0.04, L * 0.018, 0);
      seg1.rotation.z = -1.1;
      base.add(seg1);
      const seg2 = new THREE.Mesh(new THREE.CylinderGeometry(L * 0.011, L * 0.008, L * 0.10, 5), antMat);
      seg2.position.set(L * 0.10, L * 0.055, 0);
      seg2.rotation.z = -0.5;
      base.add(seg2);
      const funic = new THREE.Mesh(new THREE.SphereGeometry(L * 0.02, 8, 6), dark);
      funic.position.set(L * 0.135, L * 0.085, 0);
      base.add(funic);
      const arista = new THREE.Mesh(new THREE.CylinderGeometry(L * 0.004, L * 0.001, L * 0.14, 4), antMat);
      arista.position.set(L * 0.16, L * 0.14, 0);
      arista.rotation.z = -0.25;
      base.add(arista);
      base.rotation.x = s * 0.3;
      head.add(base);
      antennae.push({ base, side: s });
    }
    head.position.set(L * 0.30, -L * 0.02, 0);
    model.add(head);

    // -- proboscis (extends when feeding) ----------------------------------
    const prob = new THREE.Mesh(new THREE.CylinderGeometry(L * 0.012, L * 0.004, L * 0.17, 6),
      new THREE.MeshPhongMaterial({ color: 0x59431f }));
    prob.geometry.translate(0, -L * 0.085, 0);
    prob.position.set(L * 0.40, -L * 0.075, 0);
    prob.rotation.z = 2.35;                    // tucked under the head
    model.add(prob);

    // -- legs: 3 segments (coxa/femur/tibia), tripod gait -------------------
    // hip rotates about Y (sweep) and X (lift); knee folds the tibia in.
    const legs = [];
    const FEMUR = L * 0.26, TIBIA = L * 0.34;
    const SPREAD = 0.62, FOLD = 0.78;
    const mounts = [L * 0.17, L * 0.02, -L * 0.13];
    for (const s of [-1, 1]) {
      for (let i = 0; i < 3; i++) {
        const hip = new THREE.Group();
        hip.position.set(mounts[i], -L * 0.075, s * L * 0.15);
        const femurG = new THREE.Group();
        const femur = new THREE.Mesh(new THREE.CylinderGeometry(L * 0.018, L * 0.013, FEMUR, 6), legMat);
        femur.position.y = -FEMUR / 2;
        femurG.add(femur);
        hip.add(femurG);
        const knee = new THREE.Group();
        knee.position.y = -FEMUR;
        const tibia = new THREE.Mesh(new THREE.CylinderGeometry(L * 0.010, L * 0.006, TIBIA, 6), legMat);
        tibia.position.y = -TIBIA / 2;
        knee.add(tibia);
        const tarsus = new THREE.Mesh(new THREE.SphereGeometry(L * 0.016, 8, 6), darker);
        tarsus.position.y = -TIBIA;
        tarsus.scale.set(1.4, 0.5, 1.0);
        knee.add(tarsus);
        femurG.add(knee);
        model.add(hip);
        // tripod gait phasing: (front+rear same side) opposite to (mid same side),
        // mirrored across sides
        legs.push({ hip, femurG, knee, side: s, row: i,
          phase: (i % 2 === 0 ? 0 : Math.PI) + (s > 0 ? Math.PI : 0),
          spread: SPREAD, fold: FOLD });
      }
    }
    // hip base orientation: femur tilted outward, tibia folded back in
    for (const leg of legs) {
      leg.femurG.rotation.x = -leg.side * leg.spread;
      leg.knee.rotation.x = leg.side * leg.fold;
    }

    // -- wings: two membranes; folded over the abdomen at rest ---------------
    const wingMat = new THREE.MeshPhongMaterial({ color: 0xd8ecf4, transparent: true,
      opacity: 0.42, side: THREE.DoubleSide, shininess: 140 });
    const wings = [];
    // one canonical wing outline (extends backward -X, width +Y); the shape
    // lies in the XY plane → mesh.rotation.x = +π/2 lays it flat with width
    // toward +Z; mesh.scale.y = s mirrors the right wing.
    const shape = new THREE.Shape();
    shape.moveTo(0, 0);
    shape.quadraticCurveTo(-L * 0.18, L * 0.10, -L * 0.50, L * 0.085);
    shape.quadraticCurveTo(-L * 0.66, L * 0.05, -L * 0.54, 0.001);
    shape.quadraticCurveTo(-L * 0.30, -L * 0.05, 0, 0);
    const wingGeo = new THREE.ShapeGeometry(shape, 10);
    for (const s of [-1, 1]) {
      const w = new THREE.Mesh(wingGeo, wingMat);
      w.rotation.x = Math.PI / 2;
      w.scale.y = s;
      const pivot = new THREE.Group();
      pivot.add(w);
      pivot.position.set(L * 0.11, L * 0.185, s * L * 0.045);
      model.add(pivot);
      wings.push({ pivot, side: s });
    }

    const bodyY = L * 0.60;                    // origin height above ground
    return { root, model, abdomen, thorax, head, antennae, prob, legs, wings,
             L, bodyY, FOLD, SPREAD };
  }

  // ---------------------------------------------------------------- frame
  // called by the poller with the newest server frame (targets only — the
  // visual interpolation happens in render())
  update(frame) {
    const b = frame.body;
    const [px, py, pz] = b.pos;
    this.tgt.pos.set(px * M2U, pz * M2U + this.fly.bodyY, py * M2U);
    this.tgt.yaw = b.yaw;
    this.tgt.pitch = b.pitch || 0;
    this.speed = b.speed;
    this.flying = b.flying;
    this.feed = b.feed || 0;
    this.flyLevel += ((b.flying ? 1 : 0) - this.flyLevel) * 0.5;
    // unwrap phases onto the continuous counters
    const unwrap = (cur, t) => cur + ((t - cur) % TAU + 1.5 * TAU) % TAU - Math.PI;
    this.gaitTgt = unwrap(this.gait, b.walk_phase);
    this.wingTgt = unwrap(this.wing, b.wing_phase);
    this.firstFrameSeen = true;
    this.sunMesh.material.color.setHex(frame.world.light.on ? 0xffe58a : 0x4a4633);
    if (this.lampCone) this.lampCone.material.opacity = frame.world.light.on ? 0.07 : 0.0;
    this.syncSources(frame.world);
  }

  _pose(dt) {
    const f = this.fly;
    const gaitAmp = 0.16 + 0.84 * Math.min(1, this.speed / 20);   // idle shuffle → full stride
    for (const leg of f.legs) {
      const p = this.gait + leg.phase;
      const sweep = 0.62 * gaitAmp * Math.sin(p);
      const liftRaw = Math.max(0, Math.sin(p + 1.1));
      const lift = liftRaw * liftRaw * (0.28 + 0.72 * gaitAmp);
      leg.hip.rotation.y = sweep;
      leg.femurG.rotation.x = -leg.side * (leg.spread - lift * 0.42);
      leg.knee.rotation.x = leg.side * (leg.fold + lift * 0.55);
    }
    // wings — folded flat over the back on the ground, out + beating in flight
    const beat = Math.sin(this.wing);
    const out = 0.16 + this.flyLevel * 0.95;
    for (const w of f.wings) {
      w.pivot.rotation.y = -w.side * out * 0.55;
      w.pivot.rotation.x = -w.side * (0.06 + this.flyLevel * (0.55 * beat + 0.25));
      w.pivot.rotation.z = 0;
    }
    // proboscis
    f.prob.rotation.z = 2.35 - this.feed * 1.9;
    // idle head saccades + antenna sway
    const t = this._clock;
    f.head.rotation.y = 0.10 * Math.sin(t * 0.7) * Math.sin(t * 1.7 + 2.0);
    f.head.rotation.x = 0.05 * Math.sin(t * 1.1 + 0.7);
    for (const a of f.antennae) {
      a.base.rotation.x = a.side * (0.30 + 0.05 * Math.sin(t * 2.3 + a.side));
      a.base.rotation.y = a.side * 0.06 * Math.sin(t * 1.9);
    }
    // body bob (2 steps per stride) + gentle breathing at idle
    f.model.position.y =
      0.020 * f.L * gaitAmp * Math.sin(this.gait * 2 + 0.4) +
      0.008 * f.L * (1 - gaitAmp) * Math.sin(t * 2.2);
    f.model.rotation.z = -this.cur.pitch + 0.02 * gaitAmp * Math.sin(this.gait * 2);
    f.model.rotation.x = this.cur.bank;
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
      this.orbit.r = Math.min(320, Math.max(6, this.orbit.r * (1 + e.deltaY * 0.001)));
    }, { passive: false });
    el.addEventListener("touchmove", e => {
      if (e.touches.length === 2) {
        const d = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY);
        if (pinch) this.orbit.r = Math.min(320, Math.max(6, this.orbit.r * pinch / d));
        pinch = d;
      }
    });
    el.addEventListener("dblclick", () => { this.orbit.follow = !this.orbit.follow; });
  }

  render() {
    const now = performance.now();
    const dt = Math.min(0.1, (now - (this._tLast || now)) / 1000);
    this._tLast = now;
    this._clock += dt;

    // ease current pose toward the latest server targets
    const kPos = 1 - Math.exp(-dt * 7), kYaw = 1 - Math.exp(-dt * 9);
    this.cur.pos.lerp(this.tgt.pos, kPos);
    const yawPrev = this.cur.yaw;
    this.cur.yaw = lerpAngle(this.cur.yaw, this.tgt.yaw, kYaw);
    this.cur.pitch += (this.tgt.pitch - this.cur.pitch) * kPos;
    const yawRate = dt > 0 ? (this.cur.yaw - yawPrev) / dt : 0;
    const bankTgt = THREE.MathUtils.clamp(-yawRate * 0.18, -0.45, 0.45);
    this.cur.bank += (bankTgt - this.cur.bank) * (1 - Math.exp(-dt * 5));

    // advance animation phase counters at the sim's own rates
    const walkW = (2.0 + 22.0 * Math.min(1, Math.abs(this.speed) / 28)) ;
    this.gait = this.gaitTgt !== this.gait
      ? this.gait + (this.gaitTgt - this.gait) * (1 - Math.exp(-dt * 10))
      : this.gait;
    if (Math.abs(this.gaitTgt - this.gait) > 0.25) this.gait += walkW * dt;
    this.wing += (this.wingTgt - this.wing) * (1 - Math.exp(-dt * 12)) + 66 * this.flyLevel * dt * 0.2;

    const g = this.fly.root;
    g.position.copy(this.cur.pos);
    g.rotation.set(0, -this.cur.yaw, 0, "YXZ");
    this._pose(dt);
    // blob shadow tracks the fly, thinning with altitude
    const h = Math.max(0, this.cur.pos.y - this.fly.bodyY);
    this.blobShadow.position.set(this.cur.pos.x, 0.05, this.cur.pos.z);
    this.blobShadow.material.opacity = 0.28 / (1 + h * 0.12);
    this.blobShadow.scale.setScalar(1 + h * 0.02);

    // camera follow
    const o = this.orbit;
    if (o.follow) {
      o.cx += (g.position.x - o.cx) * 0.08;
      o.cz += (g.position.z - o.cz) * 0.08;
      o.cy += (g.position.y - o.cy) * 0.08;
    }
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
