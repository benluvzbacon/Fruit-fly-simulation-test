// App orchestration: polling the live simulation, panels, mode switching.
import { FlyView } from "./flyview.js";
import { NeuralView } from "./neuralview.js";

const $ = s => document.querySelector(s);
const api = async (path, body) => {
  if (body !== undefined) {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    return r.json();
  }
  return (await fetch(path)).json();
};

const wrap = $("#canvas-wrap");
const flyView = new FlyView(wrap);
const neuralView = new NeuralView(wrap);
neuralView.renderer.domElement.style.display = "none";
let mode = "normal";
let META = null;
let worldBuilt = false;

// ------------------------------------------------------------ mode switch
async function setMode(m) {
  mode = m;
  flyView.renderer.domElement.style.display = m === "normal" ? "" : "none";
  neuralView.renderer.domElement.style.display = m === "neural" ? "" : "none";
  $("#mode-normal").classList.toggle("active", m === "normal");
  $("#mode-neural").classList.toggle("active", m === "neural");
  $("#view-label").textContent = m === "normal"
    ? "NORMAL VIEW — the fly in its world"
    : "NEURAL VIEW — the real FlyWire brain (139,255 neurons from v783)";
  if (m === "neural" && !neuralView.loaded) {
    await neuralView.load();
  }
  resize();
}
$("#mode-normal").onclick = () => setMode("normal");
$("#mode-neural").onclick = () => setMode("neural");

// ------------------------------------------------------------ tabs
document.querySelectorAll(".tab").forEach(b => {
  b.onclick = () => {
    document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
    document.querySelectorAll(".tabpage").forEach(x => x.classList.add("hidden"));
    b.classList.add("active");
    $("#tab-" + b.dataset.tab).classList.remove("hidden");
  };
});

// ------------------------------------------------------------ panels
function bar(label, v, motor = false) {
  v = Math.max(0, Math.min(1, v));
  return `<div class="bar${motor ? " motor" : ""}"><span>${label}</span>
    <div class="track"><div class="fill" style="width:${(v * 100).toFixed(1)}%"></div></div>
    <span class="val">${(v).toFixed(3)}</span></div>`;
}

function renderTelemetry(f) {
  const spikeRate = f.spikes_per_window && f.target_steps
    ? Math.round(f.spikes_per_window / Math.max(1, (f.t_ms - (renderTelemetry.lastT || 0))) * 1000 / 1000) : 0;
  renderTelemetry.lastT = f.t_ms;
  $("#telemetry").innerHTML = `<div class="kv">
    <span>sim time</span><b>${(f.t_ms / 1000).toFixed(1)} s</b>
    <span>wall time</span><b>${f.wall_time_s} s</b>
    <span>brain speed</span><b>≈${f.sim_speed_x}× realtime (${f.target_steps} ms/frame)</b>
    <span>spikes (window)</span><b>${f.spikes_per_window ?? 0}</b>
    <span>position</span><b>${f.body.pos.map(x => x.toFixed(0)).join(", ")} mm</b>
    <span>speed</span><b>${f.body.speed.toFixed(1)} mm/s ${f.body.flying ? "· ✈️ FLYING" : "· 🚶 walking"}</b>
    <span>contact</span><b>${f.body.contact ? "🧱 touch" : "—"}</b>
    </div>`;
  const p = f.cmd?.pools || {};
  $("#pools").innerHTML =
    bar("forward (DNa02…)", p.forward || 0, true) +
    bar("backward (MDN)", p.backward || 0, true) +
    bar("turn L", p.turn_L || 0, true) +
    bar("turn R", p.turn_R || 0, true) +
    bar("flight (DNg42…)", p.flight || 0, true) +
    bar("halt (DNb06…)", p.halt || 0, true) +
    bar("feeding motor", p.motor || 0, true);
  const L = f.levels || {};
  $("#levels").innerHTML = ["light_L", "light_R", "odor_L", "odor_R", "taste_sugar", "taste_bitter", "mechanosensory", "thermo_hygro", "light_flash"]
    .map(k => bar(k.replace("_", " "), L[k] || 0)).join("");
  $("#events").innerHTML = (f.events || []).map(e => `<div class="event">▸ ${e}</div>`).join("");
}

// ------------------------------------------------------------ neuron info
async function showNeuron(idx) {
  if (idx < 0) { neuralView.selectNeuron(-1); $("#neuron-info").innerHTML = "<p class='dim'>no neuron under cursor</p>"; return; }
  const info = await api(`/api/neuron?idx=${idx}`);
  if (info.error) { $("#neuron-info").textContent = info.error; return; }
  neuralView.selectNeuron(idx, info);
  const kv = [
    ["root_id", info.root_id], ["cell_type", info.cell_type || "—"],
    ["supertype", info.supertype || "—"], ["super_class", info.super_class],
    ["class", info.class || "—"], ["flow", info.flow],
    ["side", info.side], ["nerve", info.nerve || "—"],
    ["neurotransmitter", info.nt || "?"],
    ["hemilineage", info.hemilineage || "—"],
    ["soma position µm", `${info.x.toFixed(1)}, ${info.y.toFixed(1)}, ${info.z.toFixed(1)}`],
    ["out-edges", `${info.out_deg} (Σ ${info.out_syn.toFixed(0)} syns)`],
    ["in-edges", `${info.in_deg} (Σ ${info.in_syn.toFixed(0)} syns)`],
    ["live rate", info.live.rate.toFixed(3)], ["live v", info.live.v.toFixed(1) + " mV"],
  ];
  if (info.labels) kv.push(["labels", info.labels.slice(0, 140)]);
  $("#neuron-info").innerHTML = `<div class="kv">` + kv.map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join("") + `</div>
    <h4>drives → (top of ${info.out_deg})</h4><div class="conn">${
      info.out_n.map(n => `<div><span>${n.cell_type || n.root_id}</span><b>${Math.abs(n.w)} syns ${n.w < 0 ? "inh" : "exc"}</b></div>`).join("")}</div>
    <h4>driven by ← (top of ${info.in_deg})</h4><div class="conn">${
      info.in_n.map(n => `<div><span>${n.cell_type || n.root_id}</span><b>${Math.abs(n.w)} syns ${n.w < 0 ? "inh" : "exc"}</b></div>`).join("")}</div>
    <button onclick="window.__stim(${info.root_id})">⚡ stimulate +30 mV</button>
    <button onclick="window.__trace(${info.root_id})">↠ trace downstream</button>`;
  document.querySelector('[data-tab="neuron"]').click();
}
window.__stim = root => api("/api/command", { action: "stimulate", root_id: root, mv: 30 });
window.__trace = root => { document.querySelector('[data-tab="trace"]').click(); $("#trace-seed").value = root; runTrace(root); };

neuralView.onPick = idx => showNeuron(idx);

$("#neuron-search-btn").onclick = findNeuron;
$("#neuron-search").addEventListener("keydown", e => { if (e.key === "Enter") findNeuron(); });
async function findNeuron() {
  const q = $("#neuron-search").value.trim();
  if (!q) return;
  const { results } = await api(`/api/search?q=${encodeURIComponent(q)}`);
  $("#search-results").innerHTML = results.length
    ? results.map(r => `<div class="hit" onclick="window.__pick(${r.idx})">${r.cell_type || "—"} <b>${r.root_id}</b><small>${r.super_class} · ${r.class || ""}</small></div>`).join("")
    : `<p class="dim">no matches</p>`;
}
window.__pick = idx => { setMode("neural").then(() => showNeuron(idx)); };

// ------------------------------------------------------------ tracing
let lastTrace = null;
async function runTrace(root, depth) {
  depth = depth || parseInt($("#trace-depth").value || "3");
  const r = await api(`/api/trace?root=${root}&depth=${depth}`);
  if (r.error) { $("#trace-info").innerHTML = r.error; return; }
  lastTrace = r;
  $("#trace-info").innerHTML = `<p class="dim">seed root <b>${r.seed}</b> → following real synapses:</p>` +
    r.layers.map((L, d) => `<div class="layer">layer ${d + 1}: <b>${L.length}</b> neurons ${
      L.slice(0, 6).map(([j, w]) => `<span class="pill">${w}×syn</span>`).join("")}</div>`).join("");
  if (mode !== "neural") await setMode("neural");
  neuralView.showTrace(r.layers, r.seed_idx);
  // also stimulate the seed so the network lights up on the same path
  api("/api/command", { action: "stimulate", root_id: root, mv: 45 });
}
$("#trace-run").onclick = () => { const r = $("#trace-seed").value.trim(); if (r) runTrace(parseInt(r)); };
$("#trace-anim").onclick = () => { if (lastTrace) neuralView.showTrace(lastTrace.layers, lastTrace.seed_idx); };

document.querySelectorAll(".trace-preset").forEach(b => {
  b.onclick = async () => {
    const kind = b.dataset.kind;
    const chan = { light: "light_R", sugar: "taste_sugar", bitter: "taste_bitter", walk: "dn_walk_forward" }[kind];
    const ex = META?.channels?.[chan]?.example_root_ids;
    if (!ex || !ex.length) return;
    if (kind === "light") api("/api/command", { action: "flash", level: 1.3 });
    runTrace(ex[0], 4);
  };
});

// ------------------------------------------------------------ controls
$("#btn-pause").onclick = () => api("/api/command", { action: paused ? "resume" : "pause" }).then(() => paused = !paused);
$("#btn-reset").onclick = () => api("/api/command", { action: "reset" }).then(() => { worldBuilt = false; });
$("#btn-flash").onclick = () => api("/api/command", { action: "flash", level: 1.6 });
$("#btn-light").onclick = () => api("/api/command", { action: "toggle_light" });
$("#btn-sugar").onclick = () => api("/api/command", { action: "drop_food", x: lastBody[0], y: lastBody[1] });
$("#btn-bitter").onclick = () => api("/api/command", { action: "drop_bitter", x: lastBody[0], y: lastBody[1] });
$("#btn-clear").onclick = () => api("/api/command", { action: "clear_sources" });
$("#hunger").oninput = e => api("/api/command", { action: "set_hunger", value: parseFloat(e.target.value) });
$("#stim-btn").onclick = () => { const r = parseInt($("#stim-root").value); if (r) window.__stim(r); };
$("#tp-btn").onclick = () => {
  const x = parseFloat($("#tp-x").value), y = parseFloat($("#tp-y").value);
  if (!isNaN(x) && !isNaN(y)) api("/api/command", { action: "teleport_fly", x, y });
};
let paused = false, lastBody = [300, 300, 0];

// ------------------------------------------------------------ main loop
async function poll() {
  try {
    const f = await api("/api/frame");
    if (!worldBuilt && f.world) { flyView.buildWorld(f.world); worldBuilt = true; }
    lastBody = f.body.pos;
    flyView.update(f);
    if (neuralView.loaded) neuralView.updateActivity(f.active_idx || [], f.active_val || []);
    renderTelemetry(f);
    if (f.paused !== paused) paused = f.paused;
  } catch (e) {
    console.warn("poll", e);
  } finally {
    setTimeout(poll, 160);
  }
}

function resize() {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  flyView.resize(w, h); neuralView.resize(w, h);
}
addEventListener("resize", resize);

(async function boot() {
  META = await api("/api/meta");
  const c = META.counts;
  $("#databadge").textContent =
    `REAL FlyWire FAFB v783 · ${c.neurons.toLocaleString()} neurons · ` +
    `${c.edges_unique_pairs.toLocaleString()} connections · ${c.synapses.toLocaleString()} synapses`;
  const legendEl = document.createElement("div");
  resize();
  poll();
  // render loop
  (function frame() {
    if (mode === "normal") flyView.render(); else neuralView.render();
    requestAnimationFrame(frame);
  })();
})();
