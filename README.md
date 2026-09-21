# 🪰 FlyWire Fruit-Fly Simulation

A runnable, embodied fruit-fly simulation whose nervous system is built from
the **real FlyWire FAFB v783 whole-brain connectome** — 139,255 real neurons,
3,732,460 real synaptic connections, 50.7 million real synapses — driving a 3D
fly body with a closed sensory-motor loop.

```
ENVIRONMENT → SENSORY INPUT → FLYWIRE NETWORK → MOTOR OUTPUT → MUSCLES → FLY BODY → ENVIRONMENT
```

The connectome is the structural basis of the simulated nervous system.  No
random stand-in network is used anywhere.

---

## 1. The dataset

| | |
|---|---|
| **Dataset** | FlyWire **FAFB v783** — whole brain of an adult female *Drosophila melanogaster* (serial-section EM volume “FAFB”) |
| **Official source** | **<https://codex.flywire.ai/api/download?dataset=fafb>** |
| **Neurons** | **139,255** |
| **Connections** | **3,732,460** unique neuron pairs (≥ 5 synapses, Princeton pipeline) |
| **Synapses** | **50,666,648** |
| **Host** | Princeton Neuroscience Institute (Codex / FlyWire) |

The raw files live in `data/flywire/raw/` — byte-identical to the official
download, verified against its SHA-256 manifest (see
[`data/flywire/raw/ORIGIN.md`](data/flywire/raw/ORIGIN.md) for the full
provenance story, per-file checksums and what was deliberately left out, and
[`data/flywire/raw/INSPECTION.md`](data/flywire/raw/INSPECTION.md) for a
generated dump of exactly what is inside each file).

Because the official endpoint requires an interactive Google sign-in, the
files were recovered from public GitHub **copies of the same official
download** and verified byte-for-byte.  To re-fetch them yourself:

```bash
python -m tools.fetch_dataset --official   # guided, requires sign-in in your browser
python -m tools.fetch_dataset --mirror     # headless, re-downloads + verifies hashes
python -m tools.fetch_dataset --verify     # hash-check whatever is on disk
```

Cell-type annotation is from the FlyWire/bibliography community work
(Schlegel et al. 2024), see `annotations.tsv.gz`.

---

## 2. Quick start

```bash
pip install -r requirements.txt             # numpy only; the server is stdlib

# 1. inspect the raw files   → data/flywire/raw/INSPECTION.md
python -m tools.inspect_dataset

# 2. build the sparse graph + neuron DB (25 s) → data/flywire/processed/
python -m tools.build_connectome

# 3. emit the validation report (real, measured numbers)
python -m tools.validate_dataset

# 4. run the simulation + interactive viewer
python server.py --port 8000
# open http://localhost:8000/
```

If the page feels heavy, the brain thread is capped at ~55 % of one CPU core
by default; the 3D scene animates smoothly regardless (the browser
interpolates between physics updates).  Tuning:

```bash
python server.py --port 8000 --duty 0.8   # let the brain use up to 80 % of a core (faster sim)
python server.py --port 8000 --duty 0.3   # cooler/quieter on a small laptop
```

The header telemetry shows the resulting brain speed (fraction of realtime).

If `data/flywire/processed/connectome_graph.npz` is missing, step 2 will be
required once.  All heavy generating artifacts are git-ignored and regenerate
deterministically from the committed raw files.

**Windows note:** everything runs on a stock Windows 10/11 Python
installation.  All FlyWire tables are read with explicit `encoding="utf-8"`
(the label files contain legal multibyte characters such as `á`/`Δ`; relying
on the Windows default codec — cp1252 — would crash), gzip archives are opened
as gzip, ZIP archives are handled via `zipfile`, and binary payloads are
never funneled through text readers.

---

## 3. What the simulation does

- A **3D fruit fly** (head with compound eyes + antennae, thorax, abdomen,
  6 tripod-gaited legs, fluttering wings, extensible proboscis) lives in a
  600 × 600 × 300 mm arena with boulder obstacles, sugar and bitter/aversive
  odor/taste sources, a directional light and a heat zone.
- Every **1 ms**, encoders write sensory drive onto **real annotated sensory
  neurons** (photoreceptors, olfactory/gustatory/mechanosensory afferents),
  activity propagates through the **real sparse connectome** (event-driven
  scatter on CSR structures; 3.7 M edges), and populations of **real named
  descending neurons** (DNa01/02/03 for forward walking, MDN for escape
  backing-up, DNp12/DNp15/DNg13/DNb01/02 for turning, DNg16/DNg42 for flight,
  DNb06/DNp13 for halting, the brain motor neurons for feeding) are decoded
  into body commands.
- The fly walks, turns, retreats from bitter/aversive stimuli, seeks and eats
  sugar, startles at looming flashes and takes off, cruises, steers and lands.
- **NORMAL VIEW** shows the fly in the world; **NEURAL VIEW** shows the real
  brain as a 139,255-neuron 3D point cloud (real soma positions, colorable by
  super-class / side / neurotransmitter), with firing neurons lighting up live.
- Click any neuron: root ID, cell type, super-class, side, nerve, real
  in/out-degree and synapse counts, current membrane state, top partners —
  then stimulate it (+30 mV) or **trace the signal** downstream through the
  actual synaptic graph and watch the cascade layer by layer.
- Preset experiments: *light → eye*, *sugar taste*, *bitter escape*,
  *DNa02 forward walk*.

---

## 4. Real data vs approximations (see the in-app legend too)

**REAL FLYWIRE DATA**
neuron identities · connectivity · synapse counts per connection ·
neurotransmitter-sign per edge (Eckstein et al. 2024 predictions) · cell types,
super-classes, hemilineages, sides, nerves · soma 3D positions (814 × 392 ×
278 µm brain cloud) · sensory & descending population membership.

**SIMULATION APPROXIMATION**
leaky integrate-and-fire membrane dynamics (EM contains no voltage
recordings) · synaptic weights = saturating function of the real synapse
count · sensory transduction (linear encoders onto real afferents) ·
spontaneous receptor background · hunger gating · loom-alert and flight-
sustain drives · muscle/body mechanics (kinematic body, procedural gait) ·
arena geometry and fields.  Approximations are marked in the code with
`APPROXIMATION` comments and listed in the UI.

---

## 5. Repository layout

```
data/flywire/
  raw/        # REAL FAFB files, protected by checksums (committed)
  extracted/  # scratch (git-ignored)
  processed/  # sparse graph / sqlite / meta / validation report (generated;
              # meta.json + validation_report.md committed, big files git-ignored)
tools/
  fetch_dataset.py      # official URL fetch guide + mirror re-download + SHA-256 verification
  inspect_dataset.py    # blind inspection of whatever raw files exist
  build_connectome.py   # importer: raw → sparse graph + sqlite + meta.json
  validate_dataset.py   # measured validation report
fly/
  connectome.py  # CSR sparse access to the real graph
  lif.py         # 139k-neuron vectorised integrate-and-fire
  sensory.py     # world state → drive onto real sensory neurons
  motor.py       # real DN populations → body commands
  body.py, world.py, engine.py
server.py        # stdlib HTTP server (sim + JSON API + static viewer)
web/             # three.js viewer (vendored three.module.js, no CDN needed)
data/flywire/processed/validation_report.md   # generated, real numbers
```

---

## 6. Citations & license of the data

FlyWire data © Princeton Neuroscience Institute, distributed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Please cite the primary papers when you use this:

- Dorkenwald S. *et al.* (2024). *Nature* 634, 124–138.
  https://doi.org/10.1038/s41586-024-07558-y
- Schlegel P. *et al.* (2024). *Nature* 634, 139–152.
  https://doi.org/10.1038/s41586-024-07686-5
- Zheng Z. *et al.* (2018). *Cell* 174, 730–743.
  https://doi.org/10.1016/j.cell.2018.06.019
- Buhmann J. *et al.* (2021). *Nat Methods* 18, 771–774.
  https://doi.org/10.1038/s41592-021-01183-7
- Eckstein N. *et al.* (2024). *Cell* 187, 2574–2594.
  https://doi.org/10.1016/j.cell.2024.03.016

Code in this repository: MIT (the *data* keeps its own CC BY-NC-SA license).
