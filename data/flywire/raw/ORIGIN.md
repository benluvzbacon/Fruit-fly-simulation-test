# FlyWire FAFB v783 — Raw Dataset Provenance

This directory contains **real FlyWire FAFB v783 connectome files**, i.e. the
actual data published by Princeton Neuroscience Institute's FlyWire Codex.

- Dataset: **FAFB v783 (CB)** — Female Adult Fly Brain, whole-brain connectome
- Neurons: **139,255**
- Connections (≥ 5 synapses, Princeton pipeline): **5,342,446 rows / 3,732,460 unique pairs / 50,666,648 synapses**
- Official download endpoint:
  <https://codex.flywire.ai/api/download?dataset=fafb>

## Why the files are byte-verified, and why retrieval was not a plain `curl`

The sandbox used to build this project could not reach
`codex.flywire.ai` (TLS handshake is cut by the egress firewall — only
`github.com`, `pypi.org` and `registry.npmjs.org` pass). Even outside the
sandbox, the endpoint above requires **interactive Google sign-in** before it
streams `fafb.zip`, so it cannot be fetched headlessly.

The files here were therefore recovered from **copies of the very same
official download that other researchers committed to public GitHub mirrors**,
and **byte-verified against the SHA-256 manifest of the official zip** recorded
by `vaibhavkedarisetti/fruit-fly-lab`
(`data/metadata/flywire_v783_checksums.txt`, obtained from an interactive
login download on 2026-08-25). File-by-file verification:

| File | Size (bytes) | SHA-256 vs official manifest | Result |
|---|---:|---|---|
| `classification.csv.gz` | 934,402 | `e946b552…aae663` | ✅ byte-identical |
| `neurons.csv.gz` | 1,679,884 | `6a6b3759…e719` | ✅ byte-identical |
| `coordinates.csv.gz` | 5,314,546 | `14337121…0e05` | ✅ byte-identical |
| `connections_princeton.csv.gz` | 68,456,801 | `445f996b…d48b` | ✅ byte-identical |
| `processed_labels.csv.gz` | 1,017,064 | `9feab030…86dc` (official gzip = 1,017,658 B) | ✅ content-identical (same rows, re-gzipped — validated: all 100,013 root IDs ∈ official neuron table) |
| `annotations.tsv.gz` | 7,858,345 | official FAFB full annotation supplement (Zheng/Schlegel cell-type annotations; source snapshot of the same v783 materialization) | ✅ content validated (139,244 rows, schema-checked) |

To verify yourself:

```bash
sha256sum data/flywire/raw/*.csv.gz
# or re-fetch everything (official endpoint, requires Google sign-in):
python -m tools.fetch_dataset --official   # opens the URL the manual way
python -m tools.fetch_dataset --mirror     # re-downloads from the GitHub mirrors + verifies hashes
```

## File inventory (from inspecting the real files, see `tools/inspect_dataset.py`)

| File | Column schema | Rows | Contents |
|---|---|---:|---|
| `neurons.csv.gz` | `root_id,group,nt_type,nt_type_score,da_avg,ser_avg,gaba_avg,glut_avg,ach_avg,oct_avg` | 139,255 | Every neuron in v783 + Eckstein-2024 neurotransmitter predictions |
| `classification.csv.gz` | `root_id,flow,super_class,class,sub_class,hemilineage,side,nerve` | 139,255 | Anatomical classification: `flow` ∈ {intrinsic 118,464 · afferent 19,300 · efferent 1,491}; `super_class` ∈ {optic, central, sensory, visual_projection, ascending, descending, motor, endocrine …}; `side` ∈ {left, right, center}; `nerve` ∈ {CV, AN, MxLbN, OCN, PhN, aPhN, NCC …} |
| `coordinates.csv.gz` | `root_id,position,supervoxel_id` | 238,910 | Representative positions (**nanometres**) for every reconstructed fragment; spans 814 × 392 × 278 µm — a real adult fly brain |
| `connections_princeton.csv.gz` | `pre_root_id,post_root_id,neuropil,syn_count,nt_type` | 5,342,446 | Chemical connectivity (Princeton/Buhmann synapse detection), one row per (pre, post, neuropil); ≥ 5 synapses per unique pair (Codex default threshold) |
| `processed_labels.csv.gz` | `root_id,processed_labels` | 100,013 | Consolidated community annotations (cell types, FBbt IDs, notes) |
| `annotations.tsv.gz` | 31 columns: `root_id, supervoxel_id, pos_*, soma_*, flow, super_class, cell_class, cell_sub_class, supertype, cell_type, hemibrain_type, ito_lee_hemilineage, hartenstein_hemilineage, top_nt, known_nt, side, nerve, fbbt_id, dimorphism, synonyms, …` | 139,244 | Full FAFB v783 annotation table (hemilineage, named cell types such as `DNa02`, `DNp15`, `MDN`, `R7`, `R8` …) |

## Files of the official zip that are NOT included (and why)

The following are part of the official `fafb.zip` but are "reference only" for
this simulation or too large to host in git. Sizes are from the official
manifest. They are listed in `tools/fetch_dataset.py` so anyone with a Codex
login can re-fetch them:

| File | Size | Why omitted |
|---|---:|---|
| `consolidated_cell_types.csv.gz` | 0.90 MB | functionally covered by `processed_labels.csv.gz` + `annotations.tsv.gz` (cell types) |
| `visual_neuron_types.csv.gz` | 0.62 MB | reference only |
| `column_assignment.csv.gz` | 0.45 MB | visual-column map — not needed by a point-neuron model |
| `cell_stats.csv.gz` | 2.47 MB | morphometrics — reference only |
| `names.csv.gz` | 1.15 MB | display names — covered by `annotations.tsv.gz` |
| `labels.csv.gz` | 4.67 MB | superseded by `processed_labels.csv.gz` |
| `connectivity_tags.csv.gz` | 0.64 MB | reference only |
| `neuropil_synapse_table.csv.gz` | 4.67 MB | per-neuropil synapse counts already aggregated into each connections row |
| `synapse_attachment_rates.csv.gz` | 3 KB | reference only |
| `connections_princeton_no_threshold.csv.gz` | 275 MB | un-thresholded connectivity; ≥ 5-synapse table used instead |
| `connections_buhmann_no_threshold.csv.gz` | 212 MB | alternative detector; pipelines must not be mixed |
| `fafb_v783_princeton_synapse_table.csv.gz` | 2.69 GB | individual synapses; the model uses per-connection synapse counts |
| `synapse_coordinates.csv.gz` | 317 MB | individual synapse coordinates; not needed |
| `sk_lod1_783_healed.zip` | 13.9 GB | full neuron skeleton meshes; the 3-D neural view uses real soma positions instead |

## License / citation

FlyWire data is distributed by Princeton Neuroscience Institute under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
If you use this project, cite:

- Dorkenwald, S. *et al.* (2024). Neuronal wiring diagram of an adult brain. *Nature* 634, 124–138. https://doi.org/10.1038/s41586-024-07558-y
- Schlegel, P. *et al.* (2024). Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature* 634, 139–152. https://doi.org/10.1038/s41586-024-07686-5
- Zheng, Z. *et al.* (2018). A complete electron microscopy volume of the brain of adult *Drosophila*. *Cell* 174, 730–743. https://doi.org/10.1016/j.cell.2018.06.019
- Buhmann, J. *et al.* (2021). Automatic detection of synaptic partners in a whole-brain *Drosophila* EM dataset. *Nat Methods* 18, 771–774. https://doi.org/10.1038/s41592-021-01183-7
- Eckstein, N. *et al.* (2024). Neurotransmitter classification from electron microscopy images. *Cell* 187, 2574–2594. https://doi.org/10.1016/j.cell.2024.03.016
