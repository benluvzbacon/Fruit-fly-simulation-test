# FlyWire FAFB v783 — dataset validation report

generated: 2026-09-21T01:22:19Z by `tools/validate_dataset.py`


## 1. Provenance / integrity of the raw files

| raw file | bytes | sha-256 | verdict |
|---|---:|---|---|
| `classification.csv.gz` | 934,402 | `e946b552f4056dfc…` | ✅ **byte-identical to the official zip** |
| `neurons.csv.gz` | 1,679,884 | `6a6b3759e635f0f3…` | ✅ **byte-identical to the official zip** |
| `coordinates.csv.gz` | 5,314,546 | `14337121f451f98c…` | ✅ **byte-identical to the official zip** |
| `connections_princeton.csv.gz` | 68,456,801 | `445f996bf6c4b180…` | ✅ **byte-identical to the official zip** |
| `processed_labels.csv.gz` | 1,017,064 | `a520c743fc47a794…` | ✅ content verified (100,013 rows; every root ID ∈ official neuron set; container re-gzipped) |
| `annotations.tsv.gz` | 7,858,345 | `173cd43846e42176…` | ✅ content verified (139,244 annotated rows, schema-checked) |

## 2. What the dataset contains (measured)

- **FlyWire FAFB detected:** YES (v783 neuron and connectivity tables verified byte-identical)
- archives resolved: 6 usable files of the official download (see `data/flywire/raw/ORIGIN.md` for the rest)
- **neuron records:** 139,255
- **synapse records (raw connections rows):** 5,342,446
- **unique connections (pre, post):** 3,732,460
- **total synapses:** 50,666,648
- cell-type / annotation records: 100,013 labelled + 139,244 annotation-table rows
- **brain regions (neuropils):** 79 (`ME_L, LA_L, UNASGD, LOP_L, LO_L, AVLP_L, SLP_L, LH_L, PVLP_L, PLP_L, AOTU_L, AME_L, GNG, AL_R, LAL_R, ME_R…`)
- **positions:** 139,255/139,255 neurons with real 3-D coordinates; span 813.5 × 391.5 × 278.5 µm (nanometre data, stored as µm)
- **morphologies:** soma/anchor positions only — full skeleton meshes (`sk_lod1_783_healed.zip`, 13.9 GB) deliberately not imported; documented in ORIGIN.md

### Neurotransmitter record

- ACH: 82,298
- ∅(unknown): 19,658
- GLUT: 19,605
- GABA: 16,017
- SER: 1,021
- DA: 584
- OCT: 72

### Super-classes

- optic: 77,873
- central: 32,381
- sensory: 16,938
- visual_projection: 7,684
- ascending: 1,750
- descending: 1,305
- sensory_ascending: 612
- visual_centrifugal: 522
- motor: 110
- endocrine: 80
- ∅: 0

### Largest classes

- optic_lobe_intrinsic: 77,382
- ∅: 31,664
- visual: 11,426
- Kenyon_Cell: 5,177
- CX: 2,878
- mechanosensory: 2,674
- olfactory: 2,281
- AN: 2,276
- ALPN: 685
- LHLN: 517
- ALLN: 429
- gustatory: 408
- DAN: 331
- bilateral: 220
- TuBu: 150
- unknown_sensory: 132

## 3. Sensory / motor populations resolved from the annotations

| channel | real neurons | example root IDs |
|---|---:|---|
| light_L | 5,892 | 720575940599736492, 720575940599744172, 720575940600010668 |
| light_R | 5,458 | 720575940601049097, 720575940605160550, 720575940605179360 |
| light_all | 11,426 | 720575940599736492, 720575940599744172, 720575940600010668 |
| odor_L | 1,117 | 720575940603430112, 720575940603659488, 720575940603870688 |
| odor_R | 1,134 | 720575940603356128, 720575940603828704, 720575940603832288 |
| taste_sugar | 23 | 720575940610788069, 720575940611875570, 720575940612670570 |
| taste_bitter | 38 | 720575940602353632, 720575940603266592, 720575940604027168 |
| taste_all | 334 | 720575940602353632, 720575940603266592, 720575940604018208 |
| mechanosensory | 2,674 | 720575940600433181, 720575940600646173, 720575940602132509 |
| thermo_hygro | 103 | 720575940603820716, 720575940604931249, 720575940605264305 |
| dn_walk_forward | 6 | 720575940604737708, 720575940620918789, 720575940627787609 |
| dn_walk_backward | 4 | 720575940610236514, 720575940616026939, 720575940631082808 |
| dn_turn | 12 | 720575940606112940, 720575940616293855, 720575940616471052 |
| dn_flight | 4 | 720575940621336852, 720575940624448956, 720575940641480949 |
| dn_halt | 4 | 720575940619486891, 720575940625295339, 720575940629041879 |
| dn_all | 1,305 | 720575940602534112, 720575940602864300, 720575940603489824 |
| brain_motor | 106 | 720575940605287358, 720575940607193986, 720575940607641010 |

Named descending neurons found in the annotation table and used for motor readout:
- walk_forward: `DNa01`, `DNa02`, `DNa03`
- walk_backward: `MDN`
- turn: `DNb01`, `DNb02`, `DNg13`, `DNp12`, `DNp15`
- flight: `DNg16`, `DNg42`
- halt: `DNb06`, `DNp13`

## 4. Graph integrity checks

- edge indices within range: `True`
- CSR indptr starts/ends consistent: `0 == 0`, `3732460 == 3732460`
- neurons with ≥1 outgoing edge: 137,518
- neurons with ≥1 incoming edge: 130,183
- largest out-degree: 6,523 · largest in-degree: 6,261
- inhibitory edges (GABA/GLUT sources): 1,274,177
- excitatory edges: 2,458,283
- weights are real synapse counts (range 5…2633)

## 5. What is NOT in this dataset (source-level gaps, documented)

- membrane/membrane-channel parameters — not measurable from EM; LIF approximation used
- muscle mechanics & body physics — outside the connectome; kinematic body used
- sensory transduction physics — encoders are linear approximations
- the 275 MB un-thresholded connection table, 2.7 GB per-synapse table and 13.9 GB
  skeleton meshes remain available from the official endpoint (GitHub cannot host them)
