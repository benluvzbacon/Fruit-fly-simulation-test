#!/usr/bin/env python3
"""Build the simulation-ready connectome from the raw FlyWire FAFB files.

Reads (all REAL data, see data/flywire/raw/ORIGIN.md):

- ``neurons.csv.gz``            official v783 neuron table (139,255 rows)
- ``classification.csv.gz``     flow / super_class / class / sub_class / side / nerve
- ``coordinates.csv.gz``        representative positions, **nanometres**
- ``annotations.tsv.gz``        full annotation table (cell_type, supertype,
                                hemilineages, soma position in 4×4×40 nm voxels)
- ``processed_labels.csv.gz``   consolidated community labels
- ``connections_princeton.csv.gz``  5,342,446 (pre, post, neuropil) rows,
                                ≥ 5 synapses per unique pair

Writes (into ``data/flywire/processed``):

- ``connectome_graph.npz``  sparse graph, both directions (CSR via indptr/indices):
    * out-edges: ``out_indptr int32[N+1]``, ``out_idx int32[E]``, ``out_w float32[E]``
      (signed synapse count: negative = inhibitory GABA/GLUT source),
      ``out_neu int16[E]`` (dominant neuropil code)
    * in-edges:  reciprocal arrays (``in_indptr``, ``in_idx``, ``in_w``)
    * node arrays: ``root_id int64[N]``, ``pos_um float32[N,3]``,
      ``has_pos uint8[N]``, categorical codes ``flow/sup/cls/side/nerve/nt int16[N]``,
      degree statistics, and ``ch_<name> int32[]`` sensory/motor channel indices
- ``neurons.sqlite``        per-neuron metadata incl. cell types and labels
- ``meta.json``             codebooks, counts, channel summary, provenance hashes
- ``validation_report.md``  the numbers that were actually detected

Usage::

    python -m tools.build_connectome
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "flywire", "raw")
PROC = os.path.join(ROOT, "data", "flywire", "processed")

INHIBITORY = {"GABA", "GLUT"}        # insect central-brain Glu is inhibitory (GluCl)
NT_ALL = ["ACH", "GABA", "GLUT", "SER", "DA", "OCT"]

# ---------------------------------------------------------------- helpers
def _gz(path):
    return gzip.open(path, "rt", newline="")


def _sha(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- loading
def load_classification(path):
    """root_id -> (flow, super_class, class, sub_class, hemilineage, side, nerve)"""
    out = {}
    with _gz(path) as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            out[r[0]] = (r[1], r[2], r[3], r[4], r[5], r[6], r[7])
    return out


def load_neurons(path):
    """root_id -> (group, nt_type, nt_score) — the authoritative neuron list"""
    out = {}
    with _gz(path) as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            out[r[0]] = (r[1], r[2], float(r[3]) if r[3] else 0.0)
    return out


def load_coordinates(path):
    """root_id -> xyz in **nm** (first fragment per root; median is overkill here)"""
    pos = {}
    with _gz(path) as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            if r[0] in pos:
                continue
            v = r[1].strip("[]").split()
            if len(v) == 3:
                pos[r[0]] = (float(v[0]), float(v[1]), float(v[2]))
    return pos


def load_annotations(path):
    """root_id -> dict(cell_type, supertype, hl_ito, hl_hart, soma_um, top_nt, top_conf, known_nt, fbbt)"""
    out = {}
    with _gz(path) as f:
        rd = csv.DictReader(f, delimiter="\t")
        for r in rd:
            rid = r["root_id"]
            try:
                soma = (float(r["soma_x"]), float(r["soma_y"]), float(r["soma_z"]))
                soma = np.array(soma, dtype=float) * np.array([4.0, 4.0, 40.0]) / 1000.0  # voxels -> µm
                if not np.all(np.isfinite(soma)):
                    soma = None
            except (ValueError, TypeError):
                soma = None
            out[rid] = {
                "cell_type": r["cell_type"], "supertype": r["supertype"],
                "hl_ito": r["ito_lee_hemilineage"], "hl_hart": r["hartenstein_hemilineage"],
                "soma_um": soma, "top_nt": r["top_nt"],
                "top_nt_conf": float(r["top_nt_conf"]) if r["top_nt_conf"] else 0.0,
                "known_nt": r["known_nt"], "fbbt": r["fbbt_id"],
            }
    return out


def load_labels(path):
    """root_id -> consolidated label string"""
    out = {}
    with _gz(path) as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            if len(r) >= 2:
                # stored like "['T4b; FBbt_00003733']" — clean lightly
                s = ",".join(r[1:]).strip().strip("'\"").strip("[]")
                out[r[0]] = s[:400]
    return out


# ---------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    os.makedirs(PROC, exist_ok=True)

    f_class = os.path.join(RAW, "classification.csv.gz")
    f_neur = os.path.join(RAW, "neurons.csv.gz")
    f_coord = os.path.join(RAW, "coordinates.csv.gz")
    f_anno = os.path.join(RAW, "annotations.tsv.gz")
    f_lab = os.path.join(RAW, "processed_labels.csv.gz")
    f_conn = os.path.join(RAW, "connections_princeton.csv.gz")
    for f in (f_class, f_neur, f_conn):
        if not os.path.exists(f):
            raise SystemExit(f"missing {f} — run `python -m tools.fetch_dataset` first")

    log("loading tables …")
    classification = load_classification(f_class)
    neurons = load_neurons(f_neur)
    coordinates = load_coordinates(f_coord) if os.path.exists(f_coord) else {}
    annotations = load_annotations(f_anno) if os.path.exists(f_anno) else {}
    labels = load_labels(f_lab) if os.path.exists(f_lab) else {}
    log(f"  neurons table: {len(neurons):,} | classification: {len(classification):,} | "
        f"coordinates: {len(coordinates):,} | annotations: {len(annotations):,} | labels: {len(labels):,}")

    # ---------------- node table -------------------------------------------
    root_ids = np.array(sorted(int(r) for r in neurons), dtype=np.int64)
    N = len(root_ids)
    ridx = {int(r): i for i, r in enumerate(root_ids)}
    idx_of_str = {}
    log(f"nodes: {N:,}")

    codebooks: dict[str, list[str]] = {}
    def encode(field_vals, book):
        b = {"": 0}
        arr = np.empty(len(field_vals), dtype=np.int16)
        for i, v in enumerate(field_vals):
            if v not in b:
                b[v] = len(b)
            arr[i] = b[v]
        book.extend([None] * len(b))
        for v, c in b.items():
            book[c] = v
        return arr

    flows, sups, clss, subcs, hems, sids, nrvs, grps, nts, labels_arr, ct_arr, st_arr = ([] for _ in range(12))
    pos = np.zeros((N, 3), dtype=np.float32)
    has_pos = np.zeros(N, dtype=np.uint8)
    # annotations positions are µm (from voxels); coordinates are nm -> /1000
    for i, r in enumerate(root_ids.tolist()):
        rs = str(r)
        idx_of_str[rs] = i
        fl, su, cl, sb, hm, sd, nv = classification.get(rs, ("afferent", "", "", "", "", "", ""))
        grp, nt, _ = neurons[rs]
        flows.append(fl); sups.append(su); clss.append(cl); subcs.append(sb)
        hems.append(hm); sids.append(sd); nrvs.append(nv); grps.append(grp)
        nts.append(nt)
        labels_arr.append(labels.get(rs, ""))
        a = annotations.get(rs)
        ct_arr.append(a["cell_type"] if a else "")
        st_arr.append(a["supertype"] if a else "")
        p = coordinates.get(rs)
        if p is not None:
            pos[i] = np.array(p, dtype=np.float32) / 1000.0
            has_pos[i] = 1
        elif a is not None and a["soma_um"] is not None:
            pos[i] = a["soma_um"].astype(np.float32)
            has_pos[i] = 1
    log(f"  positions resolved for {int(has_pos.sum()):,} / {N:,} neurons "
        f"(units: coordinates.csv position is in nanometres → stored as µm)")

    book_l: list[str] = []
    books: dict[str, list[str]] = {}
    flow_arr = encode(flows, books.setdefault("flow", []))
    sup_arr = encode(sups, books.setdefault("super_class", []))
    cls_arr = encode(clss, books.setdefault("class", []))
    sub_arr = encode(subcs, books.setdefault("sub_class", []))
    hem_arr = encode(hems, books.setdefault("hemilineage", []))
    side_arr = encode(sids, books.setdefault("side", []))
    nerve_arr = encode(nrvs, books.setdefault("nerve", []))
    grp_arr = encode(grps, books.setdefault("group", []))
    nt_arr = encode(nts, books.setdefault("nt", []))

    # ---------------- edges -------------------------------------------------
    log("loading connections_princeton.csv.gz (5.3M rows) …")
    pre_l, post_l, neu_l, syn_l, cnt_nt = [], [], [], [], []
    neuropils = Counter()
    with _gz(f_conn) as f:
        rd = csv.reader(f)
        hdr = next(rd)
        for r in rd:
            a = idx_of_str.get(r[0]); b = idx_of_str.get(r[1])
            if a is None or b is None:
                continue
            pre_l.append(a); post_l.append(b)
            neu_l.append(r[2]); syn_l.append(int(r[3]))
    Eraw = len(pre_l)
    log(f"  rows with valid endpoints: {Eraw:,}")
    pre_np = np.asarray(pre_l, dtype=np.int32)
    post_np = np.asarray(post_l, dtype=np.int32)
    syn_np = np.asarray(syn_l, dtype=np.float64)
    pre_l = post_l = None

    neu_codes: dict[str, int] = {}
    neu_np = np.empty(Eraw, dtype=np.int16)
    for i, nm in enumerate(neu_l):
        c = neu_codes.get(nm)
        if c is None:
            c = len(neu_codes)
            neu_codes[nm] = c
        neu_np[i] = c
    neu_l = None
    neu_book = [""] * len(neu_codes)
    for nm, c in neu_codes.items():
        neu_book[c] = nm

    # aggregate (pre,post): sum synapses; dominant neuropil = max-syn row
    log("aggregating to unique neuron pairs …")
    key = pre_np.astype(np.int64) * N + post_np.astype(np.int64)
    order = np.lexsort((-syn_np, key))
    key_s = key[order]; syn_s = syn_np[order]; neu_s = neu_np[order]
    uniq, first = np.unique(key_s, return_index=True)
    syn_agg = np.add.reduceat(syn_s, first)
    neu_agg = neu_s[first]
    pre_agg = (uniq // N).astype(np.int32)
    post_agg = (uniq % N).astype(np.int32)
    E = len(uniq)
    log(f"  unique pairs: {E:,} (was {Eraw:,} rows)")

    # signed weight: inhibitory sources (GABA/GLUT) get negative weight
    inhib_codes = {book_i for book_i, nm in enumerate(books["nt"]) if nm in INHIBITORY}
    inhib_pre = np.isin(nt_arr[pre_agg], list(inhib_codes))
    w_agg = np.where(inhib_pre, -syn_agg, syn_agg).astype(np.float32)

    # out-CSR (sorted by pre) and in-CSR (sorted by post)
    log("building CSR structures …")
    out_order = np.argsort(pre_agg, kind="stable")
    out_idx = post_agg[out_order]
    out_w = w_agg[out_order]
    out_neu = neu_agg[out_order]
    out_pre = pre_agg[out_order]
    out_indptr = np.zeros(N + 1, dtype=np.int64)
    np.add.at(out_indptr, out_pre + 1, 1)
    np.cumsum(out_indptr, out=out_indptr)

    in_order = np.argsort(post_agg, kind="stable")
    in_idx = pre_agg[in_order]
    in_w = w_agg[in_order]
    in_post = post_agg[in_order]
    in_indptr = np.zeros(N + 1, dtype=np.int64)
    np.add.at(in_indptr, in_post + 1, 1)
    np.cumsum(in_indptr, out=in_indptr)

    out_deg = np.diff(out_indptr).astype(np.int32)
    in_deg = np.diff(in_indptr).astype(np.int32)

    def _seg_sum(vals, indptr, deg):
        """sum per CSR segment; 0 for empty segments (reduceat would OOB otherwise)"""
        caps = np.minimum(indptr[:-1], E - 1)
        s = np.add.reduceat(vals, caps) if E else np.zeros(N)
        s[deg == 0] = 0.0
        # for segments where start==E-1 but segment is empty, reduceat returns vals[E-1];
        # the mask above fixes exactly those cases
        return s

    out_syn = _seg_sum(np.abs(out_w), out_indptr, out_deg)
    in_syn = _seg_sum(np.abs(in_w), in_indptr, in_deg)

    # ---------------- sensory / motor channels (from REAL annotations) ------
    log("deriving sensory/motor channels from annotations …")
    sup_names = books["super_class"]; cls_names = books["class"]; side_names = books["side"]
    SUP = {n: i for i, n in enumerate(sup_names)}
    CLS = {n: i for i, n in enumerate(cls_names)}
    SIDE = {n: i for i, n in enumerate(side_names)}

    def is_code(arr, code): return arr == code if code is not None else np.zeros(N, bool)
    sensory = sup_arr == SUP.get("sensory", -1)
    sideL = side_arr == SIDE.get("left", -1)
    sideR = side_arr == SIDE.get("right", -1)

    cls_visual = cls_arr == CLS.get("visual", -1)
    cls_ocellar = cls_arr == CLS.get("ocellar", -1)
    cls_olf = cls_arr == CLS.get("olfactory", -1)
    cls_gust = cls_arr == CLS.get("gustatory", -1)
    cls_mech = cls_arr == CLS.get("mechanosensory", -1)
    cls_thermo = (cls_arr == CLS.get("thermosensory", -1)) | (cls_arr == CLS.get("hygrosensory", -1))
    descend = sup_arr == SUP.get("descending", -1)
    motor_class = cls_arr == CLS.get("brain_motor_neuron", -1) | (sup_arr == SUP.get("motor", -1))

    # cell-type / label based channels
    ct_l = [c.lower() for c in ct_arr]
    lab_l = [l.lower() for l in labels_arr]
    st_l = [s.lower() for s in st_arr]

    def by_cell_type(names: set[str]) -> np.ndarray:
        """Indices whose primary cell_type is exactly one of *names* (case-insensitive)."""
        want = {n.lower() for n in names}
        return np.array([i for i, c in enumerate(ct_l) if c in want and descend[i]], dtype=np.int32)

    def by_label_re(pat: str, mask=None) -> np.ndarray:
        rx = re.compile(pat, re.I)
        out = [i for i in range(N) if (rx.search(labels_arr[i]) or rx.search(ct_arr[i])
                                       or rx.search(st_arr[i])) and (mask is None or mask[i])]
        return np.array(out, dtype=np.int32)

    channels: dict[str, np.ndarray] = {}
    # light: photoreceptors = 'visual' sensory neurons + ocellar nerve neurons
    photo = sensory & (cls_visual | cls_ocellar)
    channels["light_L"] = np.nonzero(photo & sideL)[0].astype(np.int32)
    channels["light_R"] = np.nonzero(photo & sideR)[0].astype(np.int32)
    channels["light_all"] = np.nonzero(photo)[0].astype(np.int32)
    # odour / food smell: olfactory receptor neurons
    olf = sensory & cls_olf
    channels["odor_L"] = np.nonzero(olf & sideL)[0].astype(np.int32)
    channels["odor_R"] = np.nonzero(olf & sideR)[0].astype(np.int32)
    # taste: gustatory neurons carrying 'sugar' / 'bitter' annotations
    gust = sensory & cls_gust
    sugar = by_label_re(r"sugar|sweet", gust)
    bitter = by_label_re(r"bitter", gust)
    channels["taste_sugar"] = sugar if len(sugar) else np.nonzero(gust)[0][:0].astype(np.int32)
    channels["taste_bitter"] = bitter
    channels["taste_all"] = np.nonzero(gust)[0].astype(np.int32)
    # touch / wind / obstacle contact: mechanosensory
    channels["mechanosensory"] = np.nonzero(sensory & cls_mech)[0].astype(np.int32)
    channels["thermo_hygro"] = np.nonzero(sensory & cls_thermo)[0].astype(np.int32)

    # descending motor drive (real named DNs wherever they exist in v783)
    dn = {
        "walk_forward": {"DNa02", "DNa01", "DNa03"},
        "walk_backward": {"MDN"},
        "turn": {"DNp15", "DNp12", "DNg13", "DNb01", "DNb02"},
        "flight": {"DNg42", "GF", "giant fiber", "MIDN", "DNg16"},
        "halt": {"DNp13", "DNb06", "ves", "BRK"},
    }
    dn_found: dict[str, list[str]] = {}
    for chan, names in dn.items():
        idx = by_cell_type(names)
        found = sorted({ct_arr[int(i)] for i in idx})
        dn_found[chan] = found
        channels["dn_" + chan] = idx
        if chan == "flight" and not len(idx):
            # honest fallback: nothing matched in the annotation table
            channels["dn_flight"] = idx
    channels["dn_all"] = np.nonzero(descend)[0].astype(np.int32)
    channels["brain_motor"] = np.nonzero(motor_class)[0].astype(np.int32)
    log("  channels: " + ", ".join(f"{k}={len(v):,}" for k, v in channels.items()))
    log(f"  named DNs found: {dn_found}")

    # ---------------- persist ----------------------------------------------
    graph_path = os.path.join(PROC, "connectome_graph.npz")
    log("writing connectome_graph.npz …")
    arrays = dict(
        root_id=root_ids, pos_um=pos, has_pos=has_pos,
        flow=flow_arr, super_class=sup_arr, cls=cls_arr, sub_class=sub_arr,
        hemilineage=hem_arr, side=side_arr, nerve=nerve_arr, grp=grp_arr, nt=nt_arr,
        out_indptr=out_indptr.astype(np.int64), out_idx=out_idx, out_w=out_w, out_neu=out_neu,
        in_indptr=in_indptr.astype(np.int64), in_idx=in_idx, in_w=in_w,
        out_deg=out_deg, in_deg=in_deg,
        out_syn=out_syn.astype(np.float32), in_syn=in_syn.astype(np.float32),
    )
    for k, v in channels.items():
        arrays["ch_" + k] = v
    np.savez_compressed(graph_path, **arrays)

    db_path = os.path.join(PROC, "neurons.sqlite")
    if os.path.exists(db_path):
        os.remove(db_path)
    log("writing neurons.sqlite …")
    db = sqlite3.connect(db_path)
    db.execute("""CREATE TABLE neurons(
        idx INTEGER PRIMARY KEY, root_id INTEGER, cell_type TEXT, supertype TEXT,
        flow TEXT, super_class TEXT, class TEXT, sub_class TEXT, hemilineage TEXT,
        side TEXT, nerve TEXT, grp TEXT, nt TEXT,
        x REAL, y REAL, z REAL, out_deg INT, in_deg INT, out_syn REAL, in_syn REAL,
        labels TEXT)""")
    db.execute("CREATE INDEX ix_root ON neurons(root_id)")
    db.execute("CREATE INDEX ix_ct ON neurons(cell_type)")
    q = "INSERT INTO neurons VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    bmap = {
        "flow": (books["flow"], flow_arr), "sup": (books["super_class"], sup_arr),
        "cls": (books["class"], cls_arr), "sub": (books["sub_class"], sub_arr),
        "hem": (books["hemilineage"], hem_arr), "side": (books["side"], side_arr),
        "nerve": (books["nerve"], nerve_arr), "grp": (books["group"], grp_arr),
        "nt": (books["nt"], nt_arr),
    }
    def g(name, i):
        book, arr = bmap[name]
        return book[int(arr[i])]
    batch = []
    for i in range(N):
        batch.append((i, int(root_ids[i]), ct_arr[i], st_arr[i],
                      g("flow", i), g("sup", i), g("cls", i), g("sub", i), g("hem", i),
                      g("side", i), g("nerve", i), g("grp", i), g("nt", i),
                      float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]),
                      int(out_deg[i]), int(in_deg[i]), float(out_syn[i]), float(in_syn[i]),
                      labels_arr[i]))
        if len(batch) >= 20000:
            db.executemany(q, batch); batch.clear()
    if batch:
        db.executemany(q, batch)
    db.commit()

    inhib = ("GABA", "GLUT")
    src_hashes = {os.path.basename(f): _sha(f) for f in
                  (f_neur, f_class, f_coord, f_conn, f_lab, f_anno) if os.path.exists(f)}
    meta = dict(
        dataset="FlyWire FAFB v783",
        official_source="https://codex.flywire.ai/api/download?dataset=fafb",
        built=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        counts=dict(
            neurons=int(N), edges_unique_pairs=int(E), edge_rows_raw=int(Eraw),
            synapses=int(float(np.abs(w_agg).sum())),
            with_positions=int(has_pos.sum()),
            neurotransmitters={nm: int((nt_arr == i).sum()) for i, nm in enumerate(books["nt"])},
            super_classes={nm: int((sup_arr == i).sum()) for i, nm in enumerate(books["super_class"])},
            classes={nm: int((cls_arr == i).sum()) for i, nm in enumerate(books["class"]) if (cls_arr == i).sum() > 0},
        ),
        codebooks=books,
        neuropils=neu_book,
        inhibitory=list(inhib),
        channels={k: dict(count=int(len(v)), example_root_ids=[int(root_ids[int(i)]) for i in v[:3]])
                  for k, v in channels.items()},
        named_dn_cell_types=dn_found,
        pos_bbox_um=dict(min=pos[has_pos == 1].min(0).tolist(), max=pos[has_pos == 1].max(0).tolist()),
        source_files=src_hashes,
    )
    with open(os.path.join(PROC, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    db.close()

    log(f"done in {time.time()-t0:.0f}s → {graph_path}")
    return meta


if __name__ == "__main__":
    main()
