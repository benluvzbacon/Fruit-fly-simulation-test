"""FlyWire connectome access layer (sparse, CSR).

REAL FLYWIRE DATA: everything this module exposes comes from the imported
``connectome_graph.npz`` which is built (by ``tools/build_connectome.py``)
exclusively from the official FAFB v783 tables.  No edges are invented.

The graph is kept as CSR sparse structures (never a dense matrix — 139,255²
floats would be ~77 GB):
    out-edges: ``out_indptr / out_idx / out_w``        (source → targets)
    in-edges:  ``in_indptr / in_idx / in_w``           (target → sources)
    ``out_w``/``in_w`` are *signed synapse counts*: negative where the source
    neuron is GABAergic or glutamatergic (Eckstein et al. 2024 predictions).
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(os.path.dirname(HERE), "data", "flywire", "processed")


class Connectome:
    def __init__(self, proc_dir: str = PROC):
        self.proc_dir = proc_dir
        g = np.load(os.path.join(proc_dir, "connectome_graph.npz"))
        self.g = g
        self.N = int(len(g["root_id"]))
        self.root_id: np.ndarray = g["root_id"]
        self.idx_of_root = {int(r): i for i, r in enumerate(self.root_id)}
        self.pos: np.ndarray = g["pos_um"]
        self.has_pos: np.ndarray = g["has_pos"]
        for k in ("flow", "super_class", "cls", "sub_class", "hemilineage",
                  "side", "nerve", "grp", "nt", "out_deg", "in_deg",
                  "out_syn", "in_syn"):
            setattr(self, k, g[k])
        # CSR
        self.out_indptr, self.out_idx, self.out_w, self.out_neu = (
            g["out_indptr"], g["out_idx"], g["out_w"], g["out_neu"])
        self.in_indptr, self.in_idx, self.in_w = g["in_indptr"], g["in_idx"], g["in_w"]
        # flat edge maps for vectorised propagation (built from out-CSR)
        deg = self.out_deg.astype(np.int64)
        self.edge_src = np.repeat(np.arange(self.N, dtype=np.int32), deg)
        # meta / codebooks
        with open(os.path.join(proc_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.books = self.meta["codebooks"]
        self.neuropils = self.meta["neuropils"]
        # sensory/motor channels
        self.channels: dict[str, np.ndarray] = {
            k[3:]: g[k] for k in g.files if k.startswith("ch_")
        }

    # ---------------- predicates ------------------------------------------
    def name_book(self, field: str, i: int) -> str:
        return self.books[field][int(getattr(self, field)[i])]

    def targets(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(idx, signed syn count, neuropil code) of neurons *i* drives."""
        s, e = self.out_indptr[i], self.out_indptr[i + 1]
        return self.out_idx[s:e], self.out_w[s:e], self.out_neu[s:e]

    def sources(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(idx, signed syn count) of neurons driving *i*."""
        s, e = self.in_indptr[i], self.in_indptr[i + 1]
        return self.in_idx[s:e], self.in_w[s:e]

    def signed_input_from(self, presyn_subset: np.ndarray) -> np.ndarray:
        """Total signed drive each neuron receives from a presyn spike pattern.

        Vectorised over all 3.7 M edges:
        ``I[post] = Σ w · x(pre)`` via bincount on the flat edge map.
        """
        return np.bincount(
            self._edge_tgt,
            weights=self.in_w * presyn_subset[self.in_idx],
            minlength=self.N,
        ).astype(np.float32)

    def finalize(self) -> None:
        """Derived flat arrays (built lazily to keep load fast)."""
        self._edge_tgt = np.repeat(
            np.arange(self.N, dtype=np.int32), self.in_deg.astype(np.int64))

    def downstream_layer(self, seeds: np.ndarray, depth: int,
                         k: int = 60, inhibitive_sign: bool = True):
        """Top-k strongest k-step downstream layers — used by signal tracing.

        Follows |weight| ranking so weak single-synapse noise edges don't flood
        the trace.  Returns list-of-layers of (idx, weight, parent_idx).
        """
        seeds = np.asarray(list(seeds), dtype=np.int64)
        frontier = {int(s): -1 for s in seeds}
        seen = set(int(s) for s in seeds)
        layers = []
        current = seeds
        parent_map = {int(s): -1 for s in seeds}
        for d in range(depth):
            nxt: dict[int, tuple[float, int]] = {}
            for i in current:
                t, w, _ = self.targets(int(i))
                if len(t) > k:
                    top = np.argpartition(-np.abs(w), k)[:k]
                    t, w = t[top], w[top]
                for j, wt in zip(t.tolist(), w.tolist()):
                    j = int(j)
                    if j in seen:
                        continue
                    score = abs(float(wt))
                    if j not in nxt or score > nxt[j][0]:
                        nxt[j] = (score, int(i))
                    seen.add(j)
            entries = sorted(nxt.items(), key=lambda kv: -kv[1][0])[:800]
            layer = [(j, sc, p) for j, (sc, p) in entries]
            layers.append(layer)
            current = np.array([j for j, _, _ in layer], dtype=np.int64)
            parent_map.update({j: p for j, _, p in layer})
            if len(seen) > 20000 or not len(layer):
                break
        return layers


_CONNECTOME: Connectome | None = None


def get_connectome() -> Connectome:
    global _CONNECTOME
    if _CONNECTOME is None:
        _CONNECTOME = Connectome()
        _CONNECTOME.finalize()
    return _CONNECTOME
