"""Vectorised leaky integrate-and-fire (LIF) dynamics for 139,255 neurons.

SIMULATION APPROXIMATION — the connectome (topology + synapse counts +
neurotransmitter signs) is REAL FlyWire data; membrane dynamics are a model:

- Every neuron is a point-LIF neuron (FAFB has no voltage recordings, so no
  measured membrane parameters exist — this follows the convention of
  Shiu et al. 2024 whole-brain models, which also drive the connectome with
  generic dynamics).
- Synaptic *sign* comes from the REAL presynaptic neurotransmitter prediction
  (Eckstein et al. 2024): GABA / glutamate → inhibitory (insect central
  glutamate acts on GluCl channels), acetylcholine → excitatory, amines
  (SER/DA/OCT) → weakly excitatory (scaled ×0.3).
- Synaptic *strength* = real synapse count compressed by ``min(syn, 64)**0.7``
  (saturating weighting as used by connectome-wide simulations).
- A small background current keeps the network near threshold
  ("spontaneous activity") — APPROXIMATION: half of the fly's real input
  (descending drive from the VNC / body state) is not part of FAFB.
"""
from __future__ import annotations

import numpy as np

from .connectome import Connectome


class LIFNetwork:
    DT_MS = 1.0
    MV_PER_UNIT = 3.4     # mV per unit of (normalised synaptic + injected) current

    def __init__(self, con: Connectome, seed: int = 7):
        self.con = con
        N = con.N
        self.rng = np.random.default_rng(seed)
        # state
        self.v = np.full(N, -65.0, dtype=np.float32)
        self.el = -65.0
        self.v_reset = -70.0
        self.v_thresh = -45.0
        self.tau_m = 20.0               # ms
        self.refrac_steps = 2           # ms
        self.refr_counter = np.zeros(N, dtype=np.int8)
        # per-synapse efficacy: compress synapse count, keep sign
        self.eff_out = np.sign(con.out_w) * np.power(
            np.minimum(np.abs(con.out_w), 64.0), 0.7).astype(np.float32)
        # amines are weak modulators rather than 1:1 drivers — scale down
        nt_names = con.books["nt"]
        amine_codes = [i for i, n in enumerate(nt_names) if n in ("SER", "DA", "OCT")]
        if amine_codes:
            src_nt = con.nt[con.edge_src]
            m = np.isin(src_nt, amine_codes)
            self.eff_out[m] *= 0.3
        # divisive normalisation by total inbound synapses (stability)
        insyn = np.maximum(con.in_syn, 1.0)
        self.in_norm = (1.0 / (1.0 + np.sqrt(insyn) / 30.0)).astype(np.float32)
        # state buffers
        self.i_syn = np.zeros(N, dtype=np.float32)
        self.tau_syn = 4.0              # ms
        self.spikes = np.zeros(N, dtype=bool)
        self.rate = np.zeros(N, dtype=np.float32)
        self.ext_current = np.zeros(N, dtype=np.float32)

    # --------------------------------------------------------------
    def inject(self, idx: np.ndarray, current: float) -> None:
        """Inject current into a channel (sensory drive — APPROXIMATION of
        receptor transduction, not of the connectome itself).
        Always max-combines (channels may share neurons); the caller resets
        ``ext_current`` at the start of every micro-step, so a channel that is
        silent simply does not inject."""
        if current > 0 and len(idx):
            self.ext_current[idx] = np.maximum(
                self.ext_current[idx], current).astype(np.float32)

    def stimulate_one(self, idx: int, mv: float = 25.0) -> None:
        self.v[idx] += mv

    def step(self, dt: float | None = None) -> np.ndarray:
        dt = dt or self.DT_MS
        con = self.con
        ptr, idx = con.out_indptr, con.out_idx
        # 1. event-driven scatter of presyn spikes onto the sparse graph
        if self.spikes.any():
            act = np.nonzero(self.spikes)[0]
            if len(act) <= 3000:
                for j in act:
                    s, e = ptr[j], ptr[j + 1]
                    if e > s:
                        np.add.at(self.i_syn, idx[s:e], self.eff_out[s:e])
            else:
                sel = self.spikes[con.edge_src]
                np.add.at(self.i_syn, idx[sel], self.eff_out[sel])
        # 2. synaptic dynamics (exponential decay)
        self.i_syn *= float(np.exp(-dt / self.tau_syn))
        # 3. membrane integration
        refr = self.refr_counter > 0
        drive = (self.i_syn * self.in_norm + self.ext_current) * self.MV_PER_UNIT
        dv = ((self.el - self.v) + drive) * (dt / self.tau_m)
        self.v = np.where(refr, self.v, self.v + dv).astype(np.float32)
        # 4. spike / reset / refractory
        self.spikes = (~refr) & (self.v >= self.v_thresh)
        self.v[self.spikes] = self.v_reset
        self.refr_counter[self.spikes] = self.refrac_steps
        if refr.any():
            self.refr_counter[refr] -= 1
        self.rate = (self.rate * 0.94 + self.spikes.astype(np.float32) * 0.06
                     ).astype(np.float32)
        return self.spikes

    def activity_for_view(self) -> np.ndarray:
        """0..1 per-neuron visibility (instantaneous-ish for rendering)."""
        hot = np.clip((self.v - self.el) / (self.v_thresh - self.el), 0, 1)
        return np.maximum(hot, np.minimum(self.rate * 6.0, 1.0)).astype(np.float32)
