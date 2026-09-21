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

# optional JIT acceleration — the same math is implemented in numpy below;
# numba just fuses it into one compiled pass (no temporaries, one sweep).
# Everything falls back cleanly when numba is unavailable (e.g. Python 3.14).
try:
    from numba import njit

    @njit(cache=True, fastmath=True)
    def _step_fused(v, spikes, rate, refr_counter, i_syn, in_norm, ext, bg,
                    out_ptr, out_idx, eff,
                    el, v_reset, v_thresh, tau_m, tau_syn, mv_per_unit, dt,
                    refrac_steps, rate_a, rate_b):
        n = v.shape[0]
        # 1. event scatter: every edge whose presynaptic neuron fired
        for j in range(n):
            if spikes[j]:
                s = out_ptr[j]
                e = out_ptr[j + 1]
                for k in range(s, e):
                    i_syn[out_idx[k]] += eff[k]
        decay = np.exp(-dt / tau_syn)
        dtf = dt / tau_m
        # 2-4. decay + integrate + spike/reset/refractory + rate in one sweep
        for i in range(n):
            i_syn[i] *= decay
            refr = refr_counter[i] > 0
            if not refr:
                drive = (i_syn[i] * in_norm[i] + ext[i] + bg[i]) * mv_per_unit
                v[i] += ((el - v[i]) + drive) * dtf
            sp = False
            if (not refr) and v[i] >= v_thresh:
                sp = True
            spikes[i] = sp
            if sp:
                v[i] = v_reset
                refr_counter[i] = refrac_steps
            elif refr:
                refr_counter[i] -= 1
            rate[i] = rate[i] * rate_a + (rate_b if sp else 0.0)

    _FUSED = _step_fused
except Exception:      # pragma: no cover — depends on local numba install
    _FUSED = None


class LIFNetwork:
    DT_MS = 2.0           # 2 ms substeps: halves compute per simulated second;
                          # tau_m/tau_syn stay ≥ 2× dt so dynamics are stable
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
        self.refrac_steps = 1           # × dt = 2 ms absolute refractory
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
        # slowly-refreshed background "noise" drive (kept separate so the
        # per-step channel buffer can be cheaply zeroed; rng over 139k is
        # ~2 ms, so it is only redrawn every few steps by the engine)
        self.bg_current = np.zeros(N, dtype=np.float32)
        # preallocated step buffers — the hot loop allocates NOTHING per step
        # (each 139k temporary used to cost ~0.1 ms in alloc + cache churn)
        self._drive = np.empty(N, dtype=np.float32)
        self._dv = np.empty(N, dtype=np.float32)
        self._refr = np.empty(N, dtype=bool)
        self._nrefr = np.empty(N, dtype=bool)
        # None = not JIT-compiled yet / True = fused kernel / False = numpy path
        self._fused_state = None

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
        # ---- accelerated path: compile once on the real arrays, remember
        # whether it worked, then run the fused kernel every step
        if self._fused_state is None:
            if _FUSED is not None:
                try:
                    _FUSED(self.v, self.spikes, self.rate, self.refr_counter,
                           self.i_syn, self.in_norm, self.ext_current,
                           self.bg_current, ptr, idx, self.eff_out,
                           self.el, self.v_reset, self.v_thresh, self.tau_m,
                           self.tau_syn, self.MV_PER_UNIT, dt,
                           self.refrac_steps, 0.88, 0.12)
                    self._fused_state = True
                except Exception:
                    self._fused_state = False
            else:
                self._fused_state = False
        if self._fused_state:
            _FUSED(self.v, self.spikes, self.rate, self.refr_counter,
                   self.i_syn, self.in_norm, self.ext_current,
                   self.bg_current, ptr, idx, self.eff_out,
                   self.el, self.v_reset, self.v_thresh, self.tau_m,
                   self.tau_syn, self.MV_PER_UNIT, dt,
                   self.refrac_steps, 0.88, 0.12)
            return self.spikes
        # 1. event-driven scatter of presyn spikes onto the sparse graph.
        #    Benchmarked on the full 3.7 M-edge graph: per-spiking-neuron
        #    np.add.at wins by ~8× while spikes are sparse (<~3k/ms — the
        #    normal regime); the batch mask+bincount path only makes sense
        #    for seizure-sized bursts.  Benchmarks live in this comment
        #    because a "more vectorised" version was tried and was 8× slower.
        if self.spikes.any():
            act = np.nonzero(self.spikes)[0]
            if len(act) <= 3000:
                for j in act:
                    s, e = ptr[j], ptr[j + 1]
                    if e > s:
                        np.add.at(self.i_syn, idx[s:e], self.eff_out[s:e])
            else:
                sel = self.spikes[con.edge_src]
                contrib = np.bincount(idx[sel], weights=self.eff_out[sel],
                                      minlength=self.i_syn.size)
                self.i_syn += contrib.astype(np.float32, copy=False)
        # 2. synaptic dynamics (exponential decay, in place)
        self.i_syn *= float(np.exp(-dt / self.tau_syn))
        # 3. membrane integration — fully in place, no temporaries
        refr, nrefr = self._refr, self._nrefr
        np.greater(self.refr_counter, 0, out=refr)
        np.logical_not(refr, out=nrefr)
        drive = self._drive
        np.multiply(self.i_syn, self.in_norm, out=drive)
        drive += self.ext_current
        drive += self.bg_current
        drive *= self.MV_PER_UNIT
        dv = self._dv
        np.subtract(self.el, self.v, out=dv)
        dv += drive
        dv *= (dt / self.tau_m)
        dv[refr] = 0.0                      # hold while refractory
        self.v += dv
        # 4. spike / reset / refractory (in place)
        np.greater_equal(self.v, self.v_thresh, out=self.spikes)
        self.spikes &= nrefr
        self.v[self.spikes] = self.v_reset
        self.refr_counter[self.spikes] = self.refrac_steps
        self.refr_counter[refr] -= 1
        # rate low-pass (in place; same recurrence as before)
        self.rate *= 0.88
        self.rate[self.spikes] += 0.12
        return self.spikes

    def activity_for_view(self) -> np.ndarray:
        """0..1 per-neuron visibility (instantaneous-ish for rendering)."""
        hot = np.clip((self.v - self.el) / (self.v_thresh - self.el), 0, 1)
        return np.maximum(hot, np.minimum(self.rate * 6.0, 1.0)).astype(np.float32)
