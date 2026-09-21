"""Simulation engine — the closed loop

    ENVIRONMENT → SENSORY INPUT → FLYWIRE NETWORK → MOTOR OUTPUT → BODY → WORLD

Every brain step is a 1 ms LIF update over the REAL FlyWire graph.  The only
non-connectome parts are the boundary transducers (sensory encoding) and the
effector mapping (DN population decoding), both documented as approximations.
"""
from __future__ import annotations

import time

import numpy as np

from .connectome import get_connectome
from .lif import LIFNetwork
from .sensory import SensoryEncoder, CHANNEL_GAIN
from .motor import MotorDecoder
from .world import World
from .body import FlyBody


class Simulation:
    def __init__(self, seed: int = 7):
        t0 = time.time()
        self.con = get_connectome()
        self.net = LIFNetwork(self.con, seed=seed)
        self.world = World()
        self.body = FlyBody(self.world)
        self.sensory = SensoryEncoder(self.con, self.world, self.body)
        self.motor = MotorDecoder(self.con)
        self.t_ms = 0
        self.paused = False
        self.events: list[str] = [f"connectome loaded in {time.time()-t0:.1f}s"]
        self.spikes_since_snapshot = 0
        # channel → index array (cached)
        self.ch_idx = {k: v for k, v in self.con.channels.items()}
        # spontaneous sensory firing mask (REAL sensory neurons fire
        # spontaneously at a few Hz; APPROXIMATION of receptor noise)
        sens_mask = np.zeros(self.con.N, dtype=bool)
        for ch in ("light_all", "odor_L", "odor_R", "taste_all",
                   "mechanosensory", "thermo_hygro"):
            sens_mask[self.ch_idx[ch]] = True
        self.sens_idx = np.nonzero(sens_mask)[0]
        self.spont_buf = np.zeros(len(self.sens_idx), dtype=np.float32)
        self.hunger = 0.5
        self.last_levels: dict[str, float] = {}
        self.last_cmd: dict = {}
        # stimulation requests (user-triggered)
        self._stim_targets: list[tuple[int, float]] = []

    # ------------------------------------------------------------------
    def apply_sensory(self) -> None:
        self.sensory.hunger = self.hunger
        levels = self.sensory.compute()
        self.last_levels = levels
        # fresh drive each micro-step; spontaneous sensory firing
        # (APPROXIMATION — real receptors fire spontaneously at a few Hz,
        # keeping downstream circuits warm; keeps the network alive at rest)
        self.net.ext_current = (
            self.net.rng.normal(0.4, 0.3, self.net.v.shape[0])
        ).clip(0.0, None).astype(np.float32)
        # spontaneous receptor events: ~8 ms depolarising pulses (mean rate
        # equivalent to a few Hz — receptors really do fire in the dark)
        new_ev = self.net.rng.random(len(self.sens_idx)) < 0.001
        self.spont_buf[new_ev] = 8.0
        active_sp = self.spont_buf > 0
        if active_sp.any():
            self.net.inject(self.sens_idx[active_sp], 13.0)
            self.spont_buf[active_sp] -= 1.0
        for ch, idx in self.ch_idx.items():
            gain = CHANNEL_GAIN.get(ch)
            if gain is None or not gain:
                continue
            lvl = levels.get(ch, 0.0)
            if ch == "light_all" or lvl <= 0:
                continue                       # L/R handled separately; reset is implicit
            self.net.inject(idx, gain * lvl)
        # visual flash/looming drives the full photoreceptor sheet, plus
        # direct alerting drive onto the flight-initiation DNs
        # (APPROXIMATION — see sensory.CHANNEL_GAIN["dn_flight"])
        fl = levels.get("light_flash", 0.0)
        if fl > 0:
            self.net.inject(self.ch_idx["light_all"], CHANNEL_GAIN["light_flash"] * fl)
            if len(self.ch_idx["dn_flight"]):
                self.net.inject(self.ch_idx["dn_flight"], CHANNEL_GAIN["dn_flight"] * fl)
        # hunger state gates appetitive drive (APPROXIMATION of state-
        # dependent modulation, e.g. NPF/insulin gating of feeding circuits)
        dgr = levels.get("_danger_odor", 0.0)
        if dgr > 0:
            # aversive odour: bitter afferents carry avoid signal
            # (approximation: aversive olfaction shares gustatory-avoid path)
            self.net.inject(self.ch_idx["taste_bitter"], 6.0 * dgr)
            # noxious drive onto backward-walking MDNs (escape reflex)
            if len(self.ch_idx["dn_walk_backward"]):
                self.net.inject(self.ch_idx["dn_walk_backward"], 8.0 * dgr)
        # hunger rises over time, resets when eating
        self.hunger = min(1.0, self.hunger + 0.00002)
        if self.last_levels.get("taste_sugar", 0) > 0.3:
            self.hunger = max(0.0, self.hunger - 0.002)
        # user stimulation
        for idx, mv in self._stim_targets:
            self.net.stimulate_one(idx, mv)
        self._stim_targets = []

    # ------------------------------------------------------------------
    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.apply_sensory()
            self.net.step(self.net.DT_MS)
            self.t_ms += 1
            self.spikes_since_snapshot += int(self.net.spikes.sum())
        # motor decode at tick resolution (population rate)
        cmd = self.motor.decode(self.net.rate)
        # APPROXIMATION: sustained-flight feedback — airborne + fast movement
        # keeps the flight attractor engaged (biologically: visual-flow +
        # haltere feedback routed back through the descending system)
        if self.body.flying:
            cmd["fly"] = max(cmd["fly"],
                             0.5 + 0.25 * min(1.0, self.body.speed / 200.0))
        self.last_cmd = cmd
        self.body.update(n / 1000.0, cmd)
        self.world.step(n / 1000.0)

    # ------------------------------------------------------------------
    def snapshot(self, topk: int = 12000) -> dict:
        act = self.net.activity_for_view()
        # top-K active neurons by index threshold
        if topk and topk < self.con.N:
            thr = np.partition(act, -topk)[-topk]
            sel = np.nonzero(act >= max(thr, 0.05))[0][:topk]
        else:
            sel = np.nonzero(act > 0.05)[0]
        out = dict(
            t_ms=self.t_ms, paused=self.paused,
            body=self.body.state(), world=self.world.serialise(),
            active_idx=sel.astype(np.int32).tolist(),
            active_val=[round(float(v), 3) for v in act[sel]],
            n_spiking=int(self.net.spikes.sum()),
            spikes_per_window=self.spikes_since_snapshot,
            levels={k: round(float(v), 3) for k, v in self.last_levels.items()},
            cmd={k: (round(float(v), 4) if isinstance(v, float) else
                     {kk: round(float(vv), 4) for kk, vv in v.items()})
                 for k, v in self.last_cmd.items()},
            events=self.events[-8:],
        )
        self.spikes_since_snapshot = 0
        return out

    # ------------------------------------------------------------------
    def stimulate(self, root_id: int, mv: float = 30.0) -> dict:
        idx = self.con.idx_of_root.get(int(root_id))
        if idx is None:
            return dict(error=f"root {root_id} not in v783 neuron set")
        self._stim_targets.append((idx, mv))
        return dict(ok=True, idx=int(idx))

    def trace(self, root_id: int, depth: int = 3) -> dict:
        idx = self.con.idx_of_root.get(int(root_id))
        if idx is None:
            return dict(error=f"root {root_id} not in v783 neuron set")
        layers = self.con.downstream_layer([idx], depth=depth)
        return dict(
            seed=int(root_id), seed_idx=int(idx),
            layers=[[(int(j), round(float(sc), 1), int(p)) for j, sc, p in layer]
                    for layer in layers],
        )
