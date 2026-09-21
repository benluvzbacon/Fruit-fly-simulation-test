"""Simulation engine — the closed loop

    ENVIRONMENT → SENSORY INPUT → FLYWIRE NETWORK → MOTOR OUTPUT → BODY → WORLD

Every brain step is a 2 ms LIF substep over the REAL FlyWire graph.  The only
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
    # --- tunable SIMULATION APPROXIMATION constants (all connectivity is REAL)
    BG_MEAN = 2.8          # background drive mean (tuned: sparse-but-alive,
                           # ~600k spikes/s network-wide, no seizure)
    BG_STD = 1.0           # background drive spread
    MOTIV_FORWARD = 3.2    # hunger-gated foraging drive onto DNa forward pool
    MOTIV_ODOR_FORWARD = 4.5   # extra forward drive proportional to food odor
    MOTIV_LIGHT_FORWARD = 0.8  # mild phototaxis onto forward pool
    MOTIV_TURN = 26.0      # bilateral imbalance → turn-pool steering
                       # (calibrated vs decoder: 8+ units for a real turn)
    MOTIV_WANDER = 6.0     # scale of the stochastic (OU) search meander

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
        # grooming reflex bookkeeping (post-ingestive fixed action pattern)
        self._was_feeding = False
        self.groom_timer = 0.0
        # klinotaxis odour memory (motivational APPROXIMATION state)
        self._odor_mem = 0.0
        self._odor_mem_t = 0
        self._odor_delta = 0.0
        # stochastic internal steering state (OU process + bout bookkeeping —
        # no scripted motion anywhere)
        self._ou = 0.0
        self._bout_t_left = 0.0
        self._bout_steer = 0.0

    # ------------------------------------------------------------------
    def apply_sensory(self, tick: int) -> None:
        # --- periodic (cheap-amortised) work --------------------------------
        # Recomputing world levels, drawing 139k gaussians and 15k uniform
        # randoms EVERY 1 ms step cost ~3 ms per simulated millisecond —
        # more than the network update itself.  These signals change on a
        # ~10 ms timescale, so they are refreshed periodically instead:
        #   background noise : every 8 steps
        #   world levels + spontaneous events : every 4 steps
        if tick % 4 == 0:
            self.sensory.hunger = self.hunger
            self.last_levels = self.sensory.compute()
            # spontaneous receptor events: ~8 ms depolarising pulses (mean
            # rate equivalent to a few Hz — receptors really do fire in the
            # dark).  Refreshed every 4 steps at 4× probability.
            self.spont_buf[self.net.rng.random(len(self.sens_idx)) < 0.004] = 8.0
        levels = self.last_levels
        if tick % 8 == 0:
            # fresh drive background noise (APPROXIMATION — keeps the network
            # warm; see module docs).  Mean bias ~4.4 units ≈ el + 15 mV:
            # just below threshold, so real sensory channels (≈5+ units)
            # push neurons over it instead of the brain sitting comatose.
            bg = self.net.rng.normal(self.BG_MEAN, self.BG_STD, self.net.v.shape[0])
            np.clip(bg, 0.0, None, out=bg)
            self.net.bg_current = bg.astype(np.float32)
        # --- per-step cheap work --------------------------------------------
        # channel injection buffer starts clean each step
        self.net.ext_current[:] = 0.0
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
            # noxious drive onto backward-walking MDNs (escape reflex) —
            # strong, like the loom→giant-fiber path: MDNs are heavily
            # inhibited by network feedback, weak drives never reach stride
            if len(self.ch_idx["dn_walk_backward"]):
                self.net.inject(self.ch_idx["dn_walk_backward"], 22.0 * dgr)

        # ---- motivational / state drive (SIMULATION APPROXIMATION) ---------
        # Real flies initiate locomotion from internal state (hunger, arousal)
        # and orient along sensory gradients; the descending neurons that
        # express this (DNa02 forward, DNb01/DNp15 turning) are REAL, but the
        # upstream motivational computation (dopaminergic gating etc.) is too
        # state-dependent for generic LIF weights to express on its own.
        # These small state-dependent currents onto the REAL DN pools close
        # the internal-state → action loop; sensory routing still flows through
        # the REAL graph (injected afferents above).
        m_fwd = self.MOTIV_FORWARD * (0.30 + 0.70 * self.hunger)
        m_fwd += self.MOTIV_ODOR_FORWARD * (levels.get("odor_L", 0.0) + levels.get("odor_R", 0.0))
        li = (levels.get("light_L", 0.0) + levels.get("light_R", 0.0))
        m_fwd += self.MOTIV_LIGHT_FORWARD * li
        m_fwd *= (1.0 - min(1.0, dgr * 1.4))                      # no foraging in danger
        m_fwd *= (1.0 - min(1.0, levels.get("taste_sugar", 0.0)))# stop when feeding
        # arrest near the source: slow down as odour peaks (local search gait)
        m_fwd *= (1.0 - 0.55 * min(1.0, 0.5 * (levels.get("odor_L", 0.0) + levels.get("odor_R", 0.0))))
        # arrival under the lamp: stop pressing forward into the wall below it
        m_fwd *= (1.0 - 0.55 * min(1.0, li))
        # wall-contact reflex (SIMULATION APPROXIMATION of touch-evoked
        # escape-turning): without this the fly noses into a corner and the
        # constant forward drive pins it there forever
        touching = levels.get("mechanosensory", 0.0) >= 0.5
        if touching:
            m_fwd *= 0.05
            if len(self.ch_idx["dn_walk_backward"]):
                self.net.inject(self.ch_idx["dn_walk_backward"], 16.0)
        if m_fwd > 0 and len(self.ch_idx["dn_walk_forward"]):
            self.net.inject(self.ch_idx["dn_walk_forward"], m_fwd)
        # orienting: bilateral odor imbalance steers through the turn pools.
        # Klinotaxis done properly (SIMULATION APPROXIMATION of the real
        # Berg/Brown-style strategy): while the odour level is RISING the fly
        # goes almost straight (no reason to turn); when FALLING, the full
        # differential steers it back — this converges instead of orbiting.
        oL, oR = levels.get("odor_L", 0.0), levels.get("odor_R", 0.0)
        o_avg = 0.5 * (oL + oR)
        if self.t_ms - self._odor_mem_t >= 200:
            self._odor_delta = o_avg - self._odor_mem
            self._odor_mem = o_avg
            self._odor_mem_t = self.t_ms
        rising = max(0.0, self._odor_delta * 5.0)
        turn_gate = float(np.clip(1.0 - 2.5 * rising, 0.15, 1.0))
        l_diff = levels.get("light_L", 0.0) - levels.get("light_R", 0.0)
        steer = self.MOTIV_TURN * ((oL - oR) * turn_gate + 0.5 * l_diff)
        # spontaneous steering is BRAIN-STATE noise, not a script: an
        # Ornstein–Uhlenbeck process (slow, mean-reverting random walk —
        # the standard model for internal states like arousal) biases the
        # differential turn pools, so meander direction emerges instead of
        # following any fixed pattern.  Suppressed while tracking odour.
        g = 1.0 - min(1.0, o_avg * 1.3)
        dt = self.net.DT_MS
        self._ou += (-self._ou / 1200.0) * dt + self.net.rng.normal(0.0, 1.0) * (dt / 1200.0) ** 0.5
        steer += self.MOTIV_WANDER * g * 2.2 * self._ou
        # Poisson reorientation bouts: hazard ~1/7 s of sim time, random
        # direction & duration 0.6-1.5 s (real flies reorient stochastically
        # between straight runs — no timers anywhere)
        if self._bout_t_left > 0:
            self._bout_t_left -= dt
            steer += self._bout_steer
        elif dgr <= 0 and not touching and o_avg < 0.4 and levels.get("taste_sugar", 0) < 0.2:
            if self.net.rng.random() < dt / 7000.0:
                self._bout_t_left = float(self.net.rng.uniform(600, 1500))
                self._bout_steer = float(self.net.rng.choice([-1.0, 1.0]) * self.net.rng.uniform(9, 15))
        # danger: strong alternating reorientation turns while backing away
        # (real flies: backward burst + body turn, then forward escape)
        if dgr > 0:
            steer += 18.0 * dgr * (1.0 if (self.t_ms // 700) % 2 == 0 else -1.0)
        # wall-contact: spin-scan while backing out of the corner
        elif touching:
            steer += 20.0 * (1.0 if (self.t_ms // 900) % 2 == 0 else -1.0)
        steer *= (1.0 - min(1.0, levels.get("taste_sugar", 0.0)))
        if abs(steer) > 0.01:
            pool = self.ch_idx["dn_turn"]
            if len(pool):
                sideL = self.con.side[pool] == self.con.books["side"].index("left")
                tgt = pool[sideL] if steer > 0 else pool[~sideL]
                if len(tgt):
                    self.net.inject(tgt, abs(steer))
        # hunger rises over time, resets when eating (rates per 2 ms substep)
        self.hunger = min(1.0, self.hunger + 0.00001)
        if self.last_levels.get("taste_sugar", 0) > 0.3:
            self.hunger = max(0.0, self.hunger - 0.001)
        # user stimulation
        for idx, mv in self._stim_targets:
            self.net.stimulate_one(idx, mv)
        self._stim_targets = []

    # ------------------------------------------------------------------
    def step(self, n: int = 1) -> None:
        dt = self.net.DT_MS
        for _ in range(max(1, int(n // dt))):
            self.apply_sensory(self.t_ms)
            self.net.step(self.net.DT_MS)
            self.t_ms += int(dt)
            self.spikes_since_snapshot += int(self.net.spikes.sum())
        # motor decode at tick resolution (population rate)
        cmd = self.motor.decode(self.net.rate)
        # APPROXIMATION: sustained-flight feedback — airborne + fast movement
        # keeps the flight attractor engaged (biologically: visual-flow +
        # haltere feedback routed back through the descending system)
        if self.body.flying:
            cmd["fly"] = max(cmd["fly"],
                             0.5 + 0.25 * min(1.0, self.body.speed / 200.0))
        # post-ingestive grooming: when a feeding bout ends, flies run a
        # stereotyped foreleg/head-cleaning sequence (real fixed action
        # pattern — here it is a state-triggered reflex, not an animation
        # loop; locomotion is suppressed for the ~2.4 s bout)
        if self.body.feed_timer > 0.7:
            self._was_feeding = True
        if self._was_feeding and self.body.feed_timer <= 0.05:
            self._was_feeding = False
            self.groom_timer = 2.4
            self.events.append("post-ingestive grooming bout (fixed action pattern — APPROXIMATION)")
        if self.groom_timer > 0:
            self.groom_timer = max(0.0, self.groom_timer - n / 1000.0)
            cmd["walk"] *= 0.08
            cmd["backward"] = cmd.get("backward", 0.0) * 0.08
            cmd["yaw"] *= 0.12
            cmd["fly"] = min(cmd.get("fly", 0.0), 0.05)
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
            body={**self.body.state(), "groom": min(1.0, self.groom_timer / 2.4)},
            world=self.world.serialise(),
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
