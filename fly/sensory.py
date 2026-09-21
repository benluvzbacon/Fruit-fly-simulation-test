"""Sensory encoders — world state → drive onto REAL FlyWire neurons.

REAL FLYWIRE DATA: the *target populations* are real annotated sensory neurons
(photoreceptors, olfactory/gustatory/mechanosensory afferents, resolved from
``classification.csv.gz`` and the annotation table).

SIMULATION APPROXIMATION: transduction itself (environmental value → firing
current) is a simple linear saturating encoder; receptors' biophysics are not
part of the connectome.
"""
from __future__ import annotations

import numpy as np


class SensoryEncoder:
    """Computes per-channel drive levels 0..1 from world + body state."""

    def __init__(self, con, world, body):
        self.con, self.world, self.body = con, world, body
        self.levels: dict[str, float] = {}
        self.hunger = 0.5   # set externally by the engine each step

    # ------------------------------------------------------------------
    def compute(self) -> dict[str, float]:
        w, b = self.world, self.body
        L: dict[str, float] = {}

        # ---- light (REAL photoreceptors, laterality by head orientation)
        li = w.light_intensity_at(b.pos)                # 0..1
        ang = w.light_angle(b.pos, b.yaw)               # −π..π relative bearing
        L["light_L"] = li * max(0.0, 1.0 + np.sin(ang)) / 1.0 if ang <= 0 else li * max(0.0, 1.0 - ang / np.pi)
        L["light_R"] = li * max(0.0, 1.0 - np.sin(ang)) / 1.0 if ang >= 0 else li * max(0.0, 1.0 + ang / np.pi)

        # ---- odours (REAL olfactory afferents; ORNs are extraordinarily
        # sensitive — sqrt response compresses the range like Weber's law;
        # appetitive gain is gated by hunger state, set by the engine)
        gate = 0.35 + 0.65 * self.hunger
        od_l = float(np.sqrt(w.odor_at(b.pos, kind="food"))) * gate
        od_b = float(np.sqrt(w.odor_at(b.pos, kind="danger")))
        odor_side = w.odor_angle(b.pos, b.yaw, kind="food")
        split = 0.5 + 0.5 * (np.sin(odor_side) if odor_side is not None else 0.0)
        # lateralise like the antennae: food to the LEFT (bearing > 0, CCW)
        # reads STRONGER on the left ORN set — the previous assignment was
        # anatomically flipped and produced repulsion instead of attraction
        L["odor_L"] = min(1.0, od_l * (1.0 + 0.6 * split))
        L["odor_R"] = min(1.0, od_l * (1.0 + 0.6 * (1.0 - split)))
        # aversive odour rides on the same ORN set, encoded via "danger" tag —
        # bitter is also routed through taste when contacted
        L["_danger_odor"] = od_b

        # ---- taste (REAL gustatory neurons carrying sugar/bitter labels)
        L["taste_sugar"] = w.taste_at(b.pos, "sugar")
        L["taste_bitter"] = w.taste_at(b.pos, "bitter") + min(1.0, od_b) * 0.0

        # ---- mechanosensory (REAL mechanosensory afferents: touch & air flow)
        touch = 1.0 if b.contact else 0.0
        wind = min(1.0, b.speed / 400.0) if b.flying else 0.15 * min(1.0, b.speed / 25.0)
        L["mechanosensory"] = max(touch, wind)

        # ---- thermo/hygro (REAL thermosensory + hygrosensory neurons)
        L["thermo_hygro"] = w.heat_at(b.pos)

        # ---- looming / visual surprise: bilateral bright flash
        L["light_flash"] = w.flash_level

        self.levels = L
        return L


CHANNEL_GAIN = {
    # current injected (level × gain × LIFNetwork.MV_PER_UNIT ≈ mV above EL)
    "light_L": 10.0, "light_R": 10.0, "light_flash": 14.0,
    "odor_L": 12.0, "odor_R": 12.0,
    "taste_sugar": 22.0, "taste_bitter": 24.0, "taste_all": 0.0,
    "mechanosensory": 12.0, "thermo_hygro": 6.0,
    # APPROXIMATION: looming/flash alerts the flight-initiation DNs directly
    # (in the real brain, looming LC neurons drive DNg42/giant-fiber circuits)
    "dn_flight": 25.0,
}
