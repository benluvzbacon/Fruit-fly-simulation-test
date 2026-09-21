"""3D environment — the arena the embodied fly lives in.

Everything here is a SIMULATION APPROXIMATION (world geometry is not part of
the connectome); biological realism is documented per quantity.

Units: millimetres.  Arena 600×600 mm floor, 300 mm high.
The fly is ~2.5 mm long; obstacles are raised cylindrical "boulders"
(50-70 mm), food sources sit on the floor.
"""
from __future__ import annotations

import numpy as np


class World:
    def __init__(self, seed: int = 3):
        rng = np.random.default_rng(seed)
        self.bounds = dict(x=(0.0, 600.0), y=(0.0, 600.0), z=(0.0, 300.0))
        self.obstacles = [  # (cx, cy, radius, height)
            (200.0, 150.0, 55.0, 120.0),
            (430.0, 240.0, 65.0, 150.0),
            (300.0, 420.0, 50.0, 100.0),
            (120.0, 380.0, 45.0, 90.0),
        ]
        # odor / taste sources: (x, y, z, odor_radius, kind, edible_amount, taste)
        self.sources = [
            dict(x=510.0, y=520.0, z=0.0, r=170.0, kind="food", amount=1.0, taste="sugar"),
            dict(x=90.0,  y=80.0,  z=0.0, r=140.0, kind="danger", amount=1.0, taste="bitter"),
        ]
        self.light = dict(x=520.0, y=90.0, z=300.0, intensity=1.0, on=True)
        self.heat = dict(x=80.0, y=80.0, z=0.0, r=60.0, level=0.8)
        self.flash_level = 0.0
        self.time_s = 0.0

    # ------------------------------------------------------------------
    def _dist3(self, p, q):
        return float(np.linalg.norm(np.asarray(p) - np.asarray(q)))

    def odor_at(self, pos, kind="food") -> float:
        v = 0.0
        for s in self.sources:
            if s["kind"] != kind or s["amount"] <= 0:
                continue
            d = self._dist3(pos, (s["x"], s["y"], s["z"]))
            v = max(v, max(0.0, 1.0 - d / s["r"]) * s["amount"])
        return min(1.0, v)

    def odor_angle(self, pos, yaw, kind="food"):
        """Relative bearing of the strongest source of *kind* (or None)."""
        best, bi = 0.0, None
        for s in self.sources:
            if s["kind"] != kind or s["amount"] <= 0:
                continue
            d = self._dist3(pos, (s["x"], s["y"], s["z"]))
            val = max(0.0, 1.0 - d / s["r"])
            if val > best:
                best, bi = val, s
        if bi is None:
            return None
        abs_ang = np.arctan2(bi["y"] - pos[1], bi["x"] - pos[0])
        return float((abs_ang - yaw + np.pi) % (2 * np.pi) - np.pi)

    def taste_at(self, pos, taste: str) -> float:
        v = 0.0
        for s in self.sources:
            if s["taste"] != taste or s["amount"] <= 0:
                continue
            d = self._dist2(pos, s)
            if d < 14.0:      # contact distance (fly + source radius)
                v = max(v, s["amount"])
        return v

    def _dist2(self, pos, s):
        return float(np.hypot(pos[0] - s["x"], pos[1] - s["y"]))

    def heat_at(self, pos) -> float:
        d = self._dist3(pos, (self.heat["x"], self.heat["y"], self.heat["z"]))
        return self.heat["level"] * max(0.0, 1.0 - d / self.heat["r"])

    def light_intensity_at(self, pos) -> float:
        if not self.light["on"]:
            base = 0.04     # ambient floor — fly can still see a little
        else:
            d = self._dist3(pos, (self.light["x"], self.light["y"], self.light["z"]))
            base = self.light["intensity"] * (180.0 / (60.0 + d))
        return float(min(1.2, base) + self.flash_level)

    def light_angle(self, pos, yaw) -> float:
        abs_ang = np.arctan2(self.light["y"] - pos[1], self.light["x"] - pos[0])
        return float((abs_ang - yaw + np.pi) % (2 * np.pi) - np.pi)

    # ------------------------------------------------------------------
    def resolve_collisions(self, pos: np.ndarray, radius: float = 4.0) -> tuple[np.ndarray, bool]:
        """Clamp a proposed position against obstacles / walls; returns (pos, contact)."""
        contact = False
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        x = float(np.clip(x, self.bounds["x"][0] + radius, self.bounds["x"][1] - radius))
        y = float(np.clip(y, self.bounds["y"][0] + radius, self.bounds["y"][1] - radius))
        z = float(np.clip(z, self.bounds["z"][0], self.bounds["z"][1] - radius))
        if (x, y) != (pos[0], pos[1]):
            contact = True
        for (cx, cy, r, h) in self.obstacles:
            if z < h:
                dx, dy = x - cx, y - cy
                d = np.hypot(dx, dy)
                if d < r + radius:
                    contact = True
                    if d < 1e-6:
                        dx, dy, d = 1.0, 0.0, 1.0
                    x = cx + dx / d * (r + radius)
                    y = cy + dy / d * (r + radius)
        return np.array([x, y, z], dtype=np.float32), contact

    def try_eat(self, pos, dt_s: float) -> str | None:
        """Consume a bit of an edible source in contact range; returns taste."""
        for s in self.sources:
            if s["amount"] > 0 and self._dist2(pos, s) < 14.0:
                s["amount"] = max(0.0, s["amount"] - 0.06 * dt_s)
                return s["taste"]
        return None

    def step(self, dt_s: float) -> None:
        self.time_s += dt_s
        self.flash_level = max(0.0, self.flash_level - dt_s * 2.5)

    def flash(self, level: float = 1.0) -> None:
        self.flash_level = min(1.5, level)

    def serialise(self) -> dict:
        return dict(
            bounds=self.bounds, obstacles=self.obstacles,
            sources=[dict(s) for s in self.sources], light=dict(self.light),
            heat=dict(self.heat), time_s=self.time_s,
        )
