"""Fruit-fly body — 3D kinematics driven entirely by motor commands.

SIMULATION APPROXIMATION: the body is a kinematic model (walking speed,
turning rate, simple flight dynamics + procedural leg/wing animation on the
client).  It has all the parts the UI needs — head, thorax, abdomen, 6 legs,
wings, compound eyes, antennae — but it is not a physics-grade
NeuroMechFly-style biomechanical model.
"""
from __future__ import annotations

import numpy as np


class FlyBody:
    WALK_SPEED = 28.0      # mm/s (real flies walk ~10-30 mm/s)
    BACK_SPEED = 12.0
    FLY_SPEED = 260.0
    FLY_CLIMB = 90.0

    def __init__(self, world):
        self.world = world
        self.pos = np.array([520.0, 300.0, 0.0], dtype=np.float32)
        self.yaw = 0.8
        self.pitch = 0.0
        self.speed = 0.0
        self.flying = False
        self.fly_level = 0.0     # smoothed take-off envelope
        self.contact = False
        self.walk_phase = 0.0
        self.wing_phase = 0.0
        self.feed_timer = 0.0

    # ------------------------------------------------------------------
    def update(self, dt_s: float, cmd: dict) -> None:
        w = self.world
        walk = float(np.clip(cmd["walk"], -1.0, 1.0))
        back = float(np.clip(cmd["backward"], 0.0, 1.0))
        yaw_cmd = float(np.clip(cmd["yaw"], -8.0, 8.0))
        fly = float(np.clip(cmd["fly"], 0.0, 1.0))

        # flight envelope (take off / land smoothly)
        self.fly_level += (fly - self.fly_level) * min(1.0, dt_s * 3.5)
        self.flying = self.fly_level > 0.35

        self.yaw += yaw_cmd * dt_s
        fwd = np.array([np.cos(self.yaw), np.sin(self.yaw)], dtype=np.float32)

        if self.flying:
            speed = self.FLY_SPEED * self.fly_level
            climb = self.FLY_CLIMB * (self.fly_level - 0.35) * 1.4
            target_z = 150.0 + 60.0 * np.sin(w.time_s * 0.23)
            climb += (target_z - self.pos[2]) * 0.7
        else:
            speed = self.WALK_SPEED * walk - self.BACK_SPEED * back
            climb = -200.0 * dt_s if self.pos[2] > 0.5 else 0.0

        vxy = fwd * speed
        v = np.array([vxy[0], vxy[1], climb], dtype=np.float32)
        new_pos = self.pos + v * dt_s
        new_pos, contact = w.resolve_collisions(new_pos)
        self.contact = contact
        self.speed = float(np.linalg.norm((new_pos - self.pos)[:2]) / max(dt_s, 1e-6))
        self.pos = new_pos.astype(np.float32)
        if self.pos[2] < 0.0:
            self.pos[2] = 0.0
        self.pitch = float(np.clip(np.tanh(climb / 90.0), -1, 1) * 0.5) if self.flying else 0.0

        # procedural animation phases (rendered client-side)
        self.walk_phase += dt_s * (2.0 + 22.0 * min(1.0, abs(speed) / self.WALK_SPEED))
        if not self.flying and abs(speed) < 0.5:
            self.walk_phase += dt_s * 0.4          # idle micro-movement
        self.wing_phase += dt_s * (66.0 * self.fly_level)
        if cmd.get("feed", 0.0) > 0.25:
            self.feed_timer = min(1.0, self.feed_timer + dt_s * 2.0)
        else:
            self.feed_timer = max(0.0, self.feed_timer - dt_s)
        if True:
            w.try_eat(new_pos, dt_s)

    # ------------------------------------------------------------------
    def state(self) -> dict:
        return dict(
            pos=[float(x) for x in self.pos], yaw=float(self.yaw),
            pitch=float(self.pitch), speed=self.speed, flying=self.flying,
            contact=self.contact, walk_phase=float(self.walk_phase % (2 * np.pi)),
            wing_phase=float(self.wing_phase % (2 * np.pi)),
            feed=self.feed_timer,
        )
