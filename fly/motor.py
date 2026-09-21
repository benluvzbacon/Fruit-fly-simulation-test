"""Motor decoding — REAL descending / motor neuron activity → body commands.

REAL FLYWIRE DATA: the decoder pools are real named descending neurons
(DNa01/02/03, MDN, DNg13, DNb01/02, DNp12/15, DNg16, DNg42, DNb06, DNp13)
and the annotated brain motor neurons, resolved from the v783 annotation table.

SIMULATION APPROXIMATION: the mapping activation → muscle command is a
population-rate decoder.  Real flies route these DNs through the ventral nerve
cord (not part of FAFB), and real muscle dynamics are far more complex; the
mapping below is documented and adjustable.
"""
from __future__ import annotations

import numpy as np


class MotorDecoder:
    def __init__(self, con):
        self.con = con
        ch = con.channels
        sideL = con.side == con.books["side"].index("left")
        sideR = con.side == con.books["side"].index("right")
        self.pool_forward = ch["dn_walk_forward"]
        self.pool_backward = ch["dn_walk_backward"]
        self.pool_flight = ch["dn_flight"]
        self.pool_halt = ch["dn_halt"]
        turn = ch["dn_turn"]
        self.pool_turn_L = turn[sideL[turn]] if len(turn) else turn
        self.pool_turn_R = turn[sideR[turn]] if len(turn) else turn
        self.pool_motor = ch["brain_motor"]
        # flight fallback if annotation table lacks flight DNs (it doesn't —
        # but keep the fallback code honest and documented)
        if not len(self.pool_flight):
            dn_all = ch["dn_all"]
            grp = con.grp[dn_all]
            self.pool_flight = np.zeros(0, dtype=np.int32)
        self.did_fly = False

    def _ap(self, rate: np.ndarray, pool: np.ndarray) -> float:
        if len(pool) == 0:
            return 0.0
        return float(rate[pool].mean())

    # ------------------------------------------------------------------
    def decode(self, rate: np.ndarray) -> dict:
        ap_fwd = self._ap(rate, self.pool_forward)
        ap_bwd = self._ap(rate, self.pool_backward)
        ap_tL = self._ap(rate, self.pool_turn_L)
        ap_tR = self._ap(rate, self.pool_turn_R)
        ap_flt = self._ap(rate, self.pool_flight)
        ap_halt = self._ap(rate, self.pool_halt)
        ap_mot = self._ap(rate, self.pool_motor)

        # turn rate: DN pools are bilateral; ipsilateral DNp15-type turning
        yaw_cmd = 6.0 * (ap_tL - ap_tR)                    # rad/s
        walk = 0.9 * np.tanh(6.0 * max(0.0, ap_fwd - 0.01))
        walk += 0.6 * np.tanh(3.0 * (ap_tL + ap_tR))       # turning implies gait
        walk -= 0.8 * np.tanh(8.0 * ap_bwd)
        walk += 1.0 * np.tanh(3.0 * ap_bwd) * -0.0         # (backward handled by sign below)
        backward = 0.8 * np.tanh(8.0 * ap_bwd)
        walk = float(np.clip(walk - 0.9 * ap_halt, -1.0, 1.0))
        fly_drive = float(np.tanh(10.0 * (ap_flt - 0.015))) if len(self.pool_flight) else 0.0
        fly_drive = max(0.0, fly_drive)
        feed = float(np.tanh(6.0 * ap_mot))
        return dict(
            walk=walk, backward=float(backward), yaw=yaw_cmd,
            fly=fly_drive, feed=feed,
            pools=dict(forward=ap_fwd, backward=ap_bwd, turn_L=ap_tL,
                       turn_R=ap_tR, flight=ap_flt, halt=ap_halt, motor=ap_mot),
        )
