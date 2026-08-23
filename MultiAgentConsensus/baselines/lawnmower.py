"""
Lawnmower Baseline — Sabit şerit tabanlı coverage.
"""
import numpy as np
from config import (
    AGENT_NAMES, LAWN_X_MIN, LAWN_X_MAX, LAWN_STRIPS, LAWN_STRIP_W,
    MIN_RPM, MAX_RPM, CRUISE_RPM, WP_ACCEPT_R, wrap_heading,
)


class LawnmowerPlanner:
    """Her AUV için sabit lawnmower şerit waypoint'leri."""

    def __init__(self, agent_names=None):
        names = agent_names or AGENT_NAMES
        self.waypoints = {n: self._generate(n) for n in names}
        self.wp_index = {n: 0 for n in names}
        for n in names:
            print(f"[LAWN] {n}: {len(self.waypoints[n])} waypoints")

    def _generate(self, name):
        """Paralel şerit waypoint'leri oluştur."""
        strip = LAWN_STRIPS[name]
        y_min, y_max = strip["y_min"], strip["y_max"]
        wps = []
        y = y_min + LAWN_STRIP_W / 2
        going_right = True
        while y < y_max:
            if going_right:
                wps.append(np.array([LAWN_X_MIN, y]))
                wps.append(np.array([LAWN_X_MAX, y]))
            else:
                wps.append(np.array([LAWN_X_MAX, y]))
                wps.append(np.array([LAWN_X_MIN, y]))
            y += LAWN_STRIP_W
            going_right = not going_right

        if not wps:
            mid_y = (y_min + y_max) / 2
            wps = [np.array([LAWN_X_MIN, mid_y]), np.array([LAWN_X_MAX, mid_y])]
        return wps

    def plan(self, name, pos, yaw, tick):
        """Sıradaki waypoint'e heading ve RPM döndür."""
        wps = self.waypoints[name]
        idx = self.wp_index[name]
        if idx >= len(wps):
            idx = len(wps) - 1

        target = wps[idx]
        dist = np.linalg.norm(pos[:2] - target)

        if dist < WP_ACCEPT_R and idx < len(wps) - 1:
            self.wp_index[name] = idx + 1
            target = wps[self.wp_index[name]]
            print(f"[NAV] {name} WP {self.wp_index[name]}/{len(wps)}")

        vec = target - pos[:2]
        heading = np.degrees(np.arctan2(vec[1], vec[0]))
        rpm = np.clip(CRUISE_RPM, MIN_RPM, MAX_RPM)
        return wrap_heading(heading), rpm, target
