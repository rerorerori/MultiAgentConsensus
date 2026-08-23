"""
Random Walk Baseline — Rastgele waypoint seçimi.
"""
import numpy as np
from config import BOUNDS_X, BOUNDS_Y, MIN_RPM, MAX_RPM, CRUISE_RPM, WP_ACCEPT_R, wrap_heading


class RandomWalkPlanner:
    """Sınırlar içinde rastgele waypoint seç."""

    def __init__(self):
        self.targets = {}

    def plan(self, name, pos, yaw, tick):
        """Hedefe ulaşınca yeni rastgele hedef seç."""
        if name not in self.targets:
            self.targets[name] = self._random_wp()

        dist = np.linalg.norm(pos[:2] - self.targets[name])
        if dist < WP_ACCEPT_R:
            self.targets[name] = self._random_wp()

        vec = self.targets[name] - pos[:2]
        heading = np.degrees(np.arctan2(vec[1], vec[0]))
        rpm = np.clip(CRUISE_RPM, MIN_RPM, MAX_RPM)
        return wrap_heading(heading), rpm, self.targets[name]

    def _random_wp(self):
        return np.array([
            np.random.uniform(*BOUNDS_X),
            np.random.uniform(*BOUNDS_Y),
        ])
