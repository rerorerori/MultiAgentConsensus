"""
Entropy-Only Baseline — Entropy planner WITHOUT consensus.
Local map only, no inter-agent communication.

NOT: Global magnet devre dışı (baseline olduğu için). Saf lokal greedy davranış.
"""
from core.probability_map import ProbabilityMap
from core.entropy_planner import EntropyGuidedPlanner
from config import AGENT_NAMES, MAP_SIZE_M, MAP_RES


class EntropyOnlyPlanner:
    """Her AUV sadece kendi lokal map'ini kullanır, consensus yok."""

    def __init__(self, agent_names=None):
        names = agent_names or AGENT_NAMES
        self.maps = {n: ProbabilityMap(MAP_SIZE_M, MAP_RES) for n in names}
        # agent_name vermiyoruz → global magnet registry karışmasın
        # Baseline olarak global magnet etkili OLMASIN diye özel bir planner
        self.planners = {n: _LocalOnlyPlanner(self.maps[n]) for n in names}

    def plan(self, name, pos, yaw, tick):
        """Entropy-guided hedef seç (güncelleme main döngüde yapılıyor)."""
        heading, rpm, target = self.planners[name].plan(pos, yaw, tick)
        return heading, rpm, target

    def get_coverage_pct(self, name, threshold=0.7):
        return self.maps[name].get_coverage_pct(threshold)

    def reset_target(self, name):
        """Recovery sonrasi hedefi sifirla ki yeni hedef secsin."""
        self.planners[name].current_target = None
        self.planners[name].global_target = None
        self.planners[name]._last_global_replan_tick = -99999


class _LocalOnlyPlanner(EntropyGuidedPlanner):
    """Global magnet'i devre dışı bırakan saf lokal versiyon."""

    def _update_global_target(self, pos, neighbor_grids=None):
        # Baseline için hiçbir şey yapma
        self.global_target = None

    def _hybrid_target(self, pos, yaw_rad, neighbor_grids=None):
        # Sadece lokal greedy
        return self.prob_map.get_next_target(
            pos, current_yaw=yaw_rad, neighbor_grids=neighbor_grids)