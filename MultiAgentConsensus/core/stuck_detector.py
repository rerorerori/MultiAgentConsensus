"""
StuckDetector — AUV sıkışma ve zigzag tespit & kurtarma modülü.
"""
import numpy as np

# Varsayılan sabitler (config'den felan override edilebilir)
ZIGZAG_WINDOW     = 60     # Son kaç tick'lik heading geçmişi tutulacak
ZIGZAG_VAR_THRESH = 800.0   # Heading varyansı bu değeri geçerse zigzag
ZIGZAG_COOLDOWN   = 30.0   # Zigzag kaçışından sonra bekleme süresi (s)
RECOVERY_HEADING_OFF = 180.0


class StuckDetector:
    def __init__(self, agent_names=None, cruise_depth=15.0,
                 stuck_time_s=15.0, stuck_dist_m=2.0,
                 startup_grace_s=30.0, post_recovery_grace_s=15.0,
                 recovery_dur_s=5.0):
        self.agent_names = agent_names if agent_names else ["auv0", "auv1", "auv2"]
        self.cruise_depth = cruise_depth
        self.stuck_time_s = stuck_time_s
        self.stuck_dist_m = stuck_dist_m
        self.startup_grace_s = startup_grace_s
        self.post_recovery_grace_s = post_recovery_grace_s
        self.recovery_dur_s = recovery_dur_s

        self.status           = {n: "OK" for n in self.agent_names}
        self.last_pos         = {n: np.zeros(3) for n in self.agent_names}
        self.last_move_t      = {n: 0.0 for n in self.agent_names}
        self.recovery_start   = {n: 0.0 for n in self.agent_names}
        self.recovery_heading = {n: 0.0 for n in self.agent_names}
        self.recovery_depth   = {n: self.cruise_depth for n in self.agent_names}
        self.start_sim_time   = 0.0

        # Zigzag tespiti
        self.heading_history = {n: [] for n in self.agent_names}
        self.last_escape_t   = {n: -9999.0 for n in self.agent_names}

    def check_stuck(self, name, pos, sim_time):
        if sim_time - self.start_sim_time < self.startup_grace_s:
            return False
        dist = float(np.linalg.norm(pos[:2] - self.last_pos[name][:2]))
        if dist > self.stuck_dist_m:
            self.last_pos[name]    = pos.copy()
            self.last_move_t[name] = sim_time
        return (sim_time - self.last_move_t[name] > self.stuck_time_s
                and self.status[name] == "OK")

    def update_heading(self, name, hdg):
        """Her tick heading geçmişini güncelle."""
        if name not in self.heading_history:
            return
        h = self.heading_history[name]
        h.append(hdg)
        if len(h) > ZIGZAG_WINDOW:
            h.pop(0)

    def check_zigzag(self, name, sim_time):
        """Son N tick'te heading varyansı yüksekse zigzag tespiti."""
        if name not in self.status or self.status[name] != "OK":
            return False
        if sim_time - self.last_escape_t[name] < ZIGZAG_COOLDOWN:
            return False
        if sim_time - self.start_sim_time < self.startup_grace_s * 2:
            return False
        h = self.heading_history[name]
        if len(h) < ZIGZAG_WINDOW:
            return False
        h_rad = np.radians(h)
        R = float(np.sqrt(np.mean(np.sin(h_rad)) ** 2 + np.mean(np.cos(h_rad)) ** 2))
        circular_var = (1.0 - R) * 10000.0
        return circular_var > ZIGZAG_VAR_THRESH

    def mark_escape(self, name, sim_time):
        self.last_escape_t[name] = sim_time
        print(f"[ZIGZAG] {name} kaçış hedefi seçildi", flush=True)

    def enter_recovery(self, name, hdg, depth, sim_time):
        self.status[name]         = "RECOVERING"
        self.recovery_start[name] = sim_time
        agent_idx = self.agent_names.index(name) if name in self.agent_names else 0
        offset = RECOVERY_HEADING_OFF + agent_idx * 120.0
        self.recovery_heading[name] = ((hdg + offset + 180.0) % 360.0) - 180.0
        self.recovery_depth[name]   = self.cruise_depth

    def is_done(self, name, sim_time):
        return sim_time - self.recovery_start[name] > self.recovery_dur_s

    def finish_recovery(self, name, pos, sim_time):
        self.status[name]      = "OK"
        self.last_pos[name]    = pos.copy()
        self.last_move_t[name] = sim_time + self.post_recovery_grace_s
