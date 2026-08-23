"""


Her karar adımında:
1. Global replan zamanı geldiyse → frontier regions bul → en uygununu seç
2. Lokal replan zamanı geldiyse → mevcut greedy entropy + global magnet kombinasyonu
3. Hedefe ulaşıldıysa → yeniden planla
"""
import numpy as np
from config import REPLAN_INTERVAL, MIN_RPM, MAX_RPM, TICKS_PER_SEC, wrap_heading, WP_ACCEPT_R


# ─── Global planner parametreleri ───────────────────────────────
GLOBAL_REPLAN_S     = 25.0   # Global hedef yenileme periyodu (saniye)
GLOBAL_WEIGHT       = 0.85   # %85 global + %15 lokal
GLOBAL_MAGNET_RANGE = 300.0  # Menzil artırıldı


class EntropyGuidedPlanner:
    """Entropy tabanlı planlayıcı — global frontier magnet + lokal greedy hibrit."""

    # Sınıf-seviyesi kayıt: tüm planner'ların global hedefleri (multi-agent avoid)
    _global_targets_registry = {}

    @classmethod
    def reset_registry(cls):
        """Tüm run'lar arası paylaşılan hedef registry'yi sıfırla.
        Her simülasyon başlangıcında çağrılmalı."""
        cls._global_targets_registry = {}

    def __init__(self, prob_map, replan_interval=None, agent_name=None, consensus=None):
        self.prob_map = prob_map
        self.consensus = consensus
        self._last_sim_time = 0.0
        self.replan_interval = replan_interval or REPLAN_INTERVAL
        self.global_replan_ticks = int(GLOBAL_REPLAN_S * TICKS_PER_SEC)
        self.current_target = None
        self.global_target  = None
        self.agent_name     = agent_name or f"agent_{id(self)}"
        self._last_global_replan_tick = -self.global_replan_ticks
        self._mode = "global"  # Hysteresis için mod takibi
        self.wp_accept_r = WP_ACCEPT_R
        # Önceki run'dan kalma kaydı temizle
        EntropyGuidedPlanner._global_targets_registry.pop(self.agent_name, None)

    # ─────────────────────────────────────────────────────────────
    # Multi-agent koordinasyon
    # ─────────────────────────────────────────────────────────────
    def _register_global_target(self):
        """Bu AUV'un global hedefini registry'ye kaydet."""
        if self.global_target is not None:
            if abs(self.global_target[0]) < 170.0 and abs(self.global_target[1]) < 170.0:
                EntropyGuidedPlanner._global_targets_registry[self.agent_name] = (
                    float(self.global_target[0]), float(self.global_target[1]))

    def _get_other_global_targets(self):
        """Diğer AUV'ların global hedefleri (kendimiz hariç)."""
        # Önce akustik kanaldan gelen waypoint'leri dene
        if getattr(self, 'consensus', None) is not None:
            received = self.consensus.get_received_waypoints(
                self.agent_name,
                max_age_s=30.0,
                sim_time=self._last_sim_time)
            if received:
                return received
            # Komşulardan hiç paket gelmemişse (başlangıç) fallback
        # Fallback: shared memory registry
        return [pos for name, pos in
                EntropyGuidedPlanner._global_targets_registry.items()
                if name != self.agent_name]

    # ─────────────────────────────────────────────────────────────
    # Global magnet: en büyük frontier'ı bul ve yönel
    # ─────────────────────────────────────────────────────────────
    def _update_global_target(self, pos, neighbor_grids=None):
        """Global frontier bulup kendine hedef belirle."""
        avoid = self._get_other_global_targets()
        new_target = self.prob_map.get_global_entropy_target(
            pos, avoid_positions=avoid,
            neighbor_grids=neighbor_grids,
            min_sep=150.0)

        # Hard exclusion: komşu AUV bu hedefe zaten gidiyorsa reddet
        if new_target is not None:
            for other_wp in self._get_other_global_targets():
                d = np.linalg.norm(new_target - np.array(other_wp))
                if d < 80.0:   # 80m içinde aynı hedefe gitme
                    new_target = None
                    break

        # Hard exclusion sonucu None ise: penalty ile en iyi alternatifi ara
        if new_target is None:
            avoid = self._get_other_global_targets()
            regions = self.prob_map.find_frontier_regions(
                min_size=8, uncertainty_threshold=0.7)
            best_score  = -np.inf
            best_target = None
            for r in regions:
                n_w, e_w = r["centroid"]
                candidate = np.array([n_w, e_w])
                # En az uzak olan komşu hedefe mesafe
                min_d = min(
                    [np.linalg.norm(candidate - np.array(av))
                     for av in avoid], default=999.0)
                score = r["entropy_sum"] * (min_d / 80.0)
                if score > best_score:
                    best_score  = score
                    best_target = candidate
            new_target = best_target  # None olabilir, üst kod halleder

        SAFE_MARGIN = 155.0
        if new_target is not None:
            new_target[0] = np.clip(new_target[0], -SAFE_MARGIN, SAFE_MARGIN)
            new_target[1] = np.clip(new_target[1], -SAFE_MARGIN, SAFE_MARGIN)

        old_target = self.global_target
        if new_target is not None:
            self.global_target = new_target
            self._register_global_target()
            # Global hedef değiştiyse veya ilk kez atandıysa, mevcut hedefi sıfırla (Yeni rota çiz)
            if old_target is None or np.linalg.norm(new_target - old_target) > 50.0:
                self.current_target = None
                self._mode = "global"

    # ─────────────────────────────────────────────────────────────
    # Hibrit hedef: lokal greedy + global magnet
    # ─────────────────────────────────────────────────────────────
    def _hybrid_target(self, pos, yaw_rad, neighbor_grids=None):
        """
        Lokal entropy hedefi ile global frontier magnet'i birleştir.
        Eğer global hedef uzaksa → ağırlıklı ortalama ile oraya doğru çek.
        """
        local_tgt = self.prob_map.get_next_target(
            pos, current_yaw=yaw_rad, neighbor_grids=neighbor_grids)
        if local_tgt is None:
            # AUV harita disi — ters yonde harita icine geri don
            local_tgt = np.clip(-pos[:2], -155.0, 155.0)

        SAFE_MARGIN = 155.0
        local_tgt[0] = np.clip(local_tgt[0], -SAFE_MARGIN, SAFE_MARGIN)
        local_tgt[1] = np.clip(local_tgt[1], -SAFE_MARGIN, SAFE_MARGIN)

        if self.global_target is None:
            return local_tgt

        # Hysteresis (Histerezis) Mantığı: Salınımı önle
        d_global = np.linalg.norm(self.global_target - pos[:2])
        if d_global < self.wp_accept_r:
            self.global_target = None
            self._last_global_replan_tick = -self.global_replan_ticks
            self._mode = "local"
            return local_tgt

        # Mod geçişleri
        if self._mode == "global" and d_global < 60.0:
            self._mode = "local"
        elif self._mode == "local" and d_global > 100.0:
            self._mode = "global"

        if self._mode == "global":
            return self.global_target
        else:
            return local_tgt

    # ─────────────────────────────────────────────────────────────
    # Ana plan fonksiyonu
    # ─────────────────────────────────────────────────────────────
    def plan(self, pos, yaw, tick, neighbor_grids=None, sim_time=0.0):
        """
        Döndürür: (heading_deg, rpm, target)

        yaw: radyan cinsinden
        """
        self._last_sim_time = sim_time
        yaw_rad = yaw

        # 1) Global replan (periyodik)
        if tick - self._last_global_replan_tick >= self.global_replan_ticks:
            self._update_global_target(pos, neighbor_grids=neighbor_grids)
            self._last_global_replan_tick = tick

        # 2) Replan Kararı: Sadece hedef yoksa yeni plan yap (Lokal periyodik replan İPTAL)
        should_replan = False
        if self.current_target is None:
            should_replan = True

        if should_replan:
            self.current_target = self._hybrid_target(pos, yaw_rad, neighbor_grids)

        # 3) Hedef vektörü ve varış kontrolü
        target_vec = self.current_target - pos[:2]
        distance = np.linalg.norm(target_vec)

        if distance < self.wp_accept_r:
            # Hedefe varıldı, hemen yeni bir tane seç
            self.current_target = self._hybrid_target(pos, yaw_rad, neighbor_grids)
            target_vec = self.current_target - pos[:2]
            distance = np.linalg.norm(target_vec)

        # --- KRİTİK: HER DURUMDA KLAMP (Kaçış Yok) ---
        MAP_HALF = 155.0
        self.current_target = np.array([
            np.clip(self.current_target[0], -MAP_HALF, MAP_HALF),
            np.clip(self.current_target[1], -MAP_HALF, MAP_HALF)
        ])
        
        # Klamp sonrası vektörü ve mesafeyi kontrol et
        target_vec = self.current_target - pos[:2]
        distance = np.linalg.norm(target_vec)

        # Eğer klamp sonrası hedef ayağımızın altına düştüyse, bir kez daha planla
        if distance < self.wp_accept_r:
            self.current_target = self._hybrid_target(pos, yaw_rad, neighbor_grids)
            self.current_target = np.array([
                np.clip(self.current_target[0], -MAP_HALF, MAP_HALF),
                np.clip(self.current_target[1], -MAP_HALF, MAP_HALF)
            ])
            target_vec = self.current_target - pos[:2]
            distance = max(np.linalg.norm(target_vec), 1.0) # Sıfır mesafe hatasını önle

        # 4) Çıktıları üret
        target_heading = np.degrees(np.arctan2(target_vec[1], target_vec[0]))
        rpm = np.clip(1200 + distance * 5, MIN_RPM, MAX_RPM)

        return wrap_heading(target_heading), int(rpm), self.current_target

    # ─────────────────────────────────────────────────────────────
    # Diagnostik: dashboard için
    # ─────────────────────────────────────────────────────────────
    def get_global_target(self):
        """Dashboard'un global hedefi görselleştirmesi için."""
        return self.global_target.copy() if self.global_target is not None else None