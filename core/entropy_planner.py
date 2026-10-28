"""


Her karar adımında:
1. Global replan zamanı geldiyse → frontier regions bul → en uygununu seç
2. Lokal replan zamanı geldiyse → mevcut greedy entropy + global magnet kombinasyonu
3. Hedefe ulaşıldıysa → yeniden planla
"""
import numpy as np
from config import REPLAN_INTERVAL, MIN_RPM, MAX_RPM, TICKS_PER_SEC, wrap_heading, WP_ACCEPT_R


# ─── Global planner parametreleri ───────────────────────────────
GLOBAL_REPLAN_S     = 25.0   # Global hedefin geçerliliğinin denetlendiği periyot (saniye)
GLOBAL_TARGET_TIMEOUT_S = 120.0  # Bu sürede ulaşılamayan global hedef bırakılır
# Hedefler çalışma alanı sınırından (±170 m) en az bir dönüş yarıçapı (~30 m) içeride tutulur;
# sınıra dayalı bir hedefe dışa doğru yaklaşan araç dönerken alanın dışına taşar
TARGET_LIMIT_M     = 140.0
TARGET_CONFLICT_M  = 80.0    # İki global hedef bundan yakınsa çakışma sayılır
NEIGHBOR_TARGET_MAX_AGE_S = 30.0   # Komşudan gelen hedef bilgisi bu süreden sonra bayat sayılır


class EntropyGuidedPlanner:
    """Entropy tabanlı planlayıcı — global frontier magnet + lokal greedy hibrit."""

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
        self._global_target_t = 0.0
        self._mode = "global"  # Hysteresis için mod takibi
        self.wp_accept_r = WP_ACCEPT_R

    # ─────────────────────────────────────────────────────────────
    # Multi-agent koordinasyon
    # ─────────────────────────────────────────────────────────────
    def _get_other_global_targets(self):
        """Komşuların akustik kanaldan bildirdiği global hedefler: {gönderen: np.array([x, y])}."""
        if self.consensus is None:
            return {}
        return self.consensus.get_received_targets(
            self.agent_name, max_age_s=NEIGHBOR_TARGET_MAX_AGE_S, sim_time=self._last_sim_time)

    def _yields_to_neighbor(self):
        """
        Hedefim, önceliği benden yüksek bir komşunun hedefiyle çakışıyor mu?
        Öncelik araç adının sırasıdır: çakışmada adı küçük olan hedefini korur, diğeri
        yeni hedef seçer. Böylece iki araç aynı anda vazgeçmez ya da ikisi birden ısrar etmez.
        """
        for sender, wp in self._get_other_global_targets().items():
            if sender < self.agent_name and np.linalg.norm(self.global_target - wp) < TARGET_CONFLICT_M:
                return True
        return False

    # ─────────────────────────────────────────────────────────────
    # Global magnet: en büyük frontier'ı bul ve yönel
    # ─────────────────────────────────────────────────────────────
    def _global_target_still_valid(self):
        """Mevcut global hedef hâlâ keşfedilmemiş bir bölgede ve zaman aşımına uğramamış mı?"""
        if self.global_target is None:
            return False
        if self._last_sim_time - self._global_target_t > GLOBAL_TARGET_TIMEOUT_S:
            return False
        if self._yields_to_neighbor():
            return False
        cell = self.prob_map.world_to_grid(self.global_target)
        if cell is None:
            return False
        r, c = int(cell[0]), int(cell[1])
        patch = self.prob_map.get_uncertainty()[max(r - 2, 0):r + 3, max(c - 2, 0):c + 3]
        return float(patch.mean()) > 0.7

    def _update_global_target(self, pos, neighbor_grids=None, yaw=None):
        """
        Global frontier bulup kendine hedef belirle. Geçerli bir hedef varken yenisi
        seçilmez: araç hedefe varmadan hedef değişirse sürekli dönüşte kalır.
        """
        if self._global_target_still_valid():
            return

        avoid = list(self._get_other_global_targets().values())
        new_target = self.prob_map.get_global_entropy_target(
            pos, avoid_positions=avoid,
            neighbor_grids=neighbor_grids,
            min_sep=150.0, current_yaw=yaw)

        # Hard exclusion: komşu AUV bu hedefe zaten gidiyorsa reddet
        if new_target is not None:
            for other_wp in avoid:
                d = np.linalg.norm(new_target - other_wp)
                if d < TARGET_CONFLICT_M:
                    new_target = None
                    break

        # Hard exclusion sonucu None ise: penalty ile en iyi alternatifi ara
        if new_target is None:
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
                score = r["entropy_sum"] * (min_d / TARGET_CONFLICT_M)
                if score > best_score:
                    best_score  = score
                    best_target = candidate
            new_target = best_target  # None olabilir, üst kod halleder

        if new_target is not None:
            new_target = np.clip(new_target, -TARGET_LIMIT_M, TARGET_LIMIT_M)

        old_target = self.global_target
        if new_target is not None:
            self.global_target = new_target
            self._global_target_t = self._last_sim_time
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
            local_tgt = -pos[:2]

        local_tgt = np.clip(local_tgt, -TARGET_LIMIT_M, TARGET_LIMIT_M)

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
            self._update_global_target(pos, neighbor_grids=neighbor_grids, yaw=yaw_rad)
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
        self.current_target = np.array([
            np.clip(self.current_target[0], -TARGET_LIMIT_M, TARGET_LIMIT_M),
            np.clip(self.current_target[1], -TARGET_LIMIT_M, TARGET_LIMIT_M)
        ])
        
        # Klamp sonrası vektörü ve mesafeyi kontrol et
        target_vec = self.current_target - pos[:2]
        distance = np.linalg.norm(target_vec)

        # Eğer klamp sonrası hedef ayağımızın altına düştüyse, bir kez daha planla
        if distance < self.wp_accept_r:
            self.current_target = self._hybrid_target(pos, yaw_rad, neighbor_grids)
            self.current_target = np.array([
                np.clip(self.current_target[0], -TARGET_LIMIT_M, TARGET_LIMIT_M),
                np.clip(self.current_target[1], -TARGET_LIMIT_M, TARGET_LIMIT_M)
            ])
            target_vec = self.current_target - pos[:2]
            distance = max(np.linalg.norm(target_vec), 1.0) # Sıfır mesafe hatasını önle

        # 4) Çıktıları üret
        target_heading = np.degrees(np.arctan2(target_vec[1], target_vec[0]))
        rpm = np.clip(1200 + distance * 5, MIN_RPM, MAX_RPM)

        return wrap_heading(target_heading), int(rpm), self.current_target
