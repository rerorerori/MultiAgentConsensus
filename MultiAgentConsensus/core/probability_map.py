"""
Target Probability Map (TPM) — Bayesian Coverage Grid.
Referans: Elfes (1989), Zhang et al. (2023), Zhan et al. (2024).

Her AUV kendi lokal TPM'ini tutar. Hücre değeri:
  0.01 = kesinlikle boş
  0.50 = belirsiz (başlangıç prior)
  0.99 = kesinlikle taranmış

Yenilikler (v3.2):
  • Dinamik effective_min_dist: coverage arttıkça minimum hedef mesafesi azalır
  • get_coverage_pct() döngü dışına alındı (performans)
  • _pick_subregion_target() fallback: karşı köşe stratejisi eklendi
  • get_next_target() fallback zinciri güçlendirildi
"""
import numpy as np
from config import (
    MAP_SIZE_M, MAP_RES, MAP_INIT_P,
    SENSOR_RANGE, P_DETECT, BAYESIAN_PTS,
    ENTROPY_DIST_DECAY, MIN_TARGET_DIST,
)


def _effective_min_dist(coverage_pct, base_dist):
    """
    Coverage yüzdесine göre dinamik minimum hedef mesafesi.
    Coverage arttıkça boş alanlar küçülür, kısıt gevşer.
    """
    if   coverage_pct < 30: return base_dist          # 90m
    elif coverage_pct < 50: return base_dist * 0.6    # 54m
    elif coverage_pct < 70: return base_dist * 0.4    # 36m
    else:                   return base_dist * 0.2    # 18m


class ProbabilityMap:
    """AUV başına lokal coverage probability map."""

    def __init__(self, size_m=None, resolution=None):
        sz  = size_m or MAP_SIZE_M
        res = resolution or MAP_RES
        self.size_m = sz
        self.res    = res
        self.n      = int(sz / res)
        self.grid   = np.full((self.n, self.n), MAP_INIT_P)
        self.origin = np.array([sz / 2, sz / 2])

    # ─────────────────────────────────────────────
    # Koordinat dönüşümleri
    # ─────────────────────────────────────────────
    def world_to_grid(self, pos):
        gp = ((pos[:2] + self.origin) / self.res).astype(int)
        # Harita siniri disinda ise None domdur — clip yapma!
        # Eski davranis (clip) alan-disi pozisyonlari kenar hucresine yapistiriyordu
        # ve yapay coverage sisiyor du.
        if np.any(gp < 0) or np.any(gp >= self.n):
            return None
        return gp

    def grid_to_world(self, n_idx, e_idx):
        return np.array([
            n_idx * self.res - self.origin[0] + self.res / 2,
            e_idx * self.res - self.origin[1] + self.res / 2,
        ])

    # ─────────────────────────────────────────────
    # Bayesian güncelleme
    # ─────────────────────────────────────────────
    def bayesian_update(self, pos, yaw, sensor_range=None, p_detect=None):
        sr    = sensor_range or SENSOR_RANGE
        pd    = p_detect    or P_DETECT
        n_pts = BAYESIAN_PTS

        side_vec = np.array([-np.sin(yaw), np.cos(yaw)])
        fwd_vec  = np.array([ np.cos(yaw), np.sin(yaw)])

        for fwd_d in [0, sr * 0.15, sr * 0.3]:
            for dist in np.linspace(-sr, sr, n_pts):
                world_pos    = pos[:2] + fwd_vec * fwd_d + side_vec * dist
                result       = self.world_to_grid(world_pos)
                if result is None:          # harita siniri disinda, atla
                    continue
                n_idx, e_idx = result
                prior        = self.grid[n_idx, e_idx]
                abs_dist     = np.sqrt(fwd_d**2 + dist**2)
                dist_factor  = max(0.8, 1.0 - abs_dist / (sr * 2.5))
                p_d_eff      = pd * dist_factor
                posterior    = (p_d_eff * prior) / (
                                p_d_eff * prior + (1 - p_d_eff) * (1 - prior))
                self.grid[n_idx, e_idx] = np.clip(posterior, 0.01, 0.99)

    def bayesian_update_from_sidescan(self, pos, yaw, sidescan_data):
        """
        Gerçek SidescanSonar verisiyle Bayesian güncelleme (Coverage Map).
        Sensör: Azimuth=170° (her iki yan), RangeMax=75m, RangeBins=512
        sidescan_data: 1D array [range_bins]
        """
        RANGE_MAX  = 75.0    # JSON'daki RangeMax (m)
        range_bins = len(sidescan_data)
        range_res  = RANGE_MAX / max(1, range_bins)

        # AUV'a dik vektör (NED): soldaki +side_vec, sağdaki -side_vec
        side_vec = np.array([-np.sin(yaw), np.cos(yaw)])

        for i, intensity in enumerate(sidescan_data):
            dist = (i + 0.5) * range_res            # bin merkezi
            
            # Coverage (taranma) olasılığı. Akustik yoğunluk (intensity) engeli belirtir,
            # taranma durumunu değil. Bu yüzden sabit sensör güvenilirliği (P_DETECT) kullanıyoruz.
            # Uzaklığa göre hafif bir güvenilirlik düşüşü (dist_factor) eklendi.
            dist_factor = max(0.8, 1.0 - dist / (RANGE_MAX * 1.5))
            p_detect = P_DETECT * dist_factor
            
            # Not: Eğer ileride akustik gölgelenme (shadowing) eklenmek istenirse, 
            # yüksek intensity sonrası p_detect = 0.5 (belirsiz) yapılabilir.

            for sign in [+1.0, -1.0]:               # her iki yana simetrik
                world_pos = pos[:2] + sign * side_vec * dist
                result = self.world_to_grid(world_pos)
                if result is None:          # harita siniri disinda, atla
                    continue
                n_idx, e_idx = result
                prior = self.grid[n_idx, e_idx]
                posterior = (p_detect * prior) / (
                    p_detect * prior + (1 - p_detect) * (1 - prior))
                self.grid[n_idx, e_idx] = np.clip(posterior, 0.01, 0.99)

    # ─────────────────────────────────────────────
    # Entropy
    # ─────────────────────────────────────────────
    def get_uncertainty(self):
        p = self.grid
        return -p * np.log2(p + 1e-10) - (1 - p) * np.log2(1 - p + 1e-10)

    def get_coverage_pct(self, threshold=0.7):
        return (np.sum(self.grid > threshold) / self.grid.size) * 100

    def to_dict(self, max_cells=48):
        """Taranmış / güncellenmiş hücreleri dict olarak döndür."""
        rows, cols = np.where(abs(self.grid - 0.5) > 0.1)
        if len(rows) == 0:
            return {}
        n_sel = min(max_cells, len(rows))
        idx   = np.random.choice(len(rows), n_sel, replace=False)
        return {
            (int(rows[i]), int(cols[i])): float(self.grid[rows[i], cols[i]])
            for i in idx
        }

    # ─────────────────────────────────────────────
    # Global frontier tespiti
    # ─────────────────────────────────────────────
    def find_frontier_regions(self, min_size=8, uncertainty_threshold=0.7,
                              neighbor_grids=None):
        try:
            from scipy.ndimage import label
        except ImportError:
            return self._find_frontier_fallback(uncertainty_threshold)

        unc           = self.get_uncertainty()
        frontier_mask = unc > uncertainty_threshold

        if neighbor_grids:
            for ng in neighbor_grids.values():
                frontier_mask &= ~(ng > 0.7)

        if not np.any(frontier_mask):
            return []

        labeled, num_features = label(frontier_mask)
        if num_features == 0:
            return []

        regions = []
        for comp_id in range(1, num_features + 1):
            comp_mask = labeled == comp_id
            size      = int(np.sum(comp_mask))
            if size < min_size:
                continue

            ys, xs  = np.where(comp_mask)
            weights = unc[ys, xs]
            total_w = weights.sum()
            if total_w < 1e-6:
                cy, cx = float(np.mean(ys)), float(np.mean(xs))
            else:
                cy = float(np.sum(ys * weights) / total_w)
                cx = float(np.sum(xs * weights) / total_w)

            world = self.grid_to_world(int(round(cy)), int(round(cx)))
            regions.append({
                "centroid":    (float(world[0]), float(world[1])),
                "size":        size,
                "entropy_sum": float(total_w),
                "grid_center": (cy, cx),
            })

        regions.sort(key=lambda r: r["entropy_sum"], reverse=True)
        return regions

    def _find_frontier_fallback(self, threshold=0.7):
        unc  = self.get_uncertainty()
        mask = unc > threshold
        if not np.any(mask):
            return []
        ys, xs = np.where(mask)
        cy, cx = float(np.mean(ys)), float(np.mean(xs))
        world  = self.grid_to_world(int(round(cy)), int(round(cx)))
        return [{
            "centroid":    (float(world[0]), float(world[1])),
            "size":        int(mask.sum()),
            "entropy_sum": float(unc[mask].sum()),
            "grid_center": (cy, cx),
        }]

    # ─────────────────────────────────────────────
    # Global entropi hedef seçimi
    # ─────────────────────────────────────────────
    def get_global_entropy_target(self, current_pos, avoid_positions=None,
                                  neighbor_grids=None, min_sep=80.0):
        regions = self.find_frontier_regions(
            min_size=8,
            uncertainty_threshold=0.7,
            neighbor_grids=neighbor_grids)

        if not regions:
            return None

        avoid = avoid_positions or []

        if len(regions) == 1 and regions[0]["size"] > 500:
            return self._pick_subregion_target(
                current_pos, avoid, neighbor_grids, min_sep)

        # Coverage bir kez hesapla — döngü dışında
        coverage  = self.get_coverage_pct()
        eff_min   = _effective_min_dist(coverage, MIN_TARGET_DIST)

        best_score  = -np.inf
        best_target = None

        for r in regions:
            n_w, e_w = r["centroid"]
            d_self   = np.linalg.norm(np.array([n_w, e_w]) - current_pos[:2])

            # Dinamik minimum mesafe filtresi
            if d_self < eff_min:
                continue

            avoid_penalty = 0.0
            for av in avoid:
                d_av = np.linalg.norm(np.array([n_w, e_w]) - np.array(av))
                if d_av < min_sep:
                    avoid_penalty += np.exp(-(d_av / min_sep) * 2) * 50.0

            score = r["entropy_sum"] / (1 + d_self / 200.0) - avoid_penalty
            if score > best_score:
                best_score  = score
                best_target = np.array([n_w, e_w])

        # Tüm frontier'lar filtrelendiyse subregion'a düş
        if best_target is None:
            return self._pick_subregion_target(
                current_pos, avoid, neighbor_grids, min_sep)

        return best_target

    # ─────────────────────────────────────────────
    # Alt-bölge hedef seçimi
    # ─────────────────────────────────────────────
    def _pick_subregion_target(self, current_pos, avoid_positions,
                               neighbor_grids, min_sep):
        unc           = self.get_uncertainty()
        frontier_mask = unc > 0.7

        if neighbor_grids:
            for ng in neighbor_grids.values():
                frontier_mask &= ~(ng > 0.7)

        if not np.any(frontier_mask):
            return None

        result_c = self.world_to_grid(current_pos[:2])
        if result_c is None:
            return None  # AUV harita disinda, hedef secilemiyor
        n_c, e_c    = result_c
        yi, xi      = np.mgrid[0:self.n, 0:self.n]
        d_self_grid = np.sqrt((yi - n_c)**2 + (xi - e_c)**2) * self.res

        # Komşu AUV kaçınma ağırlığı
        if avoid_positions:
            d_others = np.full((self.n, self.n), np.inf)
            for av in avoid_positions:
                av_grid = self.world_to_grid(np.array(av))
                if av_grid is not None:  # None kontrolu (world_to_grid v2)
                    d_this   = np.sqrt(
                        (yi - av_grid[0])**2 + (xi - av_grid[1])**2) * self.res
                    d_others = np.minimum(d_others, d_this)
            avoid_weight = np.clip(d_others / min_sep, 0.1, 1.0)
        else:
            avoid_weight = np.ones_like(d_self_grid)

        # Dinamik minimum mesafe
        coverage    = self.get_coverage_pct()
        eff_min     = _effective_min_dist(coverage, MIN_TARGET_DIST)

        dist_weight = np.exp(-d_self_grid / ENTROPY_DIST_DECAY)
        dist_weight[d_self_grid < eff_min] = 0.0

        score = unc * dist_weight * avoid_weight
        score = np.where(frontier_mask, score, 0.0)

        if score.max() < 1e-6:
            # Fallback: minimum mesafe kısıtı olmadan tekrar dene
            dist_weight2 = np.exp(-d_self_grid / ENTROPY_DIST_DECAY)
            score2 = unc * dist_weight2 * avoid_weight
            score2 = np.where(frontier_mask, score2, 0.0)
            if score2.max() < 1e-6:
                return None
            idx    = np.unravel_index(np.argmax(score2), score2.shape)
            return self.grid_to_world(idx[0], idx[1])

        idx    = np.unravel_index(np.argmax(score), score.shape)
        target = self.grid_to_world(idx[0], idx[1])
        return target

    # ─────────────────────────────────────────────
    # Lokal greedy hedef seçimi
    # ─────────────────────────────────────────────
    def get_next_target(self, current_pos, current_yaw=None, min_dist=None,
                        neighbor_grids=None):
        md          = min_dist or MIN_TARGET_DIST
        uncertainty = self.get_uncertainty()

        result_c = self.world_to_grid(current_pos[:2])
        if result_c is None:
            return None
        n_c, e_c   = result_c
        yi, xi     = np.mgrid[0:self.n, 0:self.n]
        dist_grid  = np.sqrt((yi - n_c)**2 + (xi - e_c)**2) * self.res

        # Dinamik minimum mesafe
        coverage    = self.get_coverage_pct()
        eff_min     = _effective_min_dist(coverage, md)

        dist_weight = np.exp(-dist_grid / ENTROPY_DIST_DECAY)
        dist_weight[dist_grid < eff_min] = 0.0

        # İleri yön ağırlığı
        frontal_weight = np.ones_like(dist_weight)
        if current_yaw is not None:
            target_yaws    = np.arctan2(xi - e_c, yi - n_c)
            diff           = np.arctan2(
                np.sin(target_yaws - current_yaw),
                np.cos(target_yaws - current_yaw))
            frontal_weight = np.where(np.abs(diff) < np.pi / 2, 1.0, 0.1)

        score = uncertainty * dist_weight * frontal_weight

        # Kendi taranmış bölgeden kaçın
        score *= (1.0 - 0.9 * (self.grid > 0.7).astype(float))

        if neighbor_grids:
            for ng in neighbor_grids.values():
                score *= (1.0 - 0.5 * (ng > 0.7).astype(float))

        # Fallback zinciri
        if score.max() < 1e-9:
            # 1. Adım: minimum mesafe kısıtı olmadan dene
            dist_weight2 = np.exp(-dist_grid / ENTROPY_DIST_DECAY)
            score2 = uncertainty * dist_weight2 * frontal_weight
            score2 *= (1.0 - 0.9 * (self.grid > 0.7).astype(float))
            if score2.max() > 1e-9:
                idx = np.unravel_index(np.argmax(score2), score2.shape)
                return self.grid_to_world(idx[0], idx[1])

            # 2. Adım: haritanın karşı köşesine git
            corner = np.array([-current_pos[0], -current_pos[1]])
            return np.clip(corner, -170.0, 170.0)

        idx    = np.unravel_index(np.argmax(score), score.shape)
        target = self.grid_to_world(idx[0], idx[1])
        return target

    def get_map_rms(self, other_grid):
        return np.sqrt(np.mean((self.grid - other_grid) ** 2))