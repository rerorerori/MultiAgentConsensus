"""
Consensus-Based Map Fusion — Weighted Average Consensus.
=========================================================
Yenilikler (v2.1):
  • Ağırlıklı consensus: w_ij = f(SNR, mesafe, veri yaşı)
  • Dayanıklılık durum makinesi: CONNECTED → DEGRADED → OFFLINE
  • Offline modda lokal entropi planlamaya geçiş
  • Yeniden bağlanma: harita senkronizasyon protokolü
  • Binary paket encode/decode (Subnero 256B uyumu)
  • DÜZELTİLDİ: RECONNECTING state loop bug'ı
  • DÜZELTİLDİ: sync_on_reconnect/consensus_update çakışması

Referans: Zhang et al. (2023) — Pheromone Matrix Consensus
          Olfati-Saber (2004) — Consensus & Cooperation
"""
# import time  -- wall-clock kaldırıldı, sim_time parametresi kullanılıyor
import struct
import numpy as np
from core.probability_map import ProbabilityMap
from config import (
    AGENT_NAMES, MAP_SIZE_M, MAP_RES,
    CONSENSUS_EPS, CONSENSUS_MAX_CELLS, CONSENSUS_THRESHOLD,
    BEACON_PACK_FMT, BEACON_CELL_FMT,
)

# ─── Dayanıklılık durum makinesi parametreleri ──────────────────
T_DEGRADED_S  = 15.0   # Son mesajdan bu kadar sonra DEGRADED
T_OFFLINE_S   = 40.0   # Son mesajdan bu kadar sonra OFFLINE
T_RECON_S     = 5.0    # Yeniden bağlanma sonrası senkron penceresi

# Durum sabitleri
STATE_CONNECTED    = "CONNECTED"
STATE_DEGRADED     = "DEGRADED"
STATE_OFFLINE      = "OFFLINE"
STATE_RECONNECTING = "RECONNECTING"


class ConsensusMapFusion:
    """Per-agent ProbabilityMap yönetimi + ağırlıklı consensus."""

    def __init__(self, agent_names=None, epsilon=None, comm_manager=None, fixed_weight=False):
        names = agent_names or AGENT_NAMES
        self.epsilon      = epsilon or CONSENSUS_EPS
        self.comm_manager = comm_manager   # SNR ağırlığı için referans
        self.fixed_weight = fixed_weight   # Ablasyon çalışması için sabit ağırlık flag'i

        self.maps          = {}
        self.last_received = {}   # receiver → {sender: grid_patch}
        self._last_rx_t    = {}   # receiver → {sender: sim_time}
        self._comm_state   = {}   # agent → STATE_*
        self._recon_start  = {}   # agent → sim_time
        self._received_waypoints = {name: {} for name in names}

        for name in names:
            self.maps[name]          = ProbabilityMap(MAP_SIZE_M, MAP_RES)
            self.last_received[name] = {}
            self._last_rx_t[name]    = {}
            self._comm_state[name]   = STATE_CONNECTED
            self._recon_start[name]  = 0.0

    # ─────────────────────────────────────────────────────────────
    # Durum makinesi (DÜZELTİLDİ)
    # ─────────────────────────────────────────────────────────────
    def update_comm_state(self, name: str, sim_time: float = 0.0) -> str:
        """
        Haberleşme durumunu güncelle.
        En az bir komşudan alınan son mesajın yaşına göre geçiş yap.

        Args:
            sim_time: Simülasyon zamanı (step / TICKS_PER_SEC). Wall-clock yerine
                      kullanılarak tekrarlanabilir sonuçlar elde edilir.
        """
        ages    = [sim_time - t for t in self._last_rx_t[name].values()]
        min_age = min(ages) if ages else float("inf")
        prev    = self._comm_state[name]

        # FIX: RECONNECTING state'i kendi içinde tutulmalı, geçiş kontrolü ayrı
        if prev == STATE_RECONNECTING:
            # Yeniden bağlanma penceresi doldu mu?
            if sim_time - self._recon_start[name] > T_RECON_S:
                self._comm_state[name] = STATE_CONNECTED
            # else: RECONNECTING'de kal, durum değişmez
            return self._comm_state[name]

        # Diğer durumlar için normal geçişler
        if min_age < T_DEGRADED_S:
            if prev in (STATE_DEGRADED, STATE_OFFLINE):
                # Yeniden bağlandı → RECONNECTING
                self._comm_state[name]  = STATE_RECONNECTING
                self._recon_start[name] = sim_time
            else:
                # CONNECTED → CONNECTED
                self._comm_state[name] = STATE_CONNECTED

        elif min_age < T_OFFLINE_S:
            self._comm_state[name] = STATE_DEGRADED

        else:
            self._comm_state[name] = STATE_OFFLINE

        return self._comm_state[name]

    def get_comm_state(self, name: str) -> str:
        return self._comm_state.get(name, STATE_CONNECTED)

    def is_online(self, name: str) -> bool:
        return self._comm_state.get(name, STATE_CONNECTED) != STATE_OFFLINE

    # ─────────────────────────────────────────────────────────────
    # Lokal güncelleme
    # ─────────────────────────────────────────────────────────────
    def update_local(self, name: str, pos, yaw, sensor_range=None, sidescan_data=None):
        """Sensör verisi ile agent'ın kendi map'ini güncelle."""
        if sidescan_data is not None:
            self.maps[name].bayesian_update_from_sidescan(pos, yaw, sidescan_data)
        else:
            self.maps[name].bayesian_update(pos, yaw, sensor_range)

    # ─────────────────────────────────────────────────────────────
    # Beacon encode / decode — Binary pack (Subnero 256B uyumlu)
    # ─────────────────────────────────────────────────────────────
    def encode_map_for_beacon(self, name: str, current_target=None) -> bytes:
        """
        Map'i binary beacon payload'ına sıkıştır.
        Format: header(3B) + cells(n×5B)
          - header: sender_id(uint8) + n_cells(uint16)
          - cell:   row(uint16) + col(uint16) + prob_u8(uint8)

        prob_u8 = int(prob * 255), 0-255 aralığı, ~%0.4 çözünürlük kaybı.
        """
        grid = self.maps[name].grid
        diff = np.abs(grid - 0.5)
        mask = diff > CONSENSUS_THRESHOLD
        idxs = np.argwhere(mask)
        if len(idxs) == 0:
            return b""

        diffs = diff[mask]
        if len(idxs) > CONSENSUS_MAX_CELLS:
            top  = np.argsort(diffs)[-CONSENSUS_MAX_CELLS:]
            idxs = idxs[top]

        # Binary pack
        try:
            sender_id = AGENT_NAMES.index(name)
        except ValueError:
            sender_id = 0

        n_cells = len(idxs)
        wp_x = float(current_target[0]) if current_target is not None else float('nan')
        wp_y = float(current_target[1]) if current_target is not None else float('nan')
        buf = bytearray(struct.pack(BEACON_PACK_FMT, sender_id, n_cells, wp_x, wp_y))
        for r, c in idxs:
            p_u8 = int(np.clip(grid[r, c] * 255, 0, 255))
            buf.extend(struct.pack(BEACON_CELL_FMT, int(r), int(c), p_u8))

        return bytes(buf)

    def decode_map_from_beacon(self, receiver: str, sender: str, payload,
                               sim_time: float = 0.0) -> bool:
        """
        Beacon'dan gelen hücre güncellemelerini kaydet.
        payload: bytes (binary) veya list (geriye uyumluluk).
        sim_time: Simülasyon zamanı (step / TICKS_PER_SEC).
        Returns: True if decode succeeded
        """
        if not payload:
            return False

        n     = self.maps[receiver].n
        patch = np.full((n, n), np.nan)

        try:
            if isinstance(payload, (bytes, bytearray)):
                # Binary format
                header_sz = struct.calcsize(BEACON_PACK_FMT)
                cell_sz   = struct.calcsize(BEACON_CELL_FMT)
                if len(payload) < header_sz:
                    return False

                import math
                _sid, n_cells, wp_x, wp_y = struct.unpack(BEACON_PACK_FMT, payload[:header_sz])

                if not math.isnan(wp_x) and not math.isnan(wp_y):
                    if receiver not in self._received_waypoints:
                        self._received_waypoints[receiver] = {}
                    self._received_waypoints[receiver][sender] = (wp_x, wp_y, sim_time)

                for i in range(n_cells):
                    offset = header_sz + i * cell_sz
                    if offset + cell_sz > len(payload):
                        break
                    r, c, p_u8 = struct.unpack(
                        BEACON_CELL_FMT, payload[offset:offset + cell_sz])
                    if 0 <= r < n and 0 <= c < n:
                        patch[r, c] = p_u8 / 255.0

            elif isinstance(payload, list):
                # Geriye uyumluluk: eski liste formatı
                for row, col, val in payload:
                    if 0 <= row < n and 0 <= col < n:
                        patch[row, col] = val
            else:
                return False
        except (struct.error, ValueError, TypeError) as e:
            print(f"[CONSENSUS] decode error: {e}", flush=True)
            return False

        self.last_received[receiver][sender] = patch
        self._last_rx_t[receiver][sender]    = sim_time
        return True

    def get_received_waypoints(self, receiver: str, max_age_s: float = 30.0, sim_time: float = 0.0) -> list:
        """
        Komşulardan akustik kanaldan gelen güncel waypoint'leri döndür.
        max_age_s saniyeden eski olanları filtrele (stale bilgi).
        Returns: [(wp_x, wp_y), ...] listesi
        """
        result = []
        rx_dict = self._received_waypoints.get(receiver, {})
        for sender, (wx, wy, t) in rx_dict.items():
            if sim_time - t <= max_age_s:
                result.append((wx, wy))
        return result

    # ─────────────────────────────────────────────────────────────
    # Ağırlıklı consensus güncelleme
    # ─────────────────────────────────────────────────────────────
    def consensus_update(self, name: str, sim_time: float = 0.0) -> bool:
        """
        Ağırlıklı average consensus:
            P_i ← P_i + w_ij · (P_j − P_i)
        w_ij = f(SNR, veri yaşı) — comm_manager üzerinden hesaplanır.

        Args:
            sim_time: Simülasyon zamanı (step / TICKS_PER_SEC).
        """
        self.update_comm_state(name, sim_time=sim_time)
        state   = self._comm_state[name]
        my_grid = self.maps[name].grid
        updated = False

        # OFFLINE'da consensus hiç çalışmasın
        if state == STATE_OFFLINE:
            # Patch'leri de tutma, eski veri birikmesin
            self.last_received[name] = {}
            return False

        for sender, patch in list(self.last_received[name].items()):
            mask = ~np.isnan(patch)
            if not np.any(mask):
                continue

            # Ağırlık hesabı
            if self.comm_manager is not None and not self.fixed_weight:
                w = self.comm_manager.get_edge_weight(sender, name, sim_time=sim_time)
            else:
                w = self.epsilon if state == STATE_CONNECTED else self.epsilon * 0.3

            if state == STATE_DEGRADED:
                # Degraded: yarı ağırlık
                w *= 0.5

            diff = patch[mask] - my_grid[mask]
            my_grid[mask] += w * diff
            updated = True

        if updated:
            self.maps[name].grid = np.clip(my_grid, 0.01, 0.99)

        self.last_received[name] = {}
        return updated

    # ─────────────────────────────────────────────────────────────
    # Yeniden bağlanma senkronizasyonu (DÜZELTİLDİ)
    # ─────────────────────────────────────────────────────────────
    def sync_on_reconnect(self, name: str):
        """
        Yeniden bağlanma sonrası: tüm alınan patch'leri yarı ağırlıkla uygula.
        Uzun süre offline kalmış ajanın haritasını hızlıca güncelleştir.

        FIX: Patch'leri tüketmeden çıkma, consensus_update ile çakışma önlenir.
        """
        my_grid = self.maps[name].grid
        patches = list(self.last_received[name].items())  # snapshot al

        for sender, patch in patches:
            mask = ~np.isnan(patch)
            if np.any(mask):
                # Tam senkron: yarı ağırlıklı ortalama
                my_grid[mask] = 0.5 * my_grid[mask] + 0.5 * patch[mask]

        self.maps[name].grid     = np.clip(my_grid, 0.01, 0.99)
        self.last_received[name] = {}
        # NOT: _comm_state artık update_comm_state() tarafından yönetiliyor

    # ─────────────────────────────────────────────────────────────
    # Metrikler
    # ─────────────────────────────────────────────────────────────
    def get_consensus_rms(self) -> float:
        """Tüm agent map'leri arası ortalama RMS farkı."""
        names = list(self.maps.keys())
        if len(names) < 2:
            return 0.0
        total, count = 0.0, 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                rms = self.maps[names[i]].get_map_rms(self.maps[names[j]].grid)
                total += rms
                count += 1
        return total / count if count else 0.0

    def get_coverage_pct(self, name: str, threshold: float = 0.7) -> float:
        return self.maps[name].get_coverage_pct(threshold)

    def get_avg_coverage_pct(self, threshold: float = 0.7) -> float:
        pcts = [self.maps[n].get_coverage_pct(threshold) for n in self.maps]
        return float(np.mean(pcts))

    def get_comm_states(self) -> dict:
        """Tüm ajanların haberleşme durumunu döndür."""
        return dict(self._comm_state)