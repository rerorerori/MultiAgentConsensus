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
import math
import struct
import numpy as np
from core.probability_map import ProbabilityMap
from config import (
    AGENT_NAMES, MAP_SIZE_M, MAP_RES,
    CONSENSUS_EPS, BEACON_PACK_FMT, BEACON_BLOCK,
)

COVERED_THRESHOLD = 0.7    # hücre bu olasılığın üstündeyse taranmış sayılır
COVERED_VALUE     = 0.99   # komşunun taranmış bildirdiği hücrenin olasılığı
HEADING_UNKNOWN   = -32768 # beacon'da baş açısı bildirilmedi

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
        self._received_nav = {name: {} for name in names}   # receiver → {sender: (konum, std, sim_time)}
        self._received_heading = {name: {} for name in names}   # receiver → {sender: (yaw [rad], sim_time)}

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
    def encode_map_for_beacon(self, name: str, current_target=None,
                              nav_pos=None, nav_std=None, nav_yaw=None) -> bytes:
        """
        Kapsama haritasını beacon payload'ına sıkıştır (256 baytlık modem paketine sığar).
        Format: header + bit haritası
          - header: sender_id(uint8) + n_bytes(uint16) + hedef(wp_x, wp_y)
                    + gönderenin kestirilen konumu (nav_x, nav_y, nav_z) ve belirsizliği (nav_std)
                    + gönderenin baş açısı (int16, 0.01 derece; verilmezse HEADING_UNKNOWN)
          - bit haritası: harita BEACON_BLOCK x BEACON_BLOCK hücrelik bloklara bölünür; bloğun
                    bütün hücreleri taranmışsa biti 1'dir (80x80 hücre -> 40x40 bit = 200 bayt)
        Her paket haritanın tamamının anlık görüntüsüdür: kaybolan paketi bir sonraki telafi eder.
        nav_pos / nav_std verilmezse NaN yazılır (mesafe güncellemesi yapılmaz).
        """
        covered = self.maps[name].grid > COVERED_THRESHOLD
        nb = covered.shape[0] // BEACON_BLOCK
        blocks = covered[:nb * BEACON_BLOCK, :nb * BEACON_BLOCK].reshape(
            nb, BEACON_BLOCK, nb, BEACON_BLOCK).all(axis=(1, 3))
        bitmap = np.packbits(blocks).tobytes()

        try:
            sender_id = AGENT_NAMES.index(name)
        except ValueError:
            sender_id = 0

        wp_x = float(current_target[0]) if current_target is not None else float('nan')
        wp_y = float(current_target[1]) if current_target is not None else float('nan')
        nav = [float(v) for v in nav_pos[:3]] if nav_pos is not None else [float('nan')] * 3
        std = float(nav_std) if nav_std is not None else float('nan')
        yaw_cdeg = HEADING_UNKNOWN if nav_yaw is None else int(round(
            math.degrees(math.atan2(math.sin(nav_yaw), math.cos(nav_yaw))) * 100.0))
        header = struct.pack(BEACON_PACK_FMT, sender_id, len(bitmap), wp_x, wp_y, *nav, std, yaw_cdeg)
        return header + bitmap

    def decode_map_from_beacon(self, receiver: str, sender: str, payload,
                               sim_time: float = 0.0) -> bool:
        """
        Beacon'dan gelen hedef, konum ve kapsama bit haritasını kaydet.
        Gönderenin taranmış bildirdiği hücreler COVERED_VALUE ile, diğerleri NaN ile
        doldurulmuş bir yama olarak saklanır; konsensüs yalnızca NaN olmayan hücrelere uygulanır.
        Returns: True if decode succeeded
        """
        if not isinstance(payload, (bytes, bytearray)):
            return False
        header_sz = struct.calcsize(BEACON_PACK_FMT)
        if len(payload) < header_sz:
            return False

        try:
            _sid, n_bytes, wp_x, wp_y, nav_x, nav_y, nav_z, nav_std, yaw_cdeg = struct.unpack(
                BEACON_PACK_FMT, payload[:header_sz])
        except struct.error as e:
            print(f"[CONSENSUS] decode error: {e}", flush=True)
            return False

        if not math.isnan(wp_x) and not math.isnan(wp_y):
            self._received_waypoints[receiver][sender] = (wp_x, wp_y, sim_time)
        if not math.isnan(nav_x) and not math.isnan(nav_std):
            self._received_nav[receiver][sender] = (
                np.array([nav_x, nav_y, nav_z]), nav_std, sim_time)
        if yaw_cdeg != HEADING_UNKNOWN:
            self._received_heading[receiver][sender] = (math.radians(yaw_cdeg / 100.0), sim_time)

        n  = self.maps[receiver].n
        nb = n // BEACON_BLOCK
        bits = np.unpackbits(np.frombuffer(payload[header_sz:header_sz + n_bytes], dtype=np.uint8))
        if bits.size < nb * nb:
            return False
        blocks = bits[:nb * nb].reshape(nb, nb).astype(bool)
        covered = np.zeros((n, n), dtype=bool)
        covered[:nb * BEACON_BLOCK, :nb * BEACON_BLOCK] = np.kron(
            blocks, np.ones((BEACON_BLOCK, BEACON_BLOCK), dtype=bool))

        self.last_received[receiver][sender] = np.where(covered, COVERED_VALUE, np.nan)
        self._last_rx_t[receiver][sender]    = sim_time
        return True

    def get_received_nav(self, receiver: str, sender: str, sim_time: float):
        """sender'ın bu sim_time'da alınan beacon'ında bildirdiği (konum, std); yoksa None."""
        entry = self._received_nav[receiver].get(sender)
        if entry is None or entry[2] != sim_time:
            return None
        return entry[0], entry[1]

    def get_received_heading(self, receiver: str, sender: str, sim_time: float):
        """sender'ın bu sim_time'da alınan beacon'ında bildirdiği baş açısı (radyan); yoksa None."""
        entry = self._received_heading[receiver].get(sender)
        return entry[0] if entry is not None and entry[1] == sim_time else None

    def get_received_targets(self, receiver: str, max_age_s: float = 30.0, sim_time: float = 0.0) -> dict:
        """
        Komşulardan akustik kanaldan gelen güncel hedefler: {gönderen: np.array([x, y])}.
        max_age_s saniyeden eski olanlar (bayat bilgi) atılır.
        """
        return {sender: np.array([wx, wy])
                for sender, (wx, wy, t) in self._received_waypoints.get(receiver, {}).items()
                if sim_time - t <= max_age_s}

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

    def get_union_coverage_pct(self, threshold: float = 0.7) -> float:
        """En az bir aracın haritasında taranmış görünen hücrelerin oranı (sürünün gerçek kapsaması)."""
        union = np.max([m.grid for m in self.maps.values()], axis=0)
        return float(np.sum(union > threshold) / union.size * 100.0)

    def get_avg_coverage_pct(self, threshold: float = 0.7) -> float:
        pcts = [self.maps[n].get_coverage_pct(threshold) for n in self.maps]
        return float(np.mean(pcts))

    def get_comm_states(self) -> dict:
        """Tüm ajanların haberleşme durumunu döndür."""
        return dict(self._comm_state)