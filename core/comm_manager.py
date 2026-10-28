"""
Communication Manager — Beacon gönderim/alım orkestratörü.
=========================================================
Yenilikler (v2.1):
  • Thorp attenuation + SNR tabanlı mesafe-bağımlı PDR
  • Yayılım gecikmesi AKTIF (tick gecikme kuyruğu + flush_delayed)
  • TDMA çoklu erişim — çakışma önleme
  • Ağırlıklı consensus için SNR ağırlığı dışa aktarımı
  • HMM kanal değişkenliği (packet_loss_sim'den)
  • Binary payload desteği (Subnero 256B uyumu)
  • Broadcast (tüm komşulara) — önceki round-robin tek-hedef düzeltildi
  • Public SNR matrix API (_snr_cache private erişimi kaldırıldı)
  • Çift kayıp modeli açıklandı
"""
# import time  -- wall-clock kaldırıldı, sim_time parametresi kullanılıyor
import random
import json
import math
import numpy as np
from collections import deque
try:
    from config_current import AGENT_NAMES, BEACON_SEND_INTERVAL, PACKET_LOSS_RATE, TICKS_PER_SEC
except ImportError:
    from config import AGENT_NAMES, BEACON_SEND_INTERVAL, PACKET_LOSS_RATE, TICKS_PER_SEC


# ─── Thorp Kanal Sabitleri ──────────────────────────────────────
CARRIER_FREQ_KHZ  = 25.0     # Subnero M25M carrier
BANDWIDTH_HZ      = 4096     # ~4 kHz
SOURCE_LEVEL_DB   = 170.0    # Tx gücü dB re μPa
NOISE_LEVEL_DB    = 60.0     # Ortam gürültüsü dB re μPa
SPREADING_FACTOR  = 1.5      # Pratik yayılım üssü
SNR_THRESHOLD_DB  = 10.0     # Minimum kabul edilebilir SNR
TAU_STALE_S       = 30.0     # Kaç saniye sonra veri "bayat" sayılır
MAX_RANGE_M       = 1500.0   # Menzil eşiği
RANGE_SIGMA_M     = 1.0      # Tek yönlü uçuş süresinden mesafe ölçümü gürültüsü (1-sigma)

# TDMA slot'ları: agent başına farklı gönderim zamanları
TDMA_OFFSETS = {"auv0": 0, "auv1": 100, "auv2": 200}


def thorp_absorption(freq_khz: float) -> float:
    """Thorp formülü — dB/km."""
    f = freq_khz
    return (0.11 * f**2 / (1 + f**2)
            + 44 * f**2 / (4100 + f**2)
            + 2.75e-4 * f**2 + 0.003)


def transmission_loss(distance_m: float) -> float:
    """TL = k·10·log10(d) + α(f)·d/1000  [dB]"""
    d = max(1.0, distance_m)
    spreading = SPREADING_FACTOR * 10.0 * math.log10(d)
    absorption = thorp_absorption(CARRIER_FREQ_KHZ) * d / 1000.0
    return spreading + absorption


def compute_snr(distance_m: float) -> float:
    """SNR = SL − TL − NL − 10·log10(BW)  [dB]"""
    tl = transmission_loss(distance_m)
    noise = NOISE_LEVEL_DB + 10.0 * math.log10(BANDWIDTH_HZ)
    return SOURCE_LEVEL_DB - tl - noise


def snr_to_pdr(snr_db: float) -> float:
    """SNR → PDR — Parçalı doğrusal Rician yaklaşımı."""
    if snr_db > 20:   return 0.95
    if snr_db > 15:   return 0.80 + 0.15 * (snr_db - 15) / 5
    if snr_db > 10:   return 0.50 + 0.30 * (snr_db - 10) / 5
    if snr_db > 5:    return 0.20 + 0.30 * (snr_db - 5) / 5
    if snr_db > 0:    return 0.02 + 0.18 * snr_db / 5
    return 0.02


class RealisticChannel:
    """Fizik tabanlı akustik kanal modeli (Thorp yayılımı + Gilbert-Elliott sönümleme)."""
    def __init__(self, ticks_per_sec=30):
        self.ticks_per_sec = ticks_per_sec
        self.state = "GOOD"
        self.p_good_to_bad = 0.02
        self.p_bad_to_good = 0.20

    def compute_pdr(self, dist_m: float) -> float:
        snr = compute_snr(dist_m)
        return snr_to_pdr(snr)

    def should_deliver(self, dist_m: float, current_tick: int = 0) -> bool:
        if self.state == "GOOD":
            if random.random() < self.p_good_to_bad:
                self.state = "BAD"
        else:
            if random.random() < self.p_bad_to_good:
                self.state = "GOOD"

        base_pdr = self.compute_pdr(dist_m)
        if self.state == "BAD":
            base_pdr *= 0.4
        return random.random() < base_pdr


class DelayedMessage:
    """Kanal modelinden geçmiş, bir sonraki adımda HoloOcean'a verilecek mesaj."""
    __slots__ = ["deliver_at_tick", "from_id", "to_id", "payload"]

    def __init__(self, deliver_at_tick, from_id, to_id, payload):
        self.deliver_at_tick = deliver_at_tick
        self.from_id = from_id
        self.to_id   = to_id
        self.payload = payload


class CommManager:
    """Akustik beacon haberleşme yöneticisi — Thorp + TDMA + gecikme kuyruğu."""

    def __init__(self, agent_names=None, send_interval=None, loss_rate=None):
        self.names = agent_names or AGENT_NAMES
        self.send_interval = send_interval or BEACON_SEND_INTERVAL
        self.loss_rate = loss_rate if loss_rate is not None else PACKET_LOSS_RATE

        # Beacon ID eşleştirmesi
        self.id_map     = {name: i for i, name in enumerate(self.names)}
        self.rev_id_map = {i: name for name, i in self.id_map.items()}

        # Round-robin gönderici indeksi (kullanılmıyor ama korundu)
        self._rr_idx = 0

        # Fizik tabanlı kanal (HMM + Thorp)
        self.channel = RealisticChannel(ticks_per_sec=TICKS_PER_SEC)

        # Yayılım gecikmesi kuyruğu: global FIFO (deliver_at_tick sıralı değil, her tick taranır)
        self._delay_queue = deque()

        # Ajanların gerçek konumları: kanal fiziği (SNR, kayıp, gecikme, mesafe) bunlarla hesaplanır
        self._agent_positions = {}
        # Gönderim anındaki gönderici konumu: (sender, receiver) → konum (mesafe ölçümü için)
        self._tx_positions = {}

        # İstatistik
        self.channel_busy_drops = 0
        self.total_sent    = 0
        self.total_dropped = 0
        self.total_delivered = 0

        # SNR geçmişi: (sender, receiver) → son SNR değeri
        self._snr_cache = {}
        # Son alım zamanı: receiver → {sender: sim_time}
        self._last_rx_time = {n: {} for n in self.names}

    # ─────────────────────────────────────────────────────────────
    # Konum güncelleme
    # ─────────────────────────────────────────────────────────────
    def update_positions(self, positions: dict):
        """Her tick'te ajan konumlarını güncelle — {name: np.array([x,y,z])}."""
        self._agent_positions = positions

    def update_ids(self, new_map: dict):
        self.id_map     = new_map
        self.rev_id_map = {i: name for name, i in self.id_map.items()}

    # ─────────────────────────────────────────────────────────────
    # SNR & ağırlık hesabı (consensus için dışa aktarım)
    # ─────────────────────────────────────────────────────────────
    def get_snr(self, sender: str, receiver: str) -> float:
        """İki ajan arasındaki anlık SNR (dB). Konum bilinmiyorsa NaN."""
        p1 = self._agent_positions.get(sender)
        p2 = self._agent_positions.get(receiver)
        if p1 is None or p2 is None:
            return float("nan")
        dist = float(np.linalg.norm(np.array(p1)[:3] - np.array(p2)[:3]))
        snr  = compute_snr(dist)
        self._snr_cache[(sender, receiver)] = snr
        return snr

    def get_snr_matrix(self) -> dict:
        """
        Tüm link'ler için anlık SNR matrisi (public API).
        Format: {"auv0→auv1": 15.2, ...}
        """
        result = {}
        for (s, r), snr in self._snr_cache.items():
            if not math.isnan(snr):
                result[f"{s}→{r}"] = round(snr, 1)
        return result

    def get_edge_weight(self, sender: str, receiver: str,
                        sim_time: float = 0.0) -> float:
        """
        Consensus kenar ağırlığı ∈ [0, CONSENSUS_EPS].
        w = ε · σ(SNR) · exp(−age / τ)

        Args:
            sim_time: Simülasyon zamanı (step / TICKS_PER_SEC).
        """
        from config import CONSENSUS_EPS
        snr = self.get_snr(sender, receiver)
        if math.isnan(snr):
            return 0.0

        # Veri yaşı: hiç mesaj gelmediyse ağırlık sıfır
        last_t = self._last_rx_time[receiver].get(sender, None)
        if last_t is None:
            return 0.0
        age = sim_time - last_t

        # SNR sigmoid ağırlığı
        snr_w = 1.0 / (1.0 + math.exp(-0.3 * (snr - SNR_THRESHOLD_DB)))
        # Yaş cezası
        age_w = math.exp(-age / TAU_STALE_S)

        return float(np.clip(CONSENSUS_EPS * snr_w * age_w, 0.0, CONSENSUS_EPS))


    def should_send_tdma(self, name: str, tick: int) -> bool:
        """
        TDMA çoklu erişim: her ajanın kendi zaman dilimi.
        Çakışma önlemek için TDMA_OFFSETS kullanılır.
        """
        offset = TDMA_OFFSETS.get(name, 0)
        t = tick - offset
        return t > 0 and t % self.send_interval == 0


    # ─────────────────────────────────────────────────────────────
    # Mesaj hazırlama — BROADCAST (tüm komşulara)
    # ─────────────────────────────────────────────────────────────
    def prepare_messages(self, sender, payload, agent_positions=None, current_tick=None):
        """
        Mesaj paketlerini hazırla — TÜM komşulara broadcast.

        FIX: Önceki round-robin tek-hedef stratejisi consensus yakınsamasını
        yavaşlatıyordu. TDMA zaten sender çakışmalarını önlüyor, her sender
        kendi slot'unda tüm komşulara broadcast yapmalı.
        """
        if sender not in self.id_map:
            return []
        sender_id = self.id_map[sender]
        positions = agent_positions or self._agent_positions
        msgs = []

        for tgt in self.names:
            if tgt == sender or tgt not in self.id_map:
                continue
            target_id = self.id_map[tgt]

            # Mesafe kontrolü: menzil dışı komşuları atla
            p1 = positions.get(sender)
            p2 = positions.get(tgt)
            if p1 is not None and p2 is not None:
                dist = float(np.linalg.norm(np.array(p1)[:3] - np.array(p2)[:3]))
                if dist > MAX_RANGE_M:
                    continue

            # Payload sarmalama: sender_id'yi payload'a ekle
            # Binary payload ise dokunma, dict/liste ise sarmala
            if isinstance(payload, (bytes, bytearray)):
                wrapped = payload  # binary: zaten sender_id header'da
            else:
                wrapped = {"f": sender_id, "p": payload}

            msgs.append((sender_id, target_id, wrapped))

        return msgs

    # ─────────────────────────────────────────────────────────────
    # Mesaj gönderme — Gecikme kuyruğuna ekler
    # ─────────────────────────────────────────────────────────────
    def send(self, env, from_agent: str, to_id: int, payload, current_tick: int = 0) -> bool:
        """
        Mesajı kanal kayıp modelinden geçirip gecikme kuyruğuna ekle.

        FIX: Anlık teslim yerine propagation delay modellendi.
        Kanal kayıp modelinde başarısız olan mesajlar kuyruğa eklenmez.
        """
        self.total_sent += 1
        try:
            from_id = self.id_map[from_agent]

            # Busy beacon kontrolü
            try:
                from holoocean.sensors import AcousticBeaconSensor
                if from_id in AcousticBeaconSensor.instances:
                    beacon = AcousticBeaconSensor.instances[from_id]
                    if getattr(beacon, "status", None) == "Transmitting":
                        self.channel_busy_drops += 1
                        self.total_dropped += 1
                        return False
            except (ImportError, KeyError, AttributeError):
                pass

            # ─── Çift kayıp modeli (açıklamalı) ────────────────────
            # 1) Sabit loss_rate (varsa): deneysel kontrol için
            # 2) Thorp+HMM (her zaman): fiziksel kanal modeli
            # İkisi bağımsız Bernoulli denemeleri: effective_PDR = (1-loss_rate)·thorp_pdr

            if self.loss_rate > 0 and random.random() < self.loss_rate:
                self.total_dropped += 1
                return False

            # Mesafe hesabı (Thorp + kuyruk gecikmesi için gerekli)
            to_name = self.rev_id_map.get(to_id)
            dist = None
            if to_name is not None:
                p1 = self._agent_positions.get(from_agent)
                p2 = self._agent_positions.get(to_name)
                if p1 is not None and p2 is not None:
                    dist = float(np.linalg.norm(np.array(p1)[:3] - np.array(p2)[:3]))

            # Thorp + HMM fizik kayıp modeli
            if dist is not None:
                if not self.channel.should_deliver(dist, current_tick):
                    self.total_dropped += 1
                    return False

            # ─── Gecikme kuyruğuna ekle ────────────────────────────
            # Yayılım gecikmesini HoloOcean'ın AcousticBeaconSensor'ü uygular (mesafe / ses hızı);
            # burada tekrar bekletilirse gecikme iki kez sayılır
            deliver_at  = current_tick

            # Payload'ı gönderilebilir forma dönüştür
            # Binary bytes → base64 benzeri encoding gerekir, ama HoloOcean
            # acoustic message string bekliyor; biz bytes'ı hex ile taşıyoruz
            if isinstance(payload, (bytes, bytearray)):
                data_to_send = payload.hex()  # hex string: 2x boyut ama güvenli
            elif isinstance(payload, str):
                data_to_send = payload
            else:
                data_to_send = json.dumps(payload)

            self._delay_queue.append(
                DelayedMessage(deliver_at, from_id, to_id, data_to_send))
            tx_pos = self._agent_positions.get(from_agent)
            if to_name is not None and tx_pos is not None:
                self._tx_positions[(from_agent, to_name)] = np.array(tx_pos, dtype=float)[:3].copy()
            return True

        except Exception as e:
            print(f"[CommManager] send exception: {e}", flush=True)
            return False

    def flush_delayed(self, env, current_tick: int) -> int:
        """
        Gecikme kuyruğundaki teslim zamanı gelen mesajları HoloOcean'a gönder.
        Her tick'te main loop'tan çağrılmalı.

        Returns: teslim edilen mesaj sayısı
        """
        due = {}
        remaining = deque()
        while self._delay_queue:
            msg = self._delay_queue.popleft()
            if msg.deliver_at_tick <= current_tick:
                due.setdefault((msg.from_id, msg.payload), []).append(msg.to_id)
            else:
                remaining.append(msg)
        self._delay_queue = remaining

        # Bir beacon aynı anda tek iletim yapabilir: aynı paket hedef başına ayrı gönderilirse
        # ikinci gönderim "still transmitting" ile reddedilir. Alıcılar tek yayında toplanır.
        delivered = 0
        for (from_id, payload), to_ids in due.items():
            try:
                env.send_acoustic_message(from_id, to_ids, "OWAY", payload)
                delivered += len(to_ids)
                self.total_delivered += len(to_ids)
            except Exception:
                self.total_dropped += len(to_ids)
        return delivered

    # ─────────────────────────────────────────────────────────────
    # Mesaj alma & ayrıştırma
    # ─────────────────────────────────────────────────────────────
    def _measure_range(self, sender: str, receiver: str):
        """
        Tek yönlü uçuş süresinden eğik mesafe (senkron saat varsayımı): göndericinin
        gönderim anındaki konumu ile alıcının alım anındaki konumu arası + gürültü.
        """
        tx_pos = self._tx_positions.get((sender, receiver))
        rx_pos = self._agent_positions.get(receiver)
        if tx_pos is None or rx_pos is None:
            return None
        dist = float(np.linalg.norm(np.array(rx_pos, dtype=float)[:3] - tx_pos))
        return dist + random.gauss(0.0, RANGE_SIGMA_M)

    def parse_beacon_data(self, sensor_data, receiver: str = None,
                          sim_time: float = 0.0) -> list:
        """
        AcousticBeaconSensor verisi → parse edilmiş mesaj listesi.
        Hem binary (hex-encoded) hem JSON payload destekler.

        Args:
            sim_time: Simülasyon zamanı (step / TICKS_PER_SEC).
        """
        if sensor_data is None:
            return []

        if isinstance(sensor_data, list) and len(sensor_data) > 0:
            raw_messages = (sensor_data
                            if isinstance(sensor_data[0], list)
                            else [sensor_data])
        else:
            return []

        parsed_list = []
        for msg in raw_messages:
            if len(msg) < 3:
                continue
            msg_type, holo_id, raw_payload = msg[0], msg[1], msg[2]
            payload = raw_payload
            real_from_id = holo_id

            # 1) Binary hex string → bytes
            if isinstance(raw_payload, str):
                # Hex string mi? (sadece hex karakterleri ve çift uzunluk)
                is_hex = (len(raw_payload) % 2 == 0 and
                          len(raw_payload) >= 6 and
                          all(c in "0123456789abcdefABCDEF" for c in raw_payload))
                if is_hex:
                    try:
                        payload = bytes.fromhex(raw_payload)
                        # Binary payload: sender_id header'da (consensus.decode çözer)
                        # İlk byte sender_id
                        if len(payload) >= 1:
                            real_from_id = payload[0]
                    except ValueError:
                        pass

                # 2) JSON fallback
                if isinstance(payload, str):
                    try:
                        payload = json.loads(raw_payload)
                    except (json.JSONDecodeError, TypeError):
                        pass

            # JSON dict sarmalaması (eski format)
            if isinstance(payload, dict) and "f" in payload and "p" in payload:
                real_from_id = payload["f"]
                payload = payload["p"]

            sender_name = self.rev_id_map.get(real_from_id, f"id{real_from_id}")

            # Alım zamanını kaydet (yaş hesabı için) — sim_time kullan
            if receiver and receiver in self._last_rx_time:
                self._last_rx_time[receiver][sender_name] = sim_time

            parsed_list.append({
                "type":      msg_type,
                "from_id":   real_from_id,
                "from_name": sender_name,
                "payload":   payload,
                "rx_time":   sim_time,
                "range":     self._measure_range(sender_name, receiver),
            })

        return parsed_list
