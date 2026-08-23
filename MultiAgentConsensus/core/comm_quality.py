"""
Communication Quality Monitor — PDR, gecikme, SNR geçmişi, bilgi yaşı.
=======================================================================
Yenilikler (v2.1):
  • SNR geçmişi per-link (gönderici-alıcı çifti)
  • Kanal durum geçmişi (CONNECTED/DEGRADED/OFFLINE)
  • Ortalama gecikme + maksimum gecikme
  • Dashboard için özet JSON
  • DÜZELTİLDİ: get_pdr() hesap tutarsızlığı (payda = self.sent[name])
"""
# import time  -- wall-clock kaldırıldı, sim_time parametresi kullanılıyor
import numpy as np
from collections import deque


MAX_HISTORY = 500  # Her metrik için tutulacak maksimum örnek sayısı


class CommQualityMonitor:
    """Haberleşme kalite metriklerini izle."""

    def __init__(self, agent_names=None):
        names = agent_names if agent_names is not None else ["auv0", "auv1", "auv2"]
        self.names = names

        # Gönderim/alım sayaçları
        self.sent     = {n: 0 for n in names}
        self.received = {n: 0 for n in names}
        self.dropped  = {n: 0 for n in names}

        # Gecikme geçmişi: receiver → deque[ms]
        self.delays = {n: deque(maxlen=MAX_HISTORY) for n in names}

        # Son alım zamanı: receiver → {sender: sim_time}
        self.last_rx_time = {n: {} for n in names}

        # SNR geçmişi: (sender, receiver) → deque[dB]
        self.snr_history = {}

        # Kanal durum geçmişi: agent → deque[(sim_time, state_str)]
        self.state_history = {n: deque(maxlen=MAX_HISTORY) for n in names}

        # Mesaj ID → gönderim zamanı (gecikme hesabı için)
        self._send_ts = {}

        # Paket iletim oranı geçmişi: deque[(sim_time, pdr)]
        self.pdr_history = deque(maxlen=MAX_HISTORY)

        self._start_sim_time = 0.0

    # ─────────────────────────────────────────────────────────────
    # Kayıt metodları
    # ─────────────────────────────────────────────────────────────
    def log_send(self, sender: str, msg_id=None, sim_time: float = 0.0):
        """Mesaj gönderimini kaydet."""
        if sender in self.sent:
            self.sent[sender] += 1
        if msg_id is not None:
            self._send_ts[msg_id] = sim_time

    def log_receive(self, receiver: str, sender: str, msg_id=None,
                    sim_time: float = 0.0):
        """Mesaj alımını kaydet, gecikme hesapla."""
        if receiver in self.received:
            self.received[receiver] += 1
        if receiver in self.last_rx_time:
            self.last_rx_time[receiver][sender] = sim_time

        if msg_id is not None and msg_id in self._send_ts:
            delay_ms = (sim_time - self._send_ts[msg_id]) * 1000.0
            self.delays[receiver].append(delay_ms)
            del self._send_ts[msg_id]

    def log_drop(self, sender: str):
        """Düşürülen paketi kaydet."""
        if sender in self.dropped:
            self.dropped[sender] += 1

    def log_snr(self, sender: str, receiver: str, snr_db: float):
        """SNR ölçümünü kaydet."""
        key = (sender, receiver)
        if key not in self.snr_history:
            self.snr_history[key] = deque(maxlen=MAX_HISTORY)
        self.snr_history[key].append(snr_db)

    def log_comm_state(self, agent: str, state: str, sim_time: float = 0.0):
        """Kanal durumunu kaydet."""
        if agent in self.state_history:
            self.state_history[agent].append((sim_time, state))

    # ─────────────────────────────────────────────────────────────
    # Hesaplama metodları
    # ─────────────────────────────────────────────────────────────
    def get_pdr(self, name: str) -> float:
        """
        Packet Delivery Ratio (per-agent).

        FIX v2.2: Broadcast çarpanı düzeltmesi.
        Her gönderici broadcast yaptığında N-1 kopya üretir.
        Bu ajanın beklenen alımı = ∑(diğer ajanların gönderdikleri).
        Broadcast'te her send = 1 gönderim (N-1 alıcıya gitse de log_send 1 kez çağrılır)
        ama log_send her hedef için ayrı çağrılıyorsa payda doğrudur.
        """
        expected = sum(self.sent[n] for n in self.names if n != name)
        if expected == 0:
            return 1.0
        return min(1.0, self.received[name] / expected)

    def get_sender_pdr(self, name: str) -> float:
        """
        Gönderici perspektifinden PDR: bu ajan kaç gönderdi, kaçı başarılı?
        Kayıp = dropped[name] ise: PDR = 1 - dropped/sent
        """
        if self.sent[name] == 0:
            return 1.0
        return 1.0 - (self.dropped[name] / self.sent[name])

    def get_global_pdr(self) -> float:
        """Tüm ağ için PDR."""
        total_s = sum(self.sent.values())
        total_r = sum(self.received.values())
        return total_r / max(1, total_s)

    def get_avg_delay(self, name: str) -> float:
        """Ortalama gecikme (ms)."""
        if not self.delays[name]:
            return 0.0
        return float(np.mean(self.delays[name]))

    def get_max_delay(self, name: str) -> float:
        """Maksimum gecikme (ms)."""
        if not self.delays[name]:
            return 0.0
        return float(np.max(self.delays[name]))

    def get_avg_snr(self, sender: str, receiver: str) -> float:
        """Ortalama SNR (dB)."""
        key = (sender, receiver)
        if key not in self.snr_history or len(self.snr_history[key]) == 0:
            return float("nan")
        return float(np.mean(self.snr_history[key]))

    def get_info_age(self, receiver: str, sender: str,
                     sim_time: float = 0.0) -> float:
        """Son alınan bilginin yaşı (saniye). sim_time ile hesaplanır."""
        if sender not in self.last_rx_time.get(receiver, {}):
            return float("inf")
        return sim_time - self.last_rx_time[receiver][sender]

    def get_current_state(self, agent: str) -> str:
        """Ajanın son kaydedilen kanal durumu."""
        hist = self.state_history.get(agent, deque())
        if hist:
            return hist[-1][1]
        return "CONNECTED"

    # ─────────────────────────────────────────────────────────────
    # Özet
    # ─────────────────────────────────────────────────────────────
    def get_summary(self) -> dict:
        """Dashboard için özet metrikler."""
        return {
            "total_sent":       sum(self.sent.values()),
            "total_received":   sum(self.received.values()),
            "total_dropped":    sum(self.dropped.values()),
            "global_pdr":       self.get_global_pdr(),
            "per_agent": {
                n: {
                    "sent":       self.sent[n],
                    "received":   self.received[n],
                    "pdr":        round(self.get_pdr(n), 3),
                    "sender_pdr": round(self.get_sender_pdr(n), 3),
                    "avg_delay":  round(self.get_avg_delay(n), 1),
                    "state":      self.get_current_state(n),
                }
                for n in self.names
            },
        }

    def get_snr_matrix(self) -> dict:
        """Tüm link'lerin ortalama SNR matrisi."""
        result = {}
        for (s, r), hist in self.snr_history.items():
            if len(hist) > 0:
                result[f"{s}→{r}"] = round(float(np.mean(hist)), 1)
        return result