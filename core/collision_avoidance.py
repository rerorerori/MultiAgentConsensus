"""
Araçlar arası çarpışmadan kaçınma — öngörülen yollar arasındaki en yakın geçişe dayalı baş açısı düzeltmesi
==========================================================================================================
Fayda/ağırlık yapısı MOOS-IvP'nin BHV_AvoidCollision davranışından uyarlanmıştır; öngörü
kısmı bu sisteme özgüdür:

  * Komşular yalnızca akustik beacon'lardan bilinir (yaklaşık 10 s'de bir): konum, baş açısı
    ve gitmekte olduğu hedef. Rapor eskidikçe temas bırakılır.
  * Komşunun yolu, hedefine dönüş hızı sınırıyla yönelen bir araç olarak ileri doğru benzetilir.
    Araçlar 10 s'de 60 dereceden fazla dönebildiği için "komşu düz gider" varsayımı (klasik
    CPA) dönüşlü karşılaşmalarda yanılır ve kaçınmayı geç başlatır; komşunun niyeti (hedefi)
    bilindiğinde dönüşü de öngörülebilir.
  * Kendi aday baş açılarının her biri için de dönüş hızı sınırlı yol benzetilir ve iki yol
    arasındaki en küçük mesafe (CPA) bulunur. CPA `MIN_UTIL_CPA_M` altındaysa aday değersiz,
    `MAX_UTIL_CPA_M` üstündeyse tam değerlidir.
  * Kaçınmanın ağırlığı komşuya olan mesafeyle artar: `PWT_OUTER_M` ötesinde sıfır,
    `PWT_INNER_M` içinde tam. Güdümün istediği baş açısına yakınlık ayrı bir fayda terimidir;
    ikisinin ağırlıklı toplamını en büyük yapan aday seçilir.
  * Roller: her çiftte adı sözlük sırasında büyük olan araç "yol veren"dir ve manevrayı o
    yapar. Diğer araç yolunu korur; böylece yol verenin öngörüsü geçerli kalır ve iki aracın
    aynı anda aynı yana kırıp yan yana kalması önlenir. Yolunu koruyan araç yalnızca komşu
    `STAND_ON_RANGE_M` içine girerse (son çare) kendisi de kaçınır.

Güdüm yönteminden bağımsız bir katmandır: planlayıcının (entropi veya şerit tarama) ürettiği
baş açısını girdi alır, gerekirse değiştirir.
"""
import math

import numpy as np

from config import wrap_heading

# Eşikler bilerek dar tutulur: 400 m'lik alanda geniş eşikler gereksiz kaçınmaya ve görevin
# uzamasına yol açar.
PWT_OUTER_M      = 60.0    # bu mesafenin ötesindeki komşu dikkate alınmaz
PWT_INNER_M      = 25.0    # bu mesafenin içinde kaçınma tam ağırlıkla uygulanır
MIN_UTIL_CPA_M   = 5.0     # CPA bunun altındaysa aday baş açısının faydası sıfır
MAX_UTIL_CPA_M   = 18.0    # CPA bunun üstündeyse fayda tam (bu geçişlere karışılmaz)
STAND_ON_RANGE_M = 25.0    # yolunu koruyan araç ancak bu mesafenin içinde kaçınır
CONTACT_STALE_S  = 35.0    # bundan eski rapor kullanılmaz

# Yol öngörüsü
PREDICT_HORIZON_S = 30.0   # yollar bu kadar ileri benzetilir
PREDICT_DT_S      = 1.0
MAX_YAW_RATE_RPS  = 0.10   # araçların ulaşabildiği dönüş hızı (rad/s)
TARGET_REACHED_M  = 15.0   # komşu hedefine bu kadar yaklaşınca sonrası bilinmez: düz devam varsayılır
NOMINAL_SPEED_MPS = 2.3    # sürati kestirilemeyen komşu için varsayılan
SPEED_LIMITS_MPS  = (1.5, 3.0)
SPEED_MAX_DT_S    = 25.0   # sürat kestirimi için iki rapor arasındaki en uzun süre

PWT_GUIDANCE     = 100.0   # güdümün istediği baş açısına yakınlığın ağırlığı
PWT_AVOID        = 250.0   # kaçınmanın (tam ağırlıktaki) ağırlığı
STARBOARD_BIAS   = 0.5     # eşitlikte sancağa dönüşü seç
KEEP_SIDE_BONUS  = 5.0     # bir kez seçilen kaçınma yönünü sürdür (kararsızlığı önler)
HEADING_OFFSETS_DEG = np.arange(-120.0, 121.0, 10.0)
ACTIVE_OFFSET_DEG = 5.0    # bundan büyük düzeltme "kaçınıyor" sayılır

_STEPS = int(round(PREDICT_HORIZON_S / PREDICT_DT_S))


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def predict_pursuit(pos, yaw, speed, target, duration, dt=PREDICT_DT_S):
    """
    Hedefine dönüş hızı sınırıyla yönelen aracın yolu: `duration` saniye, `dt` adımlarla.
    Döndürür: (konumlar [(n+1) x 2], son yaw, kalan hedef). Hedef yoksa veya hedefe varıldıysa
    araç baş açısını korur.
    """
    n = max(0, int(round(duration / dt)))
    x, y = float(pos[0]), float(pos[1])
    path = np.empty((n + 1, 2))
    path[0] = (x, y)
    for k in range(n):
        if target is not None:
            dx, dy = target[0] - x, target[1] - y
            if dx * dx + dy * dy <= TARGET_REACHED_M ** 2:
                target = None
            else:
                err = _wrap(math.atan2(dy, dx) - yaw)
                yaw += max(-MAX_YAW_RATE_RPS * dt, min(MAX_YAW_RATE_RPS * dt, err))
        x += speed * math.cos(yaw) * dt
        y += speed * math.sin(yaw) * dt
        path[k + 1] = (x, y)
    return path, yaw, target


def predict_candidates(pos, yaw, speed, headings, steps=_STEPS, dt=PREDICT_DT_S):
    """Her aday baş açısına dönüş hızı sınırıyla dönen aracın yolları: [aday x (steps+1) x 2]."""
    headings = np.asarray(headings, dtype=float)
    yaws = np.full(headings.shape, float(yaw))
    p = np.tile(np.asarray(pos[:2], dtype=float), (headings.size, 1))
    paths = np.empty((headings.size, steps + 1, 2))
    paths[:, 0] = p
    max_step = MAX_YAW_RATE_RPS * dt
    for k in range(steps):
        err = (headings - yaws + np.pi) % (2.0 * np.pi) - np.pi
        yaws = yaws + np.clip(err, -max_step, max_step)
        p = p + speed * dt * np.column_stack([np.cos(yaws), np.sin(yaws)])
        paths[:, k + 1] = p
    return paths


class CollisionAvoidance:
    """Bir aracın komşularına ilişkin temas kayıtlarını tutar ve baş açısını düzeltir."""

    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.contacts = {}           # komşu → {"pos", "yaw", "speed", "target", "t"}
        self.active_with = None      # o anda kaçınılan komşu (yoksa None)
        self.min_cpa = np.inf        # son adımda, seçilen baş açısıyla öngörülen en küçük CPA
        self._last_side = 0.0

    def update_contact(self, name, pos, sim_time, target=None, heading=None):
        """
        Komşunun beacon'da bildirdiği durumu işler.
        heading: komşunun baş açısı (radyan). Verilmezse ardışık iki rapordan, o da yoksa
        hedef yönünden kestirilir.
        """
        pos = np.asarray(pos[:2], dtype=float)
        target = None if target is None else np.asarray(target[:2], dtype=float)
        prev = self.contacts.get(name)
        speed, moved_yaw = NOMINAL_SPEED_MPS, None
        if prev is not None and 1.0 <= sim_time - prev["t"] <= SPEED_MAX_DT_S:
            delta = pos - prev["pos"]
            speed = float(np.clip(np.linalg.norm(delta) / (sim_time - prev["t"]), *SPEED_LIMITS_MPS))
            if np.linalg.norm(delta) > 1.0:
                moved_yaw = math.atan2(delta[1], delta[0])
        if heading is None:
            heading = moved_yaw
        if heading is None and target is not None:
            heading = math.atan2(target[1] - pos[1], target[0] - pos[0])
        self.contacts[name] = {"pos": pos, "yaw": heading, "speed": speed, "target": target, "t": sim_time}

    def _threats(self, own, sim_time):
        """Dikkate alınacak komşular: (ad, şimdiden itibaren öngörülen yol, ağırlık)."""
        threats = []
        for name, c in self.contacts.items():
            age = sim_time - c["t"]
            if not 0.0 <= age <= CONTACT_STALE_S or c["yaw"] is None:
                continue
            # Rapor anından şimdiye, sonra ufka kadar tek benzetim
            now_path, yaw, target = predict_pursuit(c["pos"], c["yaw"], c["speed"], c["target"], age)
            rng = float(np.linalg.norm(now_path[-1] - own))
            weight = float(np.clip((PWT_OUTER_M - rng) / (PWT_OUTER_M - PWT_INNER_M), 0.0, 1.0))
            gives_way = self.agent_name > name
            if weight > 0.0 and (gives_way or rng < STAND_ON_RANGE_M):
                path, _, _ = predict_pursuit(now_path[-1], yaw, c["speed"], target, PREDICT_HORIZON_S)
                threats.append((name, path, weight))
        return threats

    def adjust(self, pos, yaw, desired_heading_deg, speed, sim_time):
        """
        Güdümün istediği baş açısını çarpışma riskine göre düzeltir.
        yaw: aracın şimdiki baş açısı (radyan). Döndürür: baş açısı (derece).
        Kaçınma etkinse `active_with` komşunun adıdır.
        """
        own = np.asarray(pos[:2], dtype=float)
        threats = self._threats(own, sim_time)
        if not threats:
            self.active_with, self.min_cpa, self._last_side = None, np.inf, 0.0
            return wrap_heading(desired_heading_deg)

        own_paths = predict_candidates(own, yaw, max(float(speed), 0.5),
                                       np.radians(desired_heading_deg + HEADING_OFFSETS_DEG))
        score = PWT_GUIDANCE * (1.0 - np.abs(HEADING_OFFSETS_DEG) / 180.0)
        score = score + STARBOARD_BIAS * (HEADING_OFFSETS_DEG < 0)      # NWU'da negatif yaw = sancak
        if self._last_side != 0.0:
            score = score + KEEP_SIDE_BONUS * (np.sign(HEADING_OFFSETS_DEG) == self._last_side)
        worst_cpa = np.full(HEADING_OFFSETS_DEG.shape, np.inf)
        worst_name = np.full(HEADING_OFFSETS_DEG.shape, None, dtype=object)
        for name, path, weight in threats:
            cpa = np.linalg.norm(own_paths - path[None, :, :], axis=2).min(axis=1)
            utility = np.clip((cpa - MIN_UTIL_CPA_M) / (MAX_UTIL_CPA_M - MIN_UTIL_CPA_M), 0.0, 1.0)
            score = score + PWT_AVOID * weight * utility
            closer = cpa < worst_cpa
            worst_cpa = np.where(closer, cpa, worst_cpa)
            worst_name[closer] = name

        best = int(np.argmax(score))
        offset = float(HEADING_OFFSETS_DEG[best])
        self.min_cpa = float(worst_cpa[best])
        if abs(offset) > ACTIVE_OFFSET_DEG:
            self.active_with, self._last_side = worst_name[best], float(np.sign(offset))
        else:
            self.active_with, self._last_side = None, 0.0
        return wrap_heading(desired_heading_deg + offset)
