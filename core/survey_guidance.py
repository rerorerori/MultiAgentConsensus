"""
Şerit tarama (lawnmower) güdümü — entropi planlayıcıyla karşılaştırma için klasik yöntem
========================================================================================
Görev öncesi sabit plan: alan paralel şeritlere bölünür, şeritler araçlara bitişik
bloklar halinde paylaştırılır. Araçlar arası haberleşmeye ihtiyaç duymaz.

Güdüm kuralları MOOS-IvP'nin BHV_Waypoint / BHV_LegRun davranışlarından uyarlanmıştır:
  * Hat takibi : araç sıradaki noktaya değil, önceki-sonraki nokta hattı üzerinde kendi
                 dik izdüşümünün `lead` metre ilerisine yönelir; hatta `lead_damper`'dan
                 yakınken bu mesafe uzatılarak hattın öbür yanına taşma azaltılır.
  * Varış      : yakalama yarıçapı, "slip" ölçütü (mesafe artmaya başladıysa ve en yakın
                 geçiş slip yarıçapı içindeyse) veya noktadan geçen dik çizginin aşılması.
  * Dönüşler   : şeritler alanın dışına taşırılır; araç şerit sonundan doğrudan sonraki
                 şeridin hattına yönelir, dönüş alan dışında ve düşük devirle yapılır.
                 Şerit aralığı dönüş çapından darsa şeritler atlamalı sıralanır.

Koordinatlar NWU'dur (x = kuzey, y = batı); şeritler kuzey-güney doğrultusundadır.
EntropyGuidedPlanner ile aynı arayüzü sunar: plan() → (baş açısı [derece], devir, hedef).
"""
import numpy as np

from config import MAP_SIZE_M, MIN_RPM, MAX_RPM, wrap_heading

# ─── Tarama deseni ──────────────────────────────────────────────
SWATH_HALF_M     = 75.0    # yan taramalı sonarın tek yana menzili (senaryodaki RangeMax)
LANE_OVERLAP_M   = 20.0    # komşu şeritlerin örtüşmesi (seyrüsefer hatası ve akıntı payı)
LANE_OVERRUN_M   = 30.0    # şeridin alan dışına taşırılan kısmı (dönüş ve hatta oturma payı)
TURN_RADIUS_M    = 30.0    # aracın yaklaşık dönüş yarıçapı

# ─── Hat takibi ve varış ölçütleri ──────────────────────────────
LEAD_M           = 25.0
LEAD_DAMPER_M    = 6.0
CAPTURE_RADIUS_M = 8.0
SLIP_RADIUS_M    = 25.0
SLIP_HYST_M      = 1.0     # kestirim gürültüsünün slip ölçütünü erken tetiklememesi için

# ─── Hız ────────────────────────────────────────────────────────
LANE_RPM = MAX_RPM
TURN_RPM = MIN_RPM + 300


def lane_offsets(width, spacing, swath_half=SWATH_HALF_M):
    """Genişliği `width` olan alanı kaplayan, merkeze göre simetrik şerit konumları."""
    n = max(1, int(np.ceil((width - 2.0 * swath_half) / spacing - 1e-9)) + 1)
    return (np.arange(n) - (n - 1) / 2.0) * spacing


def lane_visit_order(n, spacing, turn_radius=TURN_RADIUS_M):
    """Şerit aralığı dönüş çapından darsa şeritleri atlamalı sıralar (0, m, 2m, ..., 1, m+1, ...)."""
    stride = max(1, int(np.ceil(2.0 * turn_radius / spacing - 1e-9)))
    if stride == 1 or n <= stride:
        return list(range(n))
    return [i for start in range(stride) for i in range(start, n, stride)]


class SurveyGuidance:
    """Bir aracın şerit tarama rotasını üretir ve hat takibiyle izler."""

    def __init__(self, agent_name, slot, n_agents, lane_spacing=None, area_size=None):
        """slot: aracın batı-doğu sıralamasındaki yeri (0 = en doğudaki, yani en küçük y)."""
        self.agent_name = agent_name
        self.slot = slot
        self.half = (area_size or MAP_SIZE_M) / 2.0
        self.spacing = lane_spacing or (2.0 * SWATH_HALF_M - LANE_OVERLAP_M)
        all_lanes = lane_offsets(2.0 * self.half, self.spacing)
        self.lanes = np.array_split(all_lanes, n_agents)[slot]

        self.route = None            # [(bacak başı, bacak sonu, şerit_mi), ...]; ilk plan() çağrısında kurulur
        self.index = 0
        self.done = len(self.lanes) == 0
        self.captures = 0            # yakalama yarıçapıyla alınan nokta sayısı
        self.slips = 0               # slip veya çizgi ölçütüyle alınan nokta sayısı
        self._cpa = np.inf
        self.cross_track = 0.0       # hatta dik uzaklık (m), izleme için

        # EntropyGuidedPlanner ile ortak alanlar (ana döngü bunlara yazar/okur)
        self.current_target = None
        self.global_target = None
        self.wp_accept_r = CAPTURE_RADIUS_M
        self._last_global_replan_tick = 0

    # ─────────────────────────────────────────────────────────────
    # Rota
    # ─────────────────────────────────────────────────────────────
    def _build_route(self, pos, yaw):
        """Şerit uçlarını sıralar. İlk şeride, aracın baktığı yöndeki uçtan girilir."""
        order = lane_visit_order(len(self.lanes), self.spacing)
        lanes = self.lanes[order]
        if abs(lanes[-1] - pos[1]) < abs(lanes[0] - pos[1]):
            lanes = lanes[::-1]                       # araca yakın şeritten başla
        end_x = self.half + LANE_OVERRUN_M
        side = 1.0 if np.cos(yaw) >= 0 else -1.0      # +1: kuzey ucundan gir
        if self.slot % 2 == 1:
            side = -side      # komşu araçlar karşı uçlardan girer: yan yana intikal ve çakışan giriş noktası olmaz
        # Giriş: şerit başının yanında, dönüş çapı kadar açıkta bir noktaya gidilir. Oradan
        # şerit hattına geçiş yarım daire olur ve araç alana girmeden hatta oturur.
        entry_y = lanes[0] + np.sign(pos[1] - lanes[0] or 1.0) * 2.0 * TURN_RADIUS_M
        route = [(np.array(pos, dtype=float), np.array([side * end_x, entry_y]), False)]
        for y in lanes:
            route.append((np.array([side * end_x, y]), np.array([-side * end_x, y]), True))
            side = -side
        self.route = route
        self._cpa = np.inf

    def _advance(self, pos):
        """Sıradaki noktaya varıldıysa bir sonrakine geçer."""
        while not self.done:
            start, target, _ = self.route[self.index]
            dist = float(np.linalg.norm(pos - target))
            self._cpa = min(self._cpa, dist)
            leg = target - start
            captured = dist < max(CAPTURE_RADIUS_M, self.wp_accept_r)
            slipped = self._cpa <= SLIP_RADIUS_M and dist > self._cpa + SLIP_HYST_M
            crossed = float(np.dot(pos - target, leg)) >= 0.0      # noktadan geçen dik çizgi aşıldı
            if not (captured or slipped or crossed):
                return
            self.captures += captured
            self.slips += not captured
            self._cpa = np.inf
            self.index += 1
            self.done = self.index >= len(self.route)

    def _track_point(self, pos, start, end):
        """Hat üzerindeki yönelim noktası (MOOS-IvP `lead` / `lead_damper`)."""
        leg = end - start
        length = float(np.linalg.norm(leg))
        if length < 1e-6:
            self.cross_track = float(np.linalg.norm(pos - end))
            return end
        unit = leg / length
        along = float(np.clip(np.dot(pos - start, unit), 0.0, length))
        foot = start + along * unit
        self.cross_track = float(np.linalg.norm(pos - foot))
        lead = LEAD_M
        if self.cross_track < LEAD_DAMPER_M:
            lead *= 1.0 + 2.0 * (1.0 - self.cross_track / LEAD_DAMPER_M)
        return foot + min(lead, length - along) * unit

    # ─────────────────────────────────────────────────────────────
    # Ana döngü arayüzü
    # ─────────────────────────────────────────────────────────────
    def plan(self, pos, yaw, tick, neighbor_grids=None, sim_time=0.0):
        """Döndürür: (baş açısı [derece], devir, hedef nokta). yaw radyandır."""
        p = np.asarray(pos[:2], dtype=float)
        if self.route is None:
            self._build_route(p, yaw)
        self._advance(p)
        if self.done:
            # Rota bitti: ana döngü aracı başlangıç noktasına gönderir
            return wrap_heading(np.degrees(yaw)), MIN_RPM, self.route[-1][1]

        start, target, is_lane = self.route[self.index]
        steer = self._track_point(p, start, target) - p
        if np.linalg.norm(steer) < 1e-6:
            steer = target - p
        self.current_target = target
        self.global_target = target
        heading = np.degrees(np.arctan2(steer[1], steer[0]))
        # Şeritte ve ilk intikalde seyir devri; şeritler arası dönüşte (şerit başının gerisinde) düşük devir
        turning = is_lane and float(np.dot(p - start, target - start)) < 0.0
        return wrap_heading(heading), TURN_RPM if turning else LANE_RPM, target
