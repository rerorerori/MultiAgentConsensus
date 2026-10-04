"""
Swarm AUV Configuration — Tüm ayarlanabilir parametreler.
Alternatif Versiyon: Dinamik Okyanus Akıntısı Modeli Aktif
"""
import numpy as np
import os

# ═══════════════════════════════════════════════════════════════
# SENARYO
# ═══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HOLOOCEAN_DIR = os.path.dirname(BASE_DIR)
SCENARIO_PATH = os.path.join(HOLOOCEAN_DIR, "worlds", "Ocean", "CooperativeMappingSwarm_3.json")
AGENT_NAMES   = ["auv0", "auv1", "auv2"]
TICKS_PER_SEC = 30

# ═══════════════════════════════════════════════════════════════
# NAVİGASYON
# ═══════════════════════════════════════════════════════════════
CRUISE_RPM    = 1500
CRUISE_DEPTH  = 150.0
MIN_RPM       = 800
MAX_RPM       = 1500
WP_ACCEPT_R   = 15.0      # waypoint kabul yarıçapı (hassasiyet artırıldı)
UTURN_THRESH  = 120        # heading fark eşiği (°)

# ═══════════════════════════════════════════════════════════════
# PROBABILITY MAP
# ═══════════════════════════════════════════════════════════════
MAP_SIZE_M    = 400        # harita kenar uzunluğu (m)
MAP_RES       = 5.0        # hücre çözünürlüğü (m)
MAP_INIT_P    = 0.5        # uniform prior (belirsiz = nesnel Bayesian)
SENSOR_RANGE  = 30.0       # sidescan menzili (m)
P_DETECT      = 0.95       # sensör güvenilirliği (artırıldı)
BAYESIAN_PTS  = 30         # güncelleme noktası sayısı (artırıldı)

# ═══════════════════════════════════════════════════════════════
# CONSENSUS
# ═══════════════════════════════════════════════════════════════
CONSENSUS_EPS       = 0.3    # consensus kazancı

# Beacon: 27 bayt header + kapsama bit haritası (256 baytlık modem paketi sınırı)
# Header: sender_id(uint8) + n_bytes(uint16) + wp_x, wp_y + nav_x, nav_y, nav_z, nav_std (float32)
BEACON_PACK_FMT     = "<BHffffff"
BEACON_BLOCK        = 2      # bit haritasında bir bit = BLOCK x BLOCK hücre (80x80 -> 40x40 bit = 200 bayt)

# ═══════════════════════════════════════════════════════════════
# ENTROPY PLANNER
# ═══════════════════════════════════════════════════════════════
REPLAN_INTERVAL    = 600    # tick (20 saniye @ 30Hz)
ENTROPY_DIST_DECAY = 500.0  # mesafe penaltı katsayısı (m)
MIN_TARGET_DIST    = 90.0   # En az 90m uzakta hedef ara

# ═══════════════════════════════════════════════════════════════
# LAWNMOWER (Baseline)
# ═══════════════════════════════════════════════════════════════
LAWN_X_MIN    = -200.0
LAWN_X_MAX    = 200.0
LAWN_STRIPS   = {
    "auv0": {"y_min": -60,  "y_max": 60},
    "auv1": {"y_min": -200, "y_max": -60},
    "auv2": {"y_min": 60,   "y_max": 200},
}
LAWN_STRIP_W  = 20.0

# ═══════════════════════════════════════════════════════════════
# STUCK DETECTION
# ═══════════════════════════════════════════════════════════════
STUCK_TIME_S          = 10.0
STUCK_DIST_M          = 2.0
STARTUP_GRACE_S       = 30.0
POST_RECOVERY_GRACE_S = 15.0
RECOVERY_DUR_S        = 5.0
RECOVERY_HEADING_OFF  = 90.0
RECOVERY_RPM          = 1200

# ═══════════════════════════════════════════════════════════════
# FLS AVOIDANCE
# ═══════════════════════════════════════════════════════════════
FLS_THRESHOLD  = 0.15
FLS_BIAS_HARD  = 30.0
FLS_BIAS_SOFT  = 15.0

# ═══════════════════════════════════════════════════════════════
# HABERLEŞME
# ═══════════════════════════════════════════════════════════════
BEACON_SEND_INTERVAL = 300   # tick (~10s)
PACKET_LOSS_RATE     = 0.0

# ═══════════════════════════════════════════════════════════════
# GUI
# ═══════════════════════════════════════════════════════════════
GUI_UPDATE_TICKS     = 30
TRAJECTORY_LEN       = 500
MAP_BACKGROUND_IMAGE = "maps/seafloor.jpg"
MAP_BACKGROUND_BRIGHTNESS = 0.55
MAP_BACKGROUND_TINT_ALPHA = 0.40

# ═══════════════════════════════════════════════════════════════
# ROAMING
# ═══════════════════════════════════════════════════════════════
BOUNDS_X = [-180, 180]
BOUNDS_Y = [-180, 180]

# ═══════════════════════════════════════════════════════════════
# DİNAMİK OKYANUS AKINTISI PARAMETRELERİ (YENİ)
# ═══════════════════════════════════════════════════════════════
USE_OCEAN_CURRENT     = True
# Ege Denizi İzmir Çeşme açıkları ortalama akıntı vektörü [Kuzey, Doğu, Dikey] (m/s)
# Ege'de akıntı genel olarak güneye doğrudur (kuzey ivmesi negatiftir)
BASE_CURRENT_NED      = np.array([-0.20, 0.10, 0.0])
# Birinci derece Gauss-Markov türbülans parametreleri
# CURRENT_NOISE_VAR sürecin difüzyon şiddetidir (sigma_c^2, m^2/s^3); durağan türbülans
# standart sapması sqrt(CURRENT_NOISE_VAR * CURRENT_TAU / 2) ≈ 0.155 m/s (eksen başına).
CURRENT_NOISE_VAR     = 0.02**2
CURRENT_TAU           = 120.0     # Korelasyon zaman sabiti (saniye)

# ═══════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════
def wrap_heading(angle):
    return ((angle + 180) % 360) - 180

def heading_diff(current, target):
    d = target - current
    return ((d + 180) % 360) - 180
