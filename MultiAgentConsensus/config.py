"""
Swarm AUV Configuration — Tüm ayarlanabilir parametreler.
"""
import numpy as np

# ═══════════════════════════════════════════════════════════════
# SENARYO
# ═══════════════════════════════════════════════════════════════
import os
# BASE_DIR is .../Local/holoocean/2.3.0/MultiAgentConsensus
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Scenario is at .../Local/holoocean/2.3.0/worlds/Ocean/...
HOLOOCEAN_DIR = os.path.dirname(BASE_DIR)
SCENARIO_PATH = os.path.join(HOLOOCEAN_DIR, "worlds", "Ocean", "CooperativeMappingSwarm_2.json")
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
CONSENSUS_MAX_CELLS = 48        # 11 + 48×5 = 251B < 256B
CONSENSUS_THRESHOLD = 0.1    # değişim eşiği (|p - 0.5| > threshold)

# Binary packing (Subnero M25M 256 byte limit uyumu)
# Her hücre: row(uint16) + col(uint16) + prob(uint8 0-255) = 5 byte
# Header: sender_id(uint8) + n_cells(uint16) + wp_x(float32) + wp_y(float32) = 11 byte
# Toplam: 11 + 48×5 = 251 byte < 256
BEACON_PACK_FMT     = "<BHff"   # 11 byte header
BEACON_CELL_FMT     = "<HHB"    # uint16 row, uint16 col, uint8 prob

# ═══════════════════════════════════════════════════════════════
# ENTROPY PLANNER
# ═══════════════════════════════════════════════════════════════
REPLAN_INTERVAL    = 600    # tick (20 saniye @ 30Hz) — daha kararlı seyir (commitment ile uyumlu)
ENTROPY_DIST_DECAY = 500.0  # mesafe penaltı katsayısı (m) — uzağı çekici yapar ama yakınları da unutmaz
MIN_TARGET_DIST    = 90.0   # En az 90m uzakta hedef ara (zigzag/daire önleyici)

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
RECOVERY_DEPTH_OFF    = 2.0
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
PACKET_LOSS_RATE     = 0.0   # 0.0 = sabit kayıp devre dışı, Thorp+HMM kullanılır
                             # NOT: loss_rate > 0 verilirse Thorp+HMM İLE BİRLİKTE uygulanır
                             # (iki bağımsız Bernoulli denemesi)

# ═══════════════════════════════════════════════════════════════
# GUI
# ═══════════════════════════════════════════════════════════════
GUI_UPDATE_TICKS     = 30
TRAJECTORY_LEN       = 500
MAP_BACKGROUND_IMAGE = "maps/seafloor.jpg"   # Harita arka plan görseli (opsiyonel)
                                              # Desteklenen: JPG, PNG, TIFF
                                              # Yoksa koyu mavi gradient kullanılır
MAP_BACKGROUND_BRIGHTNESS = 0.55   # 0.0-1.0 aralığı (karartma derecesi)
MAP_BACKGROUND_TINT_ALPHA = 0.40   # mavi tint yoğunluğu (opaklık)


# ═══════════════════════════════════════════════════════════════
# ROAMING
# ═══════════════════════════════════════════════════════════════
BOUNDS_X = [-180, 180]
BOUNDS_Y = [-180, 180]

# ═══════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════
def wrap_heading(angle):
    """Heading açısını [-180, 180] aralığına sınırla."""
    return ((angle + 180) % 360) - 180

def heading_diff(current, target):
    """İki heading arası en kısa fark ([-180, 180])."""
    d = target - current
    return ((d + 180) % 360) - 180