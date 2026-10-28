"""
AUV Navigation - 18-State Error-State INS/EKF (Quaternion Attitude, Current State)
===================================================================================
Nominal durum: konum p, hız v (NED), ivmeölçer bias b_a, tutum quaternion'u q (gövde -> NED),
gyro bias b_g, okyanus akıntısı c (NED). Kovaryans 18 boyutlu hata durumu üzerinde tutulur:
    dx = [dp, dv, db_a, dtheta, db_g, dc]
dtheta gövde ekseninde küçük dönme hatasıdır (R_gerçek = R · exp([dtheta]x)); güncellemede
nominal tutuma çarpımsal olarak eklenir. Euler açıları yalnızca çıktı için türetilir, bu
yüzden filtrede pitch = ±90° tekilliği yoktur.

- Tahmin: IMU (ivmeölçer + gyro) ile INS mekanizasyonu
- Güncelleme: DVL (gövde ekseni hız), derinlik, manyetometre (yaw), akustik mesafe,
  su bağıl hız (pervane/dönüş modeli)
- Akıntı rastgele yürüyüş olarak modellenir. Su bağıl hız ölçümü R^T (v - c) = model hızı
  bağıntısıyla yer hızını akıntıya bağlar: DVL varken akıntı gözlenir, DVL yokken yer hızı
  model + kestirilen akıntıdan gelir ve akıntı belirsizliği konum belirsizliğine taşınır.
"""

import numpy as np
from scipy.spatial.transform import Rotation

# NWU <-> NED dönüşümü (kendi tersidir)
T_NWU_NED = np.diag([1.0, -1.0, -1.0])
# Yerçekimi ivmesi (NED, aşağı pozitif)
G_NED = np.array([0.0, 0.0, 9.81])

N_STATES = 18
POS, VEL, BA, ATT, BG = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15)
CUR = slice(15, 18)


def skew(v: np.ndarray) -> np.ndarray:
    """3-vektörün skew-symmetric matrisini döndürür [v]x"""
    return np.array([
        [ 0.0,   -v[2],   v[1]],
        [ v[2],   0.0,   -v[0]],
        [-v[1],   v[0],   0.0 ]
    ])


def wrap_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rotation_to_euler(R: np.ndarray) -> np.ndarray:
    """Gövde -> NED dönüşüm matrisinden ZYX Euler açıları [phi, theta, psi]."""
    return np.array([
        np.arctan2(R[2, 1], R[2, 2]),
        -np.arcsin(np.clip(R[2, 0], -1.0, 1.0)),
        np.arctan2(R[1, 0], R[0, 0]),
    ])


class AUVNavigationEKF:
    """
    18-Durumlu hata-durumlu EKF + Ataletsel Seyrüsefer Çekirdeği.

    current_compensation=False verilirse akıntı durumu sıfırda sabitlenir (baseline):
    su bağıl hız modeli yalnızca DVL kesintisinde, akıntısız olarak kullanılır.
    """

    # REMUS-100 kalıcı durum surge hızı: u_ss = rpm / RPM_PER_MPS
    RPM_PER_MPS = 600.0
    # Dönüşte su bağıl sway hızı: v = SWAY_PER_YAW_RATE * r (NED gövde, m/s per rad/s)
    SWAY_PER_YAW_RATE = -3.3

    def __init__(self, dt: float, agent_name: str, dvl_dt: float = None,
                 current_compensation: bool = True, dvl_max_range: float = None,
                 dvl_beam_sigma: float = 0.01, dvl_elevation_deg: float = 22.5,
                 current_diffusion: float = 0.02):
        """
        current_diffusion: akıntı rastgele yürüyüşünün şiddeti (m/s/sqrt(s), yatay eksen başına).
        dvl_max_range: DVL ışın menzili (m). Verilirse taban kilidi ışın mesafelerinden
                       belirlenir; None ise yalnızca geçersiz hız ölçümü kayıp sayılır.
        dvl_beam_sigma / dvl_elevation_deg: ışın hız gürültüsü ve ışınların düşeyle açısı.
        """
        self.dt          = dt
        self.dvl_dt      = dvl_dt if dvl_dt is not None else dt
        self.name        = agent_name
        self.initialized = False
        self.current_compensation = current_compensation

        # Nominal durum
        self.p  = np.zeros(3)            # NED konum (m)
        self.v  = np.zeros(3)            # NED hız (m/s) (Yer hızı - Ground Velocity)
        self.ba = np.zeros(3)            # ivmeölçer bias (m/s²)
        self.bg = np.zeros(3)            # gyro bias (rad/s)
        self.c  = np.zeros(3)            # okyanus akıntısı, NED (m/s)
        self._rot = Rotation.identity()  # tutum: gövde -> NED
        self._R   = np.eye(3)            # _rot'un matris hali (önbellek)

        # Başlangıç Kovaryansı P (18x18, hata durumu)
        self.P = np.diag([
            0.1**2, 0.1**2, 0.1**2,      # konum (bilinen başlangıç noktası)
            0.05, 0.05, 0.02,            # hız
            0.05**2, 0.05**2, 0.05**2,   # ivme bias
            0.02**2, 0.02**2, 0.02**2,   # yönelim (~1 derece)
            1e-3**2, 1e-3**2, 1e-3**2,   # gyro bias
            0.3**2, 0.3**2, 0.0,         # akıntı (yatay bilinmiyor, düşey yok)
        ])
        if not current_compensation:
            current_diffusion = 0.0
            self.P[CUR, CUR] = 0.0

        # Proses Gürültüsü Q (18x18)
        self.Q = np.diag([
            (0.005)**2 * dt, (0.005)**2 * dt, (0.001)**2 * dt,   # konum
            (0.02)**2  * dt, (0.02)**2  * dt, (0.005)**2 * dt,   # hız
            (1e-5)**2  * dt, (1e-5)**2  * dt, (1e-5)**2  * dt,   # ivme bias
            (1e-3)**2  * dt, (1e-3)**2  * dt, (1e-3)**2  * dt,   # tutum
            (5e-5)**2  * dt, (5e-5)**2  * dt, (5e-5)**2  * dt,   # gyro bias (HoloOcean AngVelBiasSigma = 5e-5)
            current_diffusion**2 * dt, current_diffusion**2 * dt, 0.0,   # akıntı
        ])

        # Ölçüm Gürültü Matrisleri
        # DVL: dört ışındaki bağımsız gürültü Janus geometrisiyle gövde hızına taşınır;
        # kol uzunluğu ve zamanlama gibi modellenmeyen hatalar için küçük bir taban eklenir
        sin_e, cos_e = np.sin(np.radians(dvl_elevation_deg)), np.cos(np.radians(dvl_elevation_deg))
        var_xy = dvl_beam_sigma ** 2 / (2.0 * sin_e ** 2)
        var_z  = dvl_beam_sigma ** 2 / (4.0 * cos_e ** 2)
        self.R_dvl     = np.diag([var_xy, var_xy, var_z]) + np.eye(3) * (0.01 ** 2)
        self.R_depth   = np.array([[0.1 ** 2]])
        self.R_mag     = np.array([[0.009 ** 2]])  # ~0.5 derece
        # Su bağıl gövde hızı modeli (surge, sway, heave). Model hatası (dönüş geçişleri, pervane
        # modeli) birkaç saniye korelasyonludur; 10 Hz'de beyaz gürültü gibi işlendiği için
        # sigma, anlık model hatasından (~0.07 m/s) büyük seçilir.
        self.R_water   = np.diag([0.3 ** 2, 0.3 ** 2, 0.1 ** 2])

        # DVL kesintisinde yer hızı yalnızca modelden gelir; modelin yavaş değişen hatası
        # (baseline'da bilinmeyen akıntının tamamı) konumda doğrusal kaymaya yol açar.
        # Konum belirsizliği sigma * T_ufuk kaymasını karşılayacak hızda büyütülür.
        outage_sigma = 0.12 if current_compensation else 0.30
        self._q_pos_outage = (outage_sigma ** 2) * 60.0    # m²/s

        # Taban kilidi: menzil içinde dönen ışın sayısı bu değerin altındaysa DVL kayıp
        self.dvl_max_range = dvl_max_range
        self.dvl_min_beams = 3

        # DVL'in IMU'ya göre konumu (NED gövde, m) — dönüşte omega x r kol etkisi
        self.r_dvl = np.array([-1.16, 0.0, 0.08])

        # Son bias düzeltilmiş IMU örneği (NED gövde)
        self._f_body = -G_NED.copy()
        self._omega  = np.zeros(3)

        # Su bağıl gövde hız modeli (pervane + dönüş kaynaklı sway)
        self._u_model   = 0.0
        self._v_model   = 0.0
        self._tau_surge = 2.0
        self._tau_sway  = 1.0

        self.dvl_lost = False
        # Son DVL güncellemesinin normalize inovasyon karesi (tutarlılık izleme, beklenen ~3)
        self.nis_dvl  = float("nan")

    # ──────────────────────────────────────────────────────────────
    # Başlatma
    # ──────────────────────────────────────────────────────────────
    def _set_rotation(self, rot: Rotation):
        self._rot = rot
        self._R   = rot.as_matrix()

    def initialize_from_pose(self, pose_nwu: np.ndarray, vel_world_nwu: np.ndarray):
        self.p = T_NWU_NED @ np.array(pose_nwu[:3, 3], dtype=float)
        self.v = T_NWU_NED @ np.array(vel_world_nwu[:3], dtype=float)
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)
        self.c  = np.zeros(3)
        self._set_rotation(Rotation.from_matrix(T_NWU_NED @ pose_nwu[:3, :3] @ T_NWU_NED))

        vel_body = self._R.T @ self.v
        self._u_model = float(vel_body[0])
        self._v_model = float(vel_body[1])

        self.initialized = True

    # ──────────────────────────────────────────────────────────────
    # Tahmin
    # ──────────────────────────────────────────────────────────────
    def predict(self, imu_raw=None):
        if not self.initialized:
            return

        if imu_raw is not None:
            imu = np.array(imu_raw, dtype=float).flatten()
            if len(imu) >= 6:
                # HoloOcean IMU: özgül kuvvet ve açısal hız, NWU gövde ekseninde
                self._f_body = T_NWU_NED @ imu[0:3] - self.ba
                self._omega  = T_NWU_NED @ imu[3:6] - self.bg
        f, omega = self._f_body, self._omega
        R = self._R

        # Hata durumu sistem matrisi F (18x18); akıntı rastgele yürüyüş olduğundan F'de satırı boş
        F = np.zeros((N_STATES, N_STATES))
        F[POS, VEL] = np.eye(3)          # d(dp)/d(dv)
        F[VEL, BA]  = -R                 # d(dv)/d(db_a)
        F[VEL, ATT] = -R @ skew(f)       # d(dv)/d(dtheta)
        F[ATT, ATT] = -skew(omega)       # d(dtheta)/d(dtheta)
        F[ATT, BG]  = -np.eye(3)         # d(dtheta)/d(db_g)

        # INS mekanizasyonu: a = R f + g ; q <- q ⊗ exp(omega dt)
        a_ned = R @ f + G_NED
        self.p = self.p + self.dt * self.v + 0.5 * (self.dt ** 2) * a_ned
        self.v = self.v + self.dt * a_ned
        self._set_rotation(self._rot * Rotation.from_rotvec(omega * self.dt))

        Phi_d = np.eye(N_STATES) + F * self.dt
        self.P = Phi_d @ self.P @ Phi_d.T + self.Q
        if self.dvl_lost:
            self.P[0, 0] += self._q_pos_outage * self.dt
            self.P[1, 1] += self._q_pos_outage * self.dt
        self.P = 0.5 * (self.P + self.P.T)

        # Dönüş kaynaklı sway modeli
        v_ss = self.SWAY_PER_YAW_RATE * omega[2]
        self._v_model += (self.dt / self._tau_sway) * (v_ss - self._v_model)

    # ──────────────────────────────────────────────────────────────
    # Güncellemeler
    # ──────────────────────────────────────────────────────────────
    def _kalman_update(self, y: np.ndarray, H: np.ndarray, R_meas: np.ndarray) -> float:
        """
        Joseph formunda Kalman güncellemesi; hata durumu nominal duruma eklenir
        (tutum çarpımsal). Normalize inovasyon karesini (NIS) döndürür.
        """
        S = H @ self.P @ H.T + R_meas
        K = np.linalg.solve(S, H @ self.P).T
        dx = K @ y

        self.p  = self.p + dx[POS]
        self.v  = self.v + dx[VEL]
        self.ba = self.ba + dx[BA]
        self.bg = self.bg + dx[BG]
        self.c  = self.c + dx[CUR]
        self._set_rotation(self._rot * Rotation.from_rotvec(dx[ATT]))

        I_KH = np.eye(N_STATES) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_meas @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return float(y @ np.linalg.solve(S, y))

    def _update_water_velocity(self):
        """
        Su bağıl hız ölçümü: pervane ve dönüş modelinden gelen gövde hızı = R^T (v - c).
        """
        v_rel_body = self._R.T @ (self.v - self.c)
        z_model = np.array([self._u_model, self._v_model, 0.0])

        H = np.zeros((3, N_STATES))
        H[:, VEL] = self._R.T
        H[:, ATT] = skew(v_rel_body)
        if self.current_compensation:
            H[:, CUR] = -self._R.T
        self._kalman_update(z_model - v_rel_body, H, self.R_water)

    def update_dvl(self, dvl_raw: np.ndarray, rpm: float = None):
        """
        dvl_raw: [vx, vy, vz] veya ışın mesafeleriyle [vx, vy, vz, r1, r2, r3, r4] (NWU gövde).
        Ölçüm yoksa, geçersizse ya da taban kilidi yoksa yalnızca su bağıl hız modeli kullanılır.
        """
        if not self.initialized:
            return

        dvl = None
        if dvl_raw is not None:
            dvl = np.array(dvl_raw, dtype=float).flatten()
            if np.any(np.isnan(dvl[0:3])) or np.all(np.abs(dvl[0:3]) < 1e-5):
                dvl = None
            elif self.dvl_max_range is not None and len(dvl) >= 7:
                # Tabana ulaşmayan ışın menzil sınırında doyar
                locked_beams = int(np.sum(dvl[3:7] < self.dvl_max_range - 1.0))
                if locked_beams < self.dvl_min_beams:
                    dvl = None
        self.dvl_lost = dvl is None

        # Pervane devrinden su bağıl surge hızı modeli
        if rpm is not None:
            u_ss = float(rpm) / self.RPM_PER_MPS
            self._u_model += (self.dvl_dt / self._tau_surge) * (u_ss - self._u_model)

        if not self.dvl_lost:
            # ──────────────────────────────────────────────────────
            # A. DVL AKTİF: gövde ekseninde ölçüm, h(x) = R^T v + omega x r_dvl
            # ──────────────────────────────────────────────────────
            z_body = T_NWU_NED @ dvl[0:3]
            v_body = self._R.T @ self.v
            lever = np.cross(self._omega, self.r_dvl)

            H = np.zeros((3, N_STATES))
            H[:, VEL] = self._R.T
            H[:, ATT] = skew(v_body)
            H[:, BG]  = skew(self.r_dvl)

            self.nis_dvl = self._kalman_update(z_body - (v_body + lever), H, self.R_dvl)

        # ──────────────────────────────────────────────────────
        # B. SU BAĞIL HIZ MODELİ: DVL varken akıntıyı gözler, DVL yokken yer hızını taşır.
        #    Baseline'da akıntı durumu olmadığından yalnızca DVL kesintisinde kullanılır.
        # ──────────────────────────────────────────────────────
        if self.current_compensation or self.dvl_lost:
            self._update_water_velocity()

    def update_depth(self, depth_m: float):
        if not self.initialized:
            return

        H = np.zeros((1, N_STATES))
        H[0, 2] = 1.0
        self._kalman_update(np.array([depth_m - self.p[2]]), H, self.R_depth)

    def update_magnetometer(self, mag_raw: np.ndarray) -> None:
        """
        Manyetometreden baş açısı güncellemesi.
        mag_raw: HoloOcean MagnetometerSensor çıkışı, NWU gövde ekseninde [mx, my, mz]

        Ölçülen alan kestirilen tutumla NED'e döndürülür; yatay izdüşümünün kuzeyden
        sapması, tutumun düşey eksen etrafındaki hatasıdır.
        """
        if not self.initialized or mag_raw is None:
            return

        mag = np.array(mag_raw, dtype=float).flatten()
        if len(mag) < 3 or np.any(np.isnan(mag[:3])):
            return

        m_ned = self._R @ (T_NWU_NED @ mag[:3])
        yaw_error = np.arctan2(m_ned[1], m_ned[0])   # kestirilen - gerçek

        H = np.zeros((1, N_STATES))
        H[0, ATT] = self._R[2, :]                    # NED düşey eksen etrafındaki dönme
        self._kalman_update(np.array([-yaw_error]), H, self.R_mag)

    def update_range(self, range_m: float, anchor_ned: np.ndarray, anchor_std: float,
                     range_sigma: float) -> bool:
        """
        Akustik mesafe güncellemesi: başka bir aracın bildirdiği konuma olan eğik mesafe.
        anchor_std, bildirilen konumun belirsizliğidir ve ölçüm gürültüsüne eklenir.
        İnovasyon 3-sigma dışındaysa ölçüm reddedilir; kullanıldıysa True döner.

        Ölçüm yalnızca kendi konum belirsizliği hem komşununkinden hem de mesafe gürültüsünden
        belirgin büyükken kullanılır. Araçların hataları birbirinden bağımsız değildir (aynı
        akıntı, karşılıklı güncellemeler); bu bağıntı modellenmediği için, zaten iyi bilinen bir
        konuma gürültülü mesafe eklemek hatayı azaltmak yerine artırır.
        """
        if not self.initialized:
            return False
        if self.get_position_std() < max(2.0 * anchor_std, 0.5 * range_sigma):
            return False

        delta = self.p - np.asarray(anchor_ned, dtype=float)
        predicted = float(np.linalg.norm(delta))
        if predicted < 1.0:
            return False

        H = np.zeros((1, N_STATES))
        H[0, POS] = delta / predicted
        R_meas = np.array([[range_sigma ** 2 + anchor_std ** 2]])
        y = np.array([range_m - predicted])

        S = float((H @ self.P @ H.T + R_meas)[0, 0])
        if y[0] ** 2 > 9.0 * S:
            return False
        self._kalman_update(y, H, R_meas)
        return True

    # ──────────────────────────────────────────────────────────────
    # Çıktılar
    # ──────────────────────────────────────────────────────────────
    @property
    def x(self) -> np.ndarray:
        """[p, v, b_a, (phi, theta, psi), b_g, c] — kayıt ve karşılaştırma için durum vektörü (kopya)."""
        return np.concatenate([self.p, self.v, self.ba, rotation_to_euler(self._R), self.bg, self.c])

    @property
    def x_current(self) -> np.ndarray:
        """Kestirilen akıntı (NED)."""
        return self.c

    def get_fossen_state(self, dynamics_nwu: np.ndarray) -> np.ndarray:
        """DynamicsSensor formatında (19 eleman, NWU) kestirilmiş durum: hız, konum, açısal hız, quaternion."""
        R_nwu = T_NWU_NED @ self._R @ T_NWU_NED

        out = np.array(dynamics_nwu, dtype=float).copy()
        out[3:6]   = T_NWU_NED @ self.v
        out[6:9]   = T_NWU_NED @ self.p
        out[12:15] = R_nwu @ (T_NWU_NED @ self._omega)
        out[15:19] = Rotation.from_matrix(R_nwu).as_quat()  # [x, y, z, w]
        return out

    def get_position_nwu(self) -> np.ndarray:
        return T_NWU_NED @ self.p

    def get_yaw_nwu(self) -> float:
        return float(-np.arctan2(self._R[1, 0], self._R[0, 0]))

    def get_current_nwu(self) -> np.ndarray:
        return T_NWU_NED @ self.c

    def get_position_std(self) -> float:
        """Yatay konum belirsizliği (1-sigma, m)."""
        return float(np.sqrt(max(self.P[0, 0] + self.P[1, 1], 0.0)))
