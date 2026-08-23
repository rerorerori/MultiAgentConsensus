"""
AUV Navigation - 15-State Strapdown INS/EKF (Current-Aided Version)
====================================================================
Durum vektörü: [pn, pe, pd, vn, ve, vd, bax, bay, baz, phi, theta, psi, bgx, bgy, bgz]
- Tutum gyro entegrasyonuyla taşınır (DynamicsSensor kullanılmaz)
- Gyro bias çevrimiçi kalibre edilir
- İvmeölçer bias çevrimiçi kalibre edilir
"""

import numpy as np
from scipy.spatial.transform import Rotation


class AUVNavigationEKF:
    """
    15-Durumlu EKF + Ataletsel Seyrüsefer Çekirdeği (Akıntı Telafili).
    """

    def __init__(self, dt: float, agent_name: str):
        self.dt          = dt
        self.name        = agent_name
        self.initialized = False

        # Durum vektörü (15 boyutlu):
        # x[0:3]   = [pn, pe, pd]       # NED konum (m)
        # x[3:6]   = [vn, ve, vd]       # NED hız (m/s) (Yer hızı - Ground Velocity)
        # x[6:9]   = [bax, bay, baz]    # ivmeölçer bias (m/s²)
        # x[9:12]  = [phi, theta, psi]  # Euler tutum (NED, radyan)
        # x[12:15] = [bgx, bgy, bgz]    # gyro bias (rad/s)
        self.x = np.zeros(15)

        # Başlangıç Kovaryansı P (15x15)
        self.P = np.diag([
            1.0,  1.0,  0.1,    # konum
            0.05, 0.05, 0.02,   # hız
            0.02, 0.02, 0.02,   # ivme bias
            0.01, 0.01, 0.01,   # tutum (rad²)
            1e-3, 1e-3, 1e-3,   # gyro bias (rad²/s²)
        ])

        # Proses Gürültüsü Q (15x15)
        self.Q = np.diag([
            (0.005)**2 * dt, (0.005)**2 * dt, (0.001)**2 * dt,   # konum
            (0.02)**2  * dt, (0.02)**2  * dt, (0.005)**2 * dt,   # hız
            (1e-5)**2  * dt, (1e-5)**2  * dt, (1e-5)**2  * dt,   # ivme bias
            (1e-3)**2  * dt, (1e-3)**2  * dt, (1e-3)**2  * dt,   # tutum
            (5e-5)**2  * dt, (5e-5)**2  * dt, (5e-5)**2  * dt,   # gyro bias (HoloOcean AngVelBiasSigma = 5e-5)
        ])

        # Ölçüm Gürültü Matrisleri
        self.R_dvl   = np.eye(3) * (0.02 ** 2)
        self.R_depth = np.array([[0.1 ** 2]])

        # Yerçekimi ivmesi (NED Down ekseninde pozitif)
        self.g_ned = np.array([0.0, 0.0, 9.81])

        # NWU → NED dönüşüm matrisi
        self.T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        # Yönelimler
        self._euler_ned     = np.zeros(3)
        self._R_body2ned    = np.eye(3)
        self._last_gyro_ned = np.zeros(3)

        # Hidrodinamik Sanal Hız Durumları (DVL kaybolduğunda çalışır)
        self._u_virtual = 0.0
        self._v_virtual = 0.0
        self._w_virtual = 0.0

        self._tau_surge = 3.0
        self._tau_sway  = 5.0
        self._tau_heave = 5.0

        self._is_dvl_lost = False
        self.use_virtual_dvl = True

        # --- AKINTI TAHMİN BİRİMİ (CURRENT ESTIMATOR) ---
        # Akıntıyı dünya (NED) ekseninde [Vcx, Vcy, Vcz] olarak kestiriyoruz.
        self.x_current = np.zeros(3)
        self._tau_current = 50.0

    def initialize_from_pose(self, pose_nwu: np.ndarray, vel_world_nwu: np.ndarray):
        pos_nwu    = pose_nwu[:3, 3]
        pos_ned    = self.T @ pos_nwu
        self.x[0:3] = pos_ned

        R_nwu = pose_nwu[:3, :3]
        R_ned = self.T @ R_nwu @ self.T

        phi   = np.arctan2(R_ned[2, 1], R_ned[2, 2])
        theta = -np.arcsin(np.clip(R_ned[2, 0], -1.0, 1.0))
        psi   = np.arctan2(R_ned[1, 0], R_ned[0, 0])

        self.x[9]     = phi
        self.x[10]    = theta
        self.x[11]    = psi
        self.x[12:15] = np.zeros(3)  # gyro bias sıfır başlar

        self._euler_ned  = np.array([phi, theta, psi])
        self._R_body2ned = R_ned.copy()

        vel_ned_world = self.T @ np.array(vel_world_nwu[:3], dtype=float)
        self.x[3:6]   = vel_ned_world
        self.x[6:9]   = np.zeros(3)

        vel_body_ned = R_ned.T @ vel_ned_world
        self._u_virtual = float(vel_body_ned[0])
        self._v_virtual = float(vel_body_ned[1])
        self._w_virtual = float(vel_body_ned[2])

        self.initialized = True

    def predict(self, imu_raw=None):
        if not self.initialized:
            return

        a_ned = np.zeros(3)
        omega_body_ned = np.zeros(3)
        f_corrected = np.zeros(3)

        if imu_raw is not None:
            imu = np.array(imu_raw, dtype=float).flatten()
            if len(imu) >= 6:
                # İvmeölçer — bias düzelt
                f_raw_body = self.T @ imu[0:3]
                b_a = self.x[6:9]
                f_corrected = f_raw_body - b_a
                a_ned = self._R_body2ned @ f_corrected - self.g_ned

                # Gyro — bias düzelt
                gyro_raw_body = self.T @ imu[3:6]
                b_g = self.x[12:15]
                omega_body_ned = gyro_raw_body - b_g
                self._last_gyro_ned = omega_body_ned

        # Konum ve hız entegrasyonu (DVL outage'da ivme entegrasyonu atlanır, hızla konum ilerletilir)
        if self._is_dvl_lost and self.use_virtual_dvl:
            self.x[0:3] += self.dt * self.x[3:6]
        else:
            self.x[0:3] += self.dt * self.x[3:6] + 0.5 * (self.dt ** 2) * a_ned
            self.x[3:6] += self.dt * a_ned

        # Tutum entegrasyonu — Euler kinematik denklemi
        phi, theta, psi = self.x[9], self.x[10], self.x[11]
        cos_theta = np.cos(theta)
        if abs(cos_theta) < 0.01:  # singülarite koruması
            cos_theta = 0.01 * np.sign(cos_theta) if cos_theta != 0 else 0.01

        Phi_mat = np.array([
            [1.0, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0.0, np.cos(phi),                -np.sin(phi)             ],
            [0.0, np.sin(phi) / cos_theta,     np.cos(phi) / cos_theta  ]
        ])
        self.x[9:12] += self.dt * (Phi_mat @ omega_body_ned)

        # R_body2ned ve _euler_ned'i güncel tutum durumundan hesapla
        phi, theta, psi = self.x[9], self.x[10], self.x[11]
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(phi), -np.sin(phi)], [0.0, np.sin(phi), np.cos(phi)]])
        Ry = np.array([[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]])
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0.0], [np.sin(psi), np.cos(psi), 0.0], [0.0, 0.0, 1.0]])
        self._R_body2ned = Rz @ Ry @ Rx
        self._euler_ned  = self.x[9:12].copy()

        # F matrisi 15x15
        F = np.zeros((15, 15))
        F[0:3, 3:6]    = np.eye(3)          # dp/dv
        F[3:6, 6:9]    = -self._R_body2ned  # dv/d(ba)
        F[9:12, 12:15] = -Phi_mat          # datt/d(bg)

        Phi = np.eye(15) + F * self.dt
        self.P = Phi @ self.P @ Phi.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)

    def update_dvl(self, dvl_raw: np.ndarray, rpm: float = None):
        if not self.initialized:
            return

        is_dvl_lost = False
        if dvl_raw is None:
            is_dvl_lost = True
        else:
            dvl = np.array(dvl_raw, dtype=float).flatten()
            if np.all(np.abs(dvl[0:3]) < 1e-5) or np.isnan(dvl[0]):
                is_dvl_lost = True

        self._is_dvl_lost = is_dvl_lost

        if not is_dvl_lost:
            # ──────────────────────────────────────────────────────
            # A. DVL AKTİF DURUM (Normal ESKF Güncellemesi & Bias/Akıntı Öğrenme)
            # ──────────────────────────────────────────────────────
            v_dvl_nwu = dvl[0:3]
            v_dvl_ned_body = self.T @ v_dvl_nwu
            v_dvl_ned_world = self._R_body2ned @ v_dvl_ned_body

            H = np.zeros((3, 15))
            H[0:3, 3:6] = np.eye(3)

            y = v_dvl_ned_world - self.x[3:6]
            S = H @ self.P @ H.T + self.R_dvl
            K = self.P @ H.T @ np.linalg.inv(S)

            self.x += K @ y
            self.P = (np.eye(15) - K @ H) @ self.P
            self.P = 0.5 * (self.P + self.P.T)

            # Tutum açılarını sınırla ve R_body2ned matrisini güncelle
            self.x[9]  = (self.x[9]  + np.pi) % (2.0 * np.pi) - np.pi
            self.x[10] = np.clip(self.x[10], -np.pi/2.0 + 0.01, np.pi/2.0 - 0.01)
            self.x[11] = (self.x[11] + np.pi) % (2.0 * np.pi) - np.pi

            phi, theta, psi = self.x[9], self.x[10], self.x[11]
            Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(phi), -np.sin(phi)], [0.0, np.sin(phi), np.cos(phi)]])
            Ry = np.array([[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]])
            Rz = np.array([[np.cos(psi), -np.sin(psi), 0.0], [np.sin(psi), np.cos(psi), 0.0], [0.0, 0.0, 1.0]])
            self._R_body2ned = Rz @ Ry @ Rx
            self._euler_ned  = self.x[9:12].copy()

            # Sanal hızları senkronize et (suya bağıl hız)
            self._u_virtual = float(v_dvl_ned_body[0])
            self._v_virtual = float(v_dvl_ned_body[1])
            self._w_virtual = float(v_dvl_ned_body[2])

            # --- DİNAMİK AKINTI ÖĞRENME ---
            rpm_val = float(rpm) if rpm is not None else 1500.0
            u_ss = rpm_val / 600.0
            v_body_theoretical = np.array([u_ss, 0.0, 0.0])
            v_world_theoretical = self._R_body2ned @ v_body_theoretical

            # Anlık akıntı ölçüm girdisi = gerçek yer hızı - teorik su bağıl hızı
            current_measurement = v_dvl_ned_world - v_world_theoretical

            # Alçak geçiren filtreyle akıntıyı süz ve öğren
            self.x_current += (self.dt / self._tau_current) * (current_measurement - self.x_current)

        else:
            # ──────────────────────────────────────────────────────
            # B. DVL KAYIP DURUMU (Akıntı Telafili Sanal Fallback)
            # ──────────────────────────────────────────────────────
            if self.use_virtual_dvl:
                rpm_val = float(rpm) if rpm is not None else 1500.0
                u_ss = rpm_val / 600.0

                self._u_virtual += (self.dt / self._tau_surge) * (u_ss - self._u_virtual)
                self._v_virtual += (self.dt / self._tau_sway)  * (0.0 - self._v_virtual)
                self._w_virtual += (self.dt / self._tau_heave) * (0.0 - self._w_virtual)

                v_body_virtual = np.array([self._u_virtual, self._v_virtual, self._w_virtual])
                v_world_virtual = self._R_body2ned @ v_body_virtual

                # YER HIZI TAHMİNİ = Su Bağıl Hızı + Outage Öncesi Öğrenilen Akıntı Hızı
                self.x[3:6] = v_world_virtual + self.x_current

    def update_depth(self, depth_m: float):
        if not self.initialized:
            return

        if self._is_dvl_lost:
            self.x[2] = depth_m
            return

        H = np.zeros((1, 15))
        H[0, 2] = 1.0

        y = np.array([depth_m]) - self.x[2:3]
        S = H @ self.P @ H.T + self.R_depth
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += (K @ y).flatten()
        self.P = (np.eye(15) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

    def get_fossen_state(self, dynamics_nwu: np.ndarray) -> np.ndarray:
        pos_nwu = self.T @ self.x[0:3]
        vel_nwu = self.T @ self.x[3:6]

        out = np.array(dynamics_nwu, dtype=float).copy()
        out[3:6] = vel_nwu
        out[6:9] = pos_nwu

        # Tutum: x[9:12]'den NED Euler → NWU quaternion'a çevir
        phi, theta, psi = self.x[9], self.x[10], self.x[11]
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(phi), -np.sin(phi)], [0.0, np.sin(phi), np.cos(phi)]])
        Ry = np.array([[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]])
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0.0], [np.sin(psi), np.cos(psi), 0.0], [0.0, 0.0, 1.0]])
        R_ned = Rz @ Ry @ Rx
        R_nwu = self.T @ R_ned @ self.T

        quat_nwu = Rotation.from_matrix(R_nwu).as_quat()  # [x, y, z, w]
        out[15:19] = quat_nwu
        return out

    def get_position_ned(self) -> np.ndarray:
        return self.x[0:3].copy()

    def get_position_nwu(self) -> np.ndarray:
        return self.T @ self.x[0:3]

    def get_attitude_ned(self) -> np.ndarray:
        """[phi, theta, psi] radyan cinsinden NED Euler açıları."""
        return self.x[9:12].copy()
