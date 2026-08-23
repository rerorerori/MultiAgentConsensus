"""
AUV Navigation - Advanced 9-State Strapdown INS/EKF
===================================================
Features:
  - 9-State EKF: [p_n, p_e, p_d, v_n, v_e, v_d, b_ax, b_ay, b_az]
  - Double integration of IMU specific force (accelerometer)
  - Online accelerometer bias calibration using active DVL updates
  - DVL Bottom-Lock loss (outage) auto-detection
  - Hard Fallback Switch: Kinematic dead reckoning + Fossen virtual speed integration during DVL loss
  - Saf SINS Modu: İstenildiğinde outage sırasında pure IMU çift entegrasyon seyrüseferi (karşılaştırma için)
  - Direct depth integration to prevent vertical drift

This class replaces the simple 6-state dead reckoning with a resilient, professional seyrüsefer filter.
"""

import numpy as np


class AUVNavigationEKF:
    """
    9-State Strapdown INS / EKF seyrüsefer sınıfı.
    Geriye dönük uyumluluk için sınıf ismi 'AUVNavigationEKF' olarak korunmuştur.
    """

    def __init__(self, dt: float, agent_name: str):
        self.dt          = dt
        self.name        = agent_name
        self.initialized = False

        # Durum vektörü (9 boyutlu):
        # x[0:3] = [pn, pe, pd]  - NED konum (m)
        # x[3:6] = [vn, ve, vd]  - NED hız (m/s)
        # x[6:9] = [bax, bay, baz]- Gövde ekseni ivmeölçer bias sapması (m/s²)
        self.x = np.zeros(9)

        # Başlangıç Kovaryansı P (9x9)
        self.P = np.diag([
            1.0,  1.0,  0.1,   # konum hata varyansı (m²)
            0.05, 0.05, 0.02,  # hız hata varyansı (m²/s²)
            0.02, 0.02, 0.02,  # ivmeölçer bias hata varyansı (m²/s⁴)
        ])

        # Proses Gürültüsü Q (9x9)
        self.Q = np.diag([
            (0.005)**2 * self.dt, (0.005)**2 * self.dt, (0.001)**2 * self.dt, # konum random walk
            (0.02)**2 * self.dt,  (0.02)**2 * self.dt,  (0.005)**2 * self.dt,  # ivmeölçer gürültüsü
            (1e-5)**2 * self.dt,  (1e-5)**2 * self.dt,  (1e-5)**2 * self.dt,   # bias yürüme hızı (random walk bias)
        ])

        # Ölçüm Gürültü Matrisleri
        self.R_dvl         = np.eye(3) * (0.02 ** 2)      # Gerçek DVL gürültüsü (m/s)
        self.R_depth       = np.array([[0.1 ** 2]])       # Derinlik gürültüsü (m)

        # Yerçekimi ivmesi (NED Down ekseninde pozitif)
        self.g_ned = np.array([0.0, 0.0, 9.81])

        # NWU → NED dönüşüm matrisi
        self.T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        # Son bilinen seyrüsefer durumları
        self._euler_ned     = np.zeros(3)  # [phi, theta, psi]
        self._R_body2ned    = np.eye(3)    # Body → NED dönüşüm matrisi
        self._last_gyro_ned = np.zeros(3)  # Gyro açısal hız (NED)

        # Hidrodinamik Sanal Hız Durumları (DVL kaybolduğunda çalışır)
        self._u_virtual = 0.0
        self._v_virtual = 0.0
        self._w_virtual = 0.0
        
        # Surge lag time constant (torpido AUV eylemsizliği için ~3.0s)
        self._tau_surge = 3.0
        self._tau_sway  = 5.0
        self._tau_heave = 5.0

        # DVL kilit kaybı bayrağı
        self._is_dvl_lost = False

        # Hibrit veya Saf SINS Seçimi:
        # True  -> DVL kaybında Fossen sürüklenme-itki sanal hız entegrasyonu (Drift < 5m)
        # False -> DVL kaybında Saf SINS IMU ivme çift entegrasyon seyrüseferi
        self.use_virtual_dvl = True

    # ─────────────────────────────────────────────────────────────
    # BAŞLATMA
    # ─────────────────────────────────────────────────────────────

    def initialize_from_pose(self, pose_nwu: np.ndarray, vel_world_nwu: np.ndarray):
        """
        Sistemi ilk PoseSensor (NWU) ve DynamicsSensor (NWU) dünya hızı ile başlat.
        """
        # Konum NWU → NED
        pos_nwu    = pose_nwu[:3, 3]
        pos_ned    = self.T @ pos_nwu
        self.x[0:3] = pos_ned

        # Attitude NWU → NED
        R_nwu = pose_nwu[:3, :3]
        R_ned = self.T @ R_nwu @ self.T

        phi   = np.arctan2(R_ned[2, 1], R_ned[2, 2])
        theta = -np.arcsin(np.clip(R_ned[2, 0], -1.0, 1.0))
        psi   = np.arctan2(R_ned[1, 0], R_ned[0, 0])

        self._euler_ned  = np.array([phi, theta, psi])
        self._R_body2ned = R_ned.copy()

        # Hız: NWU world → NED world
        vel_ned_world = self.T @ np.array(vel_world_nwu[:3], dtype=float)
        self.x[3:6]   = vel_ned_world

        # İvmeölçer bias tahmini başlangıçta sıfır
        self.x[6:9]   = np.zeros(3)

        # Sanal hızları başlat
        vel_body_ned = R_ned.T @ vel_ned_world
        self._u_virtual = float(vel_body_ned[0])
        self._v_virtual = float(vel_body_ned[1])
        self._w_virtual = float(vel_body_ned[2])

        self.initialized = True

    # ─────────────────────────────────────────────────────────────
    # ATTİTUDE GÜNCELLEMESİ (Her tick çağrılır)
    # ─────────────────────────────────────────────────────────────

    def update_attitude_from_dynamics(self, dynamics_nwu: np.ndarray):
        """
        DynamicsSensor verilerinden araç yönelimini (Attitude) güncelle.
        """
        try:
            from scipy.spatial.transform import Rotation
            quat_nwu = np.array(dynamics_nwu[15:19], dtype=float)
            if np.linalg.norm(quat_nwu) < 0.5:
                return
            R_nwu            = Rotation.from_quat(quat_nwu).as_matrix()
            R_ned            = self.T @ R_nwu @ self.T
            self._R_body2ned = R_ned
            phi   = np.arctan2(R_ned[2, 1], R_ned[2, 2])
            theta = -np.arcsin(np.clip(R_ned[2, 0], -1.0, 1.0))
            psi   = np.arctan2(R_ned[1, 0], R_ned[0, 0])
            self._euler_ned  = np.array([phi, theta, psi])
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────
    # PREDICTION (INS Çift Entegrasyon - Her tick çağrılır)
    # ─────────────────────────────────────────────────────────────

    def predict(self, imu_raw=None):
        """
        IMU ivmeölçer verilerini bias düzeltmesi ve yerçekimi kompanzasyonu ile 
        çift entegre ederek konum ve hız durumlarını ilerletir (Predict adımı).
        
        NOT: DVL kaybolduğunda ve hibrit mod aktifken (use_virtual_dvl=True), 
        ucuz MEMS ivmeölçer gürültüsünün seyrüseferi saniyeler içinde saptırmasını (quadratic drift) 
        önlemek için ivme entegrasyonu tamamen askıya alınır. Hız sanal itki modeli ile sürdürülür.
        
        Eğer Saf SINS modu aktifse (use_virtual_dvl=False), DVL kilit kaybı sırasında da 
        ivmeölçer verileri bias düzeltmesi ile kesintisiz çift entegre edilmeye devam edilir.
        """
        if not self.initialized:
            return

        # DVL outage durumunda ve hibrit mod aktifken ivme entegrasyonunu askıya al, 
        # konumu her tick (30Hz) mevcut tahmini hız ile kararlı ilerlet
        if self._is_dvl_lost and self.use_virtual_dvl:
            self.x[0:3] += self.dt * self.x[3:6]
            return

        # SINS İvme Entegrasyon Hazırlığı (Normal EKF veya Saf SINS outage modu)
        a_ned = np.zeros(3)
        
        if imu_raw is not None:
            imu = np.array(imu_raw, dtype=float).flatten()
            if len(imu) >= 6:
                # 1. Ham ivmeyi (ilk 3 eleman, NWU) NED Gövde çerçevesine dönüştür
                f_raw_body = self.T @ imu[0:3]
                
                # 2. Tahmini Bias değerini çıkar (f_corrected = f_raw - b_a)
                b_a = self.x[6:9]
                f_corrected = f_raw_body - b_a
                
                # 3. Yönelim matrisi ile NED dünyasına döndür ve yerçekimini çıkar
                # Specific force zaten gravity içerdiği için NED ivmesinden yerçekimi çıkarılır
                a_ned = self._R_body2ned @ f_corrected - self.g_ned
                
                # 4. Gyro verisini (sonraki 3 eleman, NWU) NED olarak kaydet
                self._last_gyro_ned = self.T @ imu[3:6]
        
        # 5. Çift Entegrasyon / Konum İlerletmesi
        # p_new = p + dt * v + 0.5 * dt^2 * a
        self.x[0:3] += self.dt * self.x[3:6] + 0.5 * (self.dt ** 2) * a_ned
        # v_new = v + dt * a
        self.x[3:6] += self.dt * a_ned

        # 6. Kovaryans İlerletme (Phi = I + F * dt)
        F = np.zeros((9, 9))
        F[0:3, 3:6] = np.eye(3)
        
        # Eğer DVL kayıpsa ve bias dondurulmuşsa, d(vel)/d(bias) türevi sıfır alınır
        if not self._is_dvl_lost:
            F[3:6, 6:9] = -self._R_body2ned

        Phi = np.eye(9) + F * self.dt
        self.P = Phi @ self.P @ Phi.T + self.Q
        
        # Simetriyi garanti et
        self.P = 0.5 * (self.P + self.P.T)

    # ─────────────────────────────────────────────────────────────
    # DVL GÜNCELLEMESİ / OUTAGE YÖNETİMİ (~10 Hz)
    # ─────────────────────────────────────────────────────────────

    def update_dvl(self, dvl_raw: np.ndarray, rpm: float = None):
        """
        DVL hız ölçümleriyle seyrüsefer ve IMU bias kalibrasyonunu gerçekleştirir.
        DVL bottom-lock kaybı durumunda basitleştirilmiş Fossen itki-sürüklenme (virtual DVL)
        modeline geçiş yapar ve drifti sınırlar.
        
        Args:
            dvl_raw: HoloOcean DVL Sensor verisi (7 elemanlı NWU gövde hızı) veya None (Outage)
            rpm: Ajanın o anki motor devri (sanal hız tahmini için gerekir)
        """
        if not self.initialized:
            return

        # 1. DVL Outage (Kayıp) Tespiti
        is_dvl_lost = False
        if dvl_raw is None:
            is_dvl_lost = True
        else:
            dvl = np.array(dvl_raw, dtype=float).flatten()
            # Eğer tüm hız girdileri 0 veya NaN ise bottom lock kaybedilmiştir
            if np.all(np.abs(dvl[0:3]) < 1e-5) or np.isnan(dvl[0]):
                is_dvl_lost = True

        # Dahili outage bayrağını güncelle
        self._is_dvl_lost = is_dvl_lost

        if not is_dvl_lost:
            # ──────────────────────────────────────────────────────
            # A. DVL AKTİF DURUM (Normal ESKF Güncellemesi & Bias Öğrenme)
            # ──────────────────────────────────────────────────────
            v_dvl_nwu = dvl[0:3]
            v_dvl_ned_body = self.T @ v_dvl_nwu
            v_dvl_ned_world = self._R_body2ned @ v_dvl_ned_body

            # EKF için ölçüm matrisi H (3x9): Ölçüm sadece Hız durumlarını doğrudan okur
            H = np.zeros((3, 9))
            H[0:3, 3:6] = np.eye(3)

            y = v_dvl_ned_world - self.x[3:6]
            S = H @ self.P @ H.T + self.R_dvl
            K = self.P @ H.T @ np.linalg.inv(S)

            # Durum vektörünü güncelle (Bias bu adımda hız hatası üzerinden çevrimiçi kalibre edilir)
            self.x += K @ y
            self.P = (np.eye(9) - K @ H) @ self.P
            
            # Kovaryans simetrisini koru
            self.P = 0.5 * (self.P + self.P.T)

            # Sanal hızları senkronize et
            self._u_virtual = float(v_dvl_ned_body[0])
            self._v_virtual = float(v_dvl_ned_body[1])
            self._w_virtual = float(v_dvl_ned_body[2])
            
        else:
            # ──────────────────────────────────────────────────────
            # B. DVL KAYIP DURUMU
            # ──────────────────────────────────────────────────────
            if self.use_virtual_dvl:
                # [DOĞRUDAN KİNEMATİK HIZ ATAMASI] (Hibrit Sanal DVL Modu)
                rpm_val = float(rpm) if rpm is not None else 1500.0
                u_ss = rpm_val / 600.0
                
                self._u_virtual += (self.dt / self._tau_surge) * (u_ss - self._u_virtual)
                self._v_virtual += (self.dt / self._tau_sway) * (0.0 - self._v_virtual)
                self._w_virtual += (self.dt / self._tau_heave) * (0.0 - self._w_virtual)

                v_body_virtual = np.array([self._u_virtual, self._v_virtual, self._w_virtual])
                v_world_virtual = self._R_body2ned @ v_body_virtual

                self.x[3:6] = v_world_virtual
            else:
                # [SAF SINS MODU] (Bypass EKF update, predict() içinde ivme entegrasyonu devam eder)
                pass

    # ─────────────────────────────────────────────────────────────
    # DEPTH GÜNCELLEMESİ (Her tick çağrılır)
    # ─────────────────────────────────────────────────────────────

    def update_depth(self, depth_m: float):
        """
        Derinlik ölçümüyle dikey drifti kesin olarak sıfırlar.
        """
        if not self.initialized:
            return

        # DVL outage durumunda derinliği doğrudan güncelle, EKF matris işlemlerini ve kovaryans sızıntısını bypass et
        if self._is_dvl_lost:
            self.x[2] = depth_m
            return

        # H matrisi (1x9) - Sadece pd durumunu okur (x[2])
        H = np.zeros((1, 9))
        H[0, 2] = 1.0

        y = np.array([depth_m]) - self.x[2:3]
        S = H @ self.P @ H.T + self.R_depth
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += (K @ y).flatten()
        self.P = (np.eye(9) - K @ H) @ self.P
        
        # Simetriyi garanti et
        self.P = 0.5 * (self.P + self.P.T)

    # ─────────────────────────────────────────────────────────────
    # FOSSEN STATE ÇIKTISI
    # ─────────────────────────────────────────────────────────────

    def get_fossen_state(self, dynamics_nwu: np.ndarray) -> np.ndarray:
        """
        Autopilot ve Fossen kontrolcüleri için hibrit seyrüsefer durum çıktısı üretir.
        Geliştirilen EKF seyrüsefer konum ve hızını entegre eder.
        """
        # Konum tahmini NED → NWU
        pos_nwu = self.T @ self.x[0:3]

        # Hız tahmini NED world → NWU world
        vel_nwu = self.T @ self.x[3:6]

        out = np.array(dynamics_nwu, dtype=float).copy()
        out[3:6] = vel_nwu   # Hız -> EKF tahmini
        out[6:9] = pos_nwu   # Konum -> EKF tahmini

        return out

    # ─────────────────────────────────────────────────────────────
    # DURUM SORGULAMA METOTLARI
    # ─────────────────────────────────────────────────────────────

    def get_position_ned(self) -> np.ndarray:
        """EKF konum tahminini NED [n, e, d] formatında döndürür."""
        return self.x[0:3].copy()

    def get_position_nwu(self) -> np.ndarray:
        """EKF konum tahminini NWU [x, y, z] formatında döndürür."""
        return self.T @ self.x[0:3]

    def get_eta_nu_ned(self):
        """Debug için NED çerçevesinde durumları döndürür."""
        phi, theta, psi = self._euler_ned
        eta = np.array([self.x[0], self.x[1], self.x[2],
                        phi, theta, psi])
        nu  = np.zeros(6)
        nu[0:3] = self._R_body2ned.T @ self.x[3:6]
        nu[3:6] = self._last_gyro_ned
        return eta, nu
