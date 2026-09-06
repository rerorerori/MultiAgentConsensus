"""

"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Beacon "is still transmitting" spam'ini sustur
import io as _io

class _FilteredStdout:
    """Sade stdout wrapper — TextIOWrapper'dan turetme yok, super().__init__ riski yok.
    __getattr__ ile eksik tum metodlar orijinal stream'e yonlendirilir."""
    _SUPPRESS = ("is still transmitting",)
    def __init__(self, wrapped):
        self._wrapped = wrapped
    def write(self, s):
        if any(kw in s for kw in self._SUPPRESS):
            return len(s)
        return self._wrapped.write(s)
    def flush(self):
        return self._wrapped.flush()
    def fileno(self):
        return self._wrapped.fileno()
    def __getattr__(self, name):
        return getattr(self._wrapped, name)

# GUI modunda stdout replace'i devre disi birak (HoloOcean rendering ile uyumsuzluk olabilir)
# Sadece headless/CI ortaminda aktif et:
FILTER_BEACON_SPAM = False
if FILTER_BEACON_SPAM:
    sys.stdout = _FilteredStdout(sys.stdout)


APF_INFLUENCE_DIST = 50.0
APF_K_REP = 800.0
SAFE_MARGIN = 170.0

import holoocean
from holoocean.fossen_dynamics.fossen_interface import FossenInterface
import numpy as np
import json, traceback, argparse
import time as _time

# Ege akıntılı konfigürasyonu yükle
from config_current import (
    AGENT_NAMES, SCENARIO_PATH, TICKS_PER_SEC,
    CRUISE_RPM, CRUISE_DEPTH, MIN_RPM, MAX_RPM,
    WP_ACCEPT_R, UTURN_THRESH,
    STUCK_TIME_S, STUCK_DIST_M, STARTUP_GRACE_S,
    POST_RECOVERY_GRACE_S, RECOVERY_DUR_S,
    RECOVERY_HEADING_OFF, RECOVERY_DEPTH_OFF, RECOVERY_RPM,
    FLS_THRESHOLD, FLS_BIAS_HARD, FLS_BIAS_SOFT,
    BEACON_SEND_INTERVAL, PACKET_LOSS_RATE,
    GUI_UPDATE_TICKS,
    wrap_heading, heading_diff, BOUNDS_X,
    USE_OCEAN_CURRENT, BASE_CURRENT_NED, CURRENT_NOISE_VAR, CURRENT_TAU
)

from core.consensus import ConsensusMapFusion
from core.entropy_planner import EntropyGuidedPlanner
from core.comm_quality import CommQualityMonitor
from core.comm_manager import CommManager
from core.stuck_detector import StuckDetector
from visualization.dashboard import Dashboard
from core.logger import MissionLogger


def get_obstacle_bias(fls_data):
    if fls_data is None or len(fls_data.shape) != 2:
        return 0.0, False
    n_az = fls_data.shape[1]
    s = n_az // 3
    left   = np.max(fls_data[:, :s])
    center = np.max(fls_data[:, s:2*s])
    right  = np.max(fls_data[:, 2*s:])
    risk = center > FLS_THRESHOLD
    bias = 0.0
    if risk:
        bias = -FLS_BIAS_HARD if left < right else FLS_BIAS_HARD
    elif left > FLS_THRESHOLD:
        bias = FLS_BIAS_SOFT
    elif right > FLS_THRESHOLD:
        bias = -FLS_BIAS_SOFT
    return bias, risk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="proposed", choices=["proposed", "lawnmower", "random", "entropy_only"])
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=int, default=0, help="Simülasyon süresi (tick). 0=sonsuz")
    parser.add_argument("--log", type=str, default=None, help="Log dosyasının yolu")
    parser.add_argument("--nav", default="ekf", choices=["ground_truth", "ekf"])
    parser.add_argument("--fixed_weight", action="store_true", help="Ablasyon: adaptif w_ij yerine sabit epsilon")
    parser.add_argument("--current_mag", type=float, default=0.2236, help="Akıntı büyüklüğü (m/s). Yön korunur, büyüklük ölçeklenir.")
    parser.add_argument("--seed", type=int, default=42, help="Rastgelelik tohumu (seed)")
    args = parser.parse_args()

    # Seed ayarla
    np.random.seed(args.seed)

    # Akıntı büyüklüğünü ölçekle (mevcut yön korunur, büyüklük ayarlanır)
    base_curr_nom = np.array(BASE_CURRENT_NED, dtype=float)
    nom_mag = float(np.linalg.norm(base_curr_nom[:2]))
    if nom_mag > 1e-6:
        base_current_vec = base_curr_nom * (args.current_mag / nom_mag)
    else:
        base_current_vec = np.array([-args.current_mag, 0.0, 0.0])

    mode     = args.mode
    nav_mode = args.nav
    print("=" * 77)
    print(f"  Swarm AUV — V2: Hidrodinamik Akıntı Modeli (Fossen V_c/beta_c)")
    print(f"  Mode: {mode.upper()} | Navigation: {nav_mode.upper()}")
    print(f"  Weight: {'FIXED (ablation)' if args.fixed_weight else 'ADAPTIVE (w_ij)'}")
    print(f"  Okyanus Akintisi: {'AKTIF (Gauss-Markov -> Fossen)' if USE_OCEAN_CURRENT else 'PASIF'} | Mag: {args.current_mag:.2f} m/s | Seed: {args.seed}")
    print("=" * 77)

    comm_mgr  = CommManager()
    consensus = ConsensusMapFusion(comm_manager=comm_mgr, fixed_weight=args.fixed_weight)
    comm_qual = CommQualityMonitor()
    stuck     = StuckDetector(
        agent_names=AGENT_NAMES,
        cruise_depth=CRUISE_DEPTH,
        stuck_time_s=STUCK_TIME_S,
        stuck_dist_m=STUCK_DIST_M,
        startup_grace_s=STARTUP_GRACE_S,
        post_recovery_grace_s=POST_RECOVERY_GRACE_S,
        recovery_dur_s=RECOVERY_DUR_S,
    )

    # K4 FIX: ablation run’ları arasında registry’yi temizle
    EntropyGuidedPlanner.reset_registry()

    planners = {}
    if mode == "proposed":
        for name in AGENT_NAMES:
            planners[name] = EntropyGuidedPlanner(consensus.maps[name], agent_name=name, consensus=consensus)

    targets = {n: None for n in AGENT_NAMES}
    agents_metadata = {n: {} for n in AGENT_NAMES}

    logger = None
    if args.log:
        logger = MissionLogger(args.log, AGENT_NAMES)

    # Akıntı telafili EKF seyrüsefercileri
    from core.ekf_navigation_current import AUVNavigationEKF
    navigators = {}
    if nav_mode == "ekf":
        for name in AGENT_NAMES:
            navigators[name] = AUVNavigationEKF(dt=1.0 / TICKS_PER_SEC, agent_name=name)
        print("[INIT] Akıntı telafili 15-durumlu EKF seyrüseferi başlatıldı (Tutum ve Bias Kestirimli).", flush=True)

    try:
        with open(SCENARIO_PATH, "r") as f:
            scenario = json.load(f)

        fossen = FossenInterface(AGENT_NAMES, scenario, multi_agent=True)
        env    = holoocean.make(scenario_cfg=scenario)
        env.should_render_viewport(not args.headless)
        _time.sleep(1)

        dashboard = None
        if not args.headless:
            dashboard = Dashboard(consensus, comm_manager=comm_mgr)

        # ── V2: current_drifts KALDIRILDI ──
        # Akıntı artık Fossen torpedo modelinin V_c/beta_c üzerinden uygulanıyor.
        # Kinematik pozisyon offset birikimi (current_drifts) yoktur.
        dt = 1.0 / TICKS_PER_SEC
        current_ned    = base_current_vec.copy()
        turbulence_ned = np.zeros(3)
        current_nwu    = np.zeros(3)  # log/EKF icin tutulur

        # Akinti min/max penceresi (son 900 tick = 30s)
        CURR_WINDOW = 900
        current_norm_window = []

        step = 0
        while True:
            if args.duration > 0 and step >= args.duration:
                break

            sim_time = step / TICKS_PER_SEC
            states = env.tick()
            comm_mgr.flush_delayed(env, step)

            # Akıntı Türbülans Güncellemesi — Ornstein-Uhlenbeck (Euler-Maruyama)
            # DOGRU: noise sqrt(dt) ile ölçeklenir → türbülans BASE_CURRENT_NED
            # civarında kalır, sonsuz şişmez.
            if USE_OCEAN_CURRENT:
                noise = np.random.normal(0, 1, 3)
                noise[2] = 0.0  # Dikey akıntıyı sıfır tut
                turbulence_ned = (turbulence_ned
                                  - (turbulence_ned / CURRENT_TAU) * dt
                                  + np.sqrt(CURRENT_NOISE_VAR) * np.sqrt(dt) * noise)
                current_ned = base_current_vec + turbulence_ned

                # Akinti normunu pencereye ekle (min/max takibi)
                c_norm = float(np.linalg.norm(current_ned[:2]))
                current_norm_window.append(c_norm)
                if len(current_norm_window) > CURR_WINDOW:
                    current_norm_window.pop(0)

                # NED to NWU (log/EKF için)
                T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                current_nwu = T @ current_ned

                # ── V2: Akıntıyı Fossen hidrodinamik modeline geçir ──
                # torpedo.dynamics(): nu_r = nu - nu_c  (göreceli hız doğru hesaplanır)
                # beta_c: NED yatay düzlemde arctan2(E, N) → radyan, __init__ bypass
                V_c    = float(np.linalg.norm(current_ned[:2]))
                beta_c = float(np.arctan2(current_ned[1], current_ned[0]))  # radyan
                for n in AGENT_NAMES:
                    fossen.vehicles[n].V_c    = V_c
                    fossen.vehicles[n].beta_c = beta_c  # torpedo.py D2R'ı __init__'te yapar
                    # Direkt attribute set → dynamics() radyan olarak okur ✓

            # Beacon ID
            if step == 0:
                real_id_map = {}
                for name in AGENT_NAMES:
                    if hasattr(env, "agents") and name in env.agents:
                        a = env.agents[name]
                        if "AcousticBeaconSensor" in a.sensors:
                            real_id_map[name] = a.sensors["AcousticBeaconSensor"].id
                if len(real_id_map) == len(AGENT_NAMES):
                    comm_mgr.update_ids(real_id_map)

            # ── V2: Konumlar — SNR mesafesi için EKF tahmini (başlatılmışsa), yoksa GT fallback ──
            # Thorp SNR hesabı GT yerine EKF pozisyonu kullanarak gerçekçiliği artırır.
            agent_positions = {}
            for name in AGENT_NAMES:
                if name in states and "PoseSensor" in states[name]:
                    if (nav_mode == "ekf"
                            and name in navigators
                            and navigators[name].initialized):
                        # EKF pozisyon tahmini (NWU) — GT'den bağımsız
                        ekf_pos_nwu = navigators[name].get_position_nwu()
                        gt_z = states[name]["PoseSensor"][2, 3]   # derinlik GT (yüzey mesafesi)
                        agent_positions[name] = np.array([ekf_pos_nwu[0], ekf_pos_nwu[1], gt_z])
                    else:
                        # EKF hazır değilse GT fallback (başlangıç 90 tick)
                        agent_positions[name] = states[name]["PoseSensor"][:3, 3]
            comm_mgr.update_positions(agent_positions)

            # Per-agent döngü
            for name in AGENT_NAMES:
                if name not in states:
                    continue

                # ── V2: Pose ve pozisyon — drift offset YOK ──
                pose = states[name]["PoseSensor"].copy()
                pos  = pose[:3, 3]
                rot  = pose[:3, :3]
                yaw  = np.arctan2(rot[1, 0], rot[0, 0])
                hdg  = np.degrees(yaw)

                # ── V2: VelocitySensor — current_nwu EKLENMEZ ──
                # Fossen akıntıyı modellediği için AUV zaten akıntıyla sürükleniyor.
                # Manuel ekleme çift sayım yaratır.
                vel_raw = states[name].get("VelocitySensor", np.zeros(3))
                vel = float(np.linalg.norm(vel_raw[:2]))

                ss_data = states[name].get("SSS", states[name].get("SidescanSonar"))
                if ss_data is not None and len(ss_data) > 0:
                    consensus.maps[name].bayesian_update_from_sidescan(pos, yaw, ss_data)
                else:
                    consensus.update_local(name, pos, yaw)

                # SNR & Kanal
                for other in AGENT_NAMES:
                    if other != name:
                        snr = comm_mgr.get_snr(name, other)
                        if not np.isnan(snr):
                            comm_qual.log_snr(name, other, snr)

                comm_state = consensus.update_comm_state(name, sim_time=sim_time)
                comm_qual.log_comm_state(name, comm_state, sim_time=sim_time)

                # Engel ve Sıkışma
                coll = bool(np.any(states[name].get("CollisionSensor", False)))
                stuck.update_heading(name, hdg)
                if (coll or stuck.check_stuck(name, pos, sim_time)) and stuck.status[name] == "OK":
                    stuck.enter_recovery(name, hdg, abs(pos[2]), sim_time)
                    planners[name].current_target = None
                    planners[name].global_target = None
                    planners[name]._last_global_replan_tick = -99999
                elif stuck.status[name] == "RECOVERING" and stuck.is_done(name, sim_time):
                    stuck.finish_recovery(name, pos, sim_time)
                    planners[name].current_target = None
                    planners[name].global_target = None
                    planners[name]._last_global_replan_tick = -99999

                # APF & Sınırlar
                final_rpm_used = CRUISE_RPM
                if stuck.status[name] == "OK":
                    neighbor_grids = None
                    if comm_state != "OFFLINE":
                        neighbor_grids = {n: consensus.maps[n].grid for n in AGENT_NAMES if n != name}

                    # Akıntı ve rota yönü ilişkisine bağlı akıllı dinamik kabul yarıçapı
                    heading_vec = np.array([np.cos(yaw), np.sin(yaw)])
                    current_along_track = np.dot(current_nwu[:2], heading_vec) if USE_OCEAN_CURRENT else 0.0
                    planners[name].wp_accept_r = 15.0 + 15.0 * max(0.0, current_along_track)

                    tgt_h, tgt_rpm, tgt_wp = planners[name].plan(
                        pos, yaw, step, neighbor_grids=neighbor_grids, sim_time=sim_time)
                    targets[name] = tgt_wp

                    hdiff = abs(heading_diff(hdg, tgt_h))
                    if hdiff > UTURN_THRESH:
                        tgt_rpm = MIN_RPM + 200

                    final_h   = wrap_heading(tgt_h)
                    final_rpm = int(np.clip(tgt_rpm, MIN_RPM, MAX_RPM))

                    # APF Sınır İtimi
                    px, py = float(pos[0]), float(pos[1])
                    dx_edge = SAFE_MARGIN - abs(px)
                    dy_edge = SAFE_MARGIN - abs(py)
                    F_rep = np.zeros(2)
                    if dx_edge < APF_INFLUENCE_DIST:
                        dx_eff = max(dx_edge, 1.0)
                        mag = APF_K_REP * (1/dx_eff - 1/APF_INFLUENCE_DIST) / (dx_eff**2)
                        F_rep[0] += -np.sign(px) * mag
                    if dy_edge < APF_INFLUENCE_DIST:
                        dy_eff = max(dy_edge, 1.0)
                        mag = APF_K_REP * (1/dy_eff - 1/APF_INFLUENCE_DIST) / (dy_eff**2)
                        F_rep[1] += -np.sign(py) * mag

                    if np.linalg.norm(F_rep) > 0.01:
                        current_vec = np.array([np.cos(np.radians(final_h)), np.sin(np.radians(final_h))])
                        min_edge = min(dx_edge, dy_edge)
                        blend = 1.0 if min_edge <= 0 else 0.4 + 0.6 * (1 - min_edge / APF_INFLUENCE_DIST)
                        F_total = current_vec + F_rep * blend
                        final_h = wrap_heading(np.degrees(np.arctan2(F_total[1], F_total[0])))

                    # Sınır dışı acil merkeze dönüş
                    if abs(pos[0]) > 170.0 or abs(pos[1]) > 170.0:
                        inward_h = np.degrees(np.arctan2(-pos[1], -pos[0]))
                        final_h = wrap_heading(inward_h)
                        final_rpm = MIN_RPM + 300
                        targets[name] = np.array([0.0, 0.0])

                    final_rpm_used = final_rpm

                    cmd_depth = CRUISE_DEPTH

                    fossen.set_goal(name, depth=max(0.5, cmd_depth), heading=final_h, rpm=final_rpm)
                else:
                    if abs(pos[0]) > 170.0 or abs(pos[1]) > 170.0:
                        stuck.finish_recovery(name, pos, sim_time)
                        inward_h = np.degrees(np.arctan2(-pos[1], -pos[0]))
                        fossen.set_goal(name, depth=max(0.5, CRUISE_DEPTH), heading=wrap_heading(inward_h), rpm=MIN_RPM + 300)
                        final_rpm_used = MIN_RPM + 300
                    else:
                        fossen.set_goal(name, depth=max(0.5, stuck.recovery_depth[name]), heading=stuck.recovery_heading[name], rpm=RECOVERY_RPM)
                        final_rpm_used = RECOVERY_RPM

                # ── EKF SEYRÜSEFER (AKINTI TAHMİNLİ) ──
                dr_state = None
                if nav_mode == "ekf" and name in navigators:
                    nav     = navigators[name]
                    imu_raw = states[name].get("IMUSensor")

                    # ── V2: DVL'e manuel current EKLENMEZ ──
                    # Fossen akıntıyı modelliyor → AUV zaten akıntıyla hareket ediyor
                    # → DVLSensor yer hızını (akıntı dahil) doğal olarak ölçüyor
                    dvl_raw = states[name].get("DVLSensor").copy() if "DVLSensor" in states[name] else None
                    mag_raw = states[name].get("MagnetometerSensor", None)

                    dep_raw = states[name].get("DepthSensor")
                    depth_m = float(abs(dep_raw[0])) if dep_raw is not None and len(dep_raw) > 0 else abs(float(pos[2]))

                    if not nav.initialized:
                        # ── V2: Başlangıç — drift offset YOK ──
                        pose_shifted = states[name]["PoseSensor"].copy()
                        vel_shifted  = np.array(states[name]["DynamicsSensor"][3:6], dtype=float)
                        nav.initialize_from_pose(pose_shifted, vel_shifted)
                    else:
                        # ── V2: DynamicsSensor'a manuel current/drift EKLENMEZ ──
                        dynamics_shifted = states[name]["DynamicsSensor"].copy()

                        # 15-Durumlu EKF: Tutum IMU gyro entegrasyonuyla kendi içinde kestirilir (GT sızıntısı yok)
                        if imu_raw is not None:
                            nav.predict(np.array(imu_raw, dtype=float))
                        if step % 3 == 0:
                            dvl_input = np.array(dvl_raw, dtype=float) if dvl_raw is not None else None
                            nav.update_dvl(dvl_input, rpm=final_rpm_used)
                        nav.update_depth(depth_m)
                        if mag_raw is not None:
                            nav.update_magnetometer(mag_raw)

                        if step >= 90:
                            dr_state = nav.get_fossen_state(dynamics_shifted)

                # Fossen güncelleme
                accel = fossen.update(name, states, dr_state=dr_state)
                env.act(name, accel)

                # Raporlama
                target_pos = targets.get(name, np.array([0.0, 0.0]))
                sd_entry = {
                    "x": pos[0], "y": pos[1], "z": pos[2],
                    "heading": hdg, "vel": vel,
                    "rpm": final_rpm_used,
                    "depth_cmd": CRUISE_DEPTH,
                    "target_x": target_pos[0] if target_pos is not None else 0.0,
                    "target_y": target_pos[1] if target_pos is not None else 0.0,
                    "status": stuck.status[name],
                }

                if nav_mode == "ekf" and name in navigators and navigators[name].initialized:
                    ned_est = navigators[name].x[0:3]
                    ekf_x = ned_est[0]
                    ekf_y = -ned_est[1]
                    nav_err = float(np.sqrt((pos[0] - ekf_x)**2 + (pos[1] - ekf_y)**2))
                    sd_entry["nav_error"] = nav_err
                    sd_entry["ekf_x"] = ekf_x
                    sd_entry["ekf_y"] = ekf_y
                    sd_entry["curr_true_n"] = current_ned[0]
                    sd_entry["curr_true_e"] = current_ned[1]
                    sd_entry["curr_true_mag"] = float(np.linalg.norm(current_ned[:2]))
                    sd_entry["curr_est_n"] = navigators[name].x_current[0]
                    sd_entry["curr_est_e"] = navigators[name].x_current[1]
                    sd_entry["curr_est_mag"] = float(np.linalg.norm(navigators[name].x_current[:2]))

                agents_metadata[name] = sd_entry
                if dashboard:
                    dashboard.sensor_data[name] = sd_entry

            # Akustik Haberleşme (proposed)
            if mode == "proposed":
                for sender in AGENT_NAMES:
                    if comm_mgr.should_send_tdma(sender, step):
                        current_wp = planners[sender].current_target
                        payload = consensus.encode_map_for_beacon(sender, current_target=current_wp)
                        if payload:
                            msgs = comm_mgr.prepare_messages(sender, payload, agent_positions=agent_positions, current_tick=step)
                            for fid, tid, d in msgs:
                                comm_qual.log_send(sender, sim_time=sim_time)
                                ok = comm_mgr.send(env, sender, tid, d, current_tick=step)
                                if ok and dashboard:
                                    recv_name = comm_mgr.rev_id_map.get(tid, "")
                                    snr = comm_mgr.get_snr(sender, recv_name)
                                    dashboard.trigger_beacon_pulse(sender, recv_name, snr)

            # Beacon Alımı
            beacon_received_this_tick = {n: False for n in AGENT_NAMES}
            if mode == "proposed":
                for receiver in AGENT_NAMES:
                    raw = states[receiver].get("AcousticBeaconSensor")
                    if raw is not None:
                        parsed_list = comm_mgr.parse_beacon_data(raw, receiver=receiver, sim_time=sim_time)
                        for parsed in parsed_list:
                            sender_name = parsed["from_name"]
                            payload     = parsed["payload"]
                            comm_qual.log_receive(receiver, sender_name, sim_time=sim_time)
                            if isinstance(payload, (bytes, bytearray, list)):
                                ok = consensus.decode_map_from_beacon(receiver, sender_name, payload, sim_time=sim_time)
                                if ok:
                                    # O2 FIX: RECONNECTING state’de harita senkronizasyonu
                                    if consensus.get_comm_state(receiver) == "RECONNECTING":
                                        consensus.sync_on_reconnect(receiver)
                                        log_msg = f"[SYNC] {receiver} harita senkronize edildi"
                                        print(log_msg, flush=True)
                                        if dashboard:
                                            dashboard.comm_log.append(log_msg)
                                    else:
                                        consensus.consensus_update(receiver, sim_time=sim_time)
                                        beacon_received_this_tick[receiver] = True

                # Periyodik Consensus
                if step > 0 and step % BEACON_SEND_INTERVAL == 0:
                    for name in AGENT_NAMES:
                        if not beacon_received_this_tick[name]:
                            consensus.consensus_update(name, sim_time=sim_time)
                    rms = consensus.get_consensus_rms()
                    if step % 900 == 0:
                        cov_pct    = consensus.get_avg_coverage_pct()
                        states_str = " | ".join(
                            f"{n}:{consensus.get_comm_state(n)[:3]}" for n in AGENT_NAMES)
                        print(f"[CONS] step={step:5d} t={sim_time:6.1f}s  "
                              f"RMS={rms:.4f}  Cov={cov_pct:.1f}%  | {states_str}", flush=True)

                        # [POS] -- her aracin konumu, boundary disi (*) isaretli
                        pos_parts = []
                        for nn in AGENT_NAMES:
                            p = agent_positions.get(nn, np.zeros(3))
                            flag = "*" if (abs(p[0]) > 180 or abs(p[1]) > 180) else " "
                            pos_parts.append(f"{nn}:({p[0]:+.0f},{p[1]:+.0f}){flag}")
                        print(f"[POS]  " + "  ".join(pos_parts), flush=True)

                        if nav_mode == "ekf" and navigators:
                            nav_errs      = []
                            current_norms = []
                            for nn in AGENT_NAMES:
                                if nn in navigators and navigators[nn].initialized:
                                    gt_p = agent_positions.get(nn, np.zeros(3))
                                    ekf_nwu = navigators[nn].get_position_nwu()
                                    err  = float(np.sqrt(
                                        (gt_p[0] - ekf_nwu[0])**2 + (gt_p[1] - ekf_nwu[1])**2))
                                    nav_errs.append(err)
                                    current_norms.append(
                                        np.linalg.norm(navigators[nn].x_current))
                            if nav_errs:
                                print(f"[NAV]  EKF Hata -> Ort:{np.mean(nav_errs):.2f}m  "
                                      f"Maks:{np.max(nav_errs):.2f}m", flush=True)

                        # [CURR] -- akinti detayi + son pencere min/max
                        c_now  = float(np.linalg.norm(current_ned[:2]))
                        c_min  = min(current_norm_window) if current_norm_window else c_now
                        c_max  = max(current_norm_window) if current_norm_window else c_now
                        ekf_c  = (np.mean([np.linalg.norm(navigators[nn].x_current)
                                           for nn in AGENT_NAMES
                                           if nn in navigators and navigators[nn].initialized])
                                  if nav_mode == "ekf" else 0.0)
                        print(f"[CURR] gercek={c_now:.3f} m/s  "
                              f"pencere min/max={c_min:.3f}/{c_max:.3f}  "
                              f"EKF ogrenilen={ekf_c:.3f} m/s", flush=True)
                        print("-" * 65, flush=True)

            if dashboard and step % GUI_UPDATE_TICKS == 0:
                dashboard.update(states, targets, comm_qual, mode)

            if logger:
                snr_mat = comm_qual.get_snr_matrix() if mode == "proposed" else None
                logger.log(
                    step=step,
                    sim_time=sim_time,
                    coverage=consensus.get_avg_coverage_pct(),
                    rms=consensus.get_consensus_rms(),
                    pdr=comm_qual.get_summary()["global_pdr"],
                    agents_state=agents_metadata,
                    snr_matrix=snr_mat,
                )
                if step > 0 and step % 1000 == 0:
                    logger.save()

            step += 1

    except KeyboardInterrupt:
        print("\n[STOP] Kullanıcı durdurdu.")
    except Exception:
        traceback.print_exc()
    finally:
        if "logger" in locals() and logger:
            logger.save()
        if "env" in locals():
            try:
                env.__on_exit__()
            except Exception:
                pass


if __name__ == "__main__":
    main()
