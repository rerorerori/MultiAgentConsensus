"""
Communication-Aware Cooperative Coverage — Main Simulation Loop v2.1
====================================================================
Yenilikler (v2.1):
  • flush_delayed() her tick çağrılır — gecikme kuyruğu aktif
  • Beacon döngüsü değişkeni "receiver" (shadow önleme)
  • RPM sensor_data'ya final değerle yazılır (recovery/u-turn gerçek)
  • Çift consensus update önlendi (step koşulu netleştirildi)
  • SNR kaynağı comm_mgr.get_snr_matrix() (public API)
  • Binary encode beacon payload (Subnero 256B uyumu)
  • Broadcast gönderim (consensus yakınsama hızlandırıldı)

Kullanım:
    python main.py --mode proposed
    python main.py --mode lawnmower
    python main.py --mode random
    python main.py --mode entropy_only
    python main.py --mode proposed --headless
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

APF_INFLUENCE_DIST = 50.0  # m, sınırdan bu kadar içeride devreye girer
APF_K_REP = 800.0          # itici kuvvet katsayısı
SAFE_MARGIN = 170.0        # mevcut B_VAL ile aynı

import holoocean
from holoocean.fossen_dynamics.fossen_interface import FossenInterface
import numpy as np
import json, traceback, argparse

from config import (
    AGENT_NAMES, SCENARIO_PATH, TICKS_PER_SEC,
    CRUISE_RPM, CRUISE_DEPTH, MIN_RPM, MAX_RPM,
    UTURN_THRESH, STUCK_TIME_S,
    STUCK_DIST_M, STARTUP_GRACE_S, POST_RECOVERY_GRACE_S,
    RECOVERY_DUR_S, RECOVERY_HEADING_OFF, RECOVERY_RPM,
    FLS_THRESHOLD, FLS_BIAS_HARD, FLS_BIAS_SOFT,
    BEACON_SEND_INTERVAL, GUI_UPDATE_TICKS, wrap_heading,
    heading_diff
)

# Yeni modüller
from core.consensus import ConsensusMapFusion
from core.entropy_planner import EntropyGuidedPlanner
from core.comm_quality import CommQualityMonitor

# Güncellenmiş CommManager (Thorp + TDMA + Delay Queue)
from core.comm_manager import CommManager

# Baselines
from baselines.lawnmower import LawnmowerPlanner
from baselines.random_walk import RandomWalkPlanner
from baselines.entropy_only import EntropyOnlyPlanner

# Dashboard
from visualization.dashboard import Dashboard

# Loglama
from core.logger import MissionLogger
from core.stuck_detector import StuckDetector



# ═══════════════════════════════════════════════════════════════
# FLS ENGEL KAÇINMA
# ═══════════════════════════════════════════════════════════════
def get_obstacle_bias(fls_data):
    if fls_data is None or len(fls_data.shape) != 2:
        return 0.0, False
    n_az = fls_data.shape[1]  # Sütunlar (azimuth açıları)
    s = n_az // 3
    
    left   = np.max(fls_data[:, :s])
    center = np.max(fls_data[:, s:2*s])
    right  = np.max(fls_data[:, 2*s:])
    
    risk = center > FLS_THRESHOLD
    bias = 0.0
    if risk:
        # NED: +yaw = sağa. Engel sağda → sola kaç → negatif bias
        bias = -FLS_BIAS_HARD if left < right else FLS_BIAS_HARD
    elif left > FLS_THRESHOLD:
        bias = FLS_BIAS_SOFT   # sol engel → sağa kaç
    elif right > FLS_THRESHOLD:
        bias = -FLS_BIAS_SOFT   # sağ engel → sola kaç
    return bias, risk


# ═══════════════════════════════════════════════════════════════
# ANA DÖNGÜ
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="proposed",
                        choices=["proposed", "lawnmower", "random", "entropy_only"])
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=int, default=0,
                        help="Simülasyon süresi (tick). 0=sonsuz")
    parser.add_argument("--loss_rate", type=float, default=None,
                        help="Paket kayıp oranı (0.0-1.0). None=Thorp modeli")
    parser.add_argument("--log", type=str, default=None,
                        help="Log dosyasının yolu (.m formatında)")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Senaryo JSON dosyası (varsayılan: config.py SCENARIO_PATH)")
    parser.add_argument("--nav", default="ekf",
                        choices=["ground_truth", "ekf"],
                        help="Navigasyon: ground_truth (DynamicsSensor) veya "
                             "ekf (IMU+DVL+Depth EKF dead reckoning)")
    parser.add_argument("--fixed_weight", action="store_true",
                        help="Ablasyon çalışması: adaptif w_ij yerine sabit epsilon ağırlığı kullan")
    args = parser.parse_args()

    scenario_path = args.scenario if args.scenario else SCENARIO_PATH

    mode     = args.mode
    nav_mode = args.nav
    print("=" * 65)
    print(f"  Swarm AUV — Communication-Aware Cooperative Coverage v2.1")
    print(f"  Mode: {mode.upper()}")
    print(f"  Navigation: {nav_mode.upper()}")
    print(f"  Weight: {'FIXED (ablation)' if args.fixed_weight else 'ADAPTIVE (w_ij)'}")
    print(f"  Channel: {'Fixed loss=' + str(args.loss_rate) if args.loss_rate else 'Thorp + HMM'}")
    print("=" * 65)

    # ── Modüller ──────────────────────────────────────────────────
    comm_mgr  = CommManager(loss_rate=args.loss_rate)
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
        recovery_heading_off=RECOVERY_HEADING_OFF,
    )


    # Planlayıcılar
    planners     = {}
    lawn_planner = None
    rw_planner   = None
    eo_planner   = None

    if mode == "proposed":
        for name in AGENT_NAMES:
            planners[name] = EntropyGuidedPlanner(
                consensus.maps[name], agent_name=name, consensus=consensus)
    elif mode == "lawnmower":
        lawn_planner = LawnmowerPlanner()
    elif mode == "random":
        rw_planner = RandomWalkPlanner()
    elif mode == "entropy_only":
        eo_planner = EntropyOnlyPlanner()

    targets = {n: None for n in AGENT_NAMES}
    agents_metadata = {n: {} for n in AGENT_NAMES}

    # Logger hazırlığı
    logger = None
    if args.log:
        logger = MissionLogger(args.log, AGENT_NAMES)
        print(f"[INIT] Recording telemetry to: {args.log}")

    # EKF Navigasyon nesneleri
    INIT_TICKS = 90   # İlk 3 saniye ground-truth ile EKF'i başlat (3s × 30Hz)
    navigators = {}
    if nav_mode == "ekf":
        try:
            from core.ekf_navigation import AUVNavigationEKF
            for name in AGENT_NAMES:
                navigators[name] = AUVNavigationEKF(
                    dt=1.0 / TICKS_PER_SEC,
                    agent_name=name)
            print("[INIT] EKF navigasyon aktif — IMU+DVL+Depth füzyonu", flush=True)
        except ImportError as e:
            print(f"[WARN] core/ekf_navigation.py yüklenemedi ({e}) — "
                  f"ground_truth'a düşülüyor", flush=True)
            nav_mode = "ground_truth"
    else:
        print("[INIT] Navigasyon: Ground-Truth (DynamicsSensor)", flush=True)

    try:
        with open(scenario_path, "r") as f:
            scenario = json.load(f)

        fossen = FossenInterface(AGENT_NAMES, scenario, multi_agent=True)
        env    = holoocean.make(scenario_cfg=scenario)
        env.should_render_viewport(True)
        import time as _time  # yalnızca sleep için
        _time.sleep(1)

        # Dashboard
        dashboard = None
        if not args.headless:
            dashboard = Dashboard(consensus, comm_manager=comm_mgr)

        step = 0
        while True:
            if args.duration > 0 and step >= args.duration:
                print(f"\n[END] {args.duration} tick tamamlandı.")
                break

            # Simülasyon zamanı (tekrarlanabilir, wall-clock değil)
            sim_time = step / TICKS_PER_SEC

            states = env.tick()

            # ── Gecikme kuyruğunu boşalt (her tick) ───────────────
            # FIX: Propagation delay artık aktif — mesajlar buradan teslim edilir
            if mode == "proposed":
                comm_mgr.flush_delayed(env, step)

            # ── Beacon ID senkronizasyonu (adım 0) ────────────────
            if step == 0:
                real_id_map = {}
                for name in AGENT_NAMES:
                    if hasattr(env, "agents") and name in env.agents:
                        a = env.agents[name]
                        if "AcousticBeaconSensor" in a.sensors:
                            real_id_map[name] = a.sensors["AcousticBeaconSensor"].id
                if len(real_id_map) == len(AGENT_NAMES):
                    comm_mgr.update_ids(real_id_map)
                    print(f"[INIT] Beacon ID map: {real_id_map}", flush=True)

            # ── Ajan konumlarını CommManager'a bildir ─────────────
            agent_positions = {}
            for name in AGENT_NAMES:
                if name in states and "PoseSensor" in states[name]:
                    p = states[name]["PoseSensor"][:3, 3]
                    agent_positions[name] = p
            comm_mgr.update_positions(agent_positions)

            # ── Per-agent döngü ───────────────────────────────────
            for name in AGENT_NAMES:
                if name not in states:
                    continue

                pose = states[name]["PoseSensor"]
                pos  = pose[:3, 3]
                rot  = pose[:3, :3]
                yaw  = np.arctan2(rot[1, 0], rot[0, 0])
                hdg  = np.degrees(yaw)

                vel_raw = states[name].get("VelocitySensor", np.zeros(3))
                vel     = float(np.linalg.norm(vel_raw[:2])) if hasattr(vel_raw, "__len__") else 0.0

                # SSS = sensor_name (JSON), SidescanSonar = sensor_type (fallback)
                ss_data = states[name].get("SSS", states[name].get("SidescanSonar"))
                if mode == "proposed":
                    if ss_data is not None and len(ss_data) > 0:
                        consensus.maps[name].bayesian_update_from_sidescan(pos, yaw, ss_data)
                    else:
                        consensus.update_local(name, pos, yaw)
                elif mode == "entropy_only" and eo_planner:
                    if ss_data is not None and len(ss_data) > 0:
                        eo_planner.maps[name].bayesian_update_from_sidescan(pos, yaw, ss_data)
                    else:
                        eo_planner.maps[name].bayesian_update(pos, yaw)
                else:
                    if ss_data is not None and len(ss_data) > 0:
                        consensus.maps[name].bayesian_update_from_sidescan(pos, yaw, ss_data)
                    else:
                        consensus.update_local(name, pos, yaw)

                # 2. SNR bilgisini comm_quality'e kaydet
                if mode == "proposed":
                    for other in AGENT_NAMES:
                        if other != name:
                            snr = comm_mgr.get_snr(name, other)
                            if not np.isnan(snr):
                                comm_qual.log_snr(name, other, snr)

                # 3. Kanal durumu güncelle
                comm_state = consensus.update_comm_state(name, sim_time=sim_time)
                comm_qual.log_comm_state(name, comm_state, sim_time=sim_time)

                # 4. Collision & stuck
                # 6. Engel ve Sıkışma Kontrolü
                coll = bool(np.any(states[name].get("CollisionSensor", False)))
                stuck.update_heading(name, hdg)
                if (coll or stuck.check_stuck(name, pos, sim_time)) and stuck.status[name] == "OK":
                    stuck.enter_recovery(name, hdg, abs(pos[2]), sim_time)
                    if mode == "proposed" and name in planners:
                        planners[name].current_target = None
                        planners[name].global_target = None
                        planners[name]._last_global_replan_tick = -99999
                    elif mode == "entropy_only" and getattr(eo_planner, 'reset_target', None):
                        eo_planner.reset_target(name)

                # Zigzag tespiti — recovery'ye girmeden global kaçış hedefi ver
                elif stuck.check_zigzag(name, sim_time) and mode == "proposed" and name in planners:
                    pmap = consensus.maps[name]
                    unc = pmap.get_uncertainty()
                    _gc = pmap.world_to_grid(pos[:2])
                    if _gc is None:          # AUV harita disi — merkeze fallback
                        _gc = np.array([pmap.n // 2, pmap.n // 2])
                    n_c, e_c = _gc
                    yi, xi = np.mgrid[0:pmap.n, 0:pmap.n]
                    d_grid = np.sqrt((yi - n_c)**2 + (xi - e_c)**2)
                    score = unc * d_grid
                    # Diğer AUV'ların olduğu bölgeleri cezalandır (kümelenme önleme)
                    for other_name in AGENT_NAMES:
                        if other_name != name:
                            other_pos = agent_positions.get(other_name)
                            if other_pos is not None:
                                o_grid = pmap.world_to_grid(other_pos[:2])
                                if o_grid is None:  # diger AUV harita disi, atla
                                    continue
                                d_other = np.sqrt((yi - o_grid[0])**2 + (xi - o_grid[1])**2) * pmap.res
                                score *= (1.0 - 0.9 * np.exp(-d_other / 80.0))
                    idx = np.unravel_index(np.argmax(score), score.shape)
                    escape_target = pmap.grid_to_world(idx[0], idx[1])
                    escape_target = np.clip(escape_target, -160.0, 160.0)
                    planners[name].global_target = escape_target
                    planners[name].current_target = escape_target
                    planners[name]._last_global_replan_tick = -99999
                    stuck.heading_history[name].clear()
                    stuck.mark_escape(name, sim_time)

                # 5. FLS engel kaçınma
                obs_bias = 0.0
                if "ImagingSonar" in states[name]:
                    obs_bias, _ = get_obstacle_bias(states[name]["ImagingSonar"])

                # 6. Hedef hesapla
                # FIX: final_rpm değişkenini her dalda tanımla
                final_rpm_used = CRUISE_RPM

                if stuck.status[name] == "OK":
                    # Consensus modunda komşu haritaları planlamaya ver
                    neighbor_grids = None
                    if mode == "proposed" and comm_state != "OFFLINE":
                        neighbor_grids = {
                            n: consensus.maps[n].grid
                            for n in AGENT_NAMES if n != name
                        }

                    if mode == "proposed":
                        tgt_h, tgt_rpm, tgt_wp = planners[name].plan(
                            pos, yaw, step, neighbor_grids=neighbor_grids, sim_time=sim_time)
                        targets[name] = tgt_wp
                    elif mode == "lawnmower" and lawn_planner:
                        tgt_h, tgt_rpm, tgt_wp = lawn_planner.plan(name, pos, yaw, step)
                        targets[name] = tgt_wp
                    elif mode == "random" and rw_planner:
                        tgt_h, tgt_rpm, tgt_wp = rw_planner.plan(name, pos, yaw, step)
                        targets[name] = tgt_wp
                    elif mode == "entropy_only" and eo_planner:
                        tgt_h, tgt_rpm, tgt_wp = eo_planner.plan(name, pos, yaw, step)
                        targets[name] = tgt_wp
                    else:
                        tgt_h, tgt_rpm = hdg, CRUISE_RPM
                        # targets[name] değişmez — kasıtlı

                    # U-dönüşü: yavaşla
                    hdiff = abs(heading_diff(hdg, tgt_h))
                    if hdiff > UTURN_THRESH:
                        tgt_rpm = MIN_RPM + 200

                    # Offline modda lokal entropi planlamaya devam edilecek (neighbor_grids=None yapıldı)

                    final_h   = wrap_heading(tgt_h + obs_bias)
                    final_rpm = int(np.clip(tgt_rpm, MIN_RPM, MAX_RPM))

                    # APF sınır itici kuvveti (sınır dışında da aktif)
                    px, py = float(pos[0]), float(pos[1])
                    
                    # Her eksende sınıra mesafe (negatif = sınır dışında)
                    dx_edge = SAFE_MARGIN - abs(px)
                    dy_edge = SAFE_MARGIN - abs(py)
                    
                    F_rep = np.zeros(2)
                    
                    # X ekseni sınır kuvveti — sınır dışında da çalışır
                    if dx_edge < APF_INFLUENCE_DIST:
                        dx_eff = max(dx_edge, 1.0)  # sıfıra bölme önlemi
                        mag = APF_K_REP * (1/dx_eff - 1/APF_INFLUENCE_DIST) / (dx_eff**2)
                        F_rep[0] += -np.sign(px) * mag
                    
                    # Y ekseni sınır kuvveti — sınır dışında da çalışır
                    if dy_edge < APF_INFLUENCE_DIST:
                        dy_eff = max(dy_edge, 1.0)  # sıfıra bölme önlemi
                        mag = APF_K_REP * (1/dy_eff - 1/APF_INFLUENCE_DIST) / (dy_eff**2)
                        F_rep[1] += -np.sign(py) * mag
                    
                    # AUV'lar arası itici APF kuvveti (çarpışma önleme)
                    APF_AUV_DIST = 40.0   # m, bu mesafe içinde iter
                    APF_AUV_K   = 500.0  # kuvvet katsayısı
                    for other_name in AGENT_NAMES:
                        if other_name != name:
                            other_pos = agent_positions.get(other_name)
                            if other_pos is not None:
                                diff = pos[:2] - other_pos[:2]
                                d_auv = float(np.linalg.norm(diff))
                                if 0.5 < d_auv < APF_AUV_DIST:
                                    d_eff = max(d_auv, 1.0)
                                    mag_auv = APF_AUV_K * (1/d_eff - 1/APF_AUV_DIST) / (d_eff**2)
                                    direction = diff / d_auv
                                    F_rep += direction * mag_auv

                    # Mevcut heading vektörüne APF ekle
                    if np.linalg.norm(F_rep) > 0.01:
                        current_vec = np.array([
                            np.cos(np.radians(final_h)),
                            np.sin(np.radians(final_h))
                        ])
                        # Sınır dışındaysa blend=1.0 (tam APF kontrolü)
                        min_edge = min(dx_edge, dy_edge)
                        if min_edge <= 0:
                            blend = 1.0
                        else:
                            blend = 0.4 + 0.6 * (1 - min_edge / APF_INFLUENCE_DIST)
                        F_total = current_vec + F_rep * blend
                        final_h = wrap_heading(
                            np.degrees(np.arctan2(F_total[1], F_total[0]))
                        )

                    # --- SINIR KORUMASI (Dinamik) ---
                    B_VAL = 170.0
                    if abs(pos[0]) > B_VAL or abs(pos[1]) > B_VAL:
                        inward_x = np.clip(pos[0], -B_VAL + 20.0, B_VAL - 20.0)
                        inward_y = np.clip(pos[1], -B_VAL + 20.0, B_VAL - 20.0)
                        dx = inward_x - pos[0]
                        dy = inward_y - pos[1]
                        if abs(dx) > 0.1 or abs(dy) > 0.1:
                            # APF zaten içe itiyor, sadece rpm'i artır
                            final_rpm = MIN_RPM + 300
                            targets[name] = np.array([inward_x, inward_y])
                    if abs(pos[0]) > 190.0 or abs(pos[1]) > 190.0:
                        final_rpm = MAX_RPM  # hizla donup iceri gir

                    final_rpm_used = final_rpm

                    cmd_depth = CRUISE_DEPTH

                    fossen.set_goal(name,
                                    depth=max(0.5, cmd_depth),
                                    heading=final_h,
                                    rpm=final_rpm)
                else:
                    # RECOVERY
                    fossen.set_goal(name,
                                    depth=max(0.5, stuck.recovery_depth[name]),
                                    heading=stuck.recovery_heading[name],
                                    rpm=RECOVERY_RPM)
                    final_rpm_used = RECOVERY_RPM
                    if stuck.is_done(name, sim_time):
                        stuck.finish_recovery(name, pos, sim_time)
                        if mode == "proposed" and name in planners:
                            planners[name].current_target = None
                            planners[name].global_target = None
                            planners[name]._last_global_replan_tick = -99999
                        elif mode == "entropy_only" and getattr(eo_planner, 'reset_target', None):
                            eo_planner.reset_target(name)

                # 7. EKF Navigasyon Güncelleme
                dr_state = None
                if nav_mode == "ekf" and name in navigators:
                    nav     = navigators[name]
                    imu_raw = states[name].get("IMUSensor")
                    dvl_raw = states[name].get("DVLSensor")
                    dep_raw = states[name].get("DepthSensor")
                    # Derinlik: DepthSensor varsa onu kullan, yoksa PoseSensor Z
                    depth_m = float(abs(dep_raw[0])) \
                              if dep_raw is not None and len(dep_raw) > 0 \
                              else abs(float(pos[2]))

                    if not nav.initialized:
                        # İlk INIT_TICKS tick'inde ground-truth ile başlat
                        pose_4x4 = states[name]["PoseSensor"]
                        vel_nwu  = np.array(
                            states[name]["DynamicsSensor"][3:6], dtype=float)
                        nav.initialize_from_pose(pose_4x4, vel_nwu)
                    else:
                        # Attitude güncelleme (GT'den)
                        nav.update_attitude_from_dynamics(states[name]["DynamicsSensor"])
                        # Prediction: IMU (her tick, ~30 Hz)
                        if imu_raw is not None:
                            nav.predict(np.array(imu_raw, dtype=float))
                        # Update: DVL — her 3 tick'te bir (~10 Hz)
                        if dvl_raw is not None and step % 3 == 0:
                            nav.update_dvl(np.array(dvl_raw, dtype=float), rpm=final_rpm_used)
                        # Update: Depth (her tick)
                        nav.update_depth(depth_m)
                        # INIT_TICKS'ten sonra EKF devreye girer
                        if step >= INIT_TICKS:
                            dr_state = nav.get_fossen_state(states[name]["DynamicsSensor"])

                # 7b. Fossen dinamikleri (dr_state=None → ground-truth)
                accel = fossen.update(name, states, dr_state=dr_state)
                env.act(name, accel)

                # 8. Dashboard ve Log sensör verisi (FIX: gerçek RPM)
                target_pos = targets.get(name, np.array([0.0, 0.0]))
                sd_entry = {
                    "x": pos[0], "y": pos[1], "z": pos[2],
                    "heading": hdg, "vel": vel,
                    "rpm": final_rpm_used,
                    "target_x": target_pos[0] if target_pos is not None else 0.0,
                    "target_y": target_pos[1] if target_pos is not None else 0.0,
                    "status": stuck.status[name],
                }
                # EKF navigasyon hatası logu (ground-truth vs EKF)
                if nav_mode == "ekf" and name in navigators and navigators[name].initialized:
                    ned_est  = navigators[name].x[0:3]       # EKF tahmini (NED)
                    # NED [n, e] → NWU [x, -y] dönüşümü
                    ekf_x    =  ned_est[0]
                    ekf_y    = -ned_est[1]
                    nav_err  = float(np.sqrt(
                        (pos[0] - ekf_x)**2 + (pos[1] - ekf_y)**2))
                    sd_entry["nav_error"] = nav_err
                    sd_entry["ekf_x"]    = ekf_x
                    sd_entry["ekf_y"]    = ekf_y

                agents_metadata[name] = sd_entry
                if dashboard:
                    dashboard.sensor_data[name] = sd_entry

            # ── Akustik haberleşme (sadece proposed) ─────────────────────
            if mode == "proposed":
                # TDMA: her ajan kendi slot'unda gönderir, broadcast
                for sender in AGENT_NAMES:
                    if comm_mgr.should_send_tdma(sender, step):
                        current_wp = None
                        if sender in planners:
                            current_wp = planners[sender].current_target
                        
                        payload = consensus.encode_map_for_beacon(sender, current_target=current_wp)
                        if payload:
                            # prepare_messages artık tüm komşulara broadcast yapıyor
                            msgs = comm_mgr.prepare_messages(
                                sender, payload,
                                agent_positions=agent_positions,
                                current_tick=step)
                            for fid, tid, d in msgs:
                                comm_qual.log_send(sender, sim_time=sim_time)
                                ok = comm_mgr.send(env, sender, tid, d,
                                                   current_tick=step)
                                if ok and dashboard:
                                    recv_name = comm_mgr.rev_id_map.get(tid, "")
                                    snr = comm_mgr.get_snr(sender, recv_name)
                                    dashboard.trigger_beacon_pulse(sender, recv_name, snr)
                                    dashboard._active_links[(sender, recv_name)] = True
                                if not ok:
                                    comm_qual.log_drop(sender)

            # ── Beacon alımı (FIX: receiver değişkeni, çift update önlendi) ──
            beacon_received_this_tick = {n: False for n in AGENT_NAMES}

            if mode == "proposed":
                for receiver in AGENT_NAMES:  # FIX: 'name' yerine 'receiver'
                    raw = states[receiver].get("AcousticBeaconSensor")
                    if raw is None:
                        continue

                    parsed_list = comm_mgr.parse_beacon_data(
                        raw, receiver=receiver, sim_time=sim_time)
                    for parsed in parsed_list:
                        sender_name = parsed["from_name"]
                        payload     = parsed["payload"]
                        comm_qual.log_receive(receiver, sender_name,
                                              sim_time=sim_time)

                        # decode_map_from_beacon bytes de list de kabul eder
                        if isinstance(payload, (bytes, bytearray, list)):
                            ok = consensus.decode_map_from_beacon(
                                receiver, sender_name, payload,
                                sim_time=sim_time)

                            if ok:
                                # Yeniden bağlanma kontrolü
                                if consensus.get_comm_state(receiver) == "RECONNECTING":
                                    consensus.sync_on_reconnect(receiver)
                                    log_msg = f"[SYNC] {receiver} harita senkronize edildi"
                                    print(log_msg, flush=True)
                                    if dashboard:
                                        dashboard.comm_log.append(log_msg)
                                else:
                                    consensus.consensus_update(
                                        receiver, sim_time=sim_time)
                                    beacon_received_this_tick[receiver] = True

                        # Log mesajı
                        payload_sz = len(payload) if hasattr(payload, "__len__") else 1
                        log_msg = (
                            f"t={step/TICKS_PER_SEC:.0f}s "
                            f"{receiver}←{sender_name}: {payload_sz}B"
                        )
                        if dashboard:
                            dashboard.comm_log.append(log_msg)

                # Periyodik consensus güncelleme
                # Kasıtlı: Entropy-Only baseline'da haberleşme ve consensus yoktur, 
                # o yüzden bu blok sadece 'proposed' modunda çalışır.
                # FIX: bu tick'te beacon-driven update yapılmadıysa çalış
                if step > 0 and step % BEACON_SEND_INTERVAL == 0:
                    for name in AGENT_NAMES:
                        if not beacon_received_this_tick[name]:
                            consensus.consensus_update(
                                name, sim_time=sim_time)
                    rms = consensus.get_consensus_rms()
                    if step % (BEACON_SEND_INTERVAL * 3) == 0:
                        states_str = " | ".join(
                            f"{n}:{consensus.get_comm_state(n)[:3]}"
                            for n in AGENT_NAMES)
                        print(f"[CONS] step={step} RMS={rms:.4f} "
                              f"Cov={consensus.get_avg_coverage_pct():.1f}% "
                              f"| {states_str}", flush=True)
                        # EKF navigasyon hatası özeti
                        if nav_mode == "ekf" and navigators:
                            nav_errs = []
                            for nn in AGENT_NAMES:
                                if nn in navigators and navigators[nn].initialized:
                                    gt_p = agent_positions.get(nn, np.zeros(3))
                                    ned  = navigators[nn].x[0:3]
                                    err  = float(np.sqrt(
                                        (gt_p[0] - ned[0])**2
                                        + (gt_p[1] - ned[1])**2))
                                    nav_errs.append(err)
                            if nav_errs:
                                print(f"[NAV]  EKF konum hatası — "
                                      f"ort={np.mean(nav_errs):.2f}m  "
                                      f"maks={np.max(nav_errs):.2f}m",
                                      flush=True)

            # ── GUI ve Log güncelleme ──────────────────────────────
            if dashboard and step % GUI_UPDATE_TICKS == 0:
                dashboard.update(states, targets, comm_qual, mode)
                dashboard._active_links = {}

            if logger:
                # FIX: public API kullan, ortalama SNR matrisi
                snr_mat = comm_qual.get_snr_matrix() if mode == "proposed" else None
                
                if mode == "entropy_only" and eo_planner:
                    avg_cov = np.mean([eo_planner.maps[n].get_coverage_pct() for n in AGENT_NAMES])
                else:
                    avg_cov = consensus.get_avg_coverage_pct()
                    
                logger.log(
                    step=step,
                    sim_time=sim_time,
                    coverage=avg_cov,
                    rms=consensus.get_consensus_rms(),
                    pdr=comm_qual.get_summary()["global_pdr"],
                    agents_state=agents_metadata,
                    snr_matrix=snr_mat,
                )

                # Periyodik flush: crash'te veri kaybını önle
                if step > 0 and step % 1000 == 0:
                    logger.save()

            step += 1

    except KeyboardInterrupt:
        print("\n[STOP] Kullanıcı durdurdu.")
    except Exception:
        print("\n" + "=" * 30 + " HATA " + "=" * 30)
        traceback.print_exc()
        print("=" * 66)
    finally:
        print("\nSimülasyon durduruluyor...")
        if "logger" in locals() and logger:
            logger.save()
        if "env" in locals():
            try:
                env.__on_exit__()
            except Exception:
                pass


if __name__ == "__main__":
    main()