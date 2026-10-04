"""
Swarm AUV — Okyanus akıntısı altında EKF navigasyonu ile kooperatif haritalama.

Kullanım:
    python main_ekf_current.py --headless --duration 9000 --log logs/run.m
    python main_ekf_current.py --no_current_comp ...        # baseline: akıntı telafisiz
    python main_ekf_current.py --dvl_outage 120 180 ...     # 120-180 s arası DVL kesintisi (tüm araçlar)
    python main_ekf_current.py --dvl_outage 120 180 --outage_agents auv1    # yalnızca auv1
    python main_ekf_current.py --no_range_aid ...           # akustik mesafe güncellemesi kapalı
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import holoocean
from holoocean.fossen_dynamics.fossen_interface import FossenInterface
import numpy as np
import json, traceback, argparse, random
import time as _time

# Ege akıntılı konfigürasyonu yükle
from config_current import (
    AGENT_NAMES, SCENARIO_PATH, TICKS_PER_SEC,
    CRUISE_RPM, CRUISE_DEPTH, MIN_RPM, MAX_RPM,
    UTURN_THRESH,
    STUCK_TIME_S, STUCK_DIST_M, STARTUP_GRACE_S,
    POST_RECOVERY_GRACE_S, RECOVERY_DUR_S, RECOVERY_RPM,
    BEACON_SEND_INTERVAL, GUI_UPDATE_TICKS,
    wrap_heading, heading_diff,
    USE_OCEAN_CURRENT, BASE_CURRENT_NED, CURRENT_NOISE_VAR, CURRENT_TAU
)

from core.consensus import ConsensusMapFusion
from core.entropy_planner import EntropyGuidedPlanner
from core.comm_quality import CommQualityMonitor
from core.comm_manager import CommManager, RANGE_SIGMA_M
from core.stuck_detector import StuckDetector
from core.ekf_navigation_current import AUVNavigationEKF, T_NWU_NED
from visualization.dashboard import Dashboard
from core.logger import MissionLogger
from core.status_report import format_status


APF_INFLUENCE_DIST = 50.0
APF_K_REP = 800.0
SAFE_MARGIN = 170.0

DVL_DECIMATION  = 3     # 30 Hz simülasyon → 10 Hz DVL güncellemesi
NAV_WARMUP_TICKS = 90   # EKF çıktısı bu adımdan sonra kontrol ve planlamada kullanılır
STATUS_PRINT_TICKS = 900   # konsol durum raporu periyodu (30 s)
MISSION_COMPLETE_PCT = 99.0   # aracın kendi haritasında bu kapsamaya ulaşılınca görev biter
HOME_ACCEPT_R = 25.0          # başlangıç noktasına bu mesafede varılmış sayılır (m)


def main():
    # Konsol kod sayfasında karşılığı olmayan bir karakter koşuyu düşürmesin
    sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=int, default=0, help="Simülasyon süresi (tick). 0=sonsuz")
    parser.add_argument("--log", type=str, default=None, help="Log dosyasının yolu")
    parser.add_argument("--nav", default="ekf", choices=["ground_truth", "ekf"])
    parser.add_argument("--fixed_weight", action="store_true", help="Ablasyon: adaptif w_ij yerine sabit epsilon")
    parser.add_argument("--current_mag", type=float, default=0.2236, help="Akıntı büyüklüğü (m/s). Yön korunur, büyüklük ölçeklenir.")
    parser.add_argument("--seed", type=int, default=42, help="Rastgelelik tohumu (seed)")
    parser.add_argument("--no_current_comp", action="store_true", help="Baseline: akıntı kestirimi ve telafisi kapalı")
    parser.add_argument("--dvl_outage", type=float, nargs=2, default=None, metavar=("T_START", "T_END"),
                        help="Bu zaman aralığında (s) DVL ölçümü filtreye verilmez")
    parser.add_argument("--outage_agents", nargs="+", default=AGENT_NAMES, choices=AGENT_NAMES,
                        help="DVL kesintisinin uygulanacağı araçlar (varsayılan: hepsi)")
    parser.add_argument("--oracle_neighbor_maps", action="store_true",
                        help="Planlayıcıya komşu haritalarını akustik kanal olmadan doğrudan ver (üst sınır karşılaştırması)")
    parser.add_argument("--no_consensus", action="store_true",
                        help="Ablasyon: komşulardan gelen harita güncellemelerini uygulama "
                             "(hedef ve konum bildirimi sürer)")
    parser.add_argument("--no_range_aid", action="store_true",
                        help="Beacon'lardan gelen akustik mesafe ölçümünü EKF'e verme")
    args = parser.parse_args()

    # Seed ayarla (akustik kanal modeli random modülünü kullanır)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Akıntı büyüklüğünü ölçekle (mevcut yön korunur, büyüklük ayarlanır)
    base_curr_nom = np.array(BASE_CURRENT_NED, dtype=float)
    nom_mag = float(np.linalg.norm(base_curr_nom[:2]))
    if nom_mag > 1e-6:
        base_current_vec = base_curr_nom * (args.current_mag / nom_mag)
    else:
        base_current_vec = np.array([-args.current_mag, 0.0, 0.0])

    nav_mode = args.nav
    current_comp = not args.no_current_comp
    print("=" * 77)
    print("  Swarm AUV — V2: Hidrodinamik Akıntı Modeli (Fossen V_c/beta_c)")
    print(f"  Navigation: {nav_mode.upper()} | Akıntı telafisi: {'AÇIK' if current_comp else 'KAPALI (BASELINE)'}")
    print(f"  Weight: {'FIXED (ablation)' if args.fixed_weight else 'ADAPTIVE (w_ij)'}")
    print(f"  Okyanus Akintisi: {'AKTIF (Gauss-Markov -> Fossen)' if USE_OCEAN_CURRENT else 'PASIF'} | Mag: {args.current_mag:.2f} m/s | Seed: {args.seed}")
    if args.dvl_outage:
        print(f"  DVL kesintisi: {args.dvl_outage[0]:.0f}-{args.dvl_outage[1]:.0f} s | Araçlar: {', '.join(args.outage_agents)}")
    print(f"  Akustik mesafe desteği: {'KAPALI' if args.no_range_aid else 'AÇIK'}")
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


    planners = {name: EntropyGuidedPlanner(consensus.maps[name], agent_name=name, consensus=consensus)
                for name in AGENT_NAMES}

    targets = {n: None for n in AGENT_NAMES}
    agents_metadata = {n: {} for n in AGENT_NAMES}
    last_rpm = {n: CRUISE_RPM for n in AGENT_NAMES}
    range_updates = {n: 0 for n in AGENT_NAMES}
    home = {}                                         # başlangıç konumları (NWU)
    mission_done = {n: False for n in AGENT_NAMES}    # aracın haritası tamamlandı
    at_home = {n: False for n in AGENT_NAMES}         # araç başlangıç noktasına döndü

    logger = None
    if args.log:
        logger = MissionLogger(args.log, AGENT_NAMES)

    with open(SCENARIO_PATH, "r") as f:
        scenario = json.load(f)

    dt = 1.0 / TICKS_PER_SEC
    navigators = {}
    if nav_mode == "ekf":
        for name in AGENT_NAMES:
            # Filtrenin DVL modeli senaryodaki sensör tanımından alınır
            agent_cfg = next(a for a in scenario["agents"] if a["agent_name"] == name)
            dvl_cfg = next(sen["configuration"] for sen in agent_cfg["sensors"]
                           if sen["sensor_type"] == "DVLSensor")
            navigators[name] = AUVNavigationEKF(
                dt=dt, agent_name=name, dvl_dt=DVL_DECIMATION * dt,
                current_compensation=current_comp,
                dvl_max_range=dvl_cfg["MaxRange"] if dvl_cfg.get("ReturnRange") else None,
                dvl_beam_sigma=dvl_cfg.get("VelSigma", 0.01),
                dvl_elevation_deg=dvl_cfg.get("Elevation", 22.5),
                current_diffusion=float(np.sqrt(CURRENT_NOISE_VAR)))
        print("[INIT] 18-durumlu EKF seyrüseferi başlatıldı (Tutum, Bias ve Akıntı Kestirimli).", flush=True)

    try:
        fossen = FossenInterface(AGENT_NAMES, scenario, multi_agent=True)
        env    = holoocean.make(scenario_cfg=scenario)
        env.should_render_viewport(not args.headless)
        _time.sleep(1)

        dashboard = None
        if not args.headless:
            dashboard = Dashboard(consensus, comm_manager=comm_mgr)

        # Akıntı Fossen torpedo modelinin V_c/beta_c parametreleri üzerinden uygulanır.
        # Türbülans durağan dağılımından başlatılır: std = sqrt(CURRENT_NOISE_VAR * CURRENT_TAU / 2)
        current_ned    = base_current_vec.copy()
        turbulence_ned = np.zeros(3)
        if USE_OCEAN_CURRENT:
            turbulence_ned[:2] = np.random.normal(0, np.sqrt(CURRENT_NOISE_VAR * CURRENT_TAU / 2.0), 2)

        wall_start = _time.time()
        step = 0
        while True:
            if args.duration > 0 and step >= args.duration:
                break

            sim_time = step / TICKS_PER_SEC
            states = env.tick()
            comm_mgr.flush_delayed(env, step)

            # Akıntı Türbülans Güncellemesi — Ornstein-Uhlenbeck (Euler-Maruyama)
            if USE_OCEAN_CURRENT:
                noise = np.random.normal(0, 1, 3)
                noise[2] = 0.0  # Dikey akıntıyı sıfır tut
                turbulence_ned = (turbulence_ned
                                  - (turbulence_ned / CURRENT_TAU) * dt
                                  + np.sqrt(CURRENT_NOISE_VAR) * np.sqrt(dt) * noise)
                current_ned = base_current_vec + turbulence_ned

                # torpedo.dynamics(): nu_r = nu - nu_c ; beta_c NED yatay düzlemde radyan
                V_c    = float(np.linalg.norm(current_ned[:2]))
                beta_c = float(np.arctan2(current_ned[1], current_ned[0]))
                for n in AGENT_NAMES:
                    fossen.vehicles[n].V_c    = V_c
                    fossen.vehicles[n].beta_c = beta_c

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

            # ── 1) SEYRÜSEFER: sensörlerden durum kestirimi ──
            # nav_state: otopilota verilen DynamicsSensor formatında kestirim (None → gerçek durum)
            # nav_pos / nav_yaw: haritalama ve planlamanın gördüğü konum ve baş açısı
            nav_state = {}
            nav_pos   = {}
            nav_yaw   = {}
            true_pos  = {}
            for name in AGENT_NAMES:
                if name not in states:
                    continue
                pose = states[name]["PoseSensor"]
                gt_pos = pose[:3, 3].copy()
                gt_yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
                true_pos[name]  = gt_pos
                nav_state[name] = None
                nav_pos[name], nav_yaw[name] = gt_pos, gt_yaw

                if nav_mode != "ekf":
                    continue
                nav = navigators[name]
                if not nav.initialized:
                    nav.initialize_from_pose(pose, np.array(states[name]["DynamicsSensor"][3:6], dtype=float))
                    continue

                imu_raw = states[name].get("IMUSensor")
                if imu_raw is not None:
                    nav.predict(np.array(imu_raw, dtype=float))
                if step % DVL_DECIMATION == 0:
                    # DVL yer hızını (akıntı dahil) ölçer; kesinti penceresinde filtreye verilmez
                    dvl_raw = states[name].get("DVLSensor")
                    if (args.dvl_outage and name in args.outage_agents
                            and args.dvl_outage[0] <= sim_time < args.dvl_outage[1]):
                        dvl_raw = None
                    nav.update_dvl(dvl_raw, rpm=last_rpm[name])
                # Basınç sensörü; senaryoda DepthSensor yoksa gerçek derinliğe düşülür
                dep_raw = states[name].get("DepthSensor")
                depth_m = float(abs(dep_raw[0])) if dep_raw is not None and len(dep_raw) > 0 else abs(float(gt_pos[2]))
                nav.update_depth(depth_m)
                nav.update_magnetometer(states[name].get("MagnetometerSensor"))

                if step >= NAV_WARMUP_TICKS:
                    nav_state[name] = nav.get_fossen_state(states[name]["DynamicsSensor"])
                    nav_pos[name]   = nav.get_position_nwu()
                    nav_yaw[name]   = nav.get_yaw_nwu()

            # Akustik kanal fiziği (SNR, kayıp, gecikme, mesafe) gerçek konumlara bağlıdır
            comm_mgr.update_positions(true_pos)

            # ── 2) HARİTALAMA, PLANLAMA, KONTROL ──
            for name in AGENT_NAMES:
                if name not in states:
                    continue

                gt_pos = states[name]["PoseSensor"][:3, 3]
                pos, yaw = nav_pos[name], nav_yaw[name]
                hdg = np.degrees(yaw)
                home.setdefault(name, np.array(pos[:2], dtype=float))

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

                # Çarpışma ve sıkışma tespiti simülatör gözetimidir, gerçek konumla yapılır
                coll = bool(np.any(states[name].get("CollisionSensor", False)))
                stuck.update_heading(name, hdg)
                if (coll or stuck.check_stuck(name, gt_pos, sim_time)) and stuck.status[name] == "OK":
                    stuck.enter_recovery(name, hdg, abs(gt_pos[2]), sim_time)
                    planners[name].current_target = None
                    planners[name].global_target = None
                    planners[name]._last_global_replan_tick = -99999
                elif stuck.status[name] == "RECOVERING" and stuck.is_done(name, sim_time):
                    stuck.finish_recovery(name, gt_pos, sim_time)
                    planners[name].current_target = None
                    planners[name].global_target = None
                    planners[name]._last_global_replan_tick = -99999

                out_of_bounds = abs(pos[0]) > SAFE_MARGIN or abs(pos[1]) > SAFE_MARGIN
                inward_h = wrap_heading(np.degrees(np.arctan2(-pos[1], -pos[0])))

                if stuck.status[name] == "OK":
                    # Araç komşularını yalnızca akustik kanaldan gelen ve kendi haritasına
                    # konsensüsle işlenen güncellemelerden bilir; doğrudan erişim yalnızca
                    # ideal haberleşme üst sınırını ölçmek içindir
                    neighbor_grids = None
                    if args.oracle_neighbor_maps and comm_state != "OFFLINE":
                        neighbor_grids = {n: consensus.maps[n].grid for n in AGENT_NAMES if n != name}

                    # Kestirilen akıntının rota yönündeki bileşenine bağlı dinamik kabul yarıçapı
                    current_along_track = 0.0
                    if name in navigators:
                        heading_vec = np.array([np.cos(yaw), np.sin(yaw)])
                        current_along_track = float(np.dot(navigators[name].get_current_nwu()[:2], heading_vec))
                    planners[name].wp_accept_r = 15.0 + 15.0 * max(0.0, current_along_track)

                    if not mission_done[name] and consensus.get_coverage_pct(name) >= MISSION_COMPLETE_PCT:
                        mission_done[name] = True
                        print(f"[MISSION] {name}: harita tamamlandı (t={sim_time:.0f} s), başlangıç noktasına dönüyor", flush=True)

                    if mission_done[name]:
                        # Taranacak yer kalmadı: başlangıç noktasına dön
                        to_home = home[name] - pos[:2]
                        tgt_wp, tgt_rpm = home[name], CRUISE_RPM
                        tgt_h = np.degrees(np.arctan2(to_home[1], to_home[0]))
                        if not at_home[name] and np.linalg.norm(to_home) < HOME_ACCEPT_R:
                            at_home[name] = True
                            print(f"[MISSION] {name}: başlangıç noktasına ulaştı (t={sim_time:.0f} s)", flush=True)
                    else:
                        tgt_h, tgt_rpm, tgt_wp = planners[name].plan(
                            pos, yaw, step, neighbor_grids=neighbor_grids, sim_time=sim_time)
                    targets[name] = tgt_wp

                    if abs(heading_diff(hdg, tgt_h)) > UTURN_THRESH:
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
                        heading_cmd_vec = np.array([np.cos(np.radians(final_h)), np.sin(np.radians(final_h))])
                        min_edge = min(dx_edge, dy_edge)
                        blend = 1.0 if min_edge <= 0 else 0.4 + 0.6 * (1 - min_edge / APF_INFLUENCE_DIST)
                        F_total = heading_cmd_vec + F_rep * blend
                        final_h = wrap_heading(np.degrees(np.arctan2(F_total[1], F_total[0])))

                    # Sınır dışında: hedef zaten alanın içindedir; dönüş yönünü tersine
                    # çevirmemek için doğrudan hedefe yönel ve hızı düşür
                    if out_of_bounds:
                        final_h = wrap_heading(tgt_h)
                        final_rpm = MIN_RPM + 300

                    final_rpm_used = final_rpm
                    fossen.set_goal(name, depth=max(0.5, CRUISE_DEPTH), heading=final_h, rpm=final_rpm)
                elif out_of_bounds:
                    stuck.finish_recovery(name, gt_pos, sim_time)
                    final_rpm_used = MIN_RPM + 300
                    fossen.set_goal(name, depth=max(0.5, CRUISE_DEPTH), heading=inward_h, rpm=final_rpm_used)
                else:
                    final_rpm_used = RECOVERY_RPM
                    fossen.set_goal(name, depth=max(0.5, stuck.recovery_depth[name]),
                                    heading=stuck.recovery_heading[name], rpm=final_rpm_used)
                last_rpm[name] = final_rpm_used

                # Otopilot kestirilen durumu, araç fiziği gerçek durumu kullanır
                accel = fossen.update(name, states, dr_state=nav_state[name])
                env.act(name, accel)

                # Raporlama (gerçek konum ve baş açısı)
                target_pos = targets.get(name)
                gt_rot = states[name]["PoseSensor"][:3, :3]
                sd_entry = {
                    "x": gt_pos[0], "y": gt_pos[1], "z": gt_pos[2],
                    "heading": np.degrees(np.arctan2(gt_rot[1, 0], gt_rot[0, 0])), "vel": vel,
                    "rpm": final_rpm_used,
                    "depth_cmd": CRUISE_DEPTH,
                    "target_x": target_pos[0] if target_pos is not None else 0.0,
                    "target_y": target_pos[1] if target_pos is not None else 0.0,
                    "status": stuck.status[name],
                }

                if name in navigators and navigators[name].initialized:
                    nav = navigators[name]
                    ekf_x, ekf_y = nav.get_position_nwu()[:2]
                    sd_entry["nav_error"] = float(np.hypot(gt_pos[0] - ekf_x, gt_pos[1] - ekf_y))
                    sd_entry["ekf_x"] = ekf_x
                    sd_entry["ekf_y"] = ekf_y
                    sd_entry["ekf_std"]  = nav.get_position_std()
                    sd_entry["nis_dvl"]  = nav.nis_dvl
                    sd_entry["dvl_lost"] = float(nav.dvl_lost)
                    sd_entry["curr_true_n"] = current_ned[0]
                    sd_entry["curr_true_e"] = current_ned[1]
                    sd_entry["curr_true_mag"] = float(np.linalg.norm(current_ned[:2]))
                    sd_entry["curr_est_n"] = nav.x_current[0]
                    sd_entry["curr_est_e"] = nav.x_current[1]
                    sd_entry["curr_est_mag"] = float(np.linalg.norm(nav.x_current[:2]))

                agents_metadata[name] = sd_entry
                if dashboard:
                    dashboard.sensor_data[name] = sd_entry

            # Akustik Haberleşme
            for sender in AGENT_NAMES:
                if comm_mgr.should_send_tdma(sender, step):
                    # Komşular global hedeflerden kaçınır; global hedef yoksa o anki hedef bildirilir
                    current_wp = planners[sender].global_target
                    if current_wp is None:
                        current_wp = planners[sender].current_target
                    # Beacon, gönderenin kestirilen konumunu ve belirsizliğini de taşır
                    nav = navigators.get(sender)
                    payload = consensus.encode_map_for_beacon(
                        sender, current_target=current_wp,
                        nav_pos=nav_pos[sender] if nav else None,
                        nav_std=nav.get_position_std() if nav else None)
                    if payload:
                        msgs = comm_mgr.prepare_messages(sender, payload, current_tick=step)
                        for fid, tid, d in msgs:
                            comm_qual.log_send(sender, sim_time=sim_time)
                            ok = comm_mgr.send(env, sender, tid, d, current_tick=step)
                            if ok and dashboard:
                                recv_name = comm_mgr.rev_id_map.get(tid, "")
                                snr = comm_mgr.get_snr(sender, recv_name)
                                dashboard.trigger_beacon_pulse(sender, recv_name, snr)

            # Beacon Alımı
            beacon_received_this_tick = {n: False for n in AGENT_NAMES}
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
                                # Akustik mesafe: gönderenin bildirdiği konuma göre konum düzeltmesi
                                anchor = consensus.get_received_nav(receiver, sender_name, sim_time)
                                if (not args.no_range_aid and receiver in navigators
                                        and anchor is not None and parsed["range"] is not None):
                                    range_updates[receiver] += navigators[receiver].update_range(
                                        parsed["range"], T_NWU_NED @ anchor[0], anchor[1], RANGE_SIGMA_M)

                                if args.no_consensus:
                                    # Ablasyon: gelen harita parçası uygulanmadan atılır
                                    consensus.last_received[receiver] = {}
                                # O2 FIX: RECONNECTING state’de harita senkronizasyonu
                                elif consensus.get_comm_state(receiver) == "RECONNECTING":
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

            if step > 0 and step % STATUS_PRINT_TICKS == 0:
                print(format_status(
                    wall_start, step, sim_time, args.duration, TICKS_PER_SEC,
                    coverage=consensus.get_avg_coverage_pct(),
                    rms=consensus.get_consensus_rms(),
                    pdr=comm_qual.get_summary()["global_pdr"],
                    comm_states={n: consensus.get_comm_state(n) for n in AGENT_NAMES},
                    agents=agents_metadata,
                    current_true_ned=current_ned if USE_OCEAN_CURRENT else None,
                    coverage_union=consensus.get_union_coverage_pct(),
                    range_updates=range_updates if navigators else None), flush=True)

            if dashboard and step % GUI_UPDATE_TICKS == 0:
                dashboard.update(states, targets, comm_qual, "proposed")

            if logger:
                logger.log(
                    step=step,
                    sim_time=sim_time,
                    coverage=consensus.get_avg_coverage_pct(),
                    rms=consensus.get_consensus_rms(),
                    pdr=comm_qual.get_summary()["global_pdr"],
                    agents_state=agents_metadata,
                    snr_matrix=comm_qual.get_snr_matrix(),
                    coverage_union=consensus.get_union_coverage_pct(),
                )
                if step > 0 and step % 1000 == 0:
                    logger.save()

            step += 1

            if all(at_home.values()):
                print(f"[MISSION] Görev tamamlandı: tüm araçlar başlangıç noktasında (t={sim_time:.0f} s)", flush=True)
                break

    except KeyboardInterrupt:
        print("\n[STOP] Kullanıcı durdurdu.")
    except Exception:
        traceback.print_exc()
    finally:
        if logger:
            logger.save()
        if "env" in locals():
            try:
                env.__on_exit__()
            except Exception:
                pass


if __name__ == "__main__":
    main()
