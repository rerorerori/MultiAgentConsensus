"""
Konsol durum raporu — periyodik olarak basılan özet tablo.
"""
import math
import time
from datetime import datetime


def _fmt(value, spec, missing="   -"):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return missing
    return format(value, spec)


def format_status(wall_start, step, sim_time, duration_ticks, ticks_per_sec,
                  coverage, rms, pdr, comm_states, agents, current_true_ned=None,
                  range_updates=None, coverage_union=None):
    """
    wall_start: koşunun başladığı time.time() değeri
    duration_ticks: toplam süre (tick), 0 ise sınırsız
    comm_states: {ajan: "CONNECTED" | "DEGRADED" | ...}
    agents: {ajan: main döngüsündeki sd_entry sözlüğü}
    current_true_ned: gerçek akıntı [N, E, D] (yoksa None)
    range_updates: {ajan: kabul edilen akustik mesafe güncellemesi sayısı}
    coverage: araçların kendi haritalarındaki kapsamanın ortalaması
    coverage_union: en az bir aracın taradığı hücrelerin oranı
    """
    elapsed = time.time() - wall_start
    rtf = sim_time / elapsed if elapsed > 1e-6 else float("nan")
    lines = ["=" * 100]

    head = (f"[TIME] {datetime.now():%Y-%m-%d %H:%M:%S} | sim t={sim_time:7.1f}s  step={step:6d} | "
            f"geçen={elapsed / 60.0:5.1f} dk | hız={rtf:4.2f}x gerçek zaman")
    if duration_ticks > 0 and step > 0:
        remaining = (duration_ticks - step) * elapsed / step
        head += f" | kalan~{remaining / 60.0:4.1f} dk (%{100.0 * step / duration_ticks:3.0f})"
    lines.append(head)

    states = "  ".join(f"{n}:{s[:3]}" for n, s in comm_states.items())
    union = "" if coverage_union is None else f" (birleşik %{coverage_union:5.1f})"
    lines.append(f"[SWARM] kapsama=%{coverage:5.1f}{union} | konsensüs RMS={rms:.4f} | paket teslim=%{100.0 * pdr:4.0f} | "
                 f"haberleşme: {states}")

    lines.append("[AUV]  ad    konum (x,y) m     derinlik  baş°   hız   rpm   hedef (x,y)   mesafe  durum  "
                 "nav hata  1sig    DVL    NIS  mesafe#")
    nav_errors = []
    for name, sd in agents.items():
        if not sd:
            continue
        x, y = sd.get("x", float("nan")), sd.get("y", float("nan"))
        tx, ty = sd.get("target_x", float("nan")), sd.get("target_y", float("nan"))
        dist = math.hypot(tx - x, ty - y)
        err = sd.get("nav_error")
        if err is not None and not math.isnan(err):
            nav_errors.append(err)
        lost = sd.get("dvl_lost")
        dvl = "   -" if lost is None else ("KAYIP" if lost > 0.5 else "kilit")
        n_range = range_updates.get(name, 0) if range_updates else 0
        lines.append(
            f"       {name:<5} ({x:+7.1f},{y:+7.1f})  {abs(sd.get('z', float('nan'))):7.1f}  "
            f"{sd.get('heading', float('nan')):+5.0f}  {sd.get('vel', float('nan')):4.2f}  {sd.get('rpm', 0):4.0f}  "
            f"({tx:+5.0f},{ty:+5.0f})  {dist:6.1f}  {str(sd.get('status', '-'))[:5]:<5}  "
            f"{_fmt(err, '6.2f')} m  {_fmt(sd.get('ekf_std'), '5.2f')}  {dvl:>5}  {_fmt(sd.get('nis_dvl'), '5.1f')}  {n_range:5d}")

    if nav_errors:
        lines.append(f"[NAV]  EKF yatay hata: ort={sum(nav_errors) / len(nav_errors):.2f} m  maks={max(nav_errors):.2f} m")

    if current_true_ned is not None:
        cn, ce = float(current_true_ned[0]), float(current_true_ned[1])
        line = f"[CURR] gerçek (K,D)=({cn:+.3f},{ce:+.3f}) |{math.hypot(cn, ce):.3f}| m/s"
        est = [(sd["curr_est_n"], sd["curr_est_e"]) for sd in agents.values() if sd and "curr_est_n" in sd]
        if est:
            parts = [f"({n:+.3f},{e:+.3f})" for n, e in est]
            err = [math.hypot(n - cn, e - ce) for n, e in est]
            line += f" | kestirim: {' '.join(parts)} | hata ort={sum(err) / len(err):.3f} m/s"
        lines.append(line)

    lines.append("-" * 100)
    return "\n".join(lines)
