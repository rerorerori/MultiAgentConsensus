"""
Monte Carlo koşucusu — akıntı senaryoları x seed'ler.

Kullanım:
    python run_monte_carlo.py                               # 3 senaryo x 3 seed, akıntı telafili
    python run_monte_carlo.py --seeds 10 --with_baseline    # her koşu aynı seed ile baseline'la eşlenir
    python run_monte_carlo.py --dvl_outage 150 210          # DVL kesintili karşılaştırma
    python run_monte_carlo.py --dvl_outage 150 210 --outage_agents auv1 --no_range_aid

Tüm istatistikler aynı zaman penceresinde (--t_min/--t_max) hesaplanır; senaryo
özetindeki "±" değeri seed'ler arası standart sapmadır.
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

SCENARIOS = [("low", 0.10), ("mid", 0.22), ("high", 0.40)]
TICKS_PER_SEC = 30


def run_stats(csv_path, t_min, t_max):
    """Bir koşunun pencere içi navigasyon ve akıntı kestirim istatistikleri."""
    df = pd.read_csv(csv_path)
    df = df[(df["timestamp"] >= t_min) & (df["timestamp"] <= t_max)]
    err = df["nav_error"].dropna().values
    cur_err = np.hypot(df["curr_true_n"] - df["curr_est_n"], df["curr_true_e"] - df["curr_est_e"])
    return {
        "mean": float(np.mean(err)),
        "rms": float(np.sqrt(np.mean(err ** 2))),
        "max": float(np.max(err)),
        "curr_rmse": float(np.sqrt(np.nanmean(cur_err ** 2))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3, help="Senaryo başına koşu sayısı")
    parser.add_argument("--duration_s", type=float, default=300.0, help="Koşu süresi (s)")
    parser.add_argument("--t_min", type=float, default=0.0, help="İstatistik penceresi başlangıcı (s)")
    parser.add_argument("--t_max", type=float, default=None, help="İstatistik penceresi sonu (s). Varsayılan: koşu sonu")
    parser.add_argument("--with_baseline", action="store_true", help="Her koşuyu aynı seed ile akıntı telafisiz tekrarla")
    parser.add_argument("--dvl_outage", type=float, nargs=2, default=None, metavar=("T_START", "T_END"))
    parser.add_argument("--outage_agents", nargs="+", default=None, help="Kesintinin uygulanacağı araçlar (varsayılan: hepsi)")
    parser.add_argument("--no_range_aid", action="store_true", help="Akustik mesafe güncellemesi kapalı")
    parser.add_argument("--out_dir", default="logs/mc")
    args = parser.parse_args()

    t_max = args.t_max if args.t_max is not None else args.duration_s
    methods = ["proposed", "baseline"] if args.with_baseline else ["proposed"]
    os.makedirs(args.out_dir, exist_ok=True)

    runs = [(scen, mag, seed, method)
            for scen, mag in SCENARIOS
            for seed in range(1, args.seeds + 1)
            for method in methods]

    print("=" * 80)
    print(f"  MONTE CARLO: {len(runs)} koşu ({len(SCENARIOS)} senaryo x {args.seeds} seed x {len(methods)} yöntem)")
    print(f"  İstatistik penceresi: {args.t_min:.0f}-{t_max:.0f} s")
    print("=" * 80)

    results = []
    start_total = time.time()
    for idx, (scen, mag, seed, method) in enumerate(runs, 1):
        log_path = os.path.join(args.out_dir, f"mc_{scen}_{seed}_{method}.m")
        csv_path = os.path.splitext(log_path)[0] + ".csv"

        cmd = [
            sys.executable, "main_ekf_current.py",
            "--duration", str(int(args.duration_s * TICKS_PER_SEC)),
            "--current_mag", str(mag),
            "--seed", str(seed),
            "--log", log_path,
            "--headless",
        ]
        if method == "baseline":
            cmd.append("--no_current_comp")
        if args.dvl_outage:
            cmd += ["--dvl_outage", str(args.dvl_outage[0]), str(args.dvl_outage[1])]
            if args.outage_agents:
                cmd += ["--outage_agents", *args.outage_agents]
        if args.no_range_aid:
            cmd.append("--no_range_aid")

        print(f"\n[{idx}/{len(runs)}] {scen} | {mag:.2f} m/s | seed={seed} | {method} ...", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        elapsed = time.time() - t0

        # Önceki bir koşudan kalmış CSV'yi bu koşunun sonucu sanma
        if not os.path.exists(csv_path) or os.path.getmtime(csv_path) < t0:
            print(f"[{idx}/{len(runs)}] BAŞARISIZ ({elapsed:.1f}s):\n{proc.stdout[-500:]}", flush=True)
            continue

        stats = run_stats(csv_path, args.t_min, t_max)
        results.append({"scenario": scen, "mag": mag, "seed": seed, "method": method, **stats, "csv": csv_path})
        print(f"[{idx}/{len(runs)}] {elapsed:.1f}s | Ort={stats['mean']:.3f}m | RMS={stats['rms']:.3f}m | "
              f"Maks={stats['max']:.3f}m | AkıntıRMSE={stats['curr_rmse']:.3f}m/s", flush=True)

    print("\n" + "=" * 80)
    print(f"  {len(results)}/{len(runs)} koşu tamamlandı, {(time.time() - start_total) / 60.0:.1f} dakika")
    print("=" * 80)
    if not results:
        return

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(args.out_dir, "monte_carlo_runs.csv"), index=False)

    # Senaryo özeti: seed'ler arası ortalama ve standart sapma
    summary = (df_res.groupby(["scenario", "mag", "method"], sort=False)
               .agg(n=("mean", "size"),
                    mean=("mean", "mean"), mean_std=("mean", lambda s: s.std(ddof=1) if len(s) > 1 else 0.0),
                    rms=("rms", "mean"), max=("max", "max"), curr_rmse=("curr_rmse", "mean"))
               .reset_index())
    summary.to_csv(os.path.join(args.out_dir, "monte_carlo_summary.csv"), index=False)

    print("\n--- KOŞU SONUÇLARI ---")
    print(df_res.drop(columns="csv").to_string(index=False, float_format="%.3f"))
    print("\n--- SENARYO ÖZETİ (mean_std: seed'ler arası) ---")
    print(summary.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
