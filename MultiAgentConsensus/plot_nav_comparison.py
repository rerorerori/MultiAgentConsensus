"""
Swarm AUV — EKF & GT Yörünge ve Navigasyon Hatası Karşılaştırma Grafiği
=====================================================================
Kullanım:
    python plot_nav_comparison.py <log.csv | log.m> [--t_min 0] [--t_max 200] [--out_dir DIR] [--export]

Çıktılar (--out_dir, varsayılan bu betiğin klasörü):
    nav_comparison, nav_trajectory, nav_error, nav_current  (.pdf + .png)
--export verilirse grafikler bildiri klasörüne de kopyalanır.

Konsola yazılan istatistikler ile grafiklerdeki ortalamalar aynı zaman
penceresinden (--t_min/--t_max) hesaplanır.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AGENTS = ["auv0", "auv1", "auv2"]
COLORS = {"auv0": "#1f77b4", "auv1": "#ff7f0e", "auv2": "#2ca02c"}
LABELS = {"auv0": "AUV-0", "auv1": "AUV-1", "auv2": "AUV-2"}

# .m log alanı -> CSV sütunu
M_FIELDS = {
    "x": "gt_x", "y": "gt_y", "ekf_x": "ekf_x", "ekf_y": "ekf_y", "nav_error": "nav_error",
    "ekf_std": "ekf_std", "dvl_lost": "dvl_lost",
    "curr_true_n": "curr_true_n", "curr_true_e": "curr_true_e", "curr_true_mag": "curr_true_mag",
    "curr_est_n": "curr_est_n", "curr_est_e": "curr_est_e", "curr_est_mag": "curr_est_mag",
}

EXPORT_DIRS = [Path(r"C:\Users\reror\Downloads\TOK_"), Path(r"C:\Users\reror\Downloads")]


# ── 1. LOG OKUMA ──────────────────────────────────────────────────────────────

def parse_m_log(filepath):
    """MATLAB .m formatını uzun (ajan başına satır) DataFrame formatına çevir."""
    data = {}
    pat = re.compile(r"^data\.(\w+)\s*=\s*\[([^\]]*)\];", re.MULTILINE)
    for m in pat.finditer(Path(filepath).read_text(encoding="utf-8")):
        raw = m.group(2).strip()
        data[m.group(1)] = np.array([float(tok) for tok in raw.split(",")]) if raw else np.array([])

    frames = []
    for ag in AGENTS:
        frame = pd.DataFrame({"step": data["step"], "timestamp": data["sim_time"], "agent": ag})
        for field, col in M_FIELDS.items():
            if f"{ag}_{field}" in data:
                frame[col] = data[f"{ag}_{field}"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_data(filepath):
    p = Path(filepath)
    if p.suffix.lower() == ".m":
        return parse_m_log(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Desteklenmeyen dosya formati: {filepath}")


def agent_frame(df, ag):
    return df[df["agent"] == ag].sort_values("step")


# ── 2. ÇİZİM FONKSİYONLARI ────────────────────────────────────────────────────

def plot_trajectory(ax, df):
    """XY yörünge. Log NWU çerçevesindedir: x = Kuzey, y = Batı → Doğu = -y."""
    gt_label = "Ground Truth (GT)"
    for ag in AGENTS:
        a = agent_frame(df, ag)
        if a.empty:
            continue
        ax.plot(-a["gt_y"], a["gt_x"], "--k", linewidth=1.3, alpha=0.8, label=gt_label)
        gt_label = None

        e = a.dropna(subset=["ekf_x", "ekf_y"])
        if not e.empty:
            ax.plot(-e["ekf_y"], e["ekf_x"], color=COLORS[ag], linewidth=1.5, alpha=0.9,
                    label=f"EKF {LABELS[ag]}")
            ax.plot(-e["ekf_y"].iloc[0], e["ekf_x"].iloc[0], marker="*", color=COLORS[ag],
                    markersize=12, markeredgecolor="black", markeredgewidth=0.8, zorder=5)

    ax.set_title("EKF ve GT Yörünge Karşılaştırması", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Doğu / East (m)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Kuzey / North (m)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.set_aspect("equal", "datalim")


def shade_dvl_outage(ax, df):
    """DVL'in kayıp olduğu aralıkları gri bantla işaretle."""
    if "dvl_lost" not in df.columns:
        return
    a = agent_frame(df, AGENTS[0])
    lost = a["dvl_lost"].fillna(0).values > 0.5
    if not lost.any():
        return
    t = a["timestamp"].values
    edges = np.flatnonzero(np.diff(np.r_[0, lost.astype(int), 0]))
    for i0, i1 in zip(edges[::2], edges[1::2]):
        ax.axvspan(t[i0], t[i1 - 1], color="0.85", zorder=0, label="DVL kesintisi" if i0 == edges[0] else None)


def plot_error(ax, df, stats):
    shade_dvl_outage(ax, df)
    for ag in AGENTS:
        a = agent_frame(df, ag).dropna(subset=["nav_error"])
        if a.empty:
            continue
        ax.plot(a["timestamp"], a["nav_error"], color=COLORS[ag], linewidth=1.3, alpha=0.85, label=LABELS[ag])
        if ag in stats:
            ax.axhline(stats[ag]["mean"], color=COLORS[ag], linestyle="--", linewidth=1.0, alpha=0.8,
                       label=f"{LABELS[ag]} Ort: {stats[ag]['mean']:.2f} m")

    ax.set_title("Zamana Bağlı EKF Navigasyon Hatası", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Zaman (s)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Konum Hatası (m)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)


def plot_current(ax, df):
    """auv0 için gerçek ve kestirilen akıntı büyüklüğü. Veri yoksa False döndürür."""
    if "curr_true_mag" not in df.columns or "curr_est_mag" not in df.columns:
        return False
    a = agent_frame(df, "auv0").dropna(subset=["curr_true_mag", "curr_est_mag"])
    if a.empty:
        return False
    rmse = float(np.sqrt(np.mean((a["curr_true_mag"] - a["curr_est_mag"]) ** 2)))
    shade_dvl_outage(ax, df)
    ax.plot(a["timestamp"], a["curr_true_mag"], "k-", linewidth=1.6, label=r"Gerçek Akıntı $|\mathbf{V}_c^{true}|$")
    ax.plot(a["timestamp"], a["curr_est_mag"], "b--", linewidth=1.6,
            label=rf"EKF Kestirimi $|\hat{{\mathbf{{V}}}}_c|$ (RMSE: {rmse:.3f} m/s)")
    ax.set_title("Okyanus Akıntısı Büyüklüğü Kestirim Performansı", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Zaman (s)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Akıntı Hızı (m/s)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    return True


def error_stats(df):
    """Ajan bazlı ve tüm sürü için navigasyon hatası istatistikleri."""
    stats = {}
    for ag in AGENTS + ["swarm"]:
        sel = df if ag == "swarm" else df[df["agent"] == ag]
        err = sel["nav_error"].dropna().values
        if len(err):
            stats[ag] = {"n": len(err), "mean": float(err.mean()),
                         "rms": float(np.sqrt((err ** 2).mean())), "max": float(err.max())}
    return stats


def save(fig, out_dir, name):
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    for p in paths:
        fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return paths


# ── 3. ANA PROGRAM ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", help="Log dosyası (.csv veya .m)")
    parser.add_argument("--t_min", type=float, default=0.0, help="Pencere başlangıcı (s)")
    parser.add_argument("--t_max", type=float, default=None, help="Pencere sonu (s). Varsayılan: log sonu")
    parser.add_argument("--out_dir", default=None, help="Çıktı klasörü. Varsayılan: betiğin klasörü")
    parser.add_argument("--export", action="store_true", help="Grafikleri bildiri klasörüne de kopyala")
    args = parser.parse_args()

    print(f"[INFO] Log dosyasi yukleniyor: {args.log}")
    df = load_data(args.log)
    t_max = args.t_max if args.t_max is not None else float(df["timestamp"].max())
    df = df[(df["timestamp"] >= args.t_min) & (df["timestamp"] <= t_max)]
    print(f"[INFO] Pencere: {args.t_min:.0f}-{t_max:.0f} s, kayit: {len(df)}")

    stats = error_stats(df)
    print("\n" + "=" * 58)
    print(f"  EKF Navigasyon Hatası İstatistikleri ({args.t_min:.0f}-{t_max:.0f} s)")
    print("=" * 58)
    print(f"  {'Araç':<8} {'Örnek':>8} {'Ortalama (m)':>14} {'RMS (m)':>12} {'Maks (m)':>12}")
    print("-" * 58)
    for ag, s in stats.items():
        label = LABELS.get(ag, "Sürü")
        print(f"  {label:<8} {s['n']:>8} {s['mean']:>14.3f} {s['rms']:>12.3f} {s['max']:>12.3f}")
    print("=" * 58)

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.5))
    plot_trajectory(ax1, df)
    plot_error(ax2, df, stats)
    fig.tight_layout(pad=2.0)
    outputs += save(fig, out_dir, "nav_comparison")

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    plot_trajectory(ax, df)
    fig.tight_layout()
    outputs += save(fig, out_dir, "nav_trajectory")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    plot_error(ax, df, stats)
    fig.tight_layout()
    outputs += save(fig, out_dir, "nav_error")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    if plot_current(ax, df):
        fig.tight_layout()
        outputs += save(fig, out_dir, "nav_current")
    else:
        plt.close(fig)

    print("\n[OK] Grafikler kaydedildi:")
    for p in outputs:
        print(f"     -> {p}")

    for dst_folder in EXPORT_DIRS if args.export else []:
        if dst_folder.exists():
            for p in outputs:
                try:
                    shutil.copy(p, dst_folder / p.name)
                except OSError:
                    pass


if __name__ == "__main__":
    sys.exit(main())
