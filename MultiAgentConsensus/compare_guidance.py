"""
Güdüm yöntemlerinin karşılaştırma özeti — koşu loglarından (.m) görev metriklerini çıkarır.

Kullanım:
    python compare_guidance.py etiket=logs/survey/survey_s1 etiket2=logs/mc_cons/bitmap_s2 ...
    python compare_guidance.py --csv ozet.csv entropi=logs/v10/cons_bitmap survey=logs/survey/survey_s1

Aynı etiketle verilen koşular bir yöntemin farklı tohumları sayılır; sonda ortalama ± std yazılır.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

import itertools

AGENTS = ("auv0", "auv1", "auv2")
NEAR_MISS_M = 6.0        # MOOS-IvP uFldCollisionDetect varsayılanları: yakın geçiş 6 m, çarpışma 3 m
TURN_RATE_DPS = 2.0      # bunun üzerindeki dönüş hızı "dönüşte" sayılır (2 s pencere)


def load_m(path):
    """MissionLogger'ın yazdığı .m dosyasındaki sayısal dizileri okur."""
    if not os.path.exists(path) and os.path.exists(path + ".m"):
        path += ".m"
    text = open(path, encoding="utf-8", errors="replace").read()
    data = {}
    for name, body in re.findall(r"^data\.(\w+) = \[(.*?)\];", text, flags=re.M | re.S):
        try:
            data[name] = np.array([float(v) if v.strip() not in ("NaN", "nan", "") else np.nan
                                   for v in body.split(",")])
        except ValueError:
            continue     # metin alanları (durum vb.)
    return data


def first_time(t, series, level):
    idx = np.flatnonzero(series >= level)
    return float(t[idx[0]]) if len(idx) else np.nan


def run_metrics(path):
    d = load_m(path)
    t = d["sim_time"]
    dt = float(np.median(np.diff(t)))
    window = max(1, int(round(2.0 / dt)))
    path_km, turning, nav_err = 0.0, [], []
    for a in AGENTS:
        x, y, hdg = d[f"{a}_x"], d[f"{a}_y"], d[f"{a}_hdg"]
        path_km += float(np.sum(np.hypot(np.diff(x), np.diff(y)))) / 1000.0
        dh = (hdg[window:] - hdg[:-window] + 180.0) % 360.0 - 180.0
        turning.append(np.abs(dh) / (window * dt) > TURN_RATE_DPS)
        nav_err.append(d[f"{a}_nav_error"])
    nav_err = np.concatenate(nav_err)
    ranges = np.array([np.hypot(d[f"{a}_x"] - d[f"{b}_x"], d[f"{a}_y"] - d[f"{b}_y"])
                       for a, b in itertools.combinations(AGENTS, 2)])
    return {
        "gorev_suresi_s": float(t[-1]),
        "birlesik_90_s": first_time(t, d["coverage_union"], 90.0),
        "birlesik_99_s": first_time(t, d["coverage_union"], 99.0),
        "son_birlesik_%": float(d["coverage_union"][-1]),
        "son_arac_ort_%": float(d["coverage"][-1]),
        "toplam_yol_km": path_km,
        "donuste_gecen_%": 100.0 * float(np.mean(np.concatenate(turning))),
        "en_yakin_arac_m": float(ranges.min()),
        "yakin_gecis_s": float(np.sum(ranges.min(axis=0) < NEAR_MISS_M) * dt),
        "nav_hata_ort_m": float(np.nanmean(nav_err)),
        "nav_hata_maks_m": float(np.nanmax(nav_err)),
        "paket_teslim_%": 100.0 * float(d["pdr"][-1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="etiket=log_yolu (uzantısız)")
    parser.add_argument("--csv", default=None, help="Koşu başına satırların yazılacağı dosya")
    args = parser.parse_args()

    rows = []
    for item in args.runs:
        label, path = item.split("=", 1)
        rows.append({"yontem": label, "log": path, **run_metrics(path)})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.2f}".format)
    print(df.drop(columns="log").to_string(index=False))
    if df.yontem.duplicated().any():
        numeric = df.drop(columns=["log"]).groupby("yontem", sort=False)
        print("\nOrtalama:\n" + numeric.mean().to_string())
        print("\nStd:\n" + numeric.std().to_string())
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\n{args.csv} yazıldı")


if __name__ == "__main__":
    main()
