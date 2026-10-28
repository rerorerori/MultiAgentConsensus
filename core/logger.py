"""
Mission Logger — MATLAB (.m) format telemetri kaydedicisi.
==========================================================
Yenilikler (v2.1):
  • Public SNR matrix API kullanımı (_snr_cache private erişim kaldırıldı)
  • SNR ortalamaları comm_quality'den gelir (daha anlamlı metrik)
"""
import numpy as np
import os
import time


class MissionLogger:
    """Simülasyon verilerini toplayıp MATLAB (.m) formatında kaydeden sınıf."""

    def __init__(self, filename, agent_names):
        self.filename = filename
        self.agent_names = agent_names
        self.data = {
            "step": [],
            "sim_time": [],
            "coverage": [],
            "coverage_union": [],
            "rms": [],
            "pdr": []
        }
        for name in agent_names:
            self.data[f"{name}_x"] = []
            self.data[f"{name}_y"] = []
            self.data[f"{name}_z"] = []
            self.data[f"{name}_hdg"] = []
            self.data[f"{name}_vel"] = []
            self.data[f"{name}_rpm"] = []
            self.data[f"{name}_tx"] = []
            self.data[f"{name}_ty"] = []
            self.data[f"{name}_status"] = []
            self.data[f"{name}_nav_error"] = []
            self.data[f"{name}_ekf_x"]     = []
            self.data[f"{name}_ekf_y"]     = []
            self.data[f"{name}_ekf_std"]   = []
            self.data[f"{name}_nis_dvl"]   = []
            self.data[f"{name}_dvl_lost"]  = []
            self.data[f"{name}_curr_true_n"]   = []
            self.data[f"{name}_curr_true_e"]   = []
            self.data[f"{name}_curr_true_mag"] = []
            self.data[f"{name}_curr_est_n"]    = []
            self.data[f"{name}_curr_est_e"]    = []
            self.data[f"{name}_curr_est_mag"]  = []
            # Neighbor SNR logs
            for other in agent_names:
                if other != name:
                    self.data[f"{name}_{other}_snr"] = []

    def log(self, step, sim_time, coverage, rms, pdr, agents_state, snr_matrix=None,
            coverage_union=float("nan")):
        """
        Bir zaman dilimine ait verileri tampona ekle.

        snr_matrix: dict {"sender→receiver": snr_db, ...}
                   Tercihen comm_quality.get_snr_matrix() (ortalamalar) veya
                   comm_manager.get_snr_matrix() (anlık değerler).
        """
        self.data["step"].append(step)
        self.data["sim_time"].append(sim_time)
        self.data["coverage"].append(coverage)
        self.data["coverage_union"].append(coverage_union)
        self.data["rms"].append(rms)
        self.data["pdr"].append(pdr)

        for name in self.agent_names:
            sd = agents_state.get(name, {})
            self.data[f"{name}_x"].append(sd.get("x", 0.0))
            self.data[f"{name}_y"].append(sd.get("y", 0.0))
            self.data[f"{name}_z"].append(sd.get("z", 0.0))
            self.data[f"{name}_hdg"].append(sd.get("heading", 0.0))
            self.data[f"{name}_vel"].append(sd.get("vel", 0.0))
            self.data[f"{name}_rpm"].append(sd.get("rpm", 0.0))
            self.data[f"{name}_tx"].append(sd.get("target_x", 0.0))
            self.data[f"{name}_ty"].append(sd.get("target_y", 0.0))

            # Status mapping: OK=0, RECOVERING=1
            str_status = sd.get("status", "OK")
            self.data[f"{name}_status"].append(
                1.0 if str_status == "RECOVERING" else 0.0)
            self.data[f"{name}_nav_error"].append(sd.get("nav_error", float('nan')))
            self.data[f"{name}_ekf_x"].append(sd.get("ekf_x",     float('nan')))
            self.data[f"{name}_ekf_y"].append(sd.get("ekf_y",     float('nan')))
            self.data[f"{name}_ekf_std"].append(sd.get("ekf_std",   float('nan')))
            self.data[f"{name}_nis_dvl"].append(sd.get("nis_dvl",   float('nan')))
            self.data[f"{name}_dvl_lost"].append(sd.get("dvl_lost", float('nan')))
            self.data[f"{name}_curr_true_n"].append(sd.get("curr_true_n", float('nan')))
            self.data[f"{name}_curr_true_e"].append(sd.get("curr_true_e", float('nan')))
            self.data[f"{name}_curr_true_mag"].append(sd.get("curr_true_mag", float('nan')))
            self.data[f"{name}_curr_est_n"].append(sd.get("curr_est_n", float('nan')))
            self.data[f"{name}_curr_est_e"].append(sd.get("curr_est_e", float('nan')))
            self.data[f"{name}_curr_est_mag"].append(sd.get("curr_est_mag", float('nan')))

            if snr_matrix:
                for other in self.agent_names:
                    if other != name:
                        # "auv0→auv1" formatında anahtar beklenir
                        key = f"{name}→{other}"
                        val = snr_matrix.get(key, float('nan'))
                        self.data[f"{name}_{other}_snr"].append(val)

    def save(self):
        """Mevcut verileri disk üzerindeki .m ve .csv dosyalarına yazar."""
        if not self.filename:
            return
        target_path = os.path.abspath(os.path.normpath(self.filename))
        dirname = os.path.dirname(target_path)
        if dirname and not os.path.exists(dirname):
            try:
                os.makedirs(dirname, exist_ok=True)
            except OSError:
                pass
        try:
            with open(target_path, "w", encoding="utf-8") as f:
                f.write("% Swarm AUV Mission Log File\n")
                f.write(f"% Created: {time.ctime()}\n")
                f.write(f"% Agents: {', '.join(self.agent_names)}\n\n")

                f.write("data = struct();\n")

                for key, values in self.data.items():
                    # NaN değerlerini MATLAB uyumlu yaz
                    val_strs = []
                    for v in values:
                        if v is None or (isinstance(v, float) and np.isnan(v)):
                            val_strs.append("NaN")
                        else:
                            val_strs.append(str(v))

                    arr_str = ", ".join(val_strs)
                    f.write(f"data.{key} = [{arr_str}];\n")

                f.write("\n% End of Mission Log\n")
            print(f"[LOGGER] Log file saved to: {self.filename}")

            # CSV formatında da kaydet (ekf_log.csv uyumlu)
            csv_path = os.path.splitext(target_path)[0] + ".csv"
            try:
                import pandas as pd
                # CSV sütunu -> ajan bazlı log alanı
                csv_fields = {
                    "gt_x": "x", "gt_y": "y", "ekf_x": "ekf_x", "ekf_y": "ekf_y",
                    "nav_error": "nav_error", "ekf_std": "ekf_std",
                    "nis_dvl": "nis_dvl", "dvl_lost": "dvl_lost",
                    "curr_true_n": "curr_true_n", "curr_true_e": "curr_true_e",
                    "curr_true_mag": "curr_true_mag", "curr_est_n": "curr_est_n",
                    "curr_est_e": "curr_est_e", "curr_est_mag": "curr_est_mag",
                }
                rows = []
                for i, (st, t) in enumerate(zip(self.data["step"], self.data["sim_time"])):
                    for name in self.agent_names:
                        row = {"step": st, "timestamp": t, "agent": name}
                        for col, field in csv_fields.items():
                            row[col] = self.data[f"{name}_{field}"][i]
                        rows.append(row)
                df = pd.DataFrame(rows)
                df.to_csv(csv_path, index=False)
                print(f"[LOGGER] CSV log saved to: {csv_path}")
            except Exception as e:
                print(f"[LOGGER] CSV save warning: {e}")

        except Exception as e:
            print(f"[LOGGER] Error saving log file: {e}")