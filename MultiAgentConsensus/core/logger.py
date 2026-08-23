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
            # Neighbor SNR logs
            for other in agent_names:
                if other != name:
                    self.data[f"{name}_{other}_snr"] = []

    def log(self, step, sim_time, coverage, rms, pdr, agents_state, snr_matrix=None):
        """
        Bir zaman dilimine ait verileri tampona ekle.

        snr_matrix: dict {"sender→receiver": snr_db, ...}
                   Tercihen comm_quality.get_snr_matrix() (ortalamalar) veya
                   comm_manager.get_snr_matrix() (anlık değerler).
        """
        self.data["step"].append(step)
        self.data["sim_time"].append(sim_time)
        self.data["coverage"].append(coverage)
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

            if snr_matrix:
                for other in self.agent_names:
                    if other != name:
                        # "auv0→auv1" formatında anahtar beklenir
                        key = f"{name}→{other}"
                        val = snr_matrix.get(key, float('nan'))
                        self.data[f"{name}_{other}_snr"].append(val)

    def save(self):
        """Tampondaki verileri .m dosyasına yaz."""
        if not self.filename:
            return

        # Windows için yol normalleştirme
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
                rows = []
                steps = self.data.get("step", [])
                sim_times = self.data.get("sim_time", [])
                n_steps = len(steps)
                for i in range(n_steps):
                    st = steps[i]
                    t = sim_times[i] if i < len(sim_times) else st / 30.0
                    for name in self.agent_names:
                        gx = self.data[f"{name}_x"][i] if f"{name}_x" in self.data else 0.0
                        gy = self.data[f"{name}_y"][i] if f"{name}_y" in self.data else 0.0
                        ex = self.data[f"{name}_ekf_x"][i] if f"{name}_ekf_x" in self.data else float("nan")
                        ey = self.data[f"{name}_ekf_y"][i] if f"{name}_ekf_y" in self.data else float("nan")
                        ne = self.data[f"{name}_nav_error"][i] if f"{name}_nav_error" in self.data else float("nan")
                        rows.append({
                            "step": st,
                            "timestamp": t,
                            "agent": name,
                            "gt_x": gx,
                            "gt_y": gy,
                            "ekf_x": ex,
                            "ekf_y": ey,
                            "nav_error": ne
                        })
                df = pd.DataFrame(rows)
                df.to_csv(csv_path, index=False)
                print(f"[LOGGER] CSV log saved to: {csv_path}")
            except Exception as e:
                print(f"[LOGGER] CSV save warning: {e}")

        except Exception as e:
            print(f"[LOGGER] Error saving log file: {e}")