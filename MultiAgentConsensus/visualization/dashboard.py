"""
Dashboard — PMViewer-style Mission Control GUI  v5.0 (Unified & Simplified)
========================================================================
A consolidated, visually simplified 2D swarm mission control GUI.
Combines panels.py and map_canvas.py into a single high-performance file.
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
import math
import hashlib
import io
import urllib.request
import numpy as np
import time
import threading
from collections import deque

# PIL opsiyonel
try:
    from PIL import Image, ImageTk, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec

from config import (
    AGENT_NAMES, MAP_SIZE_M, TRAJECTORY_LEN, GUI_UPDATE_TICKS,
    MAP_BACKGROUND_IMAGE, MAP_BACKGROUND_BRIGHTNESS, MAP_BACKGROUND_TINT_ALPHA,
    FLS_THRESHOLD, SENSOR_RANGE
)

# ═══════════════════════════════════════════════════════════════
# TEMA VE RENK PALETİ (~%30 Saturation Muted)
# ═══════════════════════════════════════════════════════════════
BG        = "#0a0e1a"
BG2       = "#111827"
BG3       = "#1a2332"
ACCENT    = "#22a7c4"  # Muted from #00d4ff
ACCENT2   = "#d9744b"  # Muted from #ff6b35
GREEN     = "#36c986"  # Muted from #00ff88
RED       = "#c94b62"  # Muted from #ff3355
YELLOW    = "#cfb33c"  # Muted from #ffd700
PURPLE    = "#997b94"  # Muted from #b48ead
GRAY      = "#4a5568"
TEXT      = "#e2e8f0"
TEXT_DIM  = "#718096"

AGENT_COLORS = {
    "auv0": "#22a7c4",
    "auv1": "#d9744b",
    "auv2": "#36c986",
}
ALL_AGENTS = ["auv0", "auv1", "auv2"]

STATE_COLORS = {
    "CONNECTED":    GREEN,
    "RECONNECTING": ACCENT,
    "DEGRADED":     YELLOW,
    "OFFLINE":      RED,
    "OK":           GREEN,
    "RECOVERING":   YELLOW,
}

FONT_MONO  = ("IBM Plex Mono", 9)
FONT_TITLE = ("IBM Plex Mono", 10, "bold")
FONT_SMALL = ("IBM Plex Mono", 8)
FONT_TINY  = ("IBM Plex Mono", 7)

# MapCanvas Specific Colors/Constants
BG_MAP      = "#0a1520"
GRID_MAJOR  = "#1e3048"   # biraz görünür, tile üzerine çok baskın olmaz
GRID_MINOR  = "#131e2e"   # çok soluk, sadece referans
GRID_LABEL  = "#2a4a6a"
OPAREA_COL  = "#884422"   # Daha mat kırmızı/kahve sınır

AGENT_COLOR_DEFAULT = "#cfb33c"
AGENT_COLOR_SEL     = "#c9364f"

STATE_LINK_COLORS = {
    "CONNECTED":    "#36c986",
    "DEGRADED":     "#cfb33c",
    "RECONNECTING": "#22a7c4",
    "OFFLINE":      "#c94b62",
}

TRAIL_ALPHA_STEPS = 16
STALE_WARN_S      = 10.0
STALE_DEAD_S      = 30.0
EXTRAP_START_S    = 5.0

# ─── Tile Sistemi ───────────────────────────────────────────────────────────────────
TILE_PROVIDERS = {
    "esri_satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "carto_dark":     "https://cartodb-basemaps-a.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png",
    "osm":            "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
}
TILE_DISK_CACHE = os.path.expanduser("~/.cache/swarm_dashboard/tiles")
TILE_USER_AGENT = "swarm-auv-dashboard/1.0"

# HoloOcean sim (0,0) noktasının gerçek dünya karşılığı
MAP_DATUM_LAT = 38.1800
MAP_DATUM_LON = 26.7676      # Sığacık Körfezi — kıyı tam +300m hizalandı (100m güvenlik tamponu)
EARTH_R       = 6378137.0


def _deg2tile(lat_deg, lon_deg, zoom):
    lat_r = math.radians(lat_deg)
    n = 2 ** zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def _tile2deg(tx, ty, zoom):
    n = 2 ** zoom
    lon = tx / n * 360.0 - 180.0
    lat_r = math.atan(math.sinh(math.pi * (1 - 2 * ty / n)))
    return math.degrees(lat_r), lon


def _local_to_latlon(east_m, north_m,
                     lat0=MAP_DATUM_LAT, lon0=MAP_DATUM_LON):
    lat = lat0 + math.degrees(north_m / EARTH_R)
    lon = lon0 + math.degrees(
        east_m / (EARTH_R * math.cos(math.radians(lat0))))
    return lat, lon


def _latlon_to_local(lat, lon,
                     lat0=MAP_DATUM_LAT, lon0=MAP_DATUM_LON):
    east_m  = math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    north_m = math.radians(lat - lat0) * EARTH_R
    return east_m, north_m


class TileManager:
    """
    Tkinter PhotoImage tile yöneticisi.
    PIL Image: mem cache (provider/zoom bağımsız, uzun ömrülü)
    ImageTk  : boyut+opacity bağımlı, zoom/pan sonrası temizlenir.
    """

    def __init__(self, provider="esri_satellite"):
        self.provider   = provider
        self.url_tmpl   = TILE_PROVIDERS[provider]
        self._mem_cache = {}   # (z,x,y) → PIL.Image
        self._tk_cache  = {}   # (z,x,y,w,h,op,prov) → ImageTk.PhotoImage
        self._pending   = set()
        self._lock      = threading.Lock()
        os.makedirs(TILE_DISK_CACHE, exist_ok=True)

    def _disk_path(self, z, x, y):
        key   = f"{self.provider}_{z}_{x}_{y}"
        fname = hashlib.md5(key.encode()).hexdigest() + ".png"
        return os.path.join(TILE_DISK_CACHE, fname)

    def _fetch_thread(self, z, x, y):
        url  = self.url_tmpl.format(z=z, x=x, y=y)
        disk = self._disk_path(z, x, y)
        try:
            req  = urllib.request.Request(
                url, headers={"User-Agent": TILE_USER_AGENT})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = resp.read()
            with open(disk, "wb") as f:
                f.write(data)
            img = Image.open(io.BytesIO(data)).convert("RGB")
            with self._lock:
                self._mem_cache[(z, x, y)] = img
                self._pending.discard((z, x, y))
        except Exception:
            with self._lock:
                self._pending.discard((z, x, y))

    def get_pil_image(self, z, x, y):
        """PIL Image döndür (hazir değilse None + background fetch)."""
        key = (z, x, y)
        with self._lock:
            if key in self._mem_cache:
                return self._mem_cache[key]
        disk = self._disk_path(z, x, y)
        if os.path.exists(disk):
            try:
                img = Image.open(disk).convert("RGB")
                with self._lock:
                    self._mem_cache[key] = img
                return img
            except Exception:
                pass
        with self._lock:
            if key not in self._pending:
                self._pending.add(key)
                threading.Thread(
                    target=self._fetch_thread,
                    args=(z, x, y), daemon=True).start()
        return None

    def set_provider(self, provider):
        self.provider  = provider
        self.url_tmpl  = TILE_PROVIDERS[provider]
        self._tk_cache.clear()



# ═══════════════════════════════════════════════════════════════
# AgentPanel — Sol panel (Compact list with left border stripe)
# ═══════════════════════════════════════════════════════════════
class AgentPanel(tk.Frame):
    def __init__(self, parent, on_select=None, **kwargs):
        super().__init__(parent, bg=BG2, **kwargs)
        self.on_select      = on_select
        self.selected_agent = None
        self._agent_frames  = {}
        self._cov_bars      = {}
        self._state_labels  = {}
        self._cov_labels    = {}
        self._cov_canvases  = {}
        self._build()

    def _build(self):
        hdr = tk.Frame(self, bg=BG2)
        hdr.pack(fill="x", padx=6, pady=(8, 4))
        tk.Label(hdr, text="◈ VEHICLES", font=FONT_TITLE, fg=ACCENT, bg=BG2).pack(side="left")
        tk.Frame(self, bg=GRAY, height=1).pack(fill="x", padx=6, pady=2)

        for name in ALL_AGENTS:
            card = self._build_agent_card(name)
            card.pack(fill="x", padx=6, pady=3)
            self._agent_frames[name] = card

        tk.Frame(self, bg=GRAY, height=1).pack(fill="x", padx=6, pady=4)
        self._build_view_toggles()

    def _build_agent_card(self, name):
        color  = AGENT_COLORS.get(name, "#ffffff")
        card   = tk.Frame(self, bg=BG3, relief="flat", bd=0)

        # Sol kenar şeridi (Left border stripe)
        stripe = tk.Frame(card, bg=color, width=4)
        stripe.pack(side="left", fill="y")

        # Tıklanabilir
        for widget_ref in [card, stripe]:
            widget_ref.bind("<Button-1>", lambda e, n=name: self._select(n))

        # İçerik alanı
        content = tk.Frame(card, bg=BG3)
        content.pack(side="left", fill="both", expand=True, padx=(6, 4), pady=3)
        content.bind("<Button-1>", lambda e, n=name: self._select(n))

        # Üst satır: isim + durum
        top = tk.Frame(content, bg=BG3)
        top.pack(fill="x", padx=2, pady=1)
        top.bind("<Button-1>", lambda e, n=name: self._select(n))

        lbl_name = tk.Label(top, text=name.upper(), font=FONT_MONO, fg=TEXT, bg=BG3)
        lbl_name.pack(side="left")
        lbl_name.bind("<Button-1>", lambda e, n=name: self._select(n))

        state_lbl = tk.Label(top, text="CONNECTED", font=FONT_TINY, fg=GREEN, bg=BG3)
        state_lbl.pack(side="right")
        state_lbl.bind("<Button-1>", lambda e, n=name: self._select(n))
        self._state_labels[name] = state_lbl

        # Alt satır: coverage bar
        bot = tk.Frame(content, bg=BG3)
        bot.pack(fill="x", padx=2, pady=1)
        bot.bind("<Button-1>", lambda e, n=name: self._select(n))

        tk.Label(bot, text="COV:", font=FONT_TINY, fg=TEXT_DIM, bg=BG3).pack(side="left")

        bar_canvas = tk.Canvas(bot, bg=BG2, height=6, highlightthickness=0, bd=0)
        bar_canvas.pack(side="left", fill="x", expand=True, padx=(4, 4))
        bar_canvas.bind("<Button-1>", lambda e, n=name: self._select(n))
        self._cov_canvases[name] = bar_canvas

        cov_lbl = tk.Label(bot, text="0.0%", font=FONT_TINY, fg=color, bg=BG3, width=6)
        cov_lbl.pack(side="right")
        self._cov_labels[name] = cov_lbl

        return card

    def _build_view_toggles(self):
        hdr2 = tk.Frame(self, bg=BG2)
        hdr2.pack(fill="x", padx=6, pady=(4, 2))
        tk.Label(hdr2, text="◈ VIEW", font=FONT_TITLE, fg=ACCENT, bg=BG2).pack(side="left")

        toggle_frame = tk.Frame(self, bg=BG2)
        toggle_frame.pack(fill="x", padx=6)

        self._toggle_vars = {}
        toggles = [
            ("Grid",     "grid"),
            ("Trails",   "trails"),
            ("Coverage", "coverage"),
            ("Links",    "links"),
            ("Labels",   "labels"),
        ]
        for label, key in toggles:
            var = tk.BooleanVar(value=True)
            self._toggle_vars[key] = var
            cb = tk.Checkbutton(
                toggle_frame, text=label,
                variable=var,
                font=FONT_TINY, fg=TEXT_DIM, bg=BG2,
                selectcolor=BG3, activebackground=BG2,
                highlightthickness=0, bd=0)
            cb.pack(anchor="w", pady=1)

    def _select(self, name):
        self.selected_agent = name
        for n, card in self._agent_frames.items():
            card.config(bg=BG if n == name else BG3)
            for child in card.winfo_children():
                self._update_bg(child, BG if n == name else BG3)
        if self.on_select:
            self.on_select(name)

    def _update_bg(self, widget, color):
        try:
            widget.config(bg=color)
        except Exception:
            pass
        for child in widget.winfo_children():
            self._update_bg(child, color)

    def update_agents(self, comm_states, sensor_data, fusion):
        for name in AGENT_NAMES:
            st  = comm_states.get(name, "CONNECTED")
            col = STATE_COLORS.get(st, TEXT_DIM)
            self._state_labels[name].config(text=f"{st[:6]}", fg=col)

            try:
                cov  = fusion.get_coverage_pct(name)
            except Exception:
                cov  = 0.0

            self._cov_labels[name].config(text=f"{cov:5.1f}%")

            bar = self._cov_canvases[name]
            bar.update_idletasks()
            bw  = bar.winfo_width()
            bh  = bar.winfo_height()
            bar.delete("all")
            if bw > 4:
                bar.create_rectangle(0, 0, bw, bh, fill=BG3, outline="")
                fill_w = int(bw * cov / 100.0)
                if fill_w > 0:
                    bar.create_rectangle(0, 0, fill_w, bh, fill=AGENT_COLORS.get(name, ACCENT), outline="")

    def get_toggle(self, key):
        return self._toggle_vars.get(key, tk.BooleanVar(value=True)).get()


# ═══════════════════════════════════════════════════════════════
# TelemetryPanel — Sağ üst (Key-Value Grid)
# ═══════════════════════════════════════════════════════════════
class TelemetryPanel(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG2, **kwargs)
        self.selected_agent = None
        self._build()

    def _build(self):
        hdr = tk.Frame(self, bg=BG2)
        hdr.pack(fill="x", padx=6, pady=(8, 2))
        tk.Label(hdr, text="◈ TELEMETRY", font=FONT_TITLE, fg=ACCENT, bg=BG2).pack(side="left")
        tk.Frame(self, bg=GRAY, height=1).pack(fill="x", padx=6, pady=2)

        self._sel_frame = tk.Frame(self, bg=BG3)
        self._sel_frame.pack(fill="x", padx=6, pady=4)

        self._sel_name_lbl = tk.Label(
            self._sel_frame, text="No vehicle selected",
            font=FONT_MONO, fg=TEXT_DIM, bg=BG3)
        self._sel_name_lbl.pack(anchor="w", padx=6, pady=(4, 4))

        self._grid_frame = tk.Frame(self._sel_frame, bg=BG3)
        self._grid_frame.pack(fill="x", padx=6, pady=(2, 6))

        self._grid_labels = {}
        fields = [
            ("X", "X:"),
            ("Y", "Y:"),
            ("Depth", "Depth:"),
            ("Heading", "HDG:"),
            ("Speed", "Speed:"),
            ("RPM", "RPM:"),
            ("WP X", "WP X:"),
            ("WP Y", "WP Y:"),
            ("Status", "STATUS:"),
            ("Coverage", "COVERAGE:")
        ]
        for i, (key, label_text) in enumerate(fields):
            row = i // 2
            col = (i % 2) * 2
            klbl = tk.Label(self._grid_frame, text=label_text, font=FONT_SMALL, fg=TEXT_DIM, bg=BG3, anchor="w")
            klbl.grid(row=row, column=col, sticky="w", padx=(2, 4), pady=1)
            vlbl = tk.Label(self._grid_frame, text="N/A", font=FONT_SMALL, fg=TEXT, bg=BG3, anchor="w")
            vlbl.grid(row=row, column=col+1, sticky="w", padx=(0, 6), pady=1)
            self._grid_labels[key] = vlbl

        self._grid_frame.columnconfigure(0, weight=1)
        self._grid_frame.columnconfigure(1, weight=1)
        self._grid_frame.columnconfigure(2, weight=1)
        self._grid_frame.columnconfigure(3, weight=1)

        tk.Frame(self, bg=GRAY, height=1).pack(fill="x", padx=6, pady=2)

        tbl_hdr = tk.Frame(self, bg=BG2)
        tbl_hdr.pack(fill="x", padx=6, pady=(2, 0))
        tk.Label(tbl_hdr, text="ALL VEHICLES", font=FONT_SMALL, fg=TEXT_DIM, bg=BG2).pack(side="left")

        self._all_text = tk.Text(
            self, font=FONT_TINY, bg=BG2, fg=TEXT,
            height=9, width=32, relief="flat", bd=0,
            state="disabled", wrap="none")
        self._all_text.pack(fill="x", padx=6, pady=2)

    def _set_text(self, widget, text):
        widget.config(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.config(state="disabled")

    def update_telemetry(self, sensor_data, comm_states, fusion, elapsed):
        name = self.selected_agent
        if name and name in sensor_data:
            sd  = sensor_data[name]
            x   = sd.get("x", 0);     y   = sd.get("y", 0)
            z   = sd.get("z", 0);     hdg = sd.get("heading", 0)
            vel = sd.get("vel", 0);   rpm = sd.get("rpm", 0)
            tx  = sd.get("target_x", 0)
            ty  = sd.get("target_y", 0)
            st  = sd.get("status", "OK")
            cov = fusion.get_coverage_pct(name)
            comm_st = comm_states.get(name, "CONNECTED")

            color  = AGENT_COLORS.get(name, "#ffffff")
            self._sel_name_lbl.config(text=f"● {name.upper()} — {comm_st}", fg=color)

            self._grid_labels["X"].config(text=f"{x:.1f} m")
            self._grid_labels["Y"].config(text=f"{y:.1f} m")
            self._grid_labels["Depth"].config(text=f"{abs(z):.1f} m")
            self._grid_labels["Heading"].config(text=f"{hdg:.1f}°")
            self._grid_labels["Speed"].config(text=f"{vel:.2f} m/s")
            self._grid_labels["RPM"].config(text=f"{rpm:.0f}")
            self._grid_labels["WP X"].config(text=f"{tx:.1f} m")
            self._grid_labels["WP Y"].config(text=f"{ty:.1f} m")
            self._grid_labels["Status"].config(text=st, fg=GREEN if st == "OK" else YELLOW)
            self._grid_labels["Coverage"].config(text=f"{cov:.1f}%")
        else:
            self._sel_name_lbl.config(text="No vehicle selected", fg=TEXT_DIM)
            for vlbl in self._grid_labels.values():
                vlbl.config(text="N/A", fg=TEXT)

        lines = [
            f"{'AUV':<5} {'X':>7} {'Y':>7} {'D':>5} {'H':>5} {'V':>4} {'ST':<5}",
            "─" * 42,
        ]
        for n in ALL_AGENTS:
            sd  = sensor_data.get(n, {})
            x   = sd.get("x", 0);    y   = sd.get("y", 0)
            z   = sd.get("z", 0);    hdg = sd.get("heading", 0)
            vel = sd.get("vel", 0)
            st  = sd.get("status", "OK")
            sym = "◉" if st == "OK" else "⚠"
            lines.append(
                f"{sym}{n:<4} {x:>7.1f} {y:>7.1f} {abs(z):>5.1f} "
                f"{hdg:>5.1f} {vel:>4.2f} {st[:5]:<5}")

        lines += [
            "─" * 42,
            f"AVG COV : {fusion.get_avg_coverage_pct():>6.2f}%",
            f"RMS     : {fusion.get_consensus_rms():>8.5f}",
        ]
        self._set_text(self._all_text, "\n".join(lines))


# ═══════════════════════════════════════════════════════════════
# AppCastPanel — Sağ alt (AppCast/SNR/Log Viewer)
# ═══════════════════════════════════════════════════════════════
class AppCastPanel(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG2, **kwargs)
        self._build()

    def _build(self):
        hdr = tk.Frame(self, bg=BG2)
        hdr.pack(fill="x", padx=6, pady=(4, 2))
        tk.Label(hdr, text="◈ APPCAST / COMM", font=FONT_TITLE, fg=ACCENT, bg=BG2).pack(side="left")

        tab_frame = tk.Frame(self, bg=BG2)
        tab_frame.pack(fill="x", padx=6, pady=2)

        self._active_tab = tk.StringVar(value="appcast")
        for tab_name, tab_key in [("Cast", "appcast"),
                                   ("SNR", "snr"),
                                   ("Log", "log")]:
            btn = tk.Radiobutton(
                tab_frame, text=tab_name,
                variable=self._active_tab, value=tab_key,
                font=FONT_TINY, fg=TEXT_DIM, bg=BG2,
                selectcolor=BG3, activebackground=BG2,
                highlightthickness=0, bd=0,
                indicatoron=False,
                padx=6, pady=2,
                command=self._switch_tab)
            btn.pack(side="left", padx=2)

        tk.Frame(self, bg=GRAY, height=1).pack(fill="x", padx=6, pady=2)

        self._content_frame = tk.Frame(self, bg=BG2)
        self._content_frame.pack(fill="both", expand=True, padx=4, pady=2)

        self._appcast_text = tk.Text(
            self._content_frame,
            font=FONT_TINY, bg="#05090f", fg=GREEN,
            relief="flat", bd=0, state="disabled", wrap="none")
        self._appcast_text.pack(fill="both", expand=True)

        self._snr_text = tk.Text(
            self._content_frame,
            font=FONT_TINY, bg="#05090f", fg=ACCENT,
            relief="flat", bd=0, state="disabled", wrap="none")

        self._log_frame = tk.Frame(self._content_frame, bg="#05090f")
        log_scroll = tk.Scrollbar(self._log_frame, bg=BG2)
        log_scroll.pack(side="right", fill="y")
        self._log_text = tk.Text(
            self._log_frame,
            font=FONT_TINY, bg="#05090f", fg=YELLOW,
            relief="flat", bd=0, wrap="word",
            yscrollcommand=log_scroll.set)
        self._log_text.pack(fill="both", expand=True)
        log_scroll.config(command=self._log_text.yview)

        self._switch_tab()

    def _switch_tab(self):
        key = self._active_tab.get()
        self._appcast_text.pack_forget()
        self._snr_text.pack_forget()
        self._log_frame.pack_forget()

        if key == "appcast":
            self._appcast_text.pack(fill="both", expand=True)
        elif key == "snr":
            self._snr_text.pack(fill="both", expand=True)
        elif key == "log":
            self._log_frame.pack(fill="both", expand=True)

    def _set_text(self, widget, text):
        widget.config(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.config(state="disabled")

    def update_appcast(self, comm_quality, comm_log, selected_agent,
                       sensor_data, comm_mgr=None):
        try:
            summary = comm_quality.get_summary()
        except Exception:
            summary = {"total_sent": 0, "total_received": 0,
                       "global_pdr": 0.0, "per_agent": {}}

        key = self._active_tab.get()

        if key == "appcast":
            per = summary.get("per_agent", {})
            lines = [
                f"Global PDR : {summary.get('global_pdr', 0):.4f}",
                f"Total Sent : {summary.get('total_sent', 0)}",
                f"Total RX   : {summary.get('total_received', 0)}",
                f"Dropped    : {summary.get('total_dropped', 0)}",
                "",
                f"{'AGENT':<6} {'RX':>5} {'PDR':>6}  {'STATE':<12}",
                "─" * 30,
            ]
            for n in AGENT_NAMES:
                a = per.get(n, {})
                lines.append(
                    f"{n:<6} {a.get('received', 0):>5} "
                    f"{a.get('pdr', 0):>6.3f}  {a.get('state', 'N/A'):<12}")

            if selected_agent and selected_agent in sensor_data:
                sd = sensor_data[selected_agent]
                lines += [
                    "",
                    f"─── {selected_agent.upper()} ────────────",
                    f"X:{sd.get('x',0):8.2f}  Y:{sd.get('y',0):8.2f}",
                    f"Z:{abs(sd.get('z',0)):8.2f}  HDG:{sd.get('heading',0):6.1f}°",
                    f"VEL:{sd.get('vel',0):6.3f}  RPM:{sd.get('rpm',0):5.0f}",
                    f"STATUS: {sd.get('status','OK')}",
                ]
            self._set_text(self._appcast_text, "\n".join(lines))

        elif key == "snr":
            if comm_mgr:
                lines = ["SNR MATRIX (dB)", "=" * 28]
                header = f"{'':6}" + "".join(f"{r.upper():>8}" for r in AGENT_NAMES)
                lines.append(header)
                lines.append("─" * (6 + 8 * len(AGENT_NAMES)))
                for s in AGENT_NAMES:
                    row = f"{s:<6}"
                    for r in AGENT_NAMES:
                        if s == r:
                            row += f"{'---':>8}"
                        else:
                            try:
                                snr = comm_mgr.get_snr(s, r)
                                if math.isnan(snr):
                                    row += f"{'N/A':>8}"
                                else:
                                    row += f"{snr:>7.1f}d"
                            except Exception:
                                row += f"{'ERR':>8}"
                    lines.append(row)
                lines += ["", "─" * 28, "LINK QUALITY:"]
                for s in AGENT_NAMES:
                    for r in AGENT_NAMES:
                        if s == r:
                            continue
                        try:
                            snr = comm_mgr.get_snr(s, r)
                            if not math.isnan(snr):
                                q = "GOOD" if snr > 20 else "FAIR" if snr > 10 else "POOR"
                                lines.append(f"  {s}→{r}: {snr:.1f}dB [{q}]")
                        except Exception:
                            pass
            else:
                lines = ["Comm manager not available"]
            self._set_text(self._snr_text, "\n".join(lines))

        elif key == "log":
            for msg in list(comm_log)[-5:]:
                self._log_text.insert(tk.END, f"{msg}\n")
            self._log_text.see(tk.END)
            line_count = int(self._log_text.index("end-1c").split(".")[0])
            if line_count > 300:
                self._log_text.delete("1.0", "100.0")

    def console_log(self, message):
        try:
            self._log_text.config(state="normal")
            self._log_text.insert(tk.END, f"{message}\n")
            self._log_text.see(tk.END)
            self._log_text.config(state="disabled")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# MapCanvas — 2D Harita Canvas (Butonsuz & IBM Plex Mono)
# ═══════════════════════════════════════════════════════════════
class MapCanvas(tk.Canvas):
    def __init__(self, parent, consensus_fusion, comm_manager=None,
                 on_waypoint_set=None, on_agent_select=None,
                 on_deploy=None, on_return=None, on_pause=None,
                 **kwargs):
        super().__init__(parent, bg=BG_MAP, highlightthickness=0, **kwargs)

        self.fusion       = consensus_fusion
        self.comm_mgr     = comm_manager
        self.on_waypoint  = on_waypoint_set
        self.on_select    = on_agent_select
        self.on_deploy    = on_deploy
        self.on_return    = on_return
        self.on_pause     = on_pause

        # ── Görünüm ──────────────────────────────────────────────
        self.scale    = 2.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self._pan_start = None

        # ── Veri tamponları ───────────────────────────────────────
        self.trajectories    = {n: deque(maxlen=TRAJECTORY_LEN) for n in AGENT_NAMES}
        self.positions       = {}
        self.headings        = {}
        self.speeds          = {}
        self.targets         = {}
        self.waypoint_hist   = {n: deque(maxlen=5) for n in AGENT_NAMES}
        self.sensor_data     = {n: {} for n in AGENT_NAMES}
        self.comm_states     = {n: "CONNECTED" for n in AGENT_NAMES}
        self.selected_agent  = None
        self._link_anim_phase = 0
        self._active_links    = {}

        self._last_seen = {n: time.time() for n in AGENT_NAMES}
        self._pulse_list = []
        self._drop_points = []

        self._bg_photo     = None
        self._bg_canvas_id = None
        self._bg_w = 0
        self._bg_h = 0
        self._bg_loaded = False
        self._sea_texture_photo = None

        # ── Tile sistemi ─────────────────────────────────────────────
        self._tile_mgr          = TileManager(provider="esri_satellite")
        self._tile_zoom         = 17
        self._tile_enabled      = True
        self._tile_opacity      = 0.82
        self._tile_provider_idx = 0
        self._tile_provider_keys = list(TILE_PROVIDERS.keys())
        self._coverage_photo    = None

        self.show_grid        = True
        self.show_trails      = True
        self.show_links       = True
        self.show_coverage    = True
        self.show_waypoints   = True
        self.show_labels      = True
        self.show_sonar_fp    = True
        self.show_oparea      = True
        self.show_pulses      = True
        self.trail_length     = 200
        self._paused          = False

        self.bind("<Configure>",        self._on_resize)
        self.bind("<MouseWheel>",       self._on_zoom)
        self.bind("<Button-4>",         self._on_zoom_up)
        self.bind("<Button-5>",         self._on_zoom_down)
        self.bind("<ButtonPress-3>",    self._on_pan_start)
        self.bind("<B3-Motion>",        self._on_pan_drag)
        self.bind("<ButtonRelease-3>",  self._on_pan_end)
        self.bind("<ButtonPress-1>",    self._on_left_click)
        self.bind("<Double-Button-1>",  self._on_double_click)
        self.bind("<r>",                lambda e: self._clear_drops())
        self.bind("<t>",                self._on_tile_toggle)
        self.bind("<b>",                self._on_tile_provider_cycle)
        self.bind("<minus>",            self._on_tile_opacity_down)
        self.bind("<equal>",            self._on_tile_opacity_up)
        self.focus_set()

        self.after(80, self._init_center)
        self.after(200, self._load_background)

    def _init_center(self):
        w = self.winfo_width()
        h = self.winfo_height()
        if w > 1 and h > 1:
            self.offset_x = w / 2
            self.offset_y = h / 2
            self._redraw_all()

    def _on_resize(self, event):
        self.offset_x = event.width  / 2
        self.offset_y = event.height / 2
        self._bg_loaded = False
        if hasattr(self, '_tile_mgr'):
            self._tile_mgr._tk_cache.clear()
        self._redraw_all()

    def _load_background(self):
        # Tile sistemi aktifse JPG/gradient yüklemeye gerek yok
        if self._tile_enabled and PIL_AVAILABLE:
            self._bg_loaded = True
            self._redraw_all()
            return

        w = self.winfo_width()
        h = self.winfo_height()
        if w < 2 or h < 2:
            self.after(300, self._load_background)
            return

        self._bg_w = w
        self._bg_h = h

        if PIL_AVAILABLE:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            img_path = os.path.join(base, MAP_BACKGROUND_IMAGE)
            if os.path.isfile(img_path):
                try:
                    self._load_image_file(img_path, w, h)
                    self._bg_loaded = True
                    self._redraw_all()
                    return
                except Exception as e:
                    print(f"[MapCanvas] Arka plan yüklenemedi: {e}")

        self._generate_sea_texture(w, h)
        self._bg_loaded = True
        self._redraw_all()

    def _load_image_file(self, path, w, h):
        img = Image.open(path).convert("RGBA")
        img = img.resize((w, h), Image.LANCZOS)
        enhancer = ImageEnhance.Brightness(img.convert("RGB"))
        img_dark  = enhancer.enhance(MAP_BACKGROUND_BRIGHTNESS).convert("RGBA")
        alpha_val = int(255 * MAP_BACKGROUND_TINT_ALPHA)
        tint = Image.new("RGBA", (w, h), (0, 20, 40, alpha_val))
        blended = Image.alpha_composite(img_dark, tint)
        self._bg_photo = ImageTk.PhotoImage(blended)

    def _generate_sea_texture(self, w, h):
        rng = np.random.default_rng(42)
        grad_r = np.linspace(10,  13, h).astype(np.uint8)
        grad_g = np.linspace(21,  31, h).astype(np.uint8)
        grad_b = np.linspace(32,  53, h).astype(np.uint8)

        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 0] = grad_r[:, None]
        img[:, :, 1] = grad_g[:, None]
        img[:, :, 2] = grad_b[:, None]

        noise = rng.normal(0, 1.5, (h, w)).astype(np.int16)
        img_int = img.astype(np.int16) + noise[:, :, None]
        img = np.clip(img_int, 0, 255).astype(np.uint8)

        ppm_data = f"P6\n{w} {h}\n255\n".encode()
        ppm_data += img.tobytes()
        self._bg_photo = tk.PhotoImage(data=ppm_data)

    def world_to_canvas(self, east_m, north_m):
        cx = self.offset_x + east_m  * self.scale
        cy = self.offset_y - north_m * self.scale
        return cx, cy

    def canvas_to_world(self, cx, cy):
        east_m  = (cx - self.offset_x) / self.scale
        north_m = (self.offset_y - cy) / self.scale
        return east_m, north_m

    def _on_zoom(self, event):
        factor = 1.12 if event.delta > 0 else 0.88
        self._zoom_at(event.x, event.y, factor)

    def _on_zoom_up(self, event):
        self._zoom_at(event.x, event.y, 1.12)

    def _on_zoom_down(self, event):
        self._zoom_at(event.x, event.y, 0.88)

    def _zoom_at(self, cx, cy, factor):
        wx, wy = self.canvas_to_world(cx, cy)
        self.scale = max(0.3, min(20.0, self.scale * factor))
        new_cx = self.offset_x + wx * self.scale
        new_cy = self.offset_y - wy * self.scale
        self.offset_x += cx - new_cx
        self.offset_y += cy - new_cy
        self._bg_loaded = False
        if hasattr(self, '_tile_mgr'):
            self._tile_mgr._tk_cache.clear()
        self._redraw_all()

    def _on_pan_start(self, event):
        self._pan_start = (event.x, event.y)

    def _on_pan_drag(self, event):
        if self._pan_start:
            dx = event.x - self._pan_start[0]
            dy = event.y - self._pan_start[1]
            self.offset_x += dx
            self.offset_y += dy
            self._pan_start = (event.x, event.y)
            self._redraw_all()

    def _on_pan_end(self, event):
        self._pan_start = None

    def _on_left_click(self, event):
        if event.state & 0x0001:
            east_m, north_m = self.canvas_to_world(event.x, event.y)
            self._drop_points.append((east_m, north_m))
            self._redraw_all()
            return

        clicked = self._find_agent_at(event.x, event.y)
        if clicked:
            self.selected_agent = clicked
            if self.on_select:
                self.on_select(clicked)
            self._redraw_all()
            return

        if self.selected_agent and self.on_waypoint:
            east_m, north_m = self.canvas_to_world(event.x, event.y)
            self.on_waypoint(self.selected_agent, east_m, north_m)
            self.waypoint_hist[self.selected_agent].append((east_m, north_m))

    def _on_double_click(self, event):
        self.selected_agent = None
        if self.on_select:
            self.on_select(None)
        self._redraw_all()

    def _clear_drops(self):
        self._drop_points.clear()
        self._redraw_all()

    def _find_agent_at(self, cx, cy, radius=18):
        for name, pos in self.positions.items():
            px, py = self.world_to_canvas(pos[1], pos[0])
            if abs(cx - px) < radius and abs(cy - py) < radius:
                return name
        return None

    def trigger_comms_pulse(self, sender, receiver, snr=15.0):
        if sender not in self.positions:
            return
        pos = self.positions[sender]
        now = time.time()

        if snr > 20:   color = "#36c986"
        elif snr > 10: color = "#cfb33c"
        else:           color = "#c94b62"

        recv_pos = self.positions.get(receiver)
        self._pulse_list.append({
            "kind":     "comms",
            "north":    pos[0],
            "east":     pos[1],
            "color":    color,
            "start":    now,
            "duration": 3.0,
            "max_r":    60,
            "recv_pos": recv_pos,
        })
        snr_range_m = max(20, snr * 4)
        self._pulse_list.append({
            "kind":     "range",
            "north":    pos[0],
            "east":     pos[1],
            "color":    AGENT_COLORS.get(sender, "#ffffff"),
            "start":    now,
            "duration": 5.0,
            "max_r_m":  snr_range_m,
        })

    def update_display(self, states, targets, sensor_data,
                       comm_states=None, active_links=None):
        now = time.time()
        for name in AGENT_NAMES:
            if name in states and "PoseSensor" in states[name]:
                pose = states[name]["PoseSensor"]
                pos  = pose[:3, 3]
                rot  = pose[:3, :3]
                yaw  = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
                self.positions[name] = pos.copy()
                self.headings[name]  = yaw
                self.trajectories[name].append((pos[0], pos[1]))
                self._last_seen[name] = now

            if sensor_data and name in sensor_data:
                sd = sensor_data[name]
                self.sensor_data[name] = sd
                self.speeds[name] = sd.get("vel", 0.0)

            if targets and name in targets and targets[name] is not None:
                self.targets[name] = targets[name]
            else:
                self.targets.setdefault(name, None)

        if comm_states:
            self.comm_states = comm_states
        if active_links:
            self._active_links = active_links

        for name in AGENT_NAMES:
            age = now - self._last_seen.get(name, now)
            if age >= STALE_DEAD_S:
                self.comm_states[name] = "OFFLINE"

        self._link_anim_phase = (self._link_anim_phase + 1) % 20
        self._pulse_list = [
            p for p in self._pulse_list
            if (now - p["start"]) < p["duration"]
        ]
        self._redraw_all()

    def _redraw_all(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 2 or h < 2:
            return

        self._draw_background(w, h)
        if self.show_oparea:
            self._draw_oparea()
        if self.show_grid:
            self._draw_grid(w, h)
        if self.show_coverage:
            self._draw_coverage_overlay()
        if self.show_sonar_fp:
            self._draw_sonar_footprints()
        if self.show_links:
            self._draw_comm_links()
        if self.show_pulses:
            self._draw_pulses()
        if self.show_trails:
            self._draw_trails()
        if self.show_waypoints:
            self._draw_waypoints()
        self._draw_extrapolated()
        self._draw_agents()
        self._draw_boundary_warning()
        self._draw_drop_points()
        self._draw_scale_bar(w, h)
        self._draw_info_bar(w, h)

    def _draw_background(self, w, h):
        """Tile arka plan çiz; PIL yoksa veya tile kapatılmışsa gradient fallback."""

        if not self._tile_enabled or not PIL_AVAILABLE:
            # Fallback: orijinal statik görüntü / gradient
            if self._bg_photo is not None:
                self.create_image(0, 0, image=self._bg_photo,
                                  anchor="nw", tags="bg")
            else:
                self.create_rectangle(0, 0, w, h, fill=BG_MAP, outline="")
            return

        # ── Tile zoom seviyesini mevcut scale'e göre hesapla ──
        visible_m = w / max(self.scale, 0.01)
        lat_cos   = math.cos(math.radians(MAP_DATUM_LAT))
        best_zoom = 17
        for z in range(18, 10, -1):
            tile_m = (40075016.686 * lat_cos) / (2 ** z)
            if 2.5 <= (visible_m / tile_m) <= 10.0:
                best_zoom = z
                break
        self._tile_zoom = max(13, min(18, best_zoom))
        z = self._tile_zoom

        # ── Canvas köşelerini lat/lon'a çevir ──
        east_tl,  north_tl  = self.canvas_to_world(0, 0)
        east_br,  north_br  = self.canvas_to_world(w, h)
        lat_tl, lon_tl = _local_to_latlon(east_tl,  north_tl)
        lat_br, lon_br = _local_to_latlon(east_br,  north_br)

        lat_min, lat_max = min(lat_tl, lat_br), max(lat_tl, lat_br)
        lon_min, lon_max = min(lon_tl, lon_br), max(lon_tl, lon_br)

        tx1, ty1 = _deg2tile(lat_max, lon_min, z)
        tx2, ty2 = _deg2tile(lat_min, lon_max, z)
        tx_min, tx_max = min(tx1, tx2), max(tx1, tx2)
        ty_min, ty_max = min(ty1, ty2), max(ty1, ty2)

        if (tx_max - tx_min + 1) * (ty_max - ty_min + 1) > 100:
            self.create_rectangle(0, 0, w, h, fill=BG_MAP, outline="")
            return

        # ── Her tile'i çiz ──
        any_drawn = False
        for ty in range(ty_min, ty_max + 1):
            for tx in range(tx_min, tx_max + 1):

                lat_tl_t, lon_tl_t = _tile2deg(tx,     ty,     z)
                lat_br_t, lon_br_t = _tile2deg(tx + 1, ty + 1, z)

                east_tl_t, north_tl_t = _latlon_to_local(lat_tl_t, lon_tl_t)
                east_br_t, north_br_t = _latlon_to_local(lat_br_t, lon_br_t)

                cx_tl, cy_tl = self.world_to_canvas(east_tl_t, north_tl_t)
                cx_br, cy_br = self.world_to_canvas(east_br_t, north_br_t)

                tile_w = max(1, int(abs(cx_br - cx_tl)))
                tile_h = max(1, int(abs(cy_br - cy_tl)))

                pil_img = self._tile_mgr.get_pil_image(z, tx, ty)

                if pil_img is None:
                    x0 = int(min(cx_tl, cx_br))
                    y0 = int(min(cy_tl, cy_br))
                    self.create_rectangle(
                        x0, y0, x0 + tile_w, y0 + tile_h,
                        fill="#0d1825", outline="#0a1520", tags="bg_tile")
                    self.after(400, self._redraw_all)
                    continue

                cache_key = (z, tx, ty, tile_w, tile_h,
                             self._tile_opacity,
                             self._tile_mgr.provider)

                if cache_key not in self._tile_mgr._tk_cache:
                    try:
                        scaled   = pil_img.resize(
                            (tile_w, tile_h), Image.LANCZOS)
                        enhancer = ImageEnhance.Brightness(scaled)
                        darkened = enhancer.enhance(self._tile_opacity)
                        tint     = Image.new(
                            "RGBA", (tile_w, tile_h), (0, 15, 30, 38))
                        blended  = Image.alpha_composite(
                            darkened.convert("RGBA"), tint)
                        tk_img   = ImageTk.PhotoImage(blended.convert("RGB"))
                        self._tile_mgr._tk_cache[cache_key] = tk_img
                        if len(self._tile_mgr._tk_cache) > 200:
                            oldest = next(iter(self._tile_mgr._tk_cache))
                            del self._tile_mgr._tk_cache[oldest]
                    except Exception:
                        continue

                tk_img = self._tile_mgr._tk_cache.get(cache_key)
                if tk_img:
                    x0 = int(min(cx_tl, cx_br))
                    y0 = int(min(cy_tl, cy_br))
                    self.create_image(x0, y0, image=tk_img,
                                      anchor="nw", tags="bg_tile")
                    any_drawn = True

        if not any_drawn:
            self.create_rectangle(0, 0, w, h, fill=BG_MAP, outline="")

    def _draw_oparea(self):
        half = MAP_SIZE_M / 2.0
        corners_world = [
            (-half, -half), (half, -half),
            (half,  half),  (-half,  half),
        ]
        canvas_pts = [
            self.world_to_canvas(e, n)
            for e, n in corners_world
        ]
        for i in range(4):
            x0, y0 = canvas_pts[i]
            x1, y1 = canvas_pts[(i + 1) % 4]
            self.create_line(x0, y0, x1, y1, fill=OPAREA_COL, width=1.5, dash=(8, 5), tags="oparea")

        lx, ly = canvas_pts[3]
        self.create_text(lx + 6, ly + 12, text="OPAREA", fill=OPAREA_COL, font=("IBM Plex Mono", 8, "bold"), anchor="w", tags="oparea")
        self.create_text(lx + 6, ly + 24, text=f"{MAP_SIZE_M:.0f}×{MAP_SIZE_M:.0f}m", fill=OPAREA_COL, font=("IBM Plex Mono", 7), anchor="w", tags="oparea")

    def _draw_grid(self, w, h):
        west, north = self.canvas_to_world(0, 0)
        east, south = self.canvas_to_world(w, h)

        raw_step = 60.0 / self.scale
        minor = self._nice_step(raw_step)
        major = minor * 2

        def draw_line(x0, y0, x1, y1, is_major):
            color = GRID_MAJOR if is_major else GRID_MINOR
            width = 0.8 if is_major else 0.4
            self.create_line(x0, y0, x1, y1,
                             fill=color, width=width, tags="grid")

        xe = math.floor(west / minor) * minor
        while xe <= east:
            cx, _ = self.world_to_canvas(xe, 0)
            is_maj = abs(xe % major) < 0.1
            draw_line(cx, 0, cx, h, is_maj)
            if is_maj and minor * self.scale > 25:
                self.create_text(cx, h - 10, text=f"{xe:.0f}", fill=GRID_LABEL, font=("IBM Plex Mono", 7), tags="grid_lbl")
            xe += minor

        yn = math.floor(south / minor) * minor
        while yn <= north:
            _, cy = self.world_to_canvas(0, yn)
            is_maj = abs(yn % major) < 0.1
            draw_line(0, cy, w, cy, is_maj)
            if is_maj and minor * self.scale > 25:
                self.create_text(10, cy, text=f"{yn:.0f}", fill=GRID_LABEL, font=("IBM Plex Mono", 7), anchor="w", tags="grid_lbl")
            yn += minor

        cx0, cy0 = self.world_to_canvas(0, 0)
        self.create_line(cx0 - 6, cy0, cx0 + 6, cy0, fill="#3a6a8a", width=2, tags="grid")
        self.create_line(cx0, cy0 - 6, cx0, cy0 + 6, fill="#3a6a8a", width=2, tags="grid")

    def _nice_step(self, raw):
        for s in [5, 10, 20, 25, 50, 100, 200, 500, 1000]:
            if raw <= s:
                return s
        return 2000

    # ─── Tile klavye handler'ları ───────────────────────────────────────────────
    def _on_tile_toggle(self, event=None):
        """T: uydu tile arka planı aç/kapat."""
        self._tile_enabled = not self._tile_enabled
        self._tile_mgr._tk_cache.clear()
        self._bg_loaded = False
        if not self._tile_enabled:
            self.after(50, self._load_background)
        self._redraw_all()

    def _on_tile_provider_cycle(self, event=None):
        """B: provider döngüle (esri → carto_dark → osm)."""
        keys = self._tile_provider_keys
        self._tile_provider_idx = (self._tile_provider_idx + 1) % len(keys)
        new_p = keys[self._tile_provider_idx]
        self._tile_mgr.set_provider(new_p)
        print(f"[TileMap] Provider: {new_p}")
        self._redraw_all()

    def _on_tile_opacity_down(self, event=None):
        """-: tile daha koyu."""
        self._tile_opacity = max(0.2, self._tile_opacity - 0.08)
        self._tile_mgr._tk_cache.clear()
        self._redraw_all()

    def _on_tile_opacity_up(self, event=None):
        """+: tile daha parlak."""
        self._tile_opacity = min(1.0, self._tile_opacity + 0.08)
        self._tile_mgr._tk_cache.clear()
        self._redraw_all()

    def _draw_coverage_overlay(self):
        if not self.show_coverage:
            return
        if not PIL_AVAILABLE:
            self._draw_coverage_overlay_fallback()
            return

        try:
            leader_map = self.fusion.maps[AGENT_NAMES[0]]
            grid = leader_map.grid
            n    = grid.shape[0]
            res  = MAP_SIZE_M / n
            half = MAP_SIZE_M / 2.0

            w = self.winfo_width()
            h = self.winfo_height()
            if w < 2 or h < 2:
                return

            # numpy RGBA array — tüm sıfır (şeffaf)
            overlay_arr = np.zeros((h, w, 4), dtype=np.uint8)

            stride = max(1, int(1.5 / self.scale) + 1)

            for r in range(0, n, stride):
                for c in range(0, n, stride):
                    p = float(grid[r, c])

                    if abs(p - 0.5) < 0.05:
                        continue

                    north_m = -half + (r + 0.5) * res
                    east_m  = -half + (c + 0.5) * res
                    size_px = res * self.scale * stride

                    if size_px < 1.5:
                        continue

                    cx0, cy0 = self.world_to_canvas(
                        east_m - res * stride / 2,
                        north_m + res * stride / 2)

                    ix0 = max(0, int(cx0))
                    iy0 = max(0, int(cy0))
                    ix1 = min(w, int(cx0 + size_px))
                    iy1 = min(h, int(cy0 + size_px))

                    if ix1 <= ix0 or iy1 <= iy0:
                        continue

                    r_col, g_col, b_col, alpha = self._coverage_color_rgba(p)
                    
                    # Her hücrenin 1px iç kenarını biraz daha koyu yap
                    edge_alpha = min(255, alpha + 30)
                    overlay_arr[iy0:iy1, ix0:ix1] = (r_col//2, g_col//2, b_col//2, edge_alpha)
                    
                    if ix1-ix0 > 3 and iy1-iy0 > 3:
                        overlay_arr[iy0+1:iy1-1, ix0+1:ix1-1] = (r_col, g_col, b_col, alpha)

            overlay_img = Image.fromarray(overlay_arr, mode="RGBA")
            self._coverage_photo = ImageTk.PhotoImage(overlay_img)
            self.create_image(0, 0, image=self._coverage_photo,
                              anchor="nw", tags="coverage_overlay")

        except Exception:
            self._draw_coverage_overlay_fallback()

    def _coverage_color_rgba(self, p):
        """Olasılık değerini RGBA tuple'a çevir (belirgin sonar yeşil-cyan, %43-63 opacity)."""
        if p > 0.5:
            t = (p - 0.5) * 2.0    # 0→1 arası
            r = int(0)
            g = int(200 + t * 29)  # 200→229
            b = int(180 + t * 24)  # 180→204
            alpha = int(110 + t * 50)  # 110→160 (%43→%63)
        else:
            t = (0.5 - p) * 2.0
            r = int(100 * t)
            g = int(15 * t)
            b = int(15 * t)
            alpha = int(50 * t)
        return (r, g, b, alpha)

    def _draw_coverage_overlay_fallback(self):
        """PIL yoksa eski rectangle yöntemi — stipple ile yarı şeffaf."""
        try:
            leader_map = self.fusion.maps[AGENT_NAMES[0]]
            grid = leader_map.grid
            n    = grid.shape[0]
            res  = MAP_SIZE_M / n
            half = MAP_SIZE_M / 2.0

            stride = max(1, int(1.2 / self.scale) + 1)
            for r in range(0, n, stride):
                for c in range(0, n, stride):
                    p = float(grid[r, c])
                    if abs(p - 0.5) < 0.05:
                        continue

                    north_m = -half + (r + 0.5) * res
                    east_m  = -half + (c + 0.5) * res
                    size_px = res * self.scale * stride
                    if size_px < 1.5:
                        continue

                    cx0, cy0 = self.world_to_canvas(
                        east_m - res * stride / 2,
                        north_m + res * stride / 2)
                    cx1 = cx0 + size_px
                    cy1 = cy0 + size_px

                    if p > 0.5:
                        color = "#00b4b4"
                        stipple = "gray25"
                    else:
                        color = "#882222"
                        stipple = "gray12"

                    self.create_rectangle(cx0, cy0, cx1, cy1,
                                          fill=color, outline="",
                                          stipple=stipple,
                                          tags="coverage")
        except Exception:
            pass

    def _coverage_color(self, p):
        # Eski kodların kırılmaması için fallback fonksiyonu tutuyoruz
        if p > 0.5:
            t = (p - 0.5) * 2.0
            return f"#{0:02x}{int(80*(1-t)+255*t):02x}{int(80*(1-t)+80*t):02x}"
        else:
            t = (0.5 - p) * 2.0
            return f"#{int(30*t):02x}{int(50*(1-t)+10*t):02x}{int(80*(1-t)+100*t):02x}"

    def _draw_sonar_footprints(self):
        for name in AGENT_NAMES:
            if name not in self.positions:
                continue
            pos   = self.positions[name]
            hdg   = self.headings.get(name, 0.0)
            color = AGENT_COLORS.get(name, AGENT_COLOR_DEFAULT)
            cx_v, cy_v = self.world_to_canvas(pos[1], pos[0])
            angle_rad = math.radians(-hdg + 90)

            def rot(px, py):
                ca, sa = math.cos(angle_rad), math.sin(angle_rad)
                return (cx_v + px * ca - py * sa,
                        cy_v - px * sa - py * ca)

            fls_d  = SENSOR_RANGE * self.scale
            fls_a  = FLS_THRESHOLD
            fls_pts = [
                rot(0, 0),
                rot(fls_d,  fls_d * math.tan(fls_a)),
                rot(fls_d, -fls_d * math.tan(fls_a)),
            ]
            flat_fls = [c for pt in fls_pts for c in pt]
            r_, g_, b_ = (int(color[1:3], 16),
                          int(color[3:5], 16),
                          int(color[5:7], 16))
            fp_color = f"#{max(0,r_//5):02x}{max(0,g_//5):02x}{max(0,b_//5):02x}"
            self.create_polygon(*flat_fls, fill=fp_color, outline=color, width=0.5, dash=(3, 4), tags="sonar_fp")

            ss_w = SENSOR_RANGE * self.scale
            ss_d = 5.0 * self.scale
            ss_pts_r = [
                rot(0, 0), rot(ss_d,  ss_w),
                rot(0,     ss_w),
            ]
            ss_pts_l = [
                rot(0, 0), rot(ss_d,  -ss_w),
                rot(0,     -ss_w),
            ]
            for ss_pts in (ss_pts_r, ss_pts_l):
                flat_ss = [c for pt in ss_pts for c in pt]
                self.create_polygon(*flat_ss, fill=fp_color, outline=color, width=0.5, dash=(2, 5), tags="sonar_fp")

    def _draw_comm_links(self):
        if not self.comm_mgr:
            return
        drawn = set()
    def _draw_comm_links(self):
        if not self.comm_mgr:
            return
        drawn = set()
        for s in ALL_AGENTS:
            for r in ALL_AGENTS:
                if s == r or (s, r) in drawn or (r, s) in drawn:
                    continue
                drawn.add((s, r))
                if s not in self.positions or r not in self.positions:
                    continue
                ps = self.positions[s]
                pr = self.positions[r]
                cx_s, cy_s = self.world_to_canvas(ps[1], ps[0])
                cx_r, cy_r = self.world_to_canvas(pr[1], pr[0])
                try:
                    snr = self.comm_mgr.get_snr(s, r)
                except Exception:
                    snr = float("nan")
                if math.isnan(snr):
                    continue
                if snr > 20:   color, lw = "#36c986", 1.5
                elif snr > 10: color, lw = "#cfb33c", 1.2
                else:           color, lw = "#c94b62", 1.0

                is_active = (self._active_links.get((s, r)) or
                             self._active_links.get((r, s)))
                if is_active:
                    doff = (self._link_anim_phase * 3) % 16
                    self.create_line(cx_s, cy_s, cx_r, cy_r, fill=color, width=lw + 0.5, dash=(8, 6), dashoffset=doff, tags="link")
                else:
                    self.create_line(cx_s, cy_s, cx_r, cy_r, fill=color, width=0.8, dash=(3, 8), tags="link")
                if self.show_labels:
                    mx = (cx_s + cx_r) / 2
                    my = (cy_s + cy_r) / 2
                    self.create_text(mx, my, text=f"{snr:.0f}dB", fill=color, font=("IBM Plex Mono", 7), tags="link_lbl")

    def _draw_pulses(self):
        now = time.time()
        for p in self._pulse_list:
            age      = now - p["start"]
            progress = age / p["duration"]
            if progress >= 1.0:
                continue
            cx_p, cy_p = self.world_to_canvas(p["east"], p["north"])

            if p["kind"] == "comms":
                r_px = p["max_r"] * progress
                fade = self._fade_color(p["color"], 1.0 - progress)
                self.create_oval(cx_p - r_px, cy_p - r_px, cx_p + r_px, cy_p + r_px, outline=fade, width=2.0, tags="pulse")
                recv = p.get("recv_pos")
                if recv is not None:
                    cx_r, cy_r = self.world_to_canvas(recv[1], recv[0])
                    dx = cx_r - cx_p
                    dy = cy_r - cy_p
                    dist = math.sqrt(dx**2 + dy**2) or 1
                    tip_x = cx_p + dx / dist * r_px
                    tip_y = cy_p + dy / dist * r_px
                    ang   = math.atan2(-dy, dx)
                    al    = 8
                    for side in (+0.5, -0.5):
                        ax_ = tip_x - al * math.cos(ang + side)
                        ay_ = tip_y + al * math.sin(ang + side)
                        self.create_line(tip_x, tip_y, ax_, ay_, fill=fade, width=1.5, tags="pulse_arrow")

            elif p["kind"] == "range":
                snr_range_m = p.get("max_r_m", 60.0)
                max_r_px = min(snr_range_m * self.scale, 200)
                r_px  = max_r_px * progress
                fade  = self._fade_color(p["color"], (1.0 - progress) * 0.5)
                self.create_oval(cx_p - r_px, cy_p - r_px, cx_p + r_px, cy_p + r_px, outline=fade, width=1.0, tags="pulse")

    def _fade_color(self, hex_color, alpha):
        try:
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            br, bg_, bb = 10, 21, 32
            r_ = int(r * alpha + br * (1 - alpha))
            g_ = int(g * alpha + bg_ * (1 - alpha))
            b_ = int(b * alpha + bb * (1 - alpha))
            return f"#{r_:02x}{g_:02x}{b_:02x}"
        except Exception:
            return hex_color

    def _draw_trails(self):
        for name in ALL_AGENTS:
            traj = list(self.trajectories.get(name, []))
            if len(traj) < 2:
                continue
            color  = AGENT_COLORS.get(name, AGENT_COLOR_DEFAULT)
            speed  = self.speeds.get(name, 1.0)
            pts    = traj[-self.trail_length:]
            n_pts  = len(pts)
            seg_sz = max(1, n_pts // TRAIL_ALPHA_STEPS)

            for i in range(0, n_pts - 1, max(1, seg_sz)):
                ratio     = i / max(n_pts - 1, 1)
                seg_color = self._dim_color(color, ratio)
                w_trail   = max(0.5, min(4.0, 1.0 + speed * 0.4)) * ratio
                p0 = pts[i]
                p1 = pts[min(i + seg_sz, n_pts - 1)]
                cx0, cy0 = self.world_to_canvas(p0[1], p0[0])
                cx1, cy1 = self.world_to_canvas(p1[1], p1[0])
                self.create_line(cx0, cy0, cx1, cy1, fill=seg_color, width=w_trail, capstyle=tk.ROUND, tags="trail")

    def _dim_color(self, hex_color, ratio):
        try:
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            f = ratio * 0.85
            return f"#{int(r*f):02x}{int(g*f):02x}{int(b*f):02x}"
        except Exception:
            return hex_color

    def _draw_waypoints(self):
        for name, wp in self.targets.items():
            color = AGENT_COLORS.get(name, AGENT_COLOR_DEFAULT)

            if wp is not None and name in self.positions:
                pos = self.positions[name]
                ax, ay = self.world_to_canvas(pos[1], pos[0])
                wx, wy = self.world_to_canvas(wp[1], wp[0])
                self.create_line(ax, ay, wx, wy, fill=color, width=1.0, dash=(6, 4), tags="wp_route")

            for i, (ep, np_) in enumerate(self.waypoint_hist.get(name, [])):
                ex, ey = self.world_to_canvas(ep, np_)
                s  = 5
                fc = self._dim_color(color, 0.4 + 0.6 * i / 5)
                self.create_line(ex - s, ey - s, ex + s, ey + s, fill=fc, width=1.2, tags="wp_hist")
                self.create_line(ex - s, ey + s, ex + s, ey - s, fill=fc, width=1.2, tags="wp_hist")

            if wp is not None:
                cx, cy = self.world_to_canvas(wp[1], wp[0])
                s = 8
                self.create_line(cx - s, cy - s, cx + s, cy + s, fill=color, width=2.0, tags="waypoint")
                self.create_line(cx - s, cy + s, cx + s, cy - s, fill=color, width=2.0, tags="waypoint")
                self.create_oval(cx - s - 2, cy - s - 2, cx + s + 2, cy + s + 2, outline=color, width=1.0, dash=(4, 3), tags="waypoint")

    def _draw_extrapolated(self):
        now = time.time()
        for name in ALL_AGENTS:
            if name not in self.positions:
                continue
            age = now - self._last_seen.get(name, now)
            if age < EXTRAP_START_S:
                continue
            pos   = self.positions[name]
            hdg   = self.headings.get(name, 0.0)
            speed = self.speeds.get(name, 0.5)
            hdg_rad = math.radians(hdg)
            dn = speed * age * math.cos(hdg_rad)
            de = speed * age * math.sin(hdg_rad)
            est_north = pos[0] + dn
            est_east  = pos[1] + de
            cx0, cy0  = self.world_to_canvas(pos[1], pos[0])
            cx1, cy1  = self.world_to_canvas(est_east, est_north)

            color = AGENT_COLORS.get(name, AGENT_COLOR_DEFAULT)
            self.create_line(cx0, cy0, cx1, cy1, fill=color, width=1.0, dash=(4, 6), tags="extrap")
            r = 10
            faded = self._dim_color(color, 0.4)
            self.create_oval(cx1 - r, cy1 - r, cx1 + r, cy1 + r, outline=faded, width=1.0, dash=(3, 3), tags="extrap")
            self.create_text(cx1, cy1 - 14, text=f"~{age:.0f}s", fill=faded, font=("IBM Plex Mono", 6), tags="extrap")

    def _draw_agents(self):
        now = time.time()
        for name in ALL_AGENTS:
            if name not in self.positions:
                continue

            pos      = self.positions[name]
            hdg      = self.headings.get(name, 0.0)
            selected = (name == self.selected_agent)
            age      = now - self._last_seen.get(name, now)
            stale    = age >= STALE_WARN_S
            dead     = age >= STALE_DEAD_S

            if selected:
                color = AGENT_COLOR_SEL
            elif dead:
                color = self._dim_color(AGENT_COLORS.get(name, AGENT_COLOR_DEFAULT), 0.35)
            elif stale:
                color = "#cfb33c"
            else:
                color = AGENT_COLORS.get(name, AGENT_COLOR_DEFAULT)

            cx, cy     = self.world_to_canvas(pos[1], pos[0])
            angle_rad  = math.radians(-hdg + 90)

            if selected:
                r = 22
                self.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=2, dash=(4, 2), tags="agent_sel")

            self._draw_auv_icon(cx, cy, angle_rad, color, selected=selected, dead=dead, name=name)

            arrow_len = 25.0 + self.scale * 3
            ax_ = cx + arrow_len * math.cos(angle_rad)
            ay_ = cy - arrow_len * math.sin(angle_rad)
            self.create_line(cx, cy, ax_, ay_, fill=color, width=1.5, arrow=tk.LAST, arrowshape=(8, 10, 3), tags="agent_hdg")

            comm_st  = self.comm_states.get(name, "CONNECTED")
            ring_col = STATE_LINK_COLORS.get(comm_st, color)
            self.create_oval(cx - 14, cy - 14, cx + 14, cy + 14, outline=ring_col, width=1.5, tags="agent_ring")

            if self.show_labels:
                sd   = self.sensor_data.get(name, {})
                vel  = sd.get("vel", 0.0)
                dep  = abs(sd.get("z", 0.0))
                stale_txt = f" (stale {age:.0f}s)" if stale and not dead else ""
                dead_txt  = " [OFFLINE]" if dead else ""
                lbl = (f"UUV  {name.upper()}\n"
                       f"{dep:.1f}m | {vel:.1f}m/s"
                       f"{stale_txt}{dead_txt}")
                self.create_text(cx + 18, cy - 18, text=lbl, fill=color, font=("IBM Plex Mono", 6, "bold"), anchor="sw", tags="agent_lbl")

    def _draw_boundary_warning(self):
        for name in ALL_AGENTS:
            if name not in self.positions:
                continue
            pos = self.positions[name]
            north, east = pos[0], pos[1]
            max_val = max(abs(north), abs(east))

            if max_val < 150.0:
                continue

            cx, cy = self.world_to_canvas(east, north)
            r_px = 25.0

            if max_val <= 170.0:
                self.create_oval(cx - r_px, cy - r_px, cx + r_px, cy + r_px, outline="#cfb33c", width=1.5, dash=(4, 3), tags="boundary_warning")
            else:
                self.create_oval(cx - r_px, cy - r_px, cx + r_px, cy + r_px, outline="#c94b62", width=2.0, tags="boundary_warning")
                self.create_text(cx, cy - r_px - 8, text="BOUNDARY!", fill="#c94b62", font=("IBM Plex Mono", 8, "bold"), tags="boundary_warning_lbl")

    def _draw_auv_icon(self, cx, cy, angle_rad, color, selected=False, dead=False, name=""):
        size = 14 if not selected else 16

        def rotate(px, py):
            ca, sa = math.cos(angle_rad), math.sin(angle_rad)
            return (cx + px * ca - py * sa,
                    cy - px * sa - py * ca)

        fill_color = color if not dead else self._dim_color(color, 0.3)


        body = [
            rotate( size,       0),
            rotate( size//3,    size//3),
            rotate(-size//2,    size//4),
            rotate(-size//3,    0),
            rotate(-size//2,   -size//4),
            rotate( size//3,   -size//3),
        ]
        flat = [c for pt in body for c in pt]
        fill_color = color if not dead else self._dim_color(color, 0.3)
        self.create_polygon(*flat, fill=fill_color, outline="#000a14", width=1, tags="agent_body")

        for sign in (+1, -1):
            wing = [
                rotate(-size//4,  0),
                rotate(-size//2,  sign * size//2),
                rotate(-size//2,  sign * size//4),
            ]
            flat_w = [c for pt in wing for c in pt]
            wc = self._dim_color(color, 0.55)
            self.create_polygon(*flat_w, fill=wc, outline="", tags="agent_wing")

        self.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill="white", outline="", tags="agent_dot")

    def _draw_drop_points(self):
        for east_m, north_m in self._drop_points:
            cx, cy = self.world_to_canvas(east_m, north_m)
            r = 5
            self.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#ffffff", fill="#cfb33c", width=1.5, tags="drop")
            self.create_text(cx + 8, cy - 8, text=f"E:{east_m:.1f}\nN:{north_m:.1f}", fill="#cfb33c", font=("IBM Plex Mono", 7), anchor="sw", tags="drop_lbl")

    def _draw_scale_bar(self, w, h):
        target_px = 80
        dist_m    = target_px / self.scale
        nice      = self._nice_step(dist_m)
        bar_px    = int(nice * self.scale)
        bar_y = h - 54
        x0, x1 = 20, 20 + bar_px

        self.create_line(x0, bar_y, x1, bar_y, fill="#aaaaaa", width=2)
        self.create_line(x0, bar_y - 4, x0, bar_y + 4, fill="#aaaaaa", width=2)
        self.create_line(x1, bar_y - 4, x1, bar_y + 4, fill="#aaaaaa", width=2)
        self.create_text((x0 + x1) / 2, bar_y - 8, text=f"{nice:.0f}m", fill="#aaaaaa", font=("IBM Plex Mono", 8))
        self.create_text(x0, bar_y + 12, text=f"×{self.scale:.1f}", fill="#5a8aaa", font=("IBM Plex Mono", 7), anchor="w")

    def _draw_info_bar(self, w, h):
        now  = time.time()
        name = self.selected_agent

        if name is None and AGENT_NAMES:
            name = AGENT_NAMES[0]

        if not name:
            return

        sd  = self.sensor_data.get(name, {})
        x   = sd.get("x", 0.0)
        y   = sd.get("y", 0.0)
        z   = abs(sd.get("z", 0.0))
        hdg = sd.get("heading", 0.0)
        spd = sd.get("vel", 0.0)
        age = now - self._last_seen.get(name, now)

        target = self.targets.get(name)
        if target is not None and name in self.positions:
            pos = self.positions[name]
            wp_dist = float(np.linalg.norm(pos[:2] - target[:2]))
            wp_dist_str = f"{wp_dist:.1f}m"
        else:
            wp_dist_str = "---"

        bar_h = 24
        bar_y = h - bar_h
        self.create_rectangle(0, bar_y, w, h, fill="#070c14", outline="")
        self.create_line(0, bar_y, w, bar_y, fill="#1a3050", width=1)

        fields = [
            ("VName",   name.upper()),
            ("X(m)",    f"{x:.1f}"),
            ("Y(m)",    f"{y:.1f}"),
            ("WP_DIST", wp_dist_str),
            ("Spd",     f"{spd:.2f}"),
            ("Hdg",     f"{hdg:.1f}°"),
            ("Dep(m)",  f"{z:.1f}"),
            ("Age(s)",  f"{age:.1f}"),
        ]

        cell_w = w // len(fields)
        for i, (label, val) in enumerate(fields):
            xc = i * cell_w
            self.create_rectangle(xc + 2, bar_y + 2, xc + cell_w - 2, h - 2, fill="#0d1a2a", outline="#1a3050")
            self.create_text(xc + cell_w // 2, bar_y + 5, text=label, fill="#4a6a8a", font=("IBM Plex Mono", 6), tags="info_bar")
            val_color = "#e2e8f0"
            if label == "VName":
                val_color = AGENT_COLORS.get(name.lower(), "#e2e8f0")
            elif label == "Age(s)" and age > STALE_WARN_S:
                val_color = "#cfb33c"
            self.create_text(xc + cell_w // 2, bar_y + 15, text=val, fill=val_color, font=("IBM Plex Mono", 7, "bold"), tags="info_bar")

    def reset_view(self):
        w, h = self.winfo_width(), self.winfo_height()
        self.offset_x = w / 2
        self.offset_y = h / 2
        self.scale    = 2.0
        self._bg_loaded = False
        self._redraw_all()

    def center_on_agents(self):
        if not self.positions:
            return
        norths = [p[0] for p in self.positions.values()]
        easts  = [p[1] for p in self.positions.values()]
        cw     = (max(easts) + min(easts)) / 2
        ch     = (max(norths) + min(norths)) / 2
        w, h   = self.winfo_width(), self.winfo_height()
        self.offset_x = w / 2 - cw * self.scale
        self.offset_y = h / 2 + ch * self.scale
        self._redraw_all()

    def toggle_grid(self):
        self.show_grid = not self.show_grid; self._redraw_all()

    def toggle_trails(self):
        self.show_trails = not self.show_trails; self._redraw_all()

    def toggle_coverage(self):
        self.show_coverage = not self.show_coverage; self._redraw_all()

    def toggle_links(self):
        self.show_links = not self.show_links; self._redraw_all()

    def toggle_labels(self):
        self.show_labels = not self.show_labels; self._redraw_all()

    def toggle_sonar(self):
        self.show_sonar_fp = not self.show_sonar_fp; self._redraw_all()

    def toggle_oparea(self):
        self.show_oparea = not self.show_oparea; self._redraw_all()

    def toggle_pulses(self):
        self.show_pulses = not self.show_pulses; self._redraw_all()


# ═══════════════════════════════════════════════════════════════
# Dashboard — Ana Kontrol GUI Penceresi (Sadeleştirilmiş)
# ═══════════════════════════════════════════════════════════════
class Dashboard:
    def __init__(self, consensus_fusion, comm_manager=None):
        self.fusion   = consensus_fusion
        self.comm_mgr = comm_manager
        self.start_t  = time.time()
        self.n_agents = len(AGENT_NAMES)

        # ── Paylaşılan veri tamponları (main.py'den yazılır) ──────
        self.sensor_data    = {n: {} for n in AGENT_NAMES}
        self.comm_log       = deque(maxlen=200)

        # ── İç tamponlar ──────────────────────────────────────────
        self.coverage_hist  = deque(maxlen=3000)
        self.consensus_hist = deque(maxlen=3000)
        self.pdr_hist       = deque(maxlen=3000)

        # Sonar tamponları (Tab 2 için)
        _WF = 200
        self._ss_waterfall = {n: deque(maxlen=_WF) for n in AGENT_NAMES}
        self._fls_data     = {n: None for n in AGENT_NAMES}
        self._mbes_data    = {n: None for n in AGENT_NAMES}
        self._fls_hist     = {n: deque(maxlen=3000) for n in AGENT_NAMES}
        self._WATERFALL_ROWS = _WF

        # ── Kontrol bayrakları (main.py tarafından okunur) ────────
        self._paused              = False
        self._emergency_stop      = False
        self._reset_planners_flag = False
        self._beacon_interval     = 300
        self._consensus_eps       = 0.3

        # ── Aktif link takibi ─────────────────────────────────────
        self._active_links = {}

        # ── Pencere inşa ──────────────────────────────────────────
        self._build_window()

    def _build_window(self):
        self.root = tk.Tk()
        self.root.title("SWARM AUV — Mission Control")
        self.root.configure(bg=BG)
        self.root.geometry("1500x920")
        self.root.minsize(1100, 700)

        # ── Başlık bandı (36px) ──
        self._build_header()

        # ── Pull-down menü çubuğu ──
        self._build_menubar()

        # ── Ana içerik: 3 sütun ──
        self._build_main_layout()

        # ── Durum çubuğu (8px font) ──
        self._build_statusbar()

        # ── Klavye kısayolları ──
        self.root.bind("<r>", lambda e: self.map_canvas.reset_view())
        self.root.bind("<c>", lambda e: self.map_canvas.center_on_agents())
        self.root.bind("<g>", lambda e: self.map_canvas.toggle_grid())
        self.root.bind("<t>", lambda e: self.map_canvas.toggle_trails())
        self.root.bind("<l>", lambda e: self.map_canvas.toggle_links())
        self.root.bind("<v>", lambda e: self.map_canvas.toggle_coverage())
        self.root.bind("<Escape>", lambda e: self._deselect_agent())

    def _build_header(self):
        # Header yüksekliği 44->36px
        hdr = tk.Frame(self.root, bg=BG, height=36)
        hdr.pack(fill="x", padx=0, pady=0)
        hdr.pack_propagate(False)

        # Gradient-like üst şerit
        accent_bar = tk.Frame(self.root, bg=ACCENT, height=2)
        accent_bar.pack(fill="x")

        inner = tk.Frame(hdr, bg=BG)
        inner.pack(fill="both", expand=True, padx=10, pady=2)

        # Sol: logo + başlık (sadeleştirilmiş "SWARM AUV  MISSION CONTROL")
        tk.Label(inner,
                 text="SWARM AUV  MISSION CONTROL",
                 font=("IBM Plex Mono", 12, "bold"),
                 fg=ACCENT, bg=BG).pack(side="left")

        # Sağ: saat
        right = tk.Frame(inner, bg=BG)
        right.pack(side="right")

        # Saat
        self._clock_lbl = tk.Label(right,
                                    text="T+00:00",
                                    font=FONT_MONO, fg=TEXT_DIM, bg=BG)
        self._clock_lbl.pack(side="right", padx=10)

    def _build_menubar(self):
        mbar = tk.Frame(self.root, bg=BG2, height=30)
        mbar.pack(fill="x")
        mbar.pack_propagate(False)

        inner = tk.Frame(mbar, bg=BG2)
        inner.pack(side="left", padx=8)

        def menu_btn(parent, text, menu_fn):
            btn = tk.Menubutton(
                parent, text=text,
                font=FONT_SMALL, fg=TEXT, bg=BG2,
                activeforeground=ACCENT, activebackground=BG3,
                relief="flat", bd=0, padx=10, pady=4)
            btn.pack(side="left")
            m = tk.Menu(btn, tearoff=0, bg=BG2, fg=TEXT,
                        activebackground=BG3, activeforeground=ACCENT,
                        font=FONT_SMALL, bd=0)
            menu_fn(m)
            btn["menu"] = m
            return btn

        # BackView menüsü
        def backview_menu(m):
            m.add_command(label="◈ Toggle Grid",
                          command=self.map_canvas.toggle_grid if hasattr(self, "map_canvas") else lambda: None,
                          accelerator="G")
            m.add_command(label="◈ Toggle Coverage Overlay",
                          command=lambda: self.map_canvas.toggle_coverage(),
                          accelerator="V")
            m.add_command(label="◈ Toggle Trails",
                          command=lambda: self.map_canvas.toggle_trails(),
                          accelerator="T")
            m.add_command(label="◈ Toggle Labels",
                          command=lambda: self.map_canvas.toggle_labels())
            m.add_separator()
            m.add_command(label="↺ Reset View",
                          command=lambda: self.map_canvas.reset_view(),
                          accelerator="R")
            m.add_command(label="⊕ Center on Agents",
                          command=lambda: self.map_canvas.center_on_agents(),
                          accelerator="C")
            m.add_separator()
            m.add_command(label="📊 Sonar Monitor",
                          command=self._open_sonar_window)

        btn_bv = menu_btn(inner, " BackView ▾", backview_menu)

        # Vehicles menüsü
        def vehicles_menu(m):
            m.add_command(label="◈ Toggle Comm Links",
                          command=lambda: self.map_canvas.toggle_links(),
                          accelerator="L")
            m.add_separator()
            for n in AGENT_NAMES:
                m.add_command(
                    label=f"● Focus {n.upper()}",
                    command=lambda name=n: self._select_agent(name))
            m.add_separator()
            m.add_command(label="× Deselect",
                          command=self._deselect_agent,
                          accelerator="ESC")

        menu_btn(inner, " Vehicles ▾", vehicles_menu)

        # Comms menüsü
        def comms_menu(m):
            m.add_command(label="📡 Comm Stats Window",
                          command=self._open_comms_window)
            m.add_separator()
            m.add_command(label="◈ Toggle Link Animation",
                          command=lambda: self.map_canvas.toggle_links())
            m.add_command(label="📋 Clear Comm Log",
                          command=lambda: self.comm_log.clear())

        menu_btn(inner, " Comms ▾", comms_menu)

        # AUV durum pilleri (menubar içine taşındı)
        self._status_labels = {}
        for name in AGENT_NAMES:
            color = AGENT_COLORS[name]
            pill = tk.Frame(inner, bg=BG3, padx=6, pady=2)
            pill.pack(side="left", padx=8)
            tk.Label(pill, text=f"● {name.upper()}",
                     font=FONT_TINY, fg=color, bg=BG3).pack(side="left")
            lbl = tk.Label(pill, text=" CONN",
                           font=FONT_TINY, fg=GREEN, bg=BG3)
            lbl.pack(side="left")
            self._status_labels[name] = lbl

        # Ayırıcı çizgi
        tk.Frame(self.root, bg=GRAY, height=1).pack(fill="x")

    def _build_main_layout(self):
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True)

        # Sol panel (sabit genişlik)
        left_panel = tk.Frame(main, bg=BG2, width=200)
        left_panel.pack(side="left", fill="y")
        left_panel.pack_propagate(False)

        self.agent_panel = AgentPanel(
            left_panel,
            on_select=self._select_agent)
        self.agent_panel.pack(fill="both", expand=True)

        tk.Frame(main, bg=GRAY, width=1).pack(side="left", fill="y")

        # Sağ panel (sabit genişlik)
        right_panel = tk.Frame(main, bg=BG2, width=280)
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)

        # Sağ üst: Telemetri
        self.telem_panel = TelemetryPanel(right_panel)
        self.telem_panel.pack(fill="x")

        tk.Frame(right_panel, bg=GRAY, height=1).pack(fill="x", padx=4)

        # Sağ alt: AppCast
        self.appcast_panel = AppCastPanel(right_panel)
        self.appcast_panel.pack(fill="both", expand=True)

        tk.Frame(main, bg=GRAY, width=1).pack(side="right", fill="y")

        # Merkez: Harita canvas
        map_frame = tk.Frame(main, bg=BG)
        map_frame.pack(side="left", fill="both", expand=True)

        self.map_canvas = MapCanvas(
            map_frame,
            consensus_fusion=self.fusion,
            comm_manager=self.comm_mgr,
            on_waypoint_set=self._on_waypoint_set,
            on_agent_select=self._select_agent,
            on_deploy=self._do_deploy,
            on_return=self._do_return,
            on_pause=self._toggle_pause)
        self.map_canvas.pack(fill="both", expand=True)

        self._build_map_toolbar(map_frame)

    def _build_map_toolbar(self, parent):
        toolbar = tk.Frame(parent, bg=BG2, height=28)
        toolbar.pack(fill="x", side="bottom")
        toolbar.pack_propagate(False)

        # Kısayol butonları (9 butondan 5'e düşürüldü)
        btns = [
            ("⊕ Reset",    self.map_canvas.reset_view),
            ("⊙ Center",   self.map_canvas.center_on_agents),
            ("# Grid",     self.map_canvas.toggle_grid),
            ("◉ Coverage", self.map_canvas.toggle_coverage),
            ("─ Links",    self.map_canvas.toggle_links),
        ]
        for label, cmd in btns:
            tk.Button(toolbar, text=label,
                      font=FONT_TINY, fg=TEXT_DIM, bg=BG2,
                      activebackground=BG3, activeforeground=ACCENT,
                      relief="flat", bd=0, padx=8, pady=2,
                      command=cmd).pack(side="left", padx=1)

        self._coord_lbl = tk.Label(
            toolbar, text="E:0.0  N:0.0",
            font=FONT_TINY, fg=TEXT_DIM, bg=BG2)
        self._coord_lbl.pack(side="right", padx=8)

        self.map_canvas.bind("<Motion>", self._on_map_mouse_move)

        self._wp_lbl = tk.Label(
            toolbar,
            text="[Left-click map to set waypoint for selected AUV | Right-drag to pan | Wheel to zoom]",
            font=FONT_TINY, fg=TEXT_DIM, bg=BG2)
        self._wp_lbl.pack(side="right", padx=8)

    def _on_map_mouse_move(self, event):
        try:
            east, north = self.map_canvas.canvas_to_world(event.x, event.y)
            self._coord_lbl.config(text=f"E:{east:7.1f}  N:{north:7.1f}")
        except Exception:
            pass

    def _build_statusbar(self):
        tk.Frame(self.root, bg=GRAY, height=1).pack(fill="x")

        sb = tk.Frame(self.root, bg=BG3, height=22)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)

        inner = tk.Frame(sb, bg=BG3)
        inner.pack(fill="both", expand=True, padx=8)

        FONT_STATUS = ("IBM Plex Mono", 8)

        self._sb_cov  = tk.Label(inner, text="Coverage: 0.0%",
                                  font=FONT_STATUS, fg=GREEN, bg=BG3)
        self._sb_cov.pack(side="left", padx=8)

        self._sb_rms  = tk.Label(inner, text="RMS: 0.000",
                                  font=FONT_STATUS, fg=ACCENT, bg=BG3)
        self._sb_rms.pack(side="left", padx=8)

        self._sb_pdr  = tk.Label(inner, text="PDR: N/A",
                                  font=FONT_STATUS, fg=YELLOW, bg=BG3)
        self._sb_pdr.pack(side="left", padx=8)

        self._sb_mode = tk.Label(inner, text="Mode: PROPOSED",
                                  font=FONT_STATUS, fg=TEXT_DIM, bg=BG3)
        self._sb_mode.pack(side="left", padx=8)

        self._sb_wp   = tk.Label(inner, text="",
                                  font=FONT_STATUS, fg=YELLOW, bg=BG3)
        self._sb_wp.pack(side="left", padx=8)

        self._sb_sel  = tk.Label(inner, text="Selected: none",
                                  font=FONT_STATUS, fg=TEXT_DIM, bg=BG3)
        self._sb_sel.pack(side="right", padx=8)

        self._sb_paused = tk.Label(inner, text="",
                                    font=FONT_STATUS, fg=YELLOW, bg=BG3)
        self._sb_paused.pack(side="right", padx=8)

    def update(self, states, targets, comm_quality, mode_name="proposed"):
        now     = time.time()
        elapsed = now - self.start_t

        mins, secs = divmod(int(elapsed), 60)
        self._clock_lbl.config(text=f"T+{mins:02d}:{secs:02d}")

        avg_cov = self.fusion.get_avg_coverage_pct()
        rms     = self.fusion.get_consensus_rms()

        try:
            summary = comm_quality.get_summary()
            pdr     = summary.get("global_pdr", 0.0)
        except Exception:
            summary = {}
            pdr     = 0.0

        self.coverage_hist.append((elapsed, avg_cov))
        self.consensus_hist.append((elapsed, rms))
        self.pdr_hist.append((elapsed, pdr))

        comm_states = self.fusion.get_comm_states()

        for name in AGENT_NAMES:
            st  = comm_states.get(name, "CONNECTED")
            col = STATE_COLORS.get(st, TEXT_DIM)
            abbr = st[:4].upper()
            if name in self._status_labels:
                self._status_labels[name].config(text=f" {abbr}", fg=col)

        self._store_sonar_data(states, elapsed)
        self._active_links = {}

        self.map_canvas.update_display(
            states, targets, self.sensor_data,
            comm_states=comm_states,
            active_links=self._active_links)

        self.agent_panel.update_agents(comm_states, self.sensor_data, self.fusion)

        mc = self.map_canvas
        mc.show_grid     = self.agent_panel.get_toggle("grid")
        mc.show_trails   = self.agent_panel.get_toggle("trails")
        mc.show_coverage = self.agent_panel.get_toggle("coverage")
        mc.show_links    = self.agent_panel.get_toggle("links")
        mc.show_labels   = self.agent_panel.get_toggle("labels")

        self.telem_panel.selected_agent = self.map_canvas.selected_agent
        self.telem_panel.update_telemetry(
            self.sensor_data, comm_states, self.fusion, elapsed)

        self.appcast_panel.update_appcast(
            comm_quality, self.comm_log,
            self.map_canvas.selected_agent,
            self.sensor_data, self.comm_mgr)

        self._sb_cov.config(text=f"Coverage: {avg_cov:.2f}%")
        self._sb_rms.config(text=f"RMS: {rms:.4f}")
        self._sb_pdr.config(text=f"PDR: {pdr:.3f}")
        self._sb_mode.config(text=f"Mode: {mode_name.upper()}")

        sel = self.map_canvas.selected_agent
        self._sb_sel.config(
            text=f"Selected: {sel.upper() if sel else 'none'}")

        if self._paused:
            self._sb_paused.config(text="⏸ PAUSED", fg=YELLOW)
        elif self._emergency_stop:
            self._sb_paused.config(text="⛔ STOPPED", fg=RED)
        else:
            self._sb_paused.config(text="▶ RUNNING", fg=GREEN)

        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            pass

    def _select_agent(self, name):
        self.map_canvas.selected_agent = name
        self.telem_panel.selected_agent = name
        if name:
            self._sb_sel.config(text=f"Selected: {name.upper()}")
            self._sb_wp.config(
                text="",
                fg=AGENT_COLORS.get(name, YELLOW))
        else:
            self._sb_sel.config(text="Selected: none")
            self._sb_wp.config(text="")

    def _deselect_agent(self):
        self._select_agent(None)
        self.map_canvas._redraw_all()

    def _on_waypoint_set(self, agent_name, east_m, north_m):
        msg = f"[WP] {agent_name.upper()} → E:{east_m:.1f} N:{north_m:.1f}"
        self.comm_log.append(msg)
        self._sb_wp.config(
            text=f"WP SET: {agent_name.upper()} → ({east_m:.1f},{north_m:.1f})",
            fg=YELLOW)
        if agent_name in self.sensor_data:
            self.sensor_data[agent_name]["target_x"] = east_m
            self.sensor_data[agent_name]["target_y"] = north_m
        self.map_canvas.targets[agent_name] = np.array([north_m, east_m])

    def _do_deploy(self):
        if self._paused:
            self._paused = False
            if hasattr(self, "_pause_btn_mbar") and self._pause_btn_mbar.winfo_exists():
                self._pause_btn_mbar.config(text="⏸ PAUSE")
        msg = "[DEPLOY] Mission started / resumed"
        self.comm_log.append(msg)
        self._sb_wp.config(text=msg, fg=GREEN)

    def _do_return(self):
        for name in AGENT_NAMES:
            self.map_canvas.targets[name] = np.array([0.0, 0.0])
            if name in self.sensor_data:
                self.sensor_data[name]["target_x"] = 0.0
                self.sensor_data[name]["target_y"] = 0.0
            self.map_canvas.waypoint_hist[name].append((0.0, 0.0))
        msg = "[RETURN] All AUVs ordered to (0, 0)"
        self.comm_log.append(msg)
        self._sb_wp.config(text=msg, fg=ACCENT)

    def trigger_beacon_pulse(self, sender, receiver, snr=15.0):
        try:
            self.map_canvas.trigger_comms_pulse(sender, receiver, snr)
        except Exception:
            pass

    def _toggle_pause(self):
        self._paused = not self._paused
        txt = "▶ RESUME" if self._paused else "⏸ PAUSE"
        if hasattr(self, "_pause_btn_mbar") and self._pause_btn_mbar.winfo_exists():
            self._pause_btn_mbar.config(text=txt)
        if hasattr(self, "_pause_btn_ctrl") and self._pause_btn_ctrl.winfo_exists():
            self._pause_btn_ctrl.config(text=txt)

    def _do_emergency_stop(self):
        if messagebox.askyesno(
                "EMERGENCY STOP",
                "Are you sure you want to EMERGENCY STOP?\nAll AUVs will halt.",
                parent=self.root):
            self._emergency_stop = True

    def _save_log_dialog(self):
        try:
            import tkinter.filedialog as fd
            path = fd.asksaveasfilename(
                parent=self.root,
                defaultextension=".txt",
                filetypes=[("Text", "*.txt"), ("All", "*.*")],
                initialfile="swarm_comm_log.txt")
            if path:
                with open(path, "w") as f:
                    f.write("\n".join(str(m) for m in self.comm_log))
                self._sb_wp.config(text=f"Log saved: {path}", fg=GREEN)
        except Exception as e:
            self._sb_wp.config(text=f"Save error: {e}", fg=RED)



    def _open_comms_window(self):
        if hasattr(self, "_comms_win") and self._comms_win.winfo_exists():
            self._comms_win.lift()
            return

        win = tk.Toplevel(self.root)
        win.title("Acoustic Channel Monitor")
        win.configure(bg=BG)
        win.geometry("900x600")
        self._comms_win = win

        fig = plt.Figure(figsize=(9, 5.5), facecolor=BG)
        gs  = gridspec.GridSpec(2, 2, figure=fig,
                                 hspace=0.40, wspace=0.35,
                                 left=0.07, right=0.97,
                                 top=0.92, bottom=0.10)

        ax_pdr = fig.add_subplot(gs[0, :])
        ax_pdr.set_facecolor(BG2)
        ax_pdr.set_title("PACKET DELIVERY RATIO", color=ACCENT,
                          fontsize=9, fontfamily="IBM Plex Mono")
        ax_pdr.set_xlabel("Time (s)", color=TEXT_DIM, fontsize=8)
        ax_pdr.set_ylabel("PDR", color=TEXT_DIM, fontsize=8)
        ax_pdr.tick_params(colors=TEXT_DIM, labelsize=7)
        for sp in ax_pdr.spines.values():
            sp.set_color(GRAY)

        if self.pdr_hist:
            t = [p[0] for p in self.pdr_hist]
            v = [p[1] for p in self.pdr_hist]
            ax_pdr.plot(t, v, "-", color=ACCENT, linewidth=1.8)
            ax_pdr.set_xlim(0, max(t[-1], 30))
        ax_pdr.set_ylim(0, 1.05)
        ax_pdr.axhline(0.8, color=YELLOW, linewidth=0.8, linestyle="--", alpha=0.6)
        ax_pdr.axhline(0.5, color=RED,    linewidth=0.8, linestyle=":", alpha=0.5)
        ax_pdr.grid(True, color=GRAY, alpha=0.2)

        ax_mat = fig.add_subplot(gs[1, 0])
        ax_mat.set_facecolor(BG2)
        ax_mat.set_title("SNR MATRIX (dB)", color=ACCENT,
                          fontsize=9, fontfamily="IBM Plex Mono")
        n = len(AGENT_NAMES)
        snr_data = np.full((n, n), np.nan)
        if self.comm_mgr:
            for i, s in enumerate(AGENT_NAMES):
                for j, r in enumerate(AGENT_NAMES):
                    if s != r:
                        try:
                            snr_data[i, j] = self.comm_mgr.get_snr(s, r)
                        except Exception:
                            pass
        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad(color=GRAY, alpha=0.4)
        masked = np.ma.masked_invalid(snr_data)
        im = ax_mat.imshow(masked, cmap=cmap, vmin=0, vmax=30, aspect="auto")
        ax_mat.set_xticks(range(n))
        ax_mat.set_xticklabels(AGENT_NAMES, fontsize=7, color=TEXT)
        ax_mat.set_yticks(range(n))
        ax_mat.set_yticklabels(AGENT_NAMES, fontsize=7, color=TEXT)
        ax_mat.set_xlabel("Receiver", color=TEXT_DIM, fontsize=7)
        ax_mat.set_ylabel("Sender",   color=TEXT_DIM, fontsize=7)
        for i in range(n):
            for j in range(n):
                if i != j and not math.isnan(snr_data[i, j]):
                    ax_mat.text(j, i, f"{snr_data[i,j]:.0f}",
                                ha="center", va="center",
                                color="black", fontsize=9, fontweight="bold")
        fig.colorbar(im, ax=ax_mat, fraction=0.05)

        ax_rms = fig.add_subplot(gs[1, 1])
        ax_rms.set_facecolor(BG2)
        ax_rms.set_title("CONSENSUS RMS", color=ACCENT,
                          fontsize=9, fontfamily="IBM Plex Mono")
        ax_rms.set_xlabel("Time (s)", color=TEXT_DIM, fontsize=8)
        ax_rms.tick_params(colors=TEXT_DIM, labelsize=7)
        for sp in ax_rms.spines.values():
            sp.set_color(GRAY)
        if self.consensus_hist:
            t = [p[0] for p in self.consensus_hist]
            r = [p[1] for p in self.consensus_hist]
            ax_rms.plot(t, r, "-", color=RED, linewidth=1.5)
            ax_rms.set_xlim(0, max(t[-1], 30))
        ax_rms.grid(True, color=GRAY, alpha=0.2)

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    def _open_sonar_window(self):
        if hasattr(self, "_sonar_win") and self._sonar_win.winfo_exists():
            self._sonar_win.lift()
            return

        win = tk.Toplevel(self.root)
        win.title("Sonar Monitor")
        win.configure(bg=BG)
        win.geometry("1100x880")
        self._sonar_win = win

        fig = plt.Figure(figsize=(11, 8.2), facecolor=BG)
        gs  = gridspec.GridSpec(3, 4, figure=fig,
                                 hspace=0.50, wspace=0.35,
                                 left=0.05, right=0.97,
                                 top=0.94, bottom=0.06)

        _WF = self._WATERFALL_ROWS

        # Row 0: FLS
        fls_ims  = {}
        fls_axes = {}
        for i, name in enumerate(AGENT_NAMES):
            ax = fig.add_subplot(gs[0, i])
            ax.set_facecolor("#000510")
            ax.tick_params(colors=TEXT_DIM, labelsize=6)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRAY)
            ax.set_title(f"FLS — {name}",
                          color=AGENT_COLORS[name],
                          fontsize=8, fontfamily="IBM Plex Mono")
            fls = self._fls_data.get(name)
            data = fls if (fls is not None and hasattr(fls, "ndim") and fls.ndim == 2) \
                   else np.zeros((64, 64))
            im = ax.imshow(data, aspect="auto", cmap="viridis",
                           vmin=0, vmax=1, origin="upper", interpolation="nearest")
            fls_ims[name]  = im
            fls_axes[name] = ax

        ax_risk = fig.add_subplot(gs[0, 3])
        ax_risk.set_facecolor(BG2)
        ax_risk.axis("off")
        ax_risk.set_title("FLS RISK", color=ACCENT, fontsize=8,
                           fontfamily="IBM Plex Mono")
        risk_lines = ["FLS RISK SUMMARY", "─" * 22, ""]
        for name in AGENT_NAMES:
            fls = self._fls_data.get(name)
            if fls is not None and hasattr(fls, "ndim") and fls.ndim == 2:
                n_az = fls.shape[1]; s = n_az // 3
                lft  = float(np.max(fls[:, :s]))
                ctr  = float(np.max(fls[:, s:2*s]))
                rgt  = float(np.max(fls[:, 2*s:]))
                q = "⚠ RISK" if ctr > FLS_THRESHOLD else "✓ CLEAR"
                risk_lines.append(f"{name}: {q}")
                risk_lines.append(f"  L:{lft:.2f} C:{ctr:.2f} R:{rgt:.2f}")
            else:
                risk_lines.append(f"{name}: NO DATA")
            risk_lines.append("")
        ax_risk.text(0.05, 0.95, "\n".join(risk_lines),
                     transform=ax_risk.transAxes,
                     fontsize=7, color=GREEN, va="top",
                     fontfamily="IBM Plex Mono")

        # Row 1: Sidescan
        ss_ims = {}
        for i, name in enumerate(AGENT_NAMES):
            ax = fig.add_subplot(gs[1, i])
            ax.set_facecolor("#000510")
            ax.tick_params(colors=TEXT_DIM, labelsize=6)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRAY)
            ax.set_title(f"Sidescan — {name}",
                          color=AGENT_COLORS[name],
                          fontsize=8, fontfamily="IBM Plex Mono")
            wf = self._ss_waterfall.get(name)
            if wf and len(wf) > 0:
                mat = np.array(list(wf))
                vmax = max(mat.max(), 0.3)
                mat = mat / vmax
            else:
                mat = np.zeros((_WF, 64))
            im = ax.imshow(mat, aspect="auto", cmap="gray",
                           vmin=0, vmax=1, origin="upper", interpolation="nearest")
            ss_ims[name] = im

        ax_ss = fig.add_subplot(gs[1, 3])
        ax_ss.set_facecolor(BG2)
        ax_ss.axis("off")
        ax_ss.set_title("SIDESCAN INFO", color=ACCENT, fontsize=8,
                         fontfamily="IBM Plex Mono")
        ax_ss.text(0.05, 0.95,
                   "Sidescan Sonar\nStatus: ACTIVE\n\nBayesian map\nupdate active.",
                   transform=ax_ss.transAxes,
                   fontsize=8, color=GREEN, va="top",
                   fontfamily="IBM Plex Mono")

        # Row 2: MBES (ProfilingSonar)
        mbes_ims = {}
        for i, name in enumerate(AGENT_NAMES):
            ax = fig.add_subplot(gs[2, i])
            ax.set_facecolor("#000510")
            ax.tick_params(colors=TEXT_DIM, labelsize=6)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRAY)
            ax.set_title(f"MBES — {name}",
                          color=AGENT_COLORS[name],
                          fontsize=8, fontfamily="IBM Plex Mono")
            mbes = self._mbes_data.get(name)
            data = mbes if (mbes is not None and hasattr(mbes, "ndim") and mbes.ndim == 2) \
                   else np.zeros((64, 64))
            im = ax.imshow(data, aspect="auto", cmap="plasma",
                           vmin=0, vmax=0.15, origin="upper", interpolation="nearest")
            mbes_ims[name] = im

        ax_mbes_info = fig.add_subplot(gs[2, 3])
        ax_mbes_info.set_facecolor(BG2)
        ax_mbes_info.axis("off")
        ax_mbes_info.set_title("MBES INFO", color=ACCENT, fontsize=8,
                                fontfamily="IBM Plex Mono")
        ax_mbes_info.text(0.05, 0.95,
                          "Multibeam Sonar\nStatus: ACTIVE\n\nProfiling seabed\nbathymetry.",
                          transform=ax_mbes_info.transAxes,
                          fontsize=8, color=GREEN, va="top",
                          fontfamily="IBM Plex Mono")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

        def update_plots():
            if not win.winfo_exists():
                return

            # FLS
            for name in AGENT_NAMES:
                fls = self._fls_data.get(name)
                data = fls if (fls is not None and hasattr(fls, "ndim") and fls.ndim == 2) \
                       else np.zeros((64, 64))
                fls_ims[name].set_data(data)

            # FLS Risk Text
            risk_lines = ["FLS RISK SUMMARY", "─" * 22, ""]
            for name in AGENT_NAMES:
                fls = self._fls_data.get(name)
                if fls is not None and hasattr(fls, "ndim") and fls.ndim == 2:
                    n_az = fls.shape[1]; s = n_az // 3
                    lft  = float(np.max(fls[:, :s]))
                    ctr  = float(np.max(fls[:, s:2*s]))
                    rgt  = float(np.max(fls[:, 2*s:]))
                    q = "⚠ RISK" if ctr > FLS_THRESHOLD else "✓ CLEAR"
                    risk_lines.append(f"{name}: {q}")
                    risk_lines.append(f"  L:{lft:.2f} C:{ctr:.2f} R:{rgt:.2f}")
                else:
                    risk_lines.append(f"{name}: NO DATA")
                risk_lines.append("")

            ax_risk.clear()
            ax_risk.axis("off")
            ax_risk.set_title("FLS RISK", color=ACCENT, fontsize=8, fontfamily="IBM Plex Mono")
            ax_risk.text(0.05, 0.95, "\n".join(risk_lines),
                         transform=ax_risk.transAxes,
                         fontsize=7, color=GREEN, va="top",
                         fontfamily="IBM Plex Mono")

            # SSS
            for name in AGENT_NAMES:
                wf = self._ss_waterfall.get(name)
                if wf and len(wf) > 0:
                    mat = np.array(list(wf))
                    vmax = max(mat.max(), 0.3)
                    mat = mat / vmax
                else:
                    mat = np.zeros((_WF, 64))
                ss_ims[name].set_data(mat)

            # MBES
            for name in AGENT_NAMES:
                mbes = self._mbes_data.get(name)
                data = mbes if (mbes is not None and hasattr(mbes, "ndim") and mbes.ndim == 2) \
                       else np.zeros((64, 64))
                mbes_ims[name].set_data(data)

            canvas.draw_idle()
            win.after(500, update_plots)

        win.after(500, update_plots)

    def _store_sonar_data(self, states, elapsed):
        for name in AGENT_NAMES:
            if name not in states:
                continue
            fls = states[name].get("ImagingSonar")
            if fls is not None and hasattr(fls, "ndim") and fls.ndim == 2:
                self._fls_data[name] = fls.copy()
                n_az = fls.shape[1]; s = n_az // 3
                self._fls_hist[name].append((
                     elapsed,
                     float(np.max(fls[:, :s])),
                     float(np.max(fls[:, s:2*s])),
                     float(np.max(fls[:, 2*s:]))
                ))
            ss = states[name].get("SSS", states[name].get("SidescanSonar"))
            if ss is not None and hasattr(ss, "__len__") and len(ss) > 0:
                self._ss_waterfall[name].append(np.array(ss, dtype=np.float32))
            mbes = states[name].get("MBES", states[name].get("ProfilingSonar"))
            if mbes is not None and hasattr(mbes, "ndim") and mbes.ndim == 2:
                self._mbes_data[name] = mbes.copy()

    def console_log(self, message):
        self.comm_log.append(message)
        try:
            self.appcast_panel.console_log(message)
        except Exception:
            pass